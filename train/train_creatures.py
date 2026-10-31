"""Train the creature network (creatures.py) on crops round datagen's creatures.

    python train/train_creatures.py --data data/train --out runs/creatures --steps 30

It learns the creatures' layer, rgb and alpha, from maps of their parts' rough
shapes (see creatures.py). L1 on the layer (premultiplied) and alpha, and LPIPS on
the layer laid over the level's frame, settle for a blur of every skin it might
invent; once those have shaped it, a patch discriminator and a match of feature
statistics (Gram matrices) on creature pixels ask for one sharp skin, and the
maps' noise gives it the randomness to pick one. This network draws nothing
else, so they can't hurt the level (as they did in runs 5 and 6).
"""

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from creatures import CreatureNet, Crops, composite, maps
from model import PatchDisc

WARMUP = 500
# Steps of L1 and LPIPS alone before the adversarial and Gram losses start, and then
# over which they ramp up from zero.
GAN_FROM, GAN_RAMP = 3000, 5000


class Trainer:
    def __init__(self, data, out, batch=16, size=128, lr=2e-4, workers=2, device=None, lpips_weight=0.2,
                 adv_weight=0.05, gram_weight=2.0, ema_decay=0.999):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.device == "cuda"
        n_shards = len(list(Path(data).glob("shard_*")))
        assert n_shards >= 2, "need at least two shards (the last one is held out)"
        self.train_set = Crops(data, range(n_shards - 1), size)
        self.val_set = Crops(data, [n_shards - 1], size, fixed=True)
        self.model = CreatureNet().to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema_decay = ema_decay
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.lr = lr
        self.lpips_weight, self.adv_weight, self.gram_weight = lpips_weight, adv_weight, gram_weight
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval().requires_grad_(False)
        self.disc = PatchDisc().to(self.device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.0, 0.99))
        self.disc_scaler = torch.amp.GradScaler(enabled=self.amp)
        self.step = 0
        ckpt = self.out / "ckpt.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.ema.load_state_dict(state["ema"])
            self.opt.load_state_dict(state["opt"])
            self.disc.load_state_dict(state["disc"])
            self.disc_opt.load_state_dict(state["disc_opt"])
            self.step = state["step"]
            print(f"resumed at step {self.step}")
        self.loader = DataLoader(self.train_set, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                                 persistent_workers=workers > 0)
        self.batches = (b for _ in itertools.count() for b in self.loader)
        self.watch = next(iter(DataLoader(self.val_set, batch_size=6, sampler=np.linspace(0, len(self.val_set) - 1, 6).astype(int))))
        params = sum(p.numel() for p in self.model.parameters())
        print(f"{params / 1e6:.2f}M params, {len(self.train_set)} train / {len(self.val_set)} val crops, {self.device}")

    def forward(self, batch, model=None):
        """(predicted layer, target layer, level frame, proxy cover) for a batch."""
        q, d, cam, layer, frame = (x.to(self.device, non_blocking=True) for x in batch)
        with torch.no_grad():
            feats, ids = maps(q, d, cam)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            pred = (model or self.model)(feats, ids)
        return pred.float(), layer, frame, feats[:, :1]

    def losses(self, pred, target, frame):
        """(loss, layer L1, LPIPS of the composite) for a batch."""
        pa, ta = pred[:, 3:], target[:, 3:]
        l1 = (pred[:, :3] * pa - target[:, :3] * ta).abs().mean() + (pa - ta).abs().mean()
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            perceptual = self.lpips(composite(frame, pred) * 2 - 1, composite(frame, target) * 2 - 1).float().mean()
        return l1 + self.lpips_weight * perceptual, l1, perceptual

    def gram(self, x, mask):
        """Gram matrices [b, c, c] of LPIPS's first three feature maps over the masked pixels."""
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            feats = self.lpips.net(self.lpips.scaling_layer(x * 2 - 1))
        grams = []
        for f in list(feats)[:3]:
            m = F.adaptive_avg_pool2d(mask, f.shape[-2:])
            f = f.float() * m
            grams.append(torch.einsum("bchw,bdhw->bcd", f, f) / (m.sum((1, 2, 3))[:, None, None] * f.shape[1] + 1e-6))
        return grams

    def creature_losses(self, pred, target, frame, cells):
        """(adversarial, Gram) losses for the network, on the composites' creature cells."""
        fake, real = composite(frame, pred), composite(frame, target)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            score = self.disc(fake).float()
        adv = -(score * cells).sum() / cells.sum().clamp(min=1)
        mask = target[:, 3:]
        with torch.no_grad():
            want = self.gram(real, mask)
        gram = sum((g - t).abs().mean() for g, t in zip(self.gram(fake, mask), want))
        return adv, gram

    def disc_step(self, pred, target, frame, cells):
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            both = torch.cat([composite(frame, target), composite(frame, pred.detach())])
            real, fake = self.disc(both).float().chunk(2)
        loss = ((F.relu(1 - real) + F.relu(1 + fake)) * cells).sum() / cells.sum().clamp(min=1)
        self.disc_opt.zero_grad(set_to_none=True)
        self.disc_scaler.scale(loss).backward()
        self.disc_scaler.step(self.disc_opt)
        self.disc_scaler.update()
        return loss.item()

    def run(self, minutes=None, steps=None, log_every=50, sample_every=1000):
        self.model.train()
        start, done, logs = time.time(), 0, {k: [] for k in ("l1", "lpips", "adv", "gram", "disc")}
        while (steps is None or done < steps) and (minutes is None or time.time() - start < minutes * 60):
            for g in self.opt.param_groups:
                g["lr"] = self.lr * min(1.0, (self.step + 1) / WARMUP)
            pred, target, frame, cover = self.forward(next(self.batches))
            loss, l1, perceptual = self.losses(pred, target, frame)
            ramp = min(1.0, max(0.0, (self.step - GAN_FROM) / GAN_RAMP))
            # Where the creatures are, or the network might put them: in the disc's 8x8 cells.
            cells = F.max_pool2d(torch.maximum(target[:, 3:], cover), 8)
            if ramp > 0:
                adv, gram = self.creature_losses(pred, target, frame, cells)
                loss = loss + ramp * (self.adv_weight * adv + self.gram_weight * gram)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if torch.isfinite(loss):
                self.scaler.step(self.opt)
            self.scaler.update()
            if self.step >= GAN_FROM - 1000:  # the discriminator gets a head start
                logs["disc"].append(self.disc_step(pred, target, frame, cells))
            with torch.no_grad():
                torch._foreach_lerp_(list(self.ema.parameters()), list(self.model.parameters()), 1 - self.ema_decay)
            logs["l1"].append(l1.item())
            logs["lpips"].append(perceptual.item())
            if ramp > 0:
                logs["adv"].append(adv.item())
                logs["gram"].append(gram.item())
            self.step += 1
            done += 1
            if self.step % log_every == 0:
                recent = {f"train_{k}": float(np.mean(v[-log_every:])) for k, v in logs.items() if v}
                self.log(**recent, gan_ramp=ramp, secs=time.time() - start)
            if self.step % sample_every == 0:
                self.checkpoint()
        self.checkpoint()

    @torch.no_grad()
    def validate(self, batches=8):
        loader = DataLoader(self.val_set, batch_size=16)
        ls = []
        for b in itertools.islice(loader, batches):
            pred, target, frame, _ = self.forward(b, self.ema)
            ls.append([x.item() for x in self.losses(pred, target, frame)[1:]])
        return tuple(float(x) for x in np.mean(ls, 0))

    @torch.no_grad()
    def sample(self, path):
        """Rows of target | prediction (laid over the level) | proxy cover | predicted alpha."""
        pred, target, frame, cover = self.forward(self.watch, self.ema)
        rows = [torch.cat([composite(frame, target)[i], composite(frame, pred)[i], cover[i].expand(3, -1, -1),
                           pred[i, 3:].expand(3, -1, -1)], 2) for i in range(len(pred))]
        grid = torch.cat(rows, 1).clamp(0, 1)
        Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)

    def checkpoint(self):
        val_l1, val_lpips = self.validate()
        self.log(val_l1=val_l1, val_lpips=val_lpips)
        self.sample(self.out / f"sample_{self.step:07d}.png")
        self.sample(self.out / "sample_latest.png")
        torch.save({"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                    "disc": self.disc.state_dict(), "disc_opt": self.disc_opt.state_dict(), "step": self.step},
                   self.out / "ckpt.pt")

    def log(self, **kv):
        rec = {"step": self.step, **kv}
        print(json.dumps(rec), flush=True)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=128, help="crop size, a multiple of 4")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=10**9)
    ap.add_argument("--gan-from", type=int, default=GAN_FROM, help="step the adversarial and Gram losses start at")
    a = ap.parse_args()
    GAN_FROM = a.gan_from
    t = Trainer(a.data, a.out, batch=a.batch, size=a.size, workers=a.workers)
    t.run(minutes=a.minutes, steps=a.steps, log_every=a.log_every, sample_every=a.sample_every)
