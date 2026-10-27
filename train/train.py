"""Train the neural renderer.

Local smoke test:
    python train/train.py --data data/test --out runs/smoke --small --steps 30

On Colab, run it inside the persistent kernel with train/colab_chunk.py: state
survives between `colab exec` calls, and a busy kernel keeps a free-tier VM
alive where a detached process does not. Mirror checkpoints locally with
train/colab_sync.sh in case the VM is reclaimed anyway.
"""

import argparse
import copy
import itertools
import json
import math
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from data import Shards, WidthBatches, collate
from model import Renderer, loss_fn, migrate

# Steps of learning-rate warm-up after the optimizer starts afresh.
WARMUP = 500
# Longer when a new head starts from scratch on a trained body, so its first,
# random gradients don't knock the body about.
WARMUP_NEW_HEAD = 3000


class Trainer:
    def __init__(
        self, data, out, small=False, batch=16, lr=3e-4, workers=2, device=None, lpips_weight=0.1, ema_decay=0.999
    ):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        n_shards = len(list(Path(data).glob("shard_*")))
        assert n_shards >= 2, "need at least two shards (the last one is held out)"
        self.train_set = Shards(data, range(n_shards - 1))
        self.val_set = Shards(data, [n_shards - 1])
        self.cfg = dict(d=128, heads=4, enc_layers=2, dec_layers=2) if small else {}
        ckpt = self.out / "ckpt.pt"
        state = torch.load(ckpt, map_location=self.device) if ckpt.exists() else None
        migrated, warmup = False, WARMUP
        if state is not None:
            self.cfg = state.get("cfg", self.cfg)
            # From before widths varied or creature parts: start both copies from the averaged
            # weights, migrated.
            weights = state.get("ema", state["model"])
            if "q_pos" in weights:  # the old patch head goes: its replacement needs the long warm-up
                warmup = WARMUP_NEW_HEAD
            weights, cfg, migrated = migrate(weights, self.cfg)
            if migrated:
                state = {**state, "model": weights, "ema": weights, "opt": None}
                self.cfg = cfg
        self.model = Renderer(**self.cfg).to(self.device)
        # A perceptual loss on top of L1, against the blur that L1 alone settles for.
        self.lpips_weight = lpips_weight
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval().requires_grad_(False)
        covers = ("cover", "part_cover")
        main = [p for n, p in self.model.named_parameters() if n not in covers]
        self.opt = torch.optim.AdamW(main, lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
        self.opt.add_param_group({"params": [getattr(self.model, n) for n in covers], "weight_decay": 0.0})
        self.lr = lr
        self.amp = self.device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.step = 0
        # Where the learning rate last warmed up from, and for how long.
        self.warm_from, self.warmup = 0, WARMUP
        # Each batch is views of one screen width.
        self.loader = DataLoader(
            self.train_set, batch_sampler=WidthBatches(self.train_set.widths, batch, seed=int(time.time())),
            collate_fn=collate, num_workers=workers, persistent_workers=workers > 0,
        )
        self.batches = self._forever()
        # Fixed views to watch across training, one at a time (their widths differ).
        self.watch = [collate([self.val_set[i]]) for i in np.linspace(0, len(self.val_set) - 1, 6).astype(int)]
        if state is not None:
            missing = self.model.load_state_dict(state["model"], strict=False).missing_keys
            assert set(missing) <= {"cover"}, missing
            if state["opt"] is None:  # migrated: the shapes changed, so the optimizer starts over
                self.warm_from, self.warmup = state["step"], warmup
            elif len(state["opt"]["param_groups"]) == len(self.opt.param_groups):
                self.opt.load_state_dict(state["opt"])
                self.warm_from, self.warmup = state.get("warm", (0, WARMUP))
            else:  # from before the coverage bias: restore the rest, start `cover` fresh
                cover = self.opt.param_groups.pop()
                self.opt.load_state_dict(state["opt"])
                self.opt.add_param_group(cover)
            self.step = state["step"]
            print(f"resumed at step {self.step}" + (" (migrated: fresh optimizer)" if migrated else "")
                  + (f", new {missing}" if missing else ""))
        # A slow average of the weights, used for validation, samples and export: steadier than the
        # weights of any one step.
        self.ema_decay = ema_decay
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        if ckpt.exists() and "ema" in state:
            self.ema.load_state_dict(state["ema"])
        params = sum(p.numel() for p in self.model.parameters())
        print(f"{params / 1e6:.1f}M params, {len(self.train_set)} train / {len(self.val_set)} val views, {self.device}")

    def _forever(self):
        for _ in itertools.count():
            yield from self.loader

    def _lr(self, progress=None):
        """Warm-up from the first step; with `progress` (0..1 through a run), a cosine decay to 5%."""
        lr = self.lr * min(1.0, (self.step - self.warm_from + 1) / self.warmup)
        if progress is not None:
            lr *= 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return lr

    def forward(self, batch, model=None):
        feats, tex, pad, cams, images = (x.to(self.device, non_blocking=True) for x in batch[:5])
        pfeats, pids, ppad, _ = (x.to(self.device, non_blocking=True) for x in batch[5])
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            pred = (model or self.model)(feats, tex, pad, cams, width=images.shape[-1], parts=(pfeats, pids, ppad))
        return pred.float(), images

    def losses(self, pred, target):
        """(training loss, L1, LPIPS) for a batch."""
        l1 = loss_fn(pred, target)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            perceptual = self.lpips(pred * 2 - 1, target * 2 - 1).float().mean()
        return l1 + self.lpips_weight * perceptual, l1, perceptual

    def run(self, minutes=None, steps=None, log_every=50, sample_every=500, decay=False):
        """Train for `minutes` and/or `steps`. With `decay`, the learning rate follows a cosine
        over this run (by time when `minutes` is given), so each run ends settled; the next
        run restarts at the full rate."""
        self.model.train()
        start, done, l1s, lps = time.time(), 0, [], []
        while (steps is None or done < steps) and (minutes is None or time.time() - start < minutes * 60):
            progress = None
            if decay:
                progress = (time.time() - start) / (minutes * 60) if minutes else done / steps
            for g in self.opt.param_groups:
                g["lr"] = self._lr(progress)
            pred, target = self.forward(next(self.batches))
            loss, l1, perceptual = self.losses(pred, target)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            with torch.no_grad():
                torch._foreach_lerp_(list(self.ema.parameters()), list(self.model.parameters()), 1 - self.ema_decay)
            l1s.append(l1.item())
            lps.append(perceptual.item())
            self.step += 1
            done += 1
            if self.step % log_every == 0:
                self.log(
                    train_l1=float(np.mean(l1s[-log_every:])), train_lpips=float(np.mean(lps[-log_every:])),
                    lr=self.opt.param_groups[0]["lr"], secs=time.time() - start,
                )
            if self.step % sample_every == 0:
                self.checkpoint()
        self.checkpoint()
        return float(np.mean(l1s)) if l1s else None

    @torch.no_grad()
    def validate(self, batches=8):
        """(L1, LPIPS, L1 on creature pixels) of the averaged weights over the first
        validation batches."""
        loader = DataLoader(self.val_set, batch_sampler=WidthBatches(self.val_set.widths, 16, seed=0, partial=True), collate_fn=collate)
        ls, err, px = [], 0.0, 0.0
        for b in itertools.islice(loader, batches):
            pred, target = self.forward(b, self.ema)
            ls.append([x.item() for x in self.losses(pred, target)[1:]])
            mask = b[5][3].to(self.device)[:, None]
            err += ((pred - target).abs() * mask).sum().item() / 3
            px += mask.sum().item()
        return (*(float(x) for x in np.mean(ls, 0)), err / max(px, 1.0))

    @torch.no_grad()
    def sample(self, path):
        """Rows of target | prediction (averaged weights) for the watched validation views."""
        rows = []
        for batch in self.watch:
            pred, target = self.forward(batch, self.ema)
            rows.append(torch.cat([target[0], pred[0].clamp(0, 1)], 2))  # side by side
        wide = max(r.shape[2] for r in rows)
        grid = torch.cat([F.pad(r, (0, wide - r.shape[2])) for r in rows], 1)  # stacked vertically
        Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)

    def checkpoint(self):
        val_l1, val_lpips, val_creature = self.validate()
        self.log(
            val_l1=val_l1, val_lpips=val_lpips, val_creature_l1=val_creature,
            cover=self.model.cover.abs().mean().item(), part_cover=self.model.part_cover.abs().mean().item(),
        )
        self.sample(self.out / f"sample_{self.step:07d}.png")
        self.sample(self.out / "sample_latest.png")
        state = {
            "model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
            "step": self.step, "cfg": self.cfg, "warm": (self.warm_from, self.warmup),
        }
        torch.save(state, self.out / "ckpt.pt")

    def log(self, **kv):
        rec = {"step": self.step, **kv}
        print(json.dumps(rec), flush=True)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=10**9)
    ap.add_argument("--lpips", type=float, default=0.1, help="weight of the perceptual loss")
    ap.add_argument("--decay", action="store_true", help="cosine learning-rate decay over this run")
    a = ap.parse_args()
    t = Trainer(a.data, a.out, small=a.small, batch=a.batch, workers=a.workers, lpips_weight=a.lpips)
    t.run(minutes=a.minutes, steps=a.steps, log_every=a.log_every, sample_every=a.sample_every, decay=a.decay)
