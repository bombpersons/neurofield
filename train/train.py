"""Train the neural renderer.

Local smoke test:
    python train/train.py --data data/test --out runs/smoke --small --steps 30

On Colab, run it inside the persistent kernel with train/colab_chunk.py: state
survives between `colab exec` calls, and a busy kernel keeps a free-tier VM
alive where a detached process does not. Mirror checkpoints locally with
train/colab_sync.sh in case the VM is reclaimed anyway.
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import Shards, collate
from model import Renderer, loss_fn


class Trainer:
    def __init__(self, data, out, small=False, batch=16, lr=3e-4, workers=2, device=None, lpips_weight=0.1):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        n_shards = len(list(Path(data).glob("shard_*")))
        assert n_shards >= 2, "need at least two shards (the last one is held out)"
        self.train_set = Shards(data, range(n_shards - 1))
        self.val_set = Shards(data, [n_shards - 1])
        self.cfg = dict(d=128, heads=4, enc_layers=2, dec_layers=2) if small else {}
        self.model = Renderer(**self.cfg).to(self.device)
        # A perceptual loss on top of L1, against the blur that L1 alone settles for.
        self.lpips_weight = lpips_weight
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval().requires_grad_(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
        self.lr = lr
        self.amp = self.device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.step = 0
        self.loader = DataLoader(
            self.train_set, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=workers,
            drop_last=True, persistent_workers=workers > 0,
        )
        self.batches = self._forever()
        # Fixed views to watch across training.
        self.watch = collate([self.val_set[i] for i in np.linspace(0, len(self.val_set) - 1, 6).astype(int)])
        ckpt = self.out / "ckpt.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.opt.load_state_dict(state["opt"])
            self.step = state["step"]
            print(f"resumed at step {self.step}")
        params = sum(p.numel() for p in self.model.parameters())
        print(f"{params / 1e6:.1f}M params, {len(self.train_set)} train / {len(self.val_set)} val views, {self.device}")

    def _forever(self):
        for _ in itertools.count():
            yield from self.loader

    def _lr(self, progress=None):
        """Warm-up from the first step; with `progress` (0..1 through a run), a cosine decay to 5%."""
        lr = self.lr * min(1.0, (self.step + 1) / 500)
        if progress is not None:
            lr *= 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return lr

    def forward(self, batch):
        feats, tex, pad, cams, images = (x.to(self.device, non_blocking=True) for x in batch)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            pred = self.model(feats, tex, pad, cams)
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
        """Mean (L1, LPIPS) over the first validation batches."""
        self.model.eval()
        loader = DataLoader(self.val_set, batch_size=16, shuffle=False, collate_fn=collate)
        ls = [[x.item() for x in self.losses(*self.forward(b))[1:]] for b in itertools.islice(loader, batches)]
        self.model.train()
        return tuple(float(x) for x in np.mean(ls, 0))

    @torch.no_grad()
    def sample(self, path):
        """Rows of target | prediction for the watched validation views."""
        self.model.eval()
        pred, target = self.forward(self.watch)
        self.model.train()
        rows = torch.cat([target, pred.clamp(0, 1)], 3)  # side by side
        grid = torch.cat(list(rows), 1)  # stacked vertically
        Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)

    def checkpoint(self):
        val_l1, val_lpips = self.validate()
        self.log(val_l1=val_l1, val_lpips=val_lpips)
        self.sample(self.out / f"sample_{self.step:07d}.png")
        self.sample(self.out / "sample_latest.png")
        state = {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step, "cfg": self.cfg}
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
