"""Train the neural renderer.

Local smoke test:
    python train/train.py --data data/test --out runs/smoke --small --steps 30

On Colab, as a detached process that resumes from out/ckpt.pt (see colab_launch.py):
    python train.py --data /content/data/train --out /content/out --minutes 100 --batch 32 \
        --log-every 100 --sample-every 1000

Or inside a persistent kernel (state survives between `colab exec` calls):
    import sys; sys.path.insert(0, "/content/neurofield/train")
    from train import Trainer
    t = Trainer("/content/data/train", "/content/out")
    t.run(minutes=8)        # repeat; writes samples, checkpoints and log.jsonl
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import Shards, collate
from model import Renderer, loss_fn


class Trainer:
    def __init__(self, data, out, small=False, batch=16, lr=3e-4, workers=2, device=None):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        n_shards = len(list(Path(data).glob("shard_*")))
        assert n_shards >= 2, "need at least two shards (the last one is held out)"
        self.train_set = Shards(data, range(n_shards - 1))
        self.val_set = Shards(data, [n_shards - 1])
        self.cfg = dict(d=128, heads=4, enc_layers=2, dec_layers=2) if small else {}
        self.model = Renderer(**self.cfg).to(self.device)
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

    def _lr(self):
        return self.lr * min(1.0, (self.step + 1) / 500)

    def forward(self, batch):
        feats, tex, pad, cams, images = (x.to(self.device, non_blocking=True) for x in batch)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            pred = self.model(feats, tex, pad, cams)
        return pred.float(), images

    def run(self, minutes=None, steps=None, log_every=50, sample_every=500):
        self.model.train()
        start, done, losses = time.time(), 0, []
        while (steps is None or done < steps) and (minutes is None or time.time() - start < minutes * 60):
            for g in self.opt.param_groups:
                g["lr"] = self._lr()
            pred, target = self.forward(next(self.batches))
            loss = loss_fn(pred, target)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            losses.append(loss.item())
            self.step += 1
            done += 1
            if self.step % log_every == 0:
                self.log(train_l1=float(np.mean(losses[-log_every:])), secs=time.time() - start)
            if self.step % sample_every == 0:
                self.checkpoint()
        self.checkpoint()
        return float(np.mean(losses)) if losses else None

    @torch.no_grad()
    def validate(self, batches=8):
        self.model.eval()
        loader = DataLoader(self.val_set, batch_size=16, shuffle=False, collate_fn=collate)
        ls = [loss_fn(*self.forward(b)).item() for b in itertools.islice(loader, batches)]
        self.model.train()
        return float(np.mean(ls))

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
        val = self.validate()
        self.log(val_l1=val)
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
    a = ap.parse_args()
    t = Trainer(a.data, a.out, small=a.small, batch=a.batch, workers=a.workers)
    t.run(minutes=a.minutes, steps=a.steps, log_every=a.log_every, sample_every=a.sample_every)
