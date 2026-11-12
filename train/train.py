"""Train the neural renderer.

Local smoke test:
    python train/train.py --data data/test --out runs/smoke --small --steps 30

On Colab, run it inside the persistent kernel with train/colab_chunk.py: state
survives between `colab exec` calls, and a busy kernel keeps a free-tier VM
alive where a detached process does not. Mirror checkpoints locally with
train/colab_sync.sh in case the VM is reclaimed anyway.

This network draws the level alone: frames come without creatures (datagen
keeps them apart), and it gets no part tokens and no head noise. Creatures are
a network of their own, trained apart, so nothing they ask for reaches the
weights that draw the level (runs 5 and 6 showed it does harm there).
"""

import argparse
import copy
import itertools
import json
import math
import os
import shutil
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from data import Shards, WidthBatches, collate
from model import Renderer, loss_fn, migrate, migrate_deferred, migrate_head

# Steps of learning-rate warm-up after the optimizer starts afresh.
WARMUP = 500
# Longer when a new head starts from scratch on a trained body, so its first,
# random gradients don't knock the body about.
WARMUP_NEW_HEAD = 3000


class Trainer:
    def __init__(
        self, data, out, small=False, batch=16, lr=3e-4, workers=2, device=None, lpips_weight=0.1, ema_decay=0.999,
        up=None, deferred=False, freeze=10000, trunk_scale=0.1, organ_weight=0.0, on_save=None,
    ):
        """With `deferred`, the model gets (or keeps) the per-pixel stage (model.Deferred):
        added to a model without one, everything else is frozen for `freeze` steps and then
        learns at `trunk_scale` of the rate. `organ_weight` is how much more L1 counts on
        eyes, sphincters and veins (by the per-pixel stage's organ channels). `up` None keeps
        the checkpoint's head (another starts a fresh one). `on_save` is called after each
        checkpoint is written (to commit a volume, say)."""
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.on_save = on_save
        ckpt = self.out / "ckpt.pt"
        # On the CPU, where `migrate` makes its new weights; loading puts them on the device.
        state = None
        for f in (ckpt, self.out / "ckpt.prev.pt"):
            if state is None and f.exists():
                try:
                    state = torch.load(f, map_location="cpu")
                except Exception as e:  # cut short mid-write: the one before it
                    print(f"can't load {f}: {e}", flush=True)
        # A model that has the stage keeps it, and needs its inputs.
        deferred = deferred or bool(state and state.get("cfg", {}).get("deferred"))
        up = up or (state.get("cfg", {}).get("up", "shuffle") if state else "shuffle")
        n_shards = len(list(Path(data).glob("shard_*")))
        assert n_shards >= 2, "need at least two shards (the last one is held out)"
        self.train_set = Shards(data, range(n_shards - 1), pixels=deferred)
        self.val_set = Shards(data, [n_shards - 1], pixels=deferred)
        self.organ_weight = organ_weight
        # Until which step everything but the per-pixel stage stands still, and how fast it
        # learns after.
        self.freeze_until, self.trunk_scale = 0, trunk_scale
        self.cfg = dict(d=128, heads=4, enc_layers=2, dec_layers=2) if small else {}
        self.cfg["up"] = up
        if deferred:
            self.cfg["deferred"] = True
        migrated, warmup = False, WARMUP
        if state is not None:
            self.cfg = state.get("cfg", self.cfg)
            # From before widths varied or creature parts: start both copies from the averaged
            # weights, migrated.
            weights = state.get("ema", state["model"])
            if "q_pos" in weights:  # the old patch head goes: its replacement needs the long warm-up
                warmup = WARMUP_NEW_HEAD
            weights, cfg, migrated = migrate(weights, self.cfg)
            weights, cfg, new_head = migrate_head(weights, cfg, up)
            if new_head:
                warmup = WARMUP_NEW_HEAD
                migrated = True
            if deferred:
                weights, cfg, added = migrate_deferred(weights, cfg)
                if added:
                    # The rest stands still while the stage starts: a fresh head would stay random.
                    assert not new_head, "a new head and a new stage at once: train the head first"
                    migrated = True
                    self.freeze_until = state["step"] + freeze
            self.freeze_until = max(self.freeze_until, state.get("freeze_until", 0))
            if migrated:
                state = {**state, "model": weights, "ema": weights, "opt": None}
                self.cfg = cfg
        self.model = Renderer(**self.cfg).to(self.device)
        # A perceptual loss on top of L1, against the blur that L1 alone settles for.
        self.lpips_weight = lpips_weight
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval().requires_grad_(False)
        covers = ("cover", "part_cover", "organ_cover")
        stage = lambda n: n.startswith("deferred.")  # noqa: E731
        main = [p for n, p in self.model.named_parameters() if n not in covers and not stage(n)]
        self.opt = torch.optim.AdamW(main, lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
        self.opt.add_param_group({"params": [getattr(self.model, n) for n in covers], "weight_decay": 0.0})
        if self.model.deferred is not None:
            # The latent textures aren't decayed: they are the stage's memory, not weights.
            latents = ("deferred.level", "deferred.flesh", "deferred.eye", "deferred.iris")
            mlp = [p for n, p in self.model.named_parameters() if stage(n) and n not in latents]
            self.opt.add_param_group({"params": mlp, "stage": True})
            self.opt.add_param_group({"params": [p for n, p in self.model.named_parameters() if n in latents], "weight_decay": 0.0, "stage": True})
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
                assert self.model.deferred is None, "optimizer groups don't match the checkpoint's"
                cover = self.opt.param_groups.pop()
                self.opt.load_state_dict(state["opt"])
                self.opt.add_param_group(cover)
            self.step = state["step"]
            print(f"resumed at step {self.step}" + (" (migrated: fresh optimizer)" if migrated else "")
                  + (f", new {missing}" if missing else ""))
        # How far through the learning rate's decay the run the checkpoint was saved in got
        # (minutes spent of minutes in all), for a run that resumes it.
        self.clock = state.get("clock") if state is not None else None
        # A slow average of the weights, used for validation, samples and export: steadier than the
        # weights of any one step.
        self.ema_decay = ema_decay
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        if state is not None and "ema" in state:
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

    def _scale(self, group):
        """How much of the learning rate a parameter group gets: all of it for the per-pixel
        stage; for the rest, none while frozen and then `trunk_scale` (once there is a stage)."""
        if group.get("stage") or self.model.deferred is None:
            return 1.0
        return 0.0 if self.step < self.freeze_until else self.trunk_scale

    def _freeze(self):
        """Gradients only for the groups that learn this step: a frozen trunk costs no backward
        and takes no part in clipping."""
        for g in self.opt.param_groups:
            learn = self._scale(g) > 0
            for p in g["params"]:
                p.requires_grad_(learn)

    def probe(self):
        """One training step's forward and backward on a batch of the widest views, without
        stepping: an out-of-memory shows now rather than hours in. Returns the peak GiB."""
        widths = np.array(self.train_set.widths)
        idx = np.flatnonzero(widths == widths.max())[: self.loader.batch_sampler.batch]
        self.model.train().requires_grad_(True)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pred, target, weight = self.forward(collate([self.train_set[i] for i in idx]), weight=True)
        loss = self.losses(pred, target, weight)[0]
        self.scaler.scale(loss).backward()
        self.opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2**30 if self.device == "cuda" else 0.0
        print(f"probe: {len(idx)} views {int(widths.max())} wide, peak {peak:.1f} GiB", flush=True)
        return peak

    def forward(self, batch, model=None, weight=False):
        """(prediction, target) for a batch: the level alone. With `weight`, also how much
        each pixel's L1 counts [b, 240, width] (None: all alike)."""
        feats, tex, pad, cams, images = (x.to(self.device, non_blocking=True) for x in batch[:5])
        organs = tuple(x.to(self.device, non_blocking=True) for x in batch[6])
        model = model or self.model
        px = None
        if model.deferred is not None and len(batch) > 7:
            from pixels import inputs

            px = inputs(batch[7], self.device)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            pred = model(feats, tex, pad, cams, width=images.shape[3], organs=organs, pixels=px)
        if not weight:
            return pred.float(), images
        w = None
        if px is not None and self.organ_weight > 0:
            plain = px[0]
            organ = (plain[:, 9] + plain[:, 10] + (plain[:, 18] > 0.5).float()).clamp(max=1)
            w = 1 + self.organ_weight * organ
        return pred.float(), images, w

    def losses(self, pred, target, weight=None):
        """(training loss, L1, LPIPS) for a batch; `weight` [b, 240, width] weighs the L1 the
        loss uses (the L1 it reports is plain)."""
        l1 = loss_fn(pred, target)
        fit = l1 if weight is None else ((pred - target).abs().mean(1) * weight).sum() / weight.sum()
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            perceptual = self.lpips(pred * 2 - 1, target * 2 - 1).float().mean()
        return fit + self.lpips_weight * perceptual, l1, perceptual

    def run(self, minutes=None, steps=None, log_every=50, sample_every=500, decay=False, spent=0.0, total=None):
        """Train for `minutes` and/or `steps`. With `decay`, the learning rate follows a cosine
        over this run (by time when `minutes` is given), so each run ends settled; the next
        run restarts at the full rate. A run picked up partway (after a restart) passes the
        minutes it already `spent` of `total`, and the decay carries on from there."""
        self.model.train()
        total = total or (spent + minutes if minutes else None)
        start, done, l1s, lps = time.time(), 0, [], []
        self.clock = (spent, total)
        frozen = None
        while (steps is None or done < steps) and (minutes is None or time.time() - start < minutes * 60):
            progress = None
            if decay:
                progress = (spent + (time.time() - start) / 60) / total if minutes else done / steps
            self.clock = (spent + (time.time() - start) / 60, total)
            for g in self.opt.param_groups:
                g["lr"] = self._lr(progress) * self._scale(g)
            if frozen != (self.step < self.freeze_until):
                frozen = self.step < self.freeze_until
                self._freeze()
            pred, target, weight = self.forward(next(self.batches), weight=True)
            loss, l1, perceptual = self.losses(pred, target, weight)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            with torch.no_grad():
                torch._foreach_lerp_(list(self.ema.parameters()), list(self.model.parameters()), 1 - self.ema_decay)
            l1s.append(l1.item())
            lps.append(perceptual.item())
            self.step += 1
            done += 1
            if self.step % log_every == 0:
                recent = lambda xs: float(np.mean(xs[-log_every:]))  # noqa: E731
                self.log(train_l1=recent(l1s), train_lpips=recent(lps), lr=self.opt.param_groups[0]["lr"], stage_lr=self.opt.param_groups[-1]["lr"], secs=time.time() - start)
            if self.step % sample_every == 0:
                self.checkpoint()
        self.checkpoint()
        return float(np.mean(l1s)) if l1s else None

    @torch.no_grad()
    def validate(self, batches=8):
        """(L1, LPIPS) of the averaged weights over the first validation batches."""
        loader = DataLoader(self.val_set, batch_sampler=WidthBatches(self.val_set.widths, 16, seed=0, partial=True), collate_fn=collate)
        ls = []
        for b in itertools.islice(loader, batches):
            pred, target = self.forward(b, self.ema)
            ls.append([x.item() for x in self.losses(pred, target)[1:]])
        return tuple(float(x) for x in np.mean(ls, 0))

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
        """Save, then validate and sample (which can fail without costing the save). The
        checkpoint is written beside the old one and swapped in, the old kept as ckpt.prev.pt,
        so a run stopped mid-save leaves a whole one."""
        state = {
            "model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
            "step": self.step, "cfg": self.cfg, "warm": (self.warm_from, self.warmup), "freeze_until": self.freeze_until,
            "clock": self.clock,
        }
        ckpt = self.out / "ckpt.pt"
        torch.save(state, self.out / "ckpt.pt.tmp")
        if ckpt.exists():
            shutil.copyfile(ckpt, self.out / "ckpt.prev.pt")
        os.replace(self.out / "ckpt.pt.tmp", ckpt)
        if self.on_save:
            self.on_save()
        val_l1, val_lpips = self.validate()
        self.log(val_l1=val_l1, val_lpips=val_lpips, cover=self.model.cover.abs().mean().item())
        self.sample(self.out / f"sample_{self.step:07d}.png")
        self.sample(self.out / "sample_latest.png")

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
    ap.add_argument("--up", default=None, choices=["shuffle", "resize"], help="the upsampling head; another than the checkpoint's starts fresh")
    ap.add_argument("--deferred", action="store_true", help="add (or keep) the per-pixel stage")
    ap.add_argument("--freeze", type=int, default=10000, help="steps the rest stands still when the stage is new")
    ap.add_argument("--organ-weight", type=float, default=0.0, help="extra weight of L1 on organs")
    a = ap.parse_args()
    t = Trainer(a.data, a.out, small=a.small, batch=a.batch, workers=a.workers, lpips_weight=a.lpips, up=a.up,
                deferred=a.deferred, freeze=a.freeze, organ_weight=a.organ_weight)
    t.run(minutes=a.minutes, steps=a.steps, log_every=a.log_every, sample_every=a.sample_every, decay=a.decay)
