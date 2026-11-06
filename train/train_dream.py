"""Train the dream creature network (dream.py) with CLIP as its only teacher.

    python train/train_dream.py --data data/train --out runs/dream --steps 30

There are no targets. Each crop's creature is laid over the level's frame (or, half
the time, a plain blotchy backdrop, so see-through flesh does not pay), and CLIP
ViT-B/32 scores augmented cutouts of it: round each judged part, which must match
its phrase better than every other part phrase (a softmax over them, as CLIP was
trained, so hiding the creature gains nothing), the whole crop against a
whole-creature phrase, and the whole crop pushed away from a rainbow pattern. Augmentations (scale, shift, flip, rotation, shear,
blur, colour jitter, noise) make CLIP pay only for structure that survives them.
Alpha far from every part (beyond `BAND` radii) is penalised, and so is alpha short of
1 right by a part's axis (`CORE`); the silhouette between is the network's. Later in
training alpha is pushed towards 0 or 1.

The phrases' CLIP text embeddings are saved in the checkpoint (`text`, with `vocab`),
so the engine needs a table, not CLIP.
"""

import argparse
import copy
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPTokenizer

from creatures import composite
from dream import NEGATIVE_ID, Q_PHRASE, VOCAB, VOCABS, DreamCrops, DreamNet, fields, fogged

CLIP = "openai/clip-vit-base-patch32"
MEAN = torch.tensor([0.4815, 0.4578, 0.4082])[:, None, None]
STD = torch.tensor([0.2686, 0.2613, 0.2758])[:, None, None]
WARMUP = 500
# Alpha beyond this many radii from every part's axis is penalised; within CORE radii
# of one, alpha short of 1 is.
BAND, CORE = 2.5, 0.5
# CLIP's logit scale, for telling a part's phrase from all the others.
LOGIT_SCALE = 100.0
# Steps over which alpha is pushed towards 0 or 1: from, then ramping up over.
BINARY_FROM, BINARY_RAMP = 2000, 4000
# Non-finite steps in a row after which training goes back to the last good checkpoint.
BAD_STEPS = 20


def embed(f):
    """Unit CLIP embeddings (transformers 5 wraps them in an output object)."""
    return F.normalize(f if torch.is_tensor(f) else f.pooler_output, dim=-1)


def backdrops(b, n, device):
    """Plain colours with soft blotches [b, 3, n, n]."""
    base = torch.rand(b, 3, 1, 1, device=device)
    blot = F.interpolate(torch.rand(b, 3, 8, 8, device=device), size=(n, n), mode="bilinear", align_corners=False)
    return (base * 0.7 + blot * 0.3).clamp(0, 1)


def cutouts(img, centre, extent, cuts):
    """`cuts` augmented 224 px cutouts of each image [b, 3, n, n] round `centre` [b, 2]
    (pixels), about `extent` [b] across: [b * cuts, 3, 224, 224] (image-major)."""
    b, _, n, _ = img.shape
    dev = img.device
    img = img.repeat_interleave(cuts, 0)
    centre, extent = centre.repeat_interleave(cuts, 0), extent.repeat_interleave(cuts, 0)
    m = len(img)
    u = lambda lo, hi: torch.empty(m, device=dev).uniform_(lo, hi)  # noqa: E731
    size = (extent * u(0.6, 1.7)).clamp(24, n)
    c = (centre + u(-0.2, 0.2)[:, None] * size[:, None]) / n * 2 - 1
    s = size / n
    ang, shear = u(-0.35, 0.35), u(-0.15, 0.15)
    flip = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.stack(
        [
            torch.stack([s * torch.cos(ang) * flip, s * (-torch.sin(ang) + shear), c[:, 0]], -1),
            torch.stack([s * torch.sin(ang) * flip, s * torch.cos(ang), c[:, 1]], -1),
        ],
        1,
    )
    grid = F.affine_grid(theta, (m, 3, 224, 224), align_corners=False)
    out = F.grid_sample(img, grid, padding_mode="reflection", align_corners=False)
    blur = torch.rand(m, device=dev) < 0.3  # sometimes seen blurred, as from further away
    soft = F.interpolate(F.interpolate(out, size=(112, 112), mode="area"), size=(224, 224), mode="bilinear", align_corners=False)
    out = torch.where(blur[:, None, None, None], soft, out)
    out = (out - 0.5) * u(0.85, 1.15)[:, None, None, None] + 0.5 + u(-0.06, 0.06)[:, None, None, None]
    return out + torch.randn_like(out) * 0.04


class Trainer:
    def __init__(self, data, out, batch=16, size=128, lr=3e-4, workers=2, device=None, cuts=4, ema_decay=0.999, vocab="motifs",
                 part_weight=0.1, whole_weight=0.5, negative_weight=0.3, band_weight=2.0, binary_weight=0.5):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.device == "cuda"
        n_shards = len(list(Path(data).glob("shard_*")))
        assert n_shards >= 2, "need at least two shards (the last one is held out)"
        self.train_set = DreamCrops(data, range(n_shards - 1), size, vocab=vocab)
        self.val_set = DreamCrops(data, [n_shards - 1], size, fixed=True, vocab=vocab)
        self.clip = CLIPModel.from_pretrained(CLIP).to(self.device).eval().requires_grad_(False)
        tok = CLIPTokenizer.from_pretrained(CLIP)
        with torch.no_grad():
            self.text = torch.cat([
                embed(self.clip.get_text_features(**tok(VOCAB[i : i + 256], padding=True, return_tensors="pt").to(self.device)))
                for i in range(0, len(VOCAB), 256)
            ])
        self.mean, self.std = MEAN.to(self.device), STD.to(self.device)
        # The phrases a part's phrase competes with, and each phrase's place among them.
        self.candidates = torch.tensor(VOCABS[vocab], device=self.device)
        self.position = torch.zeros(len(VOCAB), dtype=torch.long, device=self.device)
        self.position[self.candidates] = torch.arange(len(self.candidates), device=self.device)
        self.vocab = vocab
        self.model = DreamNet(text_dim=self.text.shape[1]).to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.lr, self.cuts, self.ema_decay = lr, cuts, ema_decay
        self.part_weight, self.whole_weight, self.negative_weight = part_weight, whole_weight, negative_weight
        self.band_weight, self.binary_weight = band_weight, binary_weight
        self.step = 0
        ckpt = self.out / "ckpt.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.ema.load_state_dict(state["ema"])
            self.opt.load_state_dict(state["opt"])
            self.step = state["step"]
            self.lr = state.get("lr", self.lr)
            print(f"resumed at step {self.step}")
        self.good = None
        self.keep_good()
        assert self.good is not None, "the weights are not finite: resume from an earlier checkpoint"
        self.loader = DataLoader(self.train_set, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                                 persistent_workers=workers > 0)
        self.batches = (b for _ in itertools.count() for b in self.loader)
        self.watch = next(iter(DataLoader(self.val_set, batch_size=6, sampler=np.linspace(0, len(self.val_set) - 1, 6).astype(int))))
        params = sum(p.numel() for p in self.model.parameters())
        print(f"{params / 1e6:.2f}M params, {len(VOCAB)} phrases, {len(self.train_set)} train / {len(self.val_set)} val crops, {self.device}")

    def forward(self, batch, model=None):
        """(layer [b, 4, n, n], level frame, distance from the parts, fog, the batch on the device)."""
        q, cam, frame, palette, whole, judged = (x.to(self.device, non_blocking=True) for x in batch)
        n = frame.shape[-1]
        with torch.no_grad():
            feats, weights, dmin, fog = fields(q, cam, palette, n)
        text = self.text[q[..., Q_PHRASE].long()]
        # In fp32: the net is small, and without normalisation its activations can outgrow
        # fp16 (run 1 went NaN on every step from then on). CLIP stays in fp16.
        layer = (model or self.model)(feats, weights, text, palette)
        return layer.float(), frame, dmin, fog, (q, cam, palette, whole, judged)

    def scores(self, img, q, whole, judged):
        """For composites `img` [b, 3, n, n]: CLIP's mean similarity per item (parts [b, JUDGED]
        (nan where none), whole [b], rainbow [b]) and the parts' cross-entropy against all
        part phrases (mean over judged parts)."""
        b, _, n, _ = img.shape
        J = judged.shape[1]
        idx = judged[..., 0].long()
        centres = torch.cat([judged[..., 1:3], torch.full((b, 1, 2), n / 2, device=img.device)], 1)
        extents = torch.cat([judged[..., 3], torch.full((b, 1), 0.9 * n, device=img.device)], 1)
        valid = torch.cat([idx >= 0, torch.ones(b, 1, dtype=torch.bool, device=img.device)], 1)
        # Only valid cutouts go through CLIP.
        sel = valid.flatten().nonzero()[:, 0]
        src = img.repeat_interleave(J + 1, 0)[sel]
        cuts = cutouts(src, centres.flatten(0, 1)[sel], extents.flatten()[sel], self.cuts)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            f = embed(self.clip.get_image_features(pixel_values=(cuts - self.mean) / self.std)).float()
        f = f.reshape(len(sel), self.cuts, -1)
        phrase = torch.cat([torch.gather(q[..., Q_PHRASE], 1, idx.clamp(min=0)).long(), whole[:, None]], 1).flatten()[sel]
        sims = (f * self.text[phrase][:, None]).sum(-1).mean(1)
        part_rows = (~(torch.arange(J + 1, device=img.device) == J).expand(b, -1)).flatten()[sel]
        logits = LOGIT_SCALE * f[part_rows] @ self.text[self.candidates].T  # [parts, cuts, vocab phrases]
        target = self.position[phrase[part_rows]][:, None].expand(-1, self.cuts)
        ce = F.cross_entropy(logits.flatten(0, 1), target.flatten()) if part_rows.any() else img.sum() * 0
        neg = (f @ self.text[NEGATIVE_ID]).mean(1)
        table = torch.full((b * (J + 1),), math.nan, device=img.device)
        table[sel] = sims
        table = table.reshape(b, J + 1)
        negs = torch.zeros(b * (J + 1), device=img.device)
        negs[sel] = neg
        return table[:, :J], table[:, J], negs.reshape(b, J + 1)[:, J], ce

    def losses(self, layer, frame, dmin, q, whole, judged):
        """(loss, part cross-entropy, mean part score, mean whole score, rainbow score, alpha
        outside the band, alpha missing from the core)."""
        b, _, n, _ = frame.shape
        back = torch.where(torch.rand(b, 1, 1, 1, device=frame.device) < 0.5, frame, backdrops(b, n, frame.device))
        parts, whole_s, neg, ce = self.scores(composite(back, layer), q, whole, judged)
        part_s = parts.nan_to_num(0).sum() / (~parts.isnan()).sum().clamp(min=1)
        alpha = layer[:, 3:]
        outside = (alpha * (dmin[:, None] > BAND)).mean()
        # On alpha's logit, so a creature that has vanished is still pulled back.
        logit = torch.log(alpha) - torch.log1p(-alpha)
        missing = (F.relu(3 - logit) * (dmin[:, None] < CORE)).mean()
        ramp = min(1.0, max(0.0, (self.step - BINARY_FROM) / BINARY_RAMP))
        loss = (
            self.part_weight * ce
            - self.whole_weight * whole_s.mean()
            + self.negative_weight * neg.mean()
            + self.band_weight * (outside + missing)
            + 0.02 * alpha.mean()
            + ramp * self.binary_weight * (alpha * (1 - alpha)).mean()
        )
        return loss, ce, part_s, whole_s.mean(), neg.mean(), outside, missing

    def run(self, minutes=None, steps=None, log_every=50, sample_every=1000):
        self.model.train()
        start, done, bad = time.time(), 0, 0
        logs = {k: [] for k in ("loss", "part_ce", "part", "whole", "rainbow", "outside", "missing")}
        skipped = []
        while (steps is None or done < steps) and (minutes is None or time.time() - start < minutes * 60):
            for g in self.opt.param_groups:
                g["lr"] = self.lr * min(1.0, (self.step + 1) / WARMUP)
            layer, frame, dmin, _, (q, _, _, whole, judged) = self.forward(next(self.batches))
            out = self.losses(layer, frame, dmin, q, whole, judged)
            loss = out[0]
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            finite = bool(torch.isfinite(loss))
            if finite:
                self.scaler.step(self.opt)
            self.scaler.update()
            skipped.append(0 if finite else 1)
            bad = 0 if finite else bad + 1
            if bad >= BAD_STEPS:  # stuck: back to the last good state, more gently
                self.restore_good()
                bad = 0
            with torch.no_grad():
                torch._foreach_lerp_(list(self.ema.parameters()), list(self.model.parameters()), 1 - self.ema_decay)
            for k, v in zip(logs, out):
                logs[k].append(v.item())
            self.step += 1
            done += 1
            if self.step % log_every == 0:
                recent = {f"train_{k}": float(np.nanmean(v[-log_every:])) for k, v in logs.items()}
                self.log(**recent, nonfinite=int(sum(skipped[-log_every:])), lr=self.lr, secs=time.time() - start)
            if self.step % sample_every == 0:
                self.checkpoint()
        self.checkpoint()

    @torch.no_grad()
    def validate(self, batches=8):
        loader = DataLoader(self.val_set, batch_size=16)
        torch.manual_seed(0)
        rs = []
        for b in itertools.islice(loader, batches):
            layer, frame, dmin, _, (q, _, _, whole, judged) = self.forward(b, self.ema)
            rs.append([x.item() for x in self.losses(layer, frame, dmin, q, whole, judged)[1:]])
        names = ("val_part_ce", "val_part", "val_whole", "val_rainbow", "val_outside", "val_missing")
        return dict(zip(names, np.mean(rs, 0).tolist()))

    @torch.no_grad()
    def sample(self, path):
        """Rows of the level frame | the creatures over it (fogged, as the game shows them) |
        over grey, unfogged | alpha | the parts' presence. Their phrases go in a .txt beside it."""
        layer, frame, dmin, fog, (q, cam, _, whole, judged) = self.forward(self.watch, self.ema)
        grey = torch.full_like(frame, 0.5)
        shown = composite(frame, fogged(layer, fog, cam))
        presence = torch.exp(-0.5 * dmin * dmin)[:, None]
        rows = [torch.cat([frame[i], shown[i], composite(grey, layer)[i], layer[i, 3:].expand(3, -1, -1),
                           presence[i].expand(3, -1, -1)], 2) for i in range(len(layer))]
        grid = torch.cat(rows, 1).clamp(0, 1)
        Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)
        lines = []
        for i in range(len(layer)):
            judged_phrases = [VOCAB[int(q[i, int(j), Q_PHRASE])] for j in judged[i, :, 0].tolist() if j >= 0]
            lines.append(f"row {i}: {VOCAB[int(whole[i])]} | judged: {'; '.join(judged_phrases) or '-'}")
        Path(path).with_suffix(".txt").write_text("\n".join(lines) + "\n")

    def finite(self):
        return all(bool(torch.isfinite(p).all()) for m in (self.model, self.ema) for p in m.parameters())

    def keep_good(self):
        """Remember the current state (on the CPU) to come back to if training goes non-finite,
        if it is finite itself."""
        if not self.finite():
            return
        self.good = {"model": copy.deepcopy(self.model.state_dict()), "ema": copy.deepcopy(self.ema.state_dict()),
                     "opt": copy.deepcopy(self.opt.state_dict()), "step": self.step}
        self.good = {k: (v if k == "step" else {n: t.cpu() if torch.is_tensor(t) else t for n, t in v.items()})
                     for k, v in self.good.items()}

    def restore_good(self):
        self.model.load_state_dict(self.good["model"])
        self.ema.load_state_dict(self.good["ema"])
        self.opt.load_state_dict(self.good["opt"])
        self.lr *= 0.5
        self.log(restored_from=self.good["step"], lr=self.lr)

    def checkpoint(self):
        val = self.validate()
        self.log(**val)
        self.sample(self.out / f"sample_{self.step:07d}.png")
        self.sample(self.out / "sample_latest.png")
        if not (self.finite() and all(math.isfinite(v) for v in val.values())):
            self.log(skipped_checkpoint="non-finite weights or validation")
            return
        self.keep_good()
        torch.save({"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                    "step": self.step, "lr": self.lr, "vocab": VOCAB, "vocab_set": self.vocab, "text": self.text.cpu()}, self.out / "ckpt.pt")

    def log(self, **kv):
        rec = {"step": self.step, **kv}
        print(json.dumps(rec), flush=True)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--cuts", type=int, default=4)
    ap.add_argument("--vocab", default="motifs", choices=["motifs", "full"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=1000)
    a = ap.parse_args()
    t = Trainer(a.data, a.out, batch=a.batch, size=a.size, workers=a.workers, cuts=a.cuts, vocab=a.vocab, lr=a.lr)
    t.run(minutes=a.minutes, steps=a.steps, log_every=a.log_every, sample_every=a.sample_every)


if __name__ == "__main__":
    main()
