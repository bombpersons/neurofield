"""Distillation targets for the dream net (dream.py): each crop's creature optimised
straight into pixels against CLIP, drawn the way the net draws it, and saved so the
net can be trained on them with plain supervised losses.

    python train/dream_targets.py --data data/train --out data/dream_targets --count 64

Crops come from DreamCrops (fixed: an item's phrases and colour scheme are its own, so
any process makes the same ones). Like the net, a target mixes the scheme's colours,
shades the mix and has alpha from the parts' presence; on top of that, per crop, a
pyramid (coarse to fine, finer levels slower, so detail grows on structure) of mixing
weights, shading and alpha is optimised. It starts from what the net's input already
holds: the mix and shading from the parts' fixed noise through a fixed projection, so
a target follows from the net's input and the net can learn it rather than the average
of many. CLIP scores augmented cutouts (train_dream.cutouts) round each judged part
against its phrase, the whole crop against a whole-creature phrase and away from a
rainbow pattern. Alpha beyond the band round the parts is penalised, alpha short of 1
on their axes too, and towards the end alpha is pushed to 0 or 1.

Output: shards `targets_{part}_{k:04d}.npz`, each with the crops' inputs (q, cam,
palette, whole, judged; with `fields` they rebuild the net's input exactly), the level
frame, the targets [N, len(keep), n, n, 4] (uint8 RGBA at the `--keep` steps: the last
is the target, earlier ones are the way there) and CLIP's final scores; and `vocab.json`.
A process does every `parts`-th crop from `part` on, in a fixed shuffled order, and
skips shards that already exist, so runs can be split over GPUs and resumed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer

from creatures import composite
from dream import NEGATIVE_ID, PALETTE, Q_PHRASE, VOCAB, DreamCrops, fields
from train_dream import BAND, CLIP, CORE, MEAN, STD, backdrops, cutouts, embed

# Field channels: presence, then the part-fixed noise (dream.fields).
NOISE_CHANNELS = slice(9, 17)


class Pyramid(nn.Module):
    """[b, ch, n, n] as a sum of upsampled levels, coarse to fine, starting at 0."""

    def __init__(self, b, ch, n, levels=5, device=None):
        super().__init__()
        sizes = [n >> i for i in reversed(range(levels))]
        self.levels = nn.ParameterList([nn.Parameter(torch.zeros(b, ch, s, s, device=device)) for s in sizes])

    def forward(self):
        n = self.levels[-1].shape[-1]
        return sum(F.interpolate(p, size=(n, n), mode="bilinear", align_corners=False) for p in self.levels)


def draw(o, presence, palette):
    """The net's drawing (DreamNet.forward) from its raw output `o` [b, PALETTE + 2, n, n]."""
    mix = torch.einsum("bkhw,bkc->bchw", torch.softmax(o[:, :PALETTE] * 3, 1), palette)
    rgb = (mix * (0.35 + 1.3 * torch.sigmoid(o[:, PALETTE : PALETTE + 1] + 0.3))).clamp(0, 1)
    alpha = torch.sigmoid(10 * torch.tanh((o[:, PALETTE + 1 :] + 6 * presence - 3) / 10))
    return torch.cat([rgb, alpha], 1)


def clip_scores(clip, text, img, q, whole, judged, cuts, amp):
    """CLIP's mean similarity per crop of `img` [b, 3, n, n]: parts [b, J] (nan where none),
    whole [b] and rainbow [b] (as train_dream.Trainer.scores)."""
    b, _, n, _ = img.shape
    J = judged.shape[1]
    dev = img.device
    idx = judged[..., 0].long()
    centres = torch.cat([judged[..., 1:3], torch.full((b, 1, 2), n / 2, device=dev)], 1)
    extents = torch.cat([judged[..., 3], torch.full((b, 1), 0.9 * n, device=dev)], 1)
    valid = torch.cat([idx >= 0, torch.ones(b, 1, dtype=torch.bool, device=dev)], 1)
    sel = valid.flatten().nonzero()[:, 0]
    src = img.repeat_interleave(J + 1, 0)[sel]
    c = cutouts(src, centres.flatten(0, 1)[sel], extents.flatten()[sel], cuts)
    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
        f = embed(clip.get_image_features(pixel_values=(c - MEAN.to(dev)) / STD.to(dev))).float()
    f = f.reshape(len(sel), cuts, -1)
    phrase = torch.cat([torch.gather(q[..., Q_PHRASE], 1, idx.clamp(min=0)).long(), whole[:, None]], 1).flatten()[sel]
    sims = (f * text[phrase][:, None]).sum(-1).mean(1)
    neg = (f @ text[NEGATIVE_ID]).mean(1)
    table = torch.full((b * (J + 1),), float("nan"), device=dev)
    table[sel] = sims
    negs = torch.zeros(b * (J + 1), device=dev)
    negs[sel] = neg
    table, negs = table.reshape(b, J + 1), negs.reshape(b, J + 1)
    return table[:, :J], table[:, J], negs[:, J]


def optimise(clip, text, batch, steps, keep, cuts, amp, lr=0.03):
    """Targets for one batch of crops: [b, len(keep), 4, n, n] at the `keep` steps, and
    the final scores (parts [b, J], whole [b], rainbow [b])."""
    q, cam, frame, palette, whole, judged = batch
    b, _, n, _ = frame.shape
    dev = frame.device
    feats, _, dmin, _ = fields(q, cam, palette, n)
    presence = feats[:, :1]
    noise = feats[:, NOISE_CHANNELS]
    # The fixed start: colour mix and shading from the part's noise (the same projection
    # for every crop and run).
    g = torch.Generator().manual_seed(0)
    w = torch.randn(PALETTE + 1, noise.shape[1], generator=g).to(dev) / noise.shape[1] ** 0.5
    w[PALETTE] *= 0.5
    base = torch.cat([torch.einsum("kc,bchw->bkhw", w, noise), torch.zeros_like(presence)], 1)
    mix, shade, alpha = Pyramid(b, PALETTE, n, device=dev), Pyramid(b, 1, n, device=dev), Pyramid(b, 1, n, device=dev)
    groups = [{"params": [p], "lr": lr * 0.6**i} for i, p in enumerate(mix.levels)]
    groups += [{"params": [p], "lr": lr * 0.75**i} for i, p in enumerate(shade.levels)]
    groups += [{"params": [p], "lr": lr * 0.6**i} for i, p in enumerate(alpha.levels)]
    opt = torch.optim.Adam(groups)
    near, far = (dmin < CORE)[:, None].float(), (dmin > BAND)[:, None].float()
    kept = []
    for step in range(steps + 1):
        layer = draw(base + torch.cat([mix(), shade(), alpha()], 1), presence, palette)
        if step in keep:
            kept.append(layer.detach())
        if step == steps:
            break
        back = torch.where(torch.rand(b, 1, 1, 1, device=dev) < 0.5, frame, backdrops(b, n, dev))
        parts, whole_s, neg = clip_scores(clip, text, composite(back, layer), q, whole, judged, cuts, amp)
        a = layer[:, 3:]
        logit = torch.log(a) - torch.log1p(-a)
        ramp = min(1.0, max(0.0, (step / steps - 0.3) / 0.4))
        per_crop = (
            -parts.nan_to_num(0).sum(1)
            - 0.5 * whole_s
            + 0.3 * neg
            + 2.0 * ((a * far).mean((1, 2, 3)) + (F.relu(3 - logit) * near).mean((1, 2, 3)))
            + 0.02 * a.mean((1, 2, 3))
            + ramp * 0.5 * (a * (1 - a)).mean((1, 2, 3))
        )
        opt.zero_grad(set_to_none=True)
        per_crop.sum().backward()
        opt.step()
    with torch.no_grad():  # scored over the frame, as it will be seen
        scores = clip_scores(clip, text, composite(frame, layer), q, whole, judged, cuts, amp)
    return torch.stack(kept, 1), scores


def sample(path, frame, targets, dmin_presence, q, whole, judged):
    """Rows: frame | the target over it | over grey | alpha | presence | the earlier kept steps over grey."""
    final = targets[:, -1]
    grey = torch.full_like(frame, 0.5)
    rows = []
    for i in range(len(frame)):
        cols = [frame[i], composite(frame, final)[i], composite(grey, final)[i], final[i, 3:].expand(3, -1, -1),
                dmin_presence[i].expand(3, -1, -1)]
        cols += [composite(grey[:1], targets[i, k : k + 1])[0] for k in range(targets.shape[1] - 1)]
        rows.append(torch.cat(cols, 2))
    grid = torch.cat(rows, 1).clamp(0, 1)
    Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)
    lines = []
    for i in range(len(frame)):
        judged_phrases = [VOCAB[int(q[i, int(j), Q_PHRASE])] for j in judged[i, :, 0].tolist() if j >= 0]
        lines.append(f"row {i}: {VOCAB[int(whole[i])]} | judged: {'; '.join(judged_phrases) or '-'}")
    Path(path).with_suffix(".txt").write_text("\n".join(lines) + "\n")


def to_u8(x):
    return (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, help="crops for this process (default: all of its share)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--per-shard", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--keep", default="50,100,200", help="steps whose drawings are saved; the last is the target")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--cuts", type=int, default=4)
    ap.add_argument("--judged", type=int, default=3, help="parts per crop CLIP judges")
    ap.add_argument("--vocab", default="full", choices=["motifs", "full"])
    ap.add_argument("--minutes", type=float, help="stop before a shard that would not finish in time")
    ap.add_argument("--device")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    start = time.time()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dev.type == "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(VOCAB))
    keep = sorted({int(k) for k in a.keep.split(",")} | {a.steps})

    data = DreamCrops(a.data, size=a.size, fixed=True, vocab=a.vocab)
    data.JUDGED = a.judged
    order = np.random.default_rng(0).permutation(len(data))[a.part :: a.parts][: a.count]
    clip = CLIPModel.from_pretrained(CLIP).to(dev).eval().requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained(CLIP)
    with torch.no_grad():
        text = torch.cat([
            embed(clip.get_text_features(**tok(VOCAB[i : i + 256], padding=True, return_tensors="pt").to(dev)))
            for i in range(0, len(VOCAB), 256)
        ])
    print(f"{len(order)} crops of {len(data)} for part {a.part}/{a.parts}, {dev}, keeping steps {keep}", flush=True)

    shard_secs = None
    for k, s0 in enumerate(range(0, len(order), a.per_shard)):
        path = out / f"targets_{a.part}_{k:04d}.npz"
        if path.exists():
            continue
        if a.minutes and shard_secs and time.time() - start + shard_secs > a.minutes * 60:
            print("out of time", flush=True)
            break
        t0 = time.time()
        items = order[s0 : s0 + a.per_shard]
        rec = {name: [] for name in ("item", "q", "cam", "palette", "whole", "judged", "frame", "targets", "part_scores", "whole_score", "rainbow")}
        for b0 in range(0, len(items), a.batch):
            idx = items[b0 : b0 + a.batch]
            cols = list(zip(*[data[int(i)] for i in idx]))
            batch = [torch.stack(c).to(dev) for c in cols]
            torch.manual_seed(int(idx[0]))
            targets, (parts, whole_s, neg) = optimise(clip, text, batch, a.steps, keep, a.cuts, amp)
            q, cam, frame, palette, whole, judged = batch
            if k == 0 and b0 == 0:
                dmin = fields(q, cam, palette, a.size)[2]
                sample(out / f"sample_{a.part}.png", frame, targets, torch.exp(-0.5 * dmin * dmin)[:, None], q, whole, judged)
            for name, x in (("item", torch.as_tensor(idx)), ("q", q), ("cam", cam), ("palette", palette), ("whole", whole),
                            ("judged", judged), ("part_scores", parts), ("whole_score", whole_s), ("rainbow", neg)):
                rec[name].append(x.cpu().numpy())
            rec["frame"].append(to_u8(frame.permute(0, 2, 3, 1)))
            rec["targets"].append(to_u8(targets.permute(0, 1, 3, 4, 2)))
            print(json.dumps({"shard": k, "done": s0 + b0 + len(idx), "secs": round(time.time() - start),
                              "part": float(parts[~parts.isnan()].mean()) if (~parts.isnan()).any() else None, "whole": float(whole_s.mean())}), flush=True)
        np.savez_compressed(path.with_suffix(".tmp.npz"), **{name: np.concatenate(v) for name, v in rec.items()})
        path.with_suffix(".tmp.npz").rename(path)
        shard_secs = time.time() - t0


if __name__ == "__main__":
    main()
