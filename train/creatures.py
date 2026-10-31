"""Creatures as a network of their own: from per-pixel maps of their parts' rough
shapes to the creatures in full colour, as an RGBA layer laid over the level's
frame (see format.json: datagen keeps them apart).

The engine draws the same maps the same way, so this file is what it ports
(to WGSL): every part is the closed tube `Part::profile` describes round its bent
axis (creature.rs), without lumps, smooth joins or sculpted veins and warts,
which are the network's to invent. Each pixel's ray is marched through every
part's box in its own frame; the nearest part it enters is the pixel's.

The maps, per pixel (`MAP_CHANNELS`): whether a part is there, its depth, normal
(camera axes), the point in the part's frame (as fractions of its size), the
part's baked light at that normal and the torch's, fog and the fog's colour, the
skin's density and tint, and `NOISE` channels of noise at the point, in the part's
own frame and measured in skin repeats, so it moves with the part. Plus the
part's shape and skin motif as ids, for embeddings. All zero where no part is.

Creature views are trained on crops round their creatures (`Crops`).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from data import H, Shards, view_width

# Most parts one crop considers, nearest first.
MAX_PARTS = 64
# March steps through a part's box, then halvings of the step that entered it.
STEPS, HALVINGS = 32, 5
# Noise: channels, and octaves (each twice the last's frequency, half its strength).
NOISE, OCTAVES = 6, 3
MAP_CHANNELS = 1 + 1 + 3 + 3 + 3 + 3 + 1 + 3 + 1 + 3 + NOISE
NUM_SHAPES, NUM_MOTIFS = 4, 11
# Part parameters as `view_parts` lays them out (camera space, tiles).
P_CENTRE, P_AXES, P_HALF, P_LO, P_HI = slice(0, 3), slice(3, 12), slice(12, 15), slice(15, 18), slice(18, 21)
P_SHAPE, P_TAPER, P_BEND, P_BULGE = 21, 22, 23, 24
P_MOTIF, P_DENSITY, P_REPEAT, P_TINT, P_SH = 25, 26, 27, slice(28, 31), slice(31, 43)
P_SEED = 43
PART_PARAMS = 44
# Per-view values the maps need: torch rgb, torch reach, fog near, fog far (tiles), fog colour.
CAM = 9


def view_parts(parts, v, seeds):
    """Scene parts [P, 49] (parts.npy) in camera space for view `v`, nearest first, at most
    MAX_PARTS, as float32 [P, PART_PARAMS]: centre, axes (as columns), half sizes, local
    box lo and hi, shape, taper, bend, bulge, motif, density, tiles per skin repeat, tint,
    baked light (as parts.npy, its gradient turned into camera axes) and a noise seed."""
    pos, R = v[1:4], v[4:13].reshape(3, 3)
    centre = (parts[:, 3:6] - pos) @ R.T
    axes = R @ parts[:, 6:15].reshape(-1, 3, 3)
    grad = np.einsum("ij,pjc->pic", R, parts[:, 38:47].reshape(-1, 3, 3)).reshape(-1, 9)
    out = np.concatenate(
        [centre, axes.reshape(-1, 9), parts[:, 15:28], parts[:, 29:35], parts[:, 35:38], grad, seeds[:, None]], 1
    ).astype(np.float32)
    ahead = centre[:, 2] + np.abs(parts[:, 15:18]).max(1) > 0
    out, dist = out[ahead], np.linalg.norm(centre[ahead], axis=1)
    return out[np.argsort(dist)[:MAX_PARTS]]


def cam_values(v):
    return np.concatenate([v[19:22], v[22:23], v[14:16], v[23:26]]).astype(np.float32)


def profile(shape, taper, bulge, t):
    """`Part::profile`: the radius at t (-1..1 along the axis), 1 = the part's."""
    cap = lambda e, k: (e / k).clamp(0, 1).sqrt()  # noqa: E731
    ell = (1 - t * t).clamp(min=0).sqrt()
    e = (t.abs() - 0.7) / 0.3
    capsule = torch.where(e <= 0, torch.ones_like(t), (1 - e * e).clamp(min=0).sqrt())
    cone = (1 - t) / 2 * cap(t + 1, 0.15)
    tentacle = (1 - 0.8 * (t + 1) / 2) * (1 + 0.12 * torch.sin(t * 6 * math.pi)) * cap(t + 1, 0.1) * cap(1 - t, 0.1)
    r = torch.where(shape == 0, ell, torch.where(shape == 1, capsule, torch.where(shape == 2, cone, tentacle)))
    return r * (1 + (taper - 1) * (t + 1) / 2) * (1 + bulge * (1 - t * t))


def outside(p, q):
    """How far outside part `q` its local point `p` [..., 3] is (< 0 inside), in its radii.
    `q` [..., PART_PARAMS] broadcasts against p's leading shape."""
    half = q[..., P_HALF].clamp(min=1e-4)
    t = p[..., 1] / half[..., 1]
    s = (t + 1) / 2
    x = (p[..., 0] - q[..., P_BEND] * half[..., 1] * s * s) / half[..., 0]
    z = p[..., 2] / half[..., 2]
    r = profile(q[..., P_SHAPE], q[..., P_TAPER], q[..., P_BULGE], t.clamp(-1, 1))
    f = (x * x + z * z).sqrt() - r
    return torch.where(t.abs() > 1, f.abs() + 4 * (t.abs() - 1) + 0.1, f)


def rays(x0, y0, w, h, v_h, width):
    """Camera-space ray directions [h, w, 3] (z = 1) through the centres of a box's pixels."""
    ys, xs = torch.meshgrid(torch.arange(h) + y0 + 0.5, torch.arange(w) + x0 + 0.5, indexing="ij")
    return torch.stack([(xs - width / 2) / v_h, (ys - H / 2) / v_h, torch.ones_like(xs)], -1).float()


def march(q, d):
    """Where rays `d` [B, h, w, 3] from the eye first enter parts `q` [B, P, PART_PARAMS]
    (padding has zero size): ray distance [B, P, h, w] in units of d (inf for a miss), and
    the rays in each part's frame (origin [B, P, 1, 1, 3], direction [B, P, h, w, 3])."""
    B, P = q.shape[:2]
    axes = q[..., P_AXES].reshape(B, P, 3, 3)  # columns are the local axes
    o = -torch.einsum("bpji,bpj->bpi", axes, q[..., P_CENTRE])[:, :, None, None]
    dl = torch.einsum("bpji,bhwj->bphwi", axes, d)
    lo, hi = q[..., P_LO][:, :, None, None], q[..., P_HI][:, :, None, None]
    inv = 1 / torch.where(dl.abs() < 1e-6, torch.full_like(dl, 1e-6), dl)
    a, b = (lo - o) * inv, (hi - o) * inv
    near = torch.minimum(a, b).amax(-1).clamp(min=0.02)
    far = torch.maximum(a, b).amin(-1)
    valid = (far > near) & (q[..., 13] > 0)[:, :, None, None]
    qq = q[:, :, None, None]
    step = (far - near) / STEPS
    hit = torch.full_like(near, math.inf)
    before = near.clone()
    for k in range(STEPS + 1):
        t = near + step * k
        enter = valid & torch.isinf(hit) & (outside(o + dl * t[..., None], qq) < 0)
        before = torch.where(enter, (t - step).clamp(min=0), before)
        hit = torch.where(enter, t, hit)
    found = torch.isfinite(hit)
    a, b = before, torch.where(found, hit, before)
    for _ in range(HALVINGS):
        m = (a + b) / 2
        inn = outside(o + dl * m[..., None], qq) < 0
        a, b = torch.where(inn, a, m), torch.where(inn, m, b)
    return torch.where(found, b, hit), o, dl


def hash3(i, seed, channel):
    """Integer lattice points [..., 3] to 0..1 (32-bit integer hash in int64 maths)."""
    m = 0xFFFFFFFF
    h = (i[..., 0] * 73856093) ^ (i[..., 1] * 19349663) ^ (i[..., 2] * 83492791) ^ (seed * 2654435761) ^ (channel * 40503)
    h = h & m
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & m
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & m
    h = h ^ (h >> 16)
    return (h & 0xFFFF).float() / 65535.0


def noise3(p, seed, channel):
    """Value noise in -1..1 at points p [..., 3] (lattice spacing 1), smoothly interpolated."""
    i = torch.floor(p)
    f = p - i
    f = f * f * (3 - 2 * f)
    i = i.long()
    out = 0
    for c in range(8):
        off = torch.tensor([c & 1, (c >> 1) & 1, c >> 2], device=p.device)
        w = torch.where(off.bool(), f, 1 - f).prod(-1)
        out = out + w * hash3(i + off, seed, channel)
    return out * 2 - 1


def maps(q, d, cam):
    """Per-pixel maps [B, MAP_CHANNELS, h, w] and ids [B, 2, h, w] (shape, motif; -1 where no
    part is) for parts `q` [B, P, PART_PARAMS], rays `d` [B, h, w, 3] and per-view `cam` [B, CAM]."""
    t, o, dl = march(q, d)
    best, idx = t.min(1)  # the nearest part per pixel
    hit = torch.isfinite(best)
    B, P, h, w = t.shape
    g = idx[:, None, :, :, None]
    pick = lambda x: torch.gather(x.expand(B, P, h, w, x.shape[-1]), 1, g.expand(B, 1, h, w, x.shape[-1])).squeeze(1)  # noqa: E731
    qp = pick(q[:, :, None, None])  # [B, h, w, PART_PARAMS]
    p = pick(o) + pick(dl) * torch.where(hit, best, torch.zeros_like(best))[..., None]
    # The normal from the field's gradient, in the part's frame, then in camera axes.
    e = 0.02 * qp[..., P_HALF].amin(-1, keepdim=True).clamp(min=1e-3)
    eye = torch.eye(3, device=q.device)
    grad = torch.stack([outside(p + e * eye[k], qp) - outside(p - e * eye[k], qp) for k in range(3)], -1)
    axes = qp[..., P_AXES].reshape(B, h, w, 3, 3)
    n = F.normalize(torch.einsum("bhwij,bhwj->bhwi", axes, F.normalize(grad, dim=-1)), dim=-1)
    point = torch.einsum("bhwij,bhwj->bhwi", axes, p) + qp[..., P_CENTRE]
    dist = point.norm(dim=-1).clamp(min=1e-3)
    # The part's baked light at the normal, and the torch's (at the eye), as the flesh pass has them.
    sh = qp[..., P_SH]
    irr = sh[..., 0:3] + sum(sh[..., 3 + 3 * k : 6 + 3 * k] * n[..., k : k + 1] for k in range(3))
    nv = -(n * point).sum(-1) / dist
    c = cam[:, None, None]
    fall = (1 - dist / c[..., 3].clamp(min=1e-3)).clamp(min=0) ** 2
    torch_light = c[..., :3] * (fall * ((nv + 0.3) / 1.3).clamp(min=0))[..., None]
    fog = ((dist - c[..., 4]) / (c[..., 5] - c[..., 4]).clamp(min=1e-3)).clamp(0, 1)
    # Noise fixed to the part, in skin repeats.
    at = p / qp[..., P_REPEAT : P_REPEAT + 1].clamp(min=1e-3)
    seed = qp[..., P_SEED].long()
    noise = torch.stack(
        [sum(noise3(at * 2**k, seed, ch * OCTAVES + k) / 2**k for k in range(OCTAVES)) for ch in range(NOISE)], -1
    )
    feats = torch.cat(
        [
            torch.ones_like(dist)[..., None],
            (point[..., 2] / 8)[..., None],
            n,
            p / qp[..., P_HALF].clamp(min=1e-3),
            irr,
            torch_light,
            fog[..., None],
            c[..., 6:9].expand(B, h, w, 3) * fog[..., None],
            qp[..., P_DENSITY : P_DENSITY + 1],
            qp[..., P_TINT],
            noise,
        ],
        -1,
    )
    feats = torch.where(hit[..., None], feats, torch.zeros_like(feats)).permute(0, 3, 1, 2)
    ids = torch.stack([qp[..., P_SHAPE], qp[..., P_MOTIF]], 1).long()
    ids = torch.where(hit[:, None], ids, torch.full_like(ids, -1))
    return feats.contiguous(), ids


class Crops(Dataset):
    """Square crops of `size` round the creatures of datagen views that have them: per item
    parts [MAX_PARTS, PART_PARAMS] (zero padding), rays [size, size, 3], cam [CAM], the
    creatures' layer [4, size, size] (rgb, alpha) and the level's frame [3, size, size].
    `fixed` gives every item the same crop and seeds each time (for validation)."""

    def __init__(self, root, shards=None, size=128, fixed=False):
        self.shards = Shards(root, shards)
        self.size, self.fixed = size, fixed
        self.items = []
        for i, (si, vi) in enumerate(self.shards.index):
            s = self.shards.shards[si]
            if "boxes" in s:
                x0, y0, x1, y1 = s["boxes"][vi]
                if (x1 - x0) * (y1 - y0) >= 16:
                    self.items.append(i)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i = self.items[k]
        si, vi = self.shards.index[i]
        s = self.shards.shards[si]
        v = s["views"][vi]
        width = view_width(v)
        rng = np.random.default_rng(k if self.fixed else None)
        layer = self.shards.creatures(i)
        a, b = s["image_offsets"][vi : vi + 2]
        frame = np.array(s["images"][a:b]).reshape(H, width, 3)
        # Centred on a creature pixel, within the frame.
        ys, xs = np.nonzero(layer[..., 3])
        j = rng.integers(len(ys))
        n = self.size
        x0 = int(np.clip(xs[j] - n // 2 + rng.integers(-n // 4, n // 4 + 1), 0, width - n))
        y0 = int(np.clip(ys[j] - n // 2 + rng.integers(-n // 4, n // 4 + 1), 0, H - n))
        scene = int(v[0])
        parts = s["parts"][s["part_offsets"][scene] : s["part_offsets"][scene + 1]]
        q = view_parts(parts, v, rng.integers(0, 2**31, len(parts)).astype(np.float32))
        q = np.pad(q, ((0, MAX_PARTS - len(q)), (0, 0)))
        crop = lambda x: torch.from_numpy(x[y0 : y0 + n, x0 : x0 + n].copy()).permute(2, 0, 1).float() / 255.0  # noqa: E731
        return (
            torch.from_numpy(q),
            rays(x0, y0, n, n, float(v[13]), width),
            torch.from_numpy(cam_values(v)),
            crop(layer),
            crop(frame),
        )


def lin(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def enc(c):
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def composite(frame, layer):
    """The creatures' layer [b, 4, h, w] over the level's frame [b, 3, h, w], as the game lays
    it: by alpha, in linear light, in encoded colour out."""
    a = layer[:, 3:]
    return enc(lin(frame) * (1 - a) + lin(layer[:, :3]) * a)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return x + self.b(F.gelu(self.a(x)))


class CreatureNet(nn.Module):
    """Maps to the creatures' layer: a small U-Net (full, half and quarter size) of plain
    3x3 convolutions and GELUs, for a straightforward WGSL port. Output [b, 4, h, w]:
    rgb and alpha in 0..1; h and w multiples of 4."""

    def __init__(self, c=(48, 96, 128), blocks=2):
        super().__init__()
        self.shape = nn.Embedding(NUM_SHAPES + 1, 4)
        self.motif = nn.Embedding(NUM_MOTIFS + 1, 8)
        self.inp = nn.Conv2d(MAP_CHANNELS + 12, c[0], 3, padding=1)
        self.enc = nn.ModuleList([nn.Sequential(*[Block(ch) for _ in range(blocks)]) for ch in c])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(c, c[1:])])
        self.up = nn.ModuleList([nn.Conv2d(b + a, a, 3, padding=1) for a, b in zip(c, c[1:])])
        self.dec = nn.ModuleList([nn.Sequential(*[Block(ch) for _ in range(blocks)]) for ch in c[:-1]])
        self.out = nn.Conv2d(c[0], 4, 3, padding=1)
        with torch.no_grad():  # start from an empty layer (alpha ~0.05), not a grey haze
            self.out.bias[3] = -3.0

    def forward(self, feats, ids):
        emb = torch.cat([self.shape(ids[:, 0] + 1), self.motif(ids[:, 1] + 1)], -1).permute(0, 3, 1, 2)
        x = self.inp(torch.cat([feats, emb], 1))
        skips = []
        for i, enc in enumerate(self.enc):
            if i:
                x = F.gelu(self.down[i - 1](x))
            x = enc(x)
            skips.append(x)
        for i in reversed(range(len(self.dec))):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.dec[i](F.gelu(self.up[i](torch.cat([x, skips[i]], 1))))
        return torch.sigmoid(self.out(x))
