"""A small RenderFormer-style renderer.

Triangle tokens attend to each other (so occluders can find what they hide),
then a grid of patch queries, one per 16x16 pixels, cross-attends to them.
Screen positions of both triangles and patches go through the same Fourier
features, so a patch can match the triangles that cover it; the
cross-attention is also biased toward triangles whose screen bounding box
covers or lies near the patch, by a learned strength per decoder layer and head.
Creatures come in as one token per body part (see data.part_tokens), with the
Fourier features of their screen box's corners, alongside the triangles and
with a coverage bias of their own.
Patch queries come from their positions alone, so the grid can be any width
(any aspect ratio; 20 patches across is 4:3).

The upsampling head lays the patch queries out as a grid and doubles it four
times (3x3 convolution, then pixel shuffle), so neighbouring patches share
detail across their edges. Each doubling can add per-pixel noise at a learned
strength per channel (zero to begin with; `noise` gives the noise), for
inventing detail rather than averaging it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CAM_FEATURES, H, NUM_MOTIFS, NUM_SHAPES, NUM_TEX, PART_FEATURES, TRI_FEATURES, W

PATCH = 16
DREAM_DIRS = 16
# Channels of the upsampling head after each doubling: 1/8, 1/4, 1/2 and full resolution.
UP = (64, 48, 32, 16)


def fourier(x, bands, low=0):
    """x [..., c] -> [..., c * 2 * bands] sin/cos features at octave frequencies
    2^-low pi .. 2^(bands - 1 - low) pi: the lowest repeats every 2^(low + 1)."""
    freqs = (2.0 ** (torch.arange(bands, device=x.device) - low)) * math.pi
    a = x[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def biased_attention(attn, x, mem, bias, temp=1.0):
    """`attn` (an nn.MultiheadAttention, for its weights) from x to mem, with `bias`
    [b, heads, queries, keys] added to the scores, softmax at temperature `temp`. Done by
    hand, step by step as the WGSL network (engine/src/neural/gpu_net.wgsl) does it."""
    heads = attn.num_heads
    wq, wk, wv = attn.in_proj_weight.chunk(3)
    bq, bk, bv = attn.in_proj_bias.chunk(3)
    q, k, v = (
        F.linear(t, w, b).unflatten(-1, (heads, -1)).transpose(1, 2)  # [b, heads, tokens, d / heads]
        for t, w, b in [(x, wq, bq), (mem, wk, bk), (mem, wv, bv)]
    )
    scores = q @ k.transpose(-1, -2) / (math.sqrt(q.shape[-1]) * temp) + bias.to(q.dtype)
    out = scores.softmax(-1) @ v
    return attn.out_proj(out.transpose(1, 2).reshape(x.shape))


class Block(nn.Module):
    def __init__(self, d, heads, cross):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.n2 = nn.LayerNorm(d)
            self.xattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pad=None, mem=None, mem_bias=None, temp=1.0):
        """`mem_bias` [b, heads, queries, mem tokens] is added to the cross-attention scores
        (-inf masks padding)."""
        h = self.n1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        if self.cross:
            h = self.n2(x)
            x = x + biased_attention(self.xattn, h, mem, mem_bias, temp)
        return x + self.mlp(self.n3(x))


def centres(gw, device=None):
    """Patch centres [gh * gw, 2] of a grid gw patches wide (and 15 high), in the
    triangles' NDC: a pixel's x is (px - width / 2) / 160, its y (py - 120) / 160."""
    gh = H // PATCH
    ys, xs = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
    c = torch.stack([(xs + 0.5) * PATCH - gw * PATCH / 2, (ys + 0.5) * PATCH - H / 2], -1) / (W / 2)
    return c.reshape(-1, 2).float()


class UpHead(nn.Module):
    """Patch queries [b, gh * gw, d] to an image [b, 3, 16 gh, 16 gw]: a 1x1 projection
    to the grid, four times (3x3 conv to 4c channels, pixel shuffle to c at twice the
    size, plus noise scaled per channel, GELU), then a 3x3 conv to RGB."""

    def __init__(self, d, first=128):
        super().__init__()
        self.proj = nn.Linear(d, first)
        chans = (first,) + UP
        self.ups = nn.ModuleList([nn.Conv2d(a, 4 * b, 3, padding=1) for a, b in zip(chans, chans[1:])])
        self.noise_scale = nn.ParameterList([nn.Parameter(torch.zeros(c)) for c in UP])
        self.rgb = nn.Conv2d(UP[-1], 3, 3, padding=1)

    def forward(self, q, gw, noise=None):
        """`noise` [b, 4, 16 gh, 16 gw]: channel i, at the size of doubling i (the
        top-left corner of it), is that doubling's noise; None for none."""
        b, gh = q.shape[0], q.shape[1] // gw
        x = self.proj(q).transpose(1, 2).reshape(b, -1, gh, gw)
        for i, (up, scale) in enumerate(zip(self.ups, self.noise_scale)):
            x = F.pixel_shuffle(up(x), 2)
            if noise is not None:
                x = x + scale[:, None, None] * noise[:, i : i + 1, : x.shape[2], : x.shape[3]]
            x = F.gelu(x)
        return self.rgb(x)


class Renderer(nn.Module):
    def __init__(self, d=256, heads=8, enc_layers=4, dec_layers=6, bands=8, low=2):
        super().__init__()
        self.bands, self.low = bands, low
        tri_in = TRI_FEATURES + 6 * 2 * bands  # + Fourier of the 3 vertices' NDC
        self.tri_in = nn.Sequential(nn.Linear(tri_in, d), nn.GELU(), nn.Linear(d, d))
        self.tex = nn.Embedding(NUM_TEX, d)
        part_in = PART_FEATURES + 4 * 2 * bands  # + Fourier of the screen box's x0, y0, x1, y1
        self.part_in = nn.Sequential(nn.Linear(part_in, d), nn.GELU(), nn.Linear(d, d))
        self.shape = nn.Embedding(NUM_SHAPES, d)
        self.motif = nn.Embedding(NUM_MOTIFS, d)
        self.cam_in = nn.Sequential(nn.Linear(CAM_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.encoder = nn.ModuleList([Block(d, heads, False) for _ in range(enc_layers)])
        self.q_in = nn.Sequential(nn.Linear(2 * 2 * bands, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = nn.ModuleList([Block(d, heads, True) for _ in range(dec_layers)])
        # How strongly each decoder layer's heads prefer covering triangles; zero = no preference.
        self.cover = nn.Parameter(torch.zeros(dec_layers, heads))
        # The same for creature parts and their screen boxes.
        self.part_cover = nn.Parameter(torch.zeros(dec_layers, heads))
        # Fixed directions for the dream effects (engine::neural::effects) to push patch queries along.
        noise_dirs = torch.randn(DREAM_DIRS, d, generator=torch.Generator().manual_seed(0)) / math.sqrt(DREAM_DIRS)
        self.register_buffer("noise_dirs", noise_dirs, persistent=False)
        self.out_norm = nn.LayerNorm(d)
        self.head = UpHead(d)

    def forward(self, feats, tex, pad, cams, width=W, parts=None, noise=None, tex_mix=None, temp=None, q_noise=None):
        """An image [b, 3, 240, width] (width a multiple of 16) of the triangle tokens seen
        with camera tokens `cams`. `parts` is (part feats, part ids, part pad) as `collate`
        gives them, None for no creatures. `noise` is the head's (see UpHead). The other optional
        inputs are for disturbing a trained model (the game's dream effects): `tex_mix`
        [NUM_TEX, NUM_TEX] mixes texture embeddings (identity: none), `temp` [1] is the
        cross-attention temperature (1: none), and `q_noise` [patches, DREAM_DIRS] pushes
        each patch query along fixed random directions (zeros: none)."""
        b = feats.shape[0]
        gw = width // PATCH
        ndc = feats[:, :, 9:15]
        table = self.tex.weight if tex_mix is None else tex_mix @ self.tex.weight
        x = self.tri_in(torch.cat([feats, fourier(ndc, self.bands, self.low)], -1)) + F.embedding(tex, table)
        cam = self.cam_in(cams)[:, None]
        # The camera token rides along as an extra (never padded) triangle token.
        x = torch.cat([cam, x], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=pad.device), pad], 1)
        if parts is not None:
            pfeats, pids, ppad = parts
            box = pfeats[:, :, 18:22]
            p = self.part_in(torch.cat([pfeats, fourier(box, self.bands, self.low)], -1))
            # Creature parts follow the triangles.
            x = torch.cat([x, p + self.shape(pids[..., 0]) + self.motif(pids[..., 1])], 1)
            pad = torch.cat([pad, ppad], 1)
        for blk in self.encoder:
            x = blk(x, pad=pad)

        c = centres(gw, feats.device)
        q = self.q_in(fourier(c, self.bands, self.low))
        q = q[None].expand(b, -1, -1) + cam
        if q_noise is not None:
            q = q + q_noise @ self.noise_dirs
        temp = 1.0 if temp is None else temp
        dist = cover_distance(ndc, c)
        pdist = None if parts is None else box_distance(parts[0][:, :, 18:20], parts[0][:, :, 20:22], c)
        for blk, strength, pstrength in zip(self.decoder, self.cover, self.part_cover):
            bias = -strength[None, :, None, None] * dist[:, None]  # [b, heads, patches, 1 + tris]
            if pdist is not None:
                bias = torch.cat([bias, -pstrength[None, :, None, None] * pdist[:, None]], -1)
            bias = bias.masked_fill(pad[:, None, None], float("-inf"))
            q = blk(q, mem=x, mem_bias=bias, temp=temp)
        return self.head(self.out_norm(q), gw, noise)


def cover_distance(ndc, c):
    """Distance in patch widths from each patch centre c [patches, 2] to each triangle's
    screen bounding box (0 inside it): [b, patches, 1 + tris], with 0 for the camera
    token in front."""
    xy = ndc.unflatten(-1, (3, 2))  # [b, tris, vertex, xy]
    return F.pad(box_distance(xy.min(2).values, xy.max(2).values, c), (1, 0))


def box_distance(lo, hi, c):
    """Distance in patch widths from each patch centre c [patches, 2] to each of the
    screen boxes lo..hi [b, n, 2] (0 inside it): [b, patches, n]."""
    c = c[None, :, None]  # [1, patches, 1, 2]
    gap = (lo[:, None] - c).clamp(min=0) + (c - hi[:, None]).clamp(min=0)
    return gap.norm(dim=-1) / (2 * PATCH / W)


def migrate(weights, cfg):
    """Weights (and config) of an older model, brought up to date: see `migrate_widths`
    and `migrate_parts`. Returns (weights, cfg, whether anything changed)."""
    weights, cfg, widths = migrate_widths(weights, cfg)
    weights, parts = migrate_parts(weights, cfg)
    return weights, cfg, widths or parts


def migrate_parts(weights, cfg):
    """From before creature parts: fresh part tokens, with the triangles' coverage
    strengths to start from (both measure distance to a screen box)."""
    if "part_in.0.weight" in weights:
        return weights, False
    new = Renderer(**cfg).state_dict()
    out = dict(weights)
    for k, v in new.items():
        if k.startswith(("part_in.", "shape.", "motif.")):
            out[k] = v
    out["part_cover"] = weights.get("cover", new["cover"]).clone()
    return out, True


def migrate_widths(weights, cfg):
    """Weights (and config) of a model from before widths varied: 45 triangle and 7
    camera features, 6 Fourier bands, a learned table of patch positions and a linear
    patch head. The inputs added since (strangeness; fog colour and screen reach; two
    lower Fourier bands) get zero weights, so the body computes what it did; the patch
    table and old head are dropped, so the new head starts fresh. Returns (weights, cfg,
    whether anything changed)."""
    if "q_pos" not in weights:
        return weights, cfg, False
    old = cfg.get("bands", 6)
    new = Renderer(**{**cfg, "bands": 8, "low": 2})
    bands, low = new.bands, new.low
    out = {k: v for k, v in weights.items() if not k.startswith(("head.", "refine.")) and k not in ("q_pos", "centres")}

    def widen(w, feats_old, feats_new, coords):
        """Columns [features | per coordinate: sin bands, cos bands] to the new layout."""
        cols = torch.zeros(w.shape[0], feats_new + coords * 2 * bands, dtype=w.dtype)
        cols[:, :feats_old] = w[:, :feats_old]
        for c in range(coords):
            for part in range(2):  # sin, cos
                src = feats_old + c * 2 * old + part * old
                dst = feats_new + c * 2 * bands + part * bands + low
                cols[:, dst : dst + old] = w[:, src : src + old]
        return cols

    out["tri_in.0.weight"] = widen(weights["tri_in.0.weight"], TRI_FEATURES - 1, TRI_FEATURES, 6)
    out["q_in.0.weight"] = widen(weights["q_in.0.weight"], 0, 0, 2)
    cam = torch.zeros(weights["cam_in.0.weight"].shape[0], CAM_FEATURES)
    cam[:, : weights["cam_in.0.weight"].shape[1]] = weights["cam_in.0.weight"]
    out["cam_in.0.weight"] = cam
    for k, v in new.state_dict().items():
        if k.startswith("head."):
            out[k] = v
    return out, {**cfg, "bands": bands, "low": low}, True


def load(path, device="cpu"):
    """(model, checkpoint) from a ckpt.pt: the averaged weights when it has them. Checkpoints
    from before the coverage bias load with it off (zero strength); from before widths
    varied, see `migrate`."""
    state = torch.load(path, map_location=device, weights_only=False)
    weights, cfg, _ = migrate(state.get("ema", state["model"]), state.get("cfg", {}))
    m = Renderer(**cfg).to(device)
    missing = m.load_state_dict(weights, strict=False).missing_keys
    assert set(missing) <= {"cover"}, missing
    return m.eval(), state


def loss_fn(pred, target):
    return F.l1_loss(pred, target)
