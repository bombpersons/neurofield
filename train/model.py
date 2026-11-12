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

Eyes, sphincters and lengths of vein come in the same way, one token each after the
triangles (see data.organ_tokens), with a coverage bias of their own.

The upsampling head lays the patch queries out as a grid and doubles it four
times (3x3 convolution, then pixel shuffle), so neighbouring patches share
detail across their edges. With `up="resize"` each doubling is bilinear and then
a convolution instead, which keeps pixel shuffle's 16-pixel mesh out of the frame.
Each doubling can add per-pixel noise at a learned
strength per channel (zero to begin with; `noise` gives the noise), for
inventing detail rather than averaging it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import (
    CAM_FEATURES, H, NUM_MOTIFS, NUM_ORGANS, NUM_SHAPES, NUM_TEX, ORGAN_FEATURES, PART_FEATURES, TRI_FEATURES,
    TRI_FEATURES_PLAIN, W,
)

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
    size, plus noise scaled per channel, GELU), then a 3x3 conv to RGB. With `up` "resize",
    each doubling is bilinear and then a 3x3 conv to c channels."""

    def __init__(self, d, first=128, up="shuffle"):
        super().__init__()
        self.proj = nn.Linear(d, first)
        chans = (first,) + UP
        self.resize = up == "resize"
        self.ups = nn.ModuleList([nn.Conv2d(a, b if self.resize else 4 * b, 3, padding=1) for a, b in zip(chans, chans[1:])])
        self.noise_scale = nn.ParameterList([nn.Parameter(torch.zeros(c)) for c in UP])
        self.rgb = nn.Conv2d(UP[-1], 3, 3, padding=1)

    def forward(self, q, gw, noise=None, features=False):
        """`noise` [b, 4, 16 gh, 16 gw]: channel i, at the size of doubling i (the
        top-left corner of it), is that doubling's noise; None for none. With `features`,
        also the last full-size channels [b, UP[-1], 16 gh, 16 gw] (for Deferred)."""
        b, gh = q.shape[0], q.shape[1] // gw
        x = self.proj(q).transpose(1, 2).reshape(b, -1, gh, gw)
        for i, (up, scale) in enumerate(zip(self.ups, self.noise_scale)):
            x = up(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)) if self.resize else F.pixel_shuffle(up(x), 2)
            if noise is not None:
                x = x + scale[:, None, None] * noise[:, i : i + 1, : x.shape[2], : x.shape[3]]
            x = F.gelu(x)
        return (self.rgb(x), x) if features else self.rgb(x)


class Deferred(nn.Module):
    """The per-pixel stage (docs/deferred_stage.md): at every pixel, the head's last
    channels, what the G-buffer says of the pixel (pixels.inputs) and learned latent
    textures sampled where the pixel lies on them (one per level texture, one per painted
    flesh slot, and an eye by blink with an iris that follows the gaze), through a small
    MLP, added to the head's colour. Its last layer starts at zero, so a model that gains
    one draws what it did."""

    LEVEL, LEVEL_SIZE = 8, 128  # channels, texels a side: the 12 level textures, 4 x 3
    FLESH, FLESH_SIZE = 8, 256  # channels, texels a side: one for each painted flesh slot
    EYE, EYE_SIZE, BLINKS = 8, 64, 8
    IRIS = 4

    def __init__(self, hidden=64):
        super().__init__()
        from pixels import CHANNELS

        self.level = nn.Parameter(torch.randn(self.LEVEL, 3 * self.LEVEL_SIZE, 4 * self.LEVEL_SIZE) * 0.1)
        from pixels import FLESH_SLOTS

        self.flesh = nn.Parameter(torch.randn(FLESH_SLOTS, self.FLESH, self.FLESH_SIZE, self.FLESH_SIZE) * 0.1)
        self.eye = nn.Parameter(torch.randn(self.EYE, self.BLINKS, self.EYE_SIZE, self.EYE_SIZE) * 0.1)
        self.iris = nn.Parameter(torch.randn(self.IRIS, self.EYE_SIZE, self.EYE_SIZE) * 0.1)
        inp = UP[-1] + CHANNELS + self.LEVEL + self.FLESH + self.EYE + self.IRIS
        self.mlp = nn.Sequential(nn.Conv2d(inp, hidden, 1), nn.GELU(), nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 3, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _atlas(tex, cols, rows, size, which, uv):
        """Sample tile `which` [b, h, w] of an atlas [c, rows * size, cols * size] at uv
        [b, h, w, 2] (0..1, kept half a texel inside the tile): [b, c, h, w]."""
        e = 0.5 / size
        uv = uv * (1 - 2 * e) + e
        x = ((which % cols).float() + uv[..., 0]) / cols * 2 - 1
        y = ((which // cols).float() + uv[..., 1]) / rows * 2 - 1
        grid = torch.stack([x, y], -1)
        return F.grid_sample(tex[None].expand(len(uv), -1, -1, -1), grid, align_corners=False)

    @staticmethod
    def _wrapped(tex, which, uv):
        """Sample texture `which` [b, h, w] of `tex` [n, c, size, size] at uv [b, h, w, 2]
        (in repeats), bilinearly and wrapping round, as the flesh repeats: [b, c, h, w]."""
        n, c, size, _ = tex.shape
        flat = tex.permute(0, 2, 3, 1).reshape(-1, c)
        x, y = uv[..., 0] * size - 0.5, uv[..., 1] * size - 0.5
        x0, y0 = x.floor(), y.floor()
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]
        x0, y0 = x0.long(), y0.long()
        base = which.long() * size * size
        at = lambda i, j: flat[base + (j % size) * size + i % size]  # noqa: E731
        top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
        bottom = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
        return (top * (1 - fy) + bottom * fy).permute(0, 3, 1, 2)

    def forward(self, head, plain, at):
        """`head` [b, UP[-1], h, w]; `plain`, `at` as pixels.inputs gives them."""
        level = self._atlas(self.level, 4, 3, self.LEVEL_SIZE, at["tex"], at["uv"])
        flesh = self._wrapped(self.flesh, at["slot"], at["lay"]).to(head.dtype) * at["flesh"][:, None]
        b = len(head)
        u, v = at["eye_uv"][..., 0], at["eye_uv"][..., 1]
        grid = torch.stack([u, v, at["blink"].clamp(0, 1) * 2 - 1], -1)[:, None]  # [b, 1, h, w, 3]
        eye = F.grid_sample(self.eye[None].expand(b, -1, -1, -1, -1), grid, align_corners=False)[:, :, 0]
        g = at["gaze"] - 0.5
        iris = F.grid_sample(self.iris[None].expand(b, -1, -1, -1), torch.stack([u - 0.8 * g[..., 0], v - 0.8 * g[..., 1]], -1), align_corners=False)
        on = at["eye"][:, None].float()
        x = torch.cat([head, plain, level * plain[:, :1], flesh, eye * on, iris * on], 1)
        return self.mlp(x)


class Renderer(nn.Module):
    def __init__(self, d=256, heads=8, enc_layers=4, dec_layers=6, bands=8, low=2, up="shuffle", deferred=False):
        super().__init__()
        self.bands, self.low = bands, low
        tri_in = TRI_FEATURES + 6 * 2 * bands  # + Fourier of the 3 vertices' NDC
        self.tri_in = nn.Sequential(nn.Linear(tri_in, d), nn.GELU(), nn.Linear(d, d))
        self.tex = nn.Embedding(NUM_TEX, d)
        part_in = PART_FEATURES + 4 * 2 * bands  # + Fourier of the screen box's x0, y0, x1, y1
        self.part_in = nn.Sequential(nn.Linear(part_in, d), nn.GELU(), nn.Linear(d, d))
        self.shape = nn.Embedding(NUM_SHAPES, d)
        self.motif = nn.Embedding(NUM_MOTIFS, d)
        organ_in = ORGAN_FEATURES + 4 * 2 * bands  # + Fourier of the screen box's x0, y0, x1, y1
        self.organ_in = nn.Sequential(nn.Linear(organ_in, d), nn.GELU(), nn.Linear(d, d))
        self.organ_kind = nn.Embedding(NUM_ORGANS, d)
        self.cam_in = nn.Sequential(nn.Linear(CAM_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.encoder = nn.ModuleList([Block(d, heads, False) for _ in range(enc_layers)])
        self.q_in = nn.Sequential(nn.Linear(2 * 2 * bands, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = nn.ModuleList([Block(d, heads, True) for _ in range(dec_layers)])
        # How strongly each decoder layer's heads prefer covering triangles; zero = no preference.
        self.cover = nn.Parameter(torch.zeros(dec_layers, heads))
        # The same for creature parts and their screen boxes.
        self.part_cover = nn.Parameter(torch.zeros(dec_layers, heads))
        self.organ_cover = nn.Parameter(torch.zeros(dec_layers, heads))
        # Fixed directions for the dream effects (engine::neural::effects) to push patch queries along.
        noise_dirs = torch.randn(DREAM_DIRS, d, generator=torch.Generator().manual_seed(0)) / math.sqrt(DREAM_DIRS)
        self.register_buffer("noise_dirs", noise_dirs, persistent=False)
        self.out_norm = nn.LayerNorm(d)
        self.head = UpHead(d, up=up)
        self.deferred = Deferred() if deferred else None

    def forward(self, feats, tex, pad, cams, width=W, parts=None, noise=None, tex_mix=None, temp=None, q_noise=None, organs=None, pixels=None):
        """An image [b, 3, 240, width] (width a multiple of 16) of the triangle tokens seen
        with camera tokens `cams`. `parts` is (part feats, part ids, part pad) as `collate`
        gives them, None for no creatures; `organs` likewise (feats, kinds, pad). `noise` is the head's (see UpHead). The other optional
        inputs are for disturbing a trained model (the game's dream effects): `tex_mix`
        [NUM_TEX, NUM_TEX] mixes texture embeddings (identity: none), `temp` [1] is the
        cross-attention temperature (1: none), and `q_noise` [patches, DREAM_DIRS] pushes
        each patch query along fixed random directions (zeros: none). `pixels` is what
        pixels.inputs gives the per-pixel stage, if the model has one."""
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
        if organs is not None:
            ofeats, okinds, opad = organs
            o = self.organ_in(torch.cat([ofeats, fourier(ofeats[:, :, 12:16], self.bands, self.low)], -1))
            x = torch.cat([x, o + self.organ_kind(okinds)], 1)
            pad = torch.cat([pad, opad], 1)
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
        odist = None if organs is None else box_distance(organs[0][:, :, 12:14], organs[0][:, :, 14:16], c)
        for blk, strength, pstrength, ostrength in zip(self.decoder, self.cover, self.part_cover, self.organ_cover):
            bias = -strength[None, :, None, None] * dist[:, None]  # [b, heads, patches, 1 + tris]
            if pdist is not None:
                bias = torch.cat([bias, -pstrength[None, :, None, None] * pdist[:, None]], -1)
            if odist is not None:
                bias = torch.cat([bias, -ostrength[None, :, None, None] * odist[:, None]], -1)
            bias = bias.masked_fill(pad[:, None, None], float("-inf"))
            q = blk(q, mem=x, mem_bias=bias, temp=temp)
        if self.deferred is None or pixels is None:
            return self.head(self.out_norm(q), gw, noise)
        rgb, last = self.head(self.out_norm(q), gw, noise, features=True)
        return rgb + self.deferred(last, *pixels)


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
    `migrate_parts`, `migrate_turning` and `migrate_organs`. Returns (weights, cfg, whether
    anything changed)."""
    weights, cfg, widths = migrate_widths(weights, cfg)
    weights, parts = migrate_parts(weights, cfg)
    weights, turning = migrate_turning(weights, cfg)
    weights, organs = migrate_organs(weights, cfg)
    return weights, cfg, widths or parts or turning or organs


def migrate_turning(weights, cfg):
    """From before triangles were told how they turn: those inputs get zero weights, so
    the body computes what it did."""
    w = weights["tri_in.0.weight"]
    plain = TRI_FEATURES_PLAIN + 6 * 2 * cfg.get("bands", 8)
    if w.shape[1] != plain:
        return weights, False
    zeros = torch.zeros(w.shape[0], TRI_FEATURES - TRI_FEATURES_PLAIN, dtype=w.dtype)
    return {**weights, "tri_in.0.weight": torch.cat([w[:, :TRI_FEATURES_PLAIN], zeros, w[:, TRI_FEATURES_PLAIN:]], 1)}, True


def migrate_organs(weights, cfg):
    """From before organs: fresh organ tokens, with the triangles' coverage strengths to
    start from (both measure distance to a screen box)."""
    if "organ_in.0.weight" in weights:
        return weights, False
    new = Renderer(**cfg).state_dict()
    out = dict(weights)
    for k, v in new.items():
        if k.startswith(("organ_in.", "organ_kind.")):
            out[k] = v
    out["organ_cover"] = weights.get("cover", new["cover"]).clone()
    return out, True


def migrate_deferred(weights, cfg):
    """Give a model without a per-pixel stage a fresh one (which draws nothing yet).
    Returns (weights, cfg, whether anything changed)."""
    if any(k.startswith("deferred.") for k in weights):
        return weights, {**cfg, "deferred": True}, False
    cfg = {**cfg, "deferred": True}
    out = dict(weights)
    out.update({k: v for k, v in Renderer(**cfg).state_dict().items() if k.startswith("deferred.")})
    return out, cfg, True


def migrate_head(weights, cfg, up):
    """Swap the upsampling head for a fresh one of kind `up`, if it isn't that already.
    Returns (weights, cfg, whether anything changed)."""
    if cfg.get("up", "shuffle") == up:
        return weights, cfg, False
    cfg = {**cfg, "up": up}
    out = {k: v for k, v in weights.items() if not k.startswith("head.")}
    out.update({k: v for k, v in Renderer(**cfg).state_dict().items() if k.startswith("head.")})
    return out, cfg, True


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

    out["tri_in.0.weight"] = widen(weights["tri_in.0.weight"], TRI_FEATURES_PLAIN - 1, TRI_FEATURES_PLAIN, 6)
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


class PatchDisc(nn.Module):
    """A patch discriminator for training only: an image [b, 3, H, W] to a realness score
    for each 8x8 cell [b, 1, H/8, W/8], each seeing about 46 pixels across. Spectral norm
    keeps it from overpowering the renderer."""

    def __init__(self, c=64):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        chans = (3, c, 2 * c, 4 * c)
        self.downs = nn.ModuleList([sn(nn.Conv2d(a, b, 4, 2, 1)) for a, b in zip(chans, chans[1:])])
        self.out = sn(nn.Conv2d(chans[-1], 1, 3, 1, 1))

    def forward(self, x):
        x = x * 2 - 1
        for down in self.downs:
            x = F.leaky_relu(down(x), 0.2)
        return self.out(x)
