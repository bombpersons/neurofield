"""A small RenderFormer-style renderer.

Triangle tokens attend to each other (so occluders can find what they hide),
then a grid of 16x16-pixel patch queries cross-attends to them and decodes
pixels. Screen positions of both triangles and patches go through the same
Fourier features, so a patch can match the triangles that cover it.
A light convolutional pass at full resolution cleans up patch seams.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CAM_FEATURES, H, NUM_TEX, TRI_FEATURES, W

PATCH = 16


def fourier(x, bands):
    """x [..., c] -> [..., c * 2 * bands] sin/cos features at octave frequencies."""
    freqs = (2.0 ** torch.arange(bands, device=x.device)) * math.pi
    a = x[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


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

    def forward(self, x, pad=None, mem=None, mem_pad=None):
        h = self.n1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        if self.cross:
            h = self.n2(x)
            x = x + self.xattn(h, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class Renderer(nn.Module):
    def __init__(self, d=256, heads=8, enc_layers=4, dec_layers=6, bands=6):
        super().__init__()
        self.bands = bands
        tri_in = TRI_FEATURES + 6 * 2 * bands  # + Fourier of the 3 vertices' NDC
        self.tri_in = nn.Sequential(nn.Linear(tri_in, d), nn.GELU(), nn.Linear(d, d))
        self.tex = nn.Embedding(NUM_TEX, d)
        self.cam_in = nn.Sequential(nn.Linear(CAM_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.encoder = nn.ModuleList([Block(d, heads, False) for _ in range(enc_layers)])

        gw, gh = W // PATCH, H // PATCH
        ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        # Patch centres in the same NDC as triangles (x spans -1..1, y spans -0.75..0.75).
        centres = torch.stack([(xs + 0.5) / gw * 2 - 1, ((ys + 0.5) / gh * 2 - 1) * 0.75], -1).reshape(-1, 2)
        self.register_buffer("centres", centres)
        self.q_in = nn.Sequential(nn.Linear(2 * 2 * bands, d), nn.GELU(), nn.Linear(d, d))
        self.q_pos = nn.Parameter(torch.zeros(gw * gh, d))
        self.decoder = nn.ModuleList([Block(d, heads, True) for _ in range(dec_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, PATCH * PATCH * 3)
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, feats, tex, pad, cams):
        b = feats.shape[0]
        ndc = feats[:, :, 9:15]
        x = self.tri_in(torch.cat([feats, fourier(ndc, self.bands)], -1)) + self.tex(tex)
        cam = self.cam_in(cams)[:, None]
        # The camera token rides along as an extra (never padded) triangle token.
        x = torch.cat([cam, x], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for blk in self.encoder:
            x = blk(x, pad=pad)

        q = self.q_in(fourier(self.centres, self.bands)) + self.q_pos
        q = q[None].expand(b, -1, -1) + cam
        for blk in self.decoder:
            q = blk(q, mem=x, mem_pad=pad)
        patches = self.head(self.out_norm(q))  # [b, gw*gh, P*P*3]
        gw, gh = W // PATCH, H // PATCH
        img = patches.view(b, gh, gw, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, H, W)
        return img + self.refine(img)


def loss_fn(pred, target):
    return F.l1_loss(pred, target)
