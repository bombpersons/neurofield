"""Check gbuffer.py against the game's own frames: rebuild plain views from the G-buffer the way
psx-bevy's PsxMaterial shades them (texel times vertex colour with the torch, fog, 15-bit
colour) and compare with the shard's images. If the G-buffer agrees with the renderer, the
rebuilt frames line up with the real ones to the pixel, and moving them a pixel any way only
makes them worse.

    python train/check_gbuffer.py SHARD_DIR TEXTURE_DIR [--views 8] [--out check.png]

TEXTURE_DIR holds the level textures the shard was drawn with (<slot>.png, 256x256: the
painted textures' step 0 for sculpted data). Textures 8-11 (scales, bone, fur, doors) are
drawn by the engine, so their pixels are left out.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gbuffer import H, gbuffer

SIZE = 64  # tex::SIZE


def level_textures(d):
    """Textures 0-7 as the engine loads them: 256x256 averaged down to 64x64 (without its
    16-colour palette), [8, 64, 64, 3] in 0..1."""
    out = []
    for t in range(8):
        rgb = np.asarray(Image.open(Path(d) / f"{t}.png").convert("RGB"), np.float32) / 255.0
        out.append(rgb.reshape(SIZE, 4, SIZE, 4, 3).mean((1, 3)))
    return torch.as_tensor(np.stack(out))


def rebuild(tris, v, g, tex):
    """The view as PsxMaterial draws it, from the G-buffer: [240, width, 3] in 0..1, and
    which pixels it can speak for."""
    width = int(v[26])
    tri, w, piece, lam = g["tri"], g["weights"], g["piece"], g["lam"]
    t = torch.as_tensor(np.asarray(tris), dtype=torch.float32)
    shown = tri >= 0
    rows = t[tri.clamp(min=0)]  # [H, W, 35]
    texid = rows[..., 15].long()
    ok = shown & (texid < 8)
    # Texture coordinates: tris.npy has texels / 64, the renderer texels / 63 (texel 63 is 1.0).
    uv = (rows[..., 9:15].view(H, width, 3, 2) * w[..., None]).sum(2) * SIZE / (SIZE - 1)
    tx = (uv * SIZE).floor().long().clamp(0, SIZE - 1)
    texel = tex[texid.clamp(max=7), tx[..., 1], tx[..., 0]]
    # Vertex colours with the torch, at the pieces' corners, clamped, then mixed across the piece.
    pv = torch.as_tensor(g["pieces"], dtype=torch.float32)
    pc = pv[piece.clamp(min=0)]  # [H, W, 3 corners, xyz], view space
    n = torch.linalg.cross(pc[..., 1, :] - pc[..., 0, :], pc[..., 2, :] - pc[..., 0, :])
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    # The pieces' corners' own weights on the triangle: the lam-mix of them is `w`.
    from gbuffer import project

    _, _, parent, pweights, _ = project(tris, v)
    pw = torch.as_tensor(pweights, dtype=torch.float32)[piece.clamp(min=0)]  # [H, W, 3, 3]
    col = rows[..., 16:25].view(H, width, 3, 3) * 255.0
    alb = rows[..., 25:34].view(H, width, 3, 3) * 255.0
    ccol = (pw[..., :, :, None] * col[..., None, :, :]).sum(3)  # [H, W, piece corner, rgb]
    calb = (pw[..., :, :, None] * alb[..., None, :, :]).sum(3)
    torch_rgb = torch.as_tensor(v[19:22], dtype=torch.float32)
    reach = float(v[22])
    dist = pc.norm(dim=-1).clamp(min=1.0 / 256.0)
    fall = (1 - dist / max(reach, 1.0 / 256.0)).clamp(min=0) ** 2
    lambert = ((n[..., None, :] * -pc).sum(-1) / dist).clamp(min=0)
    light = calb * torch_rgb * (fall * lambert)[..., None]
    vcol = ((ccol + light) / 128.0).clamp(max=255.0 / 128.0)
    c = (lam[..., :, None] * vcol).sum(2).clamp(max=2.0)
    rgb = texel * c
    # Fog by distance to the point on the surface.
    z = 1.0 / g["inv_z"].clamp(min=1e-6)
    ys, xs = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(width) + 0.5, indexing="ij")
    p = torch.stack([(xs - width / 2) * z / v[13], (ys - 120) * z / v[13], z], -1)
    f = ((p.norm(dim=-1) - v[14]) / max(v[15] - v[14], 1e-4)).clamp(0, 1)
    fog = torch.as_tensor(v[23:26], dtype=torch.float32)
    rgb = rgb + (fog - rgb) * f[..., None]
    q = (torch.round(rgb.clamp(0, 1) * 255) / 8).floor()
    return (q * 8 + (q / 4).floor()) / 255.0, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("textures")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--out", default="gbuffer_check.png")
    a = ap.parse_args()
    d = Path(a.shard)
    tris = np.load(d / "tris.npy", mmap_mode="r")
    offs = np.load(d / "tri_offsets.npy")
    views = np.load(d / "views.npy")
    images = np.load(d / "images.npy", mmap_mode="r")
    image_offs = np.load(d / "image_offsets.npy")
    amounts = np.load(d / "amounts.npy") if (d / "amounts.npy").exists() else np.zeros(len(views))
    tex = level_textures(a.textures)
    plain = np.flatnonzero(amounts == 0)
    pick = plain[np.linspace(0, len(plain) - 1, min(a.views, len(plain))).astype(int)]
    rows, errs = [], []
    for vi in pick:
        v = views[vi]
        width = int(v[26])
        s = int(v[0])
        scene = np.asarray(tris[offs[s] : offs[s + 1]])
        g = gbuffer(scene, v)
        got, ok = rebuild(scene, v, g, tex)
        target = torch.as_tensor(np.asarray(images[image_offs[vi] : image_offs[vi + 1]]).reshape(H, width, 3) / 255.0, dtype=torch.float32)
        err = (got - target).abs().mean(-1)
        shifted = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            moved = torch.roll(got, (dy, dx), (0, 1))
            m = ok & torch.roll(ok, (dy, dx), (0, 1))
            m[:2], m[-2:], m[:, :2], m[:, -2:] = False, False, False, False
            shifted.append((moved - target).abs().mean(-1)[m].mean().item())
        e = err[ok]
        errs.append((e.mean().item(), (e > 0.15).float().mean().item(), min(shifted), ok.float().mean().item()))
        print(f"view {vi} ({width} wide): {ok.float().mean():.0%} of pixels checked, mean error {e.mean():.4f}, "
              f"{(e > 0.15).float().mean():.1%} off by more than 0.15; moved a pixel: {min(shifted):.4f} at best", flush=True)
        diff = (err * 4).clamp(0, 1)[..., None].expand(-1, -1, 3) * ok[..., None]
        rows.append(torch.cat([target, got * ok[..., None], diff], 1))
    print(f"all: mean error {np.mean([e[0] for e in errs]):.4f}, off {np.mean([e[1] for e in errs]):.1%}, "
          f"moved a pixel {np.mean([e[2] for e in errs]):.4f}")
    wide = max(r.shape[1] for r in rows)
    grid = torch.cat([torch.nn.functional.pad(r, (0, 0, 0, wide - r.shape[1])) for r in rows], 0)
    Image.fromarray((grid.numpy() * 255).astype(np.uint8)).save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
