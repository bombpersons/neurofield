"""Make candidate textures for the deep layers of the dungeon, where things get weird.

    python3 training-data-gen/photo/make_deep.py --out data/deep [--level data/level] [--skins data/skins]

Every texture is 256x256 RGB and tiles seamlessly, saved as OUT/<name>.png, with
OUT/sheet.png showing each one large and as it would be in game (64x64, 16 colours).
Kinds:
  julia-*     Julia sets on a torus (z0 runs round two circles), in acid bands
  marble-*    domain-warped noise, like oil on water
  turing-*    Gray-Scott reaction-diffusion: coral, spots, mazes, worms
  opart-*     warped checkers and interfering rings
  kaleido-*   mirror-folded (p4m) crops of the level textures and creature photos
  acid-*      the level textures recoloured through acid palettes
  eyes-*, teeth-*  wallpapers of creature photos on flesh
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

N = 256
RNG = np.random.default_rng(7)

# Cosine palettes (a + b cos(2 pi (c t + d))), per channel.
PALETTES = {
    "acid": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.33, 0.67]),
    "toxic": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 0.5], [0.8, 0.9, 0.3]),
    "bruise": ([0.5, 0.3, 0.5], [0.5, 0.3, 0.5], [1.0, 1.0, 1.0], [0.6, 0.9, 0.3]),
    "candy": ([0.6, 0.5, 0.6], [0.4, 0.5, 0.4], [2.0, 1.0, 1.0], [0.5, 0.2, 0.25]),
    "infra": ([0.5, 0.2, 0.3], [0.5, 0.4, 0.5], [1.0, 0.7, 0.4], [0.0, 0.15, 0.2]),
    "bile": ([0.4, 0.5, 0.2], [0.5, 0.5, 0.3], [1.0, 1.0, 2.0], [0.3, 0.2, 0.2]),
}


def palette(t, name, shift=0.0):
    a, b, c, d = (np.array(v)[None, None, :] for v in PALETTES[name])
    return np.clip(a + b * np.cos(2 * np.pi * (c * t[..., None] + d + shift)), 0, 1)


def grid():
    y, x = np.mgrid[0:N, 0:N] / N
    return x, y


def spectral_noise(beta, seed):
    """Periodic noise with a 1/f^beta spectrum, 0..1."""
    r = np.random.default_rng(seed)
    fy = np.fft.fftfreq(N)[:, None]
    fx = np.fft.rfftfreq(N)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spec = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f**beta
    spec[0, 0] = 0
    n = np.fft.irfft2(spec, s=(N, N))
    return (n - n.min()) / (n.max() - n.min())


def sample(img, x, y):
    """Bilinear sample of img (N, N[, 3]) at texel coords, wrapping."""
    h, w = img.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    g = lambda xx, yy: img[yy % h, xx % w]
    return (g(x0, y0) * (1 - fx) + g(x0 + 1, y0) * fx) * (1 - fy) + (g(x0, y0 + 1) * (1 - fx) + g(x0 + 1, y0 + 1) * fx) * fy


def julia(c, r1, r2, pal, bands=0.06):
    x, y = grid()
    z = r1 * np.exp(2j * np.pi * x) + r2 * np.exp(2j * np.pi * y) * np.exp(0.7j)
    count = np.zeros(z.shape)
    alive = np.ones(z.shape, bool)
    for i in range(80):
        z[alive] = z[alive] ** 2 + c
        esc = alive & (np.abs(z) > 8)
        count[esc] = i + 1 - np.log2(np.log(np.abs(z[esc])))
        alive &= ~esc
    count[alive] = 0
    rgb = palette(count * bands, pal)
    rgb[alive] *= 0.08
    return rgb


def marble(seed, pal, warp=0.35):
    a, b, c = (spectral_noise(2.2, seed + k) for k in range(3))
    x, y = grid()
    qx = sample(a, (x + warp * b) * N, (y + warp * c) * N)
    qy = sample(b, (x + warp * c) * N, (y + warp * a) * N)
    t = sample(c, (x + warp * 2 * qx) * N, (y + warp * 2 * qy) * N)
    return palette(t * 2.5 + 0.3 * qx, pal)


def turing(f, k, pal, steps=5000, seed=0):
    """Gray-Scott as Karl Sims sets it up (Da 1, Db 0.5, weighted 3x3 Laplacian), on a
    128 torus so the features stay big enough for 64 px, then smoothly doubled."""
    n = N // 2
    r = np.random.default_rng(seed)
    u, v = np.ones((n, n)), np.zeros((n, n))
    for _ in range(25):
        cx, cy = r.integers(0, n, 2)
        v[np.ix_((np.arange(cy, cy + 6)) % n, (np.arange(cx, cx + 6)) % n)] = 1
    roll = np.roll

    def lap(a):
        side = roll(a, 1, 0) + roll(a, -1, 0) + roll(a, 1, 1) + roll(a, -1, 1)
        diag = roll(roll(a, 1, 0), 1, 1) + roll(roll(a, 1, 0), -1, 1) + roll(roll(a, -1, 0), 1, 1) + roll(roll(a, -1, 0), -1, 1)
        return 0.2 * side + 0.05 * diag - a

    for _ in range(steps):
        uvv = u * v * v
        u += lap(u) - uvv + f * (1 - u)
        v += 0.5 * lap(v) + uvv - (f + k) * v
    y, x = np.mgrid[0:N, 0:N] / 2.0
    v = sample(v, x, y)
    t = (v - v.min()) / (np.ptp(v) + 1e-9)
    return palette(t * 0.9, pal)


def opart(kind, pal):
    x, y = grid()
    if kind == "checker":
        wx = x + 0.08 * np.sin(2 * np.pi * (y * 2 + 0.3 * np.sin(2 * np.pi * x)))
        wy = y + 0.08 * np.sin(2 * np.pi * (x * 3))
        t = ((np.floor(wx * 8) + np.floor(wy * 8)) % 2) * 0.5 + 0.25 * np.sin(2 * np.pi * (wx + wy) * 2)
    else:
        t = np.zeros_like(x)
        for cx, cy, k in [(0.25, 0.25, 9), (0.75, 0.6, 7), (0.4, 0.85, 11)]:
            dx, dy = np.minimum(abs(x - cx), 1 - abs(x - cx)), np.minimum(abs(y - cy), 1 - abs(y - cy))
            t += np.sin(2 * np.pi * k * np.sqrt(dx * dx + dy * dy) * 0.6)
        t = (t > 0) * 0.6 + 0.1 * t
    return palette(t, pal)


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((N, N)), dtype=np.float64) / 255


def kaleido(src, scale, angle, origin):
    """Fold the tile into eight mirrored triangles (p4m) and fill them from a crop of src."""
    x, y = grid()
    u, v = np.abs(2 * x - 1), np.abs(2 * y - 1)
    u, v = np.maximum(u, v), np.minimum(u, v)
    ca, sa = np.cos(angle), np.sin(angle)
    sx = origin[0] + scale * (u * ca - v * sa)
    sy = origin[1] + scale * (u * sa + v * ca)
    return sample(src, sx * N, sy * N)


def acid(src, pal, shift):
    l = src @ np.array([0.3, 0.59, 0.11])
    l = (l - l.min()) / (np.ptp(l) + 1e-9)
    return palette(l * 1.6 + shift, pal) * (0.75 + 0.5 * l[..., None])


def disc_wallpaper(bg, photos, per_side, stagger, size):
    """Round crops of photos laid on a wrapping grid over bg."""
    out = bg.copy()
    x, y = grid()
    cell = 1.0 / per_side
    i = 0
    for row in range(per_side):
        for col in range(per_side):
            cx = (col + 0.5 + (0.5 if stagger and row % 2 else 0)) * cell
            cy = (row + 0.5) * cell
            ph = photos[i % len(photos)]
            i += 1
            dx = (x - cx + 0.5) % 1 - 0.5
            dy = (y - cy + 0.5) % 1 - 0.5
            r = np.sqrt(dx * dx + dy * dy) / (size * cell / 2)
            m = np.clip((1 - r) * 6, 0, 1)[..., None]
            shade = np.clip(1.1 - 0.5 * r, 0, 1)[..., None]
            crop = sample(ph, (0.5 + dx / (size * cell) * 0.6) * N, (0.5 + dy / (size * cell) * 0.6) * N)
            out = out * (1 - m) + crop * shade * m
    return out


def brick_wallpaper(photos, rows, per_row):
    """Rectangular crops of photos in brick bond, dark mortar."""
    x, y = grid()
    out = np.zeros((N, N, 3)) + 0.05
    h = 1.0 / rows
    w = 1.0 / per_row
    for row in range(rows):
        off = 0.5 * w if row % 2 else 0
        for col in range(per_row):
            ph = photos[(row * per_row + col) % len(photos)]
            x0 = col * w + off
            lx = ((x - x0) % 1) / w
            ly = (y - row * h) / h
            m = ((lx > 0.04) & (lx < 0.96) & (ly > 0.06) & (ly < 0.94))[..., None]
            crop = sample(ph, (0.2 + lx * 0.6) * N, (0.25 + ly * 0.5) * N)
            out = np.where(m, crop, out)
    return out


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", type=Path, default=Path("data/level"))
    ap.add_argument("--skins", type=Path, default=Path("data/skins"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    made = {}
    made["julia-acid"] = julia(-0.8 + 0.156j, 0.9, 0.6, "acid")
    made["julia-bruise"] = julia(0.285 + 0.01j, 0.8, 0.7, "bruise", 0.09)
    made["julia-infra"] = julia(-0.4 + 0.6j, 0.7, 0.7, "infra")
    made["julia-toxic"] = julia(-0.70176 - 0.3842j, 0.85, 0.5, "toxic", 0.05)
    made["marble-candy"] = marble(1, "candy")
    made["marble-bile"] = marble(5, "bile")
    made["marble-acid"] = marble(9, "acid", 0.5)
    made["turing-coral"] = turing(0.0545, 0.062, "infra")
    made["turing-spots"] = turing(0.0367, 0.0649, "toxic", seed=1)
    made["turing-maze"] = turing(0.029, 0.057, "bruise", seed=2)
    made["turing-worms"] = turing(0.078, 0.061, "candy", seed=3)
    made["opart-checker"] = opart("checker", "acid")
    made["opart-rings"] = opart("rings", "bruise")

    level = {p.stem: load(p) for p in sorted(args.level.glob("*.png"))}
    eyes = [load(p) for p in sorted((args.skins / "eyes").glob("*.png"))[:24]]
    teeth = [load(p) for p in sorted((args.skins / "teeth").glob("*.png"))]
    flesh = [load(p) for p in sorted((args.skins / "flesh").glob("*.png"))]
    if level:
        made["kaleido-bricks"] = kaleido(level["0"], 0.35, 0.3, (0.2, 0.3))
        made["kaleido-onyx"] = kaleido(level["6"], 0.5, 0.9, (0.1, 0.4))
        made["kaleido-tiles"] = kaleido(level["3"], 0.6, 0.0, (0.2, 0.2))
        made["acid-bricks"] = acid(level["0"], "acid", 0.0)
        made["acid-stone"] = acid(level["1"], "toxic", 0.2)
        made["acid-rust"] = acid(level["5"], "candy", 0.5)
        made["acid-wood"] = acid(level["2"], "infra", 0.1)
    if eyes:
        made["kaleido-eye"] = kaleido(eyes[0], 0.45, 0.2, (0.3, 0.3))
        made["kaleido-eye2"] = kaleido(eyes[7], 0.4, 1.1, (0.35, 0.25))
    if eyes and flesh:
        bg = kaleido(flesh[0], 0.6, 0.4, (0.1, 0.1))
        made["eyes-grid"] = disc_wallpaper(bg, eyes, 3, True, 0.8)
        made["eyes-swarm"] = disc_wallpaper(bg, eyes[5:], 5, True, 0.7)
    if teeth:
        made["teeth-bond"] = brick_wallpaper(teeth, 4, 3)
    if flesh:
        made["kaleido-flesh"] = kaleido(flesh[2 % len(flesh)], 0.5, 0.7, (0.2, 0.2))

    for name, img in made.items():
        save(img, args.out / f"{name}.png")

    # Sheet: each texture tiled 2x2 at 64 px and 16 colours (as in game), its name under it.
    cols, t = 6, 192
    names = list(made)
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * t, rows * (t + 14)), "black")
    dr = ImageDraw.Draw(sheet)
    for i, name in enumerate(names):
        im = Image.open(args.out / f"{name}.png").resize((64, 64), Image.BOX)
        im = im.quantize(16, method=Image.Quantize.MEDIANCUT).convert("RGB")
        tile = Image.new("RGB", (128, 128))
        for k in range(4):
            tile.paste(im, (k % 2 * 64, k // 2 * 64))
        x, y = i % cols * t, i // cols * (t + 14)
        sheet.paste(tile.resize((t, t), Image.NEAREST), (x, y))
        dr.text((x + 2, y + t), name, fill="yellow")
    sheet.save(args.out / "sheet.png")
    print(f"{len(made)} textures, sheet at {args.out / 'sheet.png'}")


if __name__ == "__main__":
    main()
