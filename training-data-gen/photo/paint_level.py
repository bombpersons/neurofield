"""Paint the level textures in a plain video-game style, for `datagen --strange DIR`.

    python3 training-data-gen/photo/paint_level.py --flesh data/strange/7 --out data/painted

Blocky shapes, hard bevels and a few banded colours per surface, drawn at the 64x64 the
level uses, so that they look nothing like the flesh they turn into. Writes
OUT/<step>/<slot>.png (256x256, seamless): the painted texture up to the next-to-last
step, FLESH's at the last. Also OUT/sheet.png, each texture tiled 2x2.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

N = 64
STEPS = 8


def noise(period, seed):
    """Smooth noise, 0..1, that tiles: random lattice values with cosine interpolation."""
    r = np.random.default_rng(seed).random((period, period))
    t = np.arange(N) / N * period
    i = np.floor(t).astype(int)
    f = t - i
    f = (1 - np.cos(f * np.pi)) / 2
    y, x = np.meshgrid(range(N), range(N), indexing="ij")
    a, b = i[x], i[y]
    u, v = f[x], f[y]
    at = lambda p, q: r[q % period, p % period]
    return (at(a, b) * (1 - u) + at(a + 1, b) * u) * (1 - v) + (at(a, b + 1) * (1 - u) + at(a + 1, b + 1) * u) * v


def fractal(period, seed):
    return 0.6 * noise(period, seed) + 0.28 * noise(period * 2, seed + 1) + 0.12 * noise(period * 4, seed + 2)


def inset(cells):
    """Each texel's distance in from the edge of its cell, up to 6, wrapping round."""
    d = np.zeros((N, N))
    inside = np.ones((N, N), bool)
    for k in range(1, 7):
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            inside &= np.roll(cells, (dy * k, dx * k), (0, 1)) == cells
        d += inside
    return d


def bevelled(cells, width, seed, spread=0.16):
    """Shade 0..1 for blocks: dark in the joints, lit on top and left edges, dark on bottom and right."""
    d = np.minimum(inset(cells), width) / width
    gy = (np.roll(d, -1, 0) - np.roll(d, 1, 0)) / 2
    gx = (np.roll(d, -1, 1) - np.roll(d, 1, 1)) / 2
    tone = np.random.default_rng(seed).random(int(cells.max()) + 1)[cells]
    shade = 0.5 + (tone - 0.5) * spread + 0.75 * (gy + gx) + 0.14 * (fractal(6, seed) - 0.5) * 2
    return np.where(inset(cells) == 0, 0.0, shade)


def bricks(rows, across, seed):
    h, w = N // rows, N // across
    y, x = np.mgrid[0:N, 0:N]
    r = y // h
    c = ((x + (r % 2) * w // 2) % N) // w
    return r * across + c


def voronoi(count, seed):
    pts = np.random.default_rng(seed).random((count, 2)) * N
    y, x = np.mgrid[0:N, 0:N]
    dx = np.abs(x[..., None] - pts[:, 0])
    dy = np.abs(y[..., None] - pts[:, 1])
    dx, dy = np.minimum(dx, N - dx), np.minimum(dy, N - dy)
    return np.argmin(dx**2 + dy**2, axis=-1)


def scatter(count, seed):
    """Chips: a few dark pits with a lit lower edge."""
    r = np.random.default_rng(seed)
    out = np.zeros((N, N))
    for _ in range(count):
        px, py, size = r.integers(N), r.integers(N), r.integers(1, 3)
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if dx * dx + dy * dy <= size * size:
                    out[(py + dy) % N, (px + dx) % N] = -0.3
        out[(py + size + 1) % N, px] = 0.18
    return out


def brick(seed):
    return bevelled(bricks(4, 2, seed), 2, seed) + scatter(10, seed)


def blocks(seed):
    return bevelled(bricks(2, 2, seed), 3, seed, 0.1) + scatter(14, seed)


def planks(seed):
    y, x = np.mgrid[0:N, 0:N]
    cells = y // 16
    wavy = fractal(4, seed) * 10 + y * 0.9
    grain = (np.sin(wavy * 1.5) * 0.5 + 0.5) * 0.16 - 0.08
    shade = bevelled(cells, 1, seed, 0.3) + grain
    nails = np.zeros((N, N))
    for py in range(4, N, 16):
        for px in [6, 38]:
            nails[py + 4, px] = -0.45
            nails[py + 3, px] = 0.3
    return shade + nails


def cobbles(seed):
    return bevelled(voronoi(14, seed), 4, seed, 0.3)


def ground(seed):
    return 0.35 + 0.5 * fractal(5, seed) + 0.3 * (noise(N // 2, seed + 7) > 0.78) - 0.25 * (noise(N // 2, seed + 11) > 0.8)


def tiles(seed):
    y, x = np.mgrid[0:N, 0:N]
    cells = y // 16 * 4 + x // 16
    shade = bevelled(cells, 1, seed, 0.1)
    # A diamond set in each tile.
    dx, dy = np.abs(x % 16 - 7.5), np.abs(y % 16 - 7.5)
    return np.where(dx + dy < 4.5, shade + 0.32, np.where(dx + dy < 5.5, shade - 0.3, shade))


def slab(seed):
    y, x = np.mgrid[0:N, 0:N]
    shade = bevelled(np.zeros((N, N), int) + ((y // 32) * 2 + x // 32), 2, seed, 0.06) + scatter(8, seed)
    # A carved frame in each slab.
    dx, dy = np.abs(x % 32 - 15.5), np.abs(y % 32 - 15.5)
    r = np.maximum(dx, dy)
    return np.where((r > 9) & (r < 10), shade - 0.28, np.where((r >= 10) & (r < 11), shade + 0.2, shade))


def rock(seed):
    return bevelled(voronoi(22, seed), 6, seed, 0.45)


# Each surface's colours from shadow to highlight: shadows cool, highlights warm.
PAINT = [
    (brick, ["1c1420", "4a1f24", "7d3528", "a64e30", "c9774a", "e6a874"]),
    (blocks, ["16161f", "343645", "565a68", "7d8088", "a5a59f", "d3cdb8"]),
    (planks, ["1b1014", "4a2a1a", "74451f", "9a6428", "bf8b3c", "e0b768"]),
    (cobbles, ["18141c", "3e3a44", "625c5e", "8a8071", "b0a488", "d9cfaa"]),
    (ground, ["121a14", "28361c", "48542a", "6c7236", "958f45", "c2b663"]),
    (tiles, ["1e1224", "4c2030", "85382f", "b65b35", "dd9145", "f2c776"]),
    (slab, ["12141f", "2c3448", "485671", "6c7f94", "98a8ad", "cdd2c0"]),
    (rock, ["0f0f18", "24222e", "3d3843", "5b5258", "7e736e", "a79a85"]),
]


def paint(shape, colours, seed):
    ramp = np.array([[int(c[i : i + 2], 16) for i in (0, 2, 4)] for c in colours], np.uint8)
    # Hard bands, not gradients.
    bands = np.clip(shape(seed), 0, 0.999) * len(ramp)
    return ramp[bands.astype(int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flesh", type=Path, default=Path("data/strange/7"))
    ap.add_argument("--out", type=Path, default=Path("data/painted"))
    a = ap.parse_args()
    images = [paint(shape, colours, 100 + i) for i, (shape, colours) in enumerate(PAINT)]
    for step in range(STEPS):
        (a.out / str(step)).mkdir(parents=True, exist_ok=True)
        for slot, img in enumerate(images):
            path = a.out / f"{step}/{slot}.png"
            if step == STEPS - 1:
                shutil.copy(a.flesh / f"{slot}.png", path)
            else:
                Image.fromarray(img).resize((256, 256), Image.NEAREST).save(path)
    sheet = np.concatenate([np.concatenate([np.tile(i, (2, 2, 1)) for i in images[k : k + 4]], 1) for k in (0, 4)], 0)
    Image.fromarray(sheet).resize((sheet.shape[1] * 3, sheet.shape[0] * 3), Image.NEAREST).save(a.out / "sheet.png")


if __name__ == "__main__":
    main()
