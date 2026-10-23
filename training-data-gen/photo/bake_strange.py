"""Bake the level textures at every strangeness step, for `datagen --strange DIR`.

    python3 training-data-gen/photo/bake_strange.py --upper data/level --deep data/level_deep --out data/strange

Each slot's texture runs from its upper (grounded) image at step 0 to its deep one
at the last step, through three overlapping stages:
  colour drift  its colours slide toward an acid palette of its own brightness
  swim          a smooth warp that grows
  infection     the deep texture grows through along a noise front, with a dark wet rim
Writes OUT/<step>/<slot>.png (256x256 RGB, seamless) and OUT/strips.png showing
every slot across the steps as in game (64x64, 16 colours, tiled 2x2).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_deep import N, load, palette, sample, save, spectral_noise

STEPS = 8
SLOTS = 8
PALETTES = ["acid", "toxic", "bruise", "candy", "infra", "bile", "acid", "candy"]


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def morph(up, deep, slot, s):
    y, x = np.mgrid[0:N, 0:N].astype(float)
    # Swim: a periodic warp, so the result still tiles.
    wa, wb = spectral_noise(2.4, 100 + slot), spectral_noise(2.4, 200 + slot)
    amp = 22 * smooth(0.05, 0.85, s)
    dx, dy = amp * (wa - 0.5) * 2, amp * (wb - 0.5) * 2
    img = sample(up, x + dx, y + dy)
    # Colour drift.
    l = img @ np.array([0.3, 0.59, 0.11])
    l = (l - l.min()) / (np.ptp(l) + 1e-9)
    acid = palette(l * 1.4 + 0.15 * slot, PALETTES[slot]) * (0.7 + 0.5 * l[..., None])
    c = smooth(0.0, 0.7, s)
    img = img * (1 - c) + acid * c
    # Infection, along a front whose values are evened out so it sweeps at a steady rate.
    front = spectral_noise(1.6, 300 + slot)
    front = np.argsort(np.argsort(front.ravel())).reshape(front.shape) / front.size
    reach = np.clip((s - 0.45) / 0.5, 0, 1) * 1.06
    m = np.clip((reach - front) / 0.06, 0, 1)[..., None]
    rim = np.clip(1 - np.abs(reach - front) / 0.03, 0, 1)[..., None] * (0 < reach < 1.05)
    d = sample(deep, x - dx, y - dy)
    img = img * (1 - m) + d * m
    return np.clip(img * (1 - 0.6 * rim), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", type=Path, default=Path("data/level"))
    ap.add_argument("--deep", type=Path, default=Path("data/level_deep"))
    ap.add_argument("--out", type=Path, default=Path("data/strange"))
    args = ap.parse_args()
    t = 128
    sheet = Image.new("RGB", (STEPS * t + 40, SLOTS * t), "black")
    dr = ImageDraw.Draw(sheet)
    for slot in range(SLOTS):
        up, deep = load(args.upper / f"{slot}.png"), load(args.deep / f"{slot}.png")
        for k in range(STEPS):
            d = args.out / str(k)
            d.mkdir(parents=True, exist_ok=True)
            save(morph(up, deep, slot, k / (STEPS - 1)), d / f"{slot}.png")
            im = Image.open(d / f"{slot}.png").resize((64, 64), Image.BOX).quantize(16).convert("RGB")
            tile = Image.new("RGB", (128, 128))
            for q in range(4):
                tile.paste(im, (q % 2 * 64, q // 2 * 64))
            sheet.paste(tile, (40 + k * t, slot * t))
        dr.text((4, slot * t + 4), str(slot), fill="yellow")
    sheet.save(args.out / "strips.png")
    print(f"{STEPS} steps x {SLOTS} slots in {args.out}")


if __name__ == "__main__":
    main()
