"""Estimate how raised a flesh texture's surface is, from its picture.

    python3 training-data-gen/photo/height_flesh.py data/painted/7

The textures are generated photos with no depth, but they're lit so that creases are
dark and what stands out is bright. Writes DIR/<slot>_h.png beside each DIR/<slot>.png
(256x256 RGB, grey, 128 is the wall's own level), wrapping like the texture does.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def height(rgb):
    luma = rgb @ np.array([0.3, 0.59, 0.11])
    # Broad shapes count for more than fine detail; the picture's overall lighting doesn't count.
    near = gaussian_filter(luma, 1.5, mode="wrap")
    wide = gaussian_filter(luma, 6.0, mode="wrap")
    level = gaussian_filter(luma, 36.0, mode="wrap")
    h = 0.35 * (near - wide) + 0.65 * (wide - level)
    # Creases are narrow and deep, swellings round.
    h = gaussian_filter(h, 1.0, mode="wrap")
    return np.clip(h / (2.5 * h.std()), -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    a = ap.parse_args()
    for path in sorted(a.dir.glob("[0-9].png")):
        h = height(np.asarray(Image.open(path).convert("RGB"), np.float32) / 255)
        grey = np.uint8(np.round((h + 1) * 127.5))
        Image.fromarray(np.stack([grey] * 3, -1)).save(a.dir / f"{path.stem}_h.png")


if __name__ == "__main__":
    main()
