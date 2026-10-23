"""Contact sheets and a seam check for a skin library.

    python3 training-data-gen/photo/sheets.py data/skins_gen --out /tmp/sheets [--tiled]

Writes one sheet per motif (OUT/<motif>.png), each image labelled with its
number and seam score, and prints the worst seams. The score compares the
jump across the wrapped edge with the image's ordinary neighbour-to-neighbour
differences: about 1 when it tiles, well above when there is a seam. With
--tiled each cell shows the image 2x2 so seams are visible by eye.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def seam(a):
    a = a.astype(np.float32)
    inner = (np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()) / 2
    edge = (np.abs(a[0] - a[-1]).mean() + np.abs(a[:, 0] - a[:, -1]).mean()) / 2
    return edge / max(inner, 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tiled", action="store_true")
    ap.add_argument("--cell", type=int, default=128)
    ap.add_argument("--cols", type=int, default=12)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scores = []
    for d in sorted(p for p in args.lib.iterdir() if p.is_dir() and p.name != "chunks"):
        files = sorted(d.glob("*.png"))
        if not files:
            continue
        c, cols = args.cell, args.cols
        rows = (len(files) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * c, rows * c), "black")
        draw = ImageDraw.Draw(sheet)
        for i, f in enumerate(files):
            im = Image.open(f).convert("RGB")
            s = seam(np.asarray(im))
            scores.append((s, f))
            if args.tiled:
                im = Image.fromarray(np.tile(np.asarray(im), (2, 2, 1)))
            x, y = i % cols * c, i // cols * c
            sheet.paste(im.resize((c, c), Image.LANCZOS), (x, y))
            draw.text((x + 2, y + 2), f"{f.stem} {s:.1f}", fill="white", stroke_width=1, stroke_fill="black")
        sheet.save(args.out / f"{d.name}.png")
        print(f"{d.name:9} {len(files):4} images -> {args.out / d.name}.png")
    scores.sort(reverse=True)
    print("worst seams:", ", ".join(f"{f.parent.name}/{f.stem} {s:.2f}" for s, f in scores[:10]))


if __name__ == "__main__":
    main()
