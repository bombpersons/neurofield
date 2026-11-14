"""Gather the dream classifier's classes (docs/dream_atlas.md, "Steering"): one folder of
256x256 images per class under data/dream_classes/, with a credits.json per class.

    python train/dream_classes.py openimages   # face, eye, mouth, hand crops (real photos)
    python train/dream_classes.py nothing [DIR]  # the plain dungeon, from datagen's frames
                                                 # (DIR: shards, default data/test)

Remake `nothing` whenever the dungeon's textures change, from views drawn with them (e.g.
`datagen gen --out DIR --scenes 200 --views 4` with the textures the game uses), then
fine-tune again: the classifier should know the dungeon as it now looks as "nothing".
    python train/dream_classes.py afhq         # animal faces (AFHQ: cats, dogs, wild), 600
    python train/dream_classes.py minc         # ordinary walls: brick, stone, wood, foliage...
    python train/dream_classes.py sdxl         # our SDXL flesh sets (data/skins_gen, deep_gen)

Open Images V7 (validation and test splits, whose box lists are small): images with a box of
the class, cropped round the box to a square with a margin, at least MIN_BOX of the image's
shorter side, from the dataset's own copies on S3. Most are CC BY 2.0; each crop's licence,
author and original page go in its class's credits.json.
"""

import csv
import io
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OI = ROOT / "data/downloads/openimages"
OUT = ROOT / "data/dream_classes"
SIZE = 256
PER_CLASS = 500
MIN_BOX = 0.12  # the box's longer side, as a share of the image's shorter side
MARGIN = 0.15
OI_CLASSES = {"/m/0dzct": "faces", "/m/014sv8": "eyes", "/m/0283dt1": "mouths", "/m/0k65p": "hands"}


def openimages():
    meta = {}
    for split in ("validation", "test"):
        with open(OI / f"{split}-images-with-rotation.csv", newline="") as f:
            for r in csv.DictReader(f):
                meta[r["ImageID"]] = (split, r)
    boxes = {c: [] for c in OI_CLASSES}
    for split in ("validation", "test"):
        with open(OI / f"{split}-annotations-bbox.csv", newline="") as f:
            for r in csv.DictReader(f):
                c = r["LabelName"]
                if c in boxes and r["IsGroupOf"] == "0" and r["IsTruncated"] == "0" and r["ImageID"] in meta:
                    box = [float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax")]
                    boxes[c].append((r["ImageID"], box))
    rng = random.Random(0)
    for c, name in OI_CLASSES.items():
        # One box an image, so a class isn't many crops of one crowd.
        seen, picks = set(), []
        rng.shuffle(boxes[c])
        for img, box in boxes[c]:
            if img not in seen:
                seen.add(img)
                picks.append((img, box))
        print(f"{name}: {len(boxes[c])} boxes in {len(picks)} images", flush=True)
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        credits, n = {}, 0

        def fetch(item):
            img, box = item
            split, r = meta[img]
            try:
                data = urllib.request.urlopen(f"https://open-images-dataset.s3.amazonaws.com/{split}/{img}.jpg", timeout=30).read()
                im = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:  # noqa: BLE001
                return None, f"{img}: {e}"
            rot = r.get("Rotation") or ""
            if rot not in ("", "0.0", "0"):
                im = im.rotate(-float(rot), expand=True)
            w, h = im.size
            x0, x1, y0, y1 = box[0] * w, box[1] * w, box[2] * h, box[3] * h
            side = max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN)
            if max(x1 - x0, y1 - y0) < MIN_BOX * min(w, h) or side < 64:
                return None, None
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            side = min(side, w, h)
            cx = min(max(cx, side / 2), w - side / 2)
            cy = min(max(cy, side / 2), h - side / 2)
            crop = im.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)))
            return (img, crop.resize((SIZE, SIZE), Image.LANCZOS), r), None

        with ThreadPoolExecutor(16) as pool:
            for i in range(0, len(picks), 64):
                if n >= PER_CLASS:
                    break
                for got, err in pool.map(fetch, picks[i : i + 64]):
                    if err:
                        print(err, flush=True)
                    if got and n < PER_CLASS:
                        img, crop, r = got
                        file = f"oi_{img}.png"
                        crop.save(out / file)
                        credits[file] = {"source": "Open Images V7", "licence": r["License"], "author": r["Author"],
                                         "title": r["Title"], "page": r["OriginalLandingURL"]}
                        n += 1
        json.dump(credits, open(out / "credits.json", "w"), indent=1)
        print(f"{name}: {n} crops", flush=True)


def nothing(src=None, count=600):
    """Square crops (a third to all of the frame's height) of plain PS1 dungeon frames from
    datagen's test shards (drawn without creatures), so the classifier has somewhere to put
    what is none of its classes."""
    import numpy as np
    out = OUT / "nothing"
    out.mkdir(parents=True, exist_ok=True)
    src = Path(src).resolve() if src else ROOT / "data/test"
    shards = sorted(src.glob("shard_*")) or [src]
    for old in out.glob("dg_*.png"):
        old.unlink()
    rng = np.random.default_rng(0)
    credits = {}
    for n in range(count):
        d = shards[n % len(shards)]
        images = np.load(d / "images.npy", mmap_mode="r")
        if images.ndim == 4:  # old shards: [views, 240, 320, 3]
            v = int(rng.integers(len(images)))
            frame = np.asarray(images[v])
        else:  # flat, with image_offsets.npy and each view's width in views.npy
            offs, views = np.load(d / "image_offsets.npy"), np.load(d / "views.npy", mmap_mode="r")
            v = int(rng.integers(len(views)))
            frame = np.asarray(images[offs[v] : offs[v + 1]]).reshape(240, int(views[v, 26]), 3)
        w = frame.shape[1]
        side = int(rng.integers(80, 241))
        y, x = int(rng.integers(0, 240 - side + 1)), int(rng.integers(0, w - side + 1))
        file = f"dg_{n:04}.png"
        Image.fromarray(frame[y : y + side, x : x + side]).resize((SIZE, SIZE), Image.NEAREST).save(out / file)
        credits[file] = {"source": f"datagen {d.relative_to(ROOT)} view {v}", "licence": "ours"}
    json.dump(credits, open(out / "credits.json", "w"), indent=1)
    print(f"nothing: {count} crops")


def afhq(count=600):
    """Animal faces from AFHQ (Choi, Uh, Yoo and Ha, StarGAN v2; CC BY-NC 4.0, from Flickr and
    Pixabay), unpacked in data/downloads/afhq: a third each cats, dogs and wild animals."""
    src = ROOT / "data/downloads/afhq/afhq/train"
    out = OUT / "animal_faces"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    credits = {}
    for kind in ("cat", "dog", "wild"):
        for f in rng.sample(sorted((src / kind).glob("*.jpg")), count // 3):
            file = f"afhq_{kind}_{f.stem}.png"
            Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(out / file)
            credits[file] = {"source": f"AFHQ train/{kind}/{f.name}", "licence": "CC BY-NC 4.0"}
    json.dump(credits, open(out / "credits.json", "w"), indent=1)
    print(f"animal_faces: {len(credits)}")


MINC_CLASSES = ["brick", "stone", "wood", "foliage", "tile", "metal", "polishedstone", "wallpaper"]


def minc(count=600):
    """The ordinary materials a grounded dungeon is made of, from MINC-2500 (Bell, Upchurch,
    Snavely and Bala, "Material Recognition in the Wild", CVPR 2015; photos from Flickr and
    Houzz, for research), unpacked in data/downloads/minc: 362x362 patches, scaled to SIZE."""
    src = ROOT / "data/downloads/minc/minc-2500/images"
    rng = random.Random(0)
    for c in MINC_CLASSES:
        out = OUT / c
        out.mkdir(parents=True, exist_ok=True)
        credits = {}
        for f in rng.sample(sorted((src / c).glob("*.jpg")), count):
            file = f"minc_{f.stem}.png"
            Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(out / file)
            credits[file] = {"source": f"MINC-2500 images/{c}/{f.name}", "licence": "research use (MINC-2500)"}
        json.dump(credits, open(out / "credits.json", "w"), indent=1)
        print(f"{c}: {len(credits)}", flush=True)


# Our SDXL renders (training-data-gen/photo/colab_textures.py, RealVisXL; prompts and seeds in
# their credits.json): motif -> class. Eyes and mouths join the real photos' classes.
SDXL = {"skins_gen": {m: m for m in ["flesh", "veins", "teeth", "wrinkles", "bone", "chitin", "scales", "suckers", "hair", "eyes", "mouths"]},
        "deep_gen": {"eye-wall": "eyes", "mouth-wall": "mouths", "teeth-wall": "teeth", "vein-membrane": "veins",
                     "flesh-wall": "flesh", "gut-wall": "flesh", "psychedelic-flesh": "flesh"}}


def sdxl():
    for set_, motifs in SDXL.items():
        src = ROOT / "data" / set_
        made = json.load(open(src / "credits.json")) if (src / "credits.json").exists() else {}
        for motif, name in motifs.items():
            out = OUT / name
            out.mkdir(parents=True, exist_ok=True)
            cf = out / "credits.json"
            credits = json.load(open(cf)) if cf.exists() else {}
            for f in sorted((src / motif).glob("*.png")):
                file = f"sdxl_{set_}_{motif}_{f.stem}.png"
                im = Image.open(f).convert("RGB")
                if im.size != (SIZE, SIZE):
                    im = im.resize((SIZE, SIZE), Image.LANCZOS)
                im.save(out / file)
                credits[file] = {"source": f"data/{set_}/{motif}/{f.name}", **made.get(f"{motif}/{f.name}", {})}
            json.dump(credits, open(cf, "w"), indent=1)
    for d in sorted(OUT.iterdir()):
        print(f"{d.name}: {len(list(d.glob('*.png')))}")


if __name__ == "__main__":
    {"openimages": openimages, "nothing": nothing, "afhq": afhq, "minc": minc, "sdxl": sdxl}[sys.argv[1]](*sys.argv[2:])
