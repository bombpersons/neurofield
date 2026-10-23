"""Fetch level textures from ambientCG (CC0, seamless) for the eight level slots.

    python3 training-data-gen/photo/fetch_ambientcg.py --out data/level [--pick 6=Onyx010 ...]

Downloads each material's 1K JPG zip, reads its colour map straight out of the
zip, and saves it as OUT/<slot>.png (256x256 RGB, the whole tile resized so it
still wraps), with OUT/credits.json. `datagen ... --level OUT` then uses them in
place of the drawn level textures. Slots keep their roles in the generator:
0 brick, 1 stone, 2 planks, 3 tiles, 4 moss, 5 grate, 6 wallpaper, 7 rings.
The defaults are the grounded set for the upper layers.
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

UA = {"User-Agent": "neurofield-datagen/0.1 (https://github.com/bombpersons/neurofield)"}
SIZE = 256

PICKS = {
    0: "Bricks097",
    1: "Bricks108A",
    2: "WoodFloor043",
    3: "PavingStones119",
    4: "Rock064",
    5: "MetalWalkway014",
    6: "Concrete042C",
    7: "Rock030",
}


def colour_map(asset):
    url = f"https://ambientcg.com/get?file={asset}_1K-JPG.zip"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".png")) and "color" in n.lower()]
    if not names:
        raise SystemExit(f"{asset}: no colour map in {zf.namelist()}")
    return Image.open(io.BytesIO(zf.read(names[0]))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pick", nargs="*", default=[], help="SLOT=AssetId overrides")
    args = ap.parse_args()
    picks = dict(PICKS)
    for p in args.pick:
        slot, asset = p.split("=")
        picks[int(slot)] = asset
    args.out.mkdir(parents=True, exist_ok=True)
    credits_path = args.out / "credits.json"
    credits = json.loads(credits_path.read_text()) if credits_path.exists() else {}
    for slot, asset in sorted(picks.items()):
        name = f"{slot}.png"
        if credits.get(name, {}).get("asset") == asset and (args.out / name).exists():
            continue
        colour_map(asset).resize((SIZE, SIZE), Image.LANCZOS).save(args.out / name)
        credits[name] = {"asset": asset, "license": "CC0", "page": f"https://ambientcg.com/view?id={asset}"}
        print(f"{name}  {asset}")
    credits_path.write_text(json.dumps(credits, indent=1))


if __name__ == "__main__":
    main()
