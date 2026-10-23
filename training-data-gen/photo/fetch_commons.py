"""Fetch photos for creature skins from Wikimedia Commons.

    python3 training-data-gen/photo/fetch_commons.py --out data/skins [--per-query 20]

Searches Commons per skin motif, keeps freely licensed bitmaps (CC0, public
domain, CC BY, CC BY-SA), and saves each as a 256x256 PNG (the centre square,
resized) in OUT/<motif>/, with OUT/credits.json listing title, author, licence
and source page for attribution. Files already present are skipped, and
deleting a bad pick and re-running does not fetch it again (see OUT/rejected.txt).
"""

import argparse
import io
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
UA = "neurofield-datagen/0.1 (https://github.com/bombpersons/neurofield)"
SIZE = 256

QUERIES = {
    "flesh": [
        "raw meat texture close-up", "tongue surface macro", "tripe close-up", "skin pores macro",
        "raw beef close-up", "raw liver", "pork belly raw", "cow tongue", "raw chicken skin",
    ],
    "eyes": ["human eye iris macro", "eye close-up iris pupil", "goat eye close-up", "reptile eye close-up"],
    "teeth": ["teeth gums close-up", "human teeth dental photograph", "shark jaw teeth", "horse teeth close-up"],
    "mouths": ["open mouth teeth close-up", "lips close-up", "lamprey mouth", "fish mouth teeth close-up"],
    "suckers": [
        "octopus suckers close-up", "squid tentacle suckers", "octopus arm suckers macro",
        "octopus tentacle", "Octopus vulgaris arm", "cuttlefish tentacle", "raw octopus",
    ],
    "veins": ["fundus photograph retina", "veins skin close-up", "bloodshot eye vessels", "placenta vessels"],
    "scales": ["snake scales macro", "lizard skin macro", "fish scales close-up", "crocodile skin close-up"],
    "chitin": ["beetle elytra macro", "insect exoskeleton macro", "crab shell close-up", "cockroach macro"],
    "hair": [
        "fur macro texture", "animal fur close-up", "sheep wool close-up", "horse mane close-up",
        "dog fur texture", "goat hair close-up", "yak fur",
    ],
    "wrinkles": ["elephant skin texture", "wrinkled skin close-up", "naked mole rat", "rhinoceros skin close-up"],
    "bone": [
        "bone surface texture", "cancellous bone", "skull bone close-up", "bone fracture surface",
        "cow skull", "antler close-up", "whale bone texture", "animal skull close-up",
    ],
}

FREE = re.compile(r"^(cc0|public domain|pd\b|cc[ -]by(-sa)?[ -]\d)", re.I)


def get(params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def search(q, n):
    data = get({
        "action": "query", "generator": "search", "gsrnamespace": 6, "gsrlimit": n,
        "gsrsearch": f"{q} filetype:bitmap", "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 512,
    })
    return data.get("query", {}).get("pages", [])


def plain(html):
    return re.sub(r"<[^>]+>", "", html or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-query", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    credits_path = args.out / "credits.json"
    credits = json.loads(credits_path.read_text()) if credits_path.exists() else {}
    rejected_path = args.out / "rejected.txt"
    rejected = set(rejected_path.read_text().split("\n")) if rejected_path.exists() else set()
    # Anything credited but no longer on disk was deleted by hand: remember it as rejected.
    for name, c in list(credits.items()):
        if not (args.out / name).exists():
            rejected.add(c["title"])
            del credits[name]
    seen = {c["title"] for c in credits.values()} | rejected

    for motif, queries in QUERIES.items():
        d = args.out / motif
        d.mkdir(exist_ok=True)
        for q in queries:
            for page in search(q, args.per_query):
                title = page["title"]
                info = (page.get("imageinfo") or [{}])[0]
                meta = info.get("extmetadata", {})
                lic = plain(meta.get("LicenseShortName", {}).get("value"))
                if title in seen or info.get("mime") not in ("image/jpeg", "image/png") or not FREE.match(lic):
                    continue
                if min(info.get("width", 0), info.get("height", 0)) < 300:
                    continue
                seen.add(title)
                try:
                    req = urllib.request.Request(info["thumburl"], headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        img = Image.open(io.BytesIO(r.read())).convert("RGB")
                except Exception as e:  # noqa: BLE001 - one bad file should not stop the run
                    print(f"skip {title}: {e}")
                    continue
                w, h = img.size
                s = min(w, h)
                img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIZE, SIZE), Image.LANCZOS)
                n = 0
                while (d / f"{n:03d}.png").exists():
                    n += 1
                name = f"{motif}/{n:03d}.png"
                img.save(args.out / name)
                credits[name] = {
                    "title": title,
                    "author": plain(meta.get("Artist", {}).get("value")),
                    "license": lic,
                    "page": info.get("descriptionurl"),
                    "query": q,
                }
                print(f"{name}  {lic:14}  {title}")
                time.sleep(0.2)
        credits_path.write_text(json.dumps(credits, indent=1))
    rejected_path.write_text("\n".join(sorted(rejected)))


if __name__ == "__main__":
    main()
