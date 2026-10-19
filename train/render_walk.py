"""Render a `datagen walk` sequence with a trained model.

    python train/render_walk.py --walk data/walk0 --ckpt runs/x/ckpt.pt --out runs/x/walk0

Writes side-by-side frames (rasteriser | network) and an animated GIF.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import Shards, collate
from model import Renderer


@torch.no_grad()
def render_walk(model, walk_dir, out_dir, device, batch=16, every=1, gif_scale=1):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = Shards(walk_dir)
    model.eval()
    frames = []
    idx = list(range(0, len(ds), every))
    for i in range(0, len(idx), batch):
        items = [ds[j] for j in idx[i : i + batch]]
        feats, tex, pad, cams, images = (x.to(device) for x in collate(items))
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            pred = model(feats, tex, pad, cams).float().clamp(0, 1)
        pair = torch.cat([images, pred], 3).permute(0, 2, 3, 1).cpu().numpy()
        for k, f in enumerate(pair):
            im = Image.fromarray((f * 255).astype(np.uint8))
            im.save(out / f"{idx[i + k]:05d}.png")
            frames.append(im.resize((im.width // gif_scale, im.height // gif_scale)) if gif_scale > 1 else im)
    frames[0].save(out / "walk.gif", save_all=True, append_images=frames[1:], duration=1000 // 30 * every, loop=0)
    return out / "walk.gif"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(a.ckpt, map_location=device)
    m = Renderer(**state.get("cfg", {})).to(device)
    m.load_state_dict(state["model"])
    print(render_walk(m, a.walk, a.out, device, every=a.every))
