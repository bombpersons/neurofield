"""A fixture for checking the game's port of a sculpt-trained model (engine/src/neural/
gpu_net.rs and gbuffer.rs) against PyTorch: one validation view's raw inputs, and what the
model makes of them at each step, as raw little-endian f32 files with their shapes in
fixture.json.

    .venv/bin/python train/stage_fixture.py --ckpt runs/vibe1/ckpt.pt --shard runs/sculpt_val/shard_0007 --view 100 --out runs/stage_fixture
    cargo run --release -p training-data-gen -- stage-check --model runs/vibe1/model.safetensors --fixture runs/stage_fixture
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import Shards, collate
from model import load
from pixels import inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m, _ = load(a.ckpt)
    val = Shards(a.shard, pixels=True)
    s = val.shards[0]
    v = s["views"][a.view]
    scene = int(v[0])
    lo, hi = s["offsets"][scene], s["offsets"][scene + 1]
    item = val[a.view]
    b = collate([item])
    feats, tex, pad, cams, images = b[:5]
    organs = b[6]
    plain, coords = inputs(b[7], "cpu")
    width = images.shape[3]
    head = m(feats, tex, pad, cams, width=width, organs=organs)
    full = m(feats, tex, pad, cams, width=width, organs=organs, pixels=(plain, coords))
    shapes = {}

    def put(name, x):
        x = np.ascontiguousarray(np.asarray(x, np.float32))
        x.tofile(out / f"{name}.bin")
        shapes[name] = list(x.shape)

    put("tris", s["tris"][lo:hi])
    put("turning", s["turning"][lo:hi])
    put("organs", s["organs"][s["organ_offsets"][a.view] : s["organ_offsets"][a.view + 1]])
    put("view", v)
    put("amount", [s["amounts"][a.view]])
    put("feats", feats[0])
    put("tex", tex[0])
    put("cam", cams[0])
    real = ~organs[2][0]  # collate pads a view with none to one
    put("organ_feats", organs[0][0][real])
    put("organ_kinds", organs[1][0][real])
    put("plain", plain[0])
    put("head", head[0].permute(1, 2, 0))
    put("full", full[0].permute(1, 2, 0))
    put("target", images[0].permute(1, 2, 0))
    json.dump(shapes, open(out / "fixture.json", "w"))
    print(f"view {a.view}: {width} wide, {feats.shape[1]} triangles, {organs[0].shape[1]} organs -> {out}")


if __name__ == "__main__":
    main()
