"""Export a checkpoint to ONNX for the Rust viewer.

    .venv/bin/python train/export_onnx.py --ckpt runs/colab2/ckpt.pt --out runs/colab2/model.onnx

Inputs: feats [1, N, TRI_FEATURES] float32, tex [1, N] int64, pad [1, N] bool
(true = padding), cams [1, CAM_FEATURES] float32; N may vary per frame.
Output: image [1, 3, H, W] float32, roughly 0..1. Checks the export against
PyTorch on a view from --walk if given.
"""

import argparse

import numpy as np
import torch

from data import CAM_FEATURES, TRI_FEATURES, Shards, collate
from model import Renderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--walk", default=None)
    a = ap.parse_args()
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = Renderer(**state.get("cfg", {}))
    m.load_state_dict(state["model"])
    m.eval()

    n = 64
    example = (
        torch.randn(1, n, TRI_FEATURES),
        torch.zeros(1, n, dtype=torch.long),
        torch.zeros(1, n, dtype=torch.bool),
        torch.randn(1, CAM_FEATURES),
    )
    tris = torch.export.Dim("tris", min=1, max=4096)
    torch.onnx.export(
        m, example, a.out, input_names=["feats", "tex", "pad", "cams"], output_names=["image"],
        dynamic_shapes={"feats": {1: tris}, "tex": {1: tris}, "pad": {1: tris}, "cams": None},
        dynamo=True, external_data=False,
    )
    print(f"wrote {a.out} (step {state['step']})")

    if a.walk:
        import onnxruntime as ort

        feats, tex, pad, cams, _ = collate([Shards(a.walk)[0]])
        with torch.no_grad():
            want = m(feats, tex, pad, cams).numpy()
        sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"feats": feats.numpy(), "tex": tex.numpy(), "pad": pad.numpy(), "cams": cams.numpy()})[0]
        print(f"{feats.shape[1]} tris, max |onnx - torch| = {np.abs(got - want).max():.2e}")


if __name__ == "__main__":
    main()
