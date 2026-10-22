"""Export a checkpoint to ONNX for the Rust viewer.

    .venv/bin/python train/export_onnx.py --ckpt runs/colab2/ckpt.pt --out runs/colab2/model.onnx

Inputs: feats [1, N, TRI_FEATURES] float32, tex [1, N] int64, pad [1, N] bool
(true = padding), cams [1, CAM_FEATURES] float32; N may vary per frame. Then the
viewer's dream controls (see Renderer.forward): tex_mix [NUM_TEX, NUM_TEX], temp [1],
q_noise [patches, DREAM_DIRS], all float32; identity, 1 and zeros render normally.
Output: image [1, 3, H, W] float32, roughly 0..1. Checks the export against
PyTorch on a view from --walk if given.
"""

import argparse

import numpy as np
import torch

from data import CAM_FEATURES, NUM_TEX, TRI_FEATURES, Shards, collate
from model import DREAM_DIRS, load


def calm(m):
    """Dream controls that leave the model's output as it is."""
    return torch.eye(NUM_TEX), torch.ones(1), torch.zeros(len(m.centres), DREAM_DIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--walk", default=None)
    a = ap.parse_args()
    m, state = load(a.ckpt)

    n = 64
    example = (
        torch.randn(1, n, TRI_FEATURES),
        torch.zeros(1, n, dtype=torch.long),
        torch.zeros(1, n, dtype=torch.bool),
        torch.randn(1, CAM_FEATURES),
        *calm(m),
    )
    tris = torch.export.Dim("tris", min=1, max=4096)
    torch.onnx.export(
        m, example, a.out, input_names=["feats", "tex", "pad", "cams", "tex_mix", "temp", "q_noise"], output_names=["image"],
        dynamic_shapes={"feats": {1: tris}, "tex": {1: tris}, "pad": {1: tris}, "cams": None, "tex_mix": None, "temp": None, "q_noise": None},
        dynamo=True, external_data=False,
    )
    print(f"wrote {a.out} (step {state['step']})")

    if a.walk:
        import onnxruntime as ort

        feats, tex, pad, cams, *_ = collate([Shards(a.walk)[0]])
        with torch.no_grad():
            want = m(feats, tex, pad, cams).numpy()
        sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        mix, temp, noise = calm(m)
        inputs = {"feats": feats, "tex": tex, "pad": pad, "cams": cams, "tex_mix": mix, "temp": temp, "q_noise": noise}
        got = sess.run(None, {k: v.numpy() for k, v in inputs.items()})[0]
        print(f"{feats.shape[1]} tris, max |onnx - torch| = {np.abs(got - want).max():.2e}")


if __name__ == "__main__":
    main()
