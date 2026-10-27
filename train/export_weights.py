"""Export a checkpoint's weights for the neural renderer (engine/src/neural/gpu_net.rs).

    .venv/bin/python train/export_weights.py --ckpt runs/x/ckpt.pt --out runs/x/model.safetensors

Writes the averaged weights when the checkpoint has them, plus the fixed buffer
the forward pass needs (dream noise directions), as float32 safetensors with the
model config in the metadata. Patch centres depend on the screen's width, so
the renderer works them out (model.centres).
"""

import argparse
import json
import struct

import torch

from model import load


def save_safetensors(tensors, path, metadata):
    """The safetensors layout, written by hand: u64 header length, JSON header, raw data."""
    header, blobs, offset = {"__metadata__": metadata}, [], 0
    for name, t in tensors.items():
        data = t.detach().to(torch.float32).contiguous().numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m, state = load(a.ckpt)
    tensors = dict(m.state_dict())
    tensors["noise_dirs"] = m.noise_dirs
    cfg = {"d": m.tex.weight.shape[1], "heads": m.encoder[0].attn.num_heads,
           "enc_layers": len(m.encoder), "dec_layers": len(m.decoder), "bands": m.bands, "low": m.low}
    save_safetensors(tensors, a.out, {"cfg": json.dumps(cfg), "step": str(state["step"])})
    print(f"wrote {a.out} (step {state['step']}, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
