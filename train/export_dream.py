"""Export a dream net checkpoint (train_distil.py) for the engine (engine/src/neural/dream_net.rs).

    .venv/bin/python train/export_dream.py --ckpt runs/x/out/ckpt.pt --out runs/x/dream.safetensors

Writes the EMA weights' convolutions and, instead of CLIP, the tables the engine needs:
every phrase's vector (`phrase`, already through the net's projection), which phrases each
motif and shape may use (`kind_ids`, `kind_start`, `dream_ids`), and colour schemes
(`palettes`: dream.PALETTES, then draws of dream.random_palette). With --fixture FILE
(and --targets, a target shard) it also writes one crop's inputs and what PyTorch
makes of them, for `datagen dream-check` to compare the WGSL port against.
"""

import argparse

import numpy as np
import torch

from dream import BY_KIND, DREAM_IDS, MOTIFS, NOUNS, PALETTE, PALETTES, DreamNet, Q_PHRASE, fields, hex_rgb, random_palette
from export_weights import save_safetensors

HUE_SCHEMES = 24


def palettes():
    """[S, PALETTE, 3]: the fixed schemes, then draws of `random_palette`."""
    fixed = [np.array([hex_rgb(h) for h in p], np.float32) for p in PALETTES]
    rng, made = np.random.default_rng(0), []
    while len(made) < HUE_SCHEMES:
        r = random_palette(rng)
        if not any((r == f).all() for f in fixed):
            made.append(r)
    return torch.from_numpy(np.stack(fixed + made))


def load(path):
    state = torch.load(path, map_location="cpu")
    net = DreamNet(text_dim=state["text"].shape[1])
    net.load_state_dict(state["ema"])
    return net.eval(), state


@torch.no_grad()
def fixture(net, state, targets, path):
    """One held-out crop with judged parts: its inputs, fields and the net's layer."""
    d = np.load(targets)
    i = int(np.nonzero((d["judged"][:, :, 0] >= 0).any(1))[0][0])
    q, cam, palette = (torch.from_numpy(d[k][i : i + 1]) for k in ("q", "cam", "palette"))
    n = d["frame"].shape[1]
    feats, weights, _, fog = fields(q, cam, palette, n)
    text = state["text"][q[..., Q_PHRASE].long()]
    layer = net(feats, weights, text, palette)
    vec = net.phrase(text)
    emb = torch.einsum("bphw,bpe->behw", weights, vec)
    save_safetensors({"q": q[0], "cam": cam[0], "palette": palette[0], "feats": feats[0], "emb": emb[0], "fog": fog[0], "layer": layer[0]}, path, {})
    print(f"wrote {path} (item {i}, {n} px)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture")
    ap.add_argument("--targets")
    a = ap.parse_args()
    net, state = load(a.ckpt)
    tensors = {k: v for k, v in net.state_dict().items() if not k.startswith("phrase.")}
    with torch.no_grad():
        tensors["phrase"] = net.phrase(state["text"])
    kinds = [BY_KIND[m, s] for m in range(len(MOTIFS)) for s in range(len(NOUNS))]
    tensors["kind_ids"] = torch.tensor([i for k in kinds for i in k]).float()
    tensors["kind_start"] = torch.tensor(np.cumsum([0] + [len(k) for k in kinds])).float()
    tensors["dream_ids"] = torch.tensor([DREAM_IDS[s] for s in range(len(NOUNS))]).float()
    tensors["palettes"] = palettes()
    assert tensors["palettes"].shape[1] == PALETTE
    save_safetensors(tensors, a.out, {"step": str(state["step"])})
    print(f"wrote {a.out} (step {state['step']}, {len(tensors)} tensors)")
    if a.fixture:
        fixture(net, state, a.targets, a.fixture)


if __name__ == "__main__":
    main()
