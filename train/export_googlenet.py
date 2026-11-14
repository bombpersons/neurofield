"""GoogLeNet (torchvision IMAGENET1K_V1) for the dream renderer (dream-renderer/), up to
inception4e, with each BasicConv2d's BatchNorm folded into its convolution:

    {name}.weight [cout, cin, k, k], {name}.bias [cout]
    names: conv1, conv2, conv3, inception{3a..4e}.branch{1, 2.0, 2.1, 3.0, 3.1, 4.1}

    python train/export_googlenet.py --out runs/dream/googlenet.safetensors [--fixture FILE]

The fixture is for `cargo run -p dream-renderer --example check`: a frame of a walk, its
gradient for a few objectives (each layer's mean square over its own detached value, summed)
and some activations, all HWC.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from safetensors.torch import save_file

BLOCKS = ["inception3a", "inception3b", "inception4a", "inception4b", "inception4c", "inception4d", "inception4e"]


def fold(conv_bn):
    w = conv_bn.conv.weight.detach()
    bn = conv_bn.bn
    s = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    return (w * s[:, None, None, None]).contiguous(), (bn.bias.detach() - bn.running_mean * s).contiguous()


def weights(net):
    out = {}
    def put(name, m):  # noqa: E306
        out[f"{name}.weight"], out[f"{name}.bias"] = fold(m)
    for n in ("conv1", "conv2", "conv3"):
        put(n, getattr(net, n))
    for b in BLOCKS:
        blk = getattr(net, b)
        put(f"{b}.branch1", blk.branch1)
        put(f"{b}.branch2.0", blk.branch2[0])
        put(f"{b}.branch2.1", blk.branch2[1])
        put(f"{b}.branch3.0", blk.branch3[0])
        put(f"{b}.branch3.1", blk.branch3[1])
        put(f"{b}.branch4.1", blk.branch4[1])
    return out


ORDER = ["conv1", "maxpool1", "conv2", "conv3", "maxpool2", "inception3a", "inception3b", "maxpool3",
         "inception4a", "inception4b", "inception4c", "inception4d", "inception4e"]


def run(net, rgb, layers):
    """Activations of every layer in ORDER for rgb [1, 3, H, W] (0..1), as the renderer feeds
    it: torchvision's transform_input of the ImageNet-normalised picture is just 2 rgb - 1."""
    x = 2 * rgb - 1
    acts = {}
    for n in ORDER[: max(ORDER.index(l) for l in layers) + 1]:
        x = getattr(net, n)(x)
        acts[n] = x
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture")
    ap.add_argument("--frame", default="runs/atlas_walk1/frames/00060.png")
    a = ap.parse_args()
    net = torchvision.models.googlenet(weights="IMAGENET1K_V1").eval()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(weights(net), a.out)
    print("wrote", a.out)
    if not a.fixture:
        return
    img = np.asarray(Image.open(a.frame).convert("RGB"), np.float32) / 255
    out = {"rgb": torch.tensor(img)}
    hwc = lambda t: t[0].permute(1, 2, 0).contiguous()  # noqa: E731
    for spec in (["inception4c"], ["inception3b", "inception4d"]):
        rgb = torch.tensor(img).permute(2, 0, 1)[None].requires_grad_(True)
        acts = run(net, rgb, spec)
        loss = sum((acts[l] ** 2).mean() / (acts[l] ** 2).mean().detach() for l in spec)
        loss.backward()
        out["grad_" + "+".join(spec)] = hwc(rgb.grad)
        if spec == ["inception4c"]:
            for n in ("conv1", "maxpool1", "conv3", "maxpool2", "inception3a", "inception4c"):
                out["act_" + n] = hwc(acts[n].detach())
    save_file(out, a.fixture)
    print("wrote", a.fixture, {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    sys.exit(main())
