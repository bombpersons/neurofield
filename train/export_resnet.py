"""The robust ResNet-50 (Salman et al. 2020, L2 eps 3; `train/dream_atlas.py robust-weights`)
for the dream renderer (dream-renderer/), up to layer4, with each BatchNorm folded into its
convolution, and the ImageNet normalisation its input wants:

    {name}.weight [cout, cin, k, k], {name}.bias [cout]
    names: conv1, layer{1..4}.{block}.conv{1,2,3}, layer{1..4}.0.downsample
    input.scale [3], input.offset [3]: the network sees rgb * scale + offset

    python train/export_resnet.py --out runs/dream/robust_resnet50.safetensors [--fixture FILE]

The fixture is for `cargo run -p dream-renderer --example check`, as export_googlenet.py's.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from safetensors.torch import load_file, save_file

ROBUST = Path.home() / ".cache/neurofield/robust_resnet50_l2_eps3.safetensors"
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def fold(conv, bn):
    s = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight.detach() * s[:, None, None, None]
    return w.contiguous(), (bn.bias.detach() - bn.running_mean * s).contiguous()


def weights(net):
    out = {"input.scale": (1 / STD).contiguous(), "input.offset": (-MEAN / STD).contiguous()}
    def put(name, conv, bn):  # noqa: E306
        out[f"{name}.weight"], out[f"{name}.bias"] = fold(conv, bn)
    put("conv1", net.conv1, net.bn1)
    for i in range(1, 5):
        for j, blk in enumerate(getattr(net, f"layer{i}")):
            n = f"layer{i}.{j}"
            put(f"{n}.conv1", blk.conv1, blk.bn1)
            put(f"{n}.conv2", blk.conv2, blk.bn2)
            put(f"{n}.conv3", blk.conv3, blk.bn3)
            if blk.downsample is not None:
                put(f"{n}.downsample", blk.downsample[0], blk.downsample[1])
    return out


def run(net, rgb, upto):
    """Each block's output for rgb [1, 3, H, W] (0..1), named as the renderer names them."""
    x = (rgb - MEAN.view(1, 3, 1, 1)) / STD.view(1, 3, 1, 1)
    acts = {}
    x = net.relu(net.bn1(net.conv1(x)))
    acts["conv1"] = x
    x = net.maxpool(x)
    acts["maxpool"] = x
    for i in range(1, 5):
        for j, blk in enumerate(getattr(net, f"layer{i}")):
            x = blk(x)
            acts[f"layer{i}.{j}"] = x
            if f"layer{i}.{j}" == upto:
                return acts
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture")
    ap.add_argument("--frame", default="runs/atlas_walk1/frames/00060.png")
    a = ap.parse_args()
    net = torchvision.models.resnet50()
    net.load_state_dict(load_file(ROBUST))
    net.eval()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(weights(net), a.out)
    print("wrote", a.out)
    if not a.fixture:
        return
    img = np.asarray(Image.open(a.frame).convert("RGB"), np.float32) / 255
    out = {"rgb": torch.tensor(img)}
    hwc = lambda t: t[0].permute(1, 2, 0).contiguous()  # noqa: E731
    for spec in (["layer3.5"], ["layer2.3", "layer3.5"]):
        rgb = torch.tensor(img).permute(2, 0, 1)[None].requires_grad_(True)
        acts = run(net, rgb, spec[-1])
        loss = sum((acts[l] ** 2).mean() / (acts[l] ** 2).mean().detach() for l in spec)
        loss.backward()
        out["grad_" + "+".join(spec)] = hwc(rgb.grad)
        if spec == ["layer3.5"]:
            for n in ("conv1", "maxpool", "layer1.0", "layer2.0", "layer3.5"):
                out["act_" + n] = hwc(acts[n].detach())
    save_file(out, a.fixture)
    print("wrote", a.fixture, {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    sys.exit(main())
