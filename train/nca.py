"""A neural cellular automaton that grows one of the painted flesh textures (data/painted/7,
colour and height) as a living texture: every cell of a wrapping grid looks at itself and its
eight neighbours and nudges its state, over and over, so the texture forms out of noise and
keeps moving. After Niklasson, Mordvintsev, Randazzo and Levin, "Self-Organising Textures",
Distill 2021: trained by how alike the texture looks (Gram matrices of VGG16 features), not
pixel for pixel.

The game steps it on the GPU (engine/src/render/nca.rs, which must agree with `step` here):

    perceive: for each of the CHANNELS channels, the cell itself, Sobel x / 8, Sobel y / 8 and
              the Laplacian / 16 of its 3x3 neighbourhood, wrapping; laid out filter-major
              ([identity of every channel, then Sobel x of every channel, ...], 4 * CHANNELS)
    update:   dx = w2 @ relu(w1 @ perceived + b1), added to the cell where a coin (FIRE) says so

Channels 0-2 are the colour (+ 0.5), 3 the height (+ 0.5), the rest hidden.

    python train/nca.py --slot 1 --steps 4000 --out runs/nca1   (a CUDA card or patience)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CHANNELS = 12
HIDDEN = 96
FIRE = 0.5

IDENTITY = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
LAPLACIAN = [[1, 2, 1], [2, -12, 2], [1, 2, 1]]


class Nca(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([IDENTITY, SOBEL_X, np.transpose(SOBEL_X).tolist(), LAPLACIAN], dtype=torch.float32)
        k = k / torch.tensor([1.0, 8.0, 8.0, 16.0])[:, None, None]
        self.register_buffer("filters", k)
        self.w1 = nn.Conv2d(4 * CHANNELS, HIDDEN, 1)
        self.w2 = nn.Conv2d(HIDDEN, CHANNELS, 1, bias=False)
        nn.init.zeros_(self.w2.weight)

    def perceive(self, x):
        c = x.shape[1]
        y = F.pad(x, (1, 1, 1, 1), mode="circular")
        # [b, 4c] filter-major: every channel through filter 0, then filter 1, ...
        k = self.filters.repeat_interleave(c, 0)[:, None]  # [4c, 1, 3, 3] in filter-major order
        return F.conv2d(y.repeat(1, 4, 1, 1), k, groups=4 * c)

    def step(self, x, fire=FIRE):
        dx = self.w2(torch.relu(self.w1(self.perceive(x))))
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < fire).float()
        return x + dx * mask


def seed(n, size, device):
    """Fresh grids: a little noise, so that the cells differ from the first step."""
    return (torch.rand(n, CHANNELS, size, size, device=device) - 0.5) * 0.1


class Style(nn.Module):
    """Gram matrices of VGG16's first layers of each block, for a [b, 3, h, w] image in 0..1."""

    LAYERS = (1, 6, 11, 18, 25)

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(self.LAYERS) + 1].eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.LAYERS:
                b, c, h, w = x.shape
                f = x.reshape(b, c, h * w)
                out.append(f @ f.transpose(1, 2) / (h * w))
        return out


def style_loss(grams, target):
    return sum(((g - t) ** 2).mean() for g, t in zip(grams, target))


def colour(x):
    return (x[:, :3] + 0.5).clamp(0, 1)


def height(x):
    return (x[:, 3:4] + 0.5).clamp(0, 1).expand(-1, 3, -1, -1)


def train(slot, steps, out, painted="data/painted/7", size=128, batch=4, pool_size=256, height_weight=0.5, device=None, log=print, on_save=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda f, mode: torch.from_numpy(np.asarray(Image.open(f).convert(mode), np.float32) / 255)  # noqa: E731
    rgb = load(f"{painted}/{slot}.png", "RGB").permute(2, 0, 1)[None].to(device)
    h = load(f"{painted}/{slot}_h.png", "L")[None, None].expand(-1, 3, -1, -1).to(device)
    style = Style().to(device)
    with torch.no_grad():
        want_rgb, want_h = style(rgb), style(h)
    nca = Nca().to(device)
    opt = torch.optim.Adam(nca.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [steps // 2, steps * 3 // 4], 0.3)
    pool = seed(pool_size, size, device)
    start = time.time()
    for it in range(steps):
        idx = torch.randint(0, pool_size, (batch,), device=device)
        x = pool[idx]
        x[:1] = seed(1, size, device)  # one fresh each batch: it must grow from noise too
        for _ in range(int(np.random.randint(32, 96))):
            x = nca.step(x)
        loss = style_loss(style(colour(x)), want_rgb) + height_weight * style_loss(style(height(x)), want_h)
        # Keep the state in range.
        loss = loss + (x - x.clamp(-1.5, 1.5)).abs().mean()
        opt.zero_grad()
        loss.backward()
        for p in nca.parameters():  # normalised gradients, as the article does
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()
        with torch.no_grad():
            pool[idx] = x.detach()
        if it % 100 == 0 or it == steps - 1:
            log(json.dumps({"it": it, "loss": round(float(loss), 4), "secs": round(time.time() - start)}))
        if it % 1000 == 0 or it == steps - 1:
            save(nca, out, slot)
            sheet(nca, out / f"sample_{it:05d}.png", device)
            if on_save:
                on_save()
    return nca


def save(nca, out, slot):
    """The weights as the game reads them (raw f32, little-endian) and a checkpoint."""
    torch.save(nca.state_dict(), out / "nca.pt")
    w1 = nca.w1.weight.detach().cpu().reshape(HIDDEN, 4 * CHANNELS).numpy()
    b1 = nca.w1.bias.detach().cpu().numpy()
    w2 = nca.w2.weight.detach().cpu().reshape(CHANNELS, HIDDEN).numpy()
    np.concatenate([w1.ravel(), b1, w2.ravel()]).astype("<f4").tofile(out / "nca.bin")
    json.dump({"slot": slot, "channels": CHANNELS, "hidden": HIDDEN, "fire": FIRE, "layout": "w1 [hidden, 4*channels], b1 [hidden], w2 [channels, hidden]"},
              open(out / "nca.json", "w"))


@torch.no_grad()
def sheet(nca, path, device, size=256, at=(0, 32, 64, 128, 256, 512, 1024)):
    """The texture growing from noise on a 256 grid (the game's), colour over height, at the
    steps `at`."""
    x = seed(1, size, device)
    cols, done = [], 0
    for n in at:
        for _ in range(n - done):
            x = nca.step(x)
        done = n
        cols.append(torch.cat([colour(x)[0], height(x)[0]], 1))
    img = torch.cat(cols, 2).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--painted", default="data/painted/7")
    a = ap.parse_args()
    train(a.slot, a.steps, a.out, a.painted)
