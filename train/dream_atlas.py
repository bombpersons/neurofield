"""Prototype of the dream atlas (docs/dream_atlas.md): DeepDream that grows on the walls.

Every face of the level (one tile of a wall, a floor or a ceiling) gets a page of its own in
an atlas of RGB residuals laid over the PS1 picture. Each frame of a walk (`datagen walk`),
GoogLeNet looks at the picture and the gradient of its `inception4c` activations is written
back through the atlas lookup into the texels on screen, so the dream deepens frame by frame
and stays on the walls as the camera moves. Two octaves: the PS1 picture's 240 lines every
frame, and three times that (720) on a quarter of the screen a frame, in turn.

Frames are written both ways the game could show them:

    sharp/   the atlas always at 720 lines, the PS1 picture (and a stand-in PS1 sword) at 240
    settle/  the atlas at 240 lines while the camera moves, sharpening to 720 over a second
             once it stands still

    python train/dream_atlas.py --walk runs/atlas_walk1 --out runs/atlas_proto1 [--layer googlenet:4c]

The robust ResNet-50 (`--layer robust50:l3`) wants its weights converted once:

    python train/dream_atlas.py robust-weights resnet50_l2_eps3.ckpt

The walk holds still now and then (`--hold`) so the settling and the deepening can be seen.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbuffer import gbuffer  # noqa: E402

H = 240
UP = 3  # the fine octave and the sharp display: 720 lines
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


GOOGLENET = ["conv1", "maxpool1", "conv2", "conv3", "maxpool2", "inception3a", "inception3b", "maxpool3",
             "inception4a", "inception4b", "inception4c", "inception4d", "inception4e", "maxpool4",
             "inception5a", "inception5b"]
RESNET = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
# Robust ResNet-50 (Salman et al. 2020, "Do Adversarially Robust ImageNet Models Transfer
# Better?", L2 eps 3), its torchvision-shaped weights pulled out of the checkpoint by
# `robust_weights` into a safetensors file.
ROBUST = Path.home() / ".cache/neurofield/robust_resnet50_l2_eps3.safetensors"


def robust_weights(ckpt):
    """Pull the classifier out of a robustness-library checkpoint (madrylab/robust-imagenet-models
    on Hugging Face) into `ROBUST`. Loaded with weights_only, so the pickle can hold nothing
    but tensors and plain values; the attacker's copy and the normaliser are left behind."""
    from safetensors.torch import save_file
    s = torch.load(ckpt, map_location="cpu", weights_only=True)
    s = s.get("model", s.get("state_dict", s))
    out = {k[len("module.model."):]: v.contiguous() for k, v in s.items() if k.startswith("module.model.")}
    net = torchvision.models.resnet50()
    net.load_state_dict(out)
    ROBUST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, ROBUST)
    print(f"wrote {ROBUST} ({len(out)} tensors)")


class Dreamer:
    """A network up to its deepest named layer; `loss(rgb)` is how excited the named layers
    are. `spec` is `net:layer+layer`, net `googlenet` (layers `inception4c`, ..., or just
    `4c`) or `robust50` (`layer1`..`layer4`, or `l3`). Each layer counts alike: its mean
    square over its own (detached) value, so a mix isn't ruled by whichever is loudest."""

    def __init__(self, spec):
        net, _, layers = spec.partition(":")
        if net == "googlenet":
            self.net = torchvision.models.googlenet(weights="IMAGENET1K_V1").eval()
            order = GOOGLENET
            names = [n if n.startswith("inception") else "inception" + n for n in layers.split("+")]
            self.prep = lambda x: self.net._transform_input(x)  # noqa: E731
        elif net == "robust50":
            from safetensors.torch import load_file
            self.net = torchvision.models.resnet50()
            self.net.load_state_dict(load_file(ROBUST))
            self.net.eval()
            order = RESNET
            names = [n if n.startswith("layer") else "layer" + n.lstrip("l") for n in layers.split("+")]
            self.prep = lambda x: x  # noqa: E731
        else:
            raise SystemExit(f"unknown network {net!r} (googlenet or robust50)")
        for p in self.net.parameters():
            p.requires_grad_(False)
        for n in names:
            assert n in order, f"{spec}: no layer {n} (one of {order})"
        self.names = set(names)
        self.blocks = [(n, getattr(self.net, n)) for n in order[: max(order.index(n) for n in names) + 1]]

    def loss(self, rgb):
        x = self.prep((rgb - MEAN) / STD)
        total = 0.0
        for n, b in self.blocks:
            x = b(x)
            if n in self.names:
                sq = x ** 2
                total = total + sq.mean() / sq.mean().detach().clamp(min=1e-12)
        return total


class Atlas:
    """Pages of RES x RES RGB residuals, one per face, made as faces are first seen."""

    def __init__(self, res):
        self.res = res
        self.keys = {}
        self.tex = torch.zeros(0, res, res, 3)

    def pages(self, keys):
        """Page numbers for face keys [N, 5] (int), making pages for new ones."""
        out = np.empty(len(keys), np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        ids = []
        for k in map(tuple, uniq):
            if k not in self.keys:
                self.keys[k] = len(self.keys)
            ids.append(self.keys[k])
        out[:] = np.asarray(ids, np.int64)[inv.reshape(-1)]
        grow = len(self.keys) - len(self.tex)
        if grow > 0:
            self.tex = torch.cat([self.tex, torch.zeros(grow, self.res, self.res, 3)])
        return torch.as_tensor(out)

    def sample(self, tex, page, uv):
        """Bilinear lookups of `tex` (the atlas, maybe with grad) at pages [N] and uv [N, 2] in
        0..1, clamped inside each page. Returns [N, 3]."""
        r = self.res
        t = (uv * r - 0.5).clamp(0, r - 1)
        i0 = t.floor().long().clamp(max=r - 2)
        f = t - i0
        flat = tex.view(-1, 3)
        base = page * r * r
        def at(dx, dy):  # noqa: E306
            return flat[base + (i0[:, 1] + dy) * r + i0[:, 0] + dx]
        fx, fy = f[:, :1], f[:, 1:]
        return (at(0, 0) * (1 - fx) * (1 - fy) + at(1, 0) * fx * (1 - fy)
                + at(0, 1) * (1 - fx) * fy + at(1, 1) * fx * fy)


def faces(pos, normal):
    """Each surface point's face (key [N, 5]: axis, side, and the tile it lies in) and where on
    the face's page it is (uv [N, 2], 0..1). Faces are tiles of the world's planes, by the
    normal's main axis, so sloped floors get pages too."""
    axis = normal.abs().argmax(1)
    side = (normal.gather(1, axis[:, None])[:, 0] > 0).long()
    inside = pos - normal * 0.01
    cell = inside.floor().long()
    others = torch.tensor([[1, 2], [0, 2], [0, 1]])[axis]
    uv = (pos.gather(1, others) - pos.gather(1, others).floor()).clamp(0, 1 - 1e-6)
    key = torch.cat([axis[:, None], side[:, None], cell], 1)
    return key.numpy(), uv


def surface(tris, v, scale):
    """Where each pixel of the view at `scale` x 240 lines lands: (covered mask [h, w], world
    position [h, w, 3], normal [h, w, 3], distance [h, w]). The 240-line G-buffer says which
    triangle; finer pixels meet that triangle's plane along their own ray."""
    g = gbuffer(tris, v)
    tri = g["tri"]
    width = int(v[26])
    corners = torch.as_tensor(tris[:, :9], dtype=torch.float32).view(-1, 3, 3)
    n = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)
    if scale > 1:
        tri = tri.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
    h, w = tri.shape
    covered = tri >= 0
    t = tri.clamp(min=0)
    pos0, R, hp = torch.as_tensor(v[1:4]), torch.as_tensor(v[4:13]).view(3, 3), float(v[13])
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    d_view = torch.stack([(xs / scale - width / 2) / hp, (ys / scale - 120) / hp, torch.ones_like(xs)], -1)
    d = d_view @ R  # view = R (p - pos), so world direction = R^T view
    nn_ = n[t]
    p0 = corners[t, 0]
    denom = (d * nn_).sum(-1)
    s = ((p0 - pos0) * nn_).sum(-1) / torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    pos = pos0 + d * s[..., None]
    dist = (pos - pos0).norm(dim=-1)
    return covered, pos, nn_, dist


def sword(width):
    """A stand-in PS1 sword at 240 lines: (rgb [240, w, 3], alpha [240, w]) to scale up blocky."""
    im = Image.new("RGBA", (width, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = width - 70
    d.polygon([(x + 18, 120), (x + 26, 116), (x + 44, 210), (x + 36, 214)], fill=(190, 196, 206, 255))
    d.polygon([(x + 22, 120), (x + 26, 116), (x + 44, 210), (x + 40, 212)], fill=(120, 126, 140, 255))
    d.polygon([(x + 22, 206), (x + 58, 196), (x + 60, 204), (x + 24, 214)], fill=(178, 142, 54, 255))
    d.polygon([(x + 38, 210), (x + 46, 208), (x + 54, 240), (x + 44, 240)], fill=(96, 62, 34, 255))
    a = np.asarray(im, np.float32) / 255
    return torch.as_tensor(a[..., :3]), torch.as_tensor(a[..., 3])


def compose(atlas, tex, ps1, cov, page, uv, fog):
    """The PS1 picture with the atlas laid over where faces show (`page`, `uv`, `fog` are per
    covered pixel): [h, w, 3]."""
    out = ps1.clone().reshape(-1, 3)
    m = cov.reshape(-1)
    res = atlas.sample(tex, page, uv) * (1 - fog)[:, None]
    out = out.index_put((m.nonzero()[:, 0],), out[m] + res)
    return out.view(*ps1.shape).clamp(0, 1)


def dream_step(atlas, dreamer, ps1, cov, page, uv, fog, lr, jitter):
    """One DeepDream step on a picture, written into the atlas texels it shows: the image
    gradient goes back through the lookups, is spread over each page a little (so pages
    seen from afar don't get only every few texels) and normalised as DeepDream does."""
    tex = atlas.tex.clone().requires_grad_(True)
    img = compose(atlas, tex, ps1, cov, page, uv, fog)
    x = img.permute(2, 0, 1)[None]
    dx, dy = np.random.randint(-jitter, jitter + 1, 2)
    x = torch.roll(x, (int(dy), int(dx)), (2, 3))
    dreamer.loss(x).backward()
    g = tex.grad
    # How much each texel is seen: the same lookups' weights.
    ones = torch.zeros_like(atlas.tex, requires_grad=True)
    atlas.sample(ones, page, uv).sum().backward()
    hits = ones.grad[..., :1]
    used = torch.unique(page)
    k = torch.tensor([1.0, 4, 6, 4, 1])
    k = (k[:, None] * k[None]) / 256
    k = k.view(1, 1, 5, 5).repeat(4, 1, 1, 1)
    gh = torch.cat([g[used], hits[used]], -1).permute(0, 3, 1, 2)
    gh = F.conv2d(F.pad(gh, (2, 2, 2, 2), mode="replicate"), k, groups=4).permute(0, 2, 3, 1)
    upd = gh[..., :3] / (gh[..., 3:] + 1e-3)
    seen = gh[..., 3:] > 1e-3
    scale = upd[seen.expand_as(upd)].abs().mean().clamp(min=1e-12)
    atlas.tex[used] += lr * upd / scale * seen
    return float(scale)


def main():
    if sys.argv[1:2] == ["robust-weights"]:
        return robust_weights(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0, help="stop after this many frames (0: all)")
    ap.add_argument("--hold", type=int, default=45, help="frames to stand still every --every frames")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--layer", default="googlenet:4c", help="what the coarse octave dreams with (see Dreamer)")
    ap.add_argument("--layer-fine", default="", help="what the fine octave dreams with (default: --layer)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--res", type=int, default=256, help="atlas texels along a tile")
    ap.add_argument("--lr", type=float, default=0.012, help="coarse octave step, in colour")
    ap.add_argument("--lr-fine", type=float, default=0.008)
    ap.add_argument("--decay", type=float, default=0.002, help="share of the residual let go a frame")
    ap.add_argument("--limit", type=float, default=0.6, help="largest residual")
    ap.add_argument("--settle", type=float, default=30, help="frames to sharpen once still")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(a.threads)

    walk, out = Path(a.walk), Path(a.out)
    for d in ("sharp", "settle"):
        (out / d).mkdir(parents=True, exist_ok=True)
    tris = np.load(walk / "tris.npy")
    views = np.load(walk / "views.npy")
    width = int(views[0, 26])
    # The schedule: walk frames, standing still now and then.
    order = []
    for i in range(len(views)):
        order.append(i)
        if a.hold and i % a.every == a.every - 1:
            order += [i] * a.hold
    if a.frames:
        order = order[: a.frames]

    dreamer, atlas = Dreamer(a.layer), Atlas(a.res)
    fine = Dreamer(a.layer_fine) if a.layer_fine and a.layer_fine != a.layer else dreamer
    sw_rgb, sw_a = sword(width)
    up = lambda t: t.repeat_interleave(UP, 0).repeat_interleave(UP, 1)  # noqa: E731
    sw_rgb_hi, sw_a_hi = up(sw_rgb), up(sw_a)[..., None]
    sharpness, last_v, cache = 0.0, None, {}
    t0 = time.time()
    for n, i in enumerate(order):
        v = views[i]
        ps1 = torch.as_tensor(np.asarray(Image.open(walk / f"frames/{i:05}.png").convert("RGB"), np.float32) / 255)
        if i not in cache:
            cache.clear()
            octs = {}
            for s in (1, UP):
                cov, pos, nrm, dist = surface(tris, v, s)
                key, uv = faces(pos[cov], nrm[cov])
                fog = ((dist - v[14]) / max(v[15] - v[14], 1e-6)).clamp(0, 1)
                octs[s] = (cov, atlas.pages(key), uv, fog[cov])
            cache[i] = octs
        octs = cache[i]
        cov, page, uv, fog = octs[1]
        dream_step(atlas, dreamer, ps1, cov, page, uv, fog, a.lr, 4)
        # The fine octave on a quarter of the 720-line picture, in turn.
        cov, page, uv, fog = octs[UP]
        ps1_hi = up(ps1)
        hh, ww = cov.shape
        q = n % 4
        y0, x0 = (q // 2) * hh // 2, (q % 2) * ww // 2
        y0 = int(np.clip(y0 + np.random.randint(-hh // 8, hh // 8 + 1), 0, hh - hh // 2))
        x0 = int(np.clip(x0 + np.random.randint(-ww // 8, ww // 8 + 1), 0, ww - ww // 2))
        crop = (slice(y0, y0 + hh // 2), slice(x0, x0 + ww // 2))
        flat_idx = torch.arange(hh * ww).view(hh, ww)[crop][cov[crop]]
        sel = torch.zeros(hh * ww, dtype=torch.long)
        sel[cov.reshape(-1).nonzero()[:, 0]] = torch.arange(int(cov.sum()))
        sub = sel[flat_idx]
        dream_step(atlas, fine, ps1_hi[crop].contiguous(), cov[crop], page[sub], uv[sub], fog[sub], a.lr_fine, 6)
        atlas.tex.mul_(1 - a.decay).clamp_(-a.limit, a.limit)

        with torch.no_grad():
            lo = compose(atlas, atlas.tex, ps1, *octs[1])
            hi = compose(atlas, atlas.tex, ps1_hi, *octs[UP])
            moving = last_v is not None and not np.allclose(v[1:13], last_v[1:13])
            sharpness = 0.0 if moving else min(1.0, sharpness + 1.0 / a.settle)
            settle = up(lo) * (1 - sharpness) + hi * sharpness
            for name, img in (("sharp", hi), ("settle", settle)):
                img = img * (1 - sw_a_hi) + sw_rgb_hi * sw_a_hi
                Image.fromarray((img.numpy() * 255).round().astype(np.uint8)).save(out / name / f"{n:05}.png")
        last_v = v
        if n % 20 == 0:
            el = time.time() - t0
            print(f"frame {n}/{len(order)} (walk {i}): {len(atlas.keys)} pages, |atlas| {atlas.tex.abs().mean():.4f}, "
                  f"{el / (n + 1):.2f} s a frame", flush=True)

    for name in ("sharp", "settle"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(out / name / "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / f"{name}.mp4")], check=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    print("done:", out)


if __name__ == "__main__":
    main()
