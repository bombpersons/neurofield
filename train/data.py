"""Dataset over datagen shards: one item = (triangle tokens, camera token, image,
creature part tokens, creature mask).

Triangles are moved into camera space and culled the cheap ways a renderer
would before rasterising (behind the camera, beyond the fog, outside the
frustum, back-facing). Occlusion, perspective-correct-ish coverage, texturing,
fog and torch light are left for the network to learn.

Creatures arrive as one token per body part (pose, size, shape dials, skin
motif and density, tint, baked light), never as triangles: the dense meshes
and skin layouts behind the images are for the network to make up. Shards
from before creatures had parts load with none.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

W, H = 320, 240
MAX_TRIS = 512
NUM_TEX = 12
# Per-triangle continuous features (see `tri_tokens`).
TRI_FEATURES = 9 + 6 + 3 + 3 + 6 + 18
CAM_FEATURES = 1 + 2 + 3 + 1
MAX_PARTS = 128
NUM_SHAPES = 4
NUM_MOTIFS = 11
# Per-part continuous features (see `part_tokens`).
PART_FEATURES = 3 + 3 + 9 + 3 + 4 + 2 + 4 + 2 + 3 + 12


class Shards(Dataset):
    def __init__(self, root, shards=None):
        """`root` holds shard_* directories, or is itself one shard (e.g. a walk)."""
        root = Path(root)
        dirs = [root] if (root / "views.npy").exists() else sorted(root.glob("shard_*"))
        if shards is not None:
            dirs = [dirs[i] for i in shards if i < len(dirs)]
        self.shards = []
        self.index = []
        for d in dirs:
            s = {
                "tris": np.load(d / "tris.npy", mmap_mode="r"),
                "offsets": np.load(d / "tri_offsets.npy"),
                "views": np.load(d / "views.npy"),
                "images": np.load(d / "images.npy", mmap_mode="r"),
            }
            if (d / "parts.npy").exists():
                s["parts"] = np.load(d / "parts.npy")
                s["part_offsets"] = np.load(d / "part_offsets.npy")
                s["masks"] = np.load(d / "masks.npy", mmap_mode="r")
            self.index += [(len(self.shards), i) for i in range(len(s["views"]))]
            self.shards.append(s)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, vi = self.index[i]
        s = self.shards[si]
        v = s["views"][vi]
        scene = int(v[0])
        tris = np.asarray(s["tris"][s["offsets"][scene] : s["offsets"][scene + 1]])
        feats, tex = tri_tokens(tris, v)
        image = torch.from_numpy(np.array(s["images"][vi])).permute(2, 0, 1).float() / 255.0
        if "parts" in s:
            parts = s["parts"][s["part_offsets"][scene] : s["part_offsets"][scene + 1]]
            mask = np.unpackbits(np.array(s["masks"][vi]), axis=-1)
        else:
            parts, mask = np.zeros((0, 49), np.float32), np.zeros((H, W), np.uint8)
        pfeats, pids = part_tokens(parts, v)
        return (
            torch.from_numpy(feats),
            torch.from_numpy(tex),
            torch.from_numpy(cam_token(v)),
            image,
            torch.from_numpy(pfeats),
            torch.from_numpy(pids),
            torch.from_numpy(mask).float(),
        )


def cam_token(v):
    """Fog near/far, lens, torch colour and reach: everything per-view but the pose."""
    h = v[13]
    return np.concatenate([[h / 160.0], v[14:16] / 8.0, v[19:22], [v[22] / 4.0]]).astype(np.float32)


def tri_tokens(tris, v):
    """Camera-space triangle features, nearest first, at most MAX_TRIS."""
    pos, R, h = v[1:4], v[4:13].reshape(3, 3), v[13]
    p = (tris[:, :9].reshape(-1, 3, 3) - pos) @ R.T  # [T, 3 verts, xyz], tiles
    z = p[:, :, 2]
    near = h / 2 / 256
    k = 160.0 / h * 1.05
    zc = np.maximum(z, 1e-3)
    sx, sy = p[:, :, 0] / zc, p[:, :, 1] / zc
    ahead = z > 0
    outside = (
        ((sx > k) & ahead).all(1)
        | ((sx < -k) & ahead).all(1)
        | ((sy > 0.75 * k) & ahead).all(1)
        | ((sy < -0.75 * k) & ahead).all(1)
    )
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    n = np.cross(e1, e2)
    front = (n * -p[:, 0]).sum(1) > 0  # camera is at the origin
    keep = (z > near).any(1) & (z.min(1) < v[15]) & ~outside & front
    p, z, n, tris = p[keep], z[keep], n[keep], tris[keep]
    order = np.argsort(z.min(1))[:MAX_TRIS]
    p, z, n, tris = p[order], z[order], n[order], tris[order]

    # Screen position in NDC (x in -1..1 across the width), clamped for vertices near or behind the eye.
    zc = np.maximum(z, near * 0.5)
    ndc = np.stack([p[:, :, 0] / zc * h / 160.0, p[:, :, 1] / zc * h / 160.0], -1).clip(-4, 4)
    inv_z = (near / zc).clip(0, 2)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    feats = np.concatenate(
        [
            p.reshape(-1, 9) / 8.0,
            ndc.reshape(-1, 6),
            inv_z,
            n,
            tris[:, 9:15],
            tris[:, 16:34] * 2.0,  # baked colour and tint, 1.0 = neutral
        ],
        1,
    ).astype(np.float32)
    return feats, tris[:, 15].astype(np.int64)


def part_tokens(parts, v):
    """Camera-space features of creature parts, nearest first, at most MAX_PARTS:
    (float features [P, PART_FEATURES], int ids [P, 2] of shape and skin motif)."""
    pos, R, h = v[1:4], v[4:13].reshape(3, 3), v[13]
    near = h / 2 / 256
    centre, axes = parts[:, 3:6], parts[:, 6:15].reshape(-1, 3, 3)
    lo, hi = parts[:, 18:21], parts[:, 21:24]
    # Corners of each part's local bounding box, in camera space.
    pick = np.array([[(i >> k) & 1 for k in range(3)] for i in range(8)], np.float32)  # [8, 3]
    local = lo[:, None] + pick[None] * (hi - lo)[:, None]  # [P, 8, 3]
    corners = (centre[:, None] + np.einsum("pij,pkj->pki", axes, local) - pos) @ R.T
    z = corners[..., 2]
    zc = np.maximum(z, near * 0.5)
    ndc = (corners[..., :2] / zc[..., None] * h / 160.0).clip(-4, 4)
    ahead = z > 0
    k = 1.05
    outside = (
        ((ndc[..., 0] > k) & ahead).all(1)
        | ((ndc[..., 0] < -k) & ahead).all(1)
        | ((ndc[..., 1] > 0.75 * k) & ahead).all(1)
        | ((ndc[..., 1] < -0.75 * k) & ahead).all(1)
    )
    keep = (z > near).any(1) & (z.min(1) < v[15]) & ~outside
    order = np.argsort(z.min(1)[keep])[:MAX_PARTS]
    parts, z, ndc, axes = parts[keep][order], z[keep][order], ndc[keep][order], axes[keep][order]

    to_cam = lambda p: (p - pos) @ R.T / 8.0
    inv_z = (near / np.maximum(z, near * 0.5)).clip(0, 2)
    grad = parts[:, 38:47].reshape(-1, 3, 3)  # [P, axis, rgb]
    feats = np.concatenate(
        [
            to_cam(parts[:, 0:3]),
            to_cam(parts[:, 3:6]),
            (R @ axes).reshape(-1, 9),
            parts[:, 15:18] * 4.0,
            np.concatenate([ndc.min(1), ndc.max(1)], 1),  # screen box: x0, y0, x1, y1
            np.stack([inv_z.max(1), inv_z.min(1)], 1),
            parts[:, 25:29],  # taper, bend, bulge, lumps
            np.stack([parts[:, 30], parts[:, 31] * 4.0], 1),  # skin density, repeat size
            parts[:, 32:35],
            parts[:, 35:38],
            np.einsum("ij,pjc->pic", R, grad).reshape(-1, 9),  # light gradient, turned into camera axes
        ],
        1,
    ).astype(np.float32)
    ids = parts[:, [24, 29]].astype(np.int64)
    return feats, ids


def collate(batch):
    """Pad triangle and part tokens to the longest in the batch; the pads mark padding.
    Returns (feats, tex, pad, cams, images, (part feats, part ids, part pad, creature masks))."""
    n = max(1, max(item[0].shape[0] for item in batch))
    m = max(1, max(item[4].shape[0] for item in batch))
    feats = torch.zeros(len(batch), n, TRI_FEATURES)
    tex = torch.zeros(len(batch), n, dtype=torch.long)
    pad = torch.ones(len(batch), n, dtype=torch.bool)
    pfeats = torch.zeros(len(batch), m, PART_FEATURES)
    pids = torch.zeros(len(batch), m, 2, dtype=torch.long)
    ppad = torch.ones(len(batch), m, dtype=torch.bool)
    for i, (f, t, _, _, pf, pi, _) in enumerate(batch):
        feats[i, : len(f)] = f
        tex[i, : len(t)] = t
        pad[i, : len(f)] = False
        pfeats[i, : len(pf)] = pf
        pids[i, : len(pi)] = pi
        ppad[i, : len(pf)] = False
    cams = torch.stack([item[2] for item in batch])
    images = torch.stack([item[3] for item in batch])
    masks = torch.stack([item[6] for item in batch])
    return feats, tex, pad, cams, images, (pfeats, pids, ppad, masks)
