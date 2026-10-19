"""Dataset over datagen shards: one item = (triangle tokens, camera token, image).

Triangles are moved into camera space and culled the cheap ways a renderer
would before rasterising (behind the camera, beyond the fog, outside the
frustum, back-facing). Occlusion, perspective-correct-ish coverage, texturing,
fog and torch light are left for the network to learn.
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
        return torch.from_numpy(feats), torch.from_numpy(tex), torch.from_numpy(cam_token(v)), image


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


def collate(batch):
    """Pad triangle tokens to the longest in the batch; mask marks padding."""
    n = max(1, max(f.shape[0] for f, *_ in batch))
    feats = torch.zeros(len(batch), n, TRI_FEATURES)
    tex = torch.zeros(len(batch), n, dtype=torch.long)
    pad = torch.ones(len(batch), n, dtype=torch.bool)
    for i, (f, t, _, _) in enumerate(batch):
        feats[i, : len(f)] = f
        tex[i, : len(t)] = t
        pad[i, : len(f)] = False
    cams = torch.stack([c for *_, c, _ in batch])
    images = torch.stack([im for *_, im in batch])
    return feats, tex, pad, cams, images
