"""What the per-pixel stage (docs/deferred_stage.md, model.Deferred) is told about each
pixel, worked out from the G-buffer (gbuffer.py) and the view's own triangles and organs.

`prepare(tris, turning, organs, v, amount)` runs in the data loader (CPU): it splits and
projects the triangles and keeps the rows of those that can show. `inputs(views, device)`
runs in the trainer on a batch of them (one width): it rasterises on the device and returns
[b, CHANNELS, 240, width] plus what the stage samples its latent textures with.
"""

import numpy as np
import torch

from gbuffer import H, project, rasterise

SIZE = 64  # tex::SIZE: texture coordinates are texels / 64 in tris.npy, / 63 in the renderer
# Level texture -> painted flesh slot that replaces it (dreamed.rs CLEAN, slot_of): none past 7.
CLEAN = {3: 0, 4: 1, 5: 1, 7: 6}
FLESH_SLOTS = 8
# Plain channels, in order: covered; 1/z; facing (3); flesh there; how far it has turned; the
# view's amount; fold height; eye or sphincter (2), its u, v and radius, and state (4); vein
# nearness, across, along and beat.
CHANNELS = 1 + 1 + 3 + 1 + 1 + 1 + 1 + 2 + 3 + 4 + 4
EMPTY = np.zeros((0, 19), np.float32)


def prepare(tris, turning, organs, v, amount):
    """One view's G-buffer inputs, as numpy: the projected pieces and the rows of the
    triangles (and their turning) that they come from."""
    screen, inv_z, parent, weights, _ = project(tris, v)
    used, local = np.unique(parent, return_inverse=True)
    rows = np.asarray(tris[used], np.float32)
    turn = np.asarray(turning[used], np.float32) if turning is not None else np.zeros((len(used), 15), np.float32)
    return {
        "screen": screen.astype(np.float32),
        "inv_z": inv_z.astype(np.float32),
        "tri": local.astype(np.int64),
        "weights": weights.astype(np.float32),
        "rows": rows,
        "turning": turn,
        "organs": np.asarray(organs if organs is not None else EMPTY, np.float32),
        "view": np.asarray(v, np.float32),
        "amount": np.float32(amount),
    }


def _organ_frame(o, v, dev):
    """Organs [O, 19] in view space: (at, faces, across, up) [O, 3] each."""
    pos = torch.as_tensor(v[1:4], device=dev)
    R = torch.as_tensor(v[4:13], device=dev).view(3, 3)
    at = (o[:, 1:4] - pos) @ R.T
    return at, o[:, 4:7] @ R.T, o[:, 7:10] @ R.T, o[:, 10:13] @ R.T


def _on_plane(rays, at, n):
    """Where each pixel's ray [N, 3] meets each organ's plane (through `at`, facing `n`
    [O, 3]): the points [N, O, 3] and how far along the ray [N, O] (inf if never)."""
    num = (at * n).sum(-1)  # [O]
    den = rays @ n.T  # [N, O]
    t = num[None] / torch.where(den.abs() > 1e-6, den, torch.full_like(den, 1e-6))
    t = torch.where(t > 0, t, torch.full_like(t, float("inf")))
    return rays[:, None] * t[..., None], t


def one_view(g, dev):
    """Plain channels [CHANNELS, 240, width] and sampling coordinates for one prepared view."""
    v = g["view"]
    width, h = int(v[26]), float(v[13])
    piece, lam, iz = rasterise(g["screen"], g["inv_z"], width, dev)
    covered = piece >= 0
    pc = piece.clamp(min=0)
    tri = torch.as_tensor(g["tri"], device=dev)[pc]  # [H, W]
    pw = torch.as_tensor(g["weights"], device=dev)[pc]  # [H, W, 3 piece corners, 3 tri corners]
    w = (lam[..., :, None] * pw).sum(2) * covered[..., None]  # weights on the triangle's corners
    rows = torch.as_tensor(g["rows"], device=dev)[tri]
    turn = torch.as_tensor(g["turning"], device=dev)[tri]
    mix = lambda x, k: (x.view(H, width, 3, k) * w[..., None]).sum(2)  # noqa: E731
    cov = covered.float()

    # Texture: which, and where on it (the renderer's 0..1).
    texid = rows[..., 15].long().clamp(0, 11) * covered
    uv = mix(rows[..., 9:15], 2) * SIZE / (SIZE - 1)
    # Facing in view space.
    R = torch.as_tensor(v[4:13], device=dev).view(3, 3)
    p = rows[..., :9].view(H, width, 3, 3)
    n = torch.linalg.cross(p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 0, :]) @ R.T
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    near = h / 2 / 256
    inv = (iz * near).clamp(0, 2)

    # Flesh: where on the painted slot's texture, how far turned, and the fold (a quadratic
    # through the corners and the middles of edges ab, ac, bc).
    lay = mix(turn[..., 3:9], 2)
    slot = torch.full_like(texid, -1)
    for t in range(8):
        slot[texid == t] = CLEAN.get(t, t)
    flesh = covered & (turn[..., 3:9].abs().sum(-1) > 0) & (slot >= 0)
    reach = float(g["amount"]) * mix(turn[..., 0:3], 1)[..., 0]
    a, b, c = w[..., 0], w[..., 1], w[..., 2]
    fh = turn[..., 9:15]
    fold = (fh[..., 0] * a * (2 * a - 1) + fh[..., 1] * b * (2 * b - 1) + fh[..., 2] * c * (2 * c - 1)
            + 4 * (fh[..., 3] * a * b + fh[..., 4] * a * c + fh[..., 5] * b * c))

    # Organs, where each pixel's ray meets their planes (they stand proud of the plain wall).
    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(width, device=dev) + 0.5, indexing="ij")
    rays = torch.stack([(xs - width / 2) / h, (ys - 120) / h, torch.ones_like(xs)], -1).view(-1, 3)
    depth = torch.where(covered, 1 / iz.clamp(min=1e-6), torch.full_like(iz, 1e6)).view(-1)
    o = torch.as_tensor(g["organs"], device=dev)
    feat = torch.zeros(H * width, 13, device=dev)
    if len(o):
        kind = o[:, 0].long()
        at, faces, across, up = _organ_frame(o, v, dev)
        q, t = _on_plane(rays, at, faces)
        d = q - at[None]
        # Rays that never meet a plane are far off it, not infinitely (inf * 0 is NaN).
        far = lambda x: torch.nan_to_num(x, nan=1e3, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)  # noqa: E731
        u = far((d * across[None]).sum(-1) / o[None, :, 13].clamp(min=1e-4))
        vv = far((d * up[None]).sum(-1) / o[None, :, 14].clamp(min=1e-4))
        # Shown unless the level stands well in front of it.
        seen = t < depth[:, None] + 0.1
        # Eyes and sphincters: the nearest-centred one the pixel is on.
        r2 = u**2 + vv**2
        spot = seen & (kind[None] <= 1) & (r2 < 1.44) & (o[None, :, 15] > 0)
        score = torch.where(spot, r2, torch.full_like(r2, float("inf")))
        best = score.argmin(1)
        on = torch.isfinite(score.gather(1, best[:, None]))[:, 0]
        pick = lambda x: x.gather(1, best[:, None])[:, 0]  # noqa: E731
        k = kind[best]
        feat[:, 0] = (on & (k == 0)).float()
        feat[:, 1] = (on & (k == 1)).float()
        feat[:, 2] = pick(u) * on
        feat[:, 3] = pick(vv) * on
        feat[:, 4] = pick(r2).sqrt() * on
        feat[:, 5:9] = o[best, 15:19] * on[:, None]
        # Veins: the nearest the pixel is on, across in radii and along in half lengths.
        near_v = torch.exp(-(vv**2)) * (u.abs() <= 1) * seen * (kind[None] == 2)
        best_v = near_v.argmax(1)
        pv = lambda x: x.gather(1, best_v[:, None])[:, 0]  # noqa: E731
        feat[:, 9] = pv(near_v)
        feat[:, 10] = pv(vv).clamp(-3, 3) / 3
        feat[:, 11] = pv(u).clamp(-1, 1)
        feat[:, 12] = o[best_v, 15] * (feat[:, 9] > 0.01)
    feat = feat.view(H, width, 13)

    plain = torch.cat([
        cov[..., None], inv[..., None], n * cov[..., None], flesh.float()[..., None], reach[..., None],
        torch.full_like(cov, float(g["amount"]))[..., None], fold[..., None] * 4, feat,
    ], -1).permute(2, 0, 1)
    eye_on = feat[..., 0] > 0
    return plain, {
        "tex": texid, "uv": uv.clamp(0, 1),
        "slot": slot.clamp(min=0), "flesh": flesh, "lay": lay - lay.floor(),
        "eye": eye_on, "eye_uv": feat[..., 2:4].clamp(-1, 1), "blink": feat[..., 6], "gaze": feat[..., 7:9],
    }


def inputs(views, device):
    """A batch of prepared views (one width): (plain channels [b, CHANNELS, 240, width],
    sampling coordinates, each stacked [b, ...])."""
    dev = torch.device(device)
    outs = [one_view(g, dev) for g in views]
    plain = torch.stack([o[0] for o in outs])
    coords = {k: torch.stack([o[1][k] for o in outs]) for k in outs[0][1]}
    return plain, coords
