"""The plain level as the game rasterises it, pixel by pixel, for the per-pixel stage
(docs/deferred_stage.md): which level triangle each pixel shows and where on it.

Everything else the stage reads per pixel (texture coordinates, flesh coordinates, how far
the wall has turned, fold heights, colours) is a triangle's per-corner value mixed by those
weights, so this is all that has to agree with the game. It does what
`engine::render` and psx-bevy's PsxMaterial do:

* each triangle is split at its edges' middles until no edge is longer than a quarter tile
  (`render::MAX_EDGE`);
* back faces are dropped, and pieces crossing the near plane (0.02 tiles) are cut there;
* corners snap to whole pixels, and values are mixed linearly in screen space across each
  piece (affine, as the PS1 does), so a pixel's weights on its triangle are the piece's
  screen weights times the piece corners' weights on the triangle;
* the nearest piece wins, by perspective-correct 1/z.

`gbuffer(tris, view)` works on one view's triangles (`tris.npy` rows, positions in tiles)
on the CPU or the GPU. Pieces are binned into 16x16 tiles and tested a chunk at a time.
"""

import numpy as np
import torch

SPLIT = 0.25  # tiles: render::MAX_EDGE
NEAR = 0.02  # tiles: the level camera's near plane
H = 240
TILE = 16


def split(pos):
    """Triangles [T, 3, 3] split as the renderer splits them: (pieces [P, 3, 3], which
    triangle each came from [P], each piece corner's weights on its triangle [P, 3, 3])."""
    pos = np.asarray(pos, np.float64)
    parent = np.arange(len(pos))
    weights = np.broadcast_to(np.eye(3), (len(pos), 3, 3)).copy()
    done_p, done_t, done_w = [pos[:0]], [parent[:0]], [weights[:0]]
    while len(pos):
        edges = np.linalg.norm(pos - np.roll(pos, -1, 1), axis=2).max(1)
        small = edges <= SPLIT
        done_p.append(pos[small])
        done_t.append(parent[small])
        done_w.append(weights[small])
        pos, parent, weights = pos[~small], parent[~small], weights[~small]
        # Four pieces at the edges' middles, in the renderer's order.
        mids = lambda x: [(x[:, 0] + x[:, 1]) / 2, (x[:, 1] + x[:, 2]) / 2, (x[:, 2] + x[:, 0]) / 2]  # noqa: E731
        pa, pb, pc = pos[:, 0], pos[:, 1], pos[:, 2]
        pab, pbc, pca = mids(pos)
        wa, wb, wc = weights[:, 0], weights[:, 1], weights[:, 2]
        wab, wbc, wca = mids(weights)
        pos = np.concatenate([np.stack(c, 1) for c in [(pa, pab, pca), (pab, pb, pbc), (pca, pbc, pc), (pab, pbc, pca)]])
        weights = np.concatenate([np.stack(c, 1) for c in [(wa, wab, wca), (wab, wb, wbc), (wca, wbc, wc), (wab, wbc, wca)]])
        parent = np.tile(parent, 4)
    return np.concatenate(done_p), np.concatenate(done_t), np.concatenate(done_w)


def clip_near(p, w, parent):
    """Pieces in view space [P, 3, 3] cut at z = NEAR: what is behind goes, and a piece
    crossing it becomes one or two. Weights [P, 3, 3] follow, linearly in space (as the
    hardware's clipping does)."""
    behind = p[:, :, 2] < NEAR
    n = behind.sum(1)
    keep = n == 0
    out_p, out_w, out_t = [p[keep]], [w[keep]], [parent[keep]]
    for k in (1, 2):
        sel = n == k
        if not sel.any():
            continue
        q, qw, qt = p[sel], w[sel], parent[sel]
        # Roll each piece so its odd corner (behind for k = 1, in front for k = 2) is first.
        odd = np.argmax(behind[sel] if k == 1 else ~behind[sel], 1)
        idx = (odd[:, None] + np.arange(3)[None]) % 3
        q = np.take_along_axis(q, idx[:, :, None], 1)
        qw = np.take_along_axis(qw, idx[:, :, None], 1)

        def cut(i, j):
            t = ((NEAR - q[:, i, 2]) / (q[:, j, 2] - q[:, i, 2]))[:, None]
            return q[:, i] + (q[:, j] - q[:, i]) * t, qw[:, i] + (qw[:, j] - qw[:, i]) * t

        (x1, w1), (x2, w2) = cut(0, 1), cut(0, 2)
        if k == 1:  # a quad in front: (x1, b, c) and (x1, c, x2)
            out_p += [np.stack([x1, q[:, 1], q[:, 2]], 1), np.stack([x1, q[:, 2], x2], 1)]
            out_w += [np.stack([w1, qw[:, 1], qw[:, 2]], 1), np.stack([w1, qw[:, 2], w2], 1)]
            out_t += [qt, qt]
        else:  # only the odd corner's corner is in front
            out_p.append(np.stack([q[:, 0], x1, x2], 1))
            out_w.append(np.stack([qw[:, 0], w1, w2], 1))
            out_t.append(qt)
    return np.concatenate(out_p), np.concatenate(out_w), np.concatenate(out_t)


def project(tris, v):
    """One view's pieces, ready to rasterise: (screen corners [P, 3, 2], 1/z at the corners
    [P, 3], the triangle each is part of [P], its corners' weights on it [P, 3, 3], the
    pieces' corners in view space [P, 3, 3])."""
    width = int(v[26])
    pos, R, h = v[1:4], v[4:13].reshape(3, 3), v[13]
    t = np.asarray(tris[:, :9], np.float64).reshape(-1, 3, 3)
    tv = (t - pos) @ R.T
    # Back faces go: the winding makes (b - a) x (c - a) the outward normal.
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    front = (n * -tv[:, 0]).sum(1) > 0
    # Nothing wholly behind the camera, beyond the fog or off to a side can show.
    z = tv[:, :, 2]
    kx, ky = width / 2 / h * 1.05, 120.0 / h * 1.05
    zc = np.maximum(z, 1e-3)
    sx, sy = tv[:, :, 0] / zc, tv[:, :, 1] / zc
    ahead = z > 0
    outside = ((sx > kx) & ahead).all(1) | ((sx < -kx) & ahead).all(1) | ((sy > ky) & ahead).all(1) | ((sy < -ky) & ahead).all(1)
    seen = front & (z > NEAR).any(1) & (z.min(1) < v[15]) & ~outside
    ids = np.flatnonzero(seen)
    pieces, parent, weights = split(t[ids])
    parent = ids[parent]
    pv, weights, parent = clip_near((pieces - pos) @ R.T, weights, parent)
    z = pv[:, :, 2]
    # Corners snap to whole pixels (pixel centres are at + 0.5).
    screen = np.stack([width / 2 + h * pv[:, :, 0] / z, 120 + h * pv[:, :, 1] / z], -1)
    screen = np.round(screen)
    return screen, 1.0 / z, parent, weights, pv


def rasterise(screen, inv_z, width, device="cpu", chunk=1 << 22, view=None, views=None):
    """Pieces (`project`'s screen corners [P, 3, 2] and 1/z [P, 3], arrays or tensors) to
    pixels: (the piece each shows [240, width], -1 for none; its screen weights on the
    piece's corners [240, width, 3]; 1/z [240, width], 0 where nothing is). With `view`
    [P] (which of `views` frames each piece is in), all of them at once: [views, 240, width]."""
    dev = torch.device(device)
    S = torch.as_tensor(screen, dtype=torch.float32, device=dev)
    IZ = torch.as_tensor(inv_z, dtype=torch.float32, device=dev)
    one = view is None
    V = torch.zeros(len(S), dtype=torch.long, device=dev) if one else torch.as_tensor(view, dtype=torch.long, device=dev)
    views = 1 if one else views
    a, b, c = S[:, 0], S[:, 1], S[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    live = area.abs() > 1e-6
    # Each piece's tiles: those its pixel box (of pixel centres it could cover) touches.
    lo = (S.min(1).values - 0.5).ceil().clamp(min=0)
    hi = (S.max(1).values - 0.5).floor()
    hi = torch.stack([hi[:, 0].clamp(max=width - 1), hi[:, 1].clamp(max=H - 1)], 1)
    live &= (hi >= lo).all(1)
    gw, gh = width // TILE, H // TILE
    t_lo, t_hi = (lo // TILE).long(), (hi // TILE).long()
    pid = torch.nonzero(live).squeeze(1)
    nx = t_hi[pid, 0] - t_lo[pid, 0] + 1
    ny = t_hi[pid, 1] - t_lo[pid, 1] + 1
    count = nx * ny
    pair_piece = pid.repeat_interleave(count)
    k = torch.arange(int(count.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(count, 0) - count, count)
    nxr = nx.repeat_interleave(count)
    tx = t_lo[pair_piece, 0] + k % nxr
    ty = t_lo[pair_piece, 1] + k // nxr
    tile = (V[pair_piece] * gh + ty) * gw + tx

    plane = H * width
    best = torch.full((views * plane,), -1.0, device=dev)
    who = torch.full((views * plane,), -1, dtype=torch.long, device=dev)
    oy, ox = torch.meshgrid(torch.arange(TILE, device=dev), torch.arange(TILE, device=dev), indexing="ij")
    off = torch.stack([ox.flatten(), oy.flatten()], 1)  # [256, 2]
    step = max(1, chunk // (TILE * TILE))

    def pairs(s):
        """The pixels of pairs s.. and the screen weights of their centres on the pieces."""
        pp, tt = pair_piece[s : s + step], tile[s : s + step]
        px = (tt % gw)[:, None] * TILE + off[None, :, 0]
        py = (tt // gw % gh)[:, None] * TILE + off[None, :, 1]
        pv = (tt // (gw * gh))[:, None]
        lam = weights_at(pp[:, None], px + 0.5, py + 0.5)
        inside = (lam >= 0).all(-1) & (px < width)
        iz = (lam * IZ[pp][:, None]).sum(-1)
        flat = pv * plane + (py * width + px).clamp(max=plane - 1)
        return pp, flat, torch.where(inside, iz, torch.full_like(iz, -1.0)), inside

    def weights_at(pp, cx, cy):
        A, B, C = a[pp], b[pp], c[pp]
        ar = area[pp]
        l0 = ((B[..., 0] - cx) * (C[..., 1] - cy) - (B[..., 1] - cy) * (C[..., 0] - cx)) / ar
        l1 = ((C[..., 0] - cx) * (A[..., 1] - cy) - (C[..., 1] - cy) * (A[..., 0] - cx)) / ar
        return torch.stack([l0, l1, 1 - l0 - l1], -1)

    # Nearest first; then, of the pieces that are that near, the last, so that a pixel's piece
    # doesn't depend on the order duplicate writes land in.
    for s in range(0, len(tile), step):
        _, flat, iz, _ = pairs(s)
        best.scatter_reduce_(0, flat.flatten(), iz.flatten(), "amax")
    for s in range(0, len(tile), step):
        pp, flat, iz, inside = pairs(s)
        win = inside & (iz == best[flat])
        who.scatter_reduce_(0, flat[win], pp[:, None].expand_as(flat)[win], "amax")
    # The weights from the piece that won.
    lam = torch.zeros(views * plane, 3, device=dev)
    f = torch.nonzero(who >= 0).squeeze(1)
    if len(f):
        q = f % plane
        lam[f] = weights_at(who[f], (q % width).float() + 0.5, (q // width).float() + 0.5)
    shape = (H, width) if one else (views, H, width)
    return who.view(shape), lam.view(*shape, 3), best.clamp(min=0).view(shape)


def gbuffer(tris, v, device="cpu"):
    """Rasterise one view: a dict of per-pixel tensors [240, width]: `tri` (the tris row
    shown, -1 for none), `weights` [.., 3] (on that triangle's corners), `inv_z` (1/z,
    tiles), and for checks against the renderer `piece` and `lam` [.., 3] (screen weights
    on the piece's corners), with `pieces` the pieces' corners in view space."""
    width = int(v[26])
    screen, inv_z, parent, weights, pv = project(tris, v)
    dev = torch.device(device)
    piece, lam, iz = rasterise(screen, inv_z, width, dev)
    covered = piece >= 0
    tri = torch.full_like(piece, -1)
    tri[covered] = torch.as_tensor(parent, device=dev)[piece[covered]]
    W = torch.as_tensor(weights, dtype=torch.float32, device=dev)
    w = torch.zeros(*piece.shape, 3, device=dev)
    w[covered] = (lam[covered, :, None] * W[piece[covered]]).sum(1)
    return {"tri": tri, "weights": w, "inv_z": iz, "piece": piece, "lam": lam, "pieces": pv}
