"""Dream creatures: creatures as described parts, drawn by a network that CLIP teaches.

Every part of a creature is a phrase ("a tentacle covered in bloodshot eyes"), picked
from a vocabulary by its skin motif and shape (`VOCAB`), plus where it is: its axis
projected onto the screen as a short polyline, its radius and depth. Nothing says
where its surface is; the network invents the shape, skin and silhouette, and CLIP
(train_dream.py) judges whether each part looks like its phrase. Each creature gets a
colour scheme of four colours (`PALETTES`); the network mixes them and shades the
mix, so its colours stay in the scheme (free colour turns CLIP rainbow).

Per pixel the network sees (`fields`): how near the nearest part's axis is (in its
radii), where along and across it the pixel lies, its depth, the part's light, which
way the limb's rounded surface faces and how far round the limb the pixel is, noise
fixed to that surface (so it turns with the part as the camera goes round), and a blend
of the nearby parts' learned phrase vectors, so where a leg meets an eye the network
sees both and can morph one into the other. Parts at the same depth melt together, with
every field blended and the outline swelling at the joint; where one limb is clearly in
front of another it hides it.
The surface also lights the drawing (`lighting`), so CLIP judges what the game shows.
The engine computes the same fields the same way. Fog is laid on afterwards (`fogged`),
so CLIP judges the creatures clear.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from creatures import Block, noise3
from data import H, Shards, view_width

MAX_PARTS = 32
# Points along a part's (bent) axis, projected: the polyline has one segment fewer.
AXIS_POINTS = 5
NOISE, OCTAVES = 4, 3
PALETTE = 4
# Per part, in camera space (`view_parts`): centre, local x and y axes, radius x,
# half-length y, radius z, bend, shape, motif, baked light (mean rgb), creature, phrase,
# noise seed, valid.
Q_CENTRE, Q_AX, Q_AY, Q_HALF = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)
Q_BEND, Q_SHAPE, Q_MOTIF, Q_LIGHT, Q_CREATURE, Q_PHRASE, Q_SEED, Q_VALID = 12, 13, 14, slice(15, 18), 18, 19, 20, 21
Q_PARAMS = 22
# Per crop: x0, y0, projection distance, screen width, fog near, fog far, fog rgb.
C_PARAMS = 9
FIELD_CHANNELS = 1 + 1 + 1 + 1 + 1 + 1 + 3 + NOISE * OCTAVES + PALETTE * 3 + 3 + 2 + 1
NOISE_CHANNELS = slice(9, 9 + NOISE * OCTAVES)
# After the colours: the surface's normal in camera space, (cos, sin) of the angle round
# the limb, and how shut in the surface is by its neighbours.
F_NORMAL = 9 + NOISE * OCTAVES + PALETTE * 3
F_CREASE = F_NORMAL + 5
# Where the drawn edge of a limb is, in its radii (the net's alpha crosses a half about here).
EDGE = 1.25
# How widely parts melt into each other at their joints, and how much the outline swells there.
MELT, FILLET = 1.0, 2.2

MOTIFS = ["flesh", "eyes", "teeth", "mouths", "suckers", "veins", "scales", "chitin", "hair", "wrinkles", "bone"]
NOUNS = [
    ["lump", "sac", "growth", "head"],  # ellipsoid
    ["limb", "arm", "leg", "finger"],  # capsule
    ["horn", "claw", "spike", "beak"],  # cone
    ["tentacle", "tongue", "tail", "worm"],  # tentacle
]
TEMPLATES = {
    "flesh": ["a {} of raw pink flesh", "a swollen fleshy {}", "a {} of melting skin", "a glistening wet {}"],
    "eyes": ["a {} covered in bloodshot eyes", "a {} with a staring human eye", "a {} sprouting many small eyes", "a {} with a yellow reptile eye"],
    "teeth": ["a {} full of sharp teeth", "a {} with rows of human teeth", "a {} with jagged fangs", "a {} studded with molars"],
    "mouths": ["a {} with a gaping mouth", "a {} with wet lips", "a {} with a screaming mouth", "a {} with a lolling tongue"],
    "suckers": ["a {} covered in octopus suckers", "a {} with round pink suckers", "a {} lined with suction cups"],
    "veins": ["a {} covered in throbbing veins", "a {} with blue veins under pale skin", "a {} of red veiny flesh"],
    "scales": ["a scaly {}", "a {} covered in fish scales", "a {} with snake skin"],
    "chitin": ["a shiny insect {}", "a {} of black beetle shell", "a segmented chitin {}"],
    "hair": ["a hairy spider {}", "a furry {}", "a {} covered in coarse bristles", "a {} with matted fur"],
    "wrinkles": ["a wrinkled {}", "a {} of folded old skin", "a {} like a brain"],
    "bone": ["a {} of bare bone", "a {} with ribs showing", "a {} of yellowed bone", "a skeletal {}"],
}
# Any part, whatever its motif, sometimes dreams one of these instead.
DREAMS = ["a {} with a dog's face", "a {} made of human hands", "a {} turning into a bird's head", "a {} with a human ear", "a {} of fingers and nails", "a {} sprouting tiny mouths"]
DREAM_SHARE = 0.25
# The small vocabulary (`vocab="motifs"`): one phrase per skin, given to parts at random.
MOTIF_PHRASES = [
    "raw pink flesh", "bloodshot human eyes", "a mouth full of sharp teeth", "gaping wet mouths",
    "octopus suckers", "throbbing veins under pale skin", "snake scales", "shiny black beetle shell",
    "coarse black hair", "wrinkled folded skin", "bare yellowed bone",
]
WHOLES = ["a grotesque creature of melting flesh", "a nightmarish flesh monster", "a dreamlike monster made of body parts", "a strange fleshy beast"]
NEGATIVE = "a rainbow psychedelic pattern"


def build_vocab():
    """(phrases, {(motif, shape): phrase ids}, {shape: dream phrase ids}, whole ids, negative id)."""
    phrases, by_kind, dreams = [], {}, {}
    for m, name in enumerate(MOTIFS):
        for s, nouns in enumerate(NOUNS):
            by_kind[m, s] = []
            for tpl in TEMPLATES[name]:
                for n in nouns:
                    by_kind[m, s].append(len(phrases))
                    phrases.append(tpl.format(n))
    for s, nouns in enumerate(NOUNS):
        dreams[s] = []
        for tpl in DREAMS:
            for n in nouns:
                dreams[s].append(len(phrases))
                phrases.append(tpl.format(n))
    full = list(range(len(phrases)))
    motifs = list(range(len(phrases), len(phrases) + len(MOTIF_PHRASES)))
    phrases += MOTIF_PHRASES
    wholes = list(range(len(phrases), len(phrases) + len(WHOLES)))
    phrases += WHOLES
    phrases.append(NEGATIVE)
    return phrases, by_kind, dreams, {"full": full, "motifs": motifs}, wholes, len(phrases) - 1


# `VOCABS`: the part phrases of each vocabulary, which a part's phrase must beat all others of.
VOCAB, BY_KIND, DREAM_IDS, VOCABS, WHOLE_IDS, NEGATIVE_ID = build_vocab()

PALETTES = [
    ["#efe6d2", "#d9a3a0", "#7a1f1f", "#3b2a20"],  # bone
    ["#b5c26a", "#e3d36b", "#6f7466", "#1c1f17"],  # bile
    ["#6b4a7a", "#8193a8", "#c79bb3", "#d9cf7a"],  # bruise
    ["#f2b8a8", "#c2453d", "#5c1a24", "#f7efe4"],  # raw
    ["#d8d2c4", "#8a8f86", "#3e4442", "#a3332c"],  # corpse
    ["#e7c9a9", "#b07a4f", "#5a3a22", "#f2e8d8"],  # hide
    ["#2a2430", "#5d4e6d", "#a69cc0", "#e8e0f0"],  # night
    ["#c9d6c0", "#6e8f6a", "#2f3f2c", "#d8b4a0"],  # moss
]


def hex_rgb(h):
    return [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]


def random_palette(rng):
    """A scheme from `PALETTES`, or (a third of the time) one made from a random hue:
    a pale, a mid and a dark shade of it, and an accent."""
    if rng.random() < 2 / 3:
        p = np.array([hex_rgb(h) for h in PALETTES[rng.integers(len(PALETTES))]])
        return (p * rng.uniform(0.9, 1.1)).clip(0, 1).astype(np.float32)
    import colorsys

    h = rng.random()
    shades = [(h, 0.2, 0.92), (h + 0.04, 0.55, 0.65), (h - 0.04, 0.7, 0.25), (h + rng.uniform(0.3, 0.7), 0.5, 0.8)]
    return np.array([colorsys.hsv_to_rgb(x % 1, s, v) for x, s, v in shades], np.float32)


def view_parts(parts, v, rng, box=None, vocab="full", dream_share=DREAM_SHARE):
    """Scene parts [P, 49] (parts.npy) in camera space for view `v`, ahead of the camera
    (and reaching into screen `box` x0, y0, x1, y1 if given), nearest first, as float32
    [MAX_PARTS, Q_PARAMS] (zero padding), each given a phrase from `vocab` (VOCABS): by its
    motif and shape for "full", a motif at random for "motifs"."""
    pos, R = v[1:4], v[4:13].reshape(3, 3)
    centre = (parts[:, 3:6] - pos) @ R.T
    axes = R @ parts[:, 6:15].reshape(-1, 3, 3)  # columns: local x, y, z in camera space
    reach = np.abs(parts[:, 15:18]).max(1)
    ahead = centre[:, 2] + reach > 0.05
    if box is not None:
        z = centre[:, 2].clip(min=0.05)
        px = centre[:, 0] / z * v[13] + view_width(v) / 2
        py = centre[:, 1] / z * v[13] + H / 2
        r = 3 * reach / z * v[13]
        ahead &= (px + r > box[0]) & (px - r < box[2]) & (py + r > box[1]) & (py - r < box[3])
    keep = np.nonzero(ahead)[0]
    keep = keep[np.argsort(np.linalg.norm(centre[keep], axis=1))[:MAX_PARTS]]
    q = np.zeros((MAX_PARTS, Q_PARAMS), np.float32)
    for i, k in enumerate(keep):
        shape, motif = int(parts[k, 24]), int(parts[k, 29])
        if vocab == "motifs":
            motif = int(rng.integers(len(MOTIFS)))
            ids = [VOCABS["motifs"][motif]]
        else:
            ids = DREAM_IDS[shape] if rng.random() < dream_share else BY_KIND[motif, shape]
        q[i, Q_CENTRE] = centre[k]
        q[i, Q_AX] = axes[k][:, 0]
        q[i, Q_AY] = axes[k][:, 1]
        q[i, Q_HALF] = parts[k, 15:18]
        q[i, Q_BEND] = parts[k, 26]
        q[i, Q_SHAPE] = shape
        q[i, Q_MOTIF] = motif
        q[i, Q_LIGHT] = parts[k, 35:38]
        q[i, Q_CREATURE] = parts[k, 48]
        q[i, Q_PHRASE] = ids[rng.integers(len(ids))]
        q[i, Q_SEED] = rng.integers(0, 2**24)
        q[i, Q_VALID] = 1
    return q


def project(q, cam, n):
    """Each part's axis on the crop: points [B, P, AXIS_POINTS, 2] (crop pixels), their
    depth [B, P, AXIS_POINTS], their t (-1 at the joint end, 1 at the far end) and the
    points in camera space [B, P, AXIS_POINTS, 3], depth clamped as for the projection."""
    t = torch.linspace(-1, 1, AXIS_POINTS, device=q.device)
    s = (t + 1) / 2
    hy = q[..., 10:11]
    pts = (
        q[..., None, Q_CENTRE]
        + q[..., None, Q_AY] * (hy * t)[..., None]
        + q[..., None, Q_AX] * (q[..., Q_BEND : Q_BEND + 1] * hy * s * s)[..., None]
    )
    z = pts[..., 2].clamp(min=0.05)
    vh, width = cam[:, 2, None, None], cam[:, 3, None, None]
    x = pts[..., 0] / z * vh + width / 2 - cam[:, 0, None, None]
    y = pts[..., 1] / z * vh + H / 2 - cam[:, 1, None, None]
    return torch.stack([x, y], -1), z, t, torch.cat([pts[..., :2], z[..., None]], -1)


def smooth(a, b, x):
    """WGSL smoothstep."""
    t = ((x - a) / (b - a).clamp(min=1e-6)).clamp(0, 1) if torch.is_tensor(b) else ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def lighting(feats):
    """What the limbs' shape does to their colour, [B, 1, h, w], from `fields`: a light above
    and to the left of the viewer (camera space: y is down, z is away), a rim, dark creases."""
    n = feats[:, F_NORMAL : F_NORMAL + 3].float()
    towards = n.new_tensor([-0.3, -0.6, -0.74])
    towards = towards / towards.norm()
    wrap = (((n * towards[:, None, None]).sum(1, keepdim=True) + 0.35) / 1.35).clamp(0, 1)
    edge = 1 + n[:, 2:3]
    return (0.42 + 0.72 * wrap + 0.2 * edge * edge * edge) * (1 - 0.3 * feats[:, F_CREASE : F_CREASE + 1].float())


def fields(q, cam, palette, n):
    """Per-pixel fields [B, FIELD_CHANNELS, n, n] for parts `q` [B, P, Q_PARAMS] on crops
    `cam` [B, C_PARAMS] with colour schemes `palette` [B, PALETTE, 3]; the parts' blend
    weights [B, P, n, n] (for their phrase vectors); each pixel's distance from the
    nearest axis [B, n, n], in that part's radii; and its fog [B, 1, n, n] (0..1)."""
    B, P = q.shape[:2]
    pts, z, t, points = project(q, cam, n)
    tangents = unit(points[..., 1:, :] - points[..., :-1, :])  # [B, P, K, 3]
    a, b = pts[..., :-1, :], pts[..., 1:, :]  # segments [B, P, K, 2]
    za, zb = z[..., :-1], z[..., 1:]
    ta, tb = t[:-1], t[1:]
    ys, xs = torch.meshgrid(torch.arange(n, device=q.device) + 0.5, torch.arange(n, device=q.device) + 0.5, indexing="ij")
    p = torch.stack([xs, ys], -1)  # [n, n, 2]
    ab = b - a
    ap = p[None, None, None] - a[..., None, None, :]  # [B, P, K, n, n, 2]
    u = ((ap * ab[..., None, None, :]).sum(-1) / (ab * ab).sum(-1)[..., None, None].clamp(min=1e-6)).clamp(0, 1)
    off = ap - u[..., None] * ab[..., None, None, :]
    dist = off.norm(dim=-1)
    cross = ab[..., None, None, 0] * ap[..., 1] - ab[..., None, None, 1] * ap[..., 0]
    side = torch.sign(cross)
    depth = za[..., None, None] + u * (zb - za)[..., None, None]
    along = ta[:, None, None] + u * (tb - ta)[:, None, None]
    radius = 0.5 * (q[..., 9] + q[..., 11])  # world units
    rpix = (radius[..., None, None, None] * cam[:, 2, None, None, None, None] / depth).clamp(min=0.5)
    d = dist / rpix
    d, k = d.min(2)  # nearest segment per part
    pick = lambda x: torch.gather(x, 2, k[:, :, None]).squeeze(2)  # noqa: E731
    across = pick(side * dist) / pick(rpix)
    depth, along, rpix = pick(depth), pick(along), pick(rpix)
    off = torch.gather(off, 2, k[:, :, None, :, :, None].expand(-1, -1, -1, -1, -1, 2)).squeeze(2)
    valid = q[..., Q_VALID] > 0
    d = torch.where(valid[..., None, None], d, torch.full_like(d, 1e4))
    weights = torch.softmax(-2 * d * d, 1) * (d.amin(1, keepdim=True) < 4)
    # Each limb's rounded surface here, and among those covering a pixel the nearest wins.
    surface = depth - radius[..., None, None] * EDGE * (1 - (d / EDGE).clamp(max=1) ** 2).clamp(min=0).sqrt()
    inside = d < EDGE
    closest = d.amin(1)
    i = torch.where(inside.any(1), torch.where(inside, surface, torch.full_like(surface, float("inf"))).argmin(1), d.argmin(1))
    g = lambda x, at: torch.gather(x, 1, at[:, None]).squeeze(1)  # noqa: E731
    bi = torch.arange(B, device=q.device)[:, None, None]
    # The outline: a smooth minimum, so joints swell into each other.
    melted = closest - torch.log(torch.exp(-FILLET * (d - closest[:, None])).sum(1)) / FILLET
    # The part it melts into: the strongest of those level with this surface. One clearly
    # behind it counts for nothing, so a limb crossing in front of another stays crisp.
    s_i, r_i = g(surface, i), g(radius[..., None, None].expand_as(d), i)
    wide = torch.maximum(radius[..., None, None].expand_as(d), r_i[:, None])
    level = 1 - smooth(wide, 3 * wide, (surface - s_i[:, None]).abs())
    strength = torch.exp(-MELT * d * d)
    others = (strength * level).scatter(1, i[:, None], 0.0)
    j = others.argmax(1)
    blend = (g(others, j) / (g(strength, i) + g(others, j)).clamp(min=1e-12))[..., None]
    corner = torch.stack([cam[:, 0] - cam[:, 3] / 2, cam[:, 1] - H / 2], -1)[:, None, None]
    ray = unit(torch.cat([(p[None] + corner) / cam[:, 2, None, None, None], torch.ones_like(p[None][..., :1]).expand(B, -1, -1, -1)], -1))

    # Every part's surface normal: round the limb from the pixel's side to facing the camera.
    tangent = tangents[torch.arange(B, device=q.device)[:, None, None, None], torch.arange(P, device=q.device)[None, :, None, None], k]
    across_limb = unit(torch.linalg.cross(tangent, ray[:, None]))
    limb_side = torch.sign((across_limb[..., :2] * off).sum(-1, keepdim=True))
    facing = -unit(ray[:, None] - tangent * (ray[:, None] * tangent).sum(-1, keepdim=True))
    round_ = (d / EDGE).clamp(max=1)[..., None]
    normals = across_limb * limb_side * round_ + facing * (1 - round_**2).clamp(min=0).sqrt()

    def skin(at):
        """Part `at`'s row, its surface normal, (cos, sin) round it and the noise fixed to that."""
        row = q[bi, at]
        normal = torch.gather(normals, 1, at[:, None, :, :, None].expand(-1, -1, -1, -1, 3)).squeeze(1)
        around = torch.stack([(normal * row[..., Q_AX]).sum(-1), (normal * torch.linalg.cross(row[..., Q_AX], row[..., Q_AY])).sum(-1)], -1)
        around = around / around.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        hy_over_r = row[..., 10] / (0.5 * (row[..., 9] + row[..., 11])).clamp(min=1e-3)
        place = torch.cat([(g(along, at) * hy_over_r)[..., None], around], -1)
        seed = row[..., Q_SEED].long()
        noise = torch.stack([noise3(place * 2**o, seed, c * OCTAVES + o) for c in range(NOISE) for o in range(OCTAVES)], -1)
        return row, normal, around, noise

    row_i, _, around_i, noise_i = skin(i)
    row_j, _, around_j, noise_j = skin(j)
    t = blend.squeeze(-1)
    mix2 = lambda x: g(x, i) * (1 - t) + g(x, j) * t  # noqa: E731
    dmin, dep, alo, acr, rp = g(d, i) * (1 - t) + g(d, j) * t, mix2(depth), mix2(along), mix2(across), mix2(rpix)
    presence = torch.exp(-0.5 * melted * melted)
    light = (row_i[..., Q_LIGHT] * (1 - blend) + row_j[..., Q_LIGHT] * blend).permute(0, 3, 1, 2)
    fog = ((dep - cam[:, 4, None, None]) / (cam[:, 5] - cam[:, 4]).clamp(min=1e-3)[:, None, None]).clamp(0, 1)
    # The shading shows seams most, so the normal blends over every part level with the surface.
    normal = unit(((strength * level)[..., None] * normals).sum(1))
    around = around_i * (1 - blend) + around_j * blend
    around = around / around.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    noise = (noise_i * (1 - blend) + noise_j * blend).permute(0, 3, 1, 2)
    # Soft rings round limbs clearly in front of this surface; joints get none.
    ahead = (s_i * (1 - t) + g(surface, j) * t)[:, None] - surface
    ring = smooth(0.9, EDGE, d) * (1 - smooth(EDGE, 2.1, d))
    r = r_i[:, None]
    crease = (ring * smooth(0.4 * r, 1.2 * r, ahead) * (1 - smooth(2 * r, 6 * r, ahead))).amax(1)
    feats = torch.cat(
        [
            presence[:, None],
            dmin.clamp(max=8)[:, None] / 4,
            alo[:, None],
            acr.clamp(-8, 8)[:, None] / 4,
            (dep / 8)[:, None],
            rp.log()[:, None] / 4,
            light,
            noise,
            palette.reshape(B, -1, 1, 1).expand(B, PALETTE * 3, n, n),
            normal.permute(0, 3, 1, 2),
            around.permute(0, 3, 1, 2),
            crease[:, None],
        ],
        1,
    )
    far = g(d, i) < 4
    feats = torch.where(far[:, None], feats, torch.zeros_like(feats))
    return feats, weights, melted, fog[:, None]


def fogged(layer, fog, cam):
    """The layer's colour faded to the crops' fog colour by `fog` [B, 1, h, w]."""
    rgb = layer[:, :3] * (1 - fog) + cam[:, 6:9, None, None] * fog
    return torch.cat([rgb, layer[:, 3:]], 1)


class DreamCrops(Dataset):
    """Square crops of `size` round datagen views' creatures: per item parts [MAX_PARTS,
    Q_PARAMS] with phrases, the crop [C_PARAMS], the level's frame [3, size, size], a
    colour scheme [PALETTE, 3], a whole-creature phrase, and the parts to judge
    [JUDGED, 4] (index into the parts, -1 for none; centre x, y and extent in crop pixels):
    the biggest whose centre is in the crop.
    `fixed` gives every item the same crop, phrases and scheme each time (for validation)."""

    JUDGED = 2

    def __init__(self, root, shards=None, size=128, fixed=False, vocab="full"):
        self.shards = Shards(root, shards)
        self.size, self.fixed, self.vocab = size, fixed, vocab
        self.items = []
        for i, (si, vi) in enumerate(self.shards.index):
            s = self.shards.shards[si]
            if "boxes" in s:
                x0, y0, x1, y1 = s["boxes"][vi]
                if (x1 - x0) * (y1 - y0) >= 64:
                    self.items.append(i)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i = self.items[k]
        si, vi = self.shards.index[i]
        s = self.shards.shards[si]
        v = s["views"][vi]
        width = view_width(v)
        rng = np.random.default_rng(k if self.fixed else None)
        n = self.size
        bx0, by0, bx1, by1 = s["boxes"][vi]
        cx, cy = rng.uniform(bx0, bx1), rng.uniform(by0, by1)
        x0 = int(np.clip(cx - n / 2, 0, width - n))
        y0 = int(np.clip(cy - n / 2, 0, H - n))
        scene = int(v[0])
        parts = s["parts"][s["part_offsets"][scene] : s["part_offsets"][scene + 1]]
        q = view_parts(parts, v, rng, (x0, y0, x0 + n, y0 + n), self.vocab)
        cam = np.array([x0, y0, v[13], width, v[14], v[15], *v[23:26]], np.float32)
        a, b = s["image_offsets"][vi : vi + 2]
        frame = np.array(s["images"][a:b]).reshape(H, width, 3)[y0 : y0 + n, x0 : x0 + n]
        # The parts CLIP judges: the biggest on screen whose centre is in the crop.
        z = q[:, 2].clip(min=0.05)
        px = q[:, 0] / z * v[13] + width / 2 - x0
        py = q[:, 1] / z * v[13] + H / 2 - y0
        size = (q[:, 10] + 0.5 * (q[:, 9] + q[:, 11])) * v[13] / z
        inside = (q[:, Q_VALID] > 0) & (px > 8) & (px < n - 8) & (py > 8) & (py < n - 8) & (size > 6)
        order = [j for j in np.argsort(-size) if inside[j]][: self.JUDGED]
        judged = np.full((self.JUDGED, 4), -1, np.float32)
        for r, j in enumerate(order):
            judged[r] = j, px[j], py[j], min(2.6 * size[j], n)
        return (
            torch.from_numpy(q),
            torch.from_numpy(cam),
            torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(random_palette(rng)),
            torch.tensor(WHOLE_IDS[rng.integers(len(WHOLE_IDS))]),
            torch.from_numpy(judged),
        )


class DreamNet(nn.Module):
    """Fields and phrase vectors to the creatures' layer: a small U-Net (full, half and
    quarter size) of 3x3 convolutions, for a straightforward WGSL port. Each part's phrase
    is its CLIP text embedding through a learned projection (`phrase`), blended per pixel
    by the parts' weights. Output [b, 4, h, w]: rgb (a mix of the scheme's colours, shaded)
    and alpha, which starts from the parts' presence but is the network's to reshape."""

    def __init__(self, text_dim=512, phrase_dim=64, c=(48, 96, 128), blocks=2):
        super().__init__()
        self.phrase = nn.Linear(text_dim, phrase_dim)
        self.inp = nn.Conv2d(FIELD_CHANNELS + phrase_dim, c[0], 3, padding=1)
        self.enc = nn.ModuleList([nn.Sequential(*[Block(ch) for _ in range(blocks)]) for ch in c])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(c, c[1:])])
        self.up = nn.ModuleList([nn.Conv2d(b + a, a, 3, padding=1) for a, b in zip(c, c[1:])])
        self.dec = nn.ModuleList([nn.Sequential(*[Block(ch) for _ in range(blocks)]) for ch in c[:-1]])
        # The phrase also shapes every scale: a gain and an offset per channel.
        self.film = nn.ModuleList([nn.Conv2d(phrase_dim, 2 * ch, 3, padding=1) for ch in c])
        for f in self.film:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        self.out = nn.Conv2d(c[0], PALETTE + 2, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def modulate(self, x, emb, i):
        gain, offset = self.film[i](emb).chunk(2, 1)
        return x * (1 + gain) + offset

    def forward(self, feats, weights, text, palette):
        """`text` [B, P, text_dim]: each part's phrase embedding; `palette` [B, PALETTE, 3]."""
        vec = self.phrase(text)
        emb = torch.einsum("bphw,bpe->behw", weights, vec)
        x = self.inp(torch.cat([feats, emb], 1))
        skips = []
        for i, enc in enumerate(self.enc):
            if i:
                x = F.gelu(self.down[i - 1](x))
            x = enc(x)
            skips.append(x)
        embs = [emb, F.avg_pool2d(emb, 2), F.avg_pool2d(emb, 4)]
        x = self.modulate(x, embs[2], 2)
        for i in reversed(range(len(self.dec))):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.dec[i](F.gelu(self.up[i](torch.cat([x, skips[i]], 1))))
            x = self.modulate(x, embs[i], i)
        o = self.out(x).float()
        mix = torch.einsum("bkhw,bkc->bchw", torch.softmax(o[:, :PALETTE] * 3, 1), palette)
        rgb = (mix * (0.35 + 1.3 * torch.sigmoid(o[:, PALETTE : PALETTE + 1] + 0.3)) * lighting(feats)).clamp(0, 1)
        presence = feats[:, :1].float()
        # Soft-clamped to +-10, so alpha never saturates past where losses can reach it.
        alpha = torch.sigmoid(10 * torch.tanh((o[:, PALETTE + 1 :] + 6 * presence - 3) / 10))
        return torch.cat([rgb, alpha], 1)
