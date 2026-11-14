//! Network inputs for one view: a port of `tri_tokens`, `organ_tokens` and `cam_token` in
//! train/data.py, reading the same per-triangle, per-organ and per-view features that the
//! training data stores in `tris.npy`, `turning.npy`, `organs.npy` and `views.npy`. Keep the
//! two in step.

use crate::neural::features::TRI_FEATURES as RAW;
use crate::world::dreamed::{ORGAN_FEATURES as RAW_ORGAN, TURNING_FEATURES};

pub const MAX_TRIS: usize = 512;
/// Floats per triangle token and in the camera token (data.py TRI_FEATURES, CAM_FEATURES).
pub const TRI_FEATURES: usize = 61;
/// Of those, how many models from before triangles were told how they turn take.
pub const TRI_FEATURES_PLAIN: usize = 46;
pub const CAM_FEATURES: usize = 11;
/// Floats per organ token (data.py ORGAN_FEATURES).
pub const ORGAN_FEATURES: usize = 23;

/// A sculpted level's view, as far as the network is told: how far its dial has turned, its
/// triangles' turning (`dreamed::turning_features`, one row per raw triangle) and the organs
/// in sight (`dreamed::organs`).
pub struct Turned<'a> {
    pub amount: f32,
    pub turning: &'a [f32],
    pub organs: &'a [f32],
}

pub struct Tokens {
    /// [n, TRI_FEATURES], nearest triangle first.
    pub feats: Vec<f32>,
    /// [n] texture ids.
    pub tex: Vec<i64>,
    /// [n] index of each token's triangle in the scene: stable from frame to frame.
    pub ids: Vec<usize>,
    pub cam: [f32; CAM_FEATURES],
    /// The frame's width in pixels (a multiple of 16; 240 high).
    pub width: u32,
    /// [o, ORGAN_FEATURES] and [o] kinds: eyes, sphincters and veins in sight (none unless sculpted).
    pub organs: Vec<f32>,
    pub organ_kinds: Vec<i64>,
}

impl Tokens {
    pub fn len(&self) -> usize {
        self.tex.len()
    }

    /// Keep only the tokens for which `keep(id)` is true.
    pub fn retain(&mut self, keep: impl Fn(usize) -> bool) {
        let keep: Vec<bool> = self.ids.iter().map(|&id| keep(id)).collect();
        let mut i = 0;
        self.feats.retain(|_| {
            let k = keep[i / TRI_FEATURES];
            i += 1;
            k
        });
        let mut it = keep.iter();
        self.tex.retain(|_| *it.next().unwrap());
        let mut it = keep.iter();
        self.ids.retain(|_| *it.next().unwrap());
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// How many of the triangles `tris` view `v` would see, before `MAX_TRIS` cuts them off.
pub fn visible(tris: &[f32], v: &[f32]) -> usize {
    cull(tris, v).len()
}

/// (min z, camera-space vertices, normal, raw index) for every triangle that
/// survives culling, in the scene's order.
#[allow(clippy::type_complexity)]
fn cull(tris: &[f32], v: &[f32]) -> Vec<(f32, [[f32; 3]; 3], [f32; 3], usize)> {
    let pos = [v[1], v[2], v[3]];
    let r = [[v[4], v[5], v[6]], [v[7], v[8], v[9]], [v[10], v[11], v[12]]];
    let h = v[13];
    let fog_far = v[15];
    let near = h / 2.0 / 256.0;
    // Half the frame's width and height over the lens, with a margin.
    let kx = width(v) / 2.0 / h * 1.05;
    let ky = 120.0 / h * 1.05;

    let mut kept = Vec::new();
    for (i, t) in tris.chunks_exact(RAW).enumerate() {
        let p = [0, 1, 2].map(|j| {
            let d = sub([t[3 * j], t[3 * j + 1], t[3 * j + 2]], pos);
            [dot(r[0], d), dot(r[1], d), dot(r[2], d)]
        });
        let z = p.map(|q| q[2]);
        let zmin = z[0].min(z[1]).min(z[2]);
        if !z.iter().any(|&z| z > near) || zmin >= fog_far {
            continue;
        }
        let s = p.map(|q| {
            let zc = q[2].max(1e-3);
            (q[0] / zc, q[1] / zc, q[2] > 0.0)
        });
        let all = |f: &dyn Fn((f32, f32, bool)) -> bool| s.iter().all(|&e| f(e));
        if all(&|(x, _, a)| x > kx && a)
            || all(&|(x, _, a)| x < -kx && a)
            || all(&|(_, y, a)| y > ky && a)
            || all(&|(_, y, a)| y < -ky && a)
        {
            continue;
        }
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        if dot(n, p[0].map(|c| -c)) <= 0.0 {
            continue;
        }
        kept.push((zmin, p, n, i));
    }
    kept
}

/// Camera-space triangle tokens for view `v` over a scene's raw triangle features `tris`.
pub fn tokens(tris: &[f32], v: &[f32]) -> Tokens {
    tokens_turned(tris, v, None)
}

/// The same for a sculpted level, `turned` along its dial: each triangle's strangeness and
/// its corners' become how far along each is, and the organs come too.
pub fn tokens_turned(tris: &[f32], v: &[f32], turned: Option<&Turned>) -> Tokens {
    let h = v[13];
    let near = h / 2.0 / 256.0;
    let mut kept = cull(tris, v);
    kept.sort_by(|a, b| a.0.total_cmp(&b.0));
    kept.truncate(MAX_TRIS);

    let mut feats = Vec::with_capacity(kept.len() * TRI_FEATURES);
    let mut tex = Vec::with_capacity(kept.len());
    let ids = kept.iter().map(|k| k.3).collect();
    let zero = [0.0; TURNING_FEATURES];
    let turning_of = |i: usize| match turned {
        Some(u) => u.turning.get(i * TURNING_FEATURES..(i + 1) * TURNING_FEATURES).unwrap_or(&zero),
        None => &zero[..],
    };
    let amount = turned.map_or(0.0, |u| u.amount);
    // As data.py: once any triangle in view turns, strangeness is how far along each is.
    let along = amount != 0.0 || kept.iter().any(|k| turning_of(k.3).iter().any(|&x| x != 0.0));
    for (_, p, n, i) in &kept {
        let t = &tris[i * RAW..(i + 1) * RAW];
        feats.extend(p.iter().flatten().map(|c| c / 8.0));
        let zc = p.map(|q| q[2].max(near * 0.5));
        for j in 0..3 {
            feats.push((p[j][0] / zc[j] * h / 160.0).clamp(-4.0, 4.0));
            feats.push((p[j][1] / zc[j] * h / 160.0).clamp(-4.0, 4.0));
        }
        feats.extend(zc.map(|z| (near / z).clamp(0.0, 2.0)));
        let l = dot(*n, *n).sqrt().max(1e-6);
        feats.extend(n.map(|c| c / l));
        feats.extend_from_slice(&t[9..15]);
        feats.extend(t[16..34].iter().map(|c| c * 2.0));
        let turn = turning_of(*i);
        if along {
            feats.push(amount * (turn[0] + turn[1] + turn[2]) / 3.0);
        } else {
            feats.push(t[34]);
        }
        feats.extend(turn[0..3].iter().map(|c| amount * c));
        feats.extend_from_slice(&turn[3..9]);
        feats.extend(turn[9..15].iter().map(|c| c * 2.0));
        tex.push(t[15] as i64);
    }
    let w = width(v);
    let cam = [h / 160.0, v[14] / 8.0, v[15] / 8.0, v[19], v[20], v[21], v[22] / 4.0, v[23], v[24], v[25], w / 320.0];
    let (organs, organ_kinds) = turned.map_or((Vec::new(), Vec::new()), |u| organ_tokens(u.organs, v));
    Tokens { feats, tex, ids, cam, width: w as u32, organs, organ_kinds }
}

/// Camera-space features of a view's organs (`dreamed::organs` rows, which chose them, so
/// none is dropped): data.py `organ_tokens`.
pub fn organ_tokens(organs: &[f32], v: &[f32]) -> (Vec<f32>, Vec<i64>) {
    let pos = [v[1], v[2], v[3]];
    let r = [[v[4], v[5], v[6]], [v[7], v[8], v[9]], [v[10], v[11], v[12]]];
    let h = v[13];
    let near = h / 2.0 / 256.0;
    let turn = |d: [f32; 3]| [dot(r[0], d), dot(r[1], d), dot(r[2], d)];
    let mut feats = Vec::with_capacity(organs.len() / RAW_ORGAN * ORGAN_FEATURES);
    let mut kinds = Vec::with_capacity(organs.len() / RAW_ORGAN);
    for o in organs.chunks_exact(RAW_ORGAN) {
        let at = [o[1], o[2], o[3]];
        let x = [o[7], o[8], o[9]];
        let y = [o[10], o[11], o[12]];
        let (lo, hi) = [[-1.0f32, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]].iter().fold(
            ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]),
            |(lo, hi), k| {
                let c = turn(sub([0, 1, 2].map(|i| at[i] + k[0] * o[13] * x[i] + k[1] * o[14] * y[i]), pos));
                let zc = c[2].max(near * 0.5);
                let s = [0, 1].map(|i| (c[i] / zc * h / 160.0).clamp(-4.0, 4.0));
                ([lo[0].min(s[0]), lo[1].min(s[1])], [hi[0].max(s[0]), hi[1].max(s[1])])
            },
        );
        let rel = turn(sub(at, pos));
        feats.extend(rel.map(|c| c / 8.0));
        feats.extend(turn([o[4], o[5], o[6]]));
        feats.extend(turn(x));
        feats.extend(turn(y));
        feats.extend([lo[0], lo[1], hi[0], hi[1]]);
        feats.push((near / rel[2].max(near * 0.5)).clamp(0.0, 2.0));
        feats.extend([o[13] * 4.0, o[14] * 4.0]);
        feats.extend_from_slice(&o[15..19]);
        kinds.push(o[0] as i64);
    }
    (feats, kinds)
}

/// The view's frame width in pixels (`views.npy` column 26).
fn width(v: &[f32]) -> f32 {
    v[26]
}
