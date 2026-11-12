//! People's hair (`person`), made on demand from a few dials: a style, how long it falls, how
//! far the fringe comes down, its volume, where it parts and how messy it is, and its colour.
//! The mesh is in the head's own frame, in units of the head's half sizes (the head is the
//! unit sphere): +y up, +z the way the face looks, so it fits any head it is put on.
//!
//! Every style starts from a cap over the skull down to a hairline round the face, ears and
//! nape. A fringe hangs over the forehead; a bob or long hair falls from the hairline round
//! the sides and back; a ponytail or bun is gathered at the back.

use crate::world::rng::Rng;
use crate::world::tex::{Texture, SIZE};
use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Style {
    Bald,
    /// Close to the skull.
    Crop,
    /// Falling to the jaw.
    Bob,
    /// Falling past the shoulders.
    Long,
    Ponytail,
    Bun,
}

pub const STYLES: [Style; 6] = [Style::Bald, Style::Crop, Style::Bob, Style::Long, Style::Ponytail, Style::Bun];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hair {
    pub style: Style,
    /// 0..1: how far a bob, long hair or a ponytail falls.
    pub length: f32,
    /// 0..1: how far the fringe comes down the forehead (0, none: the hairline shows).
    pub fringe: f32,
    /// 0..1: how far it stands off the head.
    pub volume: f32,
    /// -1..1: where the fringe parts, from the head's -x side to its +x; 0 straight across.
    pub parting: f32,
    /// 0..1: ragged ends and tufts.
    pub messy: f32,
    pub colour: [f32; 3],
}

impl Hair {
    pub fn random(rng: &mut Rng) -> Hair {
        let colour = rng.pick(&[
            [0.12, 0.09, 0.07],
            [0.3, 0.19, 0.1],
            [0.5, 0.33, 0.17],
            [0.75, 0.6, 0.35],
            [0.55, 0.22, 0.1],
            [0.6, 0.6, 0.58],
            [0.85, 0.83, 0.78],
            [0.2, 0.15, 0.12],
        ]);
        let style = rng.pick(&STYLES);
        Hair {
            style,
            length: rng.f(),
            fringe: if rng.chance(0.6) { rng.fr(0.2, 1.0) } else { 0.0 },
            volume: rng.fr(0.0, 0.8),
            parting: if rng.chance(0.5) { rng.fr(-0.8, 0.8) } else { 0.0 },
            messy: rng.fr(0.0, 0.7),
            colour,
        }
    }
}

/// One triangle of hair: corners in the head's frame (unit head) with their texture
/// coordinates (0..1), and the way out at each.
#[derive(Clone, Copy, Debug)]
pub struct HairTri {
    pub p: [[f32; 3]; 3],
    pub n: [[f32; 3]; 3],
    pub uv: [[f32; 2]; 3],
}

/// Around the head: 0 the middle of the face, turning toward +x.
fn dir(e: f32, a: f32) -> [f32; 3] {
    [e.cos() * a.sin(), e.sin(), e.cos() * a.cos()]
}

/// The hairline's height (elevation, radians) at angle `a` round the head: up on the
/// forehead, down to the top of the ears at the sides, lower at the nape.
fn hairline(a: f32) -> f32 {
    let back = (1.0 - a.cos()) / 2.0;
    let forehead = 0.55;
    let ears = 0.05;
    let nape = -0.55;
    let side = (a.sin()).abs();
    let e = forehead + (nape - forehead) * back;
    e + (ears - e) * side.powi(3) * 0.7
}

/// Fixed noise in -1..1 from a seed and a few integers.
fn jitter(seed: u64, i: i32, j: i32) -> f32 {
    let h = crate::world::rng::hash(seed ^ ((i as u64) << 20) ^ ((j as u64) << 40) ^ 0x4A11);
    (h & 0xFFFF) as f32 / 32767.5 - 1.0
}

/// The hair's mesh. `seed` picks its tufts.
pub fn mesh(h: &Hair, seed: u64) -> Vec<HairTri> {
    let mut out = Vec::new();
    if h.style == Style::Bald {
        return out;
    }
    let lift = match h.style {
        Style::Crop => 1.04 + 0.05 * h.volume,
        _ => 1.07 + 0.15 * h.volume,
    };
    let messy = h.messy.clamp(0.0, 1.0);
    const AROUND: i32 = 14;
    const ROWS: i32 = 4;
    // The cap: rows from the hairline up to the crown.
    let cap = |i: i32, r: i32| {
        let a = i as f32 / AROUND as f32 * TAU;
        let e0 = hairline(a);
        let k = r as f32 / ROWS as f32;
        let e = e0 + (FRAC_PI_2 - e0) * k;
        let tuft = if r < ROWS { 1.0 + 0.06 * messy * jitter(seed, i.rem_euclid(AROUND), r) } else { 1.0 };
        let d = dir(e, a);
        (d.map(|c| c * lift * tuft), d)
    };
    for i in 0..AROUND {
        for r in 0..ROWS {
            let corner = |di: i32, dr: i32| {
                let (p, n) = cap(i + di, r + dr);
                (p, n, [(i % 3 + di) as f32 / 3.0, 1.0 - (r + dr) as f32 / ROWS as f32])
            };
            quad(&mut out, [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)]);
        }
    }

    // A fringe: a flap over the forehead from the hairline, its lower edge ragged and
    // tilted away from the parting.
    if h.fringe > 0.0 {
        const ACROSS: i32 = 8;
        let half = 1.1;
        for i in 0..ACROSS {
            let corner = |di: i32, low: bool| {
                let s = (i + di) as f32 / ACROSS as f32 * 2.0 - 1.0;
                let a = s * half;
                let top = hairline(a) + 0.05;
                let drop = h.fringe * 0.42 * (1.0 - 0.35 * (s - h.parting).abs()) * (1.0 + 0.3 * messy * jitter(seed, i + di, 99));
                let e = if low { top - drop.max(0.05) } else { top };
                let d = dir(e, a);
                let r = lift + if low { 0.03 } else { 0.0 };
                (d.map(|c| c * r), d, [(i + di) as f32 / ACROSS as f32, if low { 1.0 } else { 0.0 }])
            };
            quad(&mut out, [corner(0, false), corner(1, false), corner(0, true), corner(1, true)]);
        }
    }

    // Hair falling from the hairline round the sides and back.
    let fall = match h.style {
        Style::Bob => Some(0.75 + 0.35 * h.length),
        Style::Long => Some(1.4 + 2.2 * h.length),
        _ => None,
    };
    if let Some(fall) = fall {
        // From beside one cheek round the back to the other.
        let open = 1.15;
        const ROUND: i32 = 12;
        const DOWN: i32 = 3;
        for i in 0..ROUND {
            for r in 0..DOWN {
                let corner = |di: i32, dr: i32| {
                    let s = (i + di) as f32 / ROUND as f32;
                    let a = open + s * (TAU - 2.0 * open);
                    let top = dir(hairline(a), a).map(|c| c * lift);
                    let k = (r + dr) as f32 / DOWN as f32;
                    let ragged = if dr + r == DOWN { 0.25 * messy * jitter(seed, i + di, 7) } else { 0.0 };
                    let y = top[1] + (-fall - top[1]) * k + ragged;
                    // It swings out a little as it falls, and long hair down the back.
                    let out_by = lift * (1.0 + 0.1 * k * (1.0 + h.volume)) * a.cos().abs().max(a.sin().abs()).max(0.6);
                    let flat = [a.sin(), 0.0, a.cos()];
                    let behind = if h.style == Style::Long { 0.35 * k * (1.0 - a.cos()) / 2.0 } else { 0.0 };
                    let x = if k == 0.0 { top[0] } else { flat[0] * out_by };
                    let z = if k == 0.0 { top[2] } else { flat[2] * out_by - behind };
                    ([x, y, z], flat, [(i % 3 + di) as f32 / 3.0, k])
                };
                quad(&mut out, [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)]);
            }
        }
    }

    match h.style {
        Style::Ponytail => {
            // Gathered at the back of the head and hanging down behind.
            let len = 1.0 + 2.2 * h.length;
            let root = [0.0, 0.15, -lift];
            let path = |k: f32| [0.0, root[1] - len * k * k.sqrt(), root[2] - 0.35 * (k * PI).sin() - 0.1 * k];
            let radius = |k: f32| 0.24 * (1.0 + 0.4 * h.volume) * (1.0 - 0.6 * k) * if k == 0.0 { 0.7 } else { 1.0 };
            tube(&mut out, path, radius, 5);
        }
        Style::Bun => {
            let c = [0.0, 0.55, -0.75 * lift];
            let r = 0.38 + 0.15 * h.volume;
            tube(&mut out, |k| [c[0], c[1] + r * (1.0 - 2.0 * k), c[2] - 0.2 * (k * PI).sin()], |k| r * (k * PI).sin().max(0.05) * 1.1, 4);
        }
        _ => {}
    }
    out
}

/// Two triangles of a quad given as (position, way out, uv) corners in the order
/// (0,0), (1,0), (0,1), (1,1).
fn quad(out: &mut Vec<HairTri>, q: [([f32; 3], [f32; 3], [f32; 2]); 4]) {
    for [a, b, c] in [[0, 1, 2], [1, 3, 2]] {
        let t = HairTri { p: [q[a].0, q[b].0, q[c].0], n: [q[a].1, q[b].1, q[c].1], uv: [q[a].2, q[b].2, q[c].2] };
        let e1 = [0, 1, 2].map(|i| t.p[1][i] - t.p[0][i]);
        let e2 = [0, 1, 2].map(|i| t.p[2][i] - t.p[0][i]);
        let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        if cr.iter().map(|c| c * c).sum::<f32>() > 1e-8 {
            out.push(t);
        }
    }
}

/// A tube along `path` (k 0..1) of `radius(k)`, six sides, `rows` long, closed at its ends.
fn tube(out: &mut Vec<HairTri>, path: impl Fn(f32) -> [f32; 3], radius: impl Fn(f32) -> f32, rows: i32) {
    const SIDES: i32 = 6;
    let ring = |r: i32, s: i32| {
        let k = r as f32 / rows as f32;
        let c = path(k);
        let a = s as f32 / SIDES as f32 * TAU;
        let n = [a.cos(), 0.0, a.sin()];
        let rad = radius(k);
        ([c[0] + n[0] * rad, c[1], c[2] + n[2] * rad], n, [s as f32 / SIDES as f32, k])
    };
    for r in 0..rows {
        for s in 0..SIDES {
            quad(out, [ring(r, s), ring(r, s + 1), ring(r + 1, s), ring(r + 1, s + 1)]);
        }
    }
}

/// The hair's texture: strands running down, made to be tinted with the hair's colour
/// (neutral is mid-grey-white, as the body's skin).
pub fn texture() -> Texture {
    let mut pal = [[0u8; 3]; 16];
    for (i, c) in pal.iter_mut().enumerate() {
        let v = (60.0 + 195.0 * i as f32 / 15.0) as u8;
        *c = [v, v, v];
    }
    let mut idx = vec![0u8; SIZE * SIZE];
    for x in 0..SIZE {
        let strand = 0.5 + 0.5 * jitter(11, x as i32, 0);
        for y in 0..SIZE {
            let v = 0.55 + 0.3 * strand + 0.12 * crate::world::tex::value_noise(x as f32 * 0.5, y as f32 / 8.0, 32, 5) - 0.15 * (x % 4 == 0) as u8 as f32;
            idx[y * SIZE + x] = (v.clamp(0.0, 1.0) * 15.0).round() as u8;
        }
    }
    Texture { idx, pal }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lowest(m: &[HairTri]) -> f32 {
        m.iter().flat_map(|t| t.p).map(|p| p[1]).fold(f32::MAX, f32::min)
    }

    #[test]
    fn styles_fall_as_long_as_they_are_told_and_keep_off_the_face() {
        let base = Hair { style: Style::Crop, length: 0.5, fringe: 0.0, volume: 0.3, parting: 0.0, messy: 0.3, colour: [0.3; 3] };
        assert!(mesh(&Hair { style: Style::Bald, ..base }, 1).is_empty());
        let crop = mesh(&base, 1);
        assert!(!crop.is_empty() && lowest(&crop) > -0.7, "a crop stops at the nape");
        let bob = |l| lowest(&mesh(&Hair { style: Style::Bob, length: l, ..base }, 1));
        let long = |l| lowest(&mesh(&Hair { style: Style::Long, length: l, ..base }, 1));
        assert!(bob(1.0) < bob(0.0) && long(0.0) < bob(1.0) && long(1.0) < long(0.0) - 1.0);
        assert!(lowest(&mesh(&Hair { style: Style::Ponytail, length: 1.0, ..base }, 1)) < -2.0);
        // Nothing but a fringe comes over the middle of the face (the eyes, below the brows).
        for style in STYLES {
            let m = mesh(&Hair { style, length: 1.0, fringe: 0.0, ..base }, 3);
            for t in &m {
                for p in t.p {
                    let on_face = p[2] > 0.5 && p[0].abs() < 0.55 && p[1] < 0.3 && p[1] > -0.8;
                    assert!(!on_face, "{style:?} covers the face at {p:?}");
                }
            }
        }
        // The fringe comes further down when asked.
        let fringe = |f| mesh(&Hair { fringe: f, ..base }, 1).iter().flat_map(|t| t.p).filter(|p| p[2] > 0.6 && p[0].abs() < 0.3).map(|p| p[1]).fold(f32::MAX, f32::min);
        assert!(fringe(1.0) < fringe(0.3) - 0.1, "{} {}", fringe(1.0), fringe(0.3));
    }
}
