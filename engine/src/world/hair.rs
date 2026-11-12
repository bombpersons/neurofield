//! People's hair (`person`), made on demand from a few dials: a style, how long it falls, how
//! far the fringe comes down, its volume, where it parts, how spiky its ends are, a cowlick,
//! and its colour. The mesh is in the head's own frame, in units of the head's half sizes
//! (the head is the unit sphere): +y up, +z the way the face looks, so it fits any head.
//!
//! It is cartoonish, chunky hair in pointed locks: a cap over the skull down to a hairline
//! round the face, ears and nape; a fringe of locks over the forehead; locks down beside the
//! cheeks; a bob or long hair whose hem is locks too; or it gathered into a ponytail,
//! twintails, a bun or two, or spikes. Its texture has the glossy band of light such hair
//! has.

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
    /// One bun at the back.
    Bun,
    /// Two buns up on top.
    Buns,
    /// Two tails, one each side.
    Twintails,
    /// Spikes standing out all over.
    Spiky,
}

pub const STYLES: [Style; 9] = [Style::Bald, Style::Crop, Style::Bob, Style::Long, Style::Ponytail, Style::Bun, Style::Buns, Style::Twintails, Style::Spiky];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hair {
    pub style: Style,
    /// 0..1: how far a bob, long hair, a tail or spikes go.
    pub length: f32,
    /// 0..1: how far the fringe comes down the forehead (0, none: the hairline shows).
    pub fringe: f32,
    /// 0..1: how far it stands off the head.
    pub volume: f32,
    /// -1..1: where the fringe parts, from the head's -x side to its +x; 0 straight across.
    pub parting: f32,
    /// 0..1: how ragged and pointed its locks' ends are.
    pub messy: f32,
    /// A lock standing up from the crown.
    #[serde(default)]
    pub ahoge: bool,
    pub colour: [f32; 3],
}

impl Hair {
    pub fn random(rng: &mut Rng) -> Hair {
        let natural = [[0.15, 0.11, 0.1], [0.35, 0.22, 0.13], [0.6, 0.4, 0.2], [0.95, 0.82, 0.5], [0.75, 0.3, 0.15], [0.88, 0.88, 0.92]];
        let bright = [[0.98, 0.6, 0.75], [0.72, 0.62, 0.98], [0.5, 0.75, 1.0], [0.55, 0.92, 0.78], [1.0, 0.62, 0.3], [0.9, 0.28, 0.3], [0.35, 0.4, 0.75]];
        let colour = if rng.chance(0.45) { rng.pick(&bright) } else { rng.pick(&natural) };
        let style = rng.pick(&[Style::Crop, Style::Bob, Style::Bob, Style::Long, Style::Long, Style::Ponytail, Style::Bun, Style::Buns, Style::Twintails, Style::Spiky]);
        Hair {
            style,
            length: rng.f(),
            fringe: if rng.chance(0.85) { rng.fr(0.4, 1.0) } else { 0.0 },
            volume: rng.fr(0.2, 1.0),
            parting: if rng.chance(0.5) { rng.fr(-0.8, 0.8) } else { 0.0 },
            messy: rng.fr(0.2, 1.0),
            ahoge: rng.chance(0.3),
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
    let forehead = 0.6;
    let ears = 0.1;
    let nape = -0.5;
    let e = forehead + (nape - forehead) * back;
    e + (ears - e) * a.sin().abs().powi(3) * 0.7
}

/// Fixed noise in -1..1 from a seed and a few integers.
fn jitter(seed: u64, i: i32, j: i32) -> f32 {
    let h = crate::world::rng::hash(seed ^ ((i as u64) << 20) ^ ((j as u64) << 40) ^ 0x4A11);
    (h & 0xFFFF) as f32 / 32767.5 - 1.0
}

type Corner = ([f32; 3], [f32; 3], [f32; 2]);

/// The hair's mesh. `seed` picks the lengths of its locks.
pub fn mesh(h: &Hair, seed: u64) -> Vec<HairTri> {
    let mut out = Vec::new();
    if h.style == Style::Bald {
        if h.ahoge {
            ahoge(&mut out, 1.0);
        }
        return out;
    }
    let lift = match h.style {
        Style::Crop => 1.05 + 0.06 * h.volume,
        _ => 1.1 + 0.18 * h.volume,
    };
    let spiky = h.messy.clamp(0.0, 1.0);
    const AROUND: i32 = 14;
    const ROWS: i32 = 4;
    // The cap: rows from the hairline up to the crown.
    let cap = |i: i32, r: i32| -> Corner {
        let a = i as f32 / AROUND as f32 * TAU;
        let e0 = hairline(a);
        let k = r as f32 / ROWS as f32;
        let e = e0 + (FRAC_PI_2 - e0) * k;
        let d = dir(e, a);
        (d.map(|c| c * lift), d, [(i % 3) as f32 / 3.0, 1.0 - k])
    };
    for i in 0..AROUND {
        for r in 0..ROWS {
            let uv = |di: i32, dr: i32| [(i % 3 + di) as f32 / 3.0, 1.0 - (r + dr) as f32 / ROWS as f32];
            let c = |di: i32, dr: i32| {
                let (p, n, _) = cap(i + di, r + dr);
                (p, n, uv(di, dr))
            };
            quad(&mut out, [c(0, 0), c(1, 0), c(0, 1), c(1, 1)]);
        }
    }

    // The fringe: a row of pointed locks over the forehead, longer away from the parting.
    if h.fringe > 0.0 {
        const LOCKS: i32 = 7;
        let half = 1.15;
        for i in 0..LOCKS {
            let at = |s: f32, drop: f32, out_by: f32| -> Corner {
                let a = s * half;
                let e = hairline(a) + 0.12 - drop;
                let d = dir(e, a);
                (d.map(|c| c * (lift + out_by)), d, [(s + 1.0) / 2.0, drop / 0.6])
            };
            let (s0, s1) = (i as f32 / LOCKS as f32 * 2.0 - 1.0, (i + 1) as f32 / LOCKS as f32 * 2.0 - 1.0);
            let mid = (s0 + s1) / 2.0;
            // Down to the brows at most, clear of the eyes.
            let reach = h.fringe * 0.42 * (1.0 - 0.3 * (mid - h.parting).abs()) * (1.0 + 0.2 * spiky * jitter(seed, i, 99));
            let shoulder = reach * (0.55 - 0.25 * spiky);
            // A band from the hairline down to the lock's shoulders, then its point.
            quad(&mut out, [at(s0, 0.0, 0.0), at(s1, 0.0, 0.0), at(s0, shoulder, 0.02), at(s1, shoulder, 0.02)]);
            tri(&mut out, [at(s0, shoulder, 0.02), at(s1, shoulder, 0.02), at(mid, reach, 0.04)]);
        }
    }

    // Locks down beside the cheeks, framing the face.
    let side_locks = match h.style {
        Style::Bob | Style::Twintails => Some(0.9 + 0.4 * h.length),
        Style::Long => Some(1.1 + 0.6 * h.length),
        Style::Ponytail | Style::Buns | Style::Bun => (h.fringe > 0.0).then_some(0.7),
        _ => None,
    };
    if let Some(fall) = side_locks {
        for side in [-1.0f32, 1.0] {
            let a = side * 1.2;
            let top = |da: f32| -> Corner {
                let d = dir(hairline(a + da) + 0.1, a + da);
                (d.map(|c| c * (lift + 0.02)), d, [0.5 + da, 0.0])
            };
            let flat = [a.sin(), 0.0, a.cos()];
            let down = |da: f32, y: f32, u: f32| -> Corner {
                let r = lift + 0.05;
                ([(a + da).sin() * r, y, (a + da).cos() * r], flat, [u, (top(0.0).0[1] - y) / (fall + 1.0)])
            };
            let mid_y = -0.35 * fall;
            quad(&mut out, [top(-0.22), top(0.22), down(-0.2, mid_y, 0.3), down(0.2, mid_y, 0.7)]);
            tri(&mut out, [down(-0.2, mid_y, 0.3), down(0.2, mid_y, 0.7), down(0.02 * side, -fall, 0.5)]);
        }
    }

    // Hair falling round the back and sides, its hem a row of points.
    let fall = match h.style {
        Style::Bob => Some(0.8 + 0.35 * h.length),
        Style::Long => Some(1.5 + 2.2 * h.length),
        _ => None,
    };
    if let Some(fall) = fall {
        let open = 1.35;
        const ROUND: i32 = 12;
        for i in 0..ROUND {
            let at = |s: f32, k: f32| -> Corner {
                let a = open + s * (TAU - 2.0 * open);
                let top = dir(hairline(a), a).map(|c| c * lift);
                let y = top[1] + (-fall - top[1]) * k;
                // It swings out a little as it falls, and long hair lies down the back.
                let out_by = lift * (1.0 + 0.12 * k * (1.0 + h.volume));
                let flat = [a.sin(), 0.0, a.cos()];
                let behind = if h.style == Style::Long { 0.4 * k * (1.0 - a.cos()) / 2.0 } else { 0.0 };
                let p = if k == 0.0 { top } else { [flat[0] * out_by, y, flat[2] * out_by - behind] };
                (p, flat, [s * ROUND as f32 % 3.0 / 3.0, k])
            };
            let (s0, s1) = (i as f32 / ROUND as f32, (i + 1) as f32 / ROUND as f32);
            let shoulder = 0.78 - 0.2 * spiky;
            let tip = 1.0 + 0.12 * spiky * jitter(seed, i, 7);
            quad(&mut out, [at(s0, 0.0), at(s1, 0.0), at(s0, 0.4), at(s1, 0.4)]);
            quad(&mut out, [at(s0, 0.4), at(s1, 0.4), at(s0, shoulder), at(s1, shoulder)]);
            tri(&mut out, [at(s0, shoulder), at(s1, shoulder), at((s0 + s1) / 2.0, tip)]);
        }
    }

    let len = 0.6 + 2.0 * h.length;
    let girth = 1.0 + 0.5 * h.volume;
    match h.style {
        Style::Ponytail => {
            // Gathered at the back of the head, swelling and hanging down to a point.
            let path = |k: f32| [0.0, 0.25 - len * k * k.sqrt(), -lift - 0.1 - 0.45 * (k * PI).sin()];
            tube(&mut out, path, |k| 0.3 * girth * (k * 2.5).clamp(0.4, 1.0) * (1.0 - k).powf(0.7), 6);
        }
        Style::Twintails => {
            for side in [-1.0f32, 1.0] {
                let root = [side * 0.85 * lift, 0.45, -0.45 * lift];
                let fall = 1.2 + 2.2 * h.length;
                let path = move |k: f32| [root[0] + side * (0.35 * (k * PI).sin() + 0.15 * k), root[1] - fall * k, root[2] - 0.2 * k];
                tube(&mut out, path, |k| 0.26 * girth * (k * 3.0).clamp(0.45, 1.0) * (1.0 - k).powf(0.7), 6);
                // A tie at the root.
                tube(&mut out, move |k| [root[0] + side * 0.05 * k, root[1] + 0.12 - 0.24 * k, root[2]], |k| 0.16 * (k * PI).sin().max(0.3), 2);
            }
        }
        Style::Bun => {
            let c = [0.0, 0.5, -0.85 * lift];
            let r = 0.42 + 0.15 * h.volume;
            tube(&mut out, move |k| [c[0], c[1] + r * (1.0 - 2.0 * k), c[2] - 0.25 * (k * PI).sin()], move |k| r * (k * PI).sin().max(0.05) * 1.1, 4);
        }
        Style::Buns => {
            for side in [-1.0f32, 1.0] {
                let c = [side * 0.6 * lift, 0.95 * lift, -0.1];
                let r = 0.3 + 0.1 * h.volume;
                tube(&mut out, move |k| [c[0] + side * 0.15 * (k * PI).sin(), c[1] + r * (1.0 - 2.0 * k), c[2]], move |k| r * (k * PI).sin().max(0.05), 4);
            }
        }
        Style::Spiky => {
            // Cones out from the crown and the back, none over the face.
            let n = 9;
            for i in 0..n {
                let a = PI * 0.35 + (i as f32 / (n - 1) as f32) * PI * 1.3;
                let e = 0.55 + 0.6 * ((i * 7 % 5) as f32 / 4.0);
                let a = if i % 2 == 0 { a } else { -a };
                let d = dir(e, a);
                let base = d.map(|c| c * lift * 0.92);
                let long = (0.45 + 0.6 * h.length) * (1.0 + 0.3 * jitter(seed, i, 3));
                let tip = [base[0] + d[0] * long, base[1] + d[1] * long + 0.15, base[2] + d[2] * long - 0.1];
                tube(&mut out, move |k| [base[0] + (tip[0] - base[0]) * k, base[1] + (tip[1] - base[1]) * k, base[2] + (tip[2] - base[2]) * k], |k| 0.26 * (1.0 - k), 2);
            }
        }
        _ => {}
    }
    if h.ahoge {
        ahoge(&mut out, lift);
    }
    out
}

/// A lock standing up from the crown and curling forward.
fn ahoge(out: &mut Vec<HairTri>, lift: f32) {
    let path = |k: f32| [0.05 * k, lift + 0.55 * (k * 1.6).sin(), 0.05 + 0.45 * k * k];
    let n = 4;
    for i in 0..n {
        let (k0, k1) = (i as f32 / n as f32, (i + 1) as f32 / n as f32);
        let w = |k: f32| 0.14 * (1.0 - k);
        let c = |k: f32, s: f32| -> Corner {
            let p = path(k);
            ([p[0] + s * w(k), p[1], p[2]], [0.0, 0.3, 1.0], [0.5 + s, k])
        };
        quad(out, [c(k0, -1.0), c(k0, 1.0), c(k1, -1.0), c(k1, 1.0)]);
    }
}

fn push(out: &mut Vec<HairTri>, a: Corner, b: Corner, c: Corner) {
    let t = HairTri { p: [a.0, b.0, c.0], n: [a.1, b.1, c.1], uv: [a.2, b.2, c.2] };
    let e1 = [0, 1, 2].map(|i| t.p[1][i] - t.p[0][i]);
    let e2 = [0, 1, 2].map(|i| t.p[2][i] - t.p[0][i]);
    let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    if cr.iter().map(|c| c * c).sum::<f32>() > 1e-8 {
        out.push(t);
    }
}

fn tri(out: &mut Vec<HairTri>, c: [Corner; 3]) {
    push(out, c[0], c[1], c[2]);
}

/// Two triangles of a quad given as corners in the order (0,0), (1,0), (0,1), (1,1).
fn quad(out: &mut Vec<HairTri>, q: [Corner; 4]) {
    push(out, q[0], q[1], q[2]);
    push(out, q[1], q[3], q[2]);
}

/// A tube along `path` (k 0..1) of `radius(k)`, six sides, `rows` long.
fn tube(out: &mut Vec<HairTri>, path: impl Fn(f32) -> [f32; 3], radius: impl Fn(f32) -> f32, rows: i32) {
    const SIDES: i32 = 6;
    let ring = |r: i32, s: i32| -> Corner {
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

/// The hair's texture, to be tinted with the hair's colour: strands running down, darker
/// toward the ends, and a glossy band of light across near the top.
pub fn texture() -> Texture {
    let mut pal = [[0u8; 3]; 16];
    for (i, c) in pal.iter_mut().enumerate() {
        let v = (90.0 + 165.0 * (i as f32 / 15.0).powf(0.8)) as u8;
        *c = [v, v, v];
    }
    let mut idx = vec![0u8; SIZE * SIZE];
    for x in 0..SIZE {
        let strand = 0.5 + 0.5 * jitter(11, x as i32, 0);
        // The band dips and rises across the strands, in points.
        let band = 16.0 + 3.0 * ((x % 8) as f32 - 3.5).abs() / 3.5;
        for y in 0..SIZE {
            let yf = y as f32;
            let mut v = 0.62 + 0.08 * strand - 0.25 * (yf / SIZE as f32).powi(2) - 0.08 * (x % 6 == 0) as u8 as f32;
            if (yf - band).abs() < 2.0 {
                v = 0.97;
            } else if (yf - band).abs() < 3.5 {
                v += 0.12;
            }
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
        let base = Hair { style: Style::Crop, length: 0.5, fringe: 0.0, volume: 0.3, parting: 0.0, messy: 0.3, ahoge: false, colour: [0.3; 3] };
        assert!(mesh(&Hair { style: Style::Bald, ..base }, 1).is_empty());
        assert!(!mesh(&Hair { style: Style::Bald, ahoge: true, ..base }, 1).is_empty());
        let crop = mesh(&base, 1);
        assert!(!crop.is_empty() && lowest(&crop) > -0.7, "a crop stops at the nape");
        let low = |style, l| lowest(&mesh(&Hair { style, length: l, ..base }, 1));
        assert!(low(Style::Bob, 1.0) < low(Style::Bob, 0.0));
        assert!(low(Style::Long, 0.0) < low(Style::Bob, 1.0) && low(Style::Long, 1.0) < low(Style::Long, 0.0) - 1.0);
        for style in [Style::Ponytail, Style::Twintails] {
            assert!(low(style, 1.0) < -1.5 && low(style, 1.0) < low(style, 0.0) - 1.0, "{style:?}");
        }
        // Nothing but a fringe comes over the middle of the face (the eyes, below the brows).
        for style in STYLES {
            for seed in 0..5 {
                let m = mesh(&Hair { style, length: 1.0, fringe: 0.0, volume: 1.0, messy: 1.0, ahoge: true, ..base }, seed);
                for t in &m {
                    for p in t.p {
                        let on_face = p[2] > 0.5 && p[0].abs() < 0.6 && p[1] < 0.3 && p[1] > -0.8;
                        assert!(!on_face, "{style:?} covers the face at {p:?}");
                    }
                }
            }
        }
        // The fringe comes further down when asked, but stops above the eyes.
        let fringe = |f| mesh(&Hair { fringe: f, ..base }, 1).iter().flat_map(|t| t.p).filter(|p| p[2] > 0.6 && p[0].abs() < 0.3).map(|p| p[1]).fold(f32::MAX, f32::min);
        assert!(fringe(1.0) < fringe(0.4) - 0.1, "{} {}", fringe(1.0), fringe(0.4));
        assert!(fringe(1.0) > 0.2, "the fringe reaches {}", fringe(1.0));
    }
}
