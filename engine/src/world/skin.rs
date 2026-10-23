//! Creature skins: 64x64 4-bit textures of creature parts (eyes, teeth,
//! mouths, suckers, scales, ...), made afresh for every scene from a motif, a
//! density and a seed. The network is told the motif and density but never
//! the seed, so it can learn what such a skin looks like but not where each
//! eye sits: whatever it draws there is its own invention.
//!
//! Palettes: entries 0-9 are a dark-to-light ramp of the motif's base surface,
//! 10-15 its accents (iris, pupil, teeth, lips, ...). Vertex tint colours the lot.

use crate::world::rng::{hash, Rng};
use crate::world::tex::{fbm, Texture, SIZE};
use std::f32::consts::TAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motif {
    Flesh,
    Eyes,
    Teeth,
    Mouths,
    Suckers,
    Veins,
    Scales,
    Chitin,
    Hair,
    Wrinkles,
    Bone,
}

pub const MOTIFS: usize = 11;
pub const ALL: [Motif; MOTIFS] = [
    Motif::Flesh,
    Motif::Eyes,
    Motif::Teeth,
    Motif::Mouths,
    Motif::Suckers,
    Motif::Veins,
    Motif::Scales,
    Motif::Chitin,
    Motif::Hair,
    Motif::Wrinkles,
    Motif::Bone,
];

/// Motifs that cover a whole body, and ones better used on a part or two.
pub const BODY: [Motif; 6] = [Motif::Flesh, Motif::Scales, Motif::Chitin, Motif::Hair, Motif::Wrinkles, Motif::Bone];
pub const ACCENT: [Motif; 5] = [Motif::Eyes, Motif::Teeth, Motif::Mouths, Motif::Suckers, Motif::Veins];

const FLESH: ([u8; 3], [u8; 3]) = ([42, 12, 16], [235, 172, 150]);

fn ramp(lo: [u8; 3], hi: [u8; 3], n: usize) -> Vec<[u8; 3]> {
    (0..n)
        .map(|i| {
            let k = if n == 1 { 1.0 } else { i as f32 / (n - 1) as f32 };
            [0, 1, 2].map(|c| (lo[c] as f32 + (hi[c] as f32 - lo[c] as f32) * k).round() as u8)
        })
        .collect()
}

fn palette(m: Motif, rng: &mut Rng) -> [[u8; 3]; 16] {
    let iris_hues: [([u8; 3], [u8; 3]); 5] = [
        ([60, 40, 0], [250, 210, 40]),
        ([10, 50, 10], [140, 230, 90]),
        ([10, 25, 70], [120, 190, 250]),
        ([70, 5, 5], [250, 80, 50]),
        ([40, 10, 60], [200, 120, 250]),
    ];
    let (base, accents): (([u8; 3], [u8; 3]), Vec<[u8; 3]>) = match m {
        Motif::Flesh => (FLESH, [ramp([90, 20, 30], [190, 70, 80], 3), ramp([60, 10, 40], [230, 200, 190], 3)].concat()),
        Motif::Eyes => {
            let (lo, hi) = rng.pick(&iris_hues);
            (FLESH, [ramp(lo, hi, 3), vec![[8, 6, 6], [222, 212, 186], [255, 255, 245]]].concat())
        }
        Motif::Teeth => (
            ([30, 6, 10], [205, 92, 100]),
            [ramp([120, 100, 70], [242, 232, 200], 3), vec![[10, 2, 4], [150, 40, 60], [255, 250, 235]]].concat(),
        ),
        Motif::Mouths => (
            FLESH,
            [ramp([70, 8, 16], [215, 70, 80], 3), vec![[10, 2, 4], [236, 226, 196], [150, 40, 60]]].concat(),
        ),
        Motif::Suckers => (
            ([40, 18, 36], [222, 170, 192]),
            [ramp([150, 110, 130], [250, 222, 230], 3), vec![[20, 4, 14], [200, 120, 150], [255, 245, 250]]].concat(),
        ),
        Motif::Veins => (
            ([70, 22, 32], [244, 194, 184]),
            [ramp([30, 6, 34], [130, 34, 96], 3), ramp([60, 20, 80], [180, 60, 120], 3)].concat(),
        ),
        Motif::Scales => (([10, 22, 14], [200, 218, 160]), ramp([200, 220, 170], [255, 255, 230], 6)),
        Motif::Chitin => (([6, 6, 10], [150, 120, 92]), ramp([180, 160, 130], [245, 236, 214], 6)),
        Motif::Hair => (([12, 8, 6], [172, 142, 112]), ramp([190, 160, 130], [230, 210, 180], 6)),
        Motif::Wrinkles => (([40, 24, 20], [222, 182, 152]), ramp([20, 10, 8], [120, 70, 60], 6)),
        Motif::Bone => (([62, 52, 40], [242, 234, 212]), ramp([20, 16, 12], [120, 100, 80], 6)),
    };
    // A little per-skin drift in the base colour.
    let drift = [0, 1, 2].map(|_| rng.fr(0.92, 1.08));
    let mut pal = [[0u8; 3]; 16];
    for (i, c) in ramp(base.0, base.1, 10).into_iter().chain(accents).enumerate() {
        pal[i] = if i < 10 { [0, 1, 2].map(|k| (c[k] as f32 * drift[k]).min(255.0) as u8) } else { c };
    }
    pal
}

/// Nearest and second-nearest of `n` x `n` jittered points that tile over the texture.
struct Near {
    d1: f32,
    d2: f32,
    /// Offset from the nearest point to the texel.
    dx: f32,
    dy: f32,
    id: u32,
}

fn near(x: usize, y: usize, n: i32, jitter: f32, seed: u64) -> Near {
    let c = SIZE as f32 / n as f32;
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let (ci, cj) = ((px / c).floor() as i32, (py / c).floor() as i32);
    let mut best = Near { d1: f32::MAX, d2: f32::MAX, dx: 0.0, dy: 0.0, id: 0 };
    for j in cj - 1..=cj + 1 {
        for i in ci - 1..=ci + 1 {
            let id = (i.rem_euclid(n) + n * j.rem_euclid(n)) as u32;
            let h = hash(seed ^ (id as u64) << 24);
            let jx = ((h & 0xFFFF) as f32 / 65535.0 - 0.5) * jitter;
            let jy = ((h >> 16) as f32 / 65535.0 - 0.5) * jitter;
            let (qx, qy) = ((i as f32 + 0.5 + jx) * c, (j as f32 + 0.5 + jy) * c);
            let (dx, dy) = (px - qx, py - qy);
            let d = (dx * dx + dy * dy).sqrt();
            if d < best.d1 {
                best = Near { d1: d, d2: best.d1, dx, dy, id };
            } else if d < best.d2 {
                best.d2 = d;
            }
        }
    }
    best
}

/// 0..1 per cell id.
fn cell_rand(id: u32, seed: u64, k: u64) -> f32 {
    (hash(seed ^ (id as u64) << 24 ^ k << 56) & 0xFFFF) as f32 / 65535.0
}

fn base(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 9.0).round() as u8
}

/// Folded, mottled flesh: the ground most motifs sit on.
fn flesh(x: usize, y: usize, d: f32, s: u64) -> u8 {
    let n = fbm(x, y, 4, 4, s);
    let g = fbm(x, y, 16, 2, s + 1);
    let fold = (fbm(x, y, 2 + (d * 4.0) as i32, 3, s + 2) - 0.5).abs();
    base(0.3 + 0.5 * n + 0.15 * g - if fold < 0.035 { 0.3 } else { 0.0 })
}

/// Palette index of texel (x, y). `d` is the density, 0..1; `s` the seed.
fn texel(m: Motif, x: usize, y: usize, d: f32, s: u64) -> u8 {
    let n = fbm(x, y, 4, 4, s + 10);
    let g = fbm(x, y, 16, 2, s + 11);
    let c = |cells: i32| SIZE as f32 / cells as f32;
    match m {
        Motif::Flesh => flesh(x, y, d, s),
        // Eyes: lid, sclera, streaked iris, round or slit pupil, a glint.
        Motif::Eyes => {
            let k = 1 + (d * 2.0).round() as i32;
            let p = near(x, y, k, 0.6, s);
            let r = 0.42 * c(k) * (0.6 + 0.4 * cell_rand(p.id, s, 1));
            let rd = p.d1 / r;
            if rd >= 1.0 {
                return flesh(x, y, d, s);
            }
            let slit = cell_rand(p.id, s, 2) < 0.3;
            let (gx, gy) = (p.dx / r + 0.28, p.dy / r + 0.28);
            if gx * gx + gy * gy < 0.016 {
                15
            } else if (slit && (p.dx / r).abs() < 0.09 && rd < 0.6) || (!slit && rd < 0.22 + 0.1 * cell_rand(p.id, s, 3)) {
                13
            } else if rd < 0.62 {
                let streak = ((p.dy.atan2(p.dx) * 11.0 + n * 6.0).sin() * 0.5 + 0.5) * 0.8 + 0.2 * (1.0 - rd);
                10 + (streak * 2.99) as u8
            } else if rd < 0.88 {
                if g < 0.3 { 12 } else { 14 }
            } else {
                1
            }
        }
        // Teeth: rows of upper fangs and lower teeth over a black gap, gums between.
        Motif::Teeth => {
            let rows = 1 + (d * 2.0).round() as usize;
            let band = SIZE as f32 / rows as f32;
            let v = (y as f32 % band) / band;
            let row = (y as f32 / band) as u64;
            if !(0.2..0.85).contains(&v) {
                return base(0.25 + 0.5 * n + 0.2 * g - if v < 0.2 { 0.0 } else { 0.15 });
            }
            let k = 3 + (d * 5.0).round() as usize;
            let w = SIZE as f32 / k as f32;
            let tooth = |lower: bool| {
                let shift = if lower { 0.5 } else { 0.0 };
                let fx = x as f32 / w + shift;
                let id = (fx.floor() as i64).rem_euclid(k as i64) as u32 + (row as u32) * 64 + (lower as u32) * 32;
                let tu = (fx.fract() - 0.5) * 2.0;
                let len = if lower { 0.35 } else { 0.6 } * (0.6 + 0.6 * cell_rand(id, s, 4));
                let tv = if lower { (0.85 - v) / 0.65 } else { (v - 0.2) / 0.65 };
                (tv < len && tu.abs() < 0.9 * (1.0 - tv / len).powf(0.6)).then_some(tu)
            };
            match tooth(false).or_else(|| tooth(true)) {
                Some(tu) if tu > -0.55 && tu < -0.3 && g > 0.5 => 15,
                Some(tu) => 10 + ((1.0 - tu.abs()) * 2.99) as u8,
                None => 13,
            }
        }
        // Mouths: lipped slits at odd angles, teeth along the edges.
        Motif::Mouths => {
            let k = 1 + (d * 2.0).round() as i32;
            let p = near(x, y, k, 0.5, s);
            let r = 0.46 * c(k) * (0.6 + 0.4 * cell_rand(p.id, s, 1));
            let a = (cell_rand(p.id, s, 2) - 0.5) * 1.4;
            let (sa, ca) = a.sin_cos();
            let (lx, ly) = ((p.dx * ca + p.dy * sa) / r, (-p.dx * sa + p.dy * ca) / r);
            if lx.abs() >= 1.0 {
                return flesh(x, y, d, s);
            }
            let open = 0.3 * (0.5 + cell_rand(p.id, s, 3)) * (1.0 - lx * lx);
            if ly.abs() < open {
                if ly.abs() > open - 0.12 && (lx * 18.0).sin() > -0.2 {
                    14
                } else if ly > 0.4 * open {
                    15
                } else {
                    13
                }
            } else if ly.abs() < open + 0.16 * (1.0 - lx * lx) + 0.03 {
                10 + ((1.0 - (ly.abs() - open) / 0.2).clamp(0.0, 1.0) * 2.99) as u8
            } else {
                flesh(x, y, d, s)
            }
        }
        // Suckers: raised rims around dark cups, a nub in each.
        Motif::Suckers => {
            let k = 2 + (d * 4.0).round() as i32;
            let p = near(x, y, k, 0.55, s);
            let r = 0.4 * c(k) * (0.6 + 0.4 * cell_rand(p.id, s, 1));
            let rd = p.d1 / r;
            if rd >= 1.0 {
                base(0.3 + 0.5 * n + 0.15 * g)
            } else if rd < 0.15 {
                14
            } else if rd < 0.55 {
                13
            } else if p.dx + p.dy < -0.9 * r {
                15
            } else {
                10 + ((1.0 - rd) / 0.45 * 2.99) as u8
            }
        }
        // Veins: branching dark lines through pale membrane.
        Motif::Veins => {
            let period = 2 + (d * 4.0).round() as i32;
            let thick = (fbm(x, y, period, 4, s + 20) - 0.5).abs();
            let thin = (fbm(x, y, period * 2, 3, s + 21) - 0.5).abs();
            if thick < 0.022 {
                10
            } else if thick < 0.04 {
                11
            } else if thin < 0.014 {
                12 + (g * 2.99) as u8
            } else {
                base(0.45 + 0.4 * n + 0.15 * g)
            }
        }
        // Scales: jittered plates, dark at the rims, a glint near the top of some.
        Motif::Scales => {
            let k = 3 + (d * 5.0).round() as i32;
            let p = near(x, y, k, 0.7, s);
            if p.d2 - p.d1 < 1.3 {
                1
            } else if p.dy < -0.25 * c(k) && p.d1 < 0.3 * c(k) && cell_rand(p.id, s, 1) < 0.5 {
                15
            } else {
                base(0.35 + 0.55 * (1.0 - p.d1 / (0.75 * c(k))) + 0.1 * g)
            }
        }
        // Chitin: few large glossy plates, black seams, a highlight band.
        Motif::Chitin => {
            let k = 1 + (d * 2.0).round() as i32;
            let p = near(x, y, k, 0.8, s);
            if p.d2 - p.d1 < 1.6 {
                0
            } else if (p.dx + 0.3 * p.dy + 0.15 * c(k)).abs() < 1.4 + 2.0 * g {
                10 + (g * 5.99) as u8
            } else {
                base(0.2 + 0.6 * (0.5 - p.dy / c(k)).clamp(0.0, 1.0) + 0.15 * n)
            }
        }
        // Hair: wandering streaks, denser with density.
        Motif::Hair => {
            let w = ((x as f32 + n * 14.0) * (0.7 + 1.6 * d)).sin() * 0.5 + 0.5;
            let tip = fbm(x, y, 8, 2, s + 30);
            if w > 0.92 && tip > 0.55 {
                10 + (tip * 5.99) as u8
            } else {
                base(0.12 + 0.6 * w * (0.6 + 0.6 * g))
            }
        }
        // Wrinkles: warped creases running across.
        Motif::Wrinkles => {
            let bands = 3.0 + (d * 8.0).round();
            let phase = y as f32 / SIZE as f32 * bands * TAU + n * 6.0;
            if phase.cos() > 0.93 {
                10 + (g * 3.99) as u8
            } else {
                base(0.45 + 0.25 * (phase + 0.6).sin() + 0.25 * n)
            }
        }
        // Bone: pale, cracked, pitted.
        Motif::Bone => {
            let k = 3 + (d * 5.0).round() as i32;
            let ridge = (fbm(x, y, 4, 3, s + 40) - 0.5).abs();
            let pit = near(x, y, k, 0.9, s + 41);
            if ridge < 0.015 {
                10
            } else if pit.d1 < 1.6 && cell_rand(pit.id, s, 1) < 0.6 {
                11 + (g * 2.99) as u8
            } else {
                base(0.55 + 0.35 * n + 0.1 * g)
            }
        }
    }
}

/// A skin of motif `m`: cut from a photo when a library is loaded (see `photo`), else drawn.
pub fn make(m: Motif, density: f32, seed: u32) -> Texture {
    if let Some(t) = crate::world::photo::make(m, density, seed) {
        return t;
    }
    let mut rng = Rng::new(seed as u64 ^ 0x5EED_5EED);
    let pal = palette(m, &mut rng);
    let s = rng.next() as u64;
    let idx = (0..SIZE * SIZE).map(|i| texel(m, i % SIZE, i / SIZE, density, s)).collect();
    Texture { idx, pal }
}

/// RGB8 sheet: one row per motif, densities 0, 0.5, 1 with two seeds each,
/// every skin tiled 2x2 to show the wrap.
pub fn sheet() -> (u32, u32, Vec<u8>) {
    let cols = 6;
    let cell = 2 * SIZE;
    let (w, h) = (cols * cell, MOTIFS * cell);
    let mut out = vec![0u8; w * h * 3];
    for (r, &m) in ALL.iter().enumerate() {
        for c in 0..cols {
            let t = make(m, (c / 2) as f32 * 0.5, (r * 16 + c) as u32);
            for y in 0..cell {
                for x in 0..cell {
                    let o = ((r * cell + y) * w + c * cell + x) * 3;
                    out[o..o + 3].copy_from_slice(&t.pal[t.idx[(y % SIZE) * SIZE + x % SIZE] as usize]);
                }
            }
        }
    }
    (w as u32, h as u32, out)
}
