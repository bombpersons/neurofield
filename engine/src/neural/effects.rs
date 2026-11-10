//! Dream effects: ways of disturbing the trained network, each drifting slowly
//! with time so it never quite settles. Some change what the network is told
//! (triangles, camera), others reach inside it through the model's dream inputs
//! (texture mixing, attention temperature, pushes on the patch queries). Any set
//! of them can be on at once: texture mixes multiply, attention temperatures
//! multiply and query pushes add up.

use crate::neural::dream::{Dream, NUM_TEX};
use crate::neural::tokens::{Tokens, TRI_FEATURES};
use crate::neural::features::TRI_FEATURES as RAW;
use crate::world::scene::{Camera, T};
use std::f32::consts::TAU;

pub const EFFECTS: [&str; 12] = [
    "texture morph",
    "texture drift",
    "texture intensity",
    "forgetting",
    "vertex waves",
    "smeared attention",
    "stark attention",
    "query shimmer",
    "fog drift",
    "torch dance",
    "lens breathing",
    "colour drift",
];

/// The "dream" set: morph, waves, smear and forgetting together.
pub const DREAM: [usize; 4] = [0, 3, 4, 5];

/// A hash of (a, b) to -1..1.
fn hash(a: u64, b: u64) -> f32 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^= x >> 33;
    (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
}

/// Smooth noise in -1..1 for key `k`, changing about once per unit of `t`.
fn noise(k: u64, t: f32) -> f32 {
    let i = t.floor();
    let f = t - i;
    let (a, b) = (hash(k, i as i64 as u64), hash(k, (i as i64 + 1) as u64));
    a + (b - a) * f * f * (3.0 - 2.0 * f)
}

/// 0 -> 1 -> 0 over `period` seconds.
fn swell(t: f32, period: f32) -> f32 {
    0.5 - 0.5 * (TAU * t / period).cos()
}

/// Texture mix where each texture swells into another, a different one each cycle, and back.
fn texture_morph(t: f32, period: f32) -> Vec<f32> {
    let cycle = (t / period).floor() as u64;
    let w = swell(t, period);
    let mut m = vec![0.0; NUM_TEX * NUM_TEX];
    for i in 0..NUM_TEX {
        let shift = 1 + ((hash(cycle, i as u64) * 0.5 + 0.5) * (NUM_TEX - 1) as f32) as usize % (NUM_TEX - 1);
        m[i * NUM_TEX + i] = 1.0 - w;
        m[i * NUM_TEX + (i + shift) % NUM_TEX] += w;
    }
    m
}

/// Texture mix where every texture is a drifting blend of several, mostly itself.
fn texture_drift(t: f32) -> Vec<f32> {
    let mut m = vec![0.0; NUM_TEX * NUM_TEX];
    for i in 0..NUM_TEX {
        let row = &mut m[i * NUM_TEX..(i + 1) * NUM_TEX];
        for (j, v) in row.iter_mut().enumerate() {
            let own = if i == j { 1.5 } else { 0.0 };
            *v = (own + 3.0 * noise((i * NUM_TEX + j) as u64, t * 0.3)).exp();
        }
        let sum: f32 = row.iter().sum();
        row.iter_mut().for_each(|v| *v /= sum);
    }
    m
}

/// Mix the dream's texture mix further: tex_mix = m tex_mix.
fn mix_textures(dream: &mut Dream, m: &[f32]) {
    let old = dream.tex_mix.clone();
    for i in 0..NUM_TEX {
        for j in 0..NUM_TEX {
            dream.tex_mix[i * NUM_TEX + j] = (0..NUM_TEX).map(|k| m[i * NUM_TEX + k] * old[k * NUM_TEX + j]).sum();
        }
    }
}

/// Hide triangles whose noise falls below `threshold` (-1: none, 1: all).
fn forget(tok: &mut Tokens, t: f32, threshold: f32) {
    tok.retain(|id| noise(id as u64 ^ 0xF0, t * 0.5) > threshold);
}

/// Sway every vertex on screen by up to `amount` (NDC; the frame is 2 wide) along sine
/// waves rolling through the world, so neighbouring triangles move together and the
/// level ripples rather than cracks. `tris` is the scene's raw triangle features.
fn waves(tok: &mut Tokens, tris: &[f32], t: f32, amount: f32) {
    for (k, &id) in tok.ids.iter().enumerate() {
        for j in 0..3 {
            let w = &tris[id * RAW + 3 * j..id * RAW + 3 * j + 3]; // world position, in tiles
            let (x, y, z) = (w[0], w[1], w[2]);
            let dx = 0.6 * (1.3 * t + 2.1 * x + 1.7 * z).sin() + 0.4 * (0.9 * t - 1.4 * y + 2.6 * x).sin();
            let dy = 0.6 * (1.1 * t + 1.9 * z - 1.5 * y).sin() + 0.4 * (1.7 * t + 2.3 * y + 1.2 * x).sin();
            tok.feats[k * TRI_FEATURES + 9 + 2 * j] += amount * dx;
            tok.feats[k * TRI_FEATURES + 10 + 2 * j] += amount * dy;
        }
    }
}

/// Push the patch queries about by up to `amp`.
fn shimmer(dream: &mut Dream, t: f32, amp: f32) {
    for (k, q) in dream.q_noise.iter_mut().enumerate() {
        *q += amp * noise(k as u64, t * 0.7);
    }
}

/// Fog near and far wander between 1/3^k and 3^k times their distance.
fn fog_drift(tok: &mut Tokens, t: f32, k: f32) {
    let far = 3f32.powf(k * noise(1, t * 0.15));
    let near = 3f32.powf(k * noise(2, t * 0.15));
    tok.cam[2] *= far;
    tok.cam[1] = (tok.cam[1] * near).min(tok.cam[2] * 0.9);
}

/// The torch's colour cycles and its reach pulses, `k` of the way from how it is.
fn torch_dance(tok: &mut Tokens, t: f32, k: f32) {
    let a = TAU * t / 9.0;
    for c in 0..3 {
        let dance = 1.2 * (0.5 + 0.5 * (a + c as f32 * TAU / 3.0).cos());
        tok.cam[3 + c] += (dance - tok.cam[3 + c]) * k;
    }
    tok.cam[6] *= 1.0 + 1.5 * k * swell(t, 5.0);
}

/// Vertex colours (1 = neutral) drift `k` of the way toward their inverse, each channel in its own time.
fn colour_drift(tok: &mut Tokens, t: f32, k: f32) {
    for (n, f) in tok.feats.iter_mut().enumerate() {
        let c = n % TRI_FEATURES;
        if c >= 27 {
            let w = swell(t + (c % 3) as f32 * 3.0, 10.0);
            *f += (2.0 - 2.0 * *f) * w * k;
        }
    }
}

/// How far the swell moves the world at most, in tiles.
const SWELL: f32 = 0.3;

/// The slow swell of the world at `p` (tiles) and `t` seconds, about -1..1 on each axis: long
/// waves crossing at odd angles, each bent by a slower one so that no crest stays straight.
fn heave(p: [f32; 3], t: f32) -> [f32; 3] {
    // Direction times wave number, and speed in radians a second.
    const WAVES: [([f32; 3], f32); 4] = [
        ([0.71, 0.33, 0.52], 0.47),
        ([-0.43, 0.61, 0.89], 0.31),
        ([1.13, -0.27, -0.61], 0.59),
        ([0.37, 0.83, -1.21], 0.37),
    ];
    let along = |d: &[f32; 3], q: [f32; 3]| d[0] * q[0] + d[1] * q[1] + d[2] * q[2];
    [0, 1, 2].map(|a| {
        let q = [p[a], p[(a + 1) % 3], p[(a + 2) % 3]];
        let curl = 0.9 * (0.29 * q[1] - 0.23 * q[2] + 0.19 * t + a as f32).sin();
        WAVES.iter().enumerate().map(|(k, (d, speed))| (along(d, q) + speed * t + curl + 2.1 * (a + k) as f32).sin()).sum::<f32>() / 2.5
    })
}

/// The world's swell for one frame of affliction. Everything drawn goes through the same
/// function of where it is, so nothing comes apart.
pub struct Swelling {
    amount: f32,
    eye: [f32; 3],
    t: f32,
}

impl Swelling {
    /// For affliction `a` at `t` seconds, seen from `cam`.
    pub fn of(a: f32, t: f32, cam: &Camera) -> Swelling {
        Swelling { amount: ramp(a, 0.15, 0.8), eye: cam.pos.map(|c| c as f32 / T as f32), t }
    }

    /// Where `p` (tiles) goes: less close to the eye, so no wall reaches it, and not at all
    /// far away, where the fog has it anyway.
    pub fn tiles(&self, p: [f32; 3]) -> [f32; 3] {
        if self.amount <= 0.0 {
            return p;
        }
        let d = [p[0] - self.eye[0], p[1] - self.eye[1], p[2] - self.eye[2]];
        let away = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let k = self.amount * SWELL * ramp(away, 0.5, 2.5) * (1.0 - ramp(away, 12.0, 16.0));
        if k <= 0.0 {
            return p;
        }
        let h = heave(p, self.t);
        [p[0] + h[0] * k, p[1] + h[1] * k, p[2] + h[2] * k]
    }

    /// The raw triangle features `tris` swollen, into `out`.
    pub fn tris(&self, tris: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.extend_from_slice(tris);
        if self.amount > 0.0 {
            for t in out.chunks_exact_mut(RAW) {
                for v in t[..9].chunks_exact_mut(3) {
                    let p = self.tiles([v[0], v[1], v[2]]);
                    v.copy_from_slice(&p);
                }
            }
        }
    }
}

/// How much of the most tentacles affliction `a` calls for (`world::growths`): none until
/// nearly the end.
pub fn growth(a: f32) -> f32 {
    ramp(a, 0.85, 1.0)
}

/// `a` from `lo` to `hi` as a smooth 0..1.
fn ramp(a: f32, lo: f32, hi: f32) -> f32 {
    let x = ((a - lo) / (hi - lo)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Disturb the tokens (of a scene with raw triangle features `tris`) with every effect that
/// has a time (seconds since it was turned on); returns the network-side disturbance.
pub fn apply_set(times: &[Option<f32>; EFFECTS.len()], tok: &mut Tokens, tris: &[f32]) -> Dream {
    let mut dream = Dream::calm(tok.width);
    for (e, t) in times.iter().enumerate() {
        if let Some(t) = *t {
            apply_one(e, t, tok, tris, &mut dream);
        }
    }
    dream
}

/// Effect `e` alone at `t` seconds.
pub fn apply(choice: usize, t: f32, tok: &mut Tokens, tris: &[f32]) -> Dream {
    let mut times = [None; EFFECTS.len()];
    times[choice] = Some(t);
    apply_set(&times, tok, tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_part_comes_in_gradually() {
        assert_eq!(ramp(0.1, 0.2, 0.7), 0.0);
        assert_eq!(ramp(0.9, 0.2, 0.7), 1.0);
        assert!(ramp(0.3, 0.2, 0.7) < ramp(0.5, 0.2, 0.7));
    }
}

fn apply_one(e: usize, t: f32, tok: &mut Tokens, tris: &[f32], dream: &mut Dream) {
    match e {
        0 => mix_textures(dream, &texture_morph(t, 8.0)),
        1 => mix_textures(dream, &texture_drift(t)),
        2 => {
            let s = 1.0 + 1.5 * swell(t, 6.0);
            dream.tex_mix.iter_mut().for_each(|v| *v *= s);
        }
        3 => forget(tok, t, -1.0 + 0.9 * swell(t, 12.0)),
        4 => waves(tok, tris, t, 0.05),
        5 => dream.temp *= 1.0 + 2.5 * swell(t, 10.0),
        6 => dream.temp *= 0.6 - 0.35 * swell(t, 6.0),
        7 => shimmer(dream, t, 0.35 * swell(t, 8.0)),
        8 => fog_drift(tok, t, 1.0),
        9 => torch_dance(tok, t, 1.0),
        10 => tok.cam[0] *= 1.0 + 0.6 * (TAU * t / 7.0).sin(),
        11 => colour_drift(tok, t, 1.0),
        _ => {}
    }
}
