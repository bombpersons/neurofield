//! Dream effects: ways of disturbing the trained network, each drifting slowly
//! with time so it never quite settles. Some change what the network is told
//! (triangles, camera), others reach inside it through the model's dream inputs
//! (texture mixing, attention temperature, pushes on the patch queries).

use crate::net::{Dream, NUM_TEX};
use crate::tokens::{Tokens, TRI_FEATURES};
use datagen::TRI_FEATURES as RAW;
use std::f32::consts::TAU;

pub const EFFECTS: [&str; 14] = [
    "none",
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
    "dream (morph + smear + waves + forgetting)",
];

/// Whether `effect` reaches inside the network (needs a model exported with the dream inputs).
pub fn needs_dream_inputs(effect: usize) -> bool {
    matches!(effect, 1 | 2 | 3 | 6 | 7 | 8 | 13)
}

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

/// Each texture swells into another, a different one each cycle, and back.
fn texture_morph(dream: &mut Dream, t: f32, period: f32) {
    let cycle = (t / period).floor() as u64;
    let w = swell(t, period);
    for i in 0..NUM_TEX {
        let shift = 1 + ((hash(cycle, i as u64) * 0.5 + 0.5) * (NUM_TEX - 1) as f32) as usize % (NUM_TEX - 1);
        dream.tex_mix[i * NUM_TEX + i] = 1.0 - w;
        dream.tex_mix[i * NUM_TEX + (i + shift) % NUM_TEX] += w;
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

/// Disturb the tokens (of a scene with raw triangle features `tris`) for `effect` at `t`
/// seconds since it was chosen; returns the network-side disturbance to render with.
pub fn apply(effect: usize, t: f32, tok: &mut Tokens, tris: &[f32]) -> Dream {
    let mut dream = Dream::calm();
    match effect {
        1 => texture_morph(&mut dream, t, 8.0),
        2 => {
            // Every texture is a drifting blend of several, mostly itself.
            for i in 0..NUM_TEX {
                let row = &mut dream.tex_mix[i * NUM_TEX..(i + 1) * NUM_TEX];
                for (j, m) in row.iter_mut().enumerate() {
                    let own = if i == j { 1.5 } else { 0.0 };
                    *m = (own + 3.0 * noise((i * NUM_TEX + j) as u64, t * 0.3)).exp();
                }
                let sum: f32 = row.iter().sum();
                row.iter_mut().for_each(|m| *m /= sum);
            }
        }
        3 => {
            let s = 1.0 + 1.5 * swell(t, 6.0);
            for i in 0..NUM_TEX {
                dream.tex_mix[i * NUM_TEX + i] = s;
            }
        }
        4 => forget(tok, t, -1.0 + 0.9 * swell(t, 12.0)),
        5 => waves(tok, tris, t, 0.05),
        6 => dream.temp = 1.0 + 2.5 * swell(t, 10.0),
        7 => dream.temp = 0.6 - 0.35 * swell(t, 6.0),
        8 => {
            let amp = 0.35 * swell(t, 8.0);
            for (k, q) in dream.q_noise.iter_mut().enumerate() {
                *q = amp * noise(k as u64, t * 0.7);
            }
        }
        9 => {
            // Fog near and far wander between a third and three times their distance.
            let far = 3f32.powf(noise(1, t * 0.15));
            let near = 3f32.powf(noise(2, t * 0.15));
            tok.cam[2] *= far;
            tok.cam[1] = (tok.cam[1] * near).min(tok.cam[2] * 0.9);
        }
        10 => {
            let a = TAU * t / 9.0;
            for c in 0..3 {
                tok.cam[3 + c] = 1.2 * (0.5 + 0.5 * (a + c as f32 * TAU / 3.0).cos());
            }
            tok.cam[6] *= 1.0 + 1.5 * swell(t, 5.0);
        }
        11 => tok.cam[0] *= 1.0 + 0.6 * (TAU * t / 7.0).sin(),
        12 => {
            // Vertex colours (1 = neutral) drift toward their inverse, each channel in its own time.
            for (k, f) in tok.feats.iter_mut().enumerate() {
                let c = k % TRI_FEATURES;
                if c >= 27 {
                    let w = swell(t + (c % 3) as f32 * 3.0, 10.0);
                    *f += (2.0 - 2.0 * *f) * w;
                }
            }
        }
        13 => {
            texture_morph(&mut dream, t, 13.0);
            dream.temp = 1.0 + 1.5 * swell(t, 11.0);
            waves(tok, tris, t, 0.025);
            forget(tok, t, -1.0 + 0.6 * swell(t, 17.0));
        }
        _ => {}
    }
    dream
}
