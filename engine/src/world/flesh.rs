//! The creatures in full colour: what they are made of for the pass that draws
//! them over the PS1 frame so that they look like they do not belong in it
//! (`render::flesh`, the same in the game and the training data):
//! perspective-correct, smooth-shaded and anti-aliased, no dithering, and slick,
//! as if wet. Skins are full-colour (photos when a library is loaded, see
//! `photo`) with normal and gloss maps made from their fine detail.
//!
//! Lighting sticks to what the network is told: diffuse light from each part's
//! baked light as linear spherical harmonics (exactly its token's), highlights
//! along that light's gradient and from overhead, and the torch carried with the
//! camera. Fog falls off more gently than the level's, so they stand out of the dark.

use crate::world::skin::{self, Motif};
use std::sync::atomic::{AtomicBool, Ordering};

/// Skin size in texels (photo skins; drawn skins stay at 64).
const SKIN: usize = 128;
/// How far the skin's fine detail tilts its normals.
const BUMP: f32 = 8.0;

static PSX: AtomicBool = AtomicBool::new(false);

/// Draw creatures through the PS1 pipeline like the level instead of with this pass.
pub fn set_psx(on: bool) {
    PSX.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    !PSX.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Default)]
pub struct Texel {
    /// Albedo, 0..1.
    pub c: [f32; 3],
    /// Tangent-space normal: x along u, y along v, z out of the surface.
    pub n: [f32; 3],
    /// How glossy, 0..1.
    pub wet: f32,
}

pub struct Skin {
    /// Mip levels, largest first, each square.
    mips: Vec<(usize, Vec<Texel>)>,
    tint: [f32; 3],
}

/// A corner of a creature triangle, which blends two skins (the same one twice
/// away from joins), each with its own coordinates.
#[derive(Clone, Copy)]
pub struct Vertex {
    pub p: [f32; 3],
    pub n: [f32; 3],
    /// Weight of the first skin; the second gets the rest.
    pub w: f32,
    /// Per skin: coordinates in skin repeats, and the surface directions along increasing u and v.
    pub uv: [[f32; 2]; 2],
    pub t: [[f32; 3]; 2],
    pub b: [[f32; 3]; 2],
    /// Per skin: coordinates projected along the part's axis onto its end caps, and
    /// how much of the wrapped coordinates to use rather than those (see `wrap_weight`).
    pub cap: [[f32; 2]; 2],
    pub wrap: [f32; 2],
}

pub struct Tri {
    pub v: [Vertex; 3],
    pub skin: [u16; 2],
    /// Indices into `Flesh::lights`: the light of each skin's part.
    pub light: [u16; 2],
    /// Per skin: the world directions of increasing cap u and v (the part's local x and z).
    pub cap_axes: [[[f32; 3]; 2]; 2],
}


/// How much a part's skin should wrap around its axis rather than be projected
/// onto its end caps, given its radius there as a share of its widest: wrapped
/// skins pinch to a point where a part closes.
pub fn wrap_weight(rel_radius: f32) -> f32 {
    let t = ((rel_radius - 0.3) / 0.45).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Default)]
pub struct Flesh {
    pub skins: Vec<Skin>,
    pub tris: Vec<Tri>,
    /// Per part: baked light as linear SH (mean, then gradient along x, y, z), 1 = neutral.
    pub lights: Vec<[[f32; 3]; 4]>,
    /// Where each of `lights` was taken (the part's centre), to bake it again.
    pub light_at: Vec<[f32; 3]>,
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt().max(1e-9);
    v.map(|c| c / l)
}

fn luma(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

impl Skin {
    pub fn make(m: Motif, density: f32, seed: u32, tint: [f32; 3]) -> Skin {
        let (size, rgb) = match crate::world::photo::rgb(m, density, seed, SKIN) {
            Some(rgb) => (SKIN, rgb),
            None => {
                let t = skin::make(m, density, seed);
                (crate::world::tex::SIZE, t.idx.iter().map(|&i| t.pal[i as usize].map(|c| c as f32)).collect())
            }
        };
        let at = |x: isize, y: isize| (y.rem_euclid(size as isize) as usize) * size + x.rem_euclid(size as isize) as usize;
        // Fine detail: brightness less a blurred copy of itself, so the photo's own
        // shading does not turn into slopes.
        let h: Vec<f32> = rgb.iter().map(|&c| luma(c) / 255.0).collect();
        let r = (size / 16) as isize;
        let blur = |src: &[f32], dx: isize, dy: isize| -> Vec<f32> {
            (0..size * size)
                .map(|i| {
                    let (x, y) = ((i % size) as isize, (i / size) as isize);
                    (-r..=r).map(|k| src[at(x + k * dx, y + k * dy)]).sum::<f32>() / (2 * r + 1) as f32
                })
                .collect()
        };
        let low = blur(&blur(&h, 1, 0), 0, 1);
        let d: Vec<f32> = h.iter().zip(&low).map(|(a, b)| a - b).collect();
        let k = BUMP * size as f32 / SKIN as f32;
        let base: Vec<Texel> = (0..size * size)
            .map(|i| {
                let (x, y) = ((i % size) as isize, (i / size) as isize);
                let gx = d[at(x + 1, y)] - d[at(x - 1, y)];
                let gy = d[at(x, y + 1)] - d[at(x, y - 1)];
                Texel {
                    c: rgb[i].map(|c| c / 255.0),
                    n: norm([-gx * k, -gy * k, 1.0]),
                    // Ridges glossy, creases dull.
                    wet: (0.45 + 4.0 * d[i]).clamp(0.08, 1.0),
                }
            })
            .collect();
        let mut mips = vec![(size, base)];
        while mips.last().unwrap().0 > 1 {
            let (s, prev) = mips.last().unwrap();
            let s2 = s / 2;
            let next = (0..s2 * s2)
                .map(|i| {
                    let (x, y) = (i % s2 * 2, i / s2 * 2);
                    let q = [prev[y * s + x], prev[y * s + x + 1], prev[(y + 1) * s + x], prev[(y + 1) * s + x + 1]];
                    Texel {
                        c: [0, 1, 2].map(|k| q.iter().map(|t| t.c[k]).sum::<f32>() / 4.0),
                        n: norm([0, 1, 2].map(|k| q.iter().map(|t| t.n[k]).sum::<f32>())),
                        wet: q.iter().map(|t| t.wet).sum::<f32>() / 4.0,
                    }
                })
                .collect();
            mips.push((s2, next));
        }
        Skin { mips, tint }
    }

    /// Mip levels, largest first: (size, texels row by row).
    pub fn mips(&self) -> &[(usize, Vec<Texel>)] {
        &self.mips
    }

    /// What the albedo is multiplied by.
    pub fn tint(&self) -> [f32; 3] {
        self.tint
    }

    /// Bilinear sample of mip `level`, wrapping; `u`, `v` in 0..1.
    pub fn bilinear(&self, level: usize, u: f32, v: f32) -> Texel {
        let (s, ref tex) = self.mips[level];
        let (x, y) = (u * s as f32 - 0.5, v * s as f32 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let wrap = |i: f32| (i as isize).rem_euclid(s as isize) as usize;
        let (xa, xb, ya, yb) = (wrap(x0), wrap(x0 + 1.0), wrap(y0), wrap(y0 + 1.0));
        let mut out = Texel::default();
        for (t, w) in [
            (tex[ya * s + xa], (1.0 - fx) * (1.0 - fy)),
            (tex[ya * s + xb], fx * (1.0 - fy)),
            (tex[yb * s + xa], (1.0 - fx) * fy),
            (tex[yb * s + xb], fx * fy),
        ] {
            (0..3).for_each(|k| {
                out.c[k] += t.c[k] * w;
                out.n[k] += t.n[k] * w;
            });
            out.wet += t.wet * w;
        }
        out
    }
}
