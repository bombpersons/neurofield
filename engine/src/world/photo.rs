//! Creature skins cut from photos: a library of 256x256 RGB PNGs per motif
//! (`DIR/<motif>/*.png`, see `photo/fetch_commons.py`), loaded once with
//! `load`. Each skin takes a random rotated, flipped crop of one photo, wraps
//! it seamlessly by cross-fading its edges, and quantises it to 16 colours.
//! Density works as for the drawn skins: more eyes, teeth or mouths per tile
//! for the motifs that are things, a wider crop for the ones that are surfaces.

use crate::world::rng::Rng;
use crate::world::skin::{Motif, ALL};
use crate::world::tex::{Texture, SIZE};
use std::path::Path;
use std::sync::OnceLock;

const PHOTO: usize = 256;

struct Photo {
    rgb: Vec<[f32; 3]>,
}

static LIBRARY: OnceLock<Vec<Vec<Photo>>> = OnceLock::new();

fn dir_name(m: Motif) -> String {
    format!("{m:?}").to_lowercase()
}

/// Load the photo library under `dir`. Motifs without photos keep their drawn skins.
pub fn load(dir: &Path) {
    let lib = ALL
        .iter()
        .map(|&m| {
            let mut paths: Vec<_> = std::fs::read_dir(dir.join(dir_name(m)))
                .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e == "png")).collect())
                .unwrap_or_default();
            paths.sort();
            paths.iter().filter_map(|p| read(p)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (m, photos) in ALL.iter().zip(&lib) {
        eprintln!("{:>9}: {} photos", dir_name(*m), photos.len());
    }
    LIBRARY.set(lib).ok().expect("photo library loaded twice");
}

/// A 256x256 RGB PNG as row-major colours 0..255.
pub(crate) fn read_rgb(path: &Path) -> Option<Vec<[f32; 3]>> {
    read(path).map(|p| p.rgb)
}

fn read(path: &Path) -> Option<Photo> {
    let dec = png::Decoder::new(std::fs::File::open(path).ok()?);
    let mut r = dec.read_info().ok()?;
    let mut buf = vec![0; r.output_buffer_size()];
    let info = r.next_frame(&mut buf).ok()?;
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => return None,
    };
    if info.width as usize != PHOTO || info.height as usize != PHOTO || info.bit_depth != png::BitDepth::Eight {
        eprintln!("skipping {}: want {PHOTO}x{PHOTO} 8-bit RGB", path.display());
        return None;
    }
    let rgb = buf[..PHOTO * PHOTO * ch].chunks(ch).map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    Some(Photo { rgb })
}

/// A photo skin of motif `m` for the PS1 pass (16 colours), or None when the library has none for it.
pub fn make(m: Motif, density: f32, seed: u32) -> Option<Texture> {
    rgb(m, density, seed, SIZE).map(|t| quantise(&t))
}

/// The same skin as `make` in full colour, `size` texels square, RGB 0..255 row-major.
pub fn rgb(m: Motif, density: f32, seed: u32, size: usize) -> Option<Vec<[f32; 3]>> {
    let photos = &LIBRARY.get()?[m as usize];
    if photos.is_empty() {
        return None;
    }
    let mut rng = Rng::new(seed as u64 ^ 0xF07_0F07);
    let photo = &photos[rng.next() as usize % photos.len()];
    let thing = matches!(m, Motif::Eyes | Motif::Teeth | Motif::Mouths | Motif::Suckers);
    // Copies of the crop per tile side, and the crop's width as a share of the photo.
    let (reps, crop) = if thing {
        (1 + (density * 2.0).round() as usize, rng.fr(0.55, 0.8))
    } else {
        (1, 0.25 + 0.45 * density * rng.fr(0.8, 1.2))
    };
    // The cross-fade band: the crop runs past the tile by this share and wraps back over it.
    let fade = 0.3;
    let span = crop / (1.0 + fade);
    let angle = if thing { rng.fr(-0.35, 0.35) } else { rng.fr(0.0, std::f32::consts::TAU) };
    let (sa, ca) = angle.sin_cos();
    let flip = if rng.chance(0.5) { -1.0 } else { 1.0 };
    // Centre of the crop, kept far enough in that the rotated crop stays inside the photo.
    let reach = 0.5 * crop * (sa.abs() + ca.abs());
    let slack = (0.5 - reach).max(0.0) * if thing { 0.3 } else { 1.0 };
    let (cx, cy) = (0.5 + rng.fr(-slack, slack), 0.5 + rng.fr(-slack, slack));

    // Photo colour at crop coordinates (u, v), 0..1 + fade on each axis.
    let at = |u: f32, v: f32| {
        let (lx, ly) = ((u - 0.5 * (1.0 + fade)) * span * flip, (v - 0.5 * (1.0 + fade)) * span);
        let (px, py) = (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca);
        bilinear(photo, px * PHOTO as f32, py * PHOTO as f32)
    };
    // Seamless over 0..1: near the start of each axis, fade in from the crop's far end.
    let wrapped = |u: f32, v: f32| {
        let w = |t: f32| if t < fade { smooth(t / fade) } else { 1.0 };
        let (wu, wv) = (w(u), w(v));
        let mut c = [0.0; 3];
        for (du, ku) in [(0.0, wu), (1.0, 1.0 - wu)] {
            for (dv, kv) in [(0.0, wv), (1.0, 1.0 - wv)] {
                if ku * kv > 0.0 {
                    let s = at(u + du, v + dv);
                    (0..3).for_each(|i| c[i] += s[i] * ku * kv);
                }
            }
        }
        c
    };
    // Four samples a texel, to average down the photo.
    let texels: Vec<[f32; 3]> = (0..size * size)
        .map(|i| {
            let (x, y) = ((i % size) as f32, (i / size) as f32);
            let mut c = [0.0; 3];
            for (ox, oy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let (u, v) = (((x + ox) / size as f32 * reps as f32).fract(), ((y + oy) / size as f32 * reps as f32).fract());
                let s = wrapped(u, v);
                (0..3).for_each(|k| c[k] += s[k] * 0.25);
            }
            c
        })
        .collect();
    Some(texels)
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn bilinear(p: &Photo, x: f32, y: f32) -> [f32; 3] {
    let x = (x - 0.5).clamp(0.0, (PHOTO - 1) as f32);
    let y = (y - 0.5).clamp(0.0, (PHOTO - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(PHOTO - 1), (y0 + 1).min(PHOTO - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let g = |xx: usize, yy: usize| p.rgb[yy * PHOTO + xx];
    let (a, b, c, d) = (g(x0, y0), g(x1, y0), g(x0, y1), g(x1, y1));
    [0, 1, 2].map(|i| (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy)
}

fn luma(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// 16 colours by k-means, started from brightness quantiles; palette sorted dark to light.
pub(crate) fn quantise(px: &[[f32; 3]]) -> Texture {
    let mut order: Vec<usize> = (0..px.len()).collect();
    order.sort_by(|&a, &b| luma(px[a]).total_cmp(&luma(px[b])));
    let mut cent: Vec<[f32; 3]> = (0..16).map(|k| px[order[(2 * k + 1) * px.len() / 32]]).collect();
    let nearest = |cent: &[[f32; 3]], c: [f32; 3]| {
        (0..cent.len())
            .min_by(|&a, &b| dist(cent[a], c).total_cmp(&dist(cent[b], c)))
            .unwrap()
    };
    let mut idx = vec![0usize; px.len()];
    for _ in 0..12 {
        for (i, &c) in px.iter().enumerate() {
            idx[i] = nearest(&cent, c);
        }
        let mut sum = vec![[0.0f32; 4]; 16];
        for (i, &c) in px.iter().enumerate() {
            let s = &mut sum[idx[i]];
            (0..3).for_each(|k| s[k] += c[k]);
            s[3] += 1.0;
        }
        for (k, s) in sum.iter().enumerate() {
            if s[3] > 0.0 {
                cent[k] = [s[0] / s[3], s[1] / s[3], s[2] / s[3]];
            }
        }
    }
    let mut rank: Vec<usize> = (0..16).collect();
    rank.sort_by(|&a, &b| luma(cent[a]).total_cmp(&luma(cent[b])));
    let mut slot = [0u8; 16];
    for (s, &k) in rank.iter().enumerate() {
        slot[k] = s as u8;
    }
    let pal = std::array::from_fn(|s| cent[rank[s]].map(|v| v.round().clamp(0.0, 255.0) as u8));
    let idx = px.iter().map(|&c| slot[nearest(&cent, c)]).collect();
    Texture { idx, pal }
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|i| (a[i] - b[i]) * (a[i] - b[i])).sum()
}
