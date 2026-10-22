//! Twelve fixed 64x64 4-bit CLUT textures, generated procedurally and packed
//! into one VRAM texture page (4 across, 3 down) at x=640: eight for the
//! level, three for props and one for doors. Texture ids from `SKIN_BASE` on
//! are per-scene creature skins (see `skin`), on two more pages at x=704 and 768.

use crate::rng::hash;
use psx_gpu::prim::{Blend, Clut, Env, Prim, TexDepth, TexPage};
use psx_gpu::vram::rgb8_to_rgb15;

pub const N: usize = 12;
pub const SCALES: u8 = 8;
pub const BONE: u8 = 9;
pub const FUR: u8 = 10;
pub const DOOR: u8 = 11;
/// Rows of textures in the page.
const ROWS: usize = N.div_ceil(4);
pub const SIZE: usize = 64;
const PAGE_X: u16 = 640;
const CLUT_Y: u16 = 480;
/// First texture id of the per-scene creature skins.
pub const SKIN_BASE: u8 = 16;
/// Skins one scene can hold: two pages of 16.
pub const SKIN_SLOTS: usize = 32;
const SKIN_PAGE_X: u16 = 704;

/// Dark and light ends of each texture's 16-step palette.
const PALETTES: [([u8; 3], [u8; 3]); N] = [
    ([40, 14, 10], [205, 115, 80]),
    ([28, 28, 34], [175, 170, 150]),
    ([34, 20, 8], [195, 145, 85]),
    ([18, 26, 40], [225, 222, 200]),
    ([10, 24, 8], [115, 175, 60]),
    ([14, 17, 22], [155, 165, 175]),
    ([40, 10, 42], [215, 175, 120]),
    ([6, 10, 32], [245, 205, 90]),
    ([8, 22, 18], [120, 170, 110]),
    ([42, 36, 26], [230, 222, 195]),
    ([22, 12, 8], [150, 105, 70]),
    ([24, 14, 6], [170, 120, 70]),
];

pub struct Texture {
    /// Palette indices, row-major 64x64.
    pub idx: Vec<u8>,
    pub pal: [[u8; 3]; 16],
}

pub fn page() -> TexPage {
    TexPage::at(PAGE_X, 0, TexDepth::Clut4, Blend::Half)
}

/// The texture page holding texture `t`.
pub fn page_of(t: u8) -> TexPage {
    if t < SKIN_BASE {
        page()
    } else {
        TexPage::at(SKIN_PAGE_X + 64 * ((t - SKIN_BASE) as u16 / 16), 0, TexDepth::Clut4, Blend::Half)
    }
}

pub fn clut(t: u8) -> Clut {
    if t < SKIN_BASE {
        Clut::at(0, CLUT_Y + t as u16)
    } else {
        // Skin CLUTs sit beside the level's, 16 to a row.
        let s = (t - SKIN_BASE) as u16;
        Clut::at(16 * (1 + s % 16), CLUT_Y + s / 16)
    }
}

/// Top-left texel of texture `t` inside its page.
pub fn uv_origin(t: u8) -> (u8, u8) {
    let s = if t < SKIN_BASE { t as usize } else { (t - SKIN_BASE) as usize % 16 };
    ((s % 4 * SIZE) as u8, (s / 4 * SIZE) as u8)
}

/// Tileable value noise over a `period`-cell lattice, 0..1.
pub(crate) fn value_noise(x: f32, y: f32, period: i32, seed: u64) -> f32 {
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let corner = |i: i32, j: i32| {
        let (i, j) = (i.rem_euclid(period) as u64, j.rem_euclid(period) as u64);
        (hash(seed ^ (i << 20) ^ (j << 40)) & 0xFFFF) as f32 / 65535.0
    };
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let top = corner(x0, y0) * (1.0 - sx) + corner(x0 + 1, y0) * sx;
    let bot = corner(x0, y0 + 1) * (1.0 - sx) + corner(x0 + 1, y0 + 1) * sx;
    top * (1.0 - sy) + bot * sy
}

/// Fractal noise that tiles over the 64-texel texture.
pub(crate) fn fbm(x: usize, y: usize, base: i32, octaves: u32, seed: u64) -> f32 {
    let (mut sum, mut amp, mut norm, mut period) = (0.0, 1.0, 0.0, base);
    for o in 0..octaves {
        let scale = period as f32 / SIZE as f32;
        sum += amp * value_noise(x as f32 * scale, y as f32 * scale, period, seed + o as u64);
        norm += amp;
        amp *= 0.5;
        period *= 2;
    }
    sum / norm
}

fn texel(t: usize, x: usize, y: usize) -> f32 {
    let n = fbm(x, y, 4, 4, 100 + t as u64);
    let grain = fbm(x, y, 16, 2, 200 + t as u64);
    match t {
        // Brick: 16-texel courses, 32-texel bricks offset every other course.
        0 => {
            let row = y / 16;
            let bx = (x + if row % 2 == 1 { 16 } else { 0 }) % 32;
            if y % 16 < 2 || bx < 2 {
                0.08 + 0.1 * grain
            } else {
                0.45 + 0.35 * n + 0.15 * grain
            }
        }
        // Stone: noise with dark cracks along a ridge of a second noise field.
        1 => {
            let ridge = (fbm(x, y, 4, 3, 300) - 0.5).abs();
            if ridge < 0.025 {
                0.05
            } else {
                0.25 + 0.6 * n + 0.15 * grain
            }
        }
        // Planks: 16-texel boards with wavy grain and a dark gap.
        2 => {
            if x % 16 == 0 {
                0.05
            } else {
                let w = (y as f32 * 0.35 + n * 9.0 + (x / 16) as f32 * 2.1).sin() * 0.5 + 0.5;
                0.3 + 0.45 * w + 0.15 * grain
            }
        }
        // Tiles: 16-texel checker with grout.
        3 => {
            if x % 16 == 0 || y % 16 == 0 {
                0.15
            } else if (x / 16 + y / 16) % 2 == 0 {
                0.75 + 0.2 * grain
            } else {
                0.35 + 0.2 * grain
            }
        }
        // Moss: blotchy high-frequency noise.
        4 => (0.2 + 0.8 * fbm(x, y, 8, 3, 400) * (0.6 + 0.6 * grain)).min(1.0),
        // Grate: bars every 8 texels with rivets at crossings.
        5 => {
            let (bx, by) = (x % 8 < 2, y % 8 < 2);
            if bx && by {
                0.95
            } else if bx || by {
                0.55 + 0.25 * grain
            } else {
                0.04 + 0.08 * n
            }
        }
        // Wallpaper: diamond lattice.
        6 => {
            let (u, v) = ((x % 16) as f32 - 7.5, (y % 16) as f32 - 7.5);
            let d = u.abs() + v.abs();
            if d < 3.5 {
                0.9
            } else if (d - 7.0).abs() < 1.0 {
                0.15
            } else {
                0.45 + 0.2 * n
            }
        }
        // Rings: concentric bands around the centre, warped by noise.
        7 => {
            let (u, v) = (x as f32 - 31.5, y as f32 - 31.5);
            let r = (u * u + v * v).sqrt() + n * 10.0;
            (r * 0.45).sin() * 0.4 + 0.5 + 0.1 * grain
        }
        // Scales: offset rows of rounded plates, dark at the rims.
        8 => {
            let row = y / 8;
            let (u, v) = (((x + row % 2 * 4) % 8) as f32 - 3.5, (y % 8) as f32 - 1.0);
            let d = (u * u + v * v).sqrt() / 4.5;
            (0.85 - d * 0.6 + 0.15 * grain).clamp(0.05, 1.0)
        }
        // Bone: pale and smooth, with thin dark cracks and pitting.
        9 => {
            let ridge = (fbm(x, y, 8, 2, 900) - 0.5).abs();
            if ridge < 0.012 {
                0.15
            } else {
                0.6 + 0.3 * n + 0.1 * grain
            }
        }
        // Fur: vertical streaks that wander with noise.
        10 => {
            let w = ((x as f32 + n * 12.0) * 1.3).sin() * 0.5 + 0.5;
            0.25 + 0.45 * w * (0.6 + 0.6 * grain)
        }
        // Door: vertical planks crossed by two iron bands with studs.
        _ => {
            let band = (14..20).contains(&y) || (44..50).contains(&y);
            if band {
                if x % 10 == 4 && (y == 16 || y == 46) {
                    0.95
                } else {
                    0.12
                }
            } else if x % 16 == 0 {
                0.05
            } else {
                let w = (y as f32 * 0.3 + n * 8.0 + (x / 16) as f32 * 1.7).sin() * 0.5 + 0.5;
                0.35 + 0.4 * w + 0.1 * grain
            }
        }
    }
}

pub fn make_all() -> Vec<Texture> {
    (0..N)
        .map(|t| {
            let (lo, hi) = PALETTES[t];
            let mut pal = [[0u8; 3]; 16];
            for (i, c) in pal.iter_mut().enumerate() {
                let k = i as f32 / 15.0;
                for ch in 0..3 {
                    c[ch] = (lo[ch] as f32 + (hi[ch] as f32 - lo[ch] as f32) * k).round() as u8;
                }
            }
            let idx =
                (0..SIZE * SIZE).map(|i| (texel(t, i % SIZE, i / SIZE).clamp(0.0, 1.0) * 15.0).round() as u8).collect();
            Texture { idx, pal }
        })
        .collect()
}

/// GP0 uploads for the texture page and the CLUTs.
pub fn uploads(texs: &[Texture]) -> Vec<Prim> {
    let mut out = vec![page_upload(texs, PAGE_X)];
    for (t, tex) in texs.iter().enumerate() {
        out.push(clut_upload(tex, t as u8));
    }
    out
}

/// GP0 uploads for a scene's creature skins (at most `SKIN_SLOTS`).
pub fn skin_uploads(skins: &[Texture]) -> Vec<Prim> {
    let mut out: Vec<Prim> =
        skins.chunks(16).enumerate().map(|(p, page)| page_upload(page, SKIN_PAGE_X + 64 * p as u16)).collect();
    for (s, tex) in skins.iter().enumerate() {
        out.push(clut_upload(tex, SKIN_BASE + s as u8));
    }
    out
}

/// Up to 16 textures packed 4 across into one 4-bit page at VRAM x.
fn page_upload(texs: &[Texture], x: u16) -> Prim {
    // The page region is 256 texels wide = 64 halfwords (4 texels each).
    let (w, h) = (64usize, texs.len().div_ceil(4) * SIZE);
    let mut page = vec![0u16; w * h];
    for (t, tex) in texs.iter().enumerate() {
        let (u0, v0) = uv_origin(t as u8);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let (u, v) = (u0 as usize + x, v0 as usize + y);
                page[v * w + u / 4] |= (tex.idx[y * SIZE + x] as u16) << ((u % 4) * 4);
            }
        }
    }
    Prim::Env(Env::Upload { x, y: 0, w: w as u16, h: h as u16, data: page })
}

fn clut_upload(tex: &Texture, t: u8) -> Prim {
    // 0x0000 is transparent on the PS1, so nudge pure black off it.
    let data = tex.pal.iter().map(|&[r, g, b]| rgb8_to_rgb15(r, g, b).max(0x0421)).collect();
    let c = clut(t).0;
    Prim::Env(Env::Upload { x: (c & 0x3F) * 16, y: c >> 6, w: 16, h: 1, data })
}

/// RGB8 atlas of all textures (256 wide), laid out as in VRAM.
pub fn atlas_rgb(texs: &[Texture]) -> (u32, u32, Vec<u8>) {
    let (w, h) = (4 * SIZE, ROWS * SIZE);
    let mut out = vec![0u8; w * h * 3];
    for (t, tex) in texs.iter().enumerate() {
        let (u0, v0) = uv_origin(t as u8);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let o = ((v0 as usize + y) * w + u0 as usize + x) * 3;
                out[o..o + 3].copy_from_slice(&tex.pal[tex.idx[y * SIZE + x] as usize]);
            }
        }
    }
    (w as u32, h as u32, out)
}
