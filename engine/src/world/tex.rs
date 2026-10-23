//! Twelve fixed 64x64 16-colour textures, generated procedurally and laid out
//! as one PS1 texture page would hold them (4 across, 3 down), which is what
//! triangles' uv coordinates count in: eight for the level, three for props
//! and one for doors. Texture ids from `SKIN_BASE` on are per-scene creature
//! skins (see `skin`), on pages of 16 of their own. Ids from `STRANGE_BASE` on
//! are the level textures at higher strangeness (see `load_strange`), 16 to a page.

use crate::world::rng::hash;

pub const N: usize = 12;
pub const SCALES: u8 = 8;
pub const BONE: u8 = 9;
pub const FUR: u8 = 10;
pub const DOOR: u8 = 11;
/// Rows of textures in the page.
const ROWS: usize = N.div_ceil(4);
pub const SIZE: usize = 64;
/// First texture id of the per-scene creature skins.
pub const SKIN_BASE: u8 = 16;
/// Skins one scene can hold: two pages of 16.
pub const SKIN_SLOTS: usize = 32;
/// Level textures that morph with strangeness, and the steps each is baked at
/// (0 is the texture itself, `STRANGE_STEPS - 1` is fully strange).
pub const STRANGE_SLOTS: usize = 8;
pub const STRANGE_STEPS: usize = 8;
/// First texture id of the baked strange steps: step k (1..) of slot s is
/// `STRANGE_BASE + (k - 1) * STRANGE_SLOTS + s`.
pub const STRANGE_BASE: u8 = 64;

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

impl Texture {
    /// As 64x64 RGBA8, row-major.
    pub fn rgba(&self) -> Vec<u8> {
        self.idx.iter().flat_map(|&i| {
            let c = self.pal[i as usize];
            [c[0], c[1], c[2], 255]
        }).collect()
    }
}

/// Top-left texel of texture `t` inside its page.
pub fn uv_origin(t: u8) -> (u8, u8) {
    let s = if t >= STRANGE_BASE {
        (t - STRANGE_BASE) as usize % 16
    } else if t >= SKIN_BASE {
        (t - SKIN_BASE) as usize % 16
    } else {
        t as usize
    };
    ((s % 4 * SIZE) as u8, (s / 4 * SIZE) as u8)
}

/// The texture to draw for level texture `t` at strangeness `s` (0..1), and how far
/// down its page it sits relative to `t` (add it to the triangle's v coordinates).
/// Only level slots with baked steps change, and only once `load_strange` has run.
pub fn strange_variant(t: u8, s: f32) -> (u8, u8) {
    let step = (s.clamp(0.0, 1.0) * (STRANGE_STEPS - 1) as f32).round() as usize;
    if step == 0 || (t as usize) >= STRANGE_SLOTS || STRANGE.get().is_none() {
        return (t, 0);
    }
    let id = STRANGE_BASE + ((step - 1) * STRANGE_SLOTS) as u8 + t;
    (id, uv_origin(id).1 - uv_origin(t).1)
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

/// Level textures from images, by slot (see `load_level`).
static LEVEL: std::sync::OnceLock<Vec<Option<Texture>>> = std::sync::OnceLock::new();
/// Baked strange steps 1.. of the level textures: [step - 1][slot] (see `load_strange`).
static STRANGE: std::sync::OnceLock<Vec<Vec<Texture>>> = std::sync::OnceLock::new();

/// A 256x256 RGB PNG box-filtered to 64x64 and cut to 16 colours.
fn image_texture(path: &std::path::Path) -> Option<Texture> {
    let rgb = crate::world::photo::read_rgb(path)?;
    let k = 256 / SIZE;
    let small: Vec<[f32; 3]> = (0..SIZE * SIZE)
        .map(|i| {
            let (x, y) = (i % SIZE * k, i / SIZE * k);
            let mut c = [0.0; 3];
            for dy in 0..k {
                for dx in 0..k {
                    let p = rgb[(y + dy) * 256 + x + dx];
                    (0..3).for_each(|ch| c[ch] += p[ch] / (k * k) as f32);
                }
            }
            c
        })
        .collect();
    Some(crate::world::photo::quantise(&small))
}

/// Take level textures 0-7 from DIR/<slot>.png (256x256 RGB, seamless; see
/// `photo/fetch_ambientcg.py`). Slots without an image keep their drawn texture.
pub fn load_level(dir: &std::path::Path) {
    let level: Vec<_> = (0..STRANGE_SLOTS).map(|t| image_texture(&dir.join(format!("{t}.png")))).collect();
    eprintln!("level textures from {}: {} of 8", dir.display(), level.iter().flatten().count());
    LEVEL.set(level).ok().expect("level textures loaded twice");
}

/// Take the level textures at every strangeness step from DIR/<step>/<slot>.png
/// (see `photo/bake_strange.py`): step 0 is the level itself, as `load_level`.
pub fn load_strange(dir: &std::path::Path) {
    load_level(&dir.join("0"));
    let steps: Vec<Vec<Texture>> = (1..STRANGE_STEPS)
        .map(|k| {
            (0..STRANGE_SLOTS)
                .map(|t| {
                    let path = dir.join(format!("{k}/{t}.png"));
                    image_texture(&path).unwrap_or_else(|| panic!("missing {}", path.display()))
                })
                .collect()
        })
        .collect();
    eprintln!("strange steps from {}: {}", dir.display(), steps.len());
    STRANGE.set(steps).ok().expect("strange textures loaded twice");
}

/// Level texture `id` as drawn: one of `make_all`'s (`base`), or a strange step's
/// (ids from `STRANGE_BASE`, see `strange_variant`). Skins are the scene's own.
pub fn level_texture(base: &[Texture], id: u8) -> Option<&Texture> {
    if id >= STRANGE_BASE {
        let i = (id - STRANGE_BASE) as usize;
        STRANGE.get()?.get(i / STRANGE_SLOTS)?.get(i % STRANGE_SLOTS)
    } else {
        base.get(id as usize)
    }
}

pub fn make_all() -> Vec<Texture> {
    (0..N)
        .map(|t| {
            if let Some(tex) = LEVEL.get().and_then(|l| l.get(t)).and_then(|t| t.as_ref()) {
                return Texture { idx: tex.idx.clone(), pal: tex.pal };
            }
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

/// RGB8 atlas of all textures (256 wide), laid out as on their page.
/// The level textures at every strangeness step, one slot to a row and one step
/// to a column, once `load_strange` has run.
pub fn steps_rgb(base: &[Texture]) -> Option<(u32, u32, Vec<u8>)> {
    STRANGE.get()?;
    let (w, h) = (STRANGE_STEPS * SIZE, STRANGE_SLOTS * SIZE);
    let mut out = vec![0u8; w * h * 3];
    for slot in 0..STRANGE_SLOTS {
        for step in 0..STRANGE_STEPS {
            let id = if step == 0 { slot as u8 } else { STRANGE_BASE + ((step - 1) * STRANGE_SLOTS + slot) as u8 };
            let tex = level_texture(base, id)?;
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let o = ((slot * SIZE + y) * w + step * SIZE + x) * 3;
                    out[o..o + 3].copy_from_slice(&tex.pal[tex.idx[y * SIZE + x] as usize]);
                }
            }
        }
    }
    Some((w as u32, h as u32, out))
}

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
