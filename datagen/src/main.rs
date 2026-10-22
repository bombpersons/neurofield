//! PS1-style training data for a neural renderer.
//!
//! `datagen gen --out DIR [--scenes N] [--views K] [--shard-scenes S] [--seed X] [--camera fp|free] [--creature-views P]`
//!     Random scenes, K views each, in shards of S scenes. `fp` (default) samples
//!     views from King's Field-style first-person walks; `free` uses random cameras.
//!     With `fp`, a share P (default 0.5) of views instead stand near a creature
//!     and look at it, since walks alone rarely get close to one.
//! `datagen walk --out DIR [--frames N] [--seed X]`
//!     One scene and a continuous first-person walk through it.
//! `datagen textures --out FILE.png`
//!     The fixed texture atlas, as the network would have to memorise it.
//! `datagen skins --out FILE.png`
//!     Sample creature skins: per motif, three densities, two seeds each.
//! `datagen creatures --out FILE.png [--seed X]`
//!     16 close views of creatures with their masks, for eyeballing the generator.
//!
//! See `format.json` in each output directory for the array layouts.

use datagen::rng::Rng;
use datagen::scene::{Camera, Scene};
use datagen::{
    npy, part_features, render, tex, tri_features, view_features, walkable_scene, PART_FEATURES, TRI_FEATURES,
    VIEW_FEATURES,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};


const FORMAT: &str = r#"{
  "units": "positions and distances in tiles (256 world units); y points down, the lowest floor is y = 0 and floors rise to y = -1",
  "tris.npy": "float32 [T, 34] per level triangle (creatures are not in here; see parts.npy): v0 xyz, v1 xyz, v2 xyz (0-8); v0 uv, v1 uv, v2 uv in texture tiles, 0..1 (9-14); texture id 0-11 (15); v0 rgb, v1 rgb, v2 rgb baked vertex colour / 255 (16-24); v0 rgb, v1 rgb, v2 rgb unlit surface tint / 255 (25-33). On both colour scales 128/255 is neutral texture modulation. Triangles are wound so (v1 - v0) x (v2 - v0) is the outward normal",
  "textures": "0-7 level (brick, stone, planks, tiles, moss, grate, wallpaper, rings), 8 scales, 9 bone, 10 fur, 11 door; see `datagen textures`",
  "tri_offsets.npy": "int64 [S + 1]: scene s owns tris[tri_offsets[s] : tri_offsets[s + 1]]",
  "parts.npy": "float32 [P, 49] per creature part, world space: joint xyz (0-2); centre xyz (3-5); local axes as the columns of a row-major 3x3 (6-14; y runs from the joint end to the far end); radius x, half-length y, radius z (15-17); bounding box in the local frame, lo xyz then hi xyz (18-23); shape 0 ellipsoid, 1 capsule, 2 cone, 3 tentacle (24); taper, bend, bulge, lumps (25-28); skin motif 0-10 (29; flesh, eyes, teeth, mouths, suckers, veins, scales, chitin, hair, wrinkles, bone); skin density 0..1 (30); tiles per skin repeat (31); tint rgb, 1 = neutral (32-34); baked light at the centre as linear spherical harmonics, 1 = neutral: mean rgb (35-37), then the gradient along world x, y, z, rgb each (38-46); parent part index within the scene, -1 for a root (47); creature index within the scene (48)",
  "part_offsets.npy": "int64 [S + 1]: scene s owns parts[part_offsets[s] : part_offsets[s + 1]]",
  "views.npy": "float32 [V, 23] per view: scene index within the shard (0); camera position xyz (1-3); world-to-view rotation, row-major (4-12; row 2 is the view direction); projection distance H in pixels, screen centre (160, 120) (13); fog near, fog far (14-15); yaw, pitch, roll in turns (16-18); torch rgb strength, 0 = no torch (19-21); torch reach (22)",
  "images.npy": "uint8 [V, 240, 320, 3]",
  "masks.npy": "uint8 [V, 240, 40]: creature pixels, 1 bit per pixel, most significant bit leftmost (np.unpackbits(masks, axis=-1))",
  "projection": "view = R (p - pos); screen = (160 + H * view.x / view.z, 120 + H * view.y / view.z); vertices nearer than view.z = H / 2 are not drawn",
  "shading": "per vertex: colour = baked + tint * torch * max(0, 1 - d / reach)^2 * max(0, n . l), d and l toward the camera; then times fog = 1 - clamp((view.z - fog near) / (fog far - fog near), 0, 1); then texel * colour / 128"
}
"#;

fn write_png(path: &Path, w: u32, h: u32, rgb: &[u8]) {
    let file = std::io::BufWriter::new(std::fs::File::create(path).expect("create png"));
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().and_then(|mut wr| wr.write_image_data(rgb)).expect("write png");
}

/// Up to 16 frames in a 4x4 grid.
fn contact_sheet(path: &Path, frames: &[Vec<u8>]) {
    let (w, h) = (render::W as usize, render::H as usize);
    let mut sheet = vec![0u8; 16 * w * h * 3];
    for (i, f) in frames.iter().take(16).enumerate() {
        let (ox, oy) = (i % 4 * w, i / 4 * h);
        for y in 0..h {
            let dst = ((oy + y) * 4 * w + ox) * 3;
            sheet[dst..dst + w * 3].copy_from_slice(&f[y * w * 3..(y + 1) * w * 3]);
        }
    }
    write_png(path, 4 * w as u32, 4 * h as u32, &sheet);
}

struct Batch {
    tris: Vec<f32>,
    offsets: Vec<i64>,
    parts: Vec<f32>,
    part_offsets: Vec<i64>,
    views: Vec<f32>,
    images: Vec<u8>,
    masks: Vec<u8>,
    frames: Vec<Vec<u8>>,
}

/// Packs a 0/1 mask eight pixels to a byte, leftmost in the high bit.
fn pack_bits(mask: &[u8]) -> impl Iterator<Item = u8> + '_ {
    mask.chunks(8).map(|c| c.iter().fold(0u8, |acc, &b| acc << 1 | b))
}

impl Batch {
    fn new() -> Self {
        Batch {
            tris: Vec::new(),
            offsets: vec![0],
            parts: Vec::new(),
            part_offsets: vec![0],
            views: Vec::new(),
            images: Vec::new(),
            masks: Vec::new(),
            frames: Vec::new(),
        }
    }

    fn add_scene(&mut self, scene: &Scene) -> usize {
        tri_features(scene, &mut self.tris);
        self.offsets.push((self.tris.len() / TRI_FEATURES) as i64);
        part_features(scene, &mut self.parts);
        self.part_offsets.push((self.parts.len() / PART_FEATURES) as i64);
        self.offsets.len() - 2
    }

    /// Render `cam`'s view of `scene` and its creature mask, and keep both.
    fn add_view(&mut self, gpu: &mut render::Gpu, scene_index: usize, scene: &Scene, cam: &Camera) -> Vec<u8> {
        let frame = render::render(gpu, scene, cam);
        let mask = render::creature_mask(gpu, scene, cam);
        view_features(scene_index, cam, &mut self.views);
        self.images.extend_from_slice(&frame);
        self.masks.extend(pack_bits(&mask));
        if self.frames.len() < 16 {
            self.frames.push(frame.clone());
        }
        frame
    }

    fn write(&self, dir: &Path) {
        std::fs::create_dir_all(dir).expect("create output dir");
        let nv = self.views.len() / VIEW_FEATURES;
        npy::write(&dir.join("tris.npy"), &[self.tris.len() / TRI_FEATURES, TRI_FEATURES], &self.tris).unwrap();
        npy::write(&dir.join("tri_offsets.npy"), &[self.offsets.len()], &self.offsets).unwrap();
        npy::write(&dir.join("parts.npy"), &[self.parts.len() / PART_FEATURES, PART_FEATURES], &self.parts).unwrap();
        npy::write(&dir.join("part_offsets.npy"), &[self.part_offsets.len()], &self.part_offsets).unwrap();
        npy::write(&dir.join("views.npy"), &[nv, VIEW_FEATURES], &self.views).unwrap();
        npy::write(&dir.join("images.npy"), &[nv, render::H as usize, render::W as usize, 3], &self.images).unwrap();
        npy::write(&dir.join("masks.npy"), &[nv, render::H as usize, render::W as usize / 8], &self.masks).unwrap();
        contact_sheet(&dir.join("preview.png"), &self.frames);
    }
}

/// How training views are chosen.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    /// Frames sampled from first-person walks (eye height, lens and level gaze fixed).
    FirstPerson,
    /// Independent random cameras with varied height, tilt, roll and lens.
    Free,
}

/// Somewhere on the creature to look at: its root part's centre.
fn creature_target(scene: &Scene, rng: &mut Rng) -> Option<[f32; 3]> {
    let c = scene.creatures.get(rng.next() as usize % scene.creatures.len().max(1))?;
    Some(c.pose()[0].centre)
}

#[allow(clippy::too_many_arguments)]
fn gen(out: &Path, scenes: usize, views: usize, shard_scenes: usize, seed: u64, mode: CameraMode, creature_views: f32) {
    let textures = tex::make_all();
    let shards = scenes.div_ceil(shard_scenes);
    (0..shards).into_par_iter().for_each(|s| {
        let mut gpu = render::new_gpu(&textures);
        let mut batch = Batch::new();
        for i in s * shard_scenes..((s + 1) * shard_scenes).min(scenes) {
            let mut rng = Rng::new(seed.wrapping_mul(1_000_003) ^ i as u64);
            match mode {
                CameraMode::Free => {
                    let scene = Scene::generate(&mut rng);
                    let si = batch.add_scene(&scene);
                    for _ in 0..views {
                        let cam = scene.random_camera(&mut rng);
                        batch.add_view(&mut gpu, si, &scene, &cam);
                    }
                }
                CameraMode::FirstPerson => {
                    let (scene, mut walker) = walkable_scene(&mut rng, 6);
                    let si = batch.add_scene(&scene);
                    for _ in 0..rng.range(0, 60) {
                        walker.step(&scene, &mut rng);
                    }
                    for _ in 0..views {
                        for _ in 0..rng.range(6, 24) {
                            walker.step(&scene, &mut rng);
                        }
                        let near = if rng.chance(creature_views) {
                            creature_target(&scene, &mut rng).and_then(|t| walker.view_of(&scene, t, &mut rng))
                        } else {
                            None
                        };
                        let cam = near.unwrap_or_else(|| walker.camera());
                        batch.add_view(&mut gpu, si, &scene, &cam);
                    }
                }
            }
        }
        batch.write(&out.join(format!("shard_{s:04}")));
        eprintln!("shard {s:04} done");
    });
    std::fs::write(out.join("format.json"), FORMAT).unwrap();
}

/// One level and a continuous first-person walk through it.
fn walk(out: &Path, frames: usize, seed: u64) {
    let textures = tex::make_all();
    let mut gpu = render::new_gpu(&textures);
    let mut rng = Rng::new(seed);
    // Walks want room to move.
    let (scene, mut walker) = walkable_scene(&mut rng, 30);
    let mut batch = Batch::new();
    let si = batch.add_scene(&scene);
    std::fs::create_dir_all(out.join("frames")).unwrap();
    for f in 0..frames {
        let cam = walker.camera();
        let frame = batch.add_view(&mut gpu, si, &scene, &cam);
        write_png(&out.join(format!("frames/{f:05}.png")), render::W, render::H, &frame);
        walker.step(&scene, &mut rng);
    }
    batch.frames.clear();
    batch.write(out);
    std::fs::write(out.join("format.json"), FORMAT).unwrap();
}

/// 16 close views of creatures from random scenes, each with its mask tinted red
/// beside it, 8 rows of two pairs.
fn creatures(out: &Path, seed: u64) {
    let textures = tex::make_all();
    let mut gpu = render::new_gpu(&textures);
    let mut rng = Rng::new(seed);
    let mut frames = Vec::new();
    while frames.len() < 32 {
        let (scene, walker) = walkable_scene(&mut rng, 6);
        let Some(cam) = creature_target(&scene, &mut rng).and_then(|t| walker.view_of(&scene, t, &mut rng)) else {
            continue;
        };
        let frame = render::render(&mut gpu, &scene, &cam);
        let mask = render::creature_mask(&mut gpu, &scene, &cam);
        let tinted = frame.chunks(3).zip(&mask).flat_map(|(p, &m)| if m == 1 { [p[0] / 2 + 128, p[1] / 2, p[2] / 2] } else { [p[0] / 3, p[1] / 3, p[2] / 3] }).collect();
        eprintln!("{} creature tris, {} skins", scene.creature_tris.len(), scene.skins.len());
        frames.push(frame);
        frames.push(tinted);
    }
    let (w, h) = (render::W as usize, render::H as usize);
    let mut sheet = vec![0u8; 32 * w * h * 3];
    for (i, f) in frames.iter().enumerate() {
        let (ox, oy) = (i % 4 * w, i / 4 * h);
        for y in 0..h {
            let dst = ((oy + y) * 4 * w + ox) * 3;
            sheet[dst..dst + w * 3].copy_from_slice(&f[y * w * 3..(y + 1) * w * 3]);
        }
    }
    write_png(out, 4 * w as u32, 8 * h as u32, &sheet);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let mut opts = HashMap::new();
    for pair in args[2.min(args.len())..].chunks(2) {
        match pair {
            [k, v] if k.starts_with("--") => {
                opts.insert(k[2..].to_string(), v.clone());
            }
            _ => panic!("expected --key value pairs, got {pair:?}"),
        }
    }
    let num = |k: &str, d: u64| opts.get(k).map_or(d, |v| v.parse().expect("numeric option"));
    let out = PathBuf::from(opts.get("out").expect("--out is required"));
    match cmd {
        "gen" => {
            let mode = match opts.get("camera").map(String::as_str) {
                None | Some("fp") => CameraMode::FirstPerson,
                Some("free") => CameraMode::Free,
                Some(m) => panic!("--camera must be fp or free, got {m}"),
            };
            std::fs::create_dir_all(&out).unwrap();
            gen(
                &out,
                num("scenes", 1000) as usize,
                num("views", 8) as usize,
                num("shard-scenes", 100) as usize,
                num("seed", 1),
                mode,
                opts.get("creature-views").map_or(0.5, |v| v.parse().expect("numeric option")),
            );
        }
        "walk" => walk(&out, num("frames", 300) as usize, num("seed", 7)),
        "textures" => {
            let (w, h, rgb) = tex::atlas_rgb(&tex::make_all());
            write_png(&out, w, h, &rgb);
        }
        "creatures" => creatures(&out, num("seed", 3)),
        "skins" => {
            let (w, h, rgb) = datagen::skin::sheet();
            write_png(&out, w, h, &rgb);
        }
        _ => {
            eprintln!("usage: datagen gen|walk|textures|skins|creatures --out PATH [...]");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod debug {
    use super::*;
    use datagen::scene::{self, Cell};

    /// Print one generated level: per cell, floor level (in steps above 0) / clearance, or #.
    #[test]
    #[ignore]
    fn dump_level() {
        let i: u64 = std::env::var("SCENE").unwrap().parse().unwrap();
        let mut rng = Rng::new(1u64.wrapping_mul(1_000_003) ^ i);
        let (scene, _) = walkable_scene(&mut rng, 6);
        for z in 0..scene.d {
            let row: Vec<String> = (0..scene.w)
                .map(|x| match scene.cell(x, z) {
                    Cell::Solid { .. } => "  ## ".into(),
                    Cell::Floor { y0, y1, height, slope, .. } => {
                        format!("{}{}/{}{} ", -y0 / 64, -y1 / 64, height, if slope == scene::Slope::Flat { ' ' } else { '~' })
                    }
                })
                .collect();
            println!("z{z:2} {}", row.join(""));
        }
        for r in &scene.rooms {
            println!("room x{} z{} w{} d{} floor {} h {}", r.x, r.z, r.w, r.d, r.floor / 64, r.height);
        }
    }
}
