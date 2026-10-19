//! PS1-style training data for a neural renderer.
//!
//! `datagen gen --out DIR [--scenes N] [--views K] [--shard-scenes S] [--seed X] [--camera fp|free]`
//!     Random scenes, K views each, in shards of S scenes. `fp` (default) samples
//!     views from King's Field-style first-person walks; `free` uses random cameras.
//! `datagen walk --out DIR [--frames N] [--seed X]`
//!     One scene and a continuous first-person walk through it.
//! `datagen textures --out FILE.png`
//!     The fixed texture atlas, as the network would have to memorise it.
//!
//! See `format.json` in each output directory for the array layouts.

mod npy;
mod render;
mod rng;
mod scene;
mod tex;
mod walker;

use rayon::prelude::*;
use rng::Rng;
use scene::{Camera, Scene, T};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walker::Walker;

const TRI_FEATURES: usize = 25;
const VIEW_FEATURES: usize = 19;

const FORMAT: &str = r#"{
  "units": "positions and fog distances in tiles (256 world units); y points down, the lowest floor is y = 0 and floors rise to y = -1",
  "tris.npy": "float32 [T, 25] per triangle: v0 xyz, v1 xyz, v2 xyz (0-8); v0 uv, v1 uv, v2 uv in texture tiles, 0..1 (9-14); texture id 0-7 (15); v0 rgb, v1 rgb, v2 rgb baked vertex colour / 255, where 128/255 is neutral texture modulation (16-24)",
  "tri_offsets.npy": "int64 [S + 1]: scene s owns tris[tri_offsets[s] : tri_offsets[s + 1]]",
  "views.npy": "float32 [V, 19] per view: scene index within the shard (0); camera position xyz (1-3); world-to-view rotation, row-major (4-12; row 2 is the view direction); projection distance H in pixels, screen centre (160, 120) (13); fog near, fog far (14-15); yaw, pitch, roll in turns (16-18)",
  "images.npy": "uint8 [V, 240, 320, 3]",
  "projection": "view = R (p - pos); screen = (160 + H * view.x / view.z, 120 + H * view.y / view.z)"
}
"#;

fn tri_features(scene: &Scene, out: &mut Vec<f32>) {
    for t in &scene.tris {
        for v in &t.v {
            out.extend(v.p.map(|c| c as f32 / T as f32));
        }
        let (u0, v0) = tex::uv_origin(t.tex);
        for v in &t.v {
            out.push((v.uv[0] - u0) as f32 / tex::SIZE as f32);
            out.push((v.uv[1] - v0) as f32 / tex::SIZE as f32);
        }
        out.push(t.tex as f32);
        for v in &t.v {
            out.extend(v.col.map(|c| c as f32 / 255.0));
        }
    }
}

fn view_features(scene_index: usize, cam: &Camera, out: &mut Vec<f32>) {
    out.push(scene_index as f32);
    out.extend(cam.pos.map(|c| c as f32 / T as f32));
    let m = render::view_matrix(cam);
    out.extend(m.m.iter().flatten().map(|&e| e as f32 / 4096.0));
    out.push(cam.h as f32);
    out.push(cam.fog_near as f32 / T as f32);
    out.push(cam.fog_far as f32 / T as f32);
    out.extend([cam.yaw, cam.pitch, cam.roll].map(|a| a as f32 / 4096.0));
}

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
    views: Vec<f32>,
    images: Vec<u8>,
    frames: Vec<Vec<u8>>,
}

impl Batch {
    fn new() -> Self {
        Batch { tris: Vec::new(), offsets: vec![0], views: Vec::new(), images: Vec::new(), frames: Vec::new() }
    }

    fn add_scene(&mut self, scene: &Scene) -> usize {
        tri_features(scene, &mut self.tris);
        self.offsets.push((self.tris.len() / TRI_FEATURES) as i64);
        self.offsets.len() - 2
    }

    fn add_view(&mut self, scene_index: usize, cam: &Camera, frame: Vec<u8>) {
        view_features(scene_index, cam, &mut self.views);
        self.images.extend_from_slice(&frame);
        if self.frames.len() < 16 {
            self.frames.push(frame);
        }
    }

    fn write(&self, dir: &Path) {
        std::fs::create_dir_all(dir).expect("create output dir");
        let nv = self.views.len() / VIEW_FEATURES;
        npy::write(&dir.join("tris.npy"), &[self.tris.len() / TRI_FEATURES, TRI_FEATURES], &self.tris).unwrap();
        npy::write(&dir.join("tri_offsets.npy"), &[self.offsets.len()], &self.offsets).unwrap();
        npy::write(&dir.join("views.npy"), &[nv, VIEW_FEATURES], &self.views).unwrap();
        npy::write(&dir.join("images.npy"), &[nv, render::H as usize, render::W as usize, 3], &self.images).unwrap();
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

/// A level the first-person walker can move around in, and the walker.
fn walkable_scene(rng: &mut Rng, min_cells: usize) -> (Scene, Walker) {
    loop {
        let scene = Scene::generate(rng);
        if scene.floor_cells().len() < min_cells {
            continue;
        }
        if let Some(w) = Walker::new(&scene, rng) {
            return (scene, w);
        }
    }
}

fn gen(out: &Path, scenes: usize, views: usize, shard_scenes: usize, seed: u64, mode: CameraMode) {
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
                        batch.add_view(si, &cam, render::render(&mut gpu, &scene, &cam));
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
                        let cam = walker.camera();
                        batch.add_view(si, &cam, render::render(&mut gpu, &scene, &cam));
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
        let frame = render::render(&mut gpu, &scene, &cam);
        write_png(&out.join(format!("frames/{f:05}.png")), render::W, render::H, &frame);
        batch.add_view(si, &cam, frame);
        walker.step(&scene, &mut rng);
    }
    batch.frames.clear();
    batch.write(out);
    std::fs::write(out.join("format.json"), FORMAT).unwrap();
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
            );
        }
        "walk" => walk(&out, num("frames", 300) as usize, num("seed", 7)),
        "textures" => {
            let (w, h, rgb) = tex::atlas_rgb(&tex::make_all());
            write_png(&out, w, h, &rgb);
        }
        _ => {
            eprintln!("usage: datagen gen|walk|textures --out PATH [...]");
            std::process::exit(2);
        }
    }
}
