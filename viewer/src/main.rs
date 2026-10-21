//! Walk a generated level in real time, drawn by the PS1 rasteriser or by the
//! trained network.
//!
//! `viewer [--model runs/colab2/model.onnx] [--seed 7]`
//!     Space: rasteriser / network   B: side by side   Tab: auto-walk / drive
//!     Arrows or WASD: walk and turn (driving)   Q/E: strafe   PgUp/PgDn: look
//!     F / G: next / previous dream effect (slow disturbances of the network)
//!     N: new level   P: pause   Esc: quit
//! `viewer --check N --out DIR [--model ...] [--seed 7] [--effect E]`
//!     No window: the first N frames of the auto-walk, as tokens and network
//!     images in DIR, for comparison with the Python pipeline (with dream
//!     effect E at 30 frames a second, if given).
//!
//! With the same seed the level and auto-walk match `datagen walk --seed S`.

mod effects;
mod net;
mod tokens;

use datagen::render::{self, H, W};
use datagen::rng::Rng;
use datagen::scene::Scene;
use datagen::walker::{Walker, MAX_PITCH};
use datagen::{npy, tex, tri_features, view_features, walkable_scene};
use minifb::{Key, KeyRepeat, ScaleMode, Window, WindowOptions};
use net::Net;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Simulation steps per second: the walker moves a fixed amount per step, as in datagen.
const TICK: f64 = 30.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Raster,
    Network,
    Both,
}

struct Level {
    scene: Scene,
    walker: Walker,
    tris: Vec<f32>,
}

impl Level {
    fn new(rng: &mut Rng) -> Level {
        // Same minimum size as `datagen walk`, so seeds line up.
        let (scene, walker) = walkable_scene(rng, 30);
        let mut tris = Vec::new();
        tri_features(&scene, &mut tris);
        Level { scene, walker, tris }
    }

    fn tokens(&self) -> tokens::Tokens {
        let mut v = Vec::new();
        view_features(0, &self.walker.camera(), &mut v);
        tokens::tokens(&self.tris, &v)
    }
}

fn to_pixel(r: f32, g: f32, b: f32) -> u32 {
    let c = |x: f32| (x.clamp(0.0, 1.0) * 255.0) as u32;
    c(r) << 16 | c(g) << 8 | c(b)
}

/// Copy a 320x240 frame into `buf` (`stride` pixels wide) at column `x0`.
fn blit(buf: &mut [u32], stride: usize, x0: usize, frame: &[u32]) {
    let w = W as usize;
    for (y, row) in frame.chunks_exact(w).enumerate() {
        buf[y * stride + x0..y * stride + x0 + w].copy_from_slice(row);
    }
}

fn check(net: &mut Net, seed: u64, frames: usize, out: &Path, effect: usize) {
    std::fs::create_dir_all(out).unwrap();
    let mut rng = Rng::new(seed);
    let mut level = Level::new(&mut rng);
    let mut images = Vec::new();
    let mut counts = Vec::new();
    for f in 0..frames {
        let mut t = level.tokens();
        let dream = effects::apply(effect, f as f32 / TICK as f32, &mut t, &level.tris);
        npy::write(&out.join(format!("feats_{f:05}.npy")), &[t.len(), tokens::TRI_FEATURES], &t.feats).unwrap();
        npy::write(&out.join(format!("tex_{f:05}.npy")), &[t.len()], &t.tex).unwrap();
        npy::write(&out.join(format!("cam_{f:05}.npy")), &[tokens::CAM_FEATURES], &t.cam).unwrap();
        images.extend(net.render(&t, &dream).expect("inference"));
        counts.push(t.len() as i64);
        level.walker.step(&level.scene, &mut rng);
    }
    npy::write(&out.join("images.npy"), &[frames, H as usize, W as usize, 3], &images).unwrap();
    npy::write(&out.join("counts.npy"), &[frames], &counts).unwrap();
    println!("wrote {frames} frames to {}", out.display());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut opts = HashMap::new();
    for pair in args.chunks(2) {
        match pair {
            [k, v] if k.starts_with("--") => {
                opts.insert(k[2..].to_string(), v.clone());
            }
            _ => panic!("expected --key value pairs, got {pair:?}"),
        }
    }
    let model = PathBuf::from(opts.get("model").map_or("runs/colab2/model.onnx", String::as_str));
    let seed: u64 = opts.get("seed").map_or(7, |s| s.parse().expect("--seed is a number"));
    let mut net = Net::load(&model).unwrap_or_else(|e| panic!("loading {}: {e}", model.display()));
    if let Some(n) = opts.get("check") {
        let out = PathBuf::from(opts.get("out").expect("--check needs --out"));
        let effect = opts.get("effect").map_or(0, |e| e.parse().expect("--effect is a number"));
        check(&mut net, seed, n.parse().expect("--check is a frame count"), &out, effect);
        return;
    }

    let textures = tex::make_all();
    let mut gpu = render::new_gpu(&textures);
    let mut rng = Rng::new(seed);
    let mut level = Level::new(&mut rng);
    let (w, h) = (W as usize, H as usize);
    let mut window = Window::new(
        "neurofield",
        4 * w,
        2 * h,
        WindowOptions { resize: true, scale_mode: ScaleMode::AspectRatioStretch, ..WindowOptions::default() },
    )
    .expect("open window");

    let mut mode = Mode::Network;
    let mut driving = false;
    let mut paused = false;
    let mut effect = 0;
    let mut effect_t = 0.0f32;
    let mut last = Instant::now();
    let mut acc = 0.0;
    let (mut fps, mut net_ms, mut tris) = (0.0, 0.0, 0);
    let mut raster = vec![0u32; w * h];
    let mut neural = vec![0u32; w * h];
    let mut both = vec![0u32; 2 * w * h];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let pressed = |k| window.is_key_pressed(k, KeyRepeat::No);
        if pressed(Key::Space) {
            mode = if mode == Mode::Network { Mode::Raster } else { Mode::Network };
        }
        if pressed(Key::B) {
            mode = if mode == Mode::Both { Mode::Network } else { Mode::Both };
        }
        if pressed(Key::Tab) {
            driving = !driving;
        }
        if pressed(Key::P) {
            paused = !paused;
        }
        if pressed(Key::N) {
            level = Level::new(&mut rng);
        }
        if pressed(Key::F) || pressed(Key::G) {
            let n = effects::EFFECTS.len();
            effect = if pressed(Key::F) { (effect + 1) % n } else { (effect + n - 1) % n };
            effect_t = 0.0;
        }

        // Fixed-rate simulation, catching up (a little) when rendering is slow.
        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64();
        last = now;
        fps = fps * 0.9 + 0.1 / dt.max(1e-6);
        if !paused {
            acc = (acc + dt * TICK).min(4.0);
            effect_t += dt as f32;
        }
        while acc >= 1.0 {
            acc -= 1.0;
            if driving {
                let key = |a: Key, b: Key| (window.is_key_down(a) || window.is_key_down(b)) as i32 as f32;
                let forward = key(Key::Up, Key::W) - key(Key::Down, Key::S);
                let strafe = key(Key::E, Key::E) - key(Key::Q, Key::Q);
                let turn = key(Key::Right, Key::D) - key(Key::Left, Key::A);
                let look = (key(Key::PageDown, Key::PageDown) - key(Key::PageUp, Key::PageUp)) * MAX_PITCH;
                let len = (forward * forward + strafe * strafe).sqrt().max(1.0);
                level.walker.drive(&level.scene, forward / len, strafe / len, turn, look);
            } else {
                level.walker.step(&level.scene, &mut rng);
            }
        }

        let cam = level.walker.camera();
        if mode != Mode::Network {
            let rgb = render::render(&mut gpu, &level.scene, &cam);
            for (px, c) in raster.iter_mut().zip(rgb.chunks_exact(3)) {
                *px = (c[0] as u32) << 16 | (c[1] as u32) << 8 | c[2] as u32;
            }
        }
        if mode != Mode::Raster {
            let t0 = Instant::now();
            let mut t = level.tokens();
            let dream = effects::apply(effect, effect_t, &mut t, &level.tris);
            tris = t.len();
            let rgb = net.render(&t, &dream).expect("inference");
            for (px, c) in neural.iter_mut().zip(rgb.chunks_exact(3)) {
                *px = to_pixel(c[0], c[1], c[2]);
            }
            net_ms = net_ms * 0.9 + 0.1 * t0.elapsed().as_secs_f64() * 1000.0;
        }

        let name = match mode {
            Mode::Raster => "rasteriser",
            Mode::Network => "network",
            Mode::Both => "rasteriser | network",
        };
        let how = if driving { "driving" } else { "auto-walk" };
        let pause = if paused { " | paused" } else { "" };
        let stale =
            if effects::needs_dream_inputs(effect) && !net.dreams() { " (needs a re-exported model)" } else { "" };
        let dream = format!("effect {effect}: {}{stale}", effects::EFFECTS[effect]);
        window.set_title(&format!(
            "neurofield | {name} | {dream} | {fps:.0} fps | net {net_ms:.0} ms, {tris} tris | {how}{pause} \
             | space: switch  b: both  tab: drive  f/g: effect  n: new level"
        ));
        let shown = match mode {
            Mode::Raster => window.update_with_buffer(&raster, w, h),
            Mode::Network => window.update_with_buffer(&neural, w, h),
            Mode::Both => {
                blit(&mut both, 2 * w, 0, &raster);
                blit(&mut both, 2 * w, w, &neural);
                window.update_with_buffer(&both, 2 * w, h)
            }
        };
        shown.expect("draw frame");
    }
}
