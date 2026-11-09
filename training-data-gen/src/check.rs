//! `datagen check`: the trained network on its own, for checking it against the
//! Python pipeline (to walk a level with it, use the game: `cargo run -p game`).

use crate::npy;
use engine::neural::{effects, tokens, Level, Network, TICK};
use engine::world::rng::Rng;
use engine::world::view::{H, W};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// The first `frames` frames of the auto-walk with `--seed`, as tokens and network images in `out`.
pub fn run(opts: &HashMap<String, String>, frames: usize, out: &Path) {
    let config = engine::world::config::Config::load();
    let model = engine::neural::model_path(opts, &config);
    let seed: u64 = opts.get("seed").map_or(7, |s| s.parse().expect("--seed is a number"));
    let effect = opts.get("effect").map_or(0, |e| e.parse().expect("--effect is a number"));
    let width = opts.get("width").map_or(W, |w| w.parse().expect("--width is a number of pixels"));
    let mut net = Network::load(&model);
    check(&mut net, seed, frames, out, effect, width);
}

fn check(net: &mut Network, seed: u64, frames: usize, out: &Path, effect: usize, width: u32) {
    std::fs::create_dir_all(out).unwrap();
    let mut rng = Rng::new(seed);
    let mut level = Level::new(&mut rng);
    let mut images = Vec::new();
    let mut counts = Vec::new();
    let mut times = Vec::new();
    for f in 0..frames {
        let mut t = level.tokens_at(width);
        let dream = effects::apply(effect, f as f32 / TICK as f32, &mut t, &level.tris);
        npy::write(&out.join(format!("feats_{f:05}.npy")), &[t.len(), tokens::TRI_FEATURES], &t.feats).unwrap();
        npy::write(&out.join(format!("tex_{f:05}.npy")), &[t.len()], &t.tex).unwrap();
        npy::write(&out.join(format!("cam_{f:05}.npy")), &[tokens::CAM_FEATURES], &t.cam).unwrap();
        let t0 = Instant::now();
        images.extend(net.render(&t, &dream));
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        counts.push(t.len() as i64);
        level.walker.step(&level.scene, &mut rng);
    }
    npy::write(&out.join("images.npy"), &[frames, H as usize, width as usize, 3], &images).unwrap();
    npy::write(&out.join("counts.npy"), &[frames], &counts).unwrap();
    println!("wrote {frames} frames to {}", out.display());
    let (first, rest) = times.split_at_mut(frames.min(1));
    rest.sort_by(f64::total_cmp);
    if let (Some(first), Some(median)) = (first.first(), rest.get(rest.len() / 2)) {
        println!(
            "network: first frame {first:.0} ms, then median {median:.1} ms, slowest {:.0} ms",
            rest[rest.len() - 1]
        );
    }
}

/// `datagen dream-check`: the dream net's WGSL port against PyTorch (train/export_dream.py --fixture).
pub fn dream_check(opts: &HashMap<String, String>) {
    let dream = Path::new(opts.get("dream").expect("--dream FILE is required"));
    let fixture = Path::new(opts.get("fixture").expect("--fixture FILE is required"));
    let mut net = engine::neural::dream_net::DreamNet::load(dream).expect("loading the dream net");
    println!("running on {}", net.adapter);
    print!("{}", net.check(fixture).expect("comparing"));
}

/// `datagen dream-walk`: frames of a camera swinging round a creature, the level network with the
/// dream creatures over it, as `out/frame_NNNNN.png`.
pub fn dream_walk(opts: &HashMap<String, String>, frames: usize, out: &Path) {
    use engine::neural::dream::Dream;
    use engine::neural::dream_net::DreamLayer;
    use engine::neural::features::view_features;
    use engine::world::scene::Camera;
    use engine::world::walker::yaw_towards;

    let config = engine::world::config::Config::load();
    let model = engine::neural::model_path(opts, &config);
    let seed: u64 = opts.get("seed").map_or(7, |s| s.parse().expect("--seed is a number"));
    let dream = Path::new(opts.get("dream").expect("--dream FILE is required"));
    std::fs::create_dir_all(out).unwrap();
    let mut net = Network::load(&model);
    let mut layer = DreamLayer::load(dream, seed).expect("loading the dream net");
    let mut rng = Rng::new(seed);
    let level = Level::new(&mut rng);
    let which = opts.get("creature").map_or(0, |c| c.parse().expect("--creature is a number"));
    let (creature, start) = level
        .scene
        .creatures
        .iter()
        .cycle()
        .skip(which)
        .take(level.scene.creatures.len())
        .find_map(|c| level.walker.view_of(&level.scene, c, &mut rng).map(|cam| (c, cam)))
        .expect("no creature can be seen");
    let (cx, cz) = (creature.base[0], creature.base[2]);
    let (dx, dz) = (start.pos[0] as f32 - cx, start.pos[2] as f32 - cz);
    let radius = (dx * dx + dz * dz).sqrt();
    let angle = dz.atan2(dx);
    let (mut net_ms, mut dream_ms) = (Vec::new(), Vec::new());
    for f in 0..frames {
        let swing = 0.6 * (f as f32 / frames as f32 * std::f32::consts::TAU).sin();
        let (x, z) = (cx + radius * (angle + swing).cos(), cz + radius * (angle + swing).sin());
        let yaw = yaw_towards(cx - x, cz - z).round() as i32;
        let cam = Camera { pos: [x as i32, start.pos[1], z as i32], yaw, roll: 0, ..start };
        let mut v = Vec::new();
        view_features(&level.scene, 0, &cam, W, &mut v);
        let t = tokens::tokens(&level.tris, &v);
        let t0 = Instant::now();
        let mut rgb = net.render(&t, &Dream::calm(W));
        net_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let t0 = Instant::now();
        layer.draw(&level.scene, 0, &cam, &mut rgb, W).expect("dream layer");
        dream_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let bytes: Vec<u8> = rgb.iter().map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8).collect();
        crate::write_png(&out.join(format!("frame_{f:05}.png")), W, H, &bytes);
    }
    for (name, v) in [("level net", &mut net_ms), ("dream layer", &mut dream_ms)] {
        v.sort_by(f64::total_cmp);
        println!("{name}: median {:.1} ms, slowest {:.1} ms", v[v.len() / 2], v[v.len() - 1]);
    }
}
