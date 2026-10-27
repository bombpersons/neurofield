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
