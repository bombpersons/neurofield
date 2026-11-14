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
/// dream creatures over it, as `out/frame_NNNNN.png`. With `--animate 1` the creatures move and
/// the camera follows the one it is watching.
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
    let mut level = Level::new(&mut rng);
    let which = opts.get("creature").map_or(0, |c| c.parse().expect("--creature is a number"));
    let creatures = level.scene.creatures.len();
    let (watched, start) = (0..creatures)
        .map(|k| (which + k) % creatures)
        .find_map(|i| level.walker.view_of(&level.scene, &level.scene.creatures[i], &mut rng).map(|cam| (i, cam)))
        .expect("no creature can be seen");
    let (cx, cz) = (level.scene.creatures[watched].base[0], level.scene.creatures[watched].base[2]);
    let (dx, dz) = (start.pos[0] as f32 - cx, start.pos[2] as f32 - cz);
    let radius = (dx * dx + dz * dz).sqrt();
    let angle = dz.atan2(dx);
    let (mut net_ms, mut dream_ms) = (Vec::new(), Vec::new());
    let affliction: f32 = opts.get("affliction").map_or(0.0, |a| a.parse().expect("--affliction is 0..1"));
    let animate = opts.contains_key("animate");
    let mut herd = engine::world::animate::Herd::default();
    let mut growths = engine::world::growths::Growths::new(&level.scene);
    let (mut swollen, mut lashes, mut most) = (Vec::new(), 0, 0);
    for f in 0..frames {
        if animate {
            herd.update(&mut level.scene, 0, 1.0 / 30.0);
            if f == 0 {
                println!("body plans {:?}, watching {watched}", herd.plans());
            }
        }
        let (cx, cz) = (level.scene.creatures[watched].base[0], level.scene.creatures[watched].base[2]);
        let swing = 0.6 * (f as f32 / frames as f32 * std::f32::consts::TAU).sin();
        let (x, z) = (cx + radius * (angle + swing).cos(), cz + radius * (angle + swing).sin());
        let yaw = yaw_towards(cx - x, cz - z).round() as i32;
        let cam = Camera { pos: [x as i32, start.pos[1], z as i32], yaw, roll: 0, ..start };
        let mut v = Vec::new();
        view_features(&level.scene, 0, &cam, W, &mut v);
        let time = f as f32 / 30.0;
        let swelling = effects::Swelling::of(affliction, time, &cam);
        swelling.tris(&level.tris, &mut swollen);
        let t = tokens::tokens(&swollen, &v);
        lashes += growths.update(&level.scene, &cam, effects::growth(affliction), 1.0 / 30.0, &mut rng).is_some() as u32;
        most = most.max(growths.count());
        let t0 = Instant::now();
        let mut rgb = net.render(&t, &Dream::calm(W));
        net_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let t0 = Instant::now();
        layer.draw(&mut level.scene, 0, &swollen, growths.creature(), |p| swelling.tiles(p), &cam, &mut rgb, W).expect("dream layer");
        dream_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let bytes: Vec<u8> = rgb.iter().map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8).collect();
        crate::write_png(&out.join(format!("frame_{f:05}.png")), W, H, &bytes);
    }
    println!("tentacles: at most {most} out, {lashes} lashes landed");
    for (name, v) in [("level net", &mut net_ms), ("dream layer", &mut dream_ms)] {
        v.sort_by(f64::total_cmp);
        println!("{name}: median {:.1} ms, slowest {:.1} ms", v[v.len() / 2], v[v.len() - 1]);
    }
}

/// A fixture's arrays (train/stage_fixture.py): name -> (shape, values).
fn read_fixture(dir: &Path) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let json = std::fs::read_to_string(dir.join("fixture.json")).expect("fixture.json");
    let mut out = HashMap::new();
    // {"name": [d0, d1, ...], ...}
    for part in json.trim().trim_start_matches('{').trim_end_matches('}').split("]") {
        let Some((name, dims)) = part.split_once(':') else { continue };
        let name = name.trim().trim_start_matches(',').trim().trim_matches('"').to_string();
        let shape: Vec<usize> = dims.trim().trim_start_matches('[').split(',').filter(|d| !d.trim().is_empty()).map(|d| d.trim().parse().unwrap()).collect();
        let bytes = std::fs::read(dir.join(format!("{name}.bin"))).expect("fixture array");
        let values = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        out.insert(name, (shape, values));
    }
    out
}

/// Largest and mean absolute difference.
fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "lengths differ");
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
    (d.iter().copied().fold(0.0, f32::max), d.iter().sum::<f32>() / d.len().max(1) as f32)
}

/// `datagen stage-check`: a sculpt-trained model's port (tokens, organs, the head and the
/// per-pixel stage) against PyTorch, on train/stage_fixture.py's fixture.
pub fn stage_check(opts: &HashMap<String, String>) {
    use engine::neural::gpu_net::GpuNet;
    let model = Path::new(opts.get("model").expect("--model FILE is required"));
    let fx = read_fixture(Path::new(opts.get("fixture").expect("--fixture DIR is required")));
    let get = |k: &str| &fx.get(k).unwrap_or_else(|| panic!("the fixture has no {k}")).1;
    let turned = tokens::Turned { amount: get("amount")[0], turning: get("turning"), organs: get("organs") };
    let t = tokens::tokens_turned(get("tris"), get("view"), Some(&turned));
    // Triangles as near as each other may come in either order, which the network doesn't mind.
    let sorted = |f: &[f32]| {
        let mut rows: Vec<&[f32]> = f.chunks_exact(tokens::TRI_FEATURES).collect();
        rows.sort_by(|a, b| a.iter().zip(*b).map(|(x, y)| x.total_cmp(y)).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal));
        rows.concat()
    };
    let (m, mean) = diff(&sorted(&t.feats), &sorted(get("feats")));
    println!("triangle tokens: {} (fixture {}), largest difference {m:.2e}, mean {mean:.2e}", t.len(), get("tex").len());
    let (m, mean) = diff(&t.cam, get("cam"));
    println!("camera token: largest difference {m:.2e}, mean {mean:.2e}");
    let (m, mean) = diff(&t.organs, get("organ_feats"));
    println!("organ tokens: {}, largest difference {m:.2e}, mean {mean:.2e}", t.organ_kinds.len());
    let mut net = GpuNet::load(model).expect("loading the model");
    println!("running on {}", net.adapter);
    let dream = engine::neural::dream::Dream::calm(t.width);
    let head = net.render(&t, &dream).expect("rendering");
    let (m, mean) = diff(&head, get("head"));
    println!("head alone: largest difference {m:.2e}, mean {mean:.2e}");
    if !net.has_stage() {
        return;
    }
    let view = engine::neural::gpu_net::StageView { tris: get("tris"), turning: get("turning"), organs: get("organs"), view: get("view"), amount: get("amount")[0] };
    // Twice: the second as fast as it goes in a game.
    let full = net.render_staged(&t, &dream, Some(&view)).expect("rendering with the stage");
    let t0 = Instant::now();
    net.render_staged(&t, &dream, Some(&view)).expect("rendering with the stage");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let g = engine::neural::gbuffer::pieces(view.tris, view.turning, view.view);
    println!(
        "G-buffer: {} pieces of {} triangles, {} places in tiles, {:.1} ms on the CPU",
        g.pieces.len() / engine::neural::gbuffer::PIECE,
        g.rows.len() / engine::neural::gbuffer::ROW,
        g.lists.len(),
        t1.elapsed().as_secs_f64() * 1000.0
    );
    let t2 = Instant::now();
    net.render(&t, &dream).expect("rendering");
    println!("without the stage: {:.1} ms", t2.elapsed().as_secs_f64() * 1000.0);
    // The plain channels against pixels.py's, channel by channel.
    let inputs = net.stage_inputs(t.width).expect("reading the stage's inputs");
    if let Some(out) = opts.get("out") {
        std::fs::create_dir_all(out).unwrap();
        std::fs::write(Path::new(out).join("stage_inputs.bin"), inputs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    let px = H as usize * t.width as usize;
    let want = get("plain");
    let names = ["covered", "1/z", "facing x", "facing y", "facing z", "flesh", "reach", "amount", "fold", "eye", "sphincter", "u", "v", "radius",
                 "state 0", "blink", "gaze x", "gaze y", "vein", "across", "along", "beat"];
    for (c, name) in names.iter().enumerate() {
        let got = &inputs[(16 + c) * px..(17 + c) * px];
        let (m, mean) = diff(got, &want[c * px..(c + 1) * px]);
        let off = got.iter().zip(&want[c * px..(c + 1) * px]).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
        println!("  plain {c:2} {name:9}: largest difference {m:.2e}, mean {mean:.2e}, {off} pixels off by more than 1e-3");
    }
    let (m, mean) = diff(&full, get("full"));
    println!("with the stage: largest difference {m:.2e}, mean {mean:.2e} ({ms:.1} ms a frame)");
    let (m, mean) = diff(&full, get("target"));
    println!("(against the target: largest {m:.2e}, mean {mean:.2e})");
}
