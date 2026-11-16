//! Times the creature net (`DreamNet::draw`) over a whole crop, every tile drawn, and writes
//! its layer as raw f32s, to compare kernels by:
//!
//!     cargo run --release -p engine --example dream_net_bench -- NET W H [OUT]
use engine::neural::dream_net::{DreamNet, Q_PARAMS};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut net = DreamNet::load(Path::new(&a[1])).unwrap_or_else(|e| panic!("{e}"));
    let (w, h): (usize, usize) = (a[2].parse().expect("W"), a[3].parse().expect("H"));
    // Two parts side by side in front of the eye, filling much of the crop.
    let mut q = vec![0f32; Q_PARAMS * 2];
    for (i, p) in q.chunks_exact_mut(Q_PARAMS).enumerate() {
        p[..3].copy_from_slice(&[(i as f32 - 0.5) * 0.6, 0.1, 1.2]);
        p[3..6].copy_from_slice(&[1.0, 0.0, 0.0]);
        p[6..9].copy_from_slice(&[0.0, 1.0, 0.0]);
        p[9..12].copy_from_slice(&[0.25, 0.5, 0.25]);
        p[12] = 0.2;
        p[15..18].copy_from_slice(&[1.0, 0.9, 0.8]);
        p[20] = i as f32 * 0.37;
        p[21] = 1.0;
    }
    let palette = vec![0.5f32; 12 * 2];
    let cam = [180.0, w as f32, 4.0, 12.0, 0.0, 0.0];
    let mut times = Vec::new();
    let mut layer = None;
    for k in 0..13 {
        let t = std::time::Instant::now();
        let l = net.draw(&q, &palette, cam, 0, 0, w, h, None).unwrap_or_else(|e| panic!("{e}"));
        if k >= 3 {
            times.push(t.elapsed().as_secs_f32() * 1000.0);
        }
        layer = Some(l);
    }
    times.sort_by(f32::total_cmp);
    let l = layer.expect("drawn");
    let covered = l.data.chunks_exact(6).filter(|p| p[3] > 0.05).count();
    println!("{w} x {h}: median {:.2} ms, fastest {:.2} ms; {covered} pixels covered", times[times.len() / 2], times[0]);
    if let Some(out) = a.get(4) {
        std::fs::write(out, bytemuck::cast_slice(&l.data)).expect("written");
    }
}
