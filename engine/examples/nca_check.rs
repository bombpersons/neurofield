//! Check the game's flesh automaton (engine::neural::nca) against the trainer's: a step with
//! every cell updating, from the same state, must come out the same.
//!
//!     python train/nca_fixture.py runs/nca/slot1 DIR   # weights, a state and its next
//!     cargo run --release -p engine --example nca_check -- DIR

use engine::neural::nca::{Nca, CHANNELS, SIZE};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("the fixture's directory"));
    let read = |name: &str| -> Vec<f32> { bytemuck::cast_slice(&std::fs::read(dir.join(name)).expect(name)).to_vec() };
    let (weights, before, after) = (read("nca.bin"), read("before.bin"), read("after.bin"));
    assert_eq!(before.len(), SIZE * SIZE * CHANNELS);
    let mut nca = Nca::new(&weights, &before, 1.0).expect("a GPU");
    nca.step(1);
    let got = nca.state().expect("the state");
    let worst = got.iter().zip(&after).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mean = got.iter().zip(&after).map(|(a, b)| (a - b).abs()).sum::<f32>() / got.len() as f32;
    let moved = before.iter().zip(&after).map(|(a, b)| (a - b).abs()).sum::<f32>() / got.len() as f32;
    println!("one step: largest difference {worst:.2e}, mean {mean:.2e} (the step itself moves cells {moved:.2e} on average)");
    let t0 = std::time::Instant::now();
    nca.step(100);
    nca.read().expect("the texture");
    println!("100 steps and a read: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
    assert!(worst < 1e-4, "the game's step isn't the trainer's");
}
