//! A flesh automaton run for a long while, as the game runs it: its texture (RGBA8, colour
//! and height) written after each of the step counts given, to see whether it settles.
//!
//!     cargo run --release -p engine --example nca_run -- runs/nca/slot1/nca.bin OUT 256 4096 16384

use engine::neural::nca::{Nca, SIZE};

fn main() {
    let mut args = std::env::args().skip(1);
    let bin = std::path::PathBuf::from(args.next().expect("nca.bin"));
    let out = std::path::PathBuf::from(args.next().expect("where to write"));
    let at: Vec<usize> = args.map(|a| a.parse().expect("step counts")).collect();
    std::fs::create_dir_all(&out).unwrap();
    let mut nca = Nca::load(&bin, 1).expect("the automaton");
    let mut done = 0;
    for n in at {
        nca.step(n - done);
        done = n;
        std::fs::write(out.join(format!("{n}.rgba")), nca.read().unwrap()).unwrap();
        println!("{n} steps ({SIZE}x{SIZE})");
    }
}
