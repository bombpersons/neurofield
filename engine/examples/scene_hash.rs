//! Prints a hash of each generated scene's triangles, creatures and walk, for
//! checking that a refactor leaves the generator's output unchanged.
//!
//!     cargo run --release --example scene_hash > before.txt

use engine::world::rng::Rng;
use engine::world::scene::Scene;
use std::hash::{Hash, Hasher};

fn main() {
    engine::world::flesh::set_psx(true);
    for seed in 0..120u64 {
        let mut rng = Rng::new(seed);
        let scene = Scene::generate(&mut rng);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for t in scene.tris.iter().chain(&scene.creature_tris) {
            t.tex.hash(&mut h);
            for v in &t.v {
                (v.p, v.uv, v.col, v.alb).hash(&mut h);
            }
        }
        let mut wrng = Rng::new(seed ^ 77);
        if let Some(mut w) = engine::world::walker::Walker::new(&scene, &mut wrng) {
            for _ in 0..300 {
                w.step(&scene, &mut wrng);
                let c = w.camera();
                (c.pos, c.yaw, c.pitch).hash(&mut h);
            }
            for c in &scene.creatures {
                if let Some(cam) = w.view_of(&scene, c, &mut wrng) {
                    (cam.pos, cam.yaw, cam.pitch, cam.roll).hash(&mut h);
                }
            }
        }
        let floors = scene.floor_cells().len();
        let reach = scene.reachable(scene.floor_cells()[0]).len();
        println!("{seed} tris {} creatures {} floors {floors} reach {reach} hash {:016x}", scene.tris.len(), scene.creatures.len(), h.finish());
    }
}
