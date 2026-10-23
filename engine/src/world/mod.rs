//! The world: dungeons (generated or hand-built), what is in them (props,
//! doors, creatures with their skins), their textures and light, and walking
//! around them. Everything the renderers draw and the network is told about.

pub mod config;
pub mod creature;
pub mod level;
pub mod door;
pub mod flesh;
pub mod photo;
pub mod view;
pub mod rng;
pub mod scene;
pub mod sculpt;
pub mod skin;
pub mod tex;
pub mod walker;

use rng::Rng;
use scene::Scene;
use walker::Walker;

/// A level the first-person walker can move around in, and the walker.
pub fn walkable_scene(rng: &mut Rng, min_cells: usize) -> (Scene, Walker) {
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
