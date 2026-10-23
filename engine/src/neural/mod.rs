//! The neural renderer: the trained network drawing a scene from what it is
//! told about it. `features` and `tokens` build its inputs (the same floats the
//! training data stores), `gpu_net` runs it as WGSL compute shaders through wgpu
//! (Vulkan, Metal or DX12), and `effects` disturb it in dreamlike ways.

pub mod dream;
pub mod effects;
pub mod features;
pub mod gpu_net;
pub mod tokens;

use crate::world::rng::Rng;
use crate::world::scene::Scene;
use crate::world::walkable_scene;
use crate::world::walker::Walker;
use dream::Dream;
use features::{tri_features, view_features};
use gpu_net::GpuNet;
use std::path::Path;

/// The network, loaded from a `.safetensors` export (train/export_weights.py).
pub struct Network {
    net: GpuNet,
}

impl Network {
    pub fn load(path: &Path) -> Network {
        let net = GpuNet::load(path).unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
        eprintln!("running on {}", net.adapter);
        Network { net }
    }

    /// The 320x240 frame, RGB floats (roughly 0..1), row by row.
    pub fn render(&mut self, t: &tokens::Tokens, dream: &Dream) -> Vec<f32> {
        self.net.render(t, dream).expect("inference")
    }
}

/// The network to load: `--model`, else the config's; exits with a hint when it
/// isn't there.
pub fn model_path(opts: &std::collections::HashMap<String, String>, config: &crate::world::config::Config) -> std::path::PathBuf {
    let model = opts.get("model").map(std::path::PathBuf::from).or_else(|| config.model.clone());
    match model {
        Some(m) if m.is_file() => m,
        m => {
            eprintln!(
                "no model at {}: run ./model.sh to export the newest checkpoint (or download the latest release), \
                 or pass --model, or set `model` in neurofield.ron",
                m.map_or("(none set)".into(), |m| m.display().to_string())
            );
            std::process::exit(1);
        }
    }
}

/// Simulation steps per second: the walker moves a fixed amount per step, as in the training data.
pub const TICK: f64 = 30.0;

/// A level being walked, the walker, and the network's view of its triangles.
pub struct Level {
    pub scene: Scene,
    pub walker: Walker,
    /// The scene's triangle features (`features::tri_features`).
    pub tris: Vec<f32>,
}

impl Level {
    pub fn new(rng: &mut Rng) -> Level {
        // Same minimum size as `datagen walk`, so seeds line up.
        let (scene, walker) = walkable_scene(rng, 30);
        let mut tris = Vec::new();
        tri_features(&scene, &mut tris);
        Level { scene, walker, tris }
    }

    /// A hand-built level (`world::level`), with the walker at `start` (world
    /// position and yaw in radians) or else at the level's player start.
    pub fn from_file(level: &crate::world::level::Level, start: Option<([f32; 3], f32)>, rng: &mut Rng) -> Result<Level, String> {
        let scene = level.to_scene();
        let (pos, yaw) = start.or_else(|| level.player_start()).ok_or("the level has no PlayerStart")?;
        let walker = Walker::at(&scene, pos, yaw, rng).ok_or("the start is inside rock")?;
        let mut tris = Vec::new();
        tri_features(&scene, &mut tris);
        Ok(Level { scene, walker, tris })
    }

    /// Set how strange the level is at its base (see `Scene::set_strangeness`).
    pub fn set_strangeness(&mut self, base: f32) {
        self.scene.set_strangeness(base);
        self.tris.clear();
        tri_features(&self.scene, &mut self.tris);
    }

    /// The network's inputs for the walker's current view.
    pub fn tokens(&self) -> tokens::Tokens {
        let mut v = Vec::new();
        view_features(&self.scene, 0, &self.walker.camera(), &mut v);
        tokens::tokens(&self.tris, &v)
    }
}
