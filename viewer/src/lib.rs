//! The trained renderer at run time, shared by the viewer and the game: building the
//! network's inputs from a scene (tokens), the dream effects, and running the network on
//! the CPU (ONNX Runtime) or the GPU (WGSL compute shaders through wgpu).

pub mod effects;
pub mod gpu_net;
pub mod net;
pub mod tokens;

use datagen::rng::Rng;
use datagen::scene::Scene;
use datagen::walker::Walker;
use datagen::{tri_features, view_features, walkable_scene};
use gpu_net::GpuNet;
use net::{Dream, Net};
use std::path::Path;

/// The network, on ONNX Runtime (CPU) or the WGSL shaders (GPU), by the model file's type.
pub enum Network {
    Ort(Net),
    Gpu(GpuNet),
}

impl Network {
    pub fn load(path: &Path) -> Network {
        let fail = |e: String| -> ! { panic!("loading {}: {e}", path.display()) };
        if path.extension().is_some_and(|e| e == "safetensors") {
            let net = GpuNet::load(path).unwrap_or_else(|e| fail(e));
            eprintln!("running on {}", net.adapter);
            Network::Gpu(net)
        } else {
            Network::Ort(Net::load(path).unwrap_or_else(|e| fail(e.to_string())))
        }
    }

    pub fn render(&mut self, t: &tokens::Tokens, dream: &Dream) -> Vec<f32> {
        match self {
            Network::Ort(net) => net.render(t, dream).expect("inference"),
            Network::Gpu(net) => net.render(t, dream).expect("inference"),
        }
    }

    /// Whether the model takes the dream inputs.
    pub fn dreams(&self) -> bool {
        match self {
            Network::Ort(net) => net.dreams(),
            Network::Gpu(_) => true,
        }
    }
}

/// Simulation steps per second: the walker moves a fixed amount per step, as in datagen.
pub const TICK: f64 = 30.0;

/// A generated level and the walker in it.
pub struct Level {
    pub scene: Scene,
    pub walker: Walker,
    /// The scene's raw triangle features (datagen `tri_features`).
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

    /// The network's inputs for the walker's current view.
    pub fn tokens(&self) -> tokens::Tokens {
        let mut v = Vec::new();
        view_features(0, &self.walker.camera(), &mut v);
        tokens::tokens(&self.tris, &v)
    }
}

/// An RGB float pixel (roughly 0..1) as 0x00RRGGBB.
pub fn to_pixel(r: f32, g: f32, b: f32) -> u32 {
    let c = |x: f32| (x.clamp(0.0, 1.0) * 255.0) as u32;
    c(r) << 16 | c(g) << 8 | c(b)
}
