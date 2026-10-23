//! The network's fixed sizes and the disturbances the dream effects make inside it.

use crate::world::view::{H, W};

/// Texture ids the network knows (data.py NUM_TEX).
pub const NUM_TEX: usize = 12;
/// Patch queries, one per 16x16 block of the frame.
pub const PATCHES: usize = (W as usize / 16) * (H as usize / 16);
/// Directions the dream effects can push each patch query along (model.py DREAM_DIRS).
pub const DREAM_DIRS: usize = 16;

/// Disturbances of the network's insides, for the dream effects (see Renderer.forward).
pub struct Dream {
    /// [NUM_TEX, NUM_TEX]: row i is the mix of texture embeddings that texture i gets.
    pub tex_mix: Vec<f32>,
    /// Cross-attention softmax temperature: above 1 each patch looks at more triangles.
    pub temp: f32,
    /// [PATCHES, DREAM_DIRS] pushes on the patch queries.
    pub q_noise: Vec<f32>,
}

impl Dream {
    /// No disturbance: the network as trained.
    pub fn calm() -> Dream {
        let mut tex_mix = vec![0.0; NUM_TEX * NUM_TEX];
        for i in 0..NUM_TEX {
            tex_mix[i * NUM_TEX + i] = 1.0;
        }
        Dream { tex_mix, temp: 1.0, q_noise: vec![0.0; PATCHES * DREAM_DIRS] }
    }
}
