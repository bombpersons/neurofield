//! The network's fixed sizes and the disturbances the dream effects make inside it.

use crate::world::view::H;

/// Texture ids the network knows (data.py NUM_TEX).
pub const NUM_TEX: usize = 12;
/// Patch queries, one per 16x16 block of a frame `width` wide.
pub const fn patches(width: u32) -> usize {
    (width as usize / 16) * (H as usize / 16)
}
/// Directions the dream effects can push each patch query along (model.py DREAM_DIRS).
pub const DREAM_DIRS: usize = 16;

/// Disturbances of the network's insides, for the dream effects (see Renderer.forward).
pub struct Dream {
    /// [NUM_TEX, NUM_TEX]: row i is the mix of texture embeddings that texture i gets.
    pub tex_mix: Vec<f32>,
    /// Cross-attention softmax temperature: above 1 each patch looks at more triangles.
    pub temp: f32,
    /// [patches(width), DREAM_DIRS] pushes on the patch queries.
    pub q_noise: Vec<f32>,
    /// The per-pixel stage's latents read `warp` of a texture off true along slow waves, and
    /// the head's channels read `smear` pixels off, at `time` seconds (the glitch panel).
    pub warp: f32,
    pub smear: f32,
    pub time: f32,
    /// How much of the last frame's patch queries, and of the head's last channels, each
    /// frame keeps (0: none).
    pub echo: f32,
    pub pixel_echo: f32,
    pub gbuffer: GBuffer,
}

/// The glitch panel's lies to the per-pixel stage about each pixel.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct GBuffer {
    /// Pixels: which triangle a pixel shows, looked up this far off along slow waves.
    pub warp: f32,
    /// Each face's normal turned by up to about this (radians, roughly).
    pub twist: f32,
    /// Depth scaled by up to e^depth either way, in slow waves across the screen.
    pub depth: f32,
    /// Each face's texture coordinates turned and scaled by a share of this, 0..1.
    pub scramble: f32,
    /// Texture coordinates snapped to this many cells a texture (0: not).
    pub cells: f32,
    /// Flesh told everywhere, 0..1.
    pub flesh: f32,
    /// Organs split into two copies this far apart, and squashed.
    pub ghost: f32,
}

impl GBuffer {
    /// As the shader reads them (gpu_net.wgsl `stage_inputs`), with the time last.
    pub fn floats(&self, time: f32) -> [f32; 8] {
        [self.warp, self.twist, self.depth, self.scramble, self.cells, self.flesh, self.ghost, time]
    }
}

impl Dream {
    /// No disturbance: the network as trained, on a frame `width` wide.
    pub fn calm(width: u32) -> Dream {
        let mut tex_mix = vec![0.0; NUM_TEX * NUM_TEX];
        for i in 0..NUM_TEX {
            tex_mix[i * NUM_TEX + i] = 1.0;
        }
        Dream {
            tex_mix,
            temp: 1.0,
            q_noise: vec![0.0; patches(width) * DREAM_DIRS],
            warp: 0.0,
            smear: 0.0,
            time: 0.0,
            echo: 0.0,
            pixel_echo: 0.0,
            gbuffer: GBuffer::default(),
        }
    }
}
