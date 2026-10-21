//! The trained renderer, exported by train/export_onnx.py, run with ONNX Runtime.

use crate::tokens::{Tokens, CAM_FEATURES, TRI_FEATURES};
use datagen::render::{H, W};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

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

pub struct Net {
    session: Session,
    /// Whether the model takes the dream inputs (exports from before them do not).
    dreams: bool,
}

impl Net {
    pub fn load(path: &Path) -> ort::Result<Net> {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let env = ort::init().with_name("neurofield").with_telemetry(false).build()?;
        let session = Session::builder(&env)?.with_intra_threads(threads)?.commit_from_file(path)?;
        let dreams = session.inputs().iter().any(|i| i.name() == "tex_mix");
        Ok(Net { session, dreams })
    }

    pub fn dreams(&self) -> bool {
        self.dreams
    }

    /// The network's frame for these tokens: [H * W * 3] floats, roughly 0..1, row-major RGB.
    pub fn render(&mut self, t: &Tokens, dream: &Dream) -> ort::Result<Vec<f32>> {
        // An empty view still needs one (padding) token, as in data.py `collate`.
        let n = t.len().max(1);
        let mut feats = t.feats.clone();
        feats.resize(n * TRI_FEATURES, 0.0);
        let mut tex = t.tex.clone();
        tex.resize(n, 0);
        let pad: Vec<bool> = (0..n).map(|i| i >= t.len()).collect();
        let mut inputs = ort::inputs![
            "feats" => Tensor::from_array(([1, n, TRI_FEATURES], feats))?,
            "tex" => Tensor::from_array(([1, n], tex))?,
            "pad" => Tensor::from_array(([1, n], pad))?,
            "cams" => Tensor::from_array(([1, CAM_FEATURES], t.cam.to_vec()))?,
        ];
        if self.dreams {
            inputs.extend(ort::inputs![
                "tex_mix" => Tensor::from_array(([NUM_TEX, NUM_TEX], dream.tex_mix.clone()))?,
                "temp" => Tensor::from_array(([1], vec![dream.temp]))?,
                "q_noise" => Tensor::from_array(([PATCHES, DREAM_DIRS], dream.q_noise.clone()))?,
            ]);
        }
        let outputs = self.session.run(inputs)?;
        let (_, chw) = outputs["image"].try_extract_tensor::<f32>()?;
        // [3, H, W] planes -> interleaved RGB.
        let plane = (W * H) as usize;
        Ok((0..plane).flat_map(|i| [chw[i], chw[plane + i], chw[2 * plane + i]]).collect())
    }
}
