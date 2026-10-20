//! The trained renderer, exported by train/export_onnx.py, run with ONNX Runtime.

use crate::tokens::{Tokens, CAM_FEATURES, TRI_FEATURES};
use datagen::render::{H, W};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

pub struct Net {
    session: Session,
}

impl Net {
    pub fn load(path: &Path) -> ort::Result<Net> {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let env = ort::init().with_name("neurofield").with_telemetry(false).build()?;
        let session = Session::builder(&env)?.with_intra_threads(threads)?.commit_from_file(path)?;
        Ok(Net { session })
    }

    /// The network's frame for these tokens: [H * W * 3] floats, roughly 0..1, row-major RGB.
    pub fn render(&mut self, t: &Tokens) -> ort::Result<Vec<f32>> {
        // An empty view still needs one (padding) token, as in data.py `collate`.
        let n = t.len().max(1);
        let mut feats = t.feats.clone();
        feats.resize(n * TRI_FEATURES, 0.0);
        let mut tex = t.tex.clone();
        tex.resize(n, 0);
        let pad: Vec<bool> = (0..n).map(|i| i >= t.len()).collect();
        let outputs = self.session.run(ort::inputs![
            "feats" => Tensor::from_array(([1, n, TRI_FEATURES], feats))?,
            "tex" => Tensor::from_array(([1, n], tex))?,
            "pad" => Tensor::from_array(([1, n], pad))?,
            "cams" => Tensor::from_array(([1, CAM_FEATURES], t.cam.to_vec()))?,
        ])?;
        let (_, chw) = outputs["image"].try_extract_tensor::<f32>()?;
        // [3, H, W] planes -> interleaved RGB.
        let plane = (W * H) as usize;
        Ok((0..plane).flat_map(|i| [chw[i], chw[plane + i], chw[2 * plane + i]]).collect())
    }
}
