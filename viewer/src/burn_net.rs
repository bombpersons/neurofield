//! The trained renderer in pure Rust with burn, on the CPU or on a GPU through
//! wgpu (Vulkan, Metal, DX12). A hand port of `Renderer.forward` in
//! train/model.py for one view at a time; keep the two in step. Weights come
//! from train/export_weights.py.

use crate::net::{Dream, DREAM_DIRS, NUM_TEX, PATCHES};
use crate::tokens::{Tokens, CAM_FEATURES, TRI_FEATURES};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, DeviceKind, Int, Tensor, TensorData};
use datagen::render::{H, W};
use std::collections::HashMap;
use std::path::Path;

const PATCH: usize = 16;
/// Triangle tokens are padded to a multiple of this.
const PAD_TO: usize = 64;

type Result<T> = std::result::Result<T, String>;

struct Linear {
    /// [in, out]: transposed from PyTorch's [out, in] once, at load.
    w: Tensor<2>,
    b: Tensor<2>,
}

impl Linear {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.matmul(self.w.clone()) + self.b.clone()
    }
}

struct LayerNorm {
    g: Tensor<2>,
    b: Tensor<2>,
}

impl LayerNorm {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mean = x.clone().mean_dim(1);
        let x = x - mean;
        let var = x.clone().powi_scalar(2).mean_dim(1);
        x / (var + 1e-5).sqrt() * self.g.clone() + self.b.clone()
    }
}

/// nn.MultiheadAttention's weights, split into query, key and value projections.
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
}

impl Attention {
    /// Attention from `x` [L, d] to `mem` [S, d], with `bias` added to the scores
    /// ([heads or 1, L or 1, S]; -inf masks) and the softmax at temperature `temp`.
    fn forward(&self, x: Tensor<2>, mem: Tensor<2>, bias: Option<Tensor<3>>, temp: f32) -> Tensor<2> {
        let [l, d] = x.dims();
        let s = mem.dims()[0];
        let dh = d / self.heads;
        let split = |t: Tensor<2>, n: usize| t.reshape([n, self.heads, dh]).swap_dims(0, 1); // [heads, n, dh]
        let q = split(self.q.forward(x), l);
        let k = split(self.k.forward(mem.clone()), s);
        let v = split(self.v.forward(mem), s);
        let mut scores = q.matmul(k.swap_dims(1, 2)) / ((dh as f32).sqrt() * temp);
        if let Some(bias) = bias {
            scores = scores + bias;
        }
        let out = softmax(scores, 2).matmul(v); // [heads, L, dh]
        self.out.forward(out.swap_dims(0, 1).reshape([l, d]))
    }
}

struct Block {
    n1: LayerNorm,
    attn: Attention,
    /// Cross-attention to the triangle tokens (decoder blocks only).
    cross: Option<(LayerNorm, Attention)>,
    n3: LayerNorm,
    mlp: (Linear, Linear),
}

impl Block {
    /// `pad` biases the self-attention; `mem` is the cross-attention's tokens and bias.
    fn forward(&self, x: Tensor<2>, pad: Option<Tensor<3>>, mem: Option<(&Tensor<2>, Tensor<3>, f32)>) -> Tensor<2> {
        let h = self.n1.forward(x.clone());
        let mut x = x + self.attn.forward(h.clone(), h, pad, 1.0);
        if let (Some((n2, xattn)), Some((mem, bias, temp))) = (&self.cross, mem) {
            x = x.clone() + xattn.forward(n2.forward(x), mem.clone(), Some(bias), temp);
        }
        let h = self.mlp.1.forward(gelu(self.mlp.0.forward(self.n3.forward(x.clone()))));
        x + h
    }
}

/// The exported tensors, taken out by name as the model is built.
struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    device: Device,
}

impl Weights {
    fn take(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        self.tensors.remove(name).ok_or(format!("the weights have no {name}"))
    }

    fn t1(&mut self, name: &str) -> Result<Tensor<1>> {
        let (_, v) = self.take(name)?;
        let n = v.len();
        Ok(Tensor::from_data(TensorData::new(v, [n]), &self.device))
    }

    fn t2(&mut self, name: &str) -> Result<Tensor<2>> {
        let (s, v) = self.take(name)?;
        Ok(Tensor::from_data(TensorData::new(v, [s[0], s[1]]), &self.device))
    }

    fn t4(&mut self, name: &str) -> Result<Tensor<4>> {
        let (s, v) = self.take(name)?;
        Ok(Tensor::from_data(TensorData::new(v, [s[0], s[1], s[2], s[3]]), &self.device))
    }

    /// A [n] vector as a [1, n] row, to broadcast over tokens.
    fn row(&mut self, name: &str) -> Result<Tensor<2>> {
        Ok(self.t1(name)?.unsqueeze_dim(0))
    }

    fn linear(&mut self, p: &str) -> Result<Linear> {
        Ok(Linear { w: self.t2(&format!("{p}.weight"))?.swap_dims(0, 1), b: self.row(&format!("{p}.bias"))? })
    }

    fn mlp(&mut self, a: &str, b: &str) -> Result<(Linear, Linear)> {
        Ok((self.linear(a)?, self.linear(b)?))
    }

    fn norm(&mut self, p: &str) -> Result<LayerNorm> {
        Ok(LayerNorm { g: self.row(&format!("{p}.weight"))?, b: self.row(&format!("{p}.bias"))? })
    }

    fn attention(&mut self, p: &str, heads: usize) -> Result<Attention> {
        let w = self.t2(&format!("{p}.in_proj_weight"))?; // [3d, d]
        let b = self.t1(&format!("{p}.in_proj_bias"))?;
        let d = w.dims()[1];
        let part = |i: usize| Linear {
            w: w.clone().narrow(0, i * d, d).swap_dims(0, 1),
            b: b.clone().narrow(0, i * d, d).unsqueeze_dim(0),
        };
        Ok(Attention { q: part(0), k: part(1), v: part(2), out: self.linear(&format!("{p}.out_proj"))?, heads })
    }

    fn block(&mut self, p: &str, heads: usize, cross: bool) -> Result<Block> {
        Ok(Block {
            n1: self.norm(&format!("{p}.n1"))?,
            attn: self.attention(&format!("{p}.attn"), heads)?,
            cross: if cross {
                Some((self.norm(&format!("{p}.n2"))?, self.attention(&format!("{p}.xattn"), heads)?))
            } else {
                None
            },
            n3: self.norm(&format!("{p}.n3"))?,
            mlp: self.mlp(&format!("{p}.mlp.0"), &format!("{p}.mlp.2"))?,
        })
    }
}

/// The safetensors file's tensors as (shape, values), and its metadata.
fn read_safetensors(path: &Path) -> Result<(HashMap<String, (Vec<usize>, Vec<f32>)>, HashMap<String, String>)> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| e.to_string())?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
    let tensors = st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let values = view.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            (name, (view.shape().to_vec(), values))
        })
        .collect();
    Ok((tensors, meta.metadata().clone().unwrap_or_default()))
}

/// The number after `"key":` in a flat JSON object.
fn json_number(json: &str, key: &str) -> Result<usize> {
    let pat = format!("\"{key}\":");
    let at = json.find(&pat).ok_or(format!("cfg has no {key}"))?;
    let rest = json[at + pat.len()..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().map_err(|e| format!("cfg {key}: {e}"))
}

/// Sin/cos of `x` [n, c] at `bands` octaves: [n, c * 2 * bands], as model.py `fourier`.
fn fourier(x: Tensor<2>, bands: usize) -> Tensor<2> {
    let [n, c] = x.dims();
    let freqs: Vec<f32> = (0..bands).map(|k| 2f32.powi(k as i32) * std::f32::consts::PI).collect();
    let freqs = Tensor::<1>::from_data(TensorData::new(freqs, [bands]), &x.device()).reshape([1, 1, bands]);
    let a = x.reshape([n, c, 1]) * freqs;
    Tensor::cat(vec![a.clone().sin(), a.cos()], 2).reshape([n, c * 2 * bands])
}

pub struct BurnNet {
    device: Device,
    bands: usize,
    tri_in: (Linear, Linear),
    tex: Tensor<2>,
    cam_in: (Linear, Linear),
    encoder: Vec<Block>,
    decoder: Vec<Block>,
    /// The patch queries before the camera and dream noise: q_in(fourier(centres)) + q_pos.
    queries: Tensor<2>,
    centres: Tensor<2>,
    /// Per decoder layer, [heads, 1, 1]: how strongly each head prefers covering triangles.
    cover: Vec<Tensor<3>>,
    noise_dirs: Tensor<2>,
    out_norm: LayerNorm,
    head: Linear,
    refine: Vec<(Tensor<4>, Tensor<1>)>,
}

impl BurnNet {
    /// Load weights written by train/export_weights.py onto the "cpu" or the "gpu".
    pub fn load(path: &Path, device: &str) -> Result<BurnNet> {
        let device = match device {
            "cpu" => Device::flex(),
            "gpu" => Device::wgpu(DeviceKind::DefaultDevice),
            d => return Err(format!("unknown device {d:?} (cpu or gpu)")),
        };
        let (tensors, meta) = read_safetensors(path)?;
        let cfg = meta.get("cfg").ok_or("no cfg in the weights' metadata")?;
        let (heads, bands) = (json_number(cfg, "heads")?, json_number(cfg, "bands")?);
        let (enc_layers, dec_layers) = (json_number(cfg, "enc_layers")?, json_number(cfg, "dec_layers")?);
        let mut w = Weights { tensors, device: device.clone() };

        let tri_in = w.mlp("tri_in.0", "tri_in.2")?;
        let tex = w.t2("tex.weight")?;
        let cam_in = w.mlp("cam_in.0", "cam_in.2")?;
        let encoder = (0..enc_layers).map(|i| w.block(&format!("encoder.{i}"), heads, false)).collect::<Result<_>>()?;
        let decoder = (0..dec_layers).map(|i| w.block(&format!("decoder.{i}"), heads, true)).collect::<Result<_>>()?;
        let centres = w.t2("centres")?;
        let q_in = w.mlp("q_in.0", "q_in.2")?;
        let queries = q_in.1.forward(gelu(q_in.0.forward(fourier(centres.clone(), bands)))) + w.t2("q_pos")?;
        let cover = w.t2("cover")?;
        let cover = (0..dec_layers).map(|l| cover.clone().narrow(0, l, 1).reshape([heads, 1, 1])).collect();
        let noise_dirs = w.t2("noise_dirs")?;
        let out_norm = w.norm("out_norm")?;
        let head = w.linear("head")?;
        let refine = [0, 2, 4]
            .iter()
            .map(|i| Ok((w.t4(&format!("refine.{i}.weight"))?, w.t1(&format!("refine.{i}.bias"))?)))
            .collect::<Result<_>>()?;
        if !w.tensors.is_empty() {
            let mut left: Vec<_> = w.tensors.keys().collect();
            left.sort();
            return Err(format!("unused weights (is the port behind model.py?): {left:?}"));
        }
        Ok(BurnNet {
            device,
            bands,
            tri_in,
            tex,
            cam_in,
            encoder,
            decoder,
            queries,
            centres,
            cover,
            noise_dirs,
            out_norm,
            head,
            refine,
        })
    }

    /// Distance in patch widths from each patch centre to each triangle's screen bounding
    /// box (0 inside it): [patches, tris], as model.py `cover_distance`.
    fn cover_distance(&self, ndc: Tensor<2>) -> Tensor<2> {
        let n = ndc.dims()[0];
        let xy = ndc.reshape([n, 3, 2]);
        let lo = xy.clone().min_dim(1).reshape([1, n, 2]);
        let hi = xy.max_dim(1).reshape([1, n, 2]);
        let c = self.centres.clone().reshape([PATCHES, 1, 2]);
        let gap = (lo - c.clone()).clamp_min(0.0) + (c - hi).clamp_min(0.0);
        gap.powi_scalar(2).sum_dim(2).sqrt().reshape([PATCHES, n]) / (2.0 * PATCH as f32 / W as f32)
    }

    /// The network's frame for these tokens: [H * W * 3] floats, roughly 0..1, row-major RGB.
    pub fn render(&mut self, t: &Tokens, dream: &Dream) -> Result<Vec<f32>> {
        let dev = &self.device;
        // Pad to a multiple of PAD_TO masked tokens (an empty view still needs some, as in data.py
        // `collate`): the GPU backend compiles and tunes kernels per shape, so few shapes keep it fast.
        let n = t.len().max(1).div_ceil(PAD_TO) * PAD_TO;
        let mut feats = t.feats.clone();
        feats.resize(n * TRI_FEATURES, 0.0);
        let mut tex = t.tex.clone();
        tex.resize(n, 0);
        let feats = Tensor::<2>::from_data(TensorData::new(feats, [n, TRI_FEATURES]), dev);
        let tex = Tensor::<1, Int>::from_data(TensorData::new(tex, [n]), dev);
        let cams = Tensor::<2>::from_data(TensorData::new(t.cam.to_vec(), [1, CAM_FEATURES]), dev);
        let tex_mix = Tensor::<2>::from_data(TensorData::new(dream.tex_mix.clone(), [NUM_TEX, NUM_TEX]), dev);
        let q_noise = Tensor::<2>::from_data(TensorData::new(dream.q_noise.clone(), [PATCHES, DREAM_DIRS]), dev);
        // Additive key mask over [camera, triangles]: -inf for the padding token.
        let pad: Vec<f32> = (0..=n).map(|i| if i > t.len() { f32::NEG_INFINITY } else { 0.0 }).collect();
        let pad = Tensor::<1>::from_data(TensorData::new(pad, [n + 1]), dev).reshape([1, 1, n + 1]);

        let ndc = feats.clone().narrow(1, 9, 6);
        let table = tex_mix.matmul(self.tex.clone());
        let x = Tensor::cat(vec![feats, fourier(ndc.clone(), self.bands)], 1);
        let x = self.tri_in.1.forward(gelu(self.tri_in.0.forward(x))) + table.select(0, tex);
        let cam = self.cam_in.1.forward(gelu(self.cam_in.0.forward(cams))); // [1, d]
        let mut x = Tensor::cat(vec![cam.clone(), x], 0); // the camera rides along as token 0
        for blk in &self.encoder {
            x = blk.forward(x, Some(pad.clone()), None);
        }

        let mut q = self.queries.clone() + cam + q_noise.matmul(self.noise_dirs.clone());
        let zero = Tensor::<2>::zeros([PATCHES, 1], dev);
        let dist = Tensor::cat(vec![zero, self.cover_distance(ndc)], 1).reshape([1, PATCHES, n + 1]);
        for (blk, strength) in self.decoder.iter().zip(&self.cover) {
            let bias = -(strength.clone() * dist.clone()) + pad.clone(); // [heads, patches, 1 + n]
            q = blk.forward(q, None, Some((&x, bias, dream.temp)));
        }
        let patches = self.head.forward(self.out_norm.forward(q)); // [patches, 16 * 16 * 3]
        let (gw, gh) = (W as usize / PATCH, H as usize / PATCH);
        let img =
            patches.reshape([gh, gw, 3, PATCH, PATCH]).permute([2, 0, 3, 1, 4]).reshape([1, 3, H as usize, W as usize]);
        let mut r = img.clone();
        for (i, (w, b)) in self.refine.iter().enumerate() {
            r = conv2d(r, w.clone(), Some(b.clone()), ConvOptions::new([1, 1], [1, 1], [1, 1], 1));
            if i < self.refine.len() - 1 {
                r = gelu(r);
            }
        }
        let chw: Vec<f32> = (img + r).into_data().try_into_vec::<f32>().map_err(|e| format!("{e:?}"))?;
        // [3, H, W] planes -> interleaved RGB.
        let plane = (W * H) as usize;
        Ok((0..plane).flat_map(|i| [chw[i], chw[plane + i], chw[2 * plane + i]]).collect())
    }
}
