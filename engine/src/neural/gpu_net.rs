//! The trained renderer as hand-written compute shaders (gpu_net.wgsl) run through wgpu:
//! Vulkan, Metal or DX12, in pure Rust. A port of `Renderer.forward` in train/model.py for
//! one view at a time; keep the two in step. Weights come from train/export_weights.py.
//!
//! Weights, inputs and every intermediate result share one storage buffer, and each
//! dispatch reads its sizes and offsets from its own slot of a uniform buffer, so a frame
//! is one submission: upload the tokens, run ~150 small kernels, read back the image. The
//! kernels take any sizes, so nothing is compiled per triangle count.

use crate::neural::dream::{Dream, DREAM_DIRS, NUM_TEX};
use crate::neural::tokens::{Tokens, CAM_FEATURES, MAX_TRIS, ORGAN_FEATURES, TRI_FEATURES, TRI_FEATURES_PLAIN};
use crate::neural::gbuffer;
use crate::world::dreamed::{MAX_ORGANS, ORGAN_FEATURES as RAW_ORGAN};
use crate::world::view::{H, W};
use std::collections::HashMap;
use std::path::Path;

type Result<T> = std::result::Result<T, String>;

const PATCH: usize = 16;
/// Fourier bands per coordinate, the lowest at pi / 2^LOW (model.py `fourier`).
const BANDS: usize = 8;
const LOW: i32 = 2;
/// Columns of the triangle MLP's input: the features and 16 Fourier features per NDC coordinate.
const TRI_IN: usize = TRI_FEATURES + 6 * 2 * BANDS;
/// The organ MLP's: the features and 16 Fourier features per corner of the screen box.
const ORGAN_IN: usize = ORGAN_FEATURES + 4 * 2 * BANDS;
/// Where the NDC corners, and the organs' screen box, start among their features.
const TRI_NDC: u32 = 9;
const ORGAN_BOX: u32 = 12;
/// The widest frame (training data's widest, 2.4:1), and its patch grid.
pub const MAX_W: u32 = 576;
const GH: usize = H as usize / PATCH;
const MAX_PATCHES: usize = (MAX_W as usize / PATCH) * GH;
const NONE: u32 = u32::MAX;
const GELU: u32 = 1;
const ACCUMULATE: u32 = 2;
const SHUFFLE: u32 = 4;
const INTERLEAVE: u32 = 8;
/// Bytes per uniform slot (wgpu's minimum dynamic-offset alignment).
const SLOT: u64 = 256;
const MAX_OPS: u64 = 1024;

/// One dispatch's sizes and offsets (in floats into `mem`); see gpu_net.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Params {
    pub(super) m: u32,
    pub(super) k: u32,
    pub(super) n: u32,
    pub(super) batch: u32,
    pub(super) a_off: u32,
    pub(super) a_bs: u32,
    pub(super) a_rs: u32,
    pub(super) a_cs: u32,
    pub(super) b_off: u32,
    pub(super) b_bs: u32,
    pub(super) b_rs: u32,
    pub(super) b_cs: u32,
    pub(super) c_off: u32,
    pub(super) c_bs: u32,
    pub(super) c_rs: u32,
    pub(super) bias_off: u32,
    pub(super) flags: u32,
    pub(super) x0: u32,
    pub(super) x1: u32,
    pub(super) x2: u32,
    pub(super) f0: f32,
    pub(super) f1: f32,
    pub(super) f2: f32,
    pub(super) f3: f32,
}

#[derive(Clone, Copy)]
enum Kernel {
    Matmul,
    LayerNorm,
    Softmax,
    Prepare,
    AddTexture,
    CoverDistance,
    Transpose,
    Conv3x3,
    Add,
    Upsample2,
    StageInputs,
    Lerp,
}

const KERNELS: [(Kernel, &str); 12] = [
    (Kernel::Matmul, "matmul"),
    (Kernel::LayerNorm, "layer_norm"),
    (Kernel::Softmax, "softmax"),
    (Kernel::Prepare, "prepare"),
    (Kernel::AddTexture, "add_texture"),
    (Kernel::CoverDistance, "cover_distance"),
    (Kernel::Transpose, "transpose"),
    (Kernel::Conv3x3, "conv3x3"),
    (Kernel::Add, "add"),
    (Kernel::Upsample2, "resize2"),
    (Kernel::StageInputs, "stage_inputs"),
    (Kernel::Lerp, "lerp_into"),
];
/// Room for one view's G-buffer pieces, the triangles they are of, and their places in tiles.
const MAX_PIECES: usize = 1 << 18;
const MAX_ROWS: usize = 1 << 15;
const MAX_PAIRS: usize = 1 << 20;
/// The per-pixel stage's input channels: the head's last, plain, and the latents (model.py Deferred).
const STAGE_IN: usize = 16 + 22 + 8 + 8 + 8 + 4;

/// A sculpted level's view, for the per-pixel stage: its raw triangles and their turning
/// (one row each, as `tokens::Turned`), the organs in sight (`dreamed::organs` rows), the
/// view (`views.npy` row) and how far the dial has turned.
/// The glitch panel's changes to the weights: all of them `rot` of the way to noise of
/// the same mean and spread, then a share of the hidden units of the per-pixel stage's MLP
/// (`stage_units`) and of the transformer's MLPs (`net_units`) silenced, or turned
/// over with `invert`, and every other unit scaled by up to 1 +- `jitter`; which units,
/// by `seed`.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct Damage {
    pub rot: f32,
    pub stage_units: f32,
    pub net_units: f32,
    pub jitter: f32,
    pub invert: bool,
    pub seed: u32,
}

/// A hash of (a, b, c) to 0..1.
fn hash01(a: u64, b: u64, c: u64) -> f32 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ c.wrapping_mul(0x1656_67B1_9E37_79F9);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^= x >> 33;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// Noise with each tensor's (`spans`) mean and spread, for `Damage::rot`.
fn noise_like(w: &[f32], spans: &[(usize, usize)]) -> Vec<f32> {
    let mut out = vec![0.0; w.len()];
    for (k, &(at, len)) in spans.iter().enumerate() {
        let v = &w[at..at + len];
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / len.max(1) as f64;
        let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / len.max(1) as f64;
        let sd = var.sqrt() as f32;
        for (i, o) in out[at..at + len].iter_mut().enumerate() {
            // Roughly normal: the sum of four uniforms.
            let u: f32 = (0..4).map(|j| hash01(k as u64, i as u64, j)).sum::<f32>() - 2.0;
            *o = mean as f32 + sd * u * 1.73;
        }
    }
    out
}

pub struct StageView<'a> {
    pub tris: &'a [f32],
    pub turning: &'a [f32],
    pub organs: &'a [f32],
    pub view: &'a [f32],
    pub amount: f32,
}

/// The per-pixel stage's weights in `mem`: its latent textures and its MLP.
struct Stage {
    level: u32,
    flesh: u32,
    eye: u32,
    iris: u32,
    mlp: [Linear; 3],
}

/// A linear layer in `mem`: weight [inp, out] (row stride `rs`, which is wider than `out`
/// when it is the first columns of a bigger matrix) and bias.
#[derive(Clone, Copy)]
struct Linear {
    w: u32,
    b: u32,
    inp: usize,
    out: usize,
    rs: usize,
}

#[derive(Clone, Copy)]
struct Norm {
    g: u32,
    b: u32,
}

struct Block {
    n1: Norm,
    /// [d, 3d]: query, key and value.
    qkv: Linear,
    out: Linear,
    /// Cross-attention (decoder blocks): its norm, query projection and output projection.
    /// Its keys and values come from `GpuNet::cross_kv`, for all layers at once.
    cross: Option<(Norm, Linear, Linear)>,
    n3: Norm,
    mlp: (Linear, Linear),
}

/// Offsets of the inputs and intermediate results in `mem`, sized for the largest view.
struct Scratch {
    feats: u32,
    tex: u32,
    organs: u32,
    organ_kinds: u32,
    organ_in: u32,
    cams: u32,
    mix: u32,
    noise: u32,
    tri_in: u32,
    hidden: u32,
    cam_hidden: u32,
    /// The camera token before the encoder, which the patch queries get too.
    cam: u32,
    x: u32,
    xn: u32,
    qkv: u32,
    scores: u32,
    attn: u32,
    table: u32,
    q: u32,
    qn: u32,
    qc: u32,
    kv: u32,
    dist: u32,
    /// [patches, d] base queries and [patches, 2] centres for the current grid width.
    queries: u32,
    centres: u32,
    /// The head's projection [patches, channels], then its grids, ping and pong.
    proj: u32,
    ga: u32,
    gb: u32,
    out: u32,
    /// The per-pixel stage's G-buffer pieces, rows, tiles, tile lists, raw organs and view,
    /// its inputs [STAGE_IN][pixels] and its hidden layers [pixels][64] (none without a stage).
    pieces: u32,
    rows: u32,
    tiles: u32,
    lists: u32,
    raw_organs: u32,
    view: u32,
    stage_in: u32,
    stage_hidden: u32,
    /// Last frame's patch queries and head channels, for the glitch panel's echoes.
    echo: u32,
    pixel_echo: u32,
    /// The glitch panel's G-buffer lies (`GBuffer::floats`).
    gbuffer: u32,
    end: u32,
}

/// One doubling of the upsampling head: 3x3 conv weights [cout][cin][3][3] (cout four times
/// the next grid's channels when pixel-shuffled), bias, sizes.
#[derive(Clone, Copy)]
struct Up {
    w: u32,
    b: u32,
    cin: usize,
    cout: usize,
}

/// Host copy of the weights region of `mem`, built at load.
struct Upload {
    data: Vec<f32>,
    /// Where each tensor is: offset and length.
    spans: Vec<(usize, usize)>,
}

impl Upload {
    /// Append `v` (16-float aligned) and return its offset.
    fn push(&mut self, v: &[f32]) -> u32 {
        self.data.resize(self.data.len().div_ceil(16) * 16, 0.0);
        let at = self.data.len() as u32;
        self.data.extend_from_slice(v);
        self.spans.push((at as usize, v.len()));
        at
    }
}

/// The exported tensors, taken out by name as the model is built.
struct Weights(HashMap<String, (Vec<usize>, Vec<f32>)>);

impl Weights {
    fn take(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        self.0.remove(name).ok_or(format!("the weights have no {name}"))
    }

    /// A PyTorch [out, inp] weight as row-major [inp, out].
    fn transposed(&mut self, name: &str) -> Result<(usize, usize, Vec<f32>)> {
        let (s, v) = self.take(name)?;
        let (out, inp) = (s[0], s[1]);
        let mut t = vec![0.0; v.len()];
        for o in 0..out {
            for i in 0..inp {
                t[i * out + o] = v[o * inp + i];
            }
        }
        Ok((inp, out, t))
    }

    fn linear(&mut self, up: &mut Upload, p: &str) -> Result<Linear> {
        let (inp, out, w) = self.transposed(&format!("{p}.weight"))?;
        let (_, b) = self.take(&format!("{p}.bias"))?;
        Ok(Linear { w: up.push(&w), b: up.push(&b), inp, out, rs: out })
    }

    fn norm(&mut self, up: &mut Upload, p: &str) -> Result<Norm> {
        Ok(Norm { g: up.push(&self.take(&format!("{p}.weight"))?.1), b: up.push(&self.take(&format!("{p}.bias"))?.1) })
    }

    /// nn.MultiheadAttention's in_proj as one [d, 3d] linear, and its out_proj.
    fn attention(&mut self, up: &mut Upload, p: &str) -> Result<(Linear, Linear, Vec<f32>, Vec<f32>)> {
        let (inp, out, w) = self.transposed(&format!("{p}.in_proj_weight"))?;
        let (_, b) = self.take(&format!("{p}.in_proj_bias"))?;
        let qkv = Linear { w: up.push(&w), b: up.push(&b), inp, out, rs: out };
        Ok((qkv, self.linear(up, &format!("{p}.out_proj"))?, w, b))
    }
}

/// The safetensors file's tensors as (shape, values), and its metadata.
pub(super) fn read_safetensors(path: &Path) -> Result<(HashMap<String, (Vec<usize>, Vec<f32>)>, HashMap<String, String>)> {
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

/// The exact GELU, in f64, for the one-off patch query precompute.
fn gelu_cpu(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf64(x / std::f64::consts::SQRT_2))) as f32
}

/// erf in f64: its Taylor series near zero, a continued fraction for erfc further out.
fn erf64(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 2.5 {
        // Taylor series.
        let (mut term, mut sum, x2) = (a, a, a * a);
        for n in 1..60 {
            term *= -x2 / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // Continued fraction for erfc.
        let mut f = 0.0;
        for n in (1..60).rev() {
            f = n as f64 / 2.0 / (a + f);
        }
        1.0 - (-a * a).exp() / std::f64::consts::PI.sqrt() / (a + f)
    };
    r.copysign(x)
}

/// y = x W + b on the CPU, for the one-off patch query precompute: x [n, inp], W [inp, out].
fn cpu_linear(x: &[f32], w: &[f32], b: &[f32], inp: usize, out: usize) -> Vec<f32> {
    let n = x.len() / inp;
    let mut y = vec![0.0f32; n * out];
    for r in 0..n {
        for o in 0..out {
            let mut s = b[o] as f64;
            for i in 0..inp {
                s += x[r * inp + i] as f64 * w[i * out + o] as f64;
            }
            y[r * out + o] = s as f32;
        }
    }
    y
}

pub struct GpuNet {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    mem: wgpu::Buffer,
    params: wgpu::Buffer,
    bind: wgpu::BindGroup,
    readback: wgpu::Buffer,
    s: Scratch,
    d: usize,
    heads: usize,
    tri_in: (Linear, Linear),
    /// How many of a triangle token's features the model takes (TRI_FEATURES_PLAIN or all).
    tri_features: usize,
    tex: u32,
    /// Organ tokens' MLP, kind table and [dec_layers, heads] coverage strengths, if the model has them.
    organs: Option<((Linear, Linear), u32, u32)>,
    cam_in: (Linear, Linear),
    encoder: Vec<Block>,
    decoder: Vec<Block>,
    /// [d, layers * 2d]: every decoder layer's cross-attention keys and values, side by side.
    cross_kv: Linear,
    /// q_in's two layers on the CPU ([inp, out] weights, bias), for each grid width's
    /// patch queries, and the width they were last made for.
    q_in: [(usize, usize, Vec<f32>, Vec<f32>); 2],
    grid_w: usize,
    /// [dec_layers, heads] coverage strengths.
    cover: u32,
    noise_dirs: u32,
    out_norm: Norm,
    /// The upsampling head: projection, doublings, and the conv to RGB.
    proj: Linear,
    ups: Vec<Up>,
    /// Whether each doubling is bilinear then a convolution, not a convolution then pixel shuffle.
    resize: bool,
    rgb: Up,
    stage: Option<Stage>,
    /// The weights as loaded, where they are in `mem` (from 0), each tensor's span, and the
    /// noise `Damage::rot` goes toward (made when first wanted).
    weights: Vec<f32>,
    spans: Vec<(usize, usize)>,
    rot_target: Option<Vec<f32>>,
    damage: Damage,
    /// The layers whose hidden units `Damage` reaches: each, and whether it is the stage's.
    units: Vec<(Linear, bool)>,
    /// The frame width the echoes' last frames (`s.echo`, `s.pixel_echo`) were kept at, if any.
    echo_width: [Option<u32>; 2],
    pub adapter: String,
}

/// Patch centres [gh * gw, 2] in NDC for a grid `gw` patches wide (model.py `centres`).
fn centres(gw: usize) -> Vec<f32> {
    let mut c = Vec::with_capacity(gw * GH * 2);
    for y in 0..GH {
        for x in 0..gw {
            c.push(((x as f32 + 0.5) * PATCH as f32 - (gw * PATCH) as f32 / 2.0) / (W as f32 / 2.0));
            c.push(((y as f32 + 0.5) * PATCH as f32 - H as f32 / 2.0) / (W as f32 / 2.0));
        }
    }
    c
}

impl GpuNet {
    pub fn load(path: &Path) -> Result<GpuNet> {
        let (tensors, meta) = read_safetensors(path)?;
        let cfg = meta.get("cfg").ok_or("no cfg in the weights' metadata")?;
        let (heads, bands) = (json_number(cfg, "heads")?, json_number(cfg, "bands")?);
        let (enc_layers, dec_layers) = (json_number(cfg, "enc_layers")?, json_number(cfg, "dec_layers")?);
        let low = json_number(cfg, "low").unwrap_or(0);
        if bands != BANDS || low != LOW as usize {
            return Err(format!(
                "gpu_net.wgsl assumes {BANDS} Fourier bands from pi / {}, the model has {bands} from pi / {} \
                 (re-export it with the current train/export_weights.py)",
                1 << LOW,
                1 << low
            ));
        }
        let mut w = Weights(tensors);
        let mut up = Upload { data: Vec::new(), spans: Vec::new() };

        let tri_in = (w.linear(&mut up, "tri_in.0")?, w.linear(&mut up, "tri_in.2")?);
        let d = tri_in.1.out;
        let tri_features = tri_in.0.inp - 6 * 2 * BANDS;
        if tri_features != TRI_FEATURES && tri_features != TRI_FEATURES_PLAIN {
            return Err(format!("the model takes {tri_features} features a triangle, not {TRI_FEATURES} or {TRI_FEATURES_PLAIN}"));
        }
        let tex = up.push(&w.take("tex.weight")?.1);
        let organs = if w.0.contains_key("organ_in.0.weight") {
            let mlp = (w.linear(&mut up, "organ_in.0")?, w.linear(&mut up, "organ_in.2")?);
            if mlp.0.inp != ORGAN_IN {
                return Err(format!("the model's organ tokens take {} inputs, not {ORGAN_IN}", mlp.0.inp));
            }
            let kinds = up.push(&w.take("organ_kind.weight")?.1);
            let cover = up.push(&w.take("organ_cover")?.1);
            Some((mlp, kinds, cover))
        } else {
            None
        };
        let cam_in = (w.linear(&mut up, "cam_in.0")?, w.linear(&mut up, "cam_in.2")?);
        let block =
            |w: &mut Weights, up: &mut Upload, p: &str, cross: bool| -> Result<(Block, Option<(Vec<f32>, Vec<f32>)>)> {
                let n1 = w.norm(up, &format!("{p}.n1"))?;
                let (qkv, out, _, _) = w.attention(up, &format!("{p}.attn"))?;
                let (cross, kv) = if cross {
                    let n2 = w.norm(up, &format!("{p}.n2"))?;
                    let (xqkv, xout, xw, xb) = w.attention(up, &format!("{p}.xattn"))?;
                    (Some((n2, Linear { out: d, ..xqkv }, xout)), Some((xw, xb)))
                } else {
                    (None, None)
                };
                let n3 = w.norm(up, &format!("{p}.n3"))?;
                let mlp = (w.linear(up, &format!("{p}.mlp.0"))?, w.linear(up, &format!("{p}.mlp.2"))?);
                Ok((Block { n1, qkv, out, cross, n3, mlp }, kv))
            };
        let encoder = (0..enc_layers)
            .map(|i| Ok(block(&mut w, &mut up, &format!("encoder.{i}"), false)?.0))
            .collect::<Result<Vec<_>>>()?;
        let mut decoder = Vec::new();
        let (mut kv_w, mut kv_b) = (vec![0.0; d * dec_layers * 2 * d], Vec::new());
        for l in 0..dec_layers {
            let (blk, kv) = block(&mut w, &mut up, &format!("decoder.{l}"), true)?;
            let (xw, xb) = kv.expect("decoder blocks cross-attend");
            // Columns d..3d of this layer's [d, 3d] in_proj go to columns l*2d.. of cross_kv.
            for i in 0..d {
                let row = &xw[i * 3 * d + d..i * 3 * d + 3 * d];
                kv_w[i * dec_layers * 2 * d + l * 2 * d..][..2 * d].copy_from_slice(row);
            }
            kv_b.extend_from_slice(&xb[d..3 * d]);
            decoder.push(blk);
        }
        let cross_kv =
            Linear { w: up.push(&kv_w), b: up.push(&kv_b), inp: d, out: dec_layers * 2 * d, rs: dec_layers * 2 * d };

        // Patch queries come from their centres alone, so they are made on the CPU for
        // each grid width (`set_grid`).
        let q_in = [w.transposed("q_in.0.weight")?, w.transposed("q_in.2.weight")?];
        let q_in = [
            (q_in[0].0, q_in[0].1, q_in[0].2.clone(), w.take("q_in.0.bias")?.1),
            (q_in[1].0, q_in[1].1, q_in[1].2.clone(), w.take("q_in.2.bias")?.1),
        ];
        let cover = up.push(&w.take("cover")?.1);
        let noise_dirs = up.push(&w.take("noise_dirs")?.1);
        let out_norm = w.norm(&mut up, "out_norm")?;
        let proj = w.linear(&mut up, "head.proj")?;
        let conv = |w: &mut Weights, up: &mut Upload, name: &str| -> Result<Up> {
            let (shape, wt) = w.take(&format!("{name}.weight"))?;
            Ok(Up { w: up.push(&wt), b: up.push(&w.take(&format!("{name}.bias"))?.1), cin: shape[1], cout: shape[0] })
        };
        let mut ups = Vec::new();
        while w.0.contains_key(&format!("head.ups.{}.weight", ups.len())) {
            let name = format!("head.ups.{}", ups.len());
            ups.push(conv(&mut w, &mut up, &name)?);
            // Noise injection comes with the creatures; until then its strengths must be zero.
            let (_, scale) = w.take(&format!("head.noise_scale.{}", ups.len() - 1))?;
            if scale.iter().any(|&v| v != 0.0) {
                return Err("this model adds noise in its head, which gpu_net doesn't do yet".into());
            }
        }
        let rgb = conv(&mut w, &mut up, "head.rgb")?;
        // A resizing head's last doubling makes the channels the RGB conv takes; a shuffling
        // one's four times as many.
        let resize = ups.last().is_some_and(|u| u.cout == rgb.cin);
        let stage = if w.0.contains_key("deferred.level") {
            let mut tex = |name: &str, shape: &[usize]| -> Result<u32> {
                let (s, v) = w.take(name)?;
                if s != shape {
                    return Err(format!("{name} is {s:?}, gpu_net.wgsl takes {shape:?}"));
                }
                Ok(up.push(&v))
            };
            let level = tex("deferred.level", &[8, 384, 512])?;
            let flesh = tex("deferred.flesh", &[8, 8, 256, 256])?;
            let eye = tex("deferred.eye", &[8, 8, 64, 64])?;
            let iris = tex("deferred.iris", &[4, 64, 64])?;
            let mlp = [w.linear(&mut up, "deferred.mlp.0")?, w.linear(&mut up, "deferred.mlp.2")?, w.linear(&mut up, "deferred.mlp.4")?];
            if mlp[0].inp != STAGE_IN || mlp[2].out != 3 || rgb.cin != 16 {
                return Err(format!("the stage's MLP takes {} channels, gpu_net.wgsl makes {STAGE_IN}", mlp[0].inp));
            }
            Some(Stage { level, flesh, eye, iris, mlp })
        } else {
            None
        };
        // Creature part tokens aren't ported yet: without them the network draws the
        // level alone, as it does for views with no creature in sight.
        w.0.retain(|k, _| !(k.starts_with("part_in.") || k == "shape.weight" || k == "motif.weight" || k == "part_cover"));
        if !w.0.is_empty() {
            let mut left: Vec<_> = w.0.keys().collect();
            left.sort();
            return Err(format!("unused weights (is the port behind model.py?): {left:?}"));
        }

        // Scratch space after the weights, sized for the largest view.
        let (s_max, p) = (MAX_TRIS + 1 + if organs.is_some() { MAX_ORGANS } else { 0 }, MAX_PATCHES);
        // The biggest of the head's grids: each doubling's conv output, as channels x pixels.
        let grid = ups
            .iter()
            .enumerate()
            .map(|(i, u)| if resize { u.cin.max(u.cout) * p << (2 * i + 2) } else { u.cout * p << (2 * i) })
            .chain([proj.out * p])
            .max()
            .unwrap_or(0);
        let rows = s_max.max(p);
        let mut at = up.data.len().div_ceil(64) * 64;
        let mut alloc = |n: usize| {
            let o = at as u32;
            at += n.div_ceil(64) * 64;
            o
        };
        let plane = (MAX_W * H) as usize;
        let s = Scratch {
            feats: alloc(MAX_TRIS * TRI_FEATURES),
            tex: alloc(MAX_TRIS),
            organs: alloc(MAX_ORGANS * ORGAN_FEATURES),
            organ_kinds: alloc(MAX_ORGANS),
            organ_in: alloc(MAX_ORGANS * ORGAN_IN),
            cams: alloc(CAM_FEATURES),
            mix: alloc(NUM_TEX * NUM_TEX),
            noise: alloc(p * DREAM_DIRS),
            tri_in: alloc(MAX_TRIS * TRI_IN),
            hidden: alloc(rows * 4 * d),
            cam_hidden: alloc(d),
            cam: alloc(d),
            x: alloc(s_max * d),
            xn: alloc(rows * d),
            qkv: alloc(rows * 3 * d),
            scores: alloc(heads * rows * s_max.max(p)),
            attn: alloc(rows * d),
            table: alloc(NUM_TEX * d),
            q: alloc(p * d),
            qn: alloc(p * d),
            qc: alloc(p * d),
            kv: alloc(s_max * dec_layers * 2 * d),
            dist: alloc(p * s_max),
            queries: alloc(p * d),
            centres: alloc(p * 2),
            proj: alloc(p * proj.out),
            ga: alloc(grid),
            gb: alloc(grid),
            out: alloc(3 * plane),
            pieces: alloc(if stage.is_some() { MAX_PIECES * gbuffer::PIECE } else { 0 }),
            rows: alloc(if stage.is_some() { MAX_ROWS * gbuffer::ROW } else { 0 }),
            tiles: alloc(if stage.is_some() { 2 * plane / (gbuffer::TILE * gbuffer::TILE) } else { 0 }),
            lists: alloc(if stage.is_some() { MAX_PAIRS } else { 0 }),
            raw_organs: alloc(MAX_ORGANS * RAW_ORGAN),
            view: alloc(64),
            stage_in: alloc(if stage.is_some() { STAGE_IN * plane } else { 0 }),
            stage_hidden: alloc(if stage.is_some() { 2 * 64 * plane } else { 0 }),
            echo: alloc(p * d),
            pixel_echo: alloc(rgb.cin * plane),
            gbuffer: alloc(8),
            end: alloc(0),
        };

        let bytes = s.end as u64 * 4;
        let (device, queue, adapter) = crate::neural::gpu::device("neurofield", bytes)?;

        let mem = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mem"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&mem, 0, bytemuck::cast_slice(&up.data));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: SLOT * MAX_OPS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: 3 * plane as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("neurofield"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("neurofield"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: mem.as_entire_binding() },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("neurofield"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_net.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_net.wgsl").into()),
        });
        let pipelines = KERNELS
            .iter()
            .map(|(_, entry)| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();

        let mut units: Vec<(Linear, bool)> = encoder.iter().chain(&decoder).map(|b| (b.mlp.0, false)).collect();
        if let Some(st) = &stage {
            units.extend([(st.mlp[0], true), (st.mlp[1], true)]);
        }
        Ok(GpuNet {
            device,
            queue,
            pipelines,
            mem,
            params,
            bind,
            readback,
            s,
            d,
            heads,
            tri_in,
            tri_features,
            tex,
            organs,
            cam_in,
            encoder,
            decoder,
            cross_kv,
            q_in,
            grid_w: 0,
            cover,
            noise_dirs,
            out_norm,
            proj,
            ups,
            resize,
            rgb,
            stage,
            weights: up.data,
            spans: up.spans,
            rot_target: None,
            damage: Damage::default(),
            units,
            echo_width: [None; 2],
            adapter,
        })
    }

    /// Patch queries and centres for a grid `gw` patches wide: q_in(fourier(centres)).
    fn set_grid(&mut self, gw: usize) {
        let c = centres(gw);
        let mut four = Vec::with_capacity(c.len() * 2 * BANDS);
        for xy in c.chunks_exact(2) {
            for &x in xy {
                let a: Vec<f32> = (0..BANDS as i32).map(|k| x * (2f32.powi(k - LOW) * std::f32::consts::PI)).collect();
                four.extend(a.iter().map(|a| a.sin()));
                four.extend(a.iter().map(|a| a.cos()));
            }
        }
        let [(i0, o0, w0, b0), (i2, o2, w2, b2)] = &self.q_in;
        let h: Vec<f32> = cpu_linear(&four, w0, b0, *i0, *o0).into_iter().map(gelu_cpu).collect();
        let q = cpu_linear(&h, w2, b2, *i2, *o2);
        self.queue.write_buffer(&self.mem, self.s.queries as u64 * 4, bytemuck::cast_slice(&q));
        self.queue.write_buffer(&self.mem, self.s.centres as u64 * 4, bytemuck::cast_slice(&c));
        self.grid_w = gw;
    }

    /// Whether the model has a per-pixel stage, which draws with a `StageView`.
    pub fn has_stage(&self) -> bool {
        self.stage.is_some()
    }

    /// Change the weights as `dmg` says (from those loaded), if that isn't how they are.
    pub fn set_damage(&mut self, dmg: &Damage) {
        if *dmg == self.damage {
            return;
        }
        let mut w = self.weights.clone();
        if dmg.rot > 0.0 {
            let target = self.rot_target.get_or_insert_with(|| noise_like(&self.weights, &self.spans));
            for (a, b) in w.iter_mut().zip(target.iter()) {
                *a += (b - *a) * dmg.rot;
            }
        }
        for (k, &(l, stage)) in self.units.iter().enumerate() {
            let share = if stage { dmg.stage_units } else { dmg.net_units };
            for o in 0..l.out {
                let seed = dmg.seed as u64;
                let f = if hash01(seed, k as u64, o as u64) < share {
                    if dmg.invert { -1.0 } else { 0.0 }
                } else {
                    1.0 + dmg.jitter * (2.0 * hash01(seed ^ 0xA5A5, k as u64, o as u64) - 1.0)
                };
                if f == 1.0 {
                    continue;
                }
                for i in 0..l.inp {
                    w[l.w as usize + i * l.rs + o] *= f;
                }
                w[l.b as usize + o] *= f;
            }
        }
        self.queue.write_buffer(&self.mem, 0, bytemuck::cast_slice(&w));
        self.damage = *dmg;
    }

    /// Mix the last frame's `len` floats kept at `kept` into those at `at` (`amount` of them),
    /// then keep these; echo `which` (0: patch queries, 1: head channels) at frame `width`.
    fn echo(&mut self, ops: &mut Ops, which: usize, at: u32, kept: u32, len: usize, amount: f32, width: u32) {
        let groups = [len.div_ceil(64) as u32, 1, 1];
        if amount <= 0.0 {
            self.echo_width[which] = None;
            return;
        }
        if self.echo_width[which] == Some(width) {
            ops.push(Kernel::Lerp, Params { m: len as u32, a_off: kept, c_off: at, f0: amount, ..Default::default() }, groups);
        }
        ops.push(Kernel::Lerp, Params { m: len as u32, a_off: at, c_off: kept, f0: 1.0, ..Default::default() }, groups);
        self.echo_width[which] = Some(width);
    }

    /// The network's frame for these tokens: [240 * t.width * 3] floats, roughly 0..1, row-major RGB.
    pub fn render(&mut self, t: &Tokens, dream: &Dream) -> Result<Vec<f32>> {
        self.render_staged(t, dream, None)
    }

    /// The same, with the per-pixel stage drawing on it where the model has one and `view` is given.
    pub fn render_staged(&mut self, t: &Tokens, dream: &Dream, view: Option<&StageView>) -> Result<Vec<f32>> {
        let width = t.width;
        if width % PATCH as u32 != 0 || !(PATCH as u32..=MAX_W).contains(&width) {
            return Err(format!("a frame {width} wide: it must be a multiple of {PATCH} up to {MAX_W}"));
        }
        let gw = width as usize / PATCH;
        if gw != self.grid_w {
            self.set_grid(gw);
        }
        let patches = gw * GH;
        let n = t.len().min(MAX_TRIS);
        // The organs follow the triangles, if the model knows them.
        let o = if self.organs.is_some() { t.organ_kinds.len().min(MAX_ORGANS) } else { 0 };
        let tokens = n + 1 + o; // the camera rides along as token 0
        let s = &self.s;
        let write = |at: u32, v: &[f32]| self.queue.write_buffer(&self.mem, at as u64 * 4, bytemuck::cast_slice(v));
        if n > 0 {
            write(s.feats, &t.feats[..n * TRI_FEATURES]);
            write(s.tex, &t.tex[..n].iter().map(|&x| x as f32).collect::<Vec<_>>());
        }
        if o > 0 {
            write(s.organs, &t.organs[..o * ORGAN_FEATURES]);
            write(s.organ_kinds, &t.organ_kinds[..o].iter().map(|&x| x as f32).collect::<Vec<_>>());
        }
        write(s.cams, &t.cam);
        write(s.mix, &dream.tex_mix);
        if dream.q_noise.len() != patches * DREAM_DIRS {
            return Err(format!("dream pushes for {} patches, the frame has {patches}", dream.q_noise.len() / DREAM_DIRS));
        }
        write(s.noise, &dream.q_noise);

        let mut ops = Ops::default();
        let (d, heads) = (self.d, self.heads);
        // Triangle tokens: MLP over features and their Fourier features, plus texture embeddings.
        if n > 0 {
            ops.push(
                Kernel::Prepare,
                Params {
                    m: n as u32,
                    k: self.tri_features as u32,
                    n: (self.tri_features + 6 * 2 * BANDS) as u32,
                    a_off: s.feats,
                    a_rs: TRI_FEATURES as u32,
                    c_off: s.tri_in,
                    x0: TRI_NDC,
                    ..Default::default()
                },
                [TRI_IN.div_ceil(64) as u32, n as u32, 1],
            );
            ops.linear(s.tri_in, n, self.tri_in.0, s.hidden, GELU);
            ops.linear(s.hidden, n, self.tri_in.1, s.x + d as u32, 0);
            // The texture table, mixed for the dream effects: tex_mix [12, 12] x tex [12, d].
            ops.linear(s.mix, NUM_TEX, Linear { w: self.tex, b: NONE, inp: NUM_TEX, out: d, rs: d }, s.table, 0);
            ops.push(
                Kernel::AddTexture,
                Params { m: n as u32, n: d as u32, a_off: s.tex, b_off: s.table, c_off: s.x, ..Default::default() },
                [d.div_ceil(64) as u32, n as u32, 1],
            );
        }
        if let (Some((mlp, kinds, _)), true) = (self.organs, o > 0) {
            // Organ tokens likewise, from the Fourier features of their screen boxes, plus
            // their kinds' embeddings.
            ops.push(
                Kernel::Prepare,
                Params {
                    m: o as u32,
                    k: ORGAN_FEATURES as u32,
                    n: ORGAN_IN as u32,
                    a_off: s.organs,
                    a_rs: ORGAN_FEATURES as u32,
                    c_off: s.organ_in,
                    x0: ORGAN_BOX,
                    ..Default::default()
                },
                [ORGAN_IN.div_ceil(64) as u32, o as u32, 1],
            );
            ops.linear(s.organ_in, o, mlp.0, s.hidden, GELU);
            let first = s.x + ((1 + n) * d) as u32;
            ops.linear(s.hidden, o, mlp.1, first, 0);
            ops.push(
                Kernel::AddTexture,
                Params { m: o as u32, n: d as u32, a_off: s.organ_kinds, b_off: kinds, c_off: first - d as u32, ..Default::default() },
                [d.div_ceil(64) as u32, o as u32, 1],
            );
        }
        ops.linear(s.cams, 1, self.cam_in.0, s.cam_hidden, GELU);
        ops.linear(s.cam_hidden, 1, self.cam_in.1, s.x, 0);
        ops.linear(s.cam_hidden, 1, self.cam_in.1, s.cam, 0);
        for blk in &self.encoder {
            ops.block_self(blk, s, s.x, tokens, d, heads);
        }

        // Patch queries: the precomputed base + the dream pushes + the camera token (as a bias).
        ops.linear(s.noise, patches, Linear { w: self.noise_dirs, b: s.cam, inp: DREAM_DIRS, out: d, rs: d }, s.q, 0);
        ops.push(
            Kernel::Add,
            Params { m: (patches * d) as u32, a_off: s.queries, c_off: s.q, ..Default::default() },
            [(patches * d).div_ceil(64) as u32, 1, 1],
        );
        ops.linear(s.x, tokens, self.cross_kv, s.kv, 0);
        ops.push(
            Kernel::CoverDistance,
            Params {
                m: patches as u32,
                k: TRI_FEATURES as u32,
                n: tokens as u32,
                a_off: s.feats,
                b_off: s.centres,
                c_off: s.dist,
                x0: (1 + n) as u32,
                x1: s.organs,
                f0: 2.0 * PATCH as f32 / W as f32,
                ..Default::default()
            },
            [tokens.div_ceil(64) as u32, patches as u32, 1],
        );
        let kv_rs = self.cross_kv.out;
        for (l, blk) in self.decoder.iter().enumerate() {
            ops.block_self(blk, s, s.q, patches, d, heads);
            let (n2, xq, xout) = blk.cross.expect("decoder blocks cross-attend");
            ops.norm(s.q, patches, d, n2, s.qn);
            ops.linear(s.qn, patches, xq, s.qc, 0);
            let kv = s.kv + (l * 2 * d) as u32;
            let organ_cover = self.organs.filter(|_| o > 0).map(|(_, _, c)| (c + (l * heads) as u32, (1 + n) as u32));
            let cover = (self.cover + (l * heads) as u32, s.dist, organ_cover);
            let alpha = 1.0 / ((d / heads) as f32).sqrt() / dream.temp;
            ops.attention(
                s,
                (s.qc, d),
                (kv, kv_rs),
                (kv + d as u32, kv_rs),
                patches,
                tokens,
                d,
                heads,
                alpha,
                Some(cover),
            );
            ops.linear(s.attn, patches, xout, s.q, ACCUMULATE);
            ops.mlp(blk, s, s.q, patches, d);
        }

        let (q, kept) = (s.q, s.echo);
        self.echo(&mut ops, 0, q, kept, patches * d, dream.echo, width);
        let s = &self.s;

        // Pixels: the head projects the queries to a grid of channels, doubles it four times
        // (conv, then pixel shuffle) and convolves it to RGB.
        ops.norm(s.q, patches, d, self.out_norm, s.qn);
        ops.linear(s.qn, patches, self.proj, s.proj, 0);
        let c = self.proj.out;
        ops.push(
            Kernel::Transpose,
            Params { m: patches as u32, n: c as u32, a_off: s.proj, c_off: s.ga, ..Default::default() },
            [c.div_ceil(64) as u32, patches as u32, 1],
        );
        let conv = |ops: &mut Ops, u: Up, from: u32, to: u32, (gh, gw): (usize, usize), flags: u32| {
            let p = Params {
                m: gh as u32,
                k: u.cin as u32,
                n: gw as u32,
                a_off: from,
                b_off: u.w,
                bias_off: u.b,
                c_off: to,
                x0: u.cout as u32,
                flags,
                ..Default::default()
            };
            ops.push(Kernel::Conv3x3, p, [gw.div_ceil(16) as u32, gh.div_ceil(16) as u32, u.cout.div_ceil(8) as u32]);
        };
        let (mut from, mut to, mut size) = (s.ga, s.gb, (GH, gw));
        for &u in &self.ups {
            if self.resize {
                // Bilinear to twice the size, then the conv at that size.
                let big = (size.0 * 2, size.1 * 2);
                ops.push(
                    Kernel::Upsample2,
                    Params { m: big.0 as u32, n: big.1 as u32, a_off: from, c_off: to, ..Default::default() },
                    [big.1.div_ceil(16) as u32, big.0.div_ceil(16) as u32, u.cin as u32],
                );
                conv(&mut ops, u, to, from, big, GELU);
                size = big;
            } else {
                conv(&mut ops, u, from, to, size, SHUFFLE | GELU);
                (from, to, size) = (to, from, (size.0 * 2, size.1 * 2));
            }
        }
        if size != (H as usize, width as usize) {
            return Err(format!("the head makes {size:?} images, not 240 x {width}"));
        }
        let kept = s.pixel_echo;
        self.echo(&mut ops, 1, from, kept, self.rgb.cin * size.0 * size.1, dream.pixel_echo, width);
        conv(&mut ops, self.rgb, from, self.s.out, size, INTERLEAVE);
        if let (Some(stage), Some(view)) = (&self.stage, view) {
            self.stage_ops(&mut ops, stage, view, from, width as usize, dream)?;
        }
        self.run(&ops, 3 * H as usize * width as usize)
    }

    /// The per-pixel stage, added to the frame at `s.out`: its G-buffer and latents from the
    /// view, with the head's last channels at `head`, then its MLP.
    fn stage_ops(&self, ops: &mut Ops, stage: &Stage, v: &StageView, head: u32, width: usize, dream: &Dream) -> Result<()> {
        let s = &self.s;
        let g = gbuffer::pieces(v.tris, v.turning, v.view);
        let (pieces, rows) = (g.pieces.len() / gbuffer::PIECE, g.rows.len() / gbuffer::ROW);
        if pieces > MAX_PIECES || rows > MAX_ROWS || g.lists.len() > MAX_PAIRS {
            return Err(format!("{pieces} G-buffer pieces of {rows} triangles in {} places: too many", g.lists.len()));
        }
        let organs = (v.organs.len() / RAW_ORGAN).min(MAX_ORGANS);
        let write = |at: u32, x: &[f32]| {
            if !x.is_empty() {
                self.queue.write_buffer(&self.mem, at as u64 * 4, bytemuck::cast_slice(x));
            }
        };
        write(s.pieces, &g.pieces);
        write(s.rows, &g.rows);
        write(s.tiles, &g.tiles);
        write(s.lists, &g.lists);
        write(s.raw_organs, &v.organs[..organs * RAW_ORGAN]);
        write(s.view, v.view);
        write(s.gbuffer, &dream.gbuffer.floats(dream.time));
        let pixels = H as usize * width;
        ops.push(
            Kernel::StageInputs,
            Params {
                m: H,
                n: width as u32,
                k: organs as u32,
                a_off: s.pieces,
                b_off: s.rows,
                c_off: s.stage_in,
                bias_off: head,
                x0: s.tiles,
                x1: s.lists,
                x2: s.raw_organs,
                a_bs: s.view,
                a_rs: stage.level,
                a_cs: stage.flesh,
                b_bs: stage.eye,
                b_rs: stage.iris,
                f0: v.amount,
                f1: dream.warp,
                f2: dream.smear,
                f3: dream.time,
                c_bs: s.gbuffer,
                ..Default::default()
            },
            [(width / gbuffer::TILE) as u32, H / gbuffer::TILE as u32, 1],
        );
        // The MLP, pixel by pixel: its input is planes, so read down them.
        let [l0, l1, l2] = stage.mlp;
        let first = Params {
            m: pixels as u32,
            k: l0.inp as u32,
            n: l0.out as u32,
            batch: 1,
            a_off: s.stage_in,
            a_rs: 1,
            a_cs: pixels as u32,
            b_off: l0.w,
            b_rs: l0.rs as u32,
            b_cs: 1,
            c_off: s.stage_hidden,
            c_rs: l0.out as u32,
            bias_off: l0.b,
            flags: GELU,
            ..Default::default()
        };
        ops.push(Kernel::Matmul, first, [l0.out.div_ceil(32) as u32, pixels.div_ceil(32) as u32, 1]);
        let second = s.stage_hidden + (64 * pixels) as u32;
        ops.linear(s.stage_hidden, pixels, l1, second, GELU);
        ops.linear(second, pixels, l2, s.out, ACCUMULATE);
        Ok(())
    }

    /// The per-pixel stage's inputs from the last frame drawn with it, [STAGE_IN][240 * width]
    /// (for checking against PyTorch).
    pub fn stage_inputs(&self, width: u32) -> Result<Vec<f32>> {
        let floats = STAGE_IN * H as usize * width as usize;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stage inputs"),
            size: floats as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.mem, self.s.stage_in as u64 * 4, &buf, 0, floats as u64 * 4);
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        let out = bytemuck::cast_slice(&slice.get_mapped_range().map_err(|e| e.to_string())?).to_vec();
        Ok(out)
    }

    /// Run the dispatches as one submission and read back the first `floats` of the output.
    fn run(&mut self, ops: &Ops, floats: usize) -> Result<Vec<f32>> {
        if ops.list.len() as u64 > MAX_OPS {
            return Err(format!("{} dispatches, room for {MAX_OPS}", ops.list.len()));
        }
        let mut slots = vec![0u8; ops.list.len() * SLOT as usize];
        for (i, (_, p, _)) in ops.list.iter().enumerate() {
            slots[i * SLOT as usize..][..std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        self.queue.write_buffer(&self.params, 0, &slots);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (i, (kernel, _, groups)) in ops.list.iter().enumerate() {
                if groups.contains(&0) {
                    continue;
                }
                pass.set_pipeline(&self.pipelines[*kernel as usize]);
                pass.set_bind_group(0, &self.bind, &[(i as u64 * SLOT) as u32]);
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
            }
        }
        let size = floats as u64 * 4;
        enc.copy_buffer_to_buffer(&self.mem, self.s.out as u64 * 4, &self.readback, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        let out = bytemuck::cast_slice(&slice.get_mapped_range().map_err(|e| e.to_string())?).to_vec();
        self.readback.unmap();
        Ok(out)
    }
}

/// The frame's dispatches: kernel, parameters and workgroup counts.
#[derive(Default)]
struct Ops {
    list: Vec<(Kernel, Params, [u32; 3])>,
}

impl Ops {
    fn push(&mut self, kernel: Kernel, p: Params, groups: [u32; 3]) {
        self.list.push((kernel, p, groups));
    }

    /// C = act(A W + b) for A [rows, l.inp] (contiguous) at `a`, written (or added with
    /// ACCUMULATE) to C [rows, l.out] at `c`. A bias of NONE means none.
    fn linear(&mut self, a: u32, rows: usize, l: Linear, c: u32, flags: u32) {
        let p = Params {
            m: rows as u32,
            k: l.inp as u32,
            n: l.out as u32,
            batch: 1,
            a_off: a,
            a_rs: l.inp as u32,
            a_cs: 1,
            b_off: l.w,
            b_rs: l.rs as u32,
            b_cs: 1,
            c_off: c,
            c_rs: l.out as u32,
            bias_off: l.b,
            flags,
            ..Default::default()
        };
        self.push(Kernel::Matmul, p, [l.out.div_ceil(32) as u32, rows.div_ceil(32) as u32, 1]);
    }

    fn norm(&mut self, a: u32, rows: usize, d: usize, n: Norm, c: u32) {
        let p = Params {
            m: rows as u32,
            n: d as u32,
            a_off: a,
            a_rs: d as u32,
            b_off: n.g,
            bias_off: n.b,
            c_off: c,
            c_rs: d as u32,
            ..Default::default()
        };
        self.push(Kernel::LayerNorm, p, [rows as u32, 1, 1]);
    }

    /// Multi-head attention of queries [l, d] (at q.0, row stride q.1) to keys and values
    /// [s, d] (likewise), the scores scaled by `alpha` and, with `cover` (strengths, distances,
    /// and the organs' strengths and first column if any), biased toward covering triangles
    /// and organs; the heads' outputs side by side in `attn` [l, d].
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        sc: &Scratch,
        q: (u32, usize),
        k: (u32, usize),
        v: (u32, usize),
        l: usize,
        s: usize,
        d: usize,
        heads: usize,
        alpha: f32,
        cover: Option<(u32, u32, Option<(u32, u32)>)>,
    ) {
        let dh = d / heads;
        let scores = Params {
            m: l as u32,
            k: dh as u32,
            n: s as u32,
            batch: heads as u32,
            a_off: q.0,
            a_bs: dh as u32,
            a_rs: q.1 as u32,
            a_cs: 1,
            b_off: k.0,
            b_bs: dh as u32,
            b_rs: 1,
            b_cs: k.1 as u32,
            c_off: sc.scores,
            c_bs: (l * s) as u32,
            c_rs: s as u32,
            bias_off: NONE,
            ..Default::default()
        };
        self.push(Kernel::Matmul, scores, [s.div_ceil(32) as u32, l.div_ceil(32) as u32, heads as u32]);
        let (strength, dist, organs) = cover.unwrap_or((NONE, 0, None));
        let (organ_strength, first_organ) = organs.unwrap_or((NONE, 0));
        let soft = Params {
            m: l as u32,
            k: first_organ,
            n: s as u32,
            batch: heads as u32,
            a_off: sc.scores,
            b_off: strength,
            c_off: organ_strength,
            x1: dist,
            x2: s as u32,
            f0: alpha,
            ..Default::default()
        };
        self.push(Kernel::Softmax, soft, [l as u32, heads as u32, 1]);
        let mix = Params {
            m: l as u32,
            k: s as u32,
            n: dh as u32,
            batch: heads as u32,
            a_off: sc.scores,
            a_bs: (l * s) as u32,
            a_rs: s as u32,
            a_cs: 1,
            b_off: v.0,
            b_bs: dh as u32,
            b_rs: v.1 as u32,
            b_cs: 1,
            c_off: sc.attn,
            c_bs: dh as u32,
            c_rs: d as u32,
            bias_off: NONE,
            ..Default::default()
        };
        self.push(Kernel::Matmul, mix, [dh.div_ceil(32) as u32, l.div_ceil(32) as u32, heads as u32]);
    }

    /// x += attn(norm1(x)) over the block's own tokens [rows, d] at `x`, then the MLP for
    /// encoder blocks (decoder blocks cross-attend before theirs).
    fn block_self(&mut self, blk: &Block, s: &Scratch, x: u32, rows: usize, d: usize, heads: usize) {
        self.norm(x, rows, d, blk.n1, s.xn);
        self.linear(s.xn, rows, blk.qkv, s.qkv, 0);
        let rs = 3 * d;
        let alpha = 1.0 / ((d / heads) as f32).sqrt();
        self.attention(
            s,
            (s.qkv, rs),
            (s.qkv + d as u32, rs),
            (s.qkv + 2 * d as u32, rs),
            rows,
            rows,
            d,
            heads,
            alpha,
            None,
        );
        self.linear(s.attn, rows, blk.out, x, ACCUMULATE);
        if blk.cross.is_none() {
            self.mlp(blk, s, x, rows, d);
        }
    }

    /// x += mlp(norm3(x)).
    fn mlp(&mut self, blk: &Block, s: &Scratch, x: u32, rows: usize, d: usize) {
        self.norm(x, rows, d, blk.n3, s.xn);
        self.linear(s.xn, rows, blk.mlp.0, s.hidden, GELU);
        self.linear(s.hidden, rows, blk.mlp.1, x, ACCUMULATE);
    }
}
