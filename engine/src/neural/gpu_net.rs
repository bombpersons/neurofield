//! The trained renderer as hand-written compute shaders (gpu_net.wgsl) run through wgpu:
//! Vulkan, Metal or DX12, in pure Rust. A port of `Renderer.forward` in train/model.py for
//! one view at a time; keep the two in step. Weights come from train/export_weights.py.
//!
//! Weights, inputs and every intermediate result share one storage buffer, and each
//! dispatch reads its sizes and offsets from its own slot of a uniform buffer, so a frame
//! is one submission: upload the tokens, run ~150 small kernels, read back the image. The
//! kernels take any sizes, so nothing is compiled per triangle count.

use crate::neural::dream::{Dream, DREAM_DIRS, NUM_TEX};
use crate::neural::tokens::{Tokens, CAM_FEATURES, MAX_TRIS, TRI_FEATURES};
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
struct Params {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    a_off: u32,
    a_bs: u32,
    a_rs: u32,
    a_cs: u32,
    b_off: u32,
    b_bs: u32,
    b_rs: u32,
    b_cs: u32,
    c_off: u32,
    c_bs: u32,
    c_rs: u32,
    bias_off: u32,
    flags: u32,
    x0: u32,
    x1: u32,
    x2: u32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
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
}

const KERNELS: [(Kernel, &str); 9] = [
    (Kernel::Matmul, "matmul"),
    (Kernel::LayerNorm, "layer_norm"),
    (Kernel::Softmax, "softmax"),
    (Kernel::Prepare, "prepare"),
    (Kernel::AddTexture, "add_texture"),
    (Kernel::CoverDistance, "cover_distance"),
    (Kernel::Transpose, "transpose"),
    (Kernel::Conv3x3, "conv3x3"),
    (Kernel::Add, "add"),
];

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
    end: u32,
}

/// One doubling of the upsampling head: 3x3 conv weights [4 cout][cin][3][3], bias, sizes.
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
}

impl Upload {
    /// Append `v` (16-float aligned) and return its offset.
    fn push(&mut self, v: &[f32]) -> u32 {
        self.data.resize(self.data.len().div_ceil(16) * 16, 0.0);
        let at = self.data.len() as u32;
        self.data.extend_from_slice(v);
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
    tex: u32,
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
    rgb: Up,
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
        let mut up = Upload { data: Vec::new() };

        let tri_in = (w.linear(&mut up, "tri_in.0")?, w.linear(&mut up, "tri_in.2")?);
        let d = tri_in.1.out;
        let tex = up.push(&w.take("tex.weight")?.1);
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
        // Creature part tokens aren't ported yet: without them the network draws the
        // level alone, as it does for views with no creature in sight.
        w.0.retain(|k, _| !(k.starts_with("part_in.") || k == "shape.weight" || k == "motif.weight" || k == "part_cover"));
        if !w.0.is_empty() {
            let mut left: Vec<_> = w.0.keys().collect();
            left.sort();
            return Err(format!("unused weights (is the port behind model.py?): {left:?}"));
        }

        // Scratch space after the weights, sized for the largest view.
        let (s_max, p) = (MAX_TRIS + 1, MAX_PATCHES);
        // The biggest of the head's grids: each doubling's conv output, as channels x pixels.
        let grid = ups.iter().enumerate().map(|(i, u)| u.cout * p << (2 * i)).chain([proj.out * p]).max().unwrap_or(0);
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
            end: alloc(0),
        };

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let bytes = s.end as u64 * 4;
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.max(bytes);
        limits.max_buffer_size = limits.max_buffer_size.max(bytes);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("neurofield"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| format!("GPU device: {e}"))?;

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
            tex,
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
            rgb,
            adapter: adapter.get_info().name,
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

    /// The network's frame for these tokens: [240 * t.width * 3] floats, roughly 0..1, row-major RGB.
    pub fn render(&mut self, t: &Tokens, dream: &Dream) -> Result<Vec<f32>> {
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
        let tokens = n + 1; // the camera rides along as token 0
        let s = &self.s;
        let write = |at: u32, v: &[f32]| self.queue.write_buffer(&self.mem, at as u64 * 4, bytemuck::cast_slice(v));
        if n > 0 {
            write(s.feats, &t.feats[..n * TRI_FEATURES]);
            write(s.tex, &t.tex[..n].iter().map(|&x| x as f32).collect::<Vec<_>>());
        }
        write(s.cams, &t.cam);
        write(s.mix, &dream.tex_mix);
        // The dream's pushes are per patch of a 4:3 grid; other widths go without.
        if dream.q_noise.len() == patches * DREAM_DIRS {
            write(s.noise, &dream.q_noise);
        } else {
            write(s.noise, &vec![0.0; patches * DREAM_DIRS]);
        }

        let mut ops = Ops::default();
        let (d, heads) = (self.d, self.heads);
        // Triangle tokens: MLP over features and their Fourier features, plus texture embeddings.
        if n > 0 {
            ops.push(
                Kernel::Prepare,
                Params {
                    m: n as u32,
                    k: TRI_FEATURES as u32,
                    n: TRI_IN as u32,
                    a_off: s.feats,
                    c_off: s.tri_in,
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
            let cover = (self.cover + (l * heads) as u32, s.dist);
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
            conv(&mut ops, u, from, to, size, SHUFFLE | GELU);
            (from, to, size) = (to, from, (size.0 * 2, size.1 * 2));
        }
        if size != (H as usize, width as usize) {
            return Err(format!("the head makes {size:?} images, not 240 x {width}"));
        }
        conv(&mut ops, self.rgb, from, s.out, size, INTERLEAVE);
        self.run(&ops, 3 * H as usize * width as usize)
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
    /// [s, d] (likewise), the scores scaled by `alpha` and, with `cover` (strengths, distances),
    /// biased toward covering triangles; the heads' outputs side by side in `attn` [l, d].
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
        cover: Option<(u32, u32)>,
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
        let (strength, dist) = cover.unwrap_or((NONE, 0));
        let soft = Params {
            m: l as u32,
            n: s as u32,
            batch: heads as u32,
            a_off: sc.scores,
            b_off: strength,
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
