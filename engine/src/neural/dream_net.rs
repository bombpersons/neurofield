//! The dream creature net (train/dream.py) as WGSL compute shaders: each part's per-pixel
//! fields, the U-Net, and its layer laid over the level network's frame where the parts
//! are nearer than the level. Weights and phrase tables come from train/export_dream.py.

use super::features::{part_features, PART_FEATURES};
use super::gpu_net::{read_safetensors, Params, MAX_W};
use crate::world::rng::Rng;
use crate::world::scene::{Camera, Scene, T};
use crate::world::view::{view_matrix, H};
use std::collections::HashMap;
use std::path::Path;

type Result<T> = std::result::Result<T, String>;

const MAX_PARTS: usize = 96;
const AXIS_POINTS: usize = 5;
const PALETTE: usize = 12;
const PHRASE_DIM: usize = 64;
/// dream.py's `Q_PARAMS`, and where its fields are.
pub const Q_PARAMS: usize = 22;
const Q_HALF: usize = 9;
const Q_BEND: usize = 12;
const Q_LIGHT: usize = 15;
const Q_PHRASE: usize = 19;
const Q_SEED: usize = 20;
const Q_VALID: usize = 21;
/// Floats per part in `mem` (dream_net.wgsl).
const PART_FLOATS: usize = 104;
/// Planes the `dream_fields` kernel writes; the first `NET_IN` are the net's input.
const FIELD_PLANES: usize = 117;
const FIELDS: usize = 39;
const NET_IN: usize = FIELDS + PHRASE_DIM;
const PLANE_FOG: usize = 115;
/// Floats per pixel of the finished layer: rgb, alpha, fog, depth.
const LAYER: usize = 6;
/// Of `Dream.forward`'s share of phrases, the part that dreams instead of following its motif.
const DREAM_SHARE: f32 = 0.25;
const MAX_OPS: u64 = 128;
const SLOT: u64 = 256;
const GELU: u32 = 1;
const ACCUMULATE: u32 = 2;
/// How far round each part the drawn box reaches, in its radii: nothing shows further out.
const CROP_MARGIN: f32 = 2.5;
/// Alpha at which the drawn layer starts and finishes being opaque (`edge`).
const EDGE: (f32, f32) = (0.3, 0.7);

#[derive(Clone, Copy)]
enum Kernel {
    Conv3x3,
    Fields,
    Upsample,
    Finish,
    Pool,
    Film,
}
const KERNELS: [&str; 6] = ["conv3x3", "dream_fields", "upsample2", "dream_finish", "dream_pool2", "dream_film"];

/// A convolution's weights in `mem`.
#[derive(Clone, Copy)]
struct Conv {
    w: u32,
    b: u32,
    cin: u32,
    cout: u32,
}

/// x + b(gelu(a(x))).
#[derive(Clone, Copy)]
struct ResBlock {
    a: Conv,
    b: Conv,
}

/// A level's creature parts, with the phrase, noise seed and colour scheme each keeps.
pub struct Parts {
    rows: Vec<f32>,
    phrase: Vec<f32>,
    seed: Vec<f32>,
    palette: Vec<f32>,
}

/// The finished layer of a box of the screen.
pub struct Layer {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    /// [h][w][LAYER]: rgb, alpha, fog, depth.
    pub data: Vec<f32>,
}

pub struct DreamNet {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    mem: wgpu::Buffer,
    params: wgpu::Buffer,
    bind: wgpu::BindGroup,
    readback: wgpu::Buffer,
    /// In `mem`: the parts of this frame, and where the frame's planes may start.
    parts_at: u32,
    scratch: u32,
    inp: Conv,
    enc: [Vec<ResBlock>; 3],
    down: [Conv; 2],
    up: [Conv; 2],
    dec: [Vec<ResBlock>; 2],
    out: Conv,
    film: [Conv; 3],
    phrases: Vec<f32>,
    kind_ids: Vec<f32>,
    kind_start: Vec<f32>,
    dream_ids: Vec<f32>,
    palettes: Vec<f32>,
    pub adapter: String,
}

fn take(t: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    t.remove(name).ok_or(format!("no {name} in the weights"))
}

impl DreamNet {
    pub fn load(path: &Path) -> Result<DreamNet> {
        let (mut t, _) = read_safetensors(path)?;
        let mut data: Vec<f32> = Vec::new();
        let mut push = |v: &[f32]| {
            data.resize(data.len().div_ceil(16) * 16, 0.0);
            let at = data.len() as u32;
            data.extend_from_slice(v);
            at
        };
        let mut conv = |t: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str| -> Result<Conv> {
            let (shape, w) = take(t, &format!("{name}.weight"))?;
            let (_, b) = take(t, &format!("{name}.bias"))?;
            Ok(Conv { w: push(&w), b: push(&b), cin: shape[1] as u32, cout: shape[0] as u32 })
        };
        let inp = conv(&mut t, "inp")?;
        let mut blocks = |t: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str| -> Result<Vec<ResBlock>> {
            (0..2)
                .map(|i| Ok(ResBlock { a: conv(t, &format!("{name}.{i}.a"))?, b: conv(t, &format!("{name}.{i}.b"))? }))
                .collect()
        };
        let enc = [blocks(&mut t, "enc.0")?, blocks(&mut t, "enc.1")?, blocks(&mut t, "enc.2")?];
        let dec = [blocks(&mut t, "dec.0")?, blocks(&mut t, "dec.1")?];
        drop(blocks);
        let down = [conv(&mut t, "down.0")?, conv(&mut t, "down.1")?];
        let up = [conv(&mut t, "up.0")?, conv(&mut t, "up.1")?];
        let out = conv(&mut t, "out")?;
        let film = [conv(&mut t, "film.0")?, conv(&mut t, "film.1")?, conv(&mut t, "film.2")?];
        drop(conv);
        let phrases = take(&mut t, "phrase")?.1;
        let kind_ids = take(&mut t, "kind_ids")?.1;
        let kind_start = take(&mut t, "kind_start")?.1;
        let dream_ids = take(&mut t, "dream_ids")?.1;
        let palettes = take(&mut t, "palettes")?.1;
        if !t.is_empty() {
            return Err(format!("unused weights: {:?}", t.keys().collect::<Vec<_>>()));
        }

        let parts_at = (data.len().div_ceil(64) * 64) as u32;
        let scratch = parts_at + (MAX_PARTS * PART_FLOATS).div_ceil(64) as u32 * 64;
        // The planes of the biggest box: full, half and quarter size, and the layer.
        let plane = (MAX_W * H) as usize;
        let planes = FIELD_PLANES + 6 + 144 + 96 + 48 + (224 + 192 + 96) / 4 + (128 + 256) / 16 + (PHRASE_DIM + 192) / 4 + (PHRASE_DIM + 256) / 16 + 96 + LAYER;
        let bytes = (scratch as u64 + (planes * plane) as u64 + 64 * 32) * 4;

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.max(bytes);
        limits.max_buffer_size = limits.max_buffer_size.max(bytes);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("dream"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| format!("GPU device: {e}"))?;
        let mem = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dream mem"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&mem, 0, bytemuck::cast_slice(&data));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dream params"),
            size: SLOT * MAX_OPS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dream readback"),
            // Room for a fixture's fields too (`check`).
            size: ((plane * LAYER).max(128 * 128 * FIELD_PLANES) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream"),
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
            label: Some("dream"),
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
            label: Some("dream"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source = format!("{}\n{}", include_str!("gpu_net.wgsl"), include_str!("dream_net.wgsl"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dream_net.wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = KERNELS
            .iter()
            .map(|entry| {
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
        Ok(DreamNet {
            device,
            queue,
            pipelines,
            mem,
            params,
            bind,
            readback,
            parts_at,
            scratch,
            inp,
            enc,
            down,
            up,
            dec,
            out,
            film,
            phrases,
            kind_ids,
            kind_start,
            dream_ids,
            palettes,
            adapter: adapter.get_info().name,
        })
    }

    /// The scene's creature parts, each given the phrase, seed and colour scheme it keeps.
    pub fn parts(&self, scene: &Scene, seed: u64) -> Parts {
        let mut rows = Vec::new();
        part_features(scene, &mut rows);
        let n = rows.len() / PART_FEATURES;
        let mut parts = Parts { rows, phrase: Vec::new(), seed: Vec::new(), palette: Vec::new() };
        let schemes = self.palettes.len() / PALETTE;
        let dreams = self.dream_ids.len() / 4;
        for i in 0..n {
            let row = &parts.rows[i * PART_FEATURES..][..PART_FEATURES];
            let (shape, motif, creature) = (row[24] as usize, row[29] as usize, row[48] as u64);
            let mut rng = Rng::new(seed.wrapping_mul(31).wrapping_add(i as u64 + 1));
            let id = if rng.f() < DREAM_SHARE {
                self.dream_ids[shape * dreams + rng.next() as usize % dreams]
            } else {
                let (a, b) = (self.kind_start[motif * 4 + shape] as usize, self.kind_start[motif * 4 + shape + 1] as usize);
                self.kind_ids[a + rng.next() as usize % (b - a)]
            };
            parts.phrase.push(id);
            parts.seed.push((rng.next() >> 8) as f32);
            let mut scheme = Rng::new(seed.wrapping_mul(37).wrapping_add(creature + 1));
            let s = scheme.next() as usize % schemes;
            // The creature's colours, in this part's own order: the net paints by position in the scheme.
            let mut order = [0usize, 1, 2, 3];
            let mut shuffle = Rng::new(seed.wrapping_mul(41).wrapping_add(i as u64 + 1));
            for k in (1..order.len()).rev() {
                order.swap(k, shuffle.next() as usize % (k + 1));
            }
            for c in order {
                parts.palette.extend_from_slice(&self.palettes[s * PALETTE + c * 3..][..3]);
            }
        }
        parts
    }

    /// The parts seen from `cam` on a screen `width` wide: up to MAX_PARTS, the biggest on screen first, as dream.py's q
    /// [n, Q_PARAMS], with their colour schemes [n, PALETTE].
    fn view(&self, parts: &Parts, cam: &Camera, width: u32) -> (Vec<f32>, Vec<f32>) {
        let m = view_matrix(cam);
        let r = |i: usize, j: usize| m.m[i][j] as f32 / 4096.0;
        let pos = cam.pos.map(|c| c as f32 / T as f32);
        let vh = cam.h as f32;
        let mut near = Vec::new();
        for (i, row) in parts.rows.chunks_exact(PART_FEATURES).enumerate() {
            let d = [row[3] - pos[0], row[4] - pos[1], row[5] - pos[2]];
            let c = [0, 1, 2].map(|k| r(k, 0) * d[0] + r(k, 1) * d[1] + r(k, 2) * d[2]);
            let reach = row[15].abs().max(row[16].abs()).max(row[17].abs());
            let z = c[2].max(0.05);
            let (px, py, rp) = (c[0] / z * vh + width as f32 / 2.0, c[1] / z * vh + H as f32 / 2.0, 3.0 * reach / z * vh);
            if c[2] + reach > 0.05 && px + rp > 0.0 && px - rp < width as f32 && py + rp > 0.0 && py - rp < H as f32 {
                near.push((-reach / z, i, c));
            }
        }
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        near.truncate(MAX_PARTS);
        let (mut q, mut palette) = (Vec::new(), Vec::new());
        for &(_, i, c) in &near {
            let row = &parts.rows[i * PART_FEATURES..][..PART_FEATURES];
            let axis = |col: usize| [0, 1, 2].map(|k| r(k, 0) * row[6 + col] + r(k, 1) * row[9 + col] + r(k, 2) * row[12 + col]);
            q.extend_from_slice(&c);
            q.extend_from_slice(&axis(0));
            q.extend_from_slice(&axis(1));
            q.extend_from_slice(&row[15..18]);
            q.extend_from_slice(&[row[26], row[24], row[29]]);
            q.extend_from_slice(&row[35..38]);
            q.extend_from_slice(&[row[48], parts.phrase[i], parts.seed[i], 1.0]);
            palette.extend_from_slice(&parts.palette[i * PALETTE..][..PALETTE]);
        }
        (q, palette)
    }

    /// The layer of the parts `q` seen from a crop at (x0, y0), `w` x `h` (multiples of 4): each
    /// part is projected here and handed to the shaders with its phrase vector and scheme.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, q: &[f32], palette: &[f32], cam: [f32; 6], x0: usize, y0: usize, w: usize, h: usize) -> Result<Layer> {
        let [vh, width, fog_near, fog_far, ..] = cam;
        let n = q.len() / Q_PARAMS;
        let mut rows = vec![0f32; n * PART_FLOATS];
        for (i, row) in rows.chunks_exact_mut(PART_FLOATS).enumerate() {
            let p = &q[i * Q_PARAMS..][..Q_PARAMS];
            let hy = p[Q_HALF + 1];
            for k in 0..AXIS_POINTS {
                let t = -1.0 + 0.5 * k as f32;
                let s = (t + 1.0) / 2.0;
                let point = [0, 1, 2].map(|c| p[c] + p[6 + c] * hy * t + p[3 + c] * p[Q_BEND] * hy * s * s);
                let z = point[2].max(0.05);
                row[k * 3] = point[0] / z * vh + width / 2.0 - x0 as f32;
                row[k * 3 + 1] = point[1] / z * vh + H as f32 / 2.0 - y0 as f32;
                row[k * 3 + 2] = z;
            }
            row[15] = 0.5 * (p[Q_HALF] + p[Q_HALF + 2]);
            row[16] = hy;
            row[17] = p[Q_SEED];
            row[18..21].copy_from_slice(&p[Q_LIGHT..Q_LIGHT + 3]);
            row[21] = p[Q_VALID];
            row[22..22 + PHRASE_DIM].copy_from_slice(&self.phrases[p[Q_PHRASE] as usize * PHRASE_DIM..][..PHRASE_DIM]);
            row[86..86 + PALETTE].copy_from_slice(&palette[i * PALETTE..][..PALETTE]);
            let (ax, ay) = ([p[3], p[4], p[5]], [p[6], p[7], p[8]]);
            row[98..101].copy_from_slice(&ax);
            row[101..104].copy_from_slice(&[ax[1] * ay[2] - ax[2] * ay[1], ax[2] * ay[0] - ax[0] * ay[2], ax[0] * ay[1] - ax[1] * ay[0]]);
        }
        self.queue.write_buffer(&self.mem, self.parts_at as u64 * 4, bytemuck::cast_slice(&rows));

        let (full, half, quarter) = ((h, w), (h / 2, w / 2), (h / 4, w / 4));
        let area = |s: (usize, usize)| s.0 * s.1;
        let mut at = self.scratch as usize;
        let mut alloc = |planes: usize, size: (usize, usize)| {
            let o = at as u32;
            at += (planes * area(size)).div_ceil(64) * 64;
            o
        };
        let fields = alloc(FIELD_PLANES, full);
        // Each skip's planes right after the planes upsampled to join them, as `up` reads both.
        let u0 = alloc(96 + 48, full);
        let x0p = u0 + (96 * area(full)) as u32;
        let u1 = alloc(128 + 96, half);
        let x1p = u1 + (128 * area(half)) as u32;
        let x2p = alloc(128, quarter);
        let (t0, t1, t2) = (alloc(48, full), alloc(96, half), alloc(128, quarter));
        let (d0, d1) = (alloc(48, full), alloc(96, half));
        let (emb1, emb2) = (alloc(PHRASE_DIM, half), alloc(PHRASE_DIM, quarter));
        let (film0, film1, film2) = (alloc(96, full), alloc(192, half), alloc(256, quarter));
        let net_out = alloc(6, full);
        let layer = alloc(LAYER, full);

        let mut ops = Ops::default();
        ops.push(
            Kernel::Fields,
            Params { m: h as u32, n: w as u32, k: n as u32, a_off: self.parts_at, c_off: fields, f0: vh, f1: fog_near, f2: fog_far, f3: width / 2.0, x0: x0 as u32, x1: y0 as u32, ..Default::default() },
            [w.div_ceil(16) as u32, h.div_ceil(16) as u32, 1],
        );
        debug_assert_eq!(self.inp.cin as usize, NET_IN);
        ops.conv(self.inp, fields, x0p, full, 0, 1);
        ops.blocks(&self.enc[0], x0p, t0, full);
        ops.conv(self.down[0], x0p, x1p, half, GELU, 2);
        ops.blocks(&self.enc[1], x1p, t1, half);
        ops.conv(self.down[1], x1p, x2p, quarter, GELU, 2);
        ops.blocks(&self.enc[2], x2p, t2, quarter);
        let emb0 = fields + (FIELDS * area(full)) as u32;
        ops.pool(emb0, emb1, half, PHRASE_DIM as u32);
        ops.pool(emb1, emb2, quarter, PHRASE_DIM as u32);
        ops.film(self.film[2], emb2, film2, x2p, quarter);
        ops.upsample(x2p, u1, half, 128);
        ops.conv(self.up[1], u1, d1, half, GELU, 1);
        ops.blocks(&self.dec[1], d1, t1, half);
        ops.film(self.film[1], emb1, film1, d1, half);
        ops.upsample(d1, u0, full, 96);
        ops.conv(self.up[0], u0, d0, full, GELU, 1);
        ops.blocks(&self.dec[0], d0, t0, full);
        ops.film(self.film[0], emb0, film0, d0, full);
        ops.conv(self.out, d0, net_out, full, 0, 1);
        let pixels = area(full) as u32;
        ops.push(Kernel::Finish, Params { m: pixels, a_off: net_out, b_off: fields, c_off: layer, ..Default::default() }, [pixels.div_ceil(64), 1, 1]);
        let data = self.run(&ops, layer, area(full) * LAYER)?;
        Ok(Layer { x0, y0, w, h, data })
    }

    /// The layer for `cam`, in the smallest box (aligned to 4 pixels, so the U-Net's grid stays
    /// put on the screen) round the parts it can see; None when none are in view.
    pub fn render(&mut self, parts: &Parts, cam: &Camera, width: u32) -> Result<Option<Layer>> {
        let (q, palette) = self.view(parts, cam, width);
        let n = q.len() / Q_PARAMS;
        if n == 0 {
            return Ok(None);
        }
        let vh = cam.h as f32;
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in q.chunks_exact(Q_PARAMS) {
            let hy = p[Q_HALF + 1];
            let radius = 0.5 * (p[Q_HALF] + p[Q_HALF + 2]);
            for k in 0..AXIS_POINTS {
                let t = -1.0 + 0.5 * k as f32;
                let s = (t + 1.0) / 2.0;
                let point = [0, 1, 2].map(|c| p[c] + p[6 + c] * hy * t + p[3 + c] * p[Q_BEND] * hy * s * s);
                let z = point[2].max(0.05);
                let (x, y) = (point[0] / z * vh + width as f32 / 2.0, point[1] / z * vh + H as f32 / 2.0);
                let r = CROP_MARGIN * radius * vh / z;
                lo = [lo[0].min(x - r), lo[1].min(y - r)];
                hi = [hi[0].max(x + r), hi[1].max(y + r)];
            }
        }
        let snap = |v: f32, max: u32| (v.clamp(0.0, max as f32) as usize / 4) * 4;
        let (x0, y0) = (snap(lo[0], width), snap(lo[1], H));
        let (x1, y1) = (snap(hi[0] + 3.99, width), snap(hi[1] + 3.99, H));
        if x1 < x0 + 8 || y1 < y0 + 8 {
            return Ok(None);
        }
        let camera = [vh, width as f32, cam.fog_near as f32 / T as f32, cam.fog_far as f32 / T as f32, 0.0, 0.0];
        self.draw(&q, &palette, camera, x0, y0, x1 - x0, y1 - y0).map(Some)
    }

    /// Compare the port with PyTorch on an `export_dream.py --fixture` crop: a line each for the
    /// fields, the phrase vectors, fog and the layer, with mean and largest absolute differences.
    pub fn check(&mut self, fixture: &Path) -> Result<String> {
        let (mut t, _) = read_safetensors(fixture)?;
        let q = take(&mut t, "q")?.1;
        let cam = take(&mut t, "cam")?.1;
        let scheme = take(&mut t, "palette")?.1;
        let (feats, emb, fog) = (take(&mut t, "feats")?, take(&mut t, "emb")?.1, take(&mut t, "fog")?.1);
        let layer = take(&mut t, "layer")?.1;
        let n = feats.0[1];
        let valid = q.chunks_exact(Q_PARAMS).filter(|p| p[Q_VALID] > 0.0).count();
        let palette: Vec<f32> = (0..valid).flat_map(|_| scheme.iter().copied()).collect();
        let camera = [cam[2], cam[3], cam[4], cam[5], 0.0, 0.0];
        let got = self.draw(&q[..valid * Q_PARAMS], &palette, camera, cam[0] as usize, cam[1] as usize, n, n)?;
        let want: Vec<f32> = (0..n * n).flat_map(|i| (0..4).map(|c| layer[c * n * n + i]).collect::<Vec<_>>()).collect();
        let mine: Vec<f32> = got.data.chunks_exact(LAYER).flat_map(|p| p[..4].to_vec()).collect();
        let planes = self.run(&Ops::default(), self.scratch, FIELD_PLANES * n * n)?;
        let line = |name: &str, a: &[f32], b: &[f32]| {
            let (mut sum, mut max) = (0f32, 0f32);
            for (x, y) in a.iter().zip(b) {
                sum += (x - y).abs();
                max = max.max((x - y).abs());
            }
            format!("{name}: mean {:.2e}, largest {:.2e}\n", sum / a.len() as f32, max)
        };
        let mut report = String::new();
        for c in 0..feats.0[0] {
            report += &line(&format!("field {c}"), &planes[c * n * n..][..n * n], &feats.1[c * n * n..][..n * n]);
        }
        report += &line("phrase vectors", &planes[FIELDS * n * n..][..PHRASE_DIM * n * n], &emb);
        report += &line("fog", &planes[PLANE_FOG * n * n..][..n * n], &fog);
        report += &line("layer", &mine, &want);
        Ok(report)
    }

    fn run(&mut self, ops: &Ops, from: u32, floats: usize) -> Result<Vec<f32>> {
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
                pass.set_pipeline(&self.pipelines[*kernel as usize]);
                pass.set_bind_group(0, &self.bind, &[(i as u64 * SLOT) as u32]);
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
            }
        }
        let size = floats as u64 * 4;
        enc.copy_buffer_to_buffer(&self.mem, from as u64 * 4, &self.readback, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        let out = bytemuck::cast_slice(&slice.get_mapped_range().map_err(|e| e.to_string())?).to_vec();
        self.readback.unmap();
        Ok(out)
    }
}

#[derive(Default)]
struct Ops {
    list: Vec<(Kernel, Params, [u32; 3])>,
}

impl Ops {
    fn push(&mut self, kernel: Kernel, p: Params, groups: [u32; 3]) {
        self.list.push((kernel, p, groups));
    }

    /// C = act(conv(A)) with C of `size` (rows, columns); A is `step` times bigger each way.
    fn conv(&mut self, c: Conv, a: u32, to: u32, size: (usize, usize), flags: u32, step: u32) {
        let p = Params {
            m: size.0 as u32,
            n: size.1 as u32,
            k: c.cin,
            a_off: a,
            b_off: c.w,
            c_off: to,
            bias_off: c.b,
            flags,
            x0: c.cout,
            x1: step,
            ..Default::default()
        };
        self.push(Kernel::Conv3x3, p, [size.1.div_ceil(16) as u32, size.0.div_ceil(16) as u32, c.cout.div_ceil(8)]);
    }

    /// Residual blocks on the planes at `x` (train/creatures.py Block), with `ta` for scratch.
    fn blocks(&mut self, blocks: &[ResBlock], x: u32, ta: u32, size: (usize, usize)) {
        for b in blocks {
            self.conv(b.a, x, ta, size, GELU, 1);
            self.conv(b.b, ta, x, size, ACCUMULATE, 1);
        }
    }

    /// Halves `channels` planes at `a` to `size` (rows, columns).
    fn pool(&mut self, a: u32, to: u32, size: (usize, usize), channels: u32) {
        let p = Params { m: size.0 as u32, n: size.1 as u32, a_off: a, c_off: to, ..Default::default() };
        self.push(Kernel::Pool, p, [size.1.div_ceil(16) as u32, size.0.div_ceil(16) as u32, channels]);
    }

    /// DreamNet.modulate: the planes at `x` shaped by the phrase vectors at `emb`, with `to` for scratch.
    fn film(&mut self, c: Conv, emb: u32, to: u32, x: u32, size: (usize, usize)) {
        self.conv(c, emb, to, size, 0, 1);
        let k = c.cout / 2;
        let p = Params { m: size.0 as u32, n: size.1 as u32, k, a_off: x, b_off: to, ..Default::default() };
        self.push(Kernel::Film, p, [size.1.div_ceil(16) as u32, size.0.div_ceil(16) as u32, k]);
    }

    fn upsample(&mut self, a: u32, to: u32, size: (usize, usize), channels: u32) {
        let p = Params { m: size.0 as u32, n: size.1 as u32, a_off: a, c_off: to, ..Default::default() };
        self.push(Kernel::Upsample, p, [size.1.div_ceil(16) as u32, size.0.div_ceil(16) as u32, channels]);
    }
}

/// Each pixel's camera-space depth (in tiles) of the nearest level triangle, [H][width].
pub fn level_depth(scene: &Scene, cam: &Camera, width: u32) -> Vec<f32> {
    const NEAR: f32 = 0.05;
    let (wi, hi) = (width as usize, H as usize);
    let mut depth = vec![f32::MAX; wi * hi];
    let m = view_matrix(cam);
    let r = |i: usize, j: usize| m.m[i][j] as f32 / 4096.0;
    let vh = cam.h as f32;
    let to_cam = |p: [i32; 3]| {
        let d = [0, 1, 2].map(|c| (p[c] - cam.pos[c]) as f32 / T as f32);
        [0, 1, 2].map(|k| r(k, 0) * d[0] + r(k, 1) * d[1] + r(k, 2) * d[2])
    };
    for t in scene.tris.iter().chain(scene.part_tris()) {
        let tri = [to_cam(t.v[0].p), to_cam(t.v[1].p), to_cam(t.v[2].p)];
        // Clipped against the near plane: at most a quad.
        let mut poly: Vec<[f32; 3]> = Vec::with_capacity(4);
        for i in 0..3 {
            let (a, b) = (tri[i], tri[(i + 1) % 3]);
            if a[2] >= NEAR {
                poly.push(a);
            }
            if (a[2] >= NEAR) != (b[2] >= NEAR) {
                let u = (NEAR - a[2]) / (b[2] - a[2]);
                poly.push([0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * u));
            }
        }
        if poly.len() < 3 {
            continue;
        }
        let project = |v: [f32; 3]| [v[0] / v[2] * vh + width as f32 / 2.0, v[1] / v[2] * vh + H as f32 / 2.0, 1.0 / v[2]];
        let a = project(poly[0]);
        for pair in poly[1..].windows(2) {
            let (b, c) = (project(pair[0]), project(pair[1]));
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if area.abs() < 1e-6 {
                continue;
            }
            let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
            let x1 = (a[0].max(b[0]).max(c[0]).ceil().max(0.0) as usize).min(wi);
            let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
            let y1 = (a[1].max(b[1]).max(c[1]).ceil().max(0.0) as usize).min(hi);
            for y in y0..y1 {
                let py = y as f32 + 0.5;
                for x in x0..x1 {
                    let px = x as f32 + 0.5;
                    let w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area;
                    let w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        let z = 1.0 / (w0 * a[2] + w1 * b[2] + w2 * c[2]);
                        let at = y * wi + x;
                        if z < depth[at] {
                            depth[at] = z;
                        }
                    }
                }
            }
        }
    }
    depth
}

fn lin(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c.max(0.0) + 0.055) / 1.055).powf(2.4) }
}

fn enc(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.max(1e-8).powf(1.0 / 2.4) - 0.055 }
}

/// The net's alpha with its faint halo cut away: the CLIP phase left horizontal streaks there
/// (median alpha 0.05), and a firm edge suits the creatures anyway.
fn edge(alpha: f32) -> f32 {
    let t = ((alpha - EDGE.0) / (EDGE.1 - EDGE.0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The layer (fogged to `fog`) over the level network's frame `rgb` [H][width][3], where its
/// parts are nearer than the level (`depth`); in linear light, as train/creatures.py `composite`.
pub fn composite(rgb: &mut [f32], layer: &Layer, depth: &[f32], fog: [f32; 3], width: usize) {
    for y in 0..layer.h {
        for x in 0..layer.w {
            let px = &layer.data[(y * layer.w + x) * LAYER..][..LAYER];
            let at = (y + layer.y0) * width + x + layer.x0;
            let alpha = edge(px[3]);
            if alpha <= 0.0 || px[5] >= depth[at] {
                continue;
            }
            for c in 0..3 {
                let colour = px[c] * (1.0 - px[4]) + fog[c] * px[4];
                let under = rgb[at * 3 + c].clamp(0.0, 1.0);
                rgb[at * 3 + c] = enc(lin(under) * (1.0 - alpha) + lin(colour) * alpha);
            }
        }
    }
}

/// A level's dream creatures, ready to be drawn over the level network's frames.
pub struct DreamLayer {
    net: DreamNet,
    parts: Option<(u64, Parts)>,
    seed: u64,
}

impl DreamLayer {
    pub fn load(path: &Path, seed: u64) -> Result<DreamLayer> {
        Ok(DreamLayer { net: DreamNet::load(path)?, parts: None, seed })
    }

    pub fn adapter(&self) -> &str {
        &self.net.adapter
    }

    /// Lay the creatures of `scene` (`version` says when its parts must be read again) over `rgb`.
    pub fn draw(&mut self, scene: &Scene, version: u64, cam: &Camera, rgb: &mut [f32], width: u32) -> Result<()> {
        if self.parts.as_ref().is_none_or(|(v, _)| *v != version) {
            self.parts = Some((version, self.net.parts(scene, self.seed.wrapping_add(version))));
        }
        let (_, parts) = self.parts.as_mut().expect("just set");
        // They may have moved (`world::animate`).
        parts.rows.clear();
        part_features(scene, &mut parts.rows);
        if let Some(layer) = self.net.render(parts, cam, width)? {
            let depth = level_depth(scene, cam, width);
            composite(rgb, &layer, &depth, scene.fog_colour(cam.pos), width as usize);
        }
        Ok(())
    }
}
