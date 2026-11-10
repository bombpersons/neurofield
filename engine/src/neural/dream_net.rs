//! The dream creature net (train/dream.py) as WGSL compute shaders: each part's per-pixel
//! fields, the U-Net, and its layer laid over the level network's frame where the parts
//! are nearer than the level. Weights and phrase tables come from train/export_dream.py.

use super::features::{part_features, PART_FEATURES, TRI_FEATURES as RAW};
use super::gpu_net::{read_safetensors, Params, MAX_W};
use crate::world::creature::Creature;
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
/// Tiles of a crop the U-Net draws, in pixels of each of its scales: one workgroup.
const TILE: usize = 16;
/// How near a part a tile must be to be drawn: `TILE_MARGIN` radii and `TILE_PAD` pixels.
const TILE_MARGIN: f32 = 1.75;
const TILE_PAD: f32 = 8.0;
/// Passes of the net a frame may take, so that one creature can fade over another.
const MAX_LAYERS: usize = 2;

#[derive(Clone, Copy)]
enum Kernel {
    Conv3x3,
    Fields,
    Upsample,
    Finish,
    Pool,
    Film,
    Clear,
}
const KERNELS: [&str; 7] = ["conv3x3", "dream_fields", "upsample2", "dream_finish", "dream_pool2", "dream_film", "dream_clear"];

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
    tiles_at: u32,
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
        let tiles_at = parts_at + (MAX_PARTS * PART_FLOATS).div_ceil(64) as u32 * 64;
        // Each scale's tiles, drawn or cleared, as column and row.
        let tile_floats = 3 * 2 * (MAX_W as usize).div_ceil(TILE) * (H as usize).div_ceil(TILE);
        let scratch = tiles_at + tile_floats.div_ceil(64) as u32 * 64;
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
            tiles_at,
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
    /// [n, Q_PARAMS], with their colour schemes [n, PALETTE] and which creature each is of.
    fn view(&self, parts: &Parts, cam: &Camera, width: u32) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
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
            if reach > 0.0 && c[2] + reach > 0.05 && px + rp > 0.0 && px - rp < width as f32 && py + rp > 0.0 && py - rp < H as f32 {
                near.push((-reach / z, i, c));
            }
        }
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        near.truncate(MAX_PARTS);
        let (mut q, mut palette, mut creatures) = (Vec::new(), Vec::new(), Vec::new());
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
            creatures.push(row[48] as u32);
        }
        (q, palette, creatures)
    }

    /// The layer of the parts `q` seen from a crop at (x0, y0), `w` x `h` (multiples of 4): each
    /// part is projected here and handed to the shaders with its phrase vector and scheme.
    /// `drawn` limits the net to those of the crop's tiles, row by row (`tile_grid`).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, q: &[f32], palette: &[f32], cam: [f32; 6], x0: usize, y0: usize, w: usize, h: usize, drawn: Option<&[bool]>) -> Result<Layer> {
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
        // Each scale's planes together, those only ever read at their own pixel last: `Clear`
        // does the rest in one go. Each skip's planes come right after the planes upsampled to
        // join them, as `up` reads both.
        let mut at = self.scratch as usize;
        let mut alloc = |planes: usize, size: (usize, usize)| {
            let o = at as u32;
            at += planes * area(size);
            o
        };
        let fields = alloc(FIELD_PLANES, full);
        let u0 = alloc(96 + 48, full);
        let x0p = u0 + (96 * area(full)) as u32;
        let (t0, d0, film0, net_out) = (alloc(48, full), alloc(48, full), alloc(96, full), alloc(6, full));
        let u1 = alloc(128 + 96, half);
        let x1p = u1 + (128 * area(half)) as u32;
        let (t1, d1, emb1, film1) = (alloc(96, half), alloc(96, half), alloc(PHRASE_DIM, half), alloc(192, half));
        let (x2p, t2, emb2, film2) = (alloc(128, quarter), alloc(128, quarter), alloc(PHRASE_DIM, quarter), alloc(256, quarter));
        let layer = alloc(LAYER, full);
        let scales = [(fields, film0, full), (u1, film1, half), (x2p, film2, quarter)];

        let mut ops = Ops { sizes: [full, half, quarter], ..Default::default() };
        if let Some(drawn) = drawn {
            let lists = tile_lists(drawn, full);
            let mut floats = Vec::new();
            for (s, (draw, clear)) in lists.iter().enumerate() {
                let (from, to, size) = scales[s];
                let at = |floats: &Vec<f32>| self.tiles_at + floats.len() as u32;
                ops.tiles[s] = (at(&floats), draw.len() as u32);
                floats.extend(draw.iter().flat_map(|t| [t.0 as f32, t.1 as f32]));
                let list = at(&floats);
                floats.extend(clear.iter().flat_map(|t| [t.0 as f32, t.1 as f32]));
                if !clear.is_empty() {
                    let k = (to - from) / area(size) as u32;
                    ops.push(Kernel::Clear, Params { m: size.0 as u32, n: size.1 as u32, k, c_off: from, x2: list, ..Default::default() }, [clear.len() as u32, 1, 1]);
                }
            }
            self.queue.write_buffer(&self.mem, self.tiles_at as u64 * 4, bytemuck::cast_slice(&floats));
        }
        let (x2, groups) = ops.groups(full, 1);
        ops.push(
            Kernel::Fields,
            Params { m: h as u32, n: w as u32, k: n as u32, a_off: self.parts_at, c_off: fields, f0: vh, f1: fog_near, f2: fog_far, f3: width / 2.0, x0: x0 as u32, x1: y0 as u32, x2, ..Default::default() },
            groups,
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
        let (x2, groups) = ops.groups(full, 1);
        ops.push(Kernel::Finish, Params { m: h as u32, n: w as u32, a_off: net_out, b_off: fields, c_off: layer, x2, ..Default::default() }, groups);
        let mut data = self.run(&ops, layer, area(full) * LAYER)?;
        // Nothing was drawn in the other tiles: what is there is left from earlier frames.
        if let Some(drawn) = drawn {
            for (i, px) in data.chunks_exact_mut(LAYER).enumerate() {
                if !drawn[(i / w / TILE) * w.div_ceil(TILE) + i % w / TILE] {
                    px[3] = 0.0;
                }
            }
        }
        Ok(Layer { x0, y0, w, h, data })
    }

    /// The layers for `cam`, in no particular order: one, or one more for each creature that
    /// has to fade over another (`depth_groups`). Empty when no part is in view.
    pub fn render(&mut self, parts: &Parts, cam: &Camera, width: u32) -> Result<Vec<Layer>> {
        let (q, palette, creatures) = self.view(parts, cam, width);
        let mut layers = Vec::new();
        for group in depth_groups(&q, &creatures, cam.h as f32, width as f32) {
            let rows = |all: &[f32], size: usize| group.iter().flat_map(|&i| all[i * size..][..size].to_vec()).collect::<Vec<f32>>();
            layers.extend(self.crop(&rows(&q, Q_PARAMS), &rows(&palette, PALETTE), cam, width)?);
        }
        Ok(layers)
    }

    /// The layer of the parts `q`, in the smallest box (aligned to 4 pixels, so the U-Net's grid
    /// stays put on the screen) round them; None when that is off the screen.
    fn crop(&mut self, q: &[f32], palette: &[f32], cam: &Camera, width: u32) -> Result<Option<Layer>> {
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
        let drawn = tile_grid(q, vh, width as f32, x0, y0, x1 - x0, y1 - y0);
        self.draw(q, palette, camera, x0, y0, x1 - x0, y1 - y0, Some(&drawn)).map(Some)
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
        let got = self.draw(&q[..valid * Q_PARAMS], &palette, camera, cam[0] as usize, cam[1] as usize, n, n, None)?;
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
    /// (rows, columns) of the net's three scales, and where each's list of tiles to draw is
    /// and how long, with (0, 0) for every tile.
    sizes: [(usize, usize); 3],
    tiles: [(u32, u32); 3],
}

impl Ops {
    fn push(&mut self, kernel: Kernel, p: Params, groups: [u32; 3]) {
        self.list.push((kernel, p, groups));
    }

    /// The tile list (`x2`) for planes of `size`, and the dispatch for `depth` of them.
    fn groups(&self, size: (usize, usize), depth: u32) -> (u32, [u32; 3]) {
        let scale = self.sizes.iter().position(|s| *s == size).expect("one of the net's scales");
        match self.tiles[scale] {
            (0, _) => (0, [size.1.div_ceil(TILE) as u32, size.0.div_ceil(TILE) as u32, depth]),
            (list, n) => (list, [n, 1, depth]),
        }
    }

    /// C = act(conv(A)) with C of `size` (rows, columns); A is `step` times bigger each way.
    fn conv(&mut self, c: Conv, a: u32, to: u32, size: (usize, usize), flags: u32, step: u32) {
        let (x2, groups) = self.groups(size, c.cout.div_ceil(8));
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
            x2,
            ..Default::default()
        };
        self.push(Kernel::Conv3x3, p, groups);
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
        let (x2, groups) = self.groups(size, channels);
        let p = Params { m: size.0 as u32, n: size.1 as u32, a_off: a, c_off: to, x2, ..Default::default() };
        self.push(Kernel::Pool, p, groups);
    }

    /// DreamNet.modulate: the planes at `x` shaped by the phrase vectors at `emb`, with `to` for scratch.
    fn film(&mut self, c: Conv, emb: u32, to: u32, x: u32, size: (usize, usize)) {
        self.conv(c, emb, to, size, 0, 1);
        let k = c.cout / 2;
        let (x2, groups) = self.groups(size, k);
        let p = Params { m: size.0 as u32, n: size.1 as u32, k, a_off: x, b_off: to, x2, ..Default::default() };
        self.push(Kernel::Film, p, groups);
    }

    fn upsample(&mut self, a: u32, to: u32, size: (usize, usize), channels: u32) {
        let (x2, groups) = self.groups(size, channels);
        let p = Params { m: size.0 as u32, n: size.1 as u32, a_off: a, c_off: to, x2, ..Default::default() };
        self.push(Kernel::Upsample, p, groups);
    }
}

/// Which of the `TILE`-sized tiles of a crop are near enough a part in `q` to need drawing,
/// row by row. Further out nothing shows, and the net can treat it as beyond the crop's edge.
fn tile_grid(q: &[f32], vh: f32, width: f32, x0: usize, y0: usize, w: usize, h: usize) -> Vec<bool> {
    let (cols, rows) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let mut drawn = vec![false; cols * rows];
    let reach = TILE as f32 * std::f32::consts::FRAC_1_SQRT_2;
    for p in q.chunks_exact(Q_PARAMS) {
        let hy = p[Q_HALF + 1];
        let radius = 0.5 * (p[Q_HALF] + p[Q_HALF + 2]);
        let axis = |k: usize| {
            let t = -1.0 + 0.5 * k as f32;
            let s = (t + 1.0) / 2.0;
            let point = [0, 1, 2].map(|c| p[c] + p[6 + c] * hy * t + p[3 + c] * p[Q_BEND] * hy * s * s);
            let z = point[2].max(0.05);
            [point[0] / z * vh + width / 2.0 - x0 as f32, point[1] / z * vh + H as f32 / 2.0 - y0 as f32, z]
        };
        for k in 0..AXIS_POINTS - 1 {
            let (a, b) = (axis(k), axis(k + 1));
            let ab = [b[0] - a[0], b[1] - a[1]];
            let far = TILE_MARGIN * radius * vh / a[2].min(b[2]) + TILE_PAD + reach;
            let span = |lo: f32, hi: f32, n: usize| {
                let from = ((lo - far) / TILE as f32).floor().clamp(0.0, n as f32) as usize;
                from..((hi + far) / TILE as f32).ceil().clamp(0.0, n as f32) as usize
            };
            for row in span(a[1].min(b[1]), a[1].max(b[1]), rows) {
                for col in span(a[0].min(b[0]), a[0].max(b[0]), cols) {
                    let c = [(col * TILE) as f32 + TILE as f32 / 2.0 - a[0], (row * TILE) as f32 + TILE as f32 / 2.0 - a[1]];
                    let u = ((c[0] * ab[0] + c[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-6)).clamp(0.0, 1.0);
                    let r = TILE_MARGIN * radius * vh / (a[2] + u * (b[2] - a[2])) + TILE_PAD + reach;
                    if (c[0] - u * ab[0]).powi(2) + (c[1] - u * ab[1]).powi(2) <= r * r {
                        drawn[row * cols + col] = true;
                    }
                }
            }
        }
    }
    drawn
}

/// The rows of `q` to draw together, a group to a pass of the net. Creatures that share no
/// tiles share a pass; one that shares tiles with a nearer one goes in the pass after it, so
/// each can fade over the other. Past `MAX_LAYERS` the furthest share the last pass.
fn depth_groups(q: &[f32], creatures: &[u32], vh: f32, width: f32) -> Vec<Vec<usize>> {
    let mut ids = creatures.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let mut each: Vec<(f32, Vec<usize>, Vec<bool>)> = ids
        .iter()
        .map(|&id| {
            let rows: Vec<usize> = (0..creatures.len()).filter(|&i| creatures[i] == id).collect();
            let own: Vec<f32> = rows.iter().flat_map(|&i| q[i * Q_PARAMS..][..Q_PARAMS].to_vec()).collect();
            let depth = rows.iter().map(|&i| q[i * Q_PARAMS + 2]).sum::<f32>() / rows.len() as f32;
            (depth, rows, tile_grid(&own, vh, width, 0, 0, width as usize, H as usize))
        })
        .collect();
    each.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut pass_of: Vec<usize> = Vec::new();
    for (i, (_, rows, grid)) in each.iter().enumerate() {
        let shares = |j: &usize| each[*j].2.iter().zip(grid).any(|(a, b)| *a && *b);
        let pass = (0..i).filter(shares).map(|j| pass_of[j] + 1).max().unwrap_or(0).min(MAX_LAYERS - 1);
        pass_of.push(pass);
        if groups.len() == pass {
            groups.push(Vec::new());
        }
        groups[pass].extend_from_slice(rows);
    }
    groups
}

/// For each of the net's three scales, the tiles to draw and the tiles to clear for them.
/// A tile is drawn at a scale if any it covers at the scale above is; those to clear are
/// all a drawn tile reads besides: its neighbours, and at the scale above what it covers.
fn tile_lists(drawn: &[bool], full: (usize, usize)) -> [(Vec<(usize, usize)>, Vec<(usize, usize)>); 3] {
    let grid = |scale: usize| ((full.1 >> scale).div_ceil(TILE), (full.0 >> scale).div_ceil(TILE));
    let mut on = [drawn.to_vec(), Vec::new(), Vec::new()];
    for s in 1..3 {
        let ((above_cols, above_rows), (cols, rows)) = (grid(s - 1), grid(s));
        let mut here = vec![false; cols * rows];
        for row in 0..above_rows {
            for col in 0..above_cols {
                if on[s - 1][row * above_cols + col] {
                    here[row / 2 * cols + col / 2] = true;
                }
            }
        }
        on[s] = here;
    }
    std::array::from_fn(|s| {
        let (cols, rows) = grid(s);
        let mut read = on[s].clone();
        if s < 2 {
            let (below_cols, below_rows) = grid(s + 1);
            for row in 0..rows {
                for col in 0..cols {
                    read[row * cols + col] |= row / 2 < below_rows && col / 2 < below_cols && on[s + 1][row / 2 * below_cols + col / 2];
                }
            }
        }
        let (mut draw, mut clear) = (Vec::new(), Vec::new());
        for row in 0..rows {
            for col in 0..cols {
                let touched = (row.saturating_sub(1)..(row + 2).min(rows))
                    .any(|r| (col.saturating_sub(1)..(col + 2).min(cols)).any(|c| read[r * cols + c]));
                if on[s][row * cols + col] {
                    draw.push((col, row));
                } else if touched {
                    clear.push((col, row));
                }
            }
        }
        (draw, clear)
    })
}

/// Each pixel's camera-space depth (in tiles) of the nearest of the level triangles `tris`
/// (`features::tri_features`), [H][width].
pub fn level_depth(tris: &[f32], cam: &Camera, width: u32) -> Vec<f32> {
    const NEAR: f32 = 0.05;
    let (wi, hi) = (width as usize, H as usize);
    let mut depth = vec![f32::MAX; wi * hi];
    let m = view_matrix(cam);
    let r = |i: usize, j: usize| m.m[i][j] as f32 / 4096.0;
    let vh = cam.h as f32;
    let to_cam = |p: &[f32]| {
        let d = [0, 1, 2].map(|c| p[c] - cam.pos[c] as f32 / T as f32);
        [0, 1, 2].map(|k| r(k, 0) * d[0] + r(k, 1) * d[1] + r(k, 2) * d[2])
    };
    for t in tris.chunks_exact(RAW) {
        let tri = [to_cam(&t[0..3]), to_cam(&t[3..6]), to_cam(&t[6..9])];
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

/// The layers (fogged to `fog`) over the level network's frame `rgb` [H][width][3], where their
/// parts are nearer than the level (`depth`), furthest first at each pixel; in linear light,
/// as train/creatures.py `composite`.
pub fn composite(rgb: &mut [f32], layers: &[Layer], depth: &[f32], fog: [f32; 3], width: usize) {
    let (x0, y0) = (layers.iter().map(|l| l.x0).min().unwrap_or(0), layers.iter().map(|l| l.y0).min().unwrap_or(0));
    let (x1, y1) = (layers.iter().map(|l| l.x0 + l.w).max().unwrap_or(0), layers.iter().map(|l| l.y0 + l.h).max().unwrap_or(0));
    let mut here: Vec<&[f32]> = Vec::with_capacity(layers.len());
    for y in y0..y1 {
        for x in x0..x1 {
            let at = y * width + x;
            here.clear();
            for l in layers {
                if (l.x0..l.x0 + l.w).contains(&x) && (l.y0..l.y0 + l.h).contains(&y) {
                    let px = &l.data[((y - l.y0) * l.w + x - l.x0) * LAYER..][..LAYER];
                    if edge(px[3]) > 0.0 && px[5] < depth[at] {
                        here.push(px);
                    }
                }
            }
            here.sort_by(|a, b| b[5].total_cmp(&a[5]));
            for px in &here {
                let alpha = edge(px[3]);
                for c in 0..3 {
                    let colour = px[c] * (1.0 - px[4]) + fog[c] * px[4];
                    let under = rgb[at * 3 + c].clamp(0.0, 1.0);
                    rgb[at * 3 + c] = enc(lin(under) * (1.0 - alpha) + lin(colour) * alpha);
                }
            }
        }
    }
}

/// The layers as `composite` would lay them, laid over nothing instead: `rgba` gets sRGB
/// colour and straight alpha, which laid over any picture in linear light gives what
/// `composite` would have made of it.
pub fn composite_alone(rgba: &mut [f32], layers: &[Layer], depth: &[f32], fog: [f32; 3], width: usize) {
    let (x0, y0) = (layers.iter().map(|l| l.x0).min().unwrap_or(0), layers.iter().map(|l| l.y0).min().unwrap_or(0));
    let (x1, y1) = (layers.iter().map(|l| l.x0 + l.w).max().unwrap_or(0), layers.iter().map(|l| l.y0 + l.h).max().unwrap_or(0));
    let mut here: Vec<&[f32]> = Vec::with_capacity(layers.len());
    for y in y0..y1 {
        for x in x0..x1 {
            let at = y * width + x;
            here.clear();
            for l in layers {
                if (l.x0..l.x0 + l.w).contains(&x) && (l.y0..l.y0 + l.h).contains(&y) {
                    let px = &l.data[((y - l.y0) * l.w + x - l.x0) * LAYER..][..LAYER];
                    if edge(px[3]) > 0.0 && px[5] < depth[at] {
                        here.push(px);
                    }
                }
            }
            here.sort_by(|a, b| b[5].total_cmp(&a[5]));
            // Premultiplied in linear light, furthest first.
            let (mut c, mut a) = ([0.0f32; 3], 0.0f32);
            for px in &here {
                let alpha = edge(px[3]);
                for k in 0..3 {
                    let colour = px[k] * (1.0 - px[4]) + fog[k] * px[4];
                    c[k] = c[k] * (1.0 - alpha) + lin(colour) * alpha;
                }
                a = a * (1.0 - alpha) + alpha;
            }
            if a > 0.0 {
                let out = &mut rgba[at * 4..at * 4 + 4];
                for k in 0..3 {
                    out[k] = enc(c[k] / a);
                }
                out[3] = a;
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

    /// Lay the creatures of `scene`, with `extra` among them (it must always have the same parts;
    /// `version` says when the parts must be read again), over `rgb`, hidden by the level
    /// triangles `tris`. `warp` (in tiles) has already bent `tris`, and bends the creatures too.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, scene: &mut Scene, version: u64, tris: &[f32], extra: Creature, warp: impl Fn([f32; 3]) -> [f32; 3], cam: &Camera, rgb: &mut [f32], width: u32) -> Result<()> {
        let layers = self.layers(scene, version, extra, warp, cam, width)?;
        if !layers.is_empty() {
            let depth = level_depth(tris, cam, width);
            composite(rgb, &layers, &depth, scene.fog_colour(cam.pos), width as usize);
        }
        Ok(())
    }

    /// The creatures as `draw` lays them, but on their own: `rgba` ([H][width][4], sRGB
    /// colour and straight alpha) is cleared and then holds them, to be laid over a picture
    /// drawn some other way (the rasteriser's).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_alone(&mut self, scene: &mut Scene, version: u64, tris: &[f32], extra: Creature, warp: impl Fn([f32; 3]) -> [f32; 3], cam: &Camera, rgba: &mut [f32], width: u32) -> Result<()> {
        rgba.fill(0.0);
        let layers = self.layers(scene, version, extra, warp, cam, width)?;
        if !layers.is_empty() {
            let depth = level_depth(tris, cam, width);
            composite_alone(rgba, &layers, &depth, scene.fog_colour(cam.pos), width as usize);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn layers(&mut self, scene: &mut Scene, version: u64, extra: Creature, warp: impl Fn([f32; 3]) -> [f32; 3], cam: &Camera, width: u32) -> Result<Vec<Layer>> {
        scene.creatures.push(extra);
        if self.parts.as_ref().is_none_or(|(v, _)| *v != version) {
            self.parts = Some((version, self.net.parts(scene, self.seed.wrapping_add(version))));
        }
        let (_, parts) = self.parts.as_mut().expect("just set");
        // They may have moved (`world::animate`).
        parts.rows.clear();
        part_features(scene, &mut parts.rows);
        scene.creatures.pop();
        warp_parts(&mut parts.rows, &warp);
        self.net.render(parts, cam, width)
    }
}

/// The posed parts `rows` carried along by `warp` (tiles): each moves with the space it is in
/// and turns as that space does, keeping its size.
fn warp_parts(rows: &mut [f32], warp: &impl Fn([f32; 3]) -> [f32; 3]) {
    const STEP: f32 = 0.1;
    let unit = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        v.map(|c| c / l)
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for row in rows.chunks_exact_mut(PART_FEATURES) {
        let joint = warp([row[0], row[1], row[2]]);
        let centre = [row[3], row[4], row[5]];
        let moved = warp(centre);
        // The axes are the columns of rows 6..15.
        let axis = |row: &[f32], col: usize| {
            let a = [row[6 + col], row[9 + col], row[12 + col]];
            let end = warp([0, 1, 2].map(|c| centre[c] + a[c] * STEP));
            [0, 1, 2].map(|c| end[c] - moved[c])
        };
        // The length (y) leads, and the others are squared up to it.
        let y = unit(axis(row, 1));
        let x = axis(row, 0);
        let along = dot(x, y);
        let x = unit([0, 1, 2].map(|c| x[c] - y[c] * along));
        let z = crate::world::creature::cross(x, y);
        row[0..3].copy_from_slice(&joint);
        row[3..6].copy_from_slice(&moved);
        for (col, a) in [x, y, z].iter().enumerate() {
            for k in 0..3 {
                row[6 + 3 * k + col] = a[k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creatures_alone_laid_over_a_picture_give_what_composite_does() {
        let (w, h) = (4usize, 3usize);
        // Two overlapping layers, one in front of the other, part see-through; a wall at depth 5.
        let layer = |x0: usize, rgb: [f32; 3], alpha: f32, fog: f32, depth: f32| Layer {
            x0,
            y0: 0,
            w: 3,
            h,
            data: (0..3 * h).flat_map(|_| [rgb[0], rgb[1], rgb[2], alpha, fog, depth]).collect(),
        };
        let layers = [layer(0, [0.9, 0.2, 0.1], 0.6, 0.2, 2.0), layer(1, [0.1, 0.5, 0.8], 0.55, 0.0, 1.0), layer(1, [1.0; 3], 1.0, 0.0, 9.0)];
        let depth = vec![5.0; w * h];
        let fog = [0.1, 0.1, 0.2];
        let picture: Vec<f32> = (0..w * h * 3).map(|i| (i % 7) as f32 / 7.0).collect();
        let mut want = picture.clone();
        composite(&mut want, &layers, &depth, fog, w);
        let mut alone = vec![0.0; w * h * 4];
        composite_alone(&mut alone, &layers, &depth, fog, w);
        for i in 0..w * h {
            let a = alone[i * 4 + 3];
            for c in 0..3 {
                let got = enc(lin(picture[i * 3 + c]) * (1.0 - a) + lin(alone[i * 4 + c]) * a);
                assert!((got - want[i * 3 + c]).abs() < 1e-4, "pixel {i} channel {c}: {got} vs {}", want[i * 3 + c]);
            }
        }
        // Behind the wall: nothing.
        assert!(alone.chunks(4).all(|p| p[3] < 1.0));
    }

    #[test]
    fn a_warped_part_moves_and_stays_square() {
        let mut row = vec![0.0; PART_FEATURES];
        row[3..6].copy_from_slice(&[1.0, 2.0, 3.0]);
        row[6..15].copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        warp_parts(&mut row, &|p| [p[0] + 0.3 * p[1], p[1] + 1.0, p[2]]);
        assert!((row[3] - 1.6).abs() < 1e-4 && (row[4] - 3.0).abs() < 1e-4);
        let col = |c: usize| [row[6 + c], row[9 + c], row[12 + c]];
        // The shear leans the length over, and the others stay at right angles to it.
        assert!(col(1)[0] > 0.2);
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            let d: f32 = (0..3).map(|k| col(a)[k] * col(b)[k]).sum();
            assert!(d.abs() < 1e-4, "{a} {b}: {d}");
        }
    }

    #[test]
    fn a_drawn_tile_has_its_parents_drawn_and_what_they_read_cleared() {
        // A crop of 8 x 4 tiles at full size with one drawn, at column 5, row 1.
        let mut drawn = vec![false; 32];
        drawn[8 + 5] = true;
        let [(full, clear), (half, half_clear), (quarter, _)] = tile_lists(&drawn, (64, 128));
        assert_eq!(full, [(5, 1)]);
        assert_eq!(half, [(2, 0)]);
        assert_eq!(quarter, [(1, 0)]);
        // Round the tile, and everything round the four its parent covers (columns 4 and 5,
        // rows 0 and 1), since the parent's strided convolution reads those.
        for tile in [(4, 2), (6, 0), (6, 2), (3, 0), (3, 2)] {
            assert!(clear.contains(&tile), "{tile:?}");
        }
        assert!(!clear.contains(&(5, 1)));
        assert!(!clear.contains(&(1, 1)));
        assert!(half_clear.contains(&(1, 0)) && half_clear.contains(&(3, 1)));
    }

    /// A short upright part `depth` away, `x` tiles' widths right of the middle of the screen.
    fn part_at(x: f32, depth: f32) -> Vec<f32> {
        let mut q = vec![0.0; Q_PARAMS];
        q[0] = x * 16.0 / 160.0 * depth;
        q[2] = depth;
        q[7] = 1.0;
        q[Q_HALF] = 0.01;
        q[Q_HALF + 1] = 0.05;
        q[Q_HALF + 2] = 0.01;
        q
    }

    #[test]
    fn only_creatures_sharing_tiles_get_their_own_pass() {
        let apart = [part_at(-6.0, 1.0), part_at(6.0, 2.0)].concat();
        assert_eq!(depth_groups(&apart, &[0, 1], 160.0, 320.0), [vec![0, 1]]);
        let over = [part_at(0.0, 2.0), part_at(0.0, 1.0)].concat();
        assert_eq!(depth_groups(&over, &[0, 1], 160.0, 320.0), [vec![1], vec![0]]);
        // One creature's parts always stay together.
        assert_eq!(depth_groups(&over, &[3, 3], 160.0, 320.0), [vec![0, 1]]);
    }

    #[test]
    fn past_the_most_passes_the_furthest_share_the_last() {
        let stack = [part_at(0.0, 3.0), part_at(0.0, 1.0), part_at(0.0, 2.0)].concat();
        assert_eq!(depth_groups(&stack, &[0, 1, 2], 160.0, 320.0), [vec![1], vec![2, 0]]);
    }

    #[test]
    fn layers_blend_furthest_first_whatever_their_order() {
        let px = |grey: f32, alpha: f32, z: f32| Layer { x0: 0, y0: 0, w: 1, h: 1, data: vec![grey, grey, grey, alpha, 0.0, z] };
        let draw = |layers: &[Layer]| {
            let mut rgb = vec![0.0; 3];
            composite(&mut rgb, layers, &[9.0], [0.0; 3], 1);
            rgb[0]
        };
        let (near, far) = (px(1.0, 1.0, 1.0), px(0.5, 1.0, 2.0));
        assert!((draw(&[near, far]) - 1.0).abs() < 1e-3);
        let (near, far) = (px(1.0, 1.0, 1.0), px(0.5, 1.0, 2.0));
        assert!((draw(&[far, near]) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn tiles_are_drawn_near_a_part_only() {
        // One straight part down the middle of a crop 160 x 64, a tile's depth away.
        let mut q = vec![0.0; Q_PARAMS];
        q[2] = 1.0;
        q[7] = 1.0;
        q[Q_HALF] = 0.01;
        q[Q_HALF + 1] = 0.05;
        q[Q_HALF + 2] = 0.01;
        let vh = 160.0;
        let (x0, y0) = (80, 88);
        let drawn = tile_grid(&q, vh, 320.0, x0, y0, 160, 64);
        let row: Vec<bool> = (0..10).map(|c| drawn[10 + c]).collect();
        assert!(row[4] && row[5]);
        assert!(!row[0] && !row[9]);
    }
}
