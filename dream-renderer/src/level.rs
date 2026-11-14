//! The level drawn by the renderer itself (docs/dream_world.md, step 2): the caller gives its
//! triangles once (`DreamRenderer::set_level`) and a `View` a frame, the triangles are
//! rasterised on the GPU into which one each pixel shows (`Raster`), and dream.wgsl's `gbuffer`
//! works out each pixel's `Surface` and `Motion` from that, as a caller would on the CPU.
//!
//! A face is a unit square of the plane of a triangle's main axis, facing one way: one page of
//! the atlas. The faces a level has are listed when it's given (`build`), in a hash table the
//! GPU looks them up in, and given pages here as they come into sight: the GPU marks the faces
//! each band of rows shows, read back a frame or two later (`Feedback`), so a face seen for the
//! first time has no page until then.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Bands of rows the faces seen are reported by (for laying the screen on the pages a band at a
/// time).
pub(crate) const BANDS: u32 = 16;
/// Floats of a view in `mem`.
pub(crate) const VIEW: u32 = 16;
/// The face table's empty key.
const EMPTY: u32 = u32::MAX;

/// Where the eye is and how it looks: its position, its axes as rows (right, down, forward: a
/// point's view coordinates are their dot products with the point less `pos`), the focal length
/// in pixels (a point at view (x, y, z) is drawn at width/2 + focal x/z, height/2 + focal y/z,
/// pixels' middles at +0.5), and where the fog starts and is whole (the dream fades between).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub pos: [f32; 3],
    pub axes: [[f32; 3]; 3],
    pub focal: f32,
    pub fog: [f32; 2],
}

impl View {
    /// As `gbuffer` reads it.
    pub(crate) fn floats(&self, width: u32) -> [f32; VIEW as usize] {
        let a = self.axes;
        [self.pos[0], self.pos[1], self.pos[2], a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2], self.focal, self.fog[0], self.fog[1], width as f32]
    }
}

/// A face: its plane's main axis and which way it faces (axis * 2, + 1 facing +), and the unit
/// cell its surface lies in (a hair behind it, so a wall and the floor it stands on differ).
pub type Face = [i32; 4];

/// gbuffer's `face_hash`.
fn face_hash(k0: u32, k1: u32) -> u32 {
    let mut h = k0.wrapping_mul(0x9e37_79b1) ^ k1.wrapping_mul(0x85eb_ca77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^ (h >> 12)
}

/// A level as the GPU has it.
pub(crate) struct Level {
    pub faces: Vec<Face>,
    /// Each triangle's corners, three a triangle.
    pub vertices: Vec<[f32; 3]>,
    /// The level buffer: each triangle's normal and its plane's distance along it (4 floats),
    /// the face table (3 u32s an entry: two of key, the face), then a page a face (a float, -1
    /// for none), written each frame.
    pub data: Vec<u32>,
    pub table_at: u32,
    pub table_mask: u32,
    pub pages_at: u32,
    /// The lowest cell the face table's keys count from.
    pub origin: [i32; 3],
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A triangle's unit normal (as gbuffer and the caller's CPU G-buffer work it out) and its
/// main axis.
fn normal(t: &[[f32; 3]; 3]) -> ([f32; 3], usize) {
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let len = dot(n, n).sqrt().max(1e-12);
    let n = n.map(|k| k / len);
    let axis = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs())).unwrap();
    (n, axis)
}

/// The level's faces: every cell a triangle's plane passes through within its box (a hair
/// behind it), facing its way. More than it shows, at times, but never fewer.
pub(crate) fn build(tris: &[[f32; 9]]) -> Level {
    let tris: Vec<[[f32; 3]; 3]> = tris.iter().map(|t| [[t[0], t[1], t[2]], [t[3], t[4], t[5]], [t[6], t[7], t[8]]]).collect();
    let mut faces: Vec<Face> = Vec::new();
    let mut index: HashMap<Face, u32> = HashMap::new();
    let mut planes = Vec::with_capacity(tris.len() * 4);
    for t in &tris {
        let (n, axis) = normal(t);
        planes.extend([n[0], n[1], n[2], dot(t[0], n)].map(f32::to_bits));
        if dot(n, n) < 0.5 {
            continue;
        }
        let orient = axis as i32 * 2 + (n[axis] > 0.0) as i32;
        let q = t.map(|c| [0, 1, 2].map(|k| c[k] - n[k] * 0.01));
        let lo = [0, 1, 2].map(|k| q[0][k].min(q[1][k]).min(q[2][k]).floor() as i32);
        let hi = [0, 1, 2].map(|k| q[0][k].max(q[1][k]).max(q[2][k]).floor() as i32);
        let d = dot(q[0], n);
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    // Does the plane pass through the cell? (Its corners either side, or on it.)
                    let (mut below, mut above) = (false, false);
                    for c in 0..8 {
                        let p = [x + (c & 1), y + (c >> 1 & 1), z + (c >> 2 & 1)].map(|v| v as f32);
                        let s = dot(p, n) - d;
                        below |= s <= 1e-5;
                        above |= s >= -1e-5;
                    }
                    let face = [orient, x, y, z];
                    if below && above && !index.contains_key(&face) {
                        index.insert(face, faces.len() as u32);
                        faces.push(face);
                    }
                }
            }
        }
    }
    let origin = [1, 2, 3].map(|k| faces.iter().map(|f| f[k]).min().unwrap_or(0));
    let cap = (faces.len() * 2).next_power_of_two().max(64);
    let mut table = vec![EMPTY; cap * 3];
    for (i, f) in faces.iter().enumerate() {
        let (k0, k1) = key(f, origin);
        let mut h = face_hash(k0, k1) as usize & (cap - 1);
        while table[h * 3] != EMPTY {
            h = (h + 1) & (cap - 1);
        }
        table[h * 3..h * 3 + 3].copy_from_slice(&[k0, k1, i as u32]);
    }
    let table_at = planes.len() as u32;
    let pages_at = table_at + table.len() as u32;
    let mut data = planes;
    data.extend(table);
    data.extend(std::iter::repeat_n((-1.0f32).to_bits(), faces.len().max(1)));
    Level { faces, vertices: tris.into_iter().flatten().collect(), data, table_at, table_mask: cap as u32 - 1, pages_at, origin }
}

/// gbuffer's `face_key`: the orientation and the cell counted from `origin`.
fn key(f: &Face, origin: [i32; 3]) -> (u32, u32) {
    let r = [1, 2, 3].map(|k| (f[k] - origin[k - 1]) as u32);
    (f[0] as u32 | r[0] << 3, r[1] << 16 | r[2])
}

/// The level's triangles drawn as the triangle each pixel shows (its index + 1, 0 for none),
/// nearest first, those facing away left out, as far as the fog's far end.
pub(crate) struct Raster {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    pub ids: wgpu::Texture,
    depth: wgpu::Texture,
    vertices: Option<(wgpu::Buffer, u32)>,
}

impl Raster {
    pub fn new(device: &wgpu::Device, max_width: u32, height: u32) -> Raster {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("raster.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("raster.wgsl").into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream raster"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream raster view"), size: 80, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dream raster"), layout: &layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }] });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream raster"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dream raster"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3] })],
            },
            // Facing the eye: anticlockwise on the screen with y up.
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), front_face: wgpu::FrontFace::Ccw, ..Default::default() },
            // Depth: NEAR / z, nearer greater.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::R32Uint, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let texture = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: max_width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let ids = texture("dream raster ids", wgpu::TextureFormat::R32Uint, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let depth = texture("dream raster depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
        Raster { pipeline, bind, uniform, ids, depth, vertices: None }
    }

    pub fn set_level(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, vertices: &[[f32; 3]]) {
        self.vertices = (!vertices.is_empty()).then(|| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dream level vertices"),
                size: (vertices.len() * 12) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(vertices));
            (buf, vertices.len() as u32)
        });
    }

    /// Draw `view` on `width` by `height` pixels into `ids`.
    pub fn record(&self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, view: &View, width: u32, height: u32) {
        let a = view.axes;
        let u: [f32; 20] = [
            view.pos[0], view.pos[1], view.pos[2], 0.0, a[0][0], a[0][1], a[0][2], 0.0, a[1][0], a[1][1], a[1][2], 0.0, a[2][0], a[2][1], a[2][2], 0.0, view.focal,
            width as f32 / 2.0, height as f32 / 2.0, view.fog[1],
        ];
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&u));
        let ids = self.ids.create_view(&Default::default());
        let depth = self.depth.create_view(&Default::default());
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dream level"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ids,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let Some((buf, n)) = &self.vertices else { return };
        pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_vertex_buffer(0, buf.slice(..));
        pass.draw(0..*n, 0..1);
    }
}

enum Slot {
    Free,
    /// Copied into by a frame not yet known to be submitted.
    Copied,
    /// Being mapped: true once it is.
    Mapping(Arc<AtomicBool>),
}

/// The faces seen, read back without waiting: a few buffers taking turns, each copied into at
/// the end of a frame, mapped once a later frame starts (the first one's work submitted by then)
/// and read when that's done.
pub(crate) struct Feedback {
    bufs: Vec<(wgpu::Buffer, Slot, u64)>,
    /// Bumped with each new level, whose faces the old reports don't count.
    pub level: u64,
}

impl Feedback {
    pub fn new(device: &wgpu::Device, bytes: u64, level: u64) -> Feedback {
        let bufs = (0..4)
            .map(|_| {
                let b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream faces seen"), size: bytes.max(4), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                (b, Slot::Free, level)
            })
            .collect();
        Feedback { bufs, level }
    }

    /// The latest report come back (BANDS bitsets of faces), if any; reports copied into last
    /// frame start mapping.
    pub fn poll(&mut self, device: &wgpu::Device) -> Option<Vec<u32>> {
        let _ = device.poll(wgpu::PollType::Poll);
        let mut latest = None;
        for (buf, slot, level) in &mut self.bufs {
            match slot {
                Slot::Mapping(done) if done.load(Ordering::Acquire) => {
                    if *level == self.level
                        && let Ok(range) = buf.slice(..).get_mapped_range()
                    {
                        latest = Some(bytemuck::cast_slice::<u8, u32>(&range).to_vec());
                    }
                    buf.unmap();
                    *slot = Slot::Free;
                }
                Slot::Copied => {
                    let done = Arc::new(AtomicBool::new(false));
                    let flag = done.clone();
                    buf.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), Ordering::Release));
                    *slot = Slot::Mapping(done);
                }
                _ => {}
            }
        }
        latest
    }

    /// Copy this frame's report (`seen`) into a free buffer, if there is one.
    pub fn copy(&mut self, enc: &mut wgpu::CommandEncoder, seen: &wgpu::Buffer) {
        let level = self.level;
        if let Some((buf, slot, l)) = self.bufs.iter_mut().find(|(_, s, _)| matches!(s, Slot::Free)) {
            enc.copy_buffer_to_buffer(seen, 0, buf, 0, buf.size().min(seen.size()));
            *slot = Slot::Copied;
            *l = level;
        }
    }
}
