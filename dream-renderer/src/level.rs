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
pub(crate) const VIEW: u32 = 28;
/// The face table's empty key.
const EMPTY: u32 = u32::MAX;

/// Where the eye is and how it looks: its position, its axes as rows (right, down, forward: a
/// point's view coordinates are their dot products with the point less `pos`), the focal length
/// in pixels (a point at view (x, y, z) is drawn at width/2 + focal x/z, height/2 + focal y/z,
/// pixels' middles at +0.5), and where the fog starts and is whole (the dream fades between);
/// the torch carried at the eye (its colour, and how far it reaches, in pages), and the fog's
/// colour (in the textures' encoding).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub pos: [f32; 3],
    pub axes: [[f32; 3]; 3],
    pub focal: f32,
    pub fog: [f32; 2],
    pub torch: [f32; 4],
    pub fog_colour: [f32; 3],
    /// Where the torch is held, along the view's axes (right, down, forward) from the eye, in
    /// pages.
    pub torch_at: [f32; 3],
}

impl View {
    /// As `gbuffer` reads it.
    pub(crate) fn floats(&self, width: u32) -> [f32; VIEW as usize] {
        let (a, t, f, h) = (self.axes, self.torch, self.fog_colour, self.torch_at);
        [
            self.pos[0], self.pos[1], self.pos[2], a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2], self.focal, self.fog[0], self.fog[1], width as f32, t[0], t[1], t[2], t[3], f[0], f[1],
            f[2], h[0], h[1], h[2], 0.0, 0.0,
        ]
    }
}

/// A triangle of the level: its corners (in pages, a face being a unit square of its plane),
/// and at each corner, where on its texture it is (0..1 across the texture, which is clamped),
/// the light it has from the level (a multiplier of the texture, in its own encoding: 1 shows
/// the texture as it is), and how much of the torch's light it takes (likewise, times the
/// torch's colour, its fall-off and its angle); and which of the textures (`set_textures`) it
/// shows.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LevelTri {
    pub pos: [f32; 9],
    pub uv: [f32; 6],
    pub light: [f32; 9],
    pub albedo: [f32; 9],
    pub texture: u32,
}

/// A corner of a small triangle of the subdivided level: its vertex (in `Level::verts`), the
/// level triangle it's of and its texture, and the level triangle's place on the texture,
/// light and torch albedo there.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Corner {
    vertex: u32,
    tri: u32,
    layer: u32,
    uv: [f32; 2],
    light: [f32; 3],
    albedo: [f32; 3],
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
    /// The triangles subdivided (`subdivide`), three corners a small triangle.
    pub mesh: Vec<Corner>,
    pub verts: u32,
    /// The level's own triangles, flat (drawn for the dream when nothing's pushed out): their
    /// vertices' places after the subdivided mesh's, at `moved_at`, never moved.
    pub flat: Vec<Corner>,
    pub flat_verts: u32,
    /// The level buffer: each triangle's normal and its plane's distance along it (4 floats),
    /// the face table (3 u32s an entry: two of key, the face), a page a face (a float, -1 for
    /// none, written each frame); each vertex (VERT: its place, its normal, where its faces'
    /// list starts and how long it is), the lists (5 u32s a face: it, its orientation and cell;
    /// dream.wgsl LISTED); and room for each vertex pushed out by the
    /// height (MOVED floats: there, then where it was), at a whole 64 u32s.
    pub data: Vec<u32>,
    pub table_at: u32,
    pub table_mask: u32,
    pub pages_at: u32,
    pub verts_at: u32,
    pub moved_at: u32,
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
pub(crate) fn normal(t: &[[f32; 3]; 3]) -> ([f32; 3], usize) {
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let len = dot(n, n).sqrt().max(1e-12);
    let n = n.map(|k| k / len);
    let axis = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs())).unwrap();
    (n, axis)
}

/// u32s of a vertex's record, and floats of where it's pushed to.
pub(crate) const VERT: u32 = 8;
pub(crate) const MOVED: u32 = 8;
/// The most faces a vertex's height is the mean of.
const MOST_FACES: usize = 8;

/// The level's faces: every cell a triangle's plane passes through within its box (a hair
/// behind it), facing its way. More than it shows, at times, but never fewer. And its
/// triangles subdivided, no small triangle's edge longer than `spacing`, so they can bulge.
pub(crate) fn build(given: &[LevelTri], spacing: f32) -> Level {
    let tris: Vec<[[f32; 3]; 3]> = given.iter().map(|t| [0, 1, 2].map(|c| [t.pos[3 * c], t.pos[3 * c + 1], t.pos[3 * c + 2]])).collect();
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
    let (mesh, verts) = subdivide(&tris, given, spacing, &index);
    let verts_at = data.len() as u32;
    let lists_at = verts_at + verts.len() as u32 * VERT;
    let mut lists = Vec::new();
    for v in &verts {
        data.extend(v.pos.map(f32::to_bits));
        data.extend(v.normal.map(f32::to_bits));
        data.extend([lists_at + lists.len() as u32, v.faces.len() as u32]);
        for &f in &v.faces {
            let face = faces[f as usize];
            lists.extend([f, face[0] as u32, face[1] as u32, face[2] as u32, face[3] as u32]);
        }
    }
    data.extend(lists);
    data.resize(data.len().div_ceil(64) * 64, 0);
    let moved_at = data.len() as u32;
    let (mut flat, flat_verts) = subdivide(&tris, given, f32::INFINITY, &HashMap::new());
    for c in &mut flat {
        c.vertex += verts.len() as u32;
    }
    for v in verts.iter().chain(&flat_verts) {
        data.extend([v.pos[0], v.pos[1], v.pos[2], 0.0, v.pos[0], v.pos[1], v.pos[2], 0.0].map(f32::to_bits));
    }
    Level { faces, mesh, verts: verts.len() as u32, flat, flat_verts: flat_verts.len() as u32, data, table_at, table_mask: cap as u32 - 1, pages_at, verts_at, moved_at, origin }
}

/// A vertex of the subdivided mesh: where it is, the faces it's on (their indices), and the
/// mean of the different normals of the triangles it's on (pushed out along that, so a corner
/// bulges round).
struct Vert {
    pos: [f32; 3],
    normal: [f32; 3],
    faces: Vec<u32>,
    normals: Vec<[f32; 3]>,
}

/// Each triangle cut into n x n small ones, n so no edge is longer than `spacing`; each of its
/// edges' points put on that edge's own division (by the edge's length alone), so the
/// triangles either side of an edge share its points and nothing cracks when they're pushed
/// out (a few small triangles come out flat, which is no matter). Points in the same place are
/// one vertex.
fn subdivide(tris: &[[[f32; 3]; 3]], given: &[LevelTri], spacing: f32, index: &HashMap<Face, u32>) -> (Vec<Corner>, Vec<Vert>) {
    let mut verts: Vec<Vert> = Vec::new();
    let mut at: HashMap<[i64; 3], u32> = HashMap::new();
    let mut mesh = Vec::new();
    let cuts = |a: [f32; 3], b: [f32; 3]| (dot(sub(b, a), sub(b, a)).sqrt() / spacing).ceil().max(1.0) as u32;
    for (ti, t) in tris.iter().enumerate() {
        let (n, axis) = normal(t);
        if dot(n, n) < 0.5 {
            continue;
        }
        let orient = axis as i32 * 2 + (n[axis] > 0.0) as i32;
        let [a, b, c] = *t;
        let (nab, nbc, nca) = (cuts(a, b), cuts(b, c), cuts(c, a));
        let k = nab.max(nbc).max(nca);
        let snap = |x: f32, cuts: u32| (x * cuts as f32).round() / cuts as f32;
        // A point, and its weights on the triangle's b and c.
        let mut point = |i: u32, j: u32| -> (u32, [f32; 2]) {
            let (fi, fj) = (i as f32 / k as f32, j as f32 / k as f32);
            let w = if j == 0 {
                [snap(fi, nab), 0.0]
            } else if i == 0 {
                [0.0, 1.0 - snap(1.0 - fj, nca)]
            } else if i + j == k {
                let t = snap(fj, nbc);
                [1.0 - t, t]
            } else {
                [fi, fj]
            };
            let p = [0, 1, 2].map(|q| a[q] + (b[q] - a[q]) * w[0] + (c[q] - a[q]) * w[1]);
            let key = p.map(|v| (v as f64 * 4096.0).round() as i64);
            let id = *at.entry(key).or_insert_with(|| {
                verts.push(Vert { pos: p, normal: [0.0; 3], faces: Vec::new(), normals: Vec::new() });
                verts.len() as u32 - 1
            });
            let v = &mut verts[id as usize];
            if !v.normals.iter().any(|m| dot(*m, n) > 0.999) {
                v.normals.push(n);
            }
            // The faces it's on: its cell a hair behind it, and the cell either side along the
            // plane where it's on a cell's edge.
            let q = [0, 1, 2].map(|d| v.pos[d] - n[d] * 0.01);
            let near = |d: usize| -> Vec<i32> {
                if d == axis {
                    return vec![q[d].floor() as i32];
                }
                let (lo, hi) = ((q[d] - 1e-3).floor() as i32, (q[d] + 1e-3).floor() as i32);
                if lo == hi { vec![lo] } else { vec![lo, hi] }
            };
            for x in near(0) {
                for y in near(1) {
                    for z in near(2) {
                        if let Some(&f) = index.get(&[orient, x, y, z])
                            && !v.faces.contains(&f)
                            && v.faces.len() < MOST_FACES
                        {
                            v.faces.push(f);
                        }
                    }
                }
            }
            (id, w)
        };
        let mut ids = HashMap::new();
        for j in 0..=k {
            for i in 0..=k - j {
                ids.insert((i, j), point(i, j));
            }
        }
        for j in 0..k {
            for i in 0..k - j {
                let p = |i, j| ids[&(i, j)];
                let mut small = vec![[p(i, j), p(i + 1, j), p(i, j + 1)]];
                if i + j + 1 < k {
                    small.push([p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)]);
                }
                for tri in small {
                    if tri[0].0 != tri[1].0 && tri[1].0 != tri[2].0 && tri[0].0 != tri[2].0 {
                        let src = &given[ti];
                        mesh.extend(tri.map(|(v, w)| {
                            let at = |x: &[f32], d: usize| (0..d).map(|q| x[q] * (1.0 - w[0] - w[1]) + x[d + q] * w[0] + x[2 * d + q] * w[1]).collect::<Vec<f32>>();
                            let (uv, light, albedo) = (at(&src.uv, 2), at(&src.light, 3), at(&src.albedo, 3));
                            Corner { vertex: v, tri: ti as u32, layer: src.texture, uv: [uv[0], uv[1]], light: [light[0], light[1], light[2]], albedo: [albedo[0], albedo[1], albedo[2]] }
                        }));
                    }
                }
            }
        }
    }
    for v in &mut verts {
        let m = v.normals.iter().fold([0.0; 3], |s, n| [s[0] + n[0], s[1] + n[1], s[2] + n[2]]);
        let len = dot(m, m).sqrt().max(1e-6);
        v.normal = m.map(|x| x / len);
    }
    (mesh, verts)
}

/// gbuffer's `face_key`: the orientation and the cell counted from `origin`.
fn key(f: &Face, origin: [i32; 3]) -> (u32, u32) {
    let r = [1, 2, 3].map(|k| (f[k] - origin[k - 1]) as u32);
    (f[0] as u32 | r[0] << 3, r[1] << 16 | r[2])
}

/// The level's triangles, subdivided and pushed out by the height, drawn as the triangle each
/// pixel shows (its index + 1, 0 for none) with its texture times the level's light and times
/// the torch's albedo (raster.wgsl), and, at that pixel, the point of the level's own
/// (flat) surface the pushed-out one came from and the view depth of where it now is; nearest
/// first, those facing away left out, as far as the fog's far end.
pub(crate) struct Raster {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind: Option<wgpu::BindGroup>,
    uniform: wgpu::Buffer,
    pub ids: wgpu::Texture,
    pub points: wgpu::Texture,
    depth: wgpu::Texture,
    vertices: Option<(wgpu::Buffer, u32)>,
    /// The level's own triangles, unsubdivided (drawn when nothing's pushed out).
    flat: Option<(wgpu::Buffer, u32)>,
    pub textures: Option<wgpu::TextureView>,
    sampler: wgpu::Sampler,
    /// The level buffer, where its vertices' pushed-out places are, and how many.
    moved: Option<(wgpu::Buffer, u32, u32)>,
}

impl Raster {
    pub fn new(device: &wgpu::Device, max_width: u32, height: u32) -> Raster {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("raster.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("raster.wgsl").into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream raster"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream raster view"), size: 80, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream raster"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dream raster"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Corner>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Uint32x3, 1 => Float32x2, 2 => Float32x3, 3 => Float32x3],
                })],
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
                targets: &[
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba32Uint, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba32Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
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
        let copied = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let ids = texture("dream raster ids", wgpu::TextureFormat::Rgba32Uint, copied);
        let points = texture("dream raster points", wgpu::TextureFormat::Rgba32Float, copied);
        let depth = texture("dream raster depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("dream textures"), ..Default::default() });
        let mut raster = Raster { pipeline, layout, bind: None, uniform, ids, points, depth, vertices: None, flat: None, textures: None, sampler, moved: None };
        raster.set_textures(device, None, 1, &[vec![255; 4]]);
        raster
    }

    /// The level's textures, `size` texels a side each, RGBA8 in their own encoding.
    pub fn set_textures(&mut self, device: &wgpu::Device, queue: Option<&wgpu::Queue>, size: u32, layers: &[Vec<u8>]) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dream level textures"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: layers.len().max(1) as u32 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        if let Some(queue) = queue {
            for (k, rgba) in layers.iter().enumerate() {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: k as u32 }, aspect: wgpu::TextureAspect::All },
                    rgba,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * 4), rows_per_image: Some(size) },
                    wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                );
            }
        }
        self.textures = Some(texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() }));
        self.rebind(device);
    }

    fn rebind(&mut self, device: &wgpu::Device) {
        let (Some((level, moved_at, verts)), Some(textures)) = (&self.moved, &self.textures) else { return };
        self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dream raster"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: level, offset: *moved_at as u64 * 4, size: wgpu::BufferSize::new((*verts * MOVED) as u64 * 4) }),
                },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(textures) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        }));
    }

    /// The subdivided mesh and the flat one, their vertices' places read from `level` (from
    /// `moved_at` u32s, `verts` of them in all).
    pub fn set_level(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &[Corner], flat: &[Corner], level: &wgpu::Buffer, moved_at: u32, verts: u32) {
        let upload = |mesh: &[Corner]| {
            (!mesh.is_empty()).then(|| {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("dream level mesh"),
                    size: std::mem::size_of_val(mesh) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(mesh));
                (buf, mesh.len() as u32)
            })
        };
        self.vertices = upload(mesh);
        self.flat = upload(flat);
        self.moved = (verts > 0).then(|| (level.clone(), moved_at, verts));
        self.rebind(device);
    }

    /// Draw `view` on `width` by `height` pixels into `ids` and `points`: the subdivided mesh
    /// as pushed out (`displaced`), or the level's own triangles, flat.
    pub fn record(&self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, view: &View, width: u32, height: u32, displaced: bool) {
        let a = view.axes;
        let u: [f32; 20] = [
            view.pos[0], view.pos[1], view.pos[2], 0.0, a[0][0], a[0][1], a[0][2], 0.0, a[1][0], a[1][1], a[1][2], 0.0, a[2][0], a[2][1], a[2][2], 0.0, view.focal,
            width as f32 / 2.0, height as f32 / 2.0, view.fog[1],
        ];
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&u));
        let ids = self.ids.create_view(&Default::default());
        let points = self.points.create_view(&Default::default());
        let depth = self.depth.create_view(&Default::default());
        let target = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dream level"),
            color_attachments: &[target(&ids), target(&points)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let mesh = if displaced { &self.vertices } else { &self.flat };
        let (Some((buf, n)), Some(bind)) = (mesh, &self.bind) else { return };
        pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
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
