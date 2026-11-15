//! Shell-mapped relief (docs/shell_relief.md): a prism over each of the level's triangles, its
//! corners pushed out and in along the mean of the normals of the faces meeting there (so two
//! walls' shells meet at a corner), split into three tetrahedra, whose back faces are drawn at
//! the shown frame's size. Each fragment marches its pixel's line of sight through its
//! tetrahedron, where the place on the triangle and the height coordinate are linear in the
//! point, against the dreamed height on the pages (shell.wgsl), and writes where it first meets
//! it, nearest first: the triangle, the foot's place on its tile, its slope along the page, the
//! view depth and the page; and the level's light, the torch's albedo and the texture at the
//! foot, as the raster gives them.

use crate::level::{normal, Face, Level, LevelTri, View};
use std::collections::HashMap;

/// A tetrahedron's corner: on its triangle, the way that corner is pushed (`Pushed::dir`),
/// which way (-1 in, +1 out), the most it's pushed (`Pushed::cap`), the level triangle, which
/// of its prism's tetrahedra and its welded corner (whose lean, lean.wgsl, it's pushed by too).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Corner {
    pos: [f32; 3],
    dir: [f32; 3],
    side: f32,
    cap: f32,
    tri: u32,
    tet: u32,
    weld: u32,
}

/// vec4s of a triangle in the march's buffer (shell.wgsl TRI): its corners (the normal in their
/// w); the corners' order by welded place, its texture, the open edges the height tapers to and
/// the creases; its corners' places on the texture; their light; their torch albedo; the way
/// each corner is pushed and the most it's pushed; for each crease, the points along it where
/// its height's met and where they are in toward; its corners' welded corners (lean.wgsl).
const TRI: u32 = 19;

/// The uniform's size (shell.wgsl U).
const UNIFORM: u64 = 128;

/// The lean's uniform's size (lean.wgsl L).
const LEAN_UNIFORM: u64 = 96;

/// The most walls a welded corner leans between (lean.wgsl's base): one with more doesn't lean.
const LEAN_WALLS: usize = 4;

/// Each corner's welded corner (as `key`), and each welded corner's place, the unit mean of the
/// ways it's pushed with that mean's length, and the distinct normals of the walls meeting there
/// that it's pushed off (LEAN_WALLS at most, w 1), and the shortest edge from it (its place's w;
/// 0 where more walls meet, which doesn't lean; lean.wgsl's base).
fn welds(tris: &[LevelTri], pushes: &[Pushed]) -> (Vec<[u32; 3]>, Vec<f32>) {
    let corners = corners_of(tris);
    let ns: Vec<[f32; 3]> = corners.iter().map(|c| normal(c).0).collect();
    let mut ids: HashMap<[i64; 3], u32> = HashMap::new();
    let mut sums: Vec<([f32; 3], [f32; 3], f32)> = Vec::new();
    let mut walls: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut shortest: Vec<f32> = Vec::new();
    let mut id = vec![[0u32; 3]; corners.len()];
    for (i, (c, p)) in corners.iter().zip(pushes).enumerate() {
        for k in 0..3 {
            let next = ids.len() as u32;
            let j = *ids.entry(key(c[k])).or_insert(next);
            if j as usize == sums.len() {
                sums.push((c[k], [0.0; 3], 0.0));
                walls.push(Vec::new());
                shortest.push(f32::INFINITY);
            }
            id[i][k] = j;
            for e in [sub(c[(k + 1) % 3], c[k]), sub(c[(k + 2) % 3], c[k])] {
                shortest[j as usize] = shortest[j as usize].min(dot(e, e).sqrt());
            }
            let s = &mut sums[j as usize];
            s.1 = std::array::from_fn(|q| s.1[q] + p.dir[k][q]);
            s.2 += 1.0;
            let n = ns[i];
            let w = &mut walls[j as usize];
            if dot(n, n) > 0.5 && dot(p.dir[k], n) > 0.0 && !w.iter().any(|&m| dot(m, n) > 0.999) {
                w.push(n);
            }
        }
    }
    let mut base = Vec::with_capacity(sums.len() * 4 * (2 + LEAN_WALLS));
    for (((pos, sum, n), w), short) in sums.into_iter().zip(walls).zip(shortest) {
        let len = dot(sum, sum).sqrt();
        let (unit, along) = if len > 1e-6 { (sum.map(|x| x / len), len / n) } else { ([0.0; 3], 0.0) };
        let still = w.len() > LEAN_WALLS;
        base.extend([pos[0], pos[1], pos[2], if still { 0.0 } else { short }, unit[0], unit[1], unit[2], along]);
        for q in 0..LEAN_WALLS {
            match w.get(q) {
                Some(m) if !still => base.extend([m[0], m[1], m[2], 1.0]),
                _ => base.extend([0.0; 4]),
            }
        }
    }
    (id, base)
}

/// Floats of a crease in the creases' list (crease.wgsl): its start and the points along it,
/// its end and where its points start in toward, the triangle's plane, the wall beyond's.
const CREASE: usize = 16;

/// The most points along one crease.
const MOST_POINTS: u32 = 4096;

/// Floats of a page's mips (smooth.wgsl MIP): each level half the last's cells a side, down to
/// two, from `cells`.
fn mip_size(cells: u32) -> u32 {
    (1..).map(|l| cells >> l).take_while(|&n| n >= 2).map(|n| n * n).sum()
}

/// No corner is pushed further than this (`Pushed::cap` where nothing holds it).
const UNHELD: f32 = 1e6;

/// A prism's three tetrahedra: their corners, each a corner of the triangle by its order
/// (`Pushed::order`: 0 the lowest welded id) and in (false) or out (true). Each side quad's
/// diagonal runs from the lower id's inner corner to the higher's outer, so two prisms sharing
/// a side split it alike (shell.wgsl TETS). Each listed so its volume is positive where the
/// triangle's corners are in their own order.
pub const TETS: [[(usize, bool); 4]; 3] = [
    [(0, false), (0, true), (1, true), (2, true)],
    [(0, false), (1, false), (2, true), (1, true)],
    [(0, false), (1, false), (2, false), (2, true)],
];

/// How a triangle's prism is pushed (`pushed`): each corner along `dir` (times the height
/// coordinate, -reach in to +reach out), the mean of the distinct normals of the faces meeting
/// at its welded place, lengthened so the shell stands its full height off each of them (at
/// most twice); `cap`, the most it's pushed (half its shortest crease edge, held further where a
/// tetrahedron would turn inside out); and `order`, its corners by welded id.
#[derive(Clone, Copy, Debug)]
pub struct Pushed {
    pub dir: [[f32; 3]; 3],
    pub cap: [f32; 3],
    pub order: [usize; 3],
}

impl Pushed {
    /// The corner `k` of `c` (the triangle's corners) pushed `side` (-1 in, +1 out) by `reach`
    /// (within its cap).
    pub fn at(&self, c: &[[f32; 3]; 3], k: usize, side: f32, reach: f32) -> [f32; 3] {
        let r = side * reach.min(self.cap[k]);
        std::array::from_fn(|q| c[k][q] + self.dir[k][q] * r)
    }

    /// Tetrahedron `t`'s corners at `reach` (TETS).
    pub fn tet(&self, c: &[[f32; 3]; 3], t: usize, reach: f32) -> [[f32; 3]; 4] {
        TETS[t].map(|(o, out)| self.at(c, self.order[o], if out { 1.0 } else { -1.0 }, reach))
    }

    /// Whether the order is an odd permutation of the triangle's, its tetrahedra then wound the
    /// other way (their volume's sign is the order's).
    fn odd(&self) -> bool {
        let o = self.order;
        (o[0] > o[1]) ^ (o[1] > o[2]) ^ (o[0] > o[2])
    }
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

fn volume(p: &[[f32; 3]; 4]) -> f32 {
    dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0]))) / 6.0
}

fn corners_of(tris: &[LevelTri]) -> Vec<[[f32; 3]; 3]> {
    tris.iter().map(|t| [0, 1, 2].map(|c| [t.pos[3 * c], t.pos[3 * c + 1], t.pos[3 * c + 2]])).collect()
}

/// A corner's welded place (as `edges` keys them).
fn key(p: [f32; 3]) -> [i64; 3] {
    p.map(|v| (v as f64 * 1024.0).round() as i64)
}

/// How far the check that no tetrahedron turns inside out reaches (beyond any height the game
/// gives the shell).
const CHECKED_REACH: f32 = 1.0;

/// How each triangle's prism is pushed (`Pushed`), by the triangles meeting at its corners and
/// what lies beyond its edges (`edges`).
pub fn pushed(tris: &[LevelTri], edges: &[[Edge; 3]]) -> Vec<Pushed> {
    let corners = corners_of(tris);
    let ns: Vec<[f32; 3]> = corners.iter().map(|c| normal(c).0).collect();
    let good = |i: usize| dot(ns[i], ns[i]) > 0.5;
    let mut ids: HashMap<[i64; 3], usize> = HashMap::new();
    let mut normals: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut cap: Vec<f32> = Vec::new();
    let mut id = vec![[0usize; 3]; corners.len()];
    for (i, c) in corners.iter().enumerate() {
        for k in 0..3 {
            let next = ids.len();
            let j = *ids.entry(key(c[k])).or_insert(next);
            if j == normals.len() {
                normals.push(Vec::new());
                cap.push(UNHELD);
            }
            id[i][k] = j;
            if good(i) && !normals[j].iter().any(|&m| dot(m, ns[i]) > 0.999) {
                normals[j].push(ns[i]);
            }
        }
    }
    // Held to half the shortest crease edge at a corner, so shells don't fold through each
    // other where the geometry's thinner than they reach (a step, a pillar's edge, a jamb).
    for (i, c) in corners.iter().enumerate() {
        for k in 0..3 {
            if good(i) && edges[i][k].crease() {
                let e = sub(c[(k + 1) % 3], c[k]);
                let half = dot(e, e).sqrt() / 2.0;
                for j in [id[i][k], id[i][(k + 1) % 3]] {
                    cap[j] = cap[j].min(half);
                }
            }
        }
    }
    // The faces at the corner turned less than 120 degrees from this one (a thin wall's other
    // side has its own), each distinct normal once.
    let dir = |i: usize, k: usize| -> [f32; 3] {
        let n = ns[i];
        let near: Vec<[f32; 3]> = normals[id[i][k]].iter().copied().filter(|&m| dot(m, n) > -0.5).collect();
        let sum = near.iter().fold([0.0f32; 3], |a, m| [a[0] + m[0], a[1] + m[1], a[2] + m[2]]);
        let len = dot(sum, sum).sqrt();
        if len < 1e-3 {
            return n;
        }
        let u = sum.map(|x| x / len);
        let least = near.iter().map(|&m| dot(u, m)).fold(1.0f32, f32::min);
        if dot(u, n) < 0.3 {
            return n;
        }
        u.map(|x| x / least.max(0.5))
    };
    let order = |i: usize| {
        let mut o = [0, 1, 2];
        o.sort_by_key(|&k| id[i][k]);
        o
    };
    let mut out: Vec<Pushed> = (0..corners.len()).map(|i| Pushed { dir: if good(i) { [0, 1, 2].map(|k| dir(i, k)) } else { [ns[i]; 3] }, cap: [0.0; 3], order: order(i) }).collect();
    // Where a tetrahedron would still turn inside out (thin geometry the crease rule misjudges),
    // its corners are held further.
    for _ in 0..12 {
        let mut held = false;
        for (i, p) in out.iter_mut().enumerate() {
            p.cap = id[i].map(|j| cap[j]);
            if !good(i) {
                continue;
            }
            let sign = if p.odd() { -1.0 } else { 1.0 };
            if (0..3).any(|t| sign * volume(&p.tet(&corners[i], t, CHECKED_REACH)) <= 0.0) {
                for j in id[i] {
                    cap[j] = cap[j].min(CHECKED_REACH) * 0.5;
                }
                held = true;
            }
        }
        if !held {
            break;
        }
    }
    out
}

pub(crate) struct Shell {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind: Option<wgpu::BindGroup>,
    /// Each frame before the march, the heights at the creases (crease.wgsl): for each crease
    /// of each triangle, half the difference between the wall beyond's height and its own at
    /// points along it a height cell apart, which the march eases its height by near it, so
    /// both walls meet there.
    creases: wgpu::ComputePipeline,
    /// Each frame before the march, each triangle's home tile (the cell its middle's in) and
    /// that tile's page (crease.wgsl tri_homes), which the march keeps to hand rather than
    /// looking it up in the face table again and again.
    homes: wgpu::ComputePipeline,
    toward_layout: wgpu::BindGroupLayout,
    crease_layout: wgpu::BindGroupLayout,
    /// The bind groups of the march's toward (read) and the creases' pass (toward written, the
    /// creases, the homes), the creases' count, the most points along one and the triangles'
    /// count.
    toward: Option<(wgpu::BindGroup, wgpu::BindGroup, u32, u32, u32)>,
    /// Each frame before the creases, the pages' heights smoothed (smooth.wgsl): each page
    /// averaged down to coarser cells (`mips`, `mip_pages`, one a page), then each face's page
    /// rebuilt at full size from them by a cubic B-spline across its tile's neighbours, at the
    /// smoothing asked (`near`) and two levels smoother (`far`, mixed in with the distance).
    mip_pages: wgpu::ComputePipeline,
    smooth_faces: wgpu::ComputePipeline,
    smooth_layout: wgpu::BindGroupLayout,
    /// The smoothed heights again, HEIGHTS_ROW pages a row (smooth_faces writes them as it
    /// writes `near`), so the march fetches a sample's four cells at once (textureGather).
    heights: wgpu::TextureView,
    /// The march's hits (triangle + 1, two barycentrics, view depth), the nearest kept by the
    /// depth test, and `resolve` (resolve.wgsl) finishing them into `tris` and `looks`.
    marched: wgpu::TextureView,
    marched_bind: wgpu::BindGroup,
    marched_layout: wgpu::BindGroupLayout,
    marched_size: (u32, u32),
    resolve: wgpu::RenderPipeline,
    gather: wgpu::Sampler,
    smooth0_layout: wgpu::BindGroupLayout,
    /// The largest |height| smoothed this frame (f32 bits), which sets the reach while
    /// smoothing (shell.wgsl reach): the march reads the smoothed heights, flatter than the
    /// pages', so the shells are thinner.
    pub smooth_bound: wgpu::Buffer,
    mips: wgpu::Buffer,
    pub near: wgpu::Buffer,
    pub far: wgpu::Buffer,
    /// The smoothing passes' bind groups and the faces' count.
    smoothing: Option<(wgpu::BindGroup, wgpu::BindGroup, u32)>,
    pages: u32,
    uniform: wgpu::Buffer,
    /// What each pixel's line of sight met: (triangle + 1, 0 for none; the height's slope along
    /// the page, two halves; the view depth's bits; the page, or NONE), and (the hit's height
    /// above the plane's bits; the light, albedo (each halved) and texel at the foot, RGBA8 each,
    /// as raster.wgsl's).
    pub tris: wgpu::Texture,
    pub looks: wgpu::Texture,
    depth: wgpu::Texture,
    /// Before the march, the nearest inner shell's depth at each pixel (`caps`, the prisms'
    /// inner faces seen from above, drawn after the tetrahedra in `vertices`): a line of sight
    /// reaching an inner shell has met the relief there or before, so no tetrahedron it comes
    /// into further off can have the nearest hit, and those are let go before marching.
    inner: wgpu::Texture,
    caps: wgpu::RenderPipeline,
    /// The tetrahedra's faces, then the inner caps: the buffer and how many vertices of each.
    vertices: Option<(wgpu::Buffer, u32, u32)>,
    /// What the bind group holds: the triangles, the level buffer, mem, the textures.
    bound: Option<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)>,
    textures: Option<wgpu::TextureView>,
    sampler: wgpu::Sampler,
    /// Each frame before the march, each welded corner's way of looking at it turned a little
    /// towards the eye where it's in sight, and the corner leaned that way (lean.wgsl): its
    /// state (read by the march at 7), bind group and the welded corners' count.
    lean_pipeline: wgpu::ComputePipeline,
    lean_layout: wgpu::BindGroupLayout,
    lean_uniform: wgpu::Buffer,
    lean: Option<(wgpu::Buffer, wgpu::BindGroup, u32)>,
}

/// Pages a row of `Shell::heights` (shell.wgsl HEIGHTS_ROW).
const HEIGHTS_ROW: u32 = 16;

/// The march's four height cells (shell.wgsl raw_height): from the smoothed heights' texture
/// in one fetch when smoothing, the 2 x 2 cells round the corner between them (gathered in the
/// order (0, 1), (1, 1), (1, 0), (0, 0)), exactly the buffer's values.
const FOUR_CELLS: &str = "
@group(1) @binding(5) var heights: texture_2d<f32>;
@group(1) @binding(6) var gather: sampler;
fn four_cells(at: u32, pg: u32, local: vec2<u32>) -> vec4<f32> {
    let ru = u32(u.o.w / HEIGHT_CELL);
    if (u.pos.w <= 0.0 && u.r0.w <= 0.0) { return cells_from(at, ru); }
    let corner = vec2<f32>(vec2<u32>(pg % HEIGHTS_ROW, pg / HEIGHTS_ROW) * ru + local + 1u);
    let g = textureGather(0, heights, gather, corner / vec2<f32>(textureDimensions(heights)));
    return vec4<f32>(g.w, g.z, g.x, g.y);
}
";

impl Shell {
    /// Its targets `max_width` by `height` at most, for `pages` pages of `cells` height cells a
    /// side.
    pub fn new(device: &wgpu::Device, max_width: u32, height: u32, pages: u32, cells: u32) -> Shell {
        let source = format!(
            "{}\n@group(1) @binding(0) var<storage, read> toward: array<f32>;\n@group(1) @binding(1) var<storage, read> near: array<f32>;\n@group(1) @binding(3) var inner: texture_depth_2d;\n@group(1) @binding(4) var<storage, read> homes: array<vec4<i32>>;\n{}",
            include_str!("shell.wgsl"),
            FOUR_CELLS
        ) + include_str!("resolve.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("shell.wgsl"), source: wgpu::ShaderSource::Wgsl(source.into()) });
        let crease_source = format!("{}\n{}", include_str!("shell.wgsl"), include_str!("crease.wgsl"));
        let crease_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("crease.wgsl"), source: wgpu::ShaderSource::Wgsl(crease_source.into()) });
        let read = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        let entry = |binding: u32, visibility: wgpu::ShaderStages, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream shell"),
            entries: &[
                entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(1, wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE, read),
                entry(2, wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE, read),
                entry(3, wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE, read),
                entry(4, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }),
                entry(5, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(6, wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE, read),
                entry(7, wgpu::ShaderStages::VERTEX_FRAGMENT, read),
            ],
        });
        // The smoothing's (it writes the smoothed heights' bound, which the rest read at 6).
        let smooth0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream shell smoothing's view"),
            entries: &[
                entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(2, wgpu::ShaderStages::COMPUTE, read),
                entry(3, wgpu::ShaderStages::COMPUTE, read),
            ],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream shell view"), size: UNIFORM, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let toward_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream shell creases' heights"),
            entries: &[
                entry(0, wgpu::ShaderStages::FRAGMENT, read),
                entry(1, wgpu::ShaderStages::FRAGMENT, read),
                entry(3, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                entry(4, wgpu::ShaderStages::FRAGMENT, read),
                entry(5, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                entry(6, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering)),
            ],
        });
        let write = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let crease_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream shell creases"),
            entries: &[entry(0, wgpu::ShaderStages::COMPUTE, write), entry(1, wgpu::ShaderStages::COMPUTE, read), entry(2, wgpu::ShaderStages::COMPUTE, read), entry(4, wgpu::ShaderStages::COMPUTE, write)],
        });
        let smooth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream shell smoothing"),
            entries: &[entry(0, wgpu::ShaderStages::COMPUTE, write), entry(1, wgpu::ShaderStages::COMPUTE, write), entry(2, wgpu::ShaderStages::COMPUTE, write), entry(3, wgpu::ShaderStages::COMPUTE, read), entry(4, wgpu::ShaderStages::COMPUTE, write),
                entry(5, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::R32Float, view_dimension: wgpu::TextureViewDimension::D2 }),
            ],
        });
        let smooth_source = format!("{}\n{}", include_str!("shell.wgsl"), include_str!("smooth.wgsl"));
        let smooth_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("smooth.wgsl"), source: wgpu::ShaderSource::Wgsl(smooth_source.into()) });
        let smooth_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream shell smoothing"), bind_group_layouts: &[Some(&smooth0_layout), Some(&smooth_layout)], immediate_size: 0 });
        let smoothing = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&smooth_pipeline_layout),
                module: &smooth_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (mip_pages, smooth_faces) = (smoothing("mip_pages"), smoothing("smooth_faces"));
        let floats = |label: &str, n: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: n.max(1) * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let mips = floats("dream shell heights' mips", pages as u64 * mip_size(cells) as u64);
        let near = floats("dream shell heights smoothed", pages as u64 * (cells * cells) as u64);
        let far = floats("dream shell heights smoothed far off", pages as u64 * (cells * cells) as u64);
        let smooth_bound = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dream shell smoothed heights' bound"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let crease_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream shell creases"), bind_group_layouts: &[Some(&layout), Some(&crease_layout)], immediate_size: 0 });
        let creases = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dream shell creases"),
            layout: Some(&crease_pipeline_layout),
            module: &crease_module,
            entry_point: Some("crease_heights"),
            compilation_options: Default::default(),
            cache: None,
        });
        let homes = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dream shell homes"),
            layout: Some(&crease_pipeline_layout),
            module: &crease_module,
            entry_point: Some("tri_homes"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream shell"), bind_group_layouts: &[Some(&layout), Some(&toward_layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dream shell"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Corner>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32, 3 => Float32, 4 => Uint32, 5 => Uint32, 6 => Uint32],
                })],
            },
            // The back faces (the tetrahedra's faces wound to face out, as the level's are):
            // one fragment a tetrahedron a line of sight passes through, the eye inside one or
            // not.
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Front), front_face: wgpu::FrontFace::Ccw, ..Default::default() },
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
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba32Uint, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let marched_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream shell marched"),
            entries: &[entry(0, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false })],
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream shell resolve"), bind_group_layouts: &[Some(&layout), Some(&toward_layout), Some(&marched_layout)], immediate_size: 0 });
        let resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dream shell resolve"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("resolve"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba32Uint, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba32Uint, blend: None, write_mask: wgpu::ColorWrites::ALL }),
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
        let tris = texture("dream shell triangles", wgpu::TextureFormat::Rgba32Uint, copied);
        let looks = texture("dream shell looks", wgpu::TextureFormat::Rgba32Uint, copied);
        let depth = texture("dream shell depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let marched = texture("dream shell marched", wgpu::TextureFormat::Rgba32Uint, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let marched_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dream shell marched"),
            layout: &marched_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&marched) }],
        });
        let inner = texture("dream shell inner depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let caps_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream shell inner caps"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let caps = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dream shell inner caps"),
            layout: Some(&caps_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Corner>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32, 3 => Float32, 4 => Uint32, 5 => Uint32, 6 => Uint32],
                })],
            },
            // Seen from above only (wound as the triangle): reached from below, a line of
            // sight may not have met anything yet.
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), front_face: wgpu::FrontFace::Ccw, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("dream shell textures"), ..Default::default() });
        let heights = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dream shell heights smoothed"),
            size: wgpu::Extent3d { width: HEIGHTS_ROW * cells, height: pages.div_ceil(HEIGHTS_ROW) * cells, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
        let gather = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("dream shell heights"), ..Default::default() });
        let lean_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("lean.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("lean.wgsl").into()) });
        let lean_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream shell lean"),
            entries: &[
                entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(1, wgpu::ShaderStages::COMPUTE, read),
                entry(2, wgpu::ShaderStages::COMPUTE, write),
            ],
        });
        let lean_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dream shell lean"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream shell lean"), bind_group_layouts: &[Some(&lean_layout)], immediate_size: 0 })),
            module: &lean_module,
            entry_point: Some("lean_corners"),
            compilation_options: Default::default(),
            cache: None,
        });
        let lean_uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream shell lean"), size: LEAN_UNIFORM, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Shell { pipeline, layout, inner, caps, homes, bind: None, creases, toward_layout, crease_layout, toward: None, mip_pages, smooth_faces, smooth_layout, smooth0_layout, smooth_bound, mips, near, far, smoothing: None, pages, uniform, tris, looks, depth, vertices: None, bound: None, textures: None, sampler, heights, gather, resolve, marched, marched_bind, marched_layout, marched_size: (max_width, height), lean_pipeline, lean_layout, lean_uniform, lean: None }
    }

    /// The tetrahedra of the level's triangles' prisms (`tris`, as `level` was built from), the
    /// triangles' corners, normals and pushes for the march, and where it reads the faces' pages
    /// (`level_buf`) and their heights (`mem`), `cells` height cells a tile; the level's faces,
    /// whose pages are smoothed.
    #[allow(clippy::too_many_arguments)]
    pub fn set_level(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, tris: &[LevelTri], faces: &[Face], level_buf: &wgpu::Buffer, mem: &wgpu::Buffer, cells: u32) {
        let mut prisms = Vec::with_capacity(tris.len() * 36);
        let mut creases: Vec<f32> = Vec::new();
        let (mut points, mut most) = (0u32, 0u32);
        let mut corners = Vec::with_capacity(tris.len() * 4 * TRI as usize);
        let edges = edges(tris);
        let across = neighbours(tris);
        let pushes = pushed(tris, &edges);
        let (weld, base) = welds(tris, &pushes);
        for (i, t) in tris.iter().enumerate() {
            let p = [0, 1, 2].map(|c| [t.pos[3 * c], t.pos[3 * c + 1], t.pos[3 * c + 2]]);
            let (n, _) = normal(&p);
            let push = &pushes[i];
            for k in 0..3 {
                corners.extend([p[k][0], p[k][1], p[k][2], n[k]]);
            }
            let bits = |f: fn(Edge) -> bool| (0..3).map(|k| (f(edges[i][k]) as u32) << k).sum::<u32>();
            let order = (0..3).map(|o| (push.order[o] as u32) << (2 * o)).sum::<u32>();
            corners.extend([f32::from_bits(order), f32::from_bits(t.texture), f32::from_bits(bits(Edge::tapers)), f32::from_bits(bits(Edge::crease))]);
            corners.extend([t.uv[0], t.uv[1], t.uv[2], t.uv[3], t.uv[4], t.uv[5], 0.0, 0.0]);
            for k in 0..3 {
                corners.extend([t.light[3 * k], t.light[3 * k + 1], t.light[3 * k + 2], 0.0]);
            }
            for k in 0..3 {
                corners.extend([t.albedo[3 * k], t.albedo[3 * k + 1], t.albedo[3 * k + 2], 0.0]);
            }
            for k in 0..3 {
                corners.extend([push.dir[k][0], push.dir[k][1], push.dir[k][2], push.cap[k]]);
            }
            for k in 0..3 {
                let along = match across[i][k] {
                    Some(j) if edges[i][k].crease() && dot(n, n) > 0.5 => {
                        let q = [0, 1, 2].map(|c| [tris[j].pos[3 * c], tris[j].pos[3 * c + 1], tris[j].pos[3 * c + 2]]);
                        let (m, _) = normal(&q);
                        let (a, b) = (p[k], p[(k + 1) % 3]);
                        let e = sub(b, a);
                        let count = ((dot(e, e).sqrt() * cells as f32).ceil() as u32 + 1).clamp(2, MOST_POINTS);
                        creases.extend([a[0], a[1], a[2], f32::from_bits(count), b[0], b[1], b[2], f32::from_bits(points)]);
                        creases.extend([n[0], n[1], n[2], dot(n, p[0]), m[0], m[1], m[2], dot(m, q[0])]);
                        let at = points;
                        points += count;
                        most = most.max(count);
                        [f32::from_bits(count), f32::from_bits(at), 0.0, 0.0]
                    }
                    _ => [0.0; 4],
                };
                corners.extend(along);
            }
            corners.extend([f32::from_bits(weld[i][0]), f32::from_bits(weld[i][1]), f32::from_bits(weld[i][2]), 0.0]);
            if n.iter().map(|x| x * x).sum::<f32>() < 0.5 {
                continue;
            }
            // Each tetrahedron's faces wound to face out of it (as the level's triangles face
            // out of the rock): outward by the right hand where its volume's positive, which
            // it is in the order's sense (`Pushed::odd`).
            for (tet, corner) in TETS.iter().enumerate() {
                let mut v = corner.map(|(o, out)| {
                    let k = push.order[o];
                    Corner { pos: p[k], dir: push.dir[k], side: if out { 1.0 } else { -1.0 }, cap: push.cap[k], tri: i as u32, tet: tet as u32, weld: weld[i][k] }
                });
                if push.odd() {
                    v.swap(1, 2);
                }
                for f in [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] {
                    prisms.extend(f.map(|q| v[q]));
                }
            }
        }
        let tets = prisms.len() as u32;
        for (i, t) in tris.iter().enumerate() {
            let p = [0, 1, 2].map(|c| [t.pos[3 * c], t.pos[3 * c + 1], t.pos[3 * c + 2]]);
            let (n, _) = normal(&p);
            if dot(n, n) < 0.5 {
                continue;
            }
            let push = &pushes[i];
            prisms.extend([0, 1, 2].map(|k| Corner { pos: p[k], dir: push.dir[k], side: -1.0, cap: push.cap[k], tri: i as u32, tet: 0, weld: weld[i][k] }));
        }
        self.vertices = (!prisms.is_empty()).then(|| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dream shell prisms"),
                size: std::mem::size_of_val(prisms.as_slice()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&prisms));
            (buf, tets, prisms.len() as u32 - tets)
        });
        corners.resize(corners.len().max(4 * TRI as usize), 0.0);
        let corners_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dream shell triangles"),
            size: corners.len() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&corners_buf, 0, bytemuck::cast_slice(&corners));
        self.bound = Some((corners_buf, level_buf.clone(), mem.clone()));
        let count = (creases.len() / CREASE) as u32;
        creases.resize(creases.len().max(CREASE), 0.0);
        let list = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream shell creases"), size: creases.len() as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&list, 0, bytemuck::cast_slice(&creases));
        let toward = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream shell creases' heights"), size: points.max(1) as u64 * 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let homes = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream shell homes"), size: tris.len().max(1) as u64 * 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let read = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dream shell creases' heights"),
            layout: &self.toward_layout,
            entries: &[at(0, &toward), at(1, &self.near), wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.inner.create_view(&Default::default())) },
                at(4, &homes),
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.heights) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.gather) },
            ],
        });
        let written = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dream shell creases"), layout: &self.crease_layout, entries: &[at(0, &toward), at(1, &list), at(2, &self.near), at(4, &homes)] });
        self.toward = Some((read, written, count, most, tris.len() as u32));
        let listed: Vec<i32> = faces.iter().flat_map(|f| *f).collect();
        let face_list = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream shell faces"), size: (listed.len().max(4) * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&face_list, 0, bytemuck::cast_slice(&listed));
        let smoothing = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dream shell smoothing"),
            layout: &self.smooth_layout,
            entries: &[
                at(0, &self.mips),
                at(1, &self.near),
                at(2, &self.far),
                at(3, &face_list),
                at(4, &self.smooth_bound),
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.heights) },
            ],
        });
        let view = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dream shell smoothing's view"),
            layout: &self.smooth0_layout,
            entries: &[at(0, &self.uniform), at(2, level_buf), at(3, mem)],
        });
        self.smoothing = Some((view, smoothing, faces.len() as u32));
        // Each welded corner first looked at straight on (along the unit mean, not leaning).
        let stride = 4 * (2 + LEAN_WALLS);
        let welded = base.len() / stride;
        let mut state = vec![0.0f32; (welded * 8).max(8)];
        for w in 0..welded {
            state[8 * w..8 * w + 3].copy_from_slice(&base[stride * w + 4..stride * w + 7]);
        }
        let base_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream shell welded corners"), size: base.len().max(stride) as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&base_buf, 0, bytemuck::cast_slice(&base));
        let state_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream shell lean"), size: state.len() as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&state_buf, 0, bytemuck::cast_slice(&state));
        let lean_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dream shell lean"),
            layout: &self.lean_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.lean_uniform.as_entire_binding() }, at(1, &base_buf), at(2, &state_buf)],
        });
        self.lean = Some((state_buf, lean_bind, welded as u32));
        self.rebind(device);
    }

    /// The level's textures (level.rs Raster's).
    pub fn set_textures(&mut self, device: &wgpu::Device, textures: Option<&wgpu::TextureView>) {
        self.textures = textures.cloned();
        self.rebind(device);
    }

    fn rebind(&mut self, device: &wgpu::Device) {
        let (Some((corners, level, mem)), Some(textures), Some((lean, ..))) = (&self.bound, &self.textures, &self.lean) else { return };
        self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dream shell"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: corners.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: level.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mem.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(textures) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                at(6, &self.smooth_bound),
                at(7, lean),
            ],
        }));
    }

    /// March `view` (at the picture's size; drawn `up` times it, `width` by `height` shown
    /// pixels) through the prisms pushed out and in by `reach` of the largest |height| on any
    /// page (at `bound` in mem) times `gain`, within `limit`, against the pages' heights (at
    /// `page_height` in mem, pages `page_res` texels a side) times `gain`, tapered to nothing
    /// over `taper` towards open edges; smoothed `smooth` levels (each twice the last's size,
    /// 0 none), and more far off by `far` (two levels more at 4 / `far` tiles); the halvings
    /// stopped once the hit's narrowed to `pixel` of a pixel's width (0: all of them); each
    /// corner leaned `lean` of the way (0 none, 1 all) along the way it's been looked at from,
    /// which turns `follow` of the way to the eye a frame where it's in sight (lean.wgsl).
    #[allow(clippy::too_many_arguments)]
    /// The march's hits made `width` by `height`: as big as the frame at most, or (marching
    /// into an outside depth) the same size as it.
    pub fn fit_marched(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.marched_size == (width, height) {
            return;
        }
        self.marched_size = (width, height);
        self.marched = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("dream shell marched"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        self.marched_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dream shell marched"),
            layout: &self.marched_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.marched) }],
        });
    }

    pub fn record(&self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, view: &View, up: u32, width: u32, height: u32, level: &Level, limit: f32, gain: f32, bound: u32, taper: f32, page_height: u32, page_res: u32, smooth: f32, far: f32, pixel: f32, lean: f32, follow: f32, outside: Option<&wgpu::TextureView>, stamp: &mut dyn FnMut(&mut wgpu::CommandEncoder, &str)) {
        let a = view.axes;
        let f = |x: f32| x.to_bits();
        let u: [u32; 32] = [
            f(view.pos[0]), f(view.pos[1]), f(view.pos[2]), f(smooth.max(0.0)),
            f(a[0][0]), f(a[0][1]), f(a[0][2]), f(far.max(0.0)),
            f(a[1][0]), f(a[1][1]), f(a[1][2]), f(pixel.max(0.0)),
            f(a[2][0]), f(a[2][1]), f(a[2][2]), 0,
            f(view.focal * up as f32), f(width as f32 / 2.0), f(height as f32 / 2.0), f(view.fog[1]),
            f(limit), f(gain), bound, f(taper),
            level.table_at, level.table_mask, level.pages_at, page_height,
            level.origin[0] as u32, level.origin[1] as u32, level.origin[2] as u32, page_res,
        ];
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&u));
        if let Some((_, bind, welded)) = &self.lean
            && *welded > 0
        {
            let l: [u32; 24] = [
                f(view.pos[0]), f(view.pos[1]), f(view.pos[2]), f(follow.clamp(0.0, 1.0)),
                f(a[0][0]), f(a[0][1]), f(a[0][2]), f(lean.max(0.0)),
                f(a[1][0]), f(a[1][1]), f(a[1][2]), 0,
                f(a[2][0]), f(a[2][1]), f(a[2][2]), 0,
                f(view.focal * up as f32), f(width as f32 / 2.0), f(height as f32 / 2.0), 0,
                *welded, 0, 0, 0,
            ];
            queue.write_buffer(&self.lean_uniform, 0, bytemuck::cast_slice(&l));
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dream shell lean"), timestamp_writes: None });
            pass.set_bind_group(0, bind, &[]);
            pass.set_pipeline(&self.lean_pipeline);
            pass.dispatch_workgroups(welded.div_ceil(64).min(65535), welded.div_ceil(64 * 65535), 1);
            drop(pass);
            stamp(enc, "shell lean");
        }
        if let Some((view, smoothing, faces)) = &self.smoothing
            && (smooth > 0.0 || far > 0.0)
            && *faces > 0
        {
            enc.clear_buffer(&self.smooth_bound, 0, None);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dream shell smoothing"), timestamp_writes: None });
            pass.set_bind_group(0, view, &[]);
            pass.set_bind_group(1, smoothing, &[]);
            pass.set_pipeline(&self.mip_pages);
            pass.dispatch_workgroups(self.pages.min(65535), self.pages.div_ceil(65535), 1);
            pass.set_pipeline(&self.smooth_faces);
            pass.dispatch_workgroups((*faces).min(65535), faces.div_ceil(65535), 1);
            drop(pass);
            stamp(enc, "shell smoothing");
        }
        if let (Some((_, written, count, most, n_tris)), Some(bind)) = (&self.toward, &self.bind) {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dream shell creases"), timestamp_writes: None });
            pass.set_bind_group(0, bind, &[]);
            pass.set_bind_group(1, written, &[]);
            pass.set_pipeline(&self.homes);
            pass.dispatch_workgroups(n_tris.div_ceil(64).max(1), 1, 1);
            if *count > 0 {
                pass.set_pipeline(&self.creases);
                pass.dispatch_workgroups(most.div_ceil(64), (*count).min(65535), count.div_ceil(65535));
            }
            drop(pass);
            stamp(enc, "shell creases");
        }
        let tris = self.tris.create_view(&Default::default());
        let looks = self.looks.create_view(&Default::default());
        let depth = self.depth.create_view(&Default::default());
        {
            let inner = self.inner.create_view(&Default::default());
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dream shell inner caps"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &inner,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let (Some((buf, tets, caps)), Some(bind)) = (&self.vertices, &self.bind) {
                pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                pass.set_pipeline(&self.caps);
                pass.set_bind_group(0, bind, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(*tets..tets + caps, 0..1);
            }
        }
        stamp(enc, "shell inner caps");
        let target = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dream shell"),
            color_attachments: &[target(&self.marched)],
            // Into an outside depth (a camera's, `fit_marched` to its size) the march meets
            // what's drawn there already and is met by what's drawn after.
            depth_stencil_attachment: Some(match outside {
                Some(view) => wgpu::RenderPassDepthStencilAttachment { view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }), stencil_ops: None },
                None => wgpu::RenderPassDepthStencilAttachment { view: &depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }), stencil_ops: None },
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let (Some((buf, n, _)), Some(bind), Some((toward, ..))) = (&self.vertices, &self.bind, &self.toward) else { return };
        pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.set_bind_group(1, toward, &[]);
        pass.set_vertex_buffer(0, buf.slice(..));
        pass.draw(0..*n, 0..1);
        drop(pass);
        stamp(enc, "shell march");
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dream shell resolve"),
            color_attachments: &[target(&tris), target(&looks)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, bind, &[]);
        pass.set_bind_group(1, toward, &[]);
        pass.set_bind_group(2, &self.marched_bind, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        stamp(enc, "shell resolve");
    }
}

/// What lies beyond an edge of a triangle (`edges`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// A triangle going on from it in the same plane: the wall goes on, its prism the next.
    GoesOn,
    /// One falling away behind it: an outer corner, the other wall's prism past it.
    Outer,
    /// One rising in front of it: an inner corner, rock under the surface past it.
    Inner,
    /// None: rock under the surface past it, and nothing over it.
    Open,
}

impl Edge {
    /// The height tapers to nothing towards it where nothing's past it: else a bulge or hollow
    /// is cut off by the prism's side there.
    pub fn tapers(self) -> bool {
        self == Edge::Open
    }

    /// The wall doesn't go on past it: its corners' pushes are held to half its length
    /// (`pushed`), and both walls' heights meet there.
    pub fn crease(self) -> bool {
        self != Edge::GoesOn
    }
}

/// What lies beyond each edge of each triangle (corner k to k + 1), by the triangles sharing it.
pub fn edges(tris: &[LevelTri]) -> Vec<[Edge; 3]> {
    beyond(tris).into_iter().map(|e| e.map(|(kind, _)| kind)).collect()
}

/// The triangle beyond each edge of each triangle (`edges`' kind of it), none where it's open.
pub fn neighbours(tris: &[LevelTri]) -> Vec<[Option<usize>; 3]> {
    beyond(tris).into_iter().map(|e| e.map(|(_, j)| j)).collect()
}

fn beyond(tris: &[LevelTri]) -> Vec<[(Edge, Option<usize>); 3]> {
    let corners = corners_of(tris);
    let edge = |a: [f32; 3], b: [f32; 3]| {
        let (ka, kb) = (key(a), key(b));
        if ka < kb { (ka, kb) } else { (kb, ka) }
    };
    let mut on: HashMap<([i64; 3], [i64; 3]), Vec<usize>> = HashMap::new();
    for (i, c) in corners.iter().enumerate() {
        for k in 0..3 {
            on.entry(edge(c[k], c[(k + 1) % 3])).or_default().push(i);
        }
    }
    corners
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (n, _) = normal(c);
            std::array::from_fn(|k| {
                let (mut kind, mut who) = (Edge::Open, None);
                for &j in &on[&edge(c[k], c[(k + 1) % 3])] {
                    let o = &corners[j];
                    let (m, _) = normal(o);
                    if j == i || dot(m, m) < 0.5 {
                        continue;
                    }
                    // Its corner off the edge: in the plane, behind it or in front.
                    let off = (0..3).map(|q| dot(n, [o[q][0] - c[0][0], o[q][1] - c[0][1], o[q][2] - c[0][2]])).fold(0.0f32, |a, b| if b.abs() > a.abs() { b } else { a });
                    let this = if dot(m, n) > 0.999 && off.abs() < 1e-3 {
                        Edge::GoesOn
                    } else if off < -1e-3 {
                        Edge::Outer
                    } else {
                        Edge::Inner
                    };
                    // The wall going on wins, then an outer corner.
                    let rank = |e: Edge| match e {
                        Edge::GoesOn => 3,
                        Edge::Outer => 2,
                        Edge::Inner => 1,
                        Edge::Open => 0,
                    };
                    if rank(this) > rank(kind) {
                        (kind, who) = (this, Some(j));
                    }
                }
                (kind, who)
            })
        })
        .collect()
}

/// How far the prisms reach out and in: a hair past the largest |height| on any page (`bound`)
/// times `gain`, within `limit` (shell.wgsl `reach`), and never less than `LEAST_REACH`.
pub(crate) fn reach(bound: f32, gain: f32, limit: f32) -> f32 {
    (bound * gain * 1.01 + 1e-3).clamp(LEAST_REACH, limit.max(LEAST_REACH))
}

/// The thinnest the shells get.
const LEAST_REACH: f32 = 1e-3;

fn at(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}
