//! Neural cellular automata grown on the pages (nca.wgsl, docs/nca_walls.md): MeshNCA-style
//! models (16 channels: colour, height and the automaton's own), a model to each page by its
//! face's orientation, stepped where the pages are in sight and laid on the atlas's pages as
//! their colour (`Step::nca_albedo`, the wall's own colour) and height (for the shell's relief).

use crate::level::Face;
use crate::Result;
use std::collections::HashMap;
use std::path::Path;

/// vec4s of a model's weights (nca.wgsl MODEL).
const MODEL: usize = 128 * 21 + 129 * 4;
/// Steps a frame at most (a slot of parameters each).
pub const MAX_STEPS: u32 = 8;
const SLOT: u64 = 256;
/// i32s a page in the table (nca.wgsl cell_of): its model, axis, then for each edge (-x, +x,
/// -y, +y) the page beyond (-1: none) and the map of cells onto it (x' = m0 x + m1 y + m4,
/// y' = m2 x + m3 y + m5), padded.
const TABLE: usize = 32;

/// A page's axes in the world: (y, z) for x's faces, (x, z) for y's, (x, y) for z's.
fn page_axes(axis: usize) -> (usize, usize) {
    [(1, 2), (0, 2), (0, 1)][axis]
}

/// Where the surface goes on past each edge of face `f` (-x, +x, -y, +y of its page): the page
/// of the face it goes on onto, and how the page's cells (`r` a side, counted on past the edge)
/// land on that page's. A face's cell is the solid one behind it (level.rs), its plane on the
/// cell's far side when it faces +, near side when -. Past an edge (along the page's axis b, way
/// `d`) the surface turns in where the cell ahead and in front is solid (an inner corner: that
/// cell's face looking back), goes on flat where the next cell along has a face as this one, or
/// wraps round the cell's own side (an outer corner). Across a corner the cells are folded about
/// the crease: as far past it as they were, along the other wall.
fn edges(f: Face, r: i32, page_of: &HashMap<Face, u32>) -> [Option<(u32, [i32; 6])>; 4] {
    let axis = (f[0] / 2) as usize;
    let s = if f[0] % 2 == 1 { 1 } else { -1 };
    let cell = [f[1], f[2], f[3]];
    let plane = (cell[axis] + (s > 0) as i32) as f32;
    let (a0, a1) = page_axes(axis);
    let mut out = [None; 4];
    for (e, (b, d)) in [(a0, -1), (a0, 1), (a1, -1), (a1, 1)].into_iter().enumerate() {
        let step = |c: [i32; 3], k: usize, by: i32| {
            let mut c = c;
            c[k] += by;
            c
        };
        let face = |o: usize, pos: bool, c: [i32; 3]| [o as i32 * 2 + pos as i32, c[0], c[1], c[2]];
        let inner = face(b, d < 0, step(step(cell, b, d), axis, s));
        let flat = face(axis, s > 0, step(cell, b, d));
        let outer = face(b, d > 0, cell);
        // How far into the room the surface beyond goes, as it leaves the crease.
        let Some((g, turn)) = [(inner, s as f32), (flat, 0.0), (outer, -s as f32)].into_iter().find(|(g, _)| page_of.contains_key(g)) else { continue };
        let edge = (cell[b] + (d > 0) as i32) as f32;
        let ga = (g[0] / 2) as usize;
        let (g0, g1) = page_axes(ga);
        // A cell (x, y) of this page, counted on past its edges, to its place on g's page.
        let to_g = |x: i32, y: i32| -> [i32; 2] {
            let mut w = [0.0f32; 3];
            w[axis] = plane;
            w[a0] = cell[a0] as f32 + (x as f32 + 0.5) / r as f32;
            w[a1] = cell[a1] as f32 + (y as f32 + 0.5) / r as f32;
            if turn != 0.0 {
                let past = (w[b] - edge) * d as f32;
                w[b] = edge;
                w[axis] = plane + turn * past;
            }
            [((w[g0] - g[1 + g0] as f32) * r as f32 - 0.5).round() as i32, ((w[g1] - g[1 + g1] as f32) * r as f32 - 0.5).round() as i32]
        };
        let (o, px, py) = (to_g(0, 0), to_g(1, 0), to_g(0, 1));
        out[e] = Some((page_of[&g], [px[0] - o[0], py[0] - o[0], px[1] - o[1], py[1] - o[1], o[0], o[1]]));
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NcaParams {
    r: u32,
    page_res: u32,
    pages: u32,
    active: u32,
    seed: u32,
    height_at: u32,
    hr: u32,
    nearest: u32,
    fire: f32,
    depth: f32,
    lift: f32,
    _pad2: f32,
}

/// The models given (by name), from a safetensors file of NAME.w1 (81 x 128: the 80 inputs,
/// then the bias) and NAME.w2 (129 x 16), as train/meshnca_export.py writes MeshNCA's demo's.
pub fn read_models(path: &Path) -> Result<Vec<(String, Vec<f32>, Vec<f32>)>> {
    let t = crate::read_safetensors(path)?;
    let mut names: Vec<&String> = t.keys().filter_map(|k| k.strip_suffix(".w1").map(|_| k)).collect();
    names.sort();
    names
        .into_iter()
        .map(|k| {
            let name = k.trim_end_matches(".w1").to_string();
            let w1 = &t[k];
            let w2 = t.get(&format!("{name}.w2")).ok_or(format!("{name}.w2 missing"))?;
            if w1.0 != [81, 128] || w2.0 != [129, 16] {
                return Err(format!("{name}: {:?} and {:?}, not [81, 128] and [129, 16]", w1.0, w2.0));
            }
            Ok((name, w1.1.clone(), w2.1.clone()))
        })
        .collect()
}

pub struct Nca {
    pipelines: [wgpu::ComputePipeline; 2],
    binds: [wgpu::BindGroup; 2],
    params: wgpu::Buffer,
    states: [wgpu::Buffer; 2],
    table: wgpu::Buffer,
    list: wgpu::Buffer,
    /// Cells a page side, and the buffer the last step wrote.
    pub r: u32,
    flip: usize,
    seed: u32,
    /// The pages to start afresh (given to another face).
    resets: Vec<u32>,
    max_pages: u32,
}

impl Nca {
    #[allow(clippy::too_many_arguments)]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, models: &[(String, Vec<f32>, Vec<f32>)], r: u32, max_pages: u32, atlas: &wgpu::Buffer, mem: &wgpu::Buffer) -> Nca {
        let mut w = vec![0f32; models.len().max(1) * MODEL * 4];
        for (m, (_, w1, w2)) in models.iter().enumerate() {
            let base = m * MODEL * 4;
            for j in 0..128 {
                for i in 0..81 {
                    w[base + j * 84 + i] = w1[i * 128 + j];
                }
            }
            let second = base + 128 * 84;
            w[second..second + 129 * 16].copy_from_slice(w2);
        }
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buffer = |label: &str, size: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false });
        let weights = buffer("nca weights", w.len() as u64 * 4);
        queue.write_buffer(&weights, 0, bytemuck::cast_slice(&w));
        let cells = max_pages as u64 * (r * r) as u64;
        let states = [buffer("nca state a", cells * 32), buffer("nca state b", cells * 32)];
        let table = buffer("nca table", max_pages as u64 * TABLE as u64 * 4);
        queue.write_buffer(&table, 0, bytemuck::cast_slice(&vec![-1i32; max_pages as usize * TABLE]));
        let list = buffer("nca list", max_pages as u64 * 4);
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("nca params"), size: SLOT * (MAX_STEPS as u64 + 1), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("nca.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("nca.wgsl").into()) });
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nca"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<NcaParams>() as u64) }),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, storage(true)),
                entry(4, storage(true)),
                entry(5, storage(true)),
                entry(6, storage(false)),
                entry(7, storage(false)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("nca"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let bind = |from: usize| {
            let bufs = [&states[from], &states[1 - from], &weights, &table, &list, atlas, mem];
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &params, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<NcaParams>() as u64) }),
            }];
            for (i, b) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() });
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("nca"), layout: &layout, entries: &entries })
        };
        Nca {
            pipelines: [pipeline("nca_step"), pipeline("paint")],
            binds: [bind(0), bind(1)],
            params,
            states,
            table,
            list,
            r,
            flip: 0,
            seed: 1,
            resets: Vec::new(),
            max_pages,
        }
    }

    /// Each page's model (by its face: `model_of`, -1 for none), and across each of its edges
    /// the page the surface goes on onto with where its cells land there (`edges`).
    pub fn set_pages(&self, queue: &wgpu::Queue, owner: &[Option<Face>], page_of: &HashMap<Face, u32>, model_of: impl Fn(Face) -> i32) {
        let mut t = vec![-1i32; self.max_pages as usize * TABLE];
        for (p, face) in owner.iter().enumerate() {
            let Some(f) = *face else { continue };
            let row = &mut t[p * TABLE..(p + 1) * TABLE];
            row[0] = model_of(f);
            row[1] = f[0] / 2;
            for (e, edge) in edges(f, self.r as i32, page_of).into_iter().enumerate() {
                if let Some((q, m)) = edge {
                    row[2 + e * 7] = q as i32;
                    row[3 + e * 7..2 + (e + 1) * 7].copy_from_slice(&m);
                }
            }
        }
        queue.write_buffer(&self.table, 0, bytemuck::cast_slice(&t));
    }

    /// Start a page afresh (its cells all 0, MeshNCA's seed) when it's next stepped.
    pub fn reset(&mut self, page: u32) {
        self.resets.push(page);
    }

    pub fn reset_all(&mut self) {
        self.resets.extend(0..self.max_pages);
    }

    /// `steps` steps of the pages `active` (in sight), the rest of `all` kept, then the colour
    /// (each cell's alone, `nearest`, or bilinear) and height of those in sight laid on the atlas (`page_res` texels a side) and in mem's
    /// heights (at `height_at`, `hr` a side).
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, active: &[u32], all: &[u32], steps: u32, fire: f32, depth: f32, lift: f32, nearest: bool, page_res: u32, height_at: u32, hr: u32) {
        let cells = (self.r * self.r) as u64 * 32;
        for p in self.resets.drain(..) {
            for s in &self.states {
                enc.clear_buffer(s, p as u64 * cells, Some(cells));
            }
        }
        let mut list: Vec<u32> = active.to_vec();
        list.extend(all.iter().filter(|p| !active.contains(p)));
        if list.is_empty() {
            return;
        }
        queue.write_buffer(&self.list, 0, bytemuck::cast_slice(&list));
        let steps = steps.min(MAX_STEPS);
        for k in 0..=steps {
            self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
            let p = NcaParams { r: self.r, page_res, pages: list.len() as u32, active: active.len() as u32, seed: self.seed, height_at, hr, nearest: nearest as u32, fire, depth, lift, ..Default::default() };
            queue.write_buffer(&self.params, k as u64 * SLOT, bytemuck::bytes_of(&p));
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nca"), timestamp_writes: None });
        let g = self.r.div_ceil(8);
        for k in 0..steps {
            pass.set_pipeline(&self.pipelines[0]);
            pass.set_bind_group(0, &self.binds[self.flip], &[(k as u64 * SLOT) as u32]);
            pass.dispatch_workgroups(g, g, list.len() as u32);
            self.flip = 1 - self.flip;
        }
        // Painted from the buffer last written (bound as the one read).
        if !active.is_empty() {
            pass.set_pipeline(&self.pipelines[1]);
            pass.set_bind_group(0, &self.binds[self.flip], &[(steps as u64 * SLOT) as u32]);
            pass.dispatch_workgroups(page_res.div_ceil(16), page_res.div_ceil(16), active.len() as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_folds_up_the_wall_and_walls_go_on_flat() {
        let r = 64;
        // A room from x = 0 up: its floor (cells below y = 0, facing +y), a wall at x = 0 (cells
        // at x = -1, facing +x), and the floor's next tile along z.
        let floor = [3, 0, -1, 0];
        let wall = [1, -1, 0, 0];
        let next = [3, 0, -1, 1];
        let page_of: HashMap<Face, u32> = [(floor, 0), (wall, 1), (next, 2)].into_iter().collect();
        let e = edges(floor, r, &page_of);
        // Past the floor's -x edge: the wall's page, the first cell past the edge the wall's
        // lowest, and the floor's z (its y) the wall's z (its y).
        let (p, m) = e[0].unwrap();
        assert_eq!(p, 1);
        let at = |x: i32, y: i32| [m[0] * x + m[1] * y + m[4], m[2] * x + m[3] * y + m[5]];
        assert_eq!(at(-1, 5), [0, 5]);
        assert_eq!(at(-3, 7), [2, 7]);
        // Past +y (z): the next tile, its first row.
        let (p, m) = e[3].unwrap();
        assert_eq!(p, 2);
        assert_eq!([m[0] * 9 + m[1] * r + m[4], m[2] * 9 + m[3] * r + m[5]], [9, 0]);
        // And from the wall, down past its -x (y) edge: the floor, folded back.
        let (p, m) = edges(wall, r, &page_of)[0].unwrap();
        assert_eq!(p, 0);
        assert_eq!([m[0] * -1 + m[1] * 5 + m[4], m[2] * -1 + m[3] * 5 + m[5]], [0, 5]);
        // Nothing past the floor's +x edge.
        assert!(e[1].is_none());
    }
}
