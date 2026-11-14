//! A flesh automaton on the GPU: a texture that keeps growing and moving, trained by
//! `train/nca.py` (which says what a step is; nca.wgsl does the same). Its weights are the
//! trainer's `nca.bin` (raw f32: w1 [HIDDEN, 4 * CHANNELS], b1 [HIDDEN], w2 [CHANNELS, HIDDEN]).
//! No Bevy here: `step` it and `read` its colour and height back.

use std::path::Path;

pub const CHANNELS: usize = 12;
pub const HIDDEN: usize = 96;
/// How often a cell updates in a step.
pub const FIRE: f32 = 0.5;
/// Cells along each side of the grid, which wraps (a texture's size).
pub const SIZE: usize = 256;
const WEIGHTS: usize = HIDDEN * 4 * CHANNELS + HIDDEN + CHANNELS * HIDDEN;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    size: u32,
    step: u32,
    fire: f32,
    _pad: f32,
}

pub struct Nca {
    device: wgpu::Device,
    queue: wgpu::Queue,
    step_pipeline: wgpu::ComputePipeline,
    pack_pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    /// The two states, each read by one step and written by the other.
    binds: [wgpu::BindGroup; 2],
    states: [wgpu::Buffer; 2],
    packed: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Steps taken, and which state is the latest.
    steps: u32,
    fire: f32,
}

impl Nca {
    /// The automaton whose weights are `bin`, from a little noise (`seed`), as the trainer
    /// starts its grids.
    pub fn load(bin: &Path, seed: u64) -> Result<Nca, String> {
        let bytes = std::fs::read(bin).map_err(|e| format!("{}: {e}", bin.display()))?;
        if bytes.len() != WEIGHTS * 4 {
            return Err(format!("{}: {} bytes, not the {} of a {CHANNELS}-channel automaton", bin.display(), bytes.len(), WEIGHTS * 4));
        }
        let weights: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        let mut rng = crate::world::rng::Rng::new(seed);
        let state: Vec<f32> = (0..SIZE * SIZE * CHANNELS).map(|_| (rng.f() - 0.5) * 0.1).collect();
        Self::new(&weights, &state, FIRE)
    }

    /// From weights and a state [SIZE * SIZE * CHANNELS] (cell-major), cells updating with
    /// chance `fire` a step.
    pub fn new(weights: &[f32], state: &[f32], fire: f32) -> Result<Nca, String> {
        use wgpu::util::DeviceExt;
        assert_eq!(weights.len(), WEIGHTS);
        assert_eq!(state.len(), SIZE * SIZE * CHANNELS);
        let (device, queue, _) = super::gpu::device("nca", (SIZE * SIZE * CHANNELS * 4) as u64)?;
        let storage = |label, contents: &[u8], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let weights = storage("nca weights", bytemuck::cast_slice(weights), wgpu::BufferUsages::STORAGE);
        let states = [storage("nca state a", bytemuck::cast_slice(state), rw), storage("nca state b", bytemuck::cast_slice(state), rw)];
        let packed = device.create_buffer(&wgpu::BufferDescriptor { label: Some("nca packed"), size: (SIZE * SIZE * 4) as u64, usage: rw, mapped_at_creation: false });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nca readback"),
            size: (SIZE * SIZE * CHANNELS * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nca params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nca.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nca.wgsl").into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let buffer = |ty| wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nca"),
            entries: &[
                entry(0, buffer(wgpu::BufferBindingType::Uniform)),
                entry(1, buffer(read)),
                entry(2, buffer(read)),
                entry(3, buffer(write)),
                entry(4, buffer(write)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nca"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (step_pipeline, pack_pipeline) = (pipeline("step"), pipeline("pack"));
        let bind = |from: &wgpu::Buffer, to: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nca"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: weights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: from.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: to.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: packed.as_entire_binding() },
                ],
            })
        };
        let binds = [bind(&states[0], &states[1]), bind(&states[1], &states[0])];
        Ok(Nca { device, queue, step_pipeline, pack_pipeline, params, binds, states, packed, readback, steps: 0, fire })
    }

    fn params(&self) -> Params {
        Params { size: SIZE as u32, step: self.steps, fire: self.fire, _pad: 0.0 }
    }

    /// Take `n` steps.
    pub fn step(&mut self, n: usize) {
        for _ in 0..n {
            // One submit a step: the params change between them.
            self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&self.params()));
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.step_pipeline);
                pass.set_bind_group(0, &self.binds[(self.steps % 2) as usize], &[]);
                let groups = SIZE.div_ceil(8) as u32;
                pass.dispatch_workgroups(groups, groups, 1);
            }
            self.queue.submit([enc.finish()]);
            self.steps += 1;
        }
    }

    /// The latest state's buffer.
    fn latest(&self) -> usize {
        (self.steps % 2) as usize
    }

    /// The colour and height, 0..1, as RGBA8 [SIZE * SIZE * 4], rows top first.
    pub fn read(&mut self) -> Result<Vec<u8>, String> {
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&self.params()));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &self.binds[self.latest()], &[]);
            pass.dispatch_workgroups((SIZE * SIZE).div_ceil(64) as u32, 1, 1);
        }
        let size = (SIZE * SIZE * 4) as u64;
        enc.copy_buffer_to_buffer(&self.packed, 0, &self.readback, 0, size);
        self.map(enc, size)
    }

    /// The whole state, cell-major (for checking against the trainer).
    pub fn state(&mut self) -> Result<Vec<f32>, String> {
        let size = (SIZE * SIZE * CHANNELS * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.states[self.latest()], 0, &self.readback, 0, size);
        Ok(bytemuck::cast_slice(&self.map(enc, size)?).to_vec())
    }

    fn map(&self, enc: wgpu::CommandEncoder, size: u64) -> Result<Vec<u8>, String> {
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        let out = slice.get_mapped_range().map_err(|e| e.to_string())?.to_vec();
        self.readback.unmap();
        Ok(out)
    }
}
