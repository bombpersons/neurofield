//! How long candidate per-pixel stages would take on this GPU, next to the network as it is:
//! the deferred stage the research report suggests (reports/Neural flesh rendering
//! approaches.md) reads a G-buffer, samples learned latent textures and decodes every pixel.
//! The weights are random; only the time matters. Uses gpu_net.wgsl's own kernels.
//!
//!     cargo run --release --example pixel_stage_bench [-- --model runs/model.safetensors]

use engine::neural::dream::Dream;
use engine::world::rng::Rng;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    m: u32, k: u32, n: u32, batch: u32,
    a_off: u32, a_bs: u32, a_rs: u32, a_cs: u32,
    b_off: u32, b_bs: u32, b_rs: u32, b_cs: u32,
    c_off: u32, c_bs: u32, c_rs: u32, bias_off: u32,
    flags: u32, x0: u32, x1: u32, x2: u32,
    f0: f32, f1: f32, f2: f32, f3: f32,
}

const SLOT: u64 = 256;
const MAX_OPS: u64 = 4096;
const GELU: u32 = 1;
const SHUFFLE: u32 = 4;
const INTERLEAVE: u32 = 8;

/// A latent texture sampled bilinearly (wrapping) at each pixel's UV: the G-buffer's u and v
/// planes at a_off ([m * n] each), the texture [x0 channels][k][k] at b_off, written
/// pixel-major at c_off with x1 floats a pixel.
const SAMPLE: &str = r#"
@compute @workgroup_size(64)
fn sample_latent(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 64u + gid.x;
    let plane = p.m * p.n;
    if (i >= plane) { return; }
    let s = p.k;
    let u = fract(mem[p.a_off + i]) * f32(s) - 0.5;
    let v = fract(mem[p.a_off + plane + i]) * f32(s) - 0.5;
    let x0 = u32(floor(u) + f32(s)) % s;
    let y0 = u32(floor(v) + f32(s)) % s;
    let x1 = (x0 + 1u) % s;
    let y1 = (y0 + 1u) % s;
    let fx = fract(u);
    let fy = fract(v);
    for (var c = 0u; c < p.x0; c++) {
        let t = p.b_off + c * s * s;
        let a = mix(mem[t + y0 * s + x0], mem[t + y0 * s + x1], fx);
        let b = mix(mem[t + y1 * s + x0], mem[t + y1 * s + x1], fx);
        mem[p.c_off + i * p.x1 + c] = mix(a, b, fy);
    }
}
"#;

#[derive(Clone, Copy)]
enum K {
    Matmul,
    Conv,
    Sample,
}

struct Bench {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: [wgpu::ComputePipeline; 3],
    params: wgpu::Buffer,
    bind: wgpu::BindGroup,
    _mem: wgpu::Buffer,
    floats: u32,
}

/// Floats laid out in `mem`.
struct Alloc(u32);

impl Alloc {
    fn take(&mut self, n: usize) -> u32 {
        let at = self.0;
        self.0 += (n as u32).div_ceil(64) * 64;
        at
    }
}

type Op = (K, Params, [u32; 3]);

fn conv(cin: usize, cout: usize, (h, w): (usize, usize), from: u32, to: u32, a: &mut Alloc, flags: u32) -> Op {
    let wts = a.take(cout * cin * 9);
    let b = a.take(cout);
    let p = Params { m: h as u32, k: cin as u32, n: w as u32, a_off: from, b_off: wts, bias_off: b, c_off: to, x0: cout as u32, flags, ..Default::default() };
    (K::Conv, p, [w.div_ceil(16) as u32, h.div_ceil(16) as u32, cout.div_ceil(8) as u32])
}

fn linear(rows: usize, inp: usize, out: usize, from: u32, to: u32, a: &mut Alloc, flags: u32) -> Op {
    let wts = a.take(inp * out);
    let b = a.take(out);
    let p = Params {
        m: rows as u32, k: inp as u32, n: out as u32, batch: 1,
        a_off: from, a_rs: inp as u32, a_cs: 1,
        b_off: wts, b_rs: out as u32, b_cs: 1,
        c_off: to, c_rs: out as u32, bias_off: b, flags,
        ..Default::default()
    };
    (K::Matmul, p, [out.div_ceil(32) as u32, rows.div_ceil(32) as u32, 1])
}

fn sample(pixels: (usize, usize), chans: usize, size: usize, uv: u32, out: u32, stride: usize, a: &mut Alloc) -> Op {
    let tex = a.take(chans * size * size);
    let n = pixels.0 * pixels.1;
    let p = Params { m: pixels.0 as u32, n: pixels.1 as u32, k: size as u32, a_off: uv, b_off: tex, c_off: out, x0: chans as u32, x1: stride as u32, ..Default::default() };
    (K::Sample, p, [n.div_ceil(64).min(65535) as u32, n.div_ceil(64 * 65535) as u32, 1])
}

impl Bench {
    fn new(floats: u32) -> Bench {
        let (device, queue, name) = engine::neural::gpu::device("pixel stage bench", floats as u64 * 4).expect("a GPU");
        eprintln!("on {name}");
        let mem = device.create_buffer(&wgpu::BufferDescriptor { label: Some("mem"), size: floats as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        // Small random numbers everywhere, so nothing is denormal or NaN.
        let mut rng = Rng::new(1);
        let chunk: Vec<f32> = (0..1 << 20).map(|_| rng.fr(-0.05, 0.05)).collect();
        let mut at = 0u64;
        while at < floats as u64 {
            let n = (chunk.len() as u64).min(floats as u64 - at) as usize;
            queue.write_buffer(&mem, at * 4, bytemuck::cast_slice(&chunk[..n]));
            at += n as u64;
        }
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: SLOT * MAX_OPS, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &params, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }) },
                wgpu::BindGroupEntry { binding: 1, resource: mem.as_entire_binding() },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let source = format!("{}\n{SAMPLE}", include_str!("../src/neural/gpu_net.wgsl"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(source.into()) });
        let pipe = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&pl), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
        let pipes = [pipe("matmul"), pipe("conv3x3"), pipe("sample_latent")];
        Bench { device, queue, pipes, params, bind, _mem: mem, floats }
    }

    /// Milliseconds a run of `ops` takes, the median of several batches.
    fn time(&self, ops: &[Op], a: &Alloc) -> f64 {
        assert!(a.0 <= self.floats, "the bench's memory is too small: {} > {}", a.0, self.floats);
        let mut slots = vec![0u8; ops.len() * SLOT as usize];
        for (i, (_, p, _)) in ops.iter().enumerate() {
            slots[i * SLOT as usize..][..std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        self.queue.write_buffer(&self.params, 0, &slots);
        let once = || {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (i, (k, _, g)) in ops.iter().enumerate() {
                    pass.set_pipeline(&self.pipes[*k as usize]);
                    pass.set_bind_group(0, &self.bind, &[(i as u64 * SLOT) as u32]);
                    pass.dispatch_workgroups(g[0], g[1], g[2]);
                }
            }
            self.queue.submit([enc.finish()]);
        };
        let wait = || self.device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU wait");
        for _ in 0..5 {
            once();
        }
        wait();
        let mut batches: Vec<f64> = (0..5)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..20 {
                    once();
                }
                wait();
                t.elapsed().as_secs_f64() * 1000.0 / 20.0
            })
            .collect();
        batches.sort_by(f64::total_cmp);
        batches[2]
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.iter().position(|a| a == "--model").map_or("runs/model.safetensors".to_string(), |i| args[i + 1].clone());
    engine::neural::gpu::adapter().expect("a GPU adapter");

    // The network as it is, on a real view, for comparison.
    if std::path::Path::new(&model).is_file() {
        let mut net = engine::neural::Network::load(std::path::Path::new(&model));
        let level = engine::neural::Level::new(&mut Rng::new(7));
        for width in [320u32, 432] {
            let tokens = level.tokens_at(width);
            let dream = Dream::calm(width);
            for _ in 0..5 {
                net.render(&tokens, &dream);
            }
            let mut ms: Vec<f64> = (0..5)
                .map(|_| {
                    let t = Instant::now();
                    for _ in 0..10 {
                        net.render(&tokens, &dream);
                    }
                    t.elapsed().as_secs_f64() * 100.0
                })
                .collect();
            ms.sort_by(f64::total_cmp);
            println!("the network now, 240 x {width}: {:.2} ms a frame (with its read-back)", ms[2]);
        }
    } else {
        println!("(no model at {model}: skipping the network as it is)");
    }

    let sizes = [(240usize, 320usize), (240, 432), (480, 640)];
    let biggest = sizes.iter().map(|(h, w)| h * w).max().unwrap();
    // Room for the widest stage: inputs, two work planes of 128 channels, RGB, and weights.
    let bench = Bench::new((biggest * (64 + 128 + 128 + 4) + (1 << 22)) as u32);

    for &(h, w) in &sizes {
        let px = h * w;
        println!("\n{h} x {w} ({px} pixels):");
        let report = |name: &str, ms: f64| println!("  {name:58} {ms:6.2} ms");

        // The head as it is: 15 x w/16 patches of 128 channels doubled four times to RGB.
        let (gh, gw) = (h / 16, w / 16);
        let mut a = Alloc(0);
        let (ga, gb) = (a.take(px * 64), a.take(px * 64));
        let mut ops = Vec::new();
        let (mut from, mut to, mut size) = (ga, gb, (gh, gw));
        for (cin, c) in [(128, 64), (64, 48), (48, 32), (32, 16)] {
            ops.push(conv(cin, 4 * c, size, from, to, &mut a, SHUFFLE | GELU));
            (from, to, size) = (to, from, (size.0 * 2, size.1 * 2));
        }
        let out = a.take(px * 3);
        ops.push(conv(16, 3, size, from, out, &mut a, INTERLEAVE));
        report("the head as it is (pixel shuffle, 128 -> 16 channels)", bench.time(&ops, &a));

        // The deferred stage's inputs: two latent textures sampled per pixel (16 channels from a
        // 256^2 flesh latent, 16 from a 64^2 one standing in for the upsampled patch features).
        let mut a = Alloc(0);
        let uv = a.take(px * 2);
        let gin = a.take(px * 48);
        let ops = vec![sample((h, w), 16, 256, uv, gin, 48, &mut a), sample((h, w), 16, 64, uv, gin + 16, 48, &mut a)];
        report("sampling 2 x 16 latent channels per pixel", bench.time(&ops, &a));

        // Decoders over 48 channels a pixel (latents, patch features, G-buffer, organs).
        for (hidden, layers) in [(32, 2), (64, 2), (64, 3), (128, 2)] {
            let mut a = Alloc(0);
            let (x, y, z) = (a.take(px * 48), a.take(px * hidden), a.take(px * hidden));
            let mut ops = vec![linear(px, 48, hidden, x, y, &mut a, GELU)];
            let (mut from, mut to) = (y, z);
            for _ in 1..layers {
                ops.push(linear(px, hidden, hidden, from, to, &mut a, GELU));
                (from, to) = (to, from);
            }
            let out = a.take(px * 3);
            ops.push(linear(px, hidden, 3, from, out, &mut a, 0));
            report(&format!("per-pixel MLP 48 -> {}{} -> 3", hidden, format!(" -> {hidden}").repeat(layers - 1)), bench.time(&ops, &a));
        }
        for (hidden, layers) in [(16, 2), (32, 2), (64, 2)] {
            let mut a = Alloc(0);
            let (x, y, z) = (a.take(px * 48), a.take(px * hidden), a.take(px * hidden));
            let mut ops = vec![conv(48, hidden, (h, w), x, y, &mut a, GELU)];
            let (mut from, mut to) = (y, z);
            for _ in 1..layers {
                ops.push(conv(hidden, hidden, (h, w), from, to, &mut a, GELU));
                (from, to) = (to, from);
            }
            let out = a.take(px * 3);
            ops.push(conv(hidden, 3, (h, w), from, out, &mut a, INTERLEAVE));
            report(&format!("3x3 convs 48 -> {}{} -> 3", hidden, format!(" -> {hidden}").repeat(layers - 1)), bench.time(&ops, &a));
        }
    }
}
