//! The shaders against PyTorch: activations and the input gradient of a frame
//! (train/export_googlenet.py --fixture), then how long a frame of dreaming takes.
//!
//!     cargo run --release -p dream-renderer --example check -- WEIGHTS FIXTURE
//!
//! Then the G-buffer the renderer makes from a level drawn on the GPU against one made on the
//! CPU, as game/src/dream_atlas.rs `surfaces` makes it.

use dream_renderer::{Config, DreamRenderer, Face, Motion, Step, Surface, View};
use std::collections::HashMap;
use std::path::Path;

fn device(bytes: u64) -> (wgpu::Device, wgpu::Queue, String) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("a GPU");
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.max(bytes);
    limits.max_buffer_size = limits.max_buffer_size.max(bytes);
    let (d, q) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() })).expect("a device");
    (d, q, adapter.get_info().name)
}

fn fixture(path: &Path) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).expect("fixture");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("safetensors");
    st.tensors()
        .into_iter()
        .map(|(n, v)| (n, (v.shape().to_vec(), v.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())))
        .collect()
}

fn compare(what: &str, mine: &[f32], want: &[f32]) {
    assert_eq!(mine.len(), want.len(), "{what}: sizes");
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-12);
    let worst = mine.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let dot: f64 = mine.iter().zip(want).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
    let na: f64 = mine.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    println!("{what:>28}: largest difference {:.2e} of {scale:.2e} ({:.4}%), cosine {:.6}", worst, 100.0 * worst / scale, dot / (na * nb).max(1e-30));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (weights, fix) = (Path::new(&args[1]), Path::new(&args[2]));
    let mut cfg = Config::new(weights);
    cfg.max_width = 320;
    let (device, queue, name) = device(DreamRenderer::max_binding(&cfg));
    println!("on {name}");
    let mut dr = DreamRenderer::new(device, queue, cfg.clone()).expect("renderer");
    let f = fixture(fix);
    let (shape, rgb) = &f["rgb"];
    let (h, w) = (shape[0] as u32, shape[1] as u32);

    let wanted = ["conv1", "maxpool1", "conv3", "maxpool2", "inception3a", "inception4c"];
    let (g, acts) = dr.input_gradient(rgb, h, w, &["inception4c"], &wanted).expect("gradient");
    for (name, a) in wanted.iter().zip(&acts) {
        compare(&format!("activation {name}"), a, &f[&format!("act_{name}")].1);
    }
    compare("gradient 4c", &g, &f["grad_inception4c"].1);
    let (g, _) = dr.input_gradient(rgb, h, w, &["inception3b", "inception4d"], &[]).expect("gradient");
    compare("gradient 3b + 4d", &g, &f["grad_inception3b+inception4d"].1);

    // A frame of dreaming on the fixture's picture, every pixel on one of a grid of pages.
    let picture: Vec<u8> = rgb.chunks_exact(3).flat_map(|p| [(p[0] * 255.0) as u8, (p[1] * 255.0) as u8, (p[2] * 255.0) as u8, 255]).collect();
    let surfaces: Vec<Surface> = (0..h * w)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let (px, py) = ((x / 64.0).floor(), (y / 64.0).floor());
            Surface { page: py * 5.0 + px, uv: [x / 64.0 - px, y / 64.0 - py], duv_dx: [1.0 / 64.0, 0.0], duv_dy: [0.0, 1.0 / 64.0], strength: 1.0, slope: [0.3, -0.2] }
        })
        .collect();
    let step = Step::default();
    let mut times = Vec::new();
    for i in 0..40 {
        dr.frame(&picture, w, &surfaces, &step).expect("frame");
        if i >= 5 {
            times.push(dr.last_ms);
        }
    }
    times.sort_by(f32::total_cmp);
    println!("a frame of dreaming (both octaves, shown at x{}): median {:.1} ms, best {:.1} ms", cfg.up, times[times.len() / 2], times[0]);
    let no_fine = Step { lr_fine: 0.0, ..step };
    let mut times = Vec::new();
    for _ in 0..20 {
        dr.frame(&picture, w, &surfaces, &no_fine).expect("frame");
        times.push(dr.last_ms);
    }
    times.sort_by(f32::total_cmp);
    println!("coarse octave only: median {:.1} ms", times[times.len() / 2]);
    for (split, every) in [(3, 1), (4, 1), (2, 2), (4, 2)] {
        let s = Step { fine_split: split, fine_every: every, ..step };
        let mut times = Vec::new();
        for _ in 0..24 {
            dr.frame(&picture, w, &surfaces, &s).expect("frame");
            times.push(dr.last_ms);
        }
        let mean = times.iter().sum::<f32>() / times.len() as f32;
        println!("fine octave a 1/{split} side piece every {every} frame(s): mean {mean:.1} ms a frame");
    }
    let mut times = Vec::new();
    for i in 0..40 {
        dr.frame_still(&picture, w, &step).expect("still frame");
        if i >= 5 {
            times.push(dr.last_ms);
        }
    }
    times.sort_by(f32::total_cmp);
    println!("a still frame, dreaming on the screen: median {:.1} ms", times[times.len() / 2]);
    let quiet = Step { lr: 0.0, lr_fine: 0.0, ..step };
    let before = dr.frame_still(&picture, w, &quiet).expect("still frame").to_vec();
    let after = dr.frame(&picture, w, &surfaces, &quiet).expect("frame").to_vec();
    println!("the frame after, laying the screen's dream on the pages: {:.1} ms", dr.last_ms);
    let diff = |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(x, y)| (*x as f32 - *y as f32).abs()).sum::<f32>() / a.len() as f32;
    println!("  shown before and after it differ by {:.2} of 255 a channel", diff(&before, &after));
    // Carried along by no motion at all, the screen's dream should show as it was.
    for _ in 0..10 {
        dr.frame_still(&picture, w, &step).expect("still frame");
    }
    let before = dr.frame_still(&picture, w, &quiet).expect("still frame").to_vec();
    let motion: Vec<Motion> = (0..h * w).map(|i| Motion { from: [(i % w) as f32 + 0.5, (i / w) as f32 + 0.5], dfrom_dx: [1.0, 0.0], dfrom_dy: [0.0, 1.0] }).collect();
    let after = dr.frame_reprojected(&picture, w, &surfaces, &motion, &quiet).expect("reprojected").to_vec();
    println!("starting to carry it along (via the pages), the shown frame differs by {:.3} of 255 a channel", diff(&before, &after));
    for _ in 0..8 {
        dr.frame_reprojected(&picture, w, &surfaces, &motion, &step).expect("reprojected");
    }
    let before = dr.frame_reprojected(&picture, w, &surfaces, &motion, &quiet).expect("reprojected").to_vec();
    let after = dr.frame_reprojected(&picture, w, &surfaces, &motion, &quiet).expect("reprojected").to_vec();
    println!("carried along by no motion, the shown frame differs by {:.3} of 255 a channel", diff(&before, &after));
    let after = dr.frame(&picture, w, &surfaces, &quiet).expect("frame").to_vec();
    println!("back on the pages, the shown frame differs by {:.3} of 255 a channel", diff(&before, &after));
    dr.frame_reprojected(&picture, w, &surfaces, &motion, &quiet).expect("reprojected");
    let mut times = Vec::new();
    for _ in 0..20 {
        dr.frame_reprojected(&picture, w, &surfaces, &motion, &step).expect("reprojected");
        times.push(dr.last_ms);
    }
    times.sort_by(f32::total_cmp);
    println!("a reprojected frame: median {:.1} ms", times[times.len() / 2]);
    // The same frame recorded on the GPU, picture from a texture and the frame into one.
    let cpu = dr.frame_reprojected(&picture, w, &surfaces, &motion, &quiet).expect("reprojected").to_vec();
    let (sw, sh) = (w * cfg.up, h * cfg.up);
    let texture = |w: u32, h: u32, usage: wgpu::TextureUsages| {
        dr.device().create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage,
            view_formats: &[],
        })
    };
    let input = texture(w, h, wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST);
    let output = texture(sw + 16, sh + 8, wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST);
    let rows = (w * 4).div_ceil(256) * 256;
    let mut padded = vec![0u8; (rows * h) as usize];
    for y in 0..h as usize {
        padded[y * rows as usize..][..w as usize * 4].copy_from_slice(&picture[y * w as usize * 4..][..w as usize * 4]);
    }
    let (device, queue) = (dr.device().clone(), dr.queue().clone());
    queue.write_texture(input.as_image_copy(), &padded, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(rows), rows_per_image: None }, wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
    let mut enc = device.create_command_encoder(&Default::default());
    dr.encode(&mut enc, &input, w, Some(&surfaces), Some(&motion), &quiet, &output).expect("encode");
    let out_rows = (sw * 4).div_ceil(256) * 256;
    let back = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (out_rows * sh) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &back, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(out_rows), rows_per_image: None } },
        wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    back.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let gpu: Vec<u8> = {
        let m = back.slice(..).get_mapped_range().expect("mapped");
        (0..sh as usize).flat_map(|y| m[y * out_rows as usize..][..sw as usize * 4].to_vec()).collect()
    };
    println!("recorded on the GPU (texture in, texture out), the frame differs from the CPU path's by {:.3} of 255 a channel", diff(&cpu, &gpu));
    let relief = Step { relief: 0.1, ..step };
    let mut times = Vec::new();
    for _ in 0..20 {
        dr.frame(&picture, w, &surfaces, &relief).expect("frame");
        times.push(dr.last_ms);
    }
    times.sort_by(f32::total_cmp);
    println!("both octaves with relief: median {:.1} ms", times[times.len() / 2]);
    let mut times = Vec::new();
    for _ in 0..20 {
        dr.frame(&picture, w, &surfaces, &Step { lr: 0.0, lr_fine: 0.0, ..step }).expect("frame");
        times.push(dr.last_ms);
    }
    times.sort_by(f32::total_cmp);
    println!("showing only (no dreaming): median {:.1} ms", times[times.len() / 2]);
    for (h, w) in [(240, 320), (360, 480)] {
        println!("profile at {w} x {h} (each dispatch waited on alone, so a little slower):");
        for (k, ms, n) in dr.profile(h, w, &["inception4c"]).expect("profile") {
            println!("  {k:>16}: {ms:6.2} ms over {n}");
        }
    }
    let shown = dr.frame(&picture, w, &surfaces, &Step { lr: 0.0, lr_fine: 0.0, ..step }).expect("frame").to_vec();
    let (sw, sh) = (w * cfg.up, h * cfg.up);
    let out = args.get(3).map(String::as_str).unwrap_or("dream_check.ppm");
    let mut ppm = format!("P6 {sw} {sh} 255\n").into_bytes();
    ppm.extend(shown.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
    std::fs::write(out, ppm).expect("write");
    println!("wrote {out}");
    gbuffer_check(&mut dr, &input, &output, w, h);
}

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    a.map(|k| k / l)
}

/// A quad as two triangles facing along `n`.
fn quad(tris: &mut Vec<[f32; 9]>, c: [V3; 4], n: V3) {
    let [a, b, cc, d] = c;
    for t in [[a, b, cc], [a, cc, d]] {
        let t = if dot(cross(sub(t[1], t[0]), sub(t[2], t[0])), n) > 0.0 { t } else { [t[0], t[2], t[1]] };
        tris.push([t[0][0], t[0][1], t[0][2], t[1][0], t[1][1], t[1][2], t[2][0], t[2][1], t[2][2]]);
    }
}

/// An axis-aligned box from `lo` to `hi`, its sides facing out (`out`) or in.
fn boxed(tris: &mut Vec<[f32; 9]>, lo: V3, hi: V3, out: bool) {
    let s = if out { 1.0 } else { -1.0 };
    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        for side in [false, true] {
            let mut n = [0.0; 3];
            n[axis] = if side { s } else { -s };
            let corner = |a: f32, b: f32| {
                let mut p = [0.0; 3];
                p[axis] = if side { hi[axis] } else { lo[axis] };
                p[u] = a;
                p[v] = b;
                p
            };
            quad(tris, [corner(lo[u], lo[v]), corner(hi[u], lo[v]), corner(hi[u], hi[v]), corner(lo[u], hi[v])], n);
        }
    }
}

fn look(pos: V3, yaw: f32, pitch: f32, focal: f32) -> View {
    let f = [yaw.sin() * pitch.cos(), -pitch.sin(), yaw.cos() * pitch.cos()];
    let r = unit(cross([0.0, 1.0, 0.0], f));
    let d = cross(f, r);
    View { pos, axes: [r, d, f], focal, fog: [3.0, 12.0] }
}

/// game/src/dream_atlas.rs `surfaces`, the nearest triangle found by casting each pixel's ray:
/// each pixel's face (None for none), surface (its page left out), 1/z, and where it was in
/// `last` (with `last`'s 1/z).
fn cpu_gbuffer(tris: &[[f32; 9]], v: &View, w: u32, h: u32, last: Option<(&View, &[f32])>) -> (Vec<Option<Face>>, Vec<Surface>, Vec<f32>, Vec<Motion>) {
    let n_px = (w * h) as usize;
    let (mut faces, mut surfaces, mut nearest, mut motion) = (vec![None; n_px], vec![Surface::NONE; n_px], vec![0.0; n_px], vec![Motion::NONE; n_px]);
    let r = v.axes;
    let ray = |x: f32, y: f32| {
        let d = [(x - w as f32 / 2.0) / v.focal, (y - h as f32 / 2.0) / v.focal, 1.0];
        [0, 1, 2].map(|j| r[0][j] * d[0] + r[1][j] * d[1] + r[2][j] * d[2])
    };
    let was = |p: &View, q: V3| {
        let d = sub(q, p.pos);
        let z = dot(p.axes[2], d).max(1e-6);
        [w as f32 / 2.0 + p.focal * dot(p.axes[0], d) / z, h as f32 / 2.0 + p.focal * dot(p.axes[1], d) / z, z]
    };
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let d = ray(x as f32 + 0.5, y as f32 + 0.5);
            // The nearest front-facing triangle the ray meets, at least NEAR ahead.
            let mut best: Option<(f32, V3)> = None;
            for t in tris {
                let c = [[t[0], t[1], t[2]], [t[3], t[4], t[5]], [t[6], t[7], t[8]]];
                let n = cross(sub(c[1], c[0]), sub(c[2], c[0]));
                if dot(n, sub(c[0], v.pos)) >= 0.0 {
                    continue;
                }
                let dn = dot(d, n);
                if dn.abs() < 1e-9 {
                    continue;
                }
                let s = dot(sub(c[0], v.pos), n) / dn;
                let q = [0, 1, 2].map(|k| v.pos[k] + d[k] * s);
                let inside = (0..3).all(|k| dot(cross(sub(c[(k + 1) % 3], c[k]), sub(q, c[k])), n) >= 0.0);
                let z = dot(r[2], sub(q, v.pos));
                if inside && z >= 0.02 && z <= v.fog[1] && best.is_none_or(|(bz, _)| z < bz) {
                    best = Some((z, unit(n)));
                }
            }
            let Some((z, n)) = best else { continue };
            nearest[i] = 1.0 / z;
            let axis = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs())).unwrap();
            let p0 = [0, 1, 2].map(|k| v.pos[k] + d[k] * (z / dot(r[2], d)));
            let to = dot(sub(p0, v.pos), n);
            let hit = |d: V3| {
                let s = to / dot(d, n);
                [0, 1, 2].map(|k| v.pos[k] + d[k] * s)
            };
            let (px, py) = (hit(ray(x as f32 + 1.5, y as f32 + 0.5)), hit(ray(x as f32 + 0.5, y as f32 + 1.5)));
            let others = match axis {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            let cell = [0, 1, 2].map(|k| (p0[k] - n[k] * 0.01).floor() as i32);
            faces[i] = Some([axis as i32 * 2 + (n[axis] > 0.0) as i32, cell[0], cell[1], cell[2]]);
            let dist = dot(sub(p0, v.pos), sub(p0, v.pos)).sqrt();
            let fog = ((dist - v.fog[0]) / (v.fog[1] - v.fog[0])).clamp(0.0, 1.0);
            let eye = sub(v.pos, p0);
            let rise = dot(eye, n).abs().max(1e-4);
            surfaces[i] = Surface {
                page: 0.0,
                uv: others.map(|k| (p0[k] - p0[k].floor()).clamp(0.0, 0.9999)),
                duv_dx: others.map(|k| px[k] - p0[k]),
                duv_dy: others.map(|k| py[k] - p0[k]),
                strength: 1.0 - fog,
                slope: others.map(|k| eye[k] / rise),
            };
            if let Some((l, near)) = last {
                let [x0, y0, z] = was(l, p0);
                let (xi, yi) = (x0.floor(), y0.floor());
                if z > 0.02 && xi >= 0.0 && yi >= 0.0 && xi < w as f32 && yi < h as f32 && (near[(yi as u32 * w + xi as u32) as usize] - 1.0 / z).abs() <= 0.02 / z {
                    let (a, b) = (was(l, px), was(l, py));
                    motion[i] = Motion { from: [x0, y0], dfrom_dx: [a[0] - x0, a[1] - y0], dfrom_dy: [b[0] - x0, b[1] - y0] };
                }
            }
        }
    }
    (faces, surfaces, nearest, motion)
}

/// A room with a pillar and a step drawn on the GPU (`encode_view`) from one view a few frames
/// (for the faces seen to come back and get pages), then a step on: its G-buffer against the
/// CPU's.
fn gbuffer_check(dr: &mut DreamRenderer, input: &wgpu::Texture, output: &wgpu::Texture, w: u32, h: u32) {
    let mut tris = Vec::new();
    boxed(&mut tris, [0.0, 0.0, 0.0], [7.0, 3.0, 6.0], false);
    boxed(&mut tris, [4.0, 0.0, 3.0], [5.0, 3.0, 4.0], true);
    boxed(&mut tris, [1.0, 0.0, 4.0], [3.0, 0.5, 5.5], true);
    // A slope.
    quad(&mut tris, [[5.5, 0.0, 0.5], [6.5, 0.0, 0.5], [6.5, 1.0, 2.5], [5.5, 1.0, 2.5]], [0.0, 1.0, -0.5]);
    dr.set_level(&tris).expect("level");
    let focal = h as f32 / 2.0 / (0.5f32).tan();
    let a = look([1.2, 1.5, 0.8], 0.7, 0.15, focal);
    let b = look([1.3, 1.48, 0.86], 0.72, 0.16, focal);
    let (device, queue) = (dr.device().clone(), dr.queue().clone());
    let quiet = Step { lr: 0.0, lr_fine: 0.0, ..Step::default() };
    for v in [a, a, a, a, b] {
        let mut enc = device.create_command_encoder(&Default::default());
        dr.encode_view(&mut enc, input, w, Some(&v), None, true, &quiet, output).expect("encode_view");
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
    let (gs, gm) = dr.read_gbuffer().expect("read");
    let (_, _, near_a, _) = cpu_gbuffer(&tris, &a, w, h, None);
    let (faces, cs, _, cm) = cpu_gbuffer(&tris, &b, w, h, Some((&a, &near_a)));
    let (mut shown, mut paged, mut same_face, mut moved, mut moved_both) = (0, 0, 0, 0, 0);
    let mut only = [0, 0];
    let mut worst = [0.0f32; 5];
    for i in 0..(w * h) as usize {
        let (g, c) = (&gs[i], &cs[i]);
        let gface = (g.page >= 0.0).then(|| dr.face_of_page(g.page as u32)).flatten();
        if faces[i].is_some() {
            shown += 1;
            paged += gface.is_some() as u32;
        }
        if let (Some(f), Some(gf)) = (faces[i], gface) {
            if f == gf {
                same_face += 1;
                let d = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs().max((a[1] - b[1]).abs());
                for (k, e) in [d(g.uv, c.uv), d(g.duv_dx, c.duv_dx).max(d(g.duv_dy, c.duv_dy)), (g.strength - c.strength).abs(), d(g.slope, c.slope) / c.slope[0].abs().max(c.slope[1].abs()).max(1.0)].into_iter().enumerate() {
                    worst[k] = worst[k].max(e);
                }
            }
        }
        let (gv, cv) = (gm[i].from[0] > -1e8, cm[i].from[0] > -1e8);
        moved += cv as u32;
        if gv && cv {
            moved_both += 1;
            let d = (gm[i].from[0] - cm[i].from[0]).abs().max((gm[i].from[1] - cm[i].from[1]).abs());
            worst[4] = worst[4].max(d);
        } else if gv != cv {
            only[gv as usize] += 1;
        }
    }
    println!("G-buffer on the GPU against the CPU's, {} triangles, {w} x {h}:", tris.len());
    println!("  {shown} pixels show the level, {paged} with a page; {same_face} of those on the same face as the CPU's");
    println!("  largest difference where they agree: uv {:.2e}, uv a pixel on {:.2e}, strength {:.2e}, slope {:.2e} (relative)", worst[0], worst[1], worst[2], worst[3]);
    println!("  motion: {moved} pixels seen in the last view by the CPU, {moved_both} by both (only the CPU {}, only the GPU {}); largest difference {:.2e} pixels", only[0], only[1], worst[4]);
}
