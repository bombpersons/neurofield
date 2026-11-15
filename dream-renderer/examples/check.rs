//! The shaders against PyTorch: activations and the input gradient of a frame
//! (train/export_googlenet.py --fixture), then how long a frame of dreaming takes.
//!
//!     cargo run --release -p dream-renderer --example check -- WEIGHTS FIXTURE
//!
//! Then the G-buffer the renderer makes from a level drawn on the GPU against one made on the
//! CPU, as game/src/dream_atlas.rs `surfaces` makes it.

use dream_renderer::{Config, DreamRenderer, Edge, Face, LevelTri, Motion, Pushed, Step, Surface, View};
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
    // Timestamps for GAME_DREAM_PROFILE, where the adapter has them.
    let timing = adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    let (d, q) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, required_features: timing, ..Default::default() })).expect("a device");
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

    // GoogLeNet's fixture (train/export_googlenet.py) or ResNet's (train/export_resnet.py).
    let (wanted, one, two): (&[&str], &[&str], &[&str]) = if f.contains_key("grad_layer3.5") {
        (&["conv1", "maxpool", "layer1.0", "layer2.0", "layer3.5"], &["layer3.5"], &["layer2.3", "layer3.5"])
    } else {
        (&["conv1", "maxpool1", "conv3", "maxpool2", "inception3a", "inception4c"], &["inception4c"], &["inception3b", "inception4d"])
    };
    let (g, acts) = dr.input_gradient(rgb, h, w, one, wanted).expect("gradient");
    for (name, a) in wanted.iter().zip(&acts) {
        compare(&format!("activation {name}"), a, &f[&format!("act_{name}")].1);
    }
    compare(&format!("gradient {}", one.join(" + ")), &g, &f[&format!("grad_{}", one.join("+"))].1);
    let (g, _) = dr.input_gradient(rgb, h, w, two, &[]).expect("gradient");
    compare(&format!("gradient {}", two.join(" + ")), &g, &f[&format!("grad_{}", two.join("+"))].1);

    // A frame of dreaming on the fixture's picture, every pixel on one of a grid of pages.
    let picture: Vec<u8> = rgb.chunks_exact(3).flat_map(|p| [(p[0] * 255.0) as u8, (p[1] * 255.0) as u8, (p[2] * 255.0) as u8, 255]).collect();
    let surfaces: Vec<Surface> = (0..h * w)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let (px, py) = ((x / 64.0).floor(), (y / 64.0).floor());
            Surface { page: py * 5.0 + px, uv: [x / 64.0 - px, y / 64.0 - py], duv_dx: [1.0 / 64.0, 0.0], duv_dy: [0.0, 1.0 / 64.0], strength: 1.0, slope: [0.3, -0.2], pic: [x + 0.5, y + 0.5], dist: 2.0 }
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
        for (k, ms, n) in dr.profile(h, w, &[one[0]]).expect("profile") {
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
    height_check(&mut dr, &picture, w, h, cfg.up);
}

/// The height's gradient through the dream lamp (shade_back) against the objective's change
/// when a pixel's height is nudged either way, on the room's surfaces.
fn height_check(dr: &mut DreamRenderer, picture: &[u8], w: u32, h: u32, up: u32) {
    let (surfaces, _) = dr.read_gbuffer().expect("surfaces");
    let (sw, sh) = (w * up, h * up);
    // Smooth bumps, a few hundredths of a page high.
    let heights: Vec<f32> = (0..sw * sh).map(|i| {
        let (x, y) = ((i % sw) as f32, (i / sw) as f32);
        0.02 * ((x * 0.05).sin() * (y * 0.07).cos() + 0.5 * (x * 0.13 + y * 0.11).sin())
    }).collect();
    for (what, lamp) in [("colour and shading", [0.4, -0.3, 0.866, 0.3, 1.0, 0.0]), ("shading alone", [-0.5, 0.2, 0.84, 0.3, 1.0, 1.0])] {
        let (_, grad) = dr.lamp_gradient(picture, w, &surfaces, &heights, lamp).expect("gradient");
        let mut rng = 12345u64;
        let (mut worst, mut n, mut dots, mut na, mut nb) = (0.0f64, 0, 0.0f64, 0.0f64, 0.0f64);
        let scale = grad.iter().map(|g| g.abs()).fold(0.0f32, f32::max) as f64;
        while n < 24 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let (bx, by) = (2 + (rng >> 20) as u32 % (w - 4), 2 + (rng >> 40) as u32 % (h - 4));
            let i = (by * w + bx) as usize;
            if surfaces[i].page < 0.0 || grad[i].abs() < 0.05 * scale as f32 {
                continue;
            }
            let eps: f32 = std::env::var("EPS").ok().and_then(|e| e.parse().ok()).unwrap_or(1e-4);
            let mut obj = [0.0f64; 2];
            for (k, sign) in [1.0f32, -1.0].into_iter().enumerate() {
                let mut hh = heights.clone();
                for y in by * up..(by + 1) * up {
                    for x in bx * up..(bx + 1) * up {
                        hh[(y * sw + x) as usize] += sign * eps;
                    }
                }
                obj[k] = dr.lamp_gradient(picture, w, &surfaces, &hh, lamp).expect("gradient").0;
            }
            let fd = (obj[0] - obj[1]) / (2.0 * eps as f64);
            let g = grad[i] as f64;
            worst = worst.max((fd - g).abs() / scale);
            dots += fd * g;
            na += fd * fd;
            nb += g * g;
            n += 1;
        }
        println!("height gradient through the dream lamp ({what}) against nudging 24 pixels' heights: largest difference {:.2}% of the largest gradient, cosine {:.4}", 100.0 * worst, dots / (na * nb).sqrt());
    }
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
    View { pos, axes: [r, d, f], focal, fog: [3.0, 12.0], torch: [1.0, 0.8, 0.6, 3.0], fog_colour: [0.1, 0.0, 0.15], torch_at: [0.0; 3] }
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
                pic: [x as f32 + 0.5, y as f32 + 0.5],
                dist,
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
    // Each triangle on its own grey texture, half lit by the level and taking all the torch.
    let level: Vec<LevelTri> = tris.iter().map(|&pos| LevelTri { pos, uv: [0.0, 0.0, 1.0, 0.0, 0.0, 1.0], light: [1.0; 9], albedo: [1.0; 9], texture: 0 }).collect();
    dr.set_level(&level).expect("level");
    dr.set_textures(2, &[[96u8, 96, 96, 255].repeat(4)]).expect("textures");
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
    // Pushed out: each page its own height, so the faces' edges only meet if the vertices
    // there take the mean of every face they're on. Closed, the room shows no holes.
    let mut rng = 99u64;
    let mut heights = vec![0.0f32; 512];
    for h in &mut heights {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *h = (rng >> 40) as f32 / (1u64 << 24) as f32 * 0.3 - 0.1;
    }
    let corner = look([3.3, 1.2, 2.2], 0.9, 0.05, focal);
    let drawn = |dr: &mut DreamRenderer, step: &Step| {
        let mut enc = device.create_command_encoder(&Default::default());
        dr.encode_view(&mut enc, input, w, Some(&corner), None, true, step, output).expect("encode_view");
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        (dr.read_gbuffer().expect("read").0, dr.read_triangles().expect("triangles"))
    };
    let (flat, flat_ids) = drawn(dr, &quiet);
    let flat_dist: Vec<f32> = flat.iter().map(|s| s.dist).collect();
    dr.fill_heights(|p| heights[p as usize]);
    // Flat (displace 0, the dream's G-buffer): the height moves nothing.
    let (raised, raised_ids) = drawn(dr, &quiet);
    let same = |a: &Surface, b: &Surface| a.page == b.page && a.uv == b.uv && a.duv_dx == b.duv_dx && a.duv_dy == b.duv_dy && a.slope == b.slope && a.dist == b.dist && a.pic == b.pic;
    let changed = flat.iter().zip(&raised).filter(|(a, b)| !same(a, b)).count() + flat_ids.iter().zip(&raised_ids).filter(|(a, b)| a != b).count();
    println!("  flat, a height each page, close to the pillar's corner: {changed} of {} pixels' G-buffer changed by the height", flat.len());
    let quiet = Step { displace: 1.0, ..quiet };
    for (what, view) in [("from a", a), ("from b", b), ("close to the pillar's corner", look([3.3, 1.2, 2.2], 0.9, 0.05, focal)), ("close to a corner of the room", look([5.6, 2.3, 4.6], 0.8, -0.5, focal))] {
        let mut enc = device.create_command_encoder(&Default::default());
        dr.encode_view(&mut enc, input, w, Some(&view), None, true, &quiet, output).expect("encode_view");
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let ids = dr.read_triangles().expect("triangles");
        let holes = ids.iter().filter(|&&t| t == 0).count();
        println!("  pushed out -0.1..0.2 a page, a height each page, {what}: {holes} pixels show nothing");
        if what.starts_with("close to the pillar") {
            let d: Vec<f32> = dr.read_gbuffer().expect("read").0.iter().map(|s| s.dist).collect();
            let moved = d.iter().zip(&flat_dist).filter(|(a, b)| (*a - *b).abs() > 0.01).count();
            println!("    {moved} of {} pixels' distance changed from the flat level's by over a hundredth", d.len());
            let mut ppm = format!("P5 {w} {h} 255\n").into_bytes();
            ppm.extend(d.iter().map(|&x| (255.0 * (1.0 - x / 5.0)).clamp(0.0, 255.0) as u8));
            let _ = std::fs::write(std::env::temp_dir().join("dream_pushed_depth.pgm"), ppm);
        }
    }
    dr.fill_heights(|_| 0.0);
    shell_check(dr, input, output, w, h, &tris, focal);
}

/// examples' `Step` for the shell: the march on, nothing dreamed.
fn shell_step(limit: f32) -> Step {
    Step { lr: 0.0, lr_fine: 0.0, shell: 1.0, height_max: limit, ..Step::default() }
}

/// shell.wgsl's march done finely on the CPU for the shown pixel (x, y): the nearest (view
/// depth, triangle, foot, the hit's height above the plane) over every triangle's three
/// tetrahedra (`pushes`), `reach` out and in (within each corner's cap).
#[allow(clippy::too_many_arguments)]
fn cpu_march(tris: &[[f32; 9]], pushes: &[Pushed], edges: &[[Edge; 3]], across: &[[Option<usize>; 3]], taper: f32, v: &View, up: u32, sw: u32, sh: u32, x: u32, y: u32, reach: f32, r: i32, far: f32, height: &dyn Fn(Face, [i32; 2]) -> Option<(f32, f32)>) -> Option<(f32, usize, V3, f32)> {
    let f = v.focal * up as f32;
    let dv = [(x as f32 + 0.5 - sw as f32 / 2.0) / f, (y as f32 + 0.5 - sh as f32 / 2.0) / f, 1.0];
    let a = v.axes;
    let d = [0, 1, 2].map(|k| a[0][k] * dv[0] + a[1][k] * dv[1] + a[2][k] * dv[2]);
    let o = v.pos;
    let mut best: Option<(f32, usize, V3, f32)> = None;
    for (ti, t) in tris.iter().enumerate() {
        let c = [0, 1, 2].map(|k| [t[3 * k], t[3 * k + 1], t[3 * k + 2]]);
        let n = unit(cross(sub(c[1], c[0]), sub(c[2], c[0])));
        let d0 = dot(c[0], n);
        let push = &pushes[ti];
        let reaches = push.cap.map(|cap| reach.min(cap));
        let sides = [0, 1, 2].map(|k| {
            let m = unit(cross(sub(c[(k + 1) % 3], c[k]), n));
            (m, dot(m, c[k]))
        });
        // As shell.wgsl's raw_height: on the plane m . x = md, bilinear between the height
        // cells of its grid, each from the page of the tile it's in, those with none left out.
        let raw = |foot: V3, m: V3, md: f32| {
            let axis = (0..3).max_by(|&p, &q| m[p].abs().total_cmp(&m[q].abs())).unwrap();
            let orient = axis as i32 * 2 + (m[axis] > 0.0) as i32;
            let pick = |p: V3| match axis {
                0 => [p[1], p[2]],
                1 => [p[0], p[2]],
                _ => [p[0], p[1]],
            };
            let axes: [V3; 2] = match axis {
                0 => [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                1 => [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                _ => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            };
            let pf = pick(foot);
            let g = pf.map(|c| c * r as f32 - 0.25);
            let i0 = g.map(|c| c.floor() as i32);
            let f = [g[0] - g[0].floor(), g[1] - g[1].floor()];
            let (mut sum, mut wsum) = (0.0, 0.0);
            for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let gi = [i0[0] + di, i0[1] + dj];
                let c = gi.map(|q| (q as f32 + 0.25) / r as f32);
                let mut q: V3 = std::array::from_fn(|k| foot[k] + axes[0][k] * (c[0] - pf[0]) + axes[1][k] * (c[1] - pf[1]));
                let off = dot(m, q) - md;
                q = std::array::from_fn(|k| q[k] - m[k] * off);
                let cell = [0, 1, 2].map(|k| (q[k] - m[k] * 0.01).floor() as i32);
                let local = [0, 1].map(|k| (gi[k] - c[k].floor() as i32 * r).clamp(0, r - 1));
                if let Some((near, far_off)) = height([orient, cell[0], cell[1], cell[2]], local) {
                    // As shell.wgsl's far_weight: smoothed more by the foot's view depth.
                    let k = (far * dot(v.axes[2], sub(foot, v.pos)) / 4.0).clamp(0.0, 1.0);
                    let v = near * (1.0 - k) + far_off * k;
                    let w = (if di == 1 { f[0] } else { 1.0 - f[0] }) * (if dj == 1 { f[1] } else { 1.0 - f[1] });
                    sum += w * v;
                    wsum += w;
                }
            }
            if wsum > 0.0 { sum / wsum } else { 0.0 }
        };
        let smooth = |x: f32, width: f32| {
            let t = (x / width.max(1e-6)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        // As shell.wgsl's height_at: met at each crease by the wall beyond's, tapered to open
        // edges.
        let height_at = |foot: V3| {
            let mut h = raw(foot, n, d0);
            for k in 0..3 {
                let Some(j) = across[ti][k].filter(|_| edges[ti][k].crease()) else { continue };
                let dist = sides[k].1 - dot(sides[k].0, foot);
                let e: V3 = std::array::from_fn(|q| foot[q] + sides[k].0[q] * dist);
                let o = tris[j];
                let m = unit(cross(sub([o[3], o[4], o[5]], [o[0], o[1], o[2]]), sub([o[6], o[7], o[8]], [o[0], o[1], o[2]])));
                let toward = 0.5 * (raw(e, m, dot(m, [o[0], o[1], o[2]])) - raw(e, n, d0));
                h += toward * (1.0 - smooth(dist.max(0.0), taper.max(toward.abs() * 2.0)));
            }
            h * (0..3).filter(|&k| edges[ti][k].tapers()).map(|k| smooth(sides[k].1 - dot(sides[k].0, foot), taper.max(h.abs() * 2.0))).product::<f32>()
        };
        // As shell.wgsl's mapping: x = sum_k b_k (corner_k + s dir_k reach_k), solved for s
        // (Newton, in f64) and then b.
        let f64s = |v: V3| v.map(|x| x as f64);
        let cr = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let dt = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let add = |a: [f64; 3], b: [f64; 3], k: f64| [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
        let cs = c.map(f64s);
        let dirs: [[f64; 3]; 3] = std::array::from_fn(|k| push.dir[k].map(|x| x as f64 * reaches[k] as f64));
        let n64 = f64s(n);
        let place_of = |x: [f64; 3], guess: f64| {
            let q0 = add(x, cs[0], -1.0);
            let corner = |k: usize, s: f64| add(cs[k], dirs[k], s);
            let tri_at = |s: f64| {
                let a0 = corner(0, s);
                (add(corner(1, s), a0, -1.0), add(corner(2, s), a0, -1.0))
            };
            let f = |s: f64| {
                let (e1, e2) = tri_at(s);
                dt(cr(e1, e2), add(q0, dirs[0], -s))
            };
            let mut s = guess;
            for _ in 0..30 {
                let e = 1e-7;
                let df = (f(s + e) - f(s - e)) / (2.0 * e);
                if df.abs() < 1e-30 {
                    break;
                }
                let step = f(s) / df;
                s -= step;
                if step.abs() < 1e-13 {
                    break;
                }
            }
            let (e1, e2) = tri_at(s);
            let cc = cr(e1, e2);
            let q = add(q0, dirs[0], -s);
            let b1 = dt(cr(q, e2), cc) / dt(cc, cc);
            let b2 = dt(cr(e1, q), cc) / dt(cc, cc);
            ([1.0 - b1 - b2, b1, b2], s)
        };
        for tet in 0..3 {
            // The tetrahedron's corners, and where the line of sight's inside it (in f64).
            let p = push.tet(&c, tet, reach).map(f64s);
            let (o64, d64) = (f64s(o), f64s(d));
            let mut span = (0.02f64, v.fog[1] as f64);
            for i in 0..4 {
                let a0 = p[(i + 1) % 4];
                let mut m = cr(add(p[(i + 2) % 4], a0, -1.0), add(p[(i + 3) % 4], a0, -1.0));
                if dt(m, add(p[i], a0, -1.0)) < 0.0 {
                    m = m.map(|x| -x);
                }
                let (dm, room) = (dt(m, d64), dt(m, add(o64, a0, -1.0)));
                if dm.abs() < 1e-30 {
                    if room < 0.0 {
                        span = (1.0, 0.0);
                    }
                } else if dm > 0.0 {
                    span.0 = span.0.max(-room / dm);
                } else {
                    span.1 = span.1.min(-room / dm);
                }
            }
            if span.1 <= span.0 || best.is_some_and(|b| (b.0 as f64) < span.0) {
                continue;
            }
            let xe = add(o64, d64, span.0);
            let guess = (dt(n64, xe) - d0 as f64) / (dt(n64, add(add(dirs[0], dirs[1], 1.0), dirs[2], 1.0)) / 3.0).max(1e-9);
            let probe = |t: f64| {
                let (b, s) = place_of(add(o64, d64, t), guess);
                let foot: V3 = std::array::from_fn(|k| (b[0] * cs[0][k] + b[1] * cs[1][k] + b[2] * cs[2][k]) as f32);
                let lim = (b[0] * reaches[0] as f64 + b[1] * reaches[1] as f64 + b[2] * reaches[2] as f64) as f32;
                (s as f32 * lim - height_at(foot).clamp(-lim, lim), foot, s)
            };
            const STEPS: u32 = 700;
            let (mut t0, mut t1) = (span.0, span.0);
            let mut met = probe(span.0).0 <= 0.0;
            for i in 1..=STEPS {
                if met {
                    break;
                }
                let ti = span.0 + (span.1 - span.0) * i as f64 / STEPS as f64;
                if probe(ti).0 <= 0.0 {
                    t1 = ti;
                    met = true;
                } else {
                    t0 = ti;
                }
            }
            if !met {
                // As shell.wgsl: out through the inner shell, met there.
                if probe(span.1).2 > -1.0 + 1e-4 {
                    continue;
                }
                (t0, t1) = (span.1, span.1);
            }
            for _ in 0..40 {
                let tm = 0.5 * (t0 + t1);
                if probe(tm).0 <= 0.0 { t1 = tm } else { t0 = tm }
            }
            let (_, foot, _) = probe(t1);
            let t1 = t1 as f32;
            let hit: V3 = std::array::from_fn(|k| o[k] + d[k] * t1);
            if best.is_none_or(|b| t1 < b.0) {
                best = Some((t1, ti, foot, dot(n, hit) - d0));
            }
        }
    }
    best
}

/// The shell (docs/shell_relief.md step 2): its march on the GPU against the CPU's at a few
/// hundred shown pixels, flat and with a smooth bump on each page, and the pixels left with
/// nothing (the room is closed, so none should be).
fn shell_check(dr: &mut DreamRenderer, input: &wgpu::Texture, output: &wgpu::Texture, w: u32, h: u32, tris: &[[f32; 9]], focal: f32) {
    let (device, queue) = (dr.device().clone(), dr.queue().clone());
    let level: Vec<LevelTri> = tris.iter().map(|&pos| LevelTri { pos, ..Default::default() }).collect();
    let edges = dream_renderer::edges(&level);
    let pushes = dream_renderer::pushed(&level, &edges);
    let beyond = dream_renderer::neighbours(&level);
    let caps: Vec<f32> = pushes.iter().flat_map(|p| p.cap).filter(|&c| c < 1e5).collect();
    let mitres: Vec<f32> = pushes.iter().flat_map(|p| p.dir).map(|d| dot(d, d).sqrt()).collect();
    println!(
        "shell: {} triangles; corners held to {} reaches, the least {:.3}, {} under 0.1; pushed up to {:.3} times the height",
        tris.len(),
        caps.len(),
        caps.iter().copied().fold(f32::INFINITY, f32::min),
        caps.iter().filter(|&&c| c < 0.1).count(),
        mitres.iter().copied().fold(0.0, f32::max)
    );
    let taper = Step::default().shell_taper;
    let up = 3;
    let (sw, sh) = (w * up, h * up);
    let (r, k) = (dr.height_cells(), r_k(dr));
    let mut page_of: HashMap<Face, u32> = HashMap::new();
    for p in 0..65536 {
        if let Some(f) = dr.face_of_page(p) {
            page_of.insert(f, p);
        }
    }
    // A bump on each page, up or down by its own amount, nothing at its edges.
    let amp = |p: u32| ((p.wrapping_mul(2654435761) >> 8) % 1000) as f32 / 1000.0 * 0.24 - 0.12;
    let bump = move |p: u32, i: u32, j: u32| {
        let (u, v) = ((2 * i) as f32 + 0.5, (2 * j) as f32 + 0.5);
        amp(p) * (std::f32::consts::PI * u / k as f32).sin() * (std::f32::consts::PI * v / k as f32).sin()
    };
    let limit = 0.15;
    let views = [
        ("from a", look([1.2, 1.5, 0.8], 0.7, 0.15, focal)),
        ("close to the pillar's corner", look([3.3, 1.2, 2.2], 0.9, 0.05, focal)),
        ("glancing along a wall", look([0.3, 1.0, 0.4], 0.05, 0.1, focal)),
    ];
    // A bulge and a hollow across the pillar's corner (the edge x 4, z 3), by where each height
    // cell's middle is in the level, so both walls give the same height at the edge.
    let faces: Vec<Option<Face>> = (0..65536).map(|p| dr.face_of_page(p)).collect();
    let across = move |p: u32, i: u32, j: u32| {
        let Some(f) = faces[p as usize] else { return 0.0 };
        let axis = (f[0] / 2) as usize;
        let along = [[1, 2], [0, 2], [0, 1]][axis];
        let mut x = [0.0f32; 3];
        x[axis] = (f[1 + axis] + f[0] % 2) as f32;
        for (q, c) in along.into_iter().zip([i, j]) {
            x[q] = f[1 + q] as f32 + (c as f32 + 0.25) / r as f32;
        }
        let blob = |c: V3, a: f32| {
            let d = sub(x, c);
            a * (-dot(d, d) / (0.3f32 * 0.3)).exp()
        };
        blob([4.0, 1.2, 3.0], 0.12) + blob([4.0, 2.2, 3.0], -0.1)
    };
    // Smoothed (smooth.wgsl): two and a half levels, and two more by 4 / 0.6 tiles off.
    let (smooth, far) = (2.5, 0.6);
    for (what, case) in [("flat", 0), ("across the pillar's corner", 2), ("bumped", 1), ("bumped, smoothed", 3)] {
        let field = |p: u32, i: u32, j: u32| match case {
            0 => 0.0,
            1 | 3 => bump(p, i, j),
            _ => across(p, i, j),
        };
        let step = if case == 3 { Step { shell_smooth: smooth, shell_far: far, ..shell_step(limit) } } else { shell_step(limit) };
        // Every face in sight given its page first, so the CPU knows them all (a page given a
        // face later is clear on the GPU, where the CPU would leave it out).
        for (_, v) in &views {
            let mut enc = device.create_command_encoder(&Default::default());
            dr.encode_view(&mut enc, input, w, Some(v), None, true, &step, output).expect("encode_view");
            queue.submit([enc.finish()]);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        page_of = (0..65536).filter_map(|p| dr.face_of_page(p).map(|f| (f, p))).collect();
        dr.fill_height_cells(field);
        let smoothed = (case == 3).then(|| (cpu_smooth(&page_of, &field, r as usize, smooth), cpu_smooth(&page_of, &field, r as usize, smooth + 2.0)));
        if let Some((near, far_off)) = &smoothed {
            // The smoothing itself, against the CPU's, after a frame.
            let mut enc = device.create_command_encoder(&Default::default());
            dr.encode_view(&mut enc, input, w, Some(&views[0].1), None, true, &step, output).expect("encode_view");
            queue.submit([enc.finish()]);
            let mut worst = [0.0f32; 2];
            let mut where_worst = None;
            for (&p, want) in near {
                let (gn, gf) = dr.read_shell_smoothed(p).expect("smoothed");
                // (Near on the GPU has the far-off smoothing mixed in by the view: far alone.)
                let _ = (&gn, want);
                for (k, (g, c)) in [(1, (&gf, &far_off[&p]))] {
                    for (i, (a, b)) in g.iter().zip(c.iter()).enumerate() {
                        if (a - b).abs() > worst[k] {
                            worst[k] = (a - b).abs();
                            if k == 1 {
                                where_worst = Some((p, i, *a, *b));
                            }
                        }
                    }
                }
            }
            println!("smoothed heights (two levels more, for far off) against the CPU's on {} pages: largest difference {:.2e} (at {:?})", near.len(), worst[1], where_worst.map(|(p, i, a, b)| (dr.face_of_page(p), i, a, b)));
        }
        for (view_name, v) in &views {
            if case == 2 && !view_name.starts_with("close to the pillar") {
                continue;
            }
            // As smooth.wgsl: the far-off smoothing mixed in by each cell's view depth (its
            // middle, on an axis-aligned face's plane).
            let height = |f: Face, c: [i32; 2]| {
                page_of.get(&f).map(|&p| match &smoothed {
                    Some((near, far_off)) => {
                        let at = (c[1] * r as i32 + c[0]) as usize;
                        let axis = (f[0] / 2) as usize;
                        let along = [[1, 2], [0, 2], [0, 1]][axis];
                        let mut x = [f[1] as f32, f[2] as f32, f[3] as f32];
                        x[axis] += (f[0] % 2) as f32;
                        for (q, i) in along.into_iter().zip(c) {
                            x[q] += (i as f32 + 0.25) / r as f32;
                        }
                        let k = (far * dot(v.axes[2], sub(x, v.pos)) / 4.0).clamp(0.0, 1.0);
                        let m = near[&p][at] * (1.0 - k) + far_off[&p][at] * k;
                        (m, m)
                    }
                    None => {
                        let v = field(p, c[0] as u32, c[1] as u32);
                        (v, v)
                    }
                })
            };
            let mut enc = device.create_command_encoder(&Default::default());
            dr.encode_view(&mut enc, input, w, Some(v), None, true, &step, output).expect("encode_view");
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            // SHELL_TIME: the whole frame's time from this view on these heights, for comparing
            // the march's speed on heights that stay put (the game's grow differently each run).
            if std::env::var_os("SHELL_TIME").is_some() {
                let mut times: Vec<f32> = (0..30)
                    .map(|_| {
                        let t0 = std::time::Instant::now();
                        let mut enc = device.create_command_encoder(&Default::default());
                        dr.encode_view(&mut enc, input, w, Some(v), None, true, &step, output).expect("encode_view");
                        queue.submit([enc.finish()]);
                        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
                        t0.elapsed().as_secs_f32() * 1000.0
                    })
                    .collect();
                times.sort_by(f32::total_cmp);
                println!("  frame from this view: median {:.2} ms, best {:.2}", times[15], times[0]);
            }
            let hits = dr.read_shell().expect("shell");
            // The prisms reach just past the largest height on any page.
            let reach = dr.shell_reach(&step).expect("reach");
            let holes = hits.iter().filter(|s| s.tri == 0).count();
            let mut rng = 7u64;
            let mut examples = Vec::new();
            let (mut same, mut both_none, mut same_point, mut differ) = (0, 0, 0, 0);
            let mut worst = [0.0f32; 3];
            const PIXELS: u32 = 400;
            for _ in 0..PIXELS {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let (x, y) = ((rng >> 20) as u32 % sw, (rng >> 40) as u32 % sh);
                let g = hits[(y * sw + x) as usize];
                match cpu_march(tris, &pushes, &edges, &beyond, taper, v, up, sw, sh, x, y, reach, r as i32, 0.0, &height) {
                    None if g.tri == 0 => both_none += 1,
                    Some((t, ti, foot, hh)) if g.tri == ti as u32 + 1 => {
                        same += 1;
                        let df = (0..3).map(|q| (foot[q] - g.foot[q]).abs()).fold(0.0, f32::max);
                        worst[0] = worst[0].max(df);
                        worst[1] = worst[1].max((g.depth - t).abs() / t);
                        if (g.height - hh).abs() > worst[2] && (g.height - hh).abs() > 0.01 {
                            examples.push(format!("    height at ({x}, {y}): the GPU's {:.4} (depth {:.4}, foot {:?}), the CPU's {hh:.4} (depth {t:.4}, foot {foot:?})", g.height, g.depth, g.foot));
                        }
                        worst[2] = worst[2].max((g.height - hh).abs());
                    }
                    Some((t, _, foot, _)) if g.tri != 0 && (g.depth - t).abs() / t < 2e-3 && (0..3).all(|q| (foot[q] - g.foot[q]).abs() < 2e-3) => same_point += 1,
                    c => {
                        differ += 1;
                        if differ <= 3 {
                            examples.push(format!("    at ({x}, {y}): the GPU's triangle {} at depth {:.4} foot {:?} h {:.4}, the CPU's {:?}", g.tri, g.depth, g.foot, g.height, c.map(|c| (c.1 + 1, c.0, c.2, c.3))));
                        }
                    }
                }
            }
            println!("shell, {what}, {view_name}: reaching {reach:.3}, {holes} of {} shown pixels meet nothing", hits.len());
            if case == 2 {
                // Neighbouring pixels either side of the corner (on the pillar's two walls, near
                // the edge): how far apart their hits are, against a height cell.
                let f = v.focal * up as f32;
                let hit_at = |x: u32, y: u32| {
                    let g = hits[(y * sw + x) as usize];
                    let dv = [(x as f32 + 0.5 - sw as f32 / 2.0) / f, (y as f32 + 0.5 - sh as f32 / 2.0) / f, 1.0];
                    let p: V3 = std::array::from_fn(|k| v.pos[k] + (v.axes[0][k] * dv[0] + v.axes[1][k] * dv[1] + v.axes[2][k] * dv[2]) * g.depth);
                    (g.tri, p)
                };
                let normal_of = |t: u32| {
                    let c = tris[t as usize - 1];
                    unit(cross(sub([c[3], c[4], c[5]], [c[0], c[1], c[2]]), sub([c[6], c[7], c[8]], [c[0], c[1], c[2]])))
                };
                let (mut pairs, mut over, mut widest) = (0, 0, 0.0f32);
                for y in 0..sh - 1 {
                    for x in 0..sw - 1 {
                        let (ta, pa) = hit_at(x, y);
                        for (tb, pb) in [hit_at(x + 1, y), hit_at(x, y + 1)] {
                            if ta == 0 || tb == 0 || ta == tb || dot(normal_of(ta), normal_of(tb)) > 0.99 {
                                continue;
                            }
                            let near = |p: V3| ((p[0] - 4.0).powi(2) + (p[2] - 3.0).powi(2)).sqrt() < 0.4 && p[0] > 3.6 && p[2] > 2.6;
                            if !near(pa) || !near(pb) {
                                continue;
                            }
                            let gap = sub(pa, pb);
                            let gap = dot(gap, gap).sqrt();
                            pairs += 1;
                            widest = widest.max(gap);
                            if gap > 1.0 / r as f32 {
                                over += 1;
                            }
                        }
                    }
                }
                println!("  either side of the corner: {pairs} neighbouring pixels' hits on its two walls, {over} further apart than a height cell ({:.4}), the furthest {widest:.4}", 1.0 / r as f32);
            }
            println!("  against the CPU's march at {PIXELS} pixels: {same} on the same triangle, {same_point} the same point on another (two meeting), {both_none} nothing for both, {differ} differ");
            println!("  largest differences on the same triangle: foot {:.2e} tiles, depth {:.3}%, height {:.2e} tiles", worst[0], 100.0 * worst[1], worst[2]);
            for e in examples {
                println!("{e}");
            }
        }
    }
    // At the height 0 the shell's frame is the flat one: the wall's colour now taken at each
    // shown pixel's own point (the texture sampled there, not at its base pixel's), so it
    // differs along the texels' edges, a pixel at most.
    dr.fill_heights(|_| 0.0);
    for (view_name, v) in &views {
        let mut shown = Vec::new();
        for shell in [0.0, 1.0] {
            let mut enc = device.create_command_encoder(&Default::default());
            dr.encode_view(&mut enc, input, w, Some(v), None, true, &Step { shell, ..shell_step(limit) }, output).expect("encode_view");
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            shown.push(dr.read_shown().expect("shown"));
        }
        let channel = |c: u32, k: u32| ((c >> (8 * k)) & 255) as i32;
        let diffs: Vec<i32> = shown[0].iter().zip(&shown[1]).map(|(&a, &b)| (0..3).map(|k| (channel(a, k) - channel(b, k)).abs()).max().unwrap()).collect();
        let mean = diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len() as f64;
        let near = diffs.iter().filter(|&&d| d <= 4).count();
        // Off by more only where a neighbour 3 pixels away (a base pixel) is the same as the other.
        let (sw, sh) = (w * 3, h * 3);
        let edge = (0..diffs.len())
            .filter(|&i| diffs[i] > 4)
            .filter(|&i| {
                let (x, y) = ((i as u32 % sw) as i32, (i as u32 / sw) as i32);
                (-3..=3).flat_map(|dy| (-3..=3).map(move |dx| (dx, dy))).any(|(dx, dy)| {
                    let (qx, qy) = (x + dx, y + dy);
                    qx >= 0 && qy >= 0 && qx < sw as i32 && qy < sh as i32 && {
                        let j = (qy as u32 * sw + qx as u32) as usize;
                        (0..3).all(|k| (channel(shown[0][j], k) - channel(shown[1][i], k)).abs() <= 4)
                    }
                })
            })
            .count();
        println!("shown, height 0, {view_name}: with the shell against without, mean difference {mean:.2} of 255; {near} of {} pixels within 4, {edge} more within a base pixel's reach", diffs.len());
        if view_name.starts_with("from a") {
            let mut ppm = format!("P6 {sw} {} 255\n", 2 * sh).into_bytes();
            for frame in &shown {
                ppm.extend(frame.iter().flat_map(|&c| [c as u8, (c >> 8) as u8, (c >> 16) as u8]));
            }
            let _ = std::fs::write(std::env::temp_dir().join("dream_shell_flat.ppm"), ppm);
        }
    }
}

/// smooth.wgsl's smoothing on the CPU, of the heights `field` (page, cell i, j) on the pages of
/// `page_of`, `r` cells a side: each page's mips, then each cell from level `l` (fractions
/// mixing two) by a cubic B-spline across its tile's edges into the neighbouring tiles' pages.
fn cpu_smooth(page_of: &HashMap<Face, u32>, field: &dyn Fn(u32, u32, u32) -> f32, r: usize, l: f32) -> HashMap<u32, Vec<f32>> {
    let top = (1..).take_while(|&k| r >> k >= 2).last().unwrap_or(0);
    let mips: HashMap<u32, Vec<Vec<f32>>> = page_of
        .values()
        .map(|&p| {
            let mut levels = vec![(0..r * r).map(|c| field(p, (c % r) as u32, (c / r) as u32)).collect::<Vec<f32>>()];
            for k in 1..=top {
                let (n, prev) = (r >> k, &levels[k - 1]);
                let pn = 2 * n;
                let next = (0..n * n)
                    .map(|c| {
                        let o = 2 * (c / n) * pn + 2 * (c % n);
                        0.25 * (prev[o] + prev[o + 1] + prev[o + pn] + prev[o + pn + 1])
                    })
                    .collect();
                levels.push(next);
            }
            (p, levels)
        })
        .collect();
    let bspline = |t: f32| {
        let (t2, s) = (t * t, 1.0 - t);
        [s * s * s, 3.0 * t2 * t - 6.0 * t2 + 4.0, -3.0 * t2 * t + 3.0 * t2 + 3.0 * t + 1.0, t2 * t].map(|w| w / 6.0)
    };
    let mut out = HashMap::new();
    for (&f, &p) in page_of {
        let axes: [[i32; 3]; 2] = match f[0] / 2 {
            0 => [[0, 1, 0], [0, 0, 1]],
            1 => [[1, 0, 0], [0, 0, 1]],
            _ => [[1, 0, 0], [0, 1, 0]],
        };
        let around: [Option<u32>; 9] = std::array::from_fn(|k| {
            let (dx, dy) = (k as i32 % 3 - 1, k as i32 / 3 - 1);
            page_of.get(&[f[0], f[1] + axes[0][0] * dx + axes[1][0] * dy, f[2] + axes[0][1] * dx + axes[1][1] * dy, f[3] + axes[0][2] * dx + axes[1][2] * dy]).copied()
        });
        let at_level = |k: usize, i: i32, j: i32| -> f32 {
            if k == 0 {
                return mips[&p][0][j as usize * r + i as usize];
            }
            let (n, s) = ((r >> k) as i32, (1 << k) as f32);
            let g = [(i as f32 + 0.5) / s - 0.5, (j as f32 + 0.5) / s - 0.5];
            let (wx, wy) = (bspline(g[0] - g[0].floor()), bspline(g[1] - g[1].floor()));
            let g0 = g.map(|x| x.floor() as i32);
            let tap = |qx: i32, qy: i32| {
                let side = |q: i32| if q < 0 { -1 } else if q >= n { 1 } else { 0 };
                let (dx, dy) = (side(qx), side(qy));
                match around[((dy + 1) * 3 + dx + 1) as usize] {
                    Some(q) => mips[&q][k][((qy - dy * n) * n + qx - dx * n) as usize],
                    None => mips[&p][k][(qy.clamp(0, n - 1) * n + qx.clamp(0, n - 1)) as usize],
                }
            };
            (0..4).map(|b| wy[b] * (0..4).map(|a| wx[a] * tap(g0[0] - 1 + a as i32, g0[1] - 1 + b as i32)).sum::<f32>()).sum()
        };
        let lc = l.clamp(0.0, top as f32);
        let (l0, fr) = (lc.floor() as usize, lc - lc.floor());
        let v = (0..r * r)
            .map(|c| {
                let (i, j) = ((c % r) as i32, (c / r) as i32);
                let a = at_level(l0, i, j);
                if fr <= 0.0 { a } else { a * (1.0 - fr) + at_level(l0 + 1, i, j) * fr }
            })
            .collect();
        out.insert(p, v);
    }
    out
}

/// The pages' texels a side (as the renderer was configured).
fn r_k(dr: &DreamRenderer) -> u32 {
    dr.height_cells() * 2
}
