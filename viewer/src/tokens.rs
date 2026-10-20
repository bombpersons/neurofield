//! Network inputs for one view: a port of `tri_tokens` and `cam_token` in
//! train/data.py, reading the same per-triangle and per-view features that
//! datagen writes to `tris.npy` and `views.npy`. Keep the two in step.

use datagen::TRI_FEATURES as RAW;

pub const MAX_TRIS: usize = 512;
/// Floats per triangle token and in the camera token (data.py TRI_FEATURES, CAM_FEATURES).
pub const TRI_FEATURES: usize = 45;
pub const CAM_FEATURES: usize = 7;

pub struct Tokens {
    /// [n, TRI_FEATURES], nearest triangle first.
    pub feats: Vec<f32>,
    /// [n] texture ids.
    pub tex: Vec<i64>,
    pub cam: [f32; CAM_FEATURES],
}

impl Tokens {
    pub fn len(&self) -> usize {
        self.tex.len()
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Camera-space triangle tokens for view `v` over a scene's raw triangle features `tris`.
pub fn tokens(tris: &[f32], v: &[f32]) -> Tokens {
    let pos = [v[1], v[2], v[3]];
    let r = [[v[4], v[5], v[6]], [v[7], v[8], v[9]], [v[10], v[11], v[12]]];
    let h = v[13];
    let fog_far = v[15];
    let near = h / 2.0 / 256.0;
    let k = 160.0 / h * 1.05;

    // (min z, camera-space vertices, normal, raw index) for every triangle that survives culling.
    let mut kept: Vec<(f32, [[f32; 3]; 3], [f32; 3], usize)> = Vec::new();
    for (i, t) in tris.chunks_exact(RAW).enumerate() {
        let p = [0, 1, 2].map(|j| {
            let d = sub([t[3 * j], t[3 * j + 1], t[3 * j + 2]], pos);
            [dot(r[0], d), dot(r[1], d), dot(r[2], d)]
        });
        let z = p.map(|q| q[2]);
        let zmin = z[0].min(z[1]).min(z[2]);
        if !z.iter().any(|&z| z > near) || zmin >= fog_far {
            continue;
        }
        let s = p.map(|q| {
            let zc = q[2].max(1e-3);
            (q[0] / zc, q[1] / zc, q[2] > 0.0)
        });
        let all = |f: &dyn Fn((f32, f32, bool)) -> bool| s.iter().all(|&e| f(e));
        if all(&|(x, _, a)| x > k && a)
            || all(&|(x, _, a)| x < -k && a)
            || all(&|(_, y, a)| y > 0.75 * k && a)
            || all(&|(_, y, a)| y < -0.75 * k && a)
        {
            continue;
        }
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        if dot(n, p[0].map(|c| -c)) <= 0.0 {
            continue;
        }
        kept.push((zmin, p, n, i));
    }
    kept.sort_by(|a, b| a.0.total_cmp(&b.0));
    kept.truncate(MAX_TRIS);

    let mut feats = Vec::with_capacity(kept.len() * TRI_FEATURES);
    let mut tex = Vec::with_capacity(kept.len());
    for (_, p, n, i) in &kept {
        let t = &tris[i * RAW..(i + 1) * RAW];
        feats.extend(p.iter().flatten().map(|c| c / 8.0));
        let zc = p.map(|q| q[2].max(near * 0.5));
        for j in 0..3 {
            feats.push((p[j][0] / zc[j] * h / 160.0).clamp(-4.0, 4.0));
            feats.push((p[j][1] / zc[j] * h / 160.0).clamp(-4.0, 4.0));
        }
        feats.extend(zc.map(|z| (near / z).clamp(0.0, 2.0)));
        let l = dot(*n, *n).sqrt().max(1e-6);
        feats.extend(n.map(|c| c / l));
        feats.extend_from_slice(&t[9..15]);
        feats.extend(t[16..34].iter().map(|c| c * 2.0));
        tex.push(t[15] as i64);
    }
    let cam = [h / 160.0, v[14] / 8.0, v[15] / 8.0, v[19], v[20], v[21], v[22] / 4.0];
    Tokens { feats, tex, cam }
}
