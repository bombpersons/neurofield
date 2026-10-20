//! Procedural PS1-style levels, a first-person walker and the PS1 rasteriser,
//! shared by the `datagen` binary and the realtime viewer.

pub mod creature;
pub mod door;
pub mod npy;
pub mod render;
pub mod rng;
pub mod scene;
pub mod tex;
pub mod walker;

use rng::Rng;
use scene::{Camera, Scene, T};
use walker::Walker;

/// Floats per triangle in `tris.npy` and per view in `views.npy` (see `FORMAT` in main.rs).
pub const TRI_FEATURES: usize = 34;
pub const VIEW_FEATURES: usize = 23;

/// A level the first-person walker can move around in, and the walker.
pub fn walkable_scene(rng: &mut Rng, min_cells: usize) -> (Scene, Walker) {
    loop {
        let scene = Scene::generate(rng);
        if scene.floor_cells().len() < min_cells {
            continue;
        }
        if let Some(w) = Walker::new(&scene, rng) {
            return (scene, w);
        }
    }
}

/// Per-triangle features as written to `tris.npy`, appended to `out`.
pub fn tri_features(scene: &Scene, out: &mut Vec<f32>) {
    for t in &scene.tris {
        for v in &t.v {
            out.extend(v.p.map(|c| c as f32 / T as f32));
        }
        let (u0, v0) = tex::uv_origin(t.tex);
        for v in &t.v {
            out.push((v.uv[0] - u0) as f32 / tex::SIZE as f32);
            out.push((v.uv[1] - v0) as f32 / tex::SIZE as f32);
        }
        out.push(t.tex as f32);
        for v in &t.v {
            out.extend(v.col.map(|c| c as f32 / 255.0));
        }
        for v in &t.v {
            out.extend(v.alb.map(|c| c as f32 / 255.0));
        }
    }
}

/// Per-view features as written to `views.npy`, appended to `out`.
pub fn view_features(scene_index: usize, cam: &Camera, out: &mut Vec<f32>) {
    out.push(scene_index as f32);
    out.extend(cam.pos.map(|c| c as f32 / T as f32));
    let m = render::view_matrix(cam);
    out.extend(m.m.iter().flatten().map(|&e| e as f32 / 4096.0));
    out.push(cam.h as f32);
    out.push(cam.fog_near as f32 / T as f32);
    out.push(cam.fog_far as f32 / T as f32);
    out.extend([cam.yaw, cam.pitch, cam.roll].map(|a| a as f32 / 4096.0));
    out.extend(cam.torch);
    out.push(cam.torch_radius as f32 / T as f32);
}
