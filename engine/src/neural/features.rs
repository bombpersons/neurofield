//! What the network is told about a scene and a view, as floats: per level
//! triangle, per creature part and per view. The training data stores exactly
//! these (`tris.npy`, `parts.npy`, `views.npy`; see `FORMAT` in
//! training-data-gen) and `tokens` builds the network's inputs from them.

use crate::world::scene::{Camera, Scene, T};
use crate::world::{tex, view};

/// Floats per triangle in `tris.npy`, per creature part in `parts.npy` and per
/// view in `views.npy`.
pub const TRI_FEATURES: usize = 35;
pub const PART_FEATURES: usize = 49;
pub const VIEW_FEATURES: usize = 26;

/// Per-triangle features as written to `tris.npy`, appended to `out`: the
/// level's, then those of its doors, keys and levers (`thing_tri_features`).
pub fn tri_features(scene: &Scene, out: &mut Vec<f32>) {
    for t in &scene.tris {
        one_tri(scene, t, out);
    }
    thing_tri_features(scene, out);
}

/// Features of the triangles of the scene's moving things alone: these come
/// last, so a door's move only redoes this tail.
pub fn thing_tri_features(scene: &Scene, out: &mut Vec<f32>) {
    for t in scene.part_tris() {
        one_tri(scene, t, out);
    }
}

fn one_tri(scene: &Scene, t: &crate::world::scene::Tri, out: &mut Vec<f32>) {
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
    let centre = [0, 1, 2].map(|c| t.v.iter().map(|v| v.p[c] as f32).sum::<f32>() / 3.0);
    out.push(scene.strangeness(centre));
}

/// Per-part features of the scene's creatures as written to `parts.npy`, appended to `out`.
pub fn part_features(scene: &Scene, out: &mut Vec<f32>) {
    let mut first = 0;
    for (ci, c) in scene.creatures.iter().enumerate() {
        for (p, at) in c.parts.iter().zip(c.pose()) {
            let tiles = |v: [f32; 3]| v.map(|x| x / T as f32);
            out.extend(tiles(at.joint));
            out.extend(tiles(at.centre));
            out.extend(at.axes.iter().flatten());
            out.extend(tiles(p.half));
            let (lo, hi) = p.bounds();
            out.extend(tiles(lo));
            out.extend(tiles(hi));
            out.extend([p.shape as u8 as f32, p.taper, p.bend, p.bulge, p.lumps]);
            let skin = &c.skins[p.skin];
            out.extend([skin.motif as u8 as f32, skin.density, skin.scale / T as f32]);
            out.extend(skin.tint);
            // Baked light at the centre as linear spherical harmonics: mean, then gradient along x, y, z.
            out.extend(scene.light_sh(at.centre).iter().flatten());
            out.push(p.parent.map_or(-1.0, |i| (first + i) as f32));
            out.push(ci as f32);
        }
        first += c.parts.len();
    }
}

/// Per-view features as written to `views.npy`, appended to `out`.
pub fn view_features(scene: &Scene, scene_index: usize, cam: &Camera, out: &mut Vec<f32>) {
    out.push(scene_index as f32);
    out.extend(cam.pos.map(|c| c as f32 / T as f32));
    let m = view::view_matrix(cam);
    out.extend(m.m.iter().flatten().map(|&e| e as f32 / 4096.0));
    out.push(cam.h as f32);
    out.push(cam.fog_near as f32 / T as f32);
    out.push(cam.fog_far as f32 / T as f32);
    out.extend([cam.yaw, cam.pitch, cam.roll].map(|a| a as f32 / 4096.0));
    out.extend(cam.torch);
    out.push(cam.torch_radius as f32 / T as f32);
    out.extend(scene.fog_colour(cam.pos));
}
