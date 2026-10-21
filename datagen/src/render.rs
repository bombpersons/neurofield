//! Renders a scene the way a PS1 game would: GTE RTPT projection, NCLIP
//! back-face culling, AVSZ3 into an ordering table (no depth buffer), and
//! Gouraud-shaded, affine-textured, dithered triangles through psx-gpu.
//! Triangles near the camera are subdivided (as many games did) so that
//! near-plane rejection leaves small gaps rather than whole missing walls.
//! Light from a torch carried with the camera is added per vertex, then fog
//! darkens vertex colours by screen depth, fading to black.

use crate::scene::{Camera, Scene, Vert};
use crate::tex;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{Env, Poly, PolyVertex, Prim, Rgb, Texture, Uv, Xy};
pub use psx_gpu::Gpu;
use psx_gte::math::{mul_matrix, rot_x, rot_y, rot_z};
use psx_gte::{Gte, Matrix, Vec3i16, Vec3i32};

pub const W: u32 = 320;
pub const H: u32 = 240;
/// Triangles with a vertex closer than this are split into four, up to `MAX_SPLIT` times.
const SPLIT_Z: u16 = 384;
const MAX_SPLIT: u32 = 3;
const OT_LEN: usize = 2048;
/// Sign of NCLIP for triangles wound along their outward normal.
const FRONT: i32 = -1;

/// World → view rotation (1.3.12). Row 2 is the view direction in world space.
pub fn view_matrix(cam: &Camera) -> Matrix {
    mul_matrix(&rot_z(cam.roll), &mul_matrix(&rot_x(cam.pitch), &rot_y(cam.yaw)))
}

pub fn new_gpu(textures: &[tex::Texture]) -> Gpu {
    let mut gpu = Gpu::new();
    let mut words = Vec::new();
    for p in tex::uploads(textures) {
        p.encode(&mut words);
    }
    gpu.gp0_all(&words);
    gpu
}

fn mid(a: &Vert, b: &Vert) -> Vert {
    Vert {
        p: [0, 1, 2].map(|c| (a.p[c] + b.p[c]) / 2),
        uv: [0, 1].map(|c| ((a.uv[c] as u16 + b.uv[c] as u16) / 2) as u8),
        col: [0, 1, 2].map(|c| ((a.col[c] as u16 + b.col[c] as u16) / 2) as u8),
        alb: [0, 1, 2].map(|c| ((a.alb[c] as u16 + b.alb[c] as u16) / 2) as u8),
    }
}

struct Frame<'a> {
    gte: Gte,
    ot: OrderingTable,
    cam: &'a Camera,
}

impl Frame<'_> {
    /// Light from the torch carried with the camera: albedo × colour × falloff × Lambert.
    fn torch(&self, v: &Vert, n: [f32; 3]) -> [f32; 3] {
        let to = [0, 1, 2].map(|c| (self.cam.pos[c] - v.p[c]) as f32);
        let dist = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]).sqrt().max(1.0);
        let fall = (1.0 - dist / self.cam.torch_radius.max(1) as f32).max(0.0).powi(2);
        let lambert = ((n[0] * to[0] + n[1] * to[1] + n[2] * to[2]) / dist).max(0.0);
        [0, 1, 2].map(|c| v.alb[c] as f32 * self.cam.torch[c] * fall * lambert)
    }

    fn fog(&self, sz: u16) -> f32 {
        let t = (sz as f32 - self.cam.fog_near as f32) / (self.cam.fog_far - self.cam.fog_near).max(1) as f32;
        1.0 - t.clamp(0.0, 1.0)
    }

    fn tri(&mut self, v: [Vert; 3], tex: u8, depth: u32) {
        let r = self.gte.rtpt(v.map(|v| Vec3i16::new(v.p[0] as i16, v.p[1] as i16, v.p[2] as i16)));
        if r.sz.iter().all(|&z| z as i32 > self.cam.fog_far) {
            return;
        }
        let near = r.sz.iter().any(|&z| z < SPLIT_Z);
        let offscreen = r.sxy.iter().any(|s| s.x.abs() >= 1023 || s.y.abs() >= 1023);
        if depth < MAX_SPLIT && (near || offscreen) {
            let (ab, bc, ca) = (mid(&v[0], &v[1]), mid(&v[1], &v[2]), mid(&v[2], &v[0]));
            self.tri([v[0], ab, ca], tex, depth + 1);
            self.tri([ab, v[1], bc], tex, depth + 1);
            self.tri([ca, bc, v[2]], tex, depth + 1);
            self.tri([ab, bc, ca], tex, depth + 1);
            return;
        }
        // Below SZ = H/2 the GTE's divide saturates and vertices project to the wrong
        // place, so games used H/2 as the near plane; anything crossing it is dropped.
        if r.sz.iter().any(|&z| z < self.cam.h / 2) || offscreen {
            return;
        }
        if self.gte.nclip() * FRONT <= 0 {
            return;
        }
        let otz = self.gte.avsz3();
        // Triangles are wound along their outward normal.
        let e1 = [0, 1, 2].map(|c| (v[1].p[c] - v[0].p[c]) as f32);
        let e2 = [0, 1, 2].map(|c| (v[2].p[c] - v[0].p[c]) as f32);
        let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let len = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt().max(1e-6);
        let n = cr.map(|c| c / len);
        let verts = (0..3)
            .map(|i| {
                let f = self.fog(r.sz[i]);
                let t = self.torch(&v[i], n);
                let c = [0, 1, 2].map(|k| ((v[i].col[k] as f32 + t[k]) * f).min(255.0) as u8);
                PolyVertex {
                    xy: Xy(r.sxy[i].x, r.sxy[i].y),
                    color: Rgb(c[0], c[1], c[2]),
                    uv: Uv(v[i].uv[0], v[i].uv[1]),
                }
            })
            .collect();
        let texture = Some(Texture { page: tex::page(), clut: tex::clut(tex), raw: false });
        self.ot.add(otz as i32, &Prim::Poly(Poly { verts, gouraud: true, texture, semi: false }));
    }
}

/// Draw one view and return the 320x240 frame as RGB8.
pub fn render(gpu: &mut Gpu, scene: &Scene, cam: &Camera) -> Vec<u8> {
    let rt = view_matrix(cam);
    let mut gte = Gte::new();
    gte.set_rotation(&rt);
    let tr = [0, 1, 2].map(|r| -((0..3).map(|k| rt.m[r][k] as i64 * cam.pos[k] as i64).sum::<i64>() >> 12) as i32);
    gte.set_translation(Vec3i32::from_array(tr));
    gte.set_screen_offset(W as i32 / 2, H as i32 / 2);
    gte.set_projection_distance(cam.h);
    // OTZ = mean SZ / 4, so the 2048-entry table covers depths up to 8192.
    gte.set_z_scale(0x1000 / 3 / 4, 0x1000 / 4 / 4);

    let mut frame = Frame { gte, ot: OrderingTable::new(OT_LEN), cam };
    for t in &scene.tris {
        frame.tri(t.v, t.tex, 0);
    }

    let mut words = Vec::new();
    for env in [
        Env::TexPage { page: tex::page(), dither: true, draw_to_display: true },
        Env::DrawArea { x1: 0, y1: 0, x2: W as u16 - 1, y2: H as u16 - 1 },
        Env::DrawOffset { x: 0, y: 0 },
        Env::Fill { x: 0, y: 0, w: W as u16, h: H as u16, color: Rgb(0, 0, 0) },
    ] {
        env.encode(&mut words);
    }
    gpu.gp0_all(&words);
    frame.ot.draw(gpu);
    gpu.vram.rgba15(0, 0, W, H).chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
}
