//! Rays from the pointer into the 3D view, and what they hit: the level's
//! triangles as drawn, turned into the cell and face of the level they belong
//! to. View space is in tiles with y up (see `view3d::to_view`).

use crate::view3d::{to_view, to_world, EditorCam, TF};
use bevy::prelude::*;
use bevy_egui::egui;
use engine::world::level::{Dir, Face, Level};
use engine::world::scene::{Scene, STEP};

/// One layer in view units (tiles).
pub const LAYER: f32 = STEP as f32 / TF;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,
}

/// The ray through `p` (egui points) for the 3D view filling `screen`.
pub fn ray(cam: &EditorCam, screen: egui::Rect, p: egui::Pos2) -> Ray {
    let half = (engine::render::fov(engine::world::walker::H) / 2.0).tan();
    let aspect = screen.width() / screen.height();
    let nx = (p.x - screen.left()) / screen.width() * 2.0 - 1.0;
    let ny = 1.0 - (p.y - screen.top()) / screen.height() * 2.0;
    let d = cam.rotation() * Vec3::new(nx * half * aspect, ny * half, -1.0);
    Ray { o: cam.pos, d: d.normalize() }
}

/// Where `v` (view space) lands on `screen`, if it is in front of the camera.
pub fn project(cam: &EditorCam, screen: egui::Rect, v: Vec3) -> Option<egui::Pos2> {
    let half = (engine::render::fov(engine::world::walker::H) / 2.0).tan();
    let aspect = screen.width() / screen.height();
    let c = cam.rotation().inverse() * (v - cam.pos);
    if c.z > -0.01 {
        return None;
    }
    let (nx, ny) = (c.x / -c.z / (half * aspect), c.y / -c.z / half);
    Some(egui::pos2(screen.left() + (nx + 1.0) / 2.0 * screen.width(), screen.top() + (1.0 - ny) / 2.0 * screen.height()))
}

/// Where the ray meets the plane through `p0` facing `n`.
pub fn on_plane(r: Ray, p0: Vec3, n: Vec3) -> Option<Vec3> {
    let den = r.d.dot(n);
    if den.abs() < 1e-6 {
        return None;
    }
    let t = (p0 - r.o).dot(n) / den;
    (t > 0.0).then(|| r.o + r.d * t)
}

/// How far along the line `p0 + s·u` (u a unit vector) the point nearest the ray is.
pub fn along_line(r: Ray, p0: Vec3, u: Vec3) -> Option<f32> {
    let w = p0 - r.o;
    let b = u.dot(r.d);
    let den = 1.0 - b * b;
    if den < 1e-4 {
        return None;
    }
    Some((b * r.d.dot(w) - u.dot(w)) / den)
}

/// A face of the level under the pointer: the open cell (x, layer, z) it bounds,
/// which side of that cell it is, and the point hit (view space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub cell: [i32; 3],
    pub face: Face,
    pub point: Vec3,
}

/// The nearest front-facing triangle of `scene` along the ray, as a face of `level`.
/// A floor's cell is its run's bottom layer and a ceiling's its top one, so sloped
/// floors and stair treads count as the floor of their column.
pub fn pick(scene: &Scene, level: &Level, r: Ray) -> Option<Hit> {
    let mut best: Option<(f32, [f32; 3])> = None;
    for t in &scene.tris {
        let n = t.normal();
        // View space flips y and z, so the normal does too.
        let nv = Vec3::new(n[0], -n[1], -n[2]);
        if nv.dot(r.d) >= 0.0 {
            continue;
        }
        let [a, b, c] = t.v.map(|v| to_view(v.p.map(|c| c as f32)));
        if let Some(d) = hit_tri(r, a, b, c) {
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, n));
            }
        }
    }
    let (d, n) = best?;
    let point = r.o + r.d * d;
    Some(face_at(level, point, n))
}

/// The face through `point` (view space) facing `n` (world axes).
pub fn face_at(level: &Level, point: Vec3, n: [f32; 3]) -> Hit {
    let w = to_world(point);
    // A step into the open side picks the cell.
    let f = [w[0] + n[0] * 2.0, w[1] + n[1] * 2.0, w[2] + n[2] * 2.0];
    let mut cell = [(f[0] / TF).floor() as i32, (-f[1] / STEP as f32).floor() as i32, (f[2] / TF).floor() as i32];
    let face = if n[1] < -0.5 {
        Face::Floor
    } else if n[1] > 0.5 {
        Face::Ceiling
    } else if n[0].abs() > n[2].abs() {
        // The rock is on the side the normal points away from.
        Face::Wall(if n[0] < 0.0 { Dir::E } else { Dir::W })
    } else {
        Face::Wall(if n[2] < 0.0 { Dir::S } else { Dir::N })
    };
    if let Some(run) = level.run_at(cell[0], cell[1], cell[2]) {
        match face {
            Face::Floor => cell[1] = run.from,
            Face::Ceiling => cell[1] = run.to - 1,
            Face::Wall(_) => {}
        }
    }
    Hit { cell, face, point }
}

/// Distance along the ray to triangle `abc` (Möller–Trumbore), either side.
fn hit_tri(r: Ray, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let (e1, e2) = (b - a, c - a);
    let p = r.d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    let s = r.o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = r.d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 1e-4).then_some(t)
}

/// The plane a face lies in: a point on it and its normal, view space, facing
/// into the open cell. A sloped floor counts as flat at its low edge.
pub fn plane(cell: [i32; 3], face: Face) -> (Vec3, Vec3) {
    let [x, l, z] = cell.map(|c| c as f32);
    // The cell's lowest north-west corner, in view space.
    let corner = Vec3::new(x, l * LAYER, -z);
    match face {
        Face::Floor => (corner, Vec3::Y),
        Face::Ceiling => (corner + Vec3::Y * LAYER, Vec3::NEG_Y),
        Face::Wall(d) => {
            let (dx, dz) = d.delta();
            let n = Vec3::new(-dx as f32, 0.0, dz as f32);
            let centre = Vec3::new(x + 0.5, l * LAYER, -(z + 0.5));
            (centre - n * 0.5, n)
        }
    }
}

/// The cell whose `face` lies where the ray meets the plane of `face` at `cell`.
pub fn cell_on_plane(r: Ray, cell: [i32; 3], face: Face) -> Option<[i32; 3]> {
    let (p0, n) = plane(cell, face);
    let p = on_plane(r, p0, n)?;
    // Half a cell into the open side, then which cell that is.
    let into = match face {
        Face::Floor | Face::Ceiling => n * LAYER * 0.5,
        Face::Wall(_) => n * 0.5,
    };
    let w = to_world(p + into);
    Some([(w[0] / TF).floor() as i32, (-w[1] / STEP as f32).floor() as i32, (w[2] / TF).floor() as i32])
}

/// The four corners of a face (view space), lifted a little into the open cell
/// so lines drawn round it show. Sloped floors tilt with their slope.
pub fn quad(level: &Level, cell: [i32; 3], face: Face) -> [Vec3; 4] {
    const LIFT: f32 = 0.015;
    let [x, l, z] = cell.map(|c| c as f32);
    let p = |x: f32, layers: f32, z: f32| Vec3::new(x, layers * LAYER, -z);
    match face {
        Face::Floor | Face::Ceiling => {
            let floor = face == Face::Floor;
            let y = if floor { l } else { l + 1.0 };
            let lift = if floor { LIFT } else { -LIFT };
            // How far up a sloped floor or a ceiling that follows it is at (u, v) across the tile.
            let run = level.run_at(cell[0], cell[1], cell[2]);
            let slope = run.and_then(|r| r.slope).filter(|s| floor || s.ceiling == engine::world::level::CeilingKind::Follow);
            let up = |u: f32, v: f32| match slope {
                None => 0.0,
                Some(s) => {
                    let t = match s.dir {
                        Dir::E => u,
                        Dir::W => 1.0 - u,
                        Dir::S => v,
                        Dir::N => 1.0 - v,
                    };
                    t * s.rise as f32
                }
            };
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(u, v)| p(x + u, y + up(u, v), z + v) + Vec3::Y * lift)
        }
        Face::Wall(d) => {
            let (dx, dz) = d.delta();
            let (cx, cz) = (x + 0.5 + dx as f32 * (0.5 - LIFT), z + 0.5 + dz as f32 * (0.5 - LIFT));
            // Along the wall, half a tile each way.
            let (ax, az) = (dz.abs() as f32 * 0.5, dx.abs() as f32 * 0.5);
            [
                p(cx - ax, l, cz - az),
                p(cx + ax, l, cz + az),
                p(cx + ax, l + 1.0, cz + az),
                p(cx - ax, l + 1.0, cz - az),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn down_at(x: f32, z: f32) -> Ray {
        Ray { o: Vec3::new(x, 20.0, -z), d: Vec3::NEG_Y }
    }

    #[test]
    fn picks_floors_through_ceilings_and_rock() {
        engine::world::flesh::set_psx(true);
        let level = Level::demo();
        let scene = level.to_scene();
        // Down into the lower hall: through rock and the hall's ceiling to its floor.
        let h = pick(&scene, &level, down_at(1.5, 1.5)).unwrap();
        assert_eq!((h.cell, h.face), ([1, 0, 1], Face::Floor));
        assert!((h.point.y).abs() < 1e-3);
        // Into the pit beside the bridge: its floor is 8 layers down.
        let h = pick(&scene, &level, down_at(2.5, 1.5)).unwrap();
        assert_eq!((h.cell, h.face), ([2, -8, 1], Face::Floor));
        // A stair column counts as the floor of its run.
        let h = pick(&scene, &level, down_at(7.5, 2.5)).unwrap();
        assert_eq!((h.cell, h.face), ([7, 4, 2], Face::Floor));
    }

    #[test]
    fn picks_walls_and_ceilings() {
        engine::world::flesh::set_psx(true);
        let level = Level::demo();
        let scene = level.to_scene();
        // From inside the hall, looking west at its west wall.
        let r = Ray { o: Vec3::new(3.0, 1.0, -1.5), d: Vec3::NEG_X };
        let h = pick(&scene, &level, r).unwrap();
        assert_eq!((h.cell, h.face), ([0, 4, 1], Face::Wall(Dir::W)));
        // And up at its ceiling.
        let r = Ray { o: Vec3::new(1.5, 1.0, -1.5), d: Vec3::Y };
        let h = pick(&scene, &level, r).unwrap();
        assert_eq!((h.cell, h.face), ([1, 11, 1], Face::Ceiling));
    }

    #[test]
    fn rays_and_projection_agree() {
        let cam = EditorCam { pos: Vec3::new(1.0, 2.0, 3.0), yaw: 0.4, pitch: -0.3, focus: 5.0 };
        let screen = egui::Rect::from_min_size(egui::pos2(0.0, 20.0), egui::vec2(800.0, 500.0));
        let p = egui::pos2(612.0, 133.0);
        let r = ray(&cam, screen, p);
        let back = project(&cam, screen, r.o + r.d * 7.0).unwrap();
        assert!((back - p).length() < 0.01, "{back:?}");
    }

    #[test]
    fn plane_cells_match_faces() {
        let r = Ray { o: Vec3::new(4.3, 1.0, -2.6), d: Vec3::new(0.0, -1.0, 0.0) };
        assert_eq!(cell_on_plane(r, [0, 0, 0], Face::Floor), Some([4, 0, 2]));
        let r = Ray { o: Vec3::new(0.5, 0.4, -6.5), d: Vec3::NEG_X };
        assert_eq!(cell_on_plane(r, [0, 0, 0], Face::Wall(Dir::W)), Some([0, 1, 6]));
    }
}
