//! Procedural grid levels: rooms joined by corridors (indoor), or open plazas
//! between extruded blocks (outdoor), plus a scatter of props. Rooms sit at
//! different floor levels and corridors climb between them by ramps or
//! stairs. Lighting is baked into vertex colours, as most PS1 games did.
//!
//! World units: one tile is `T` = 256. The y axis points down (the lowest
//! floor is y = 0, everything above it is negative y), as on the PS1.

use crate::rng::{hash, Rng};
use crate::tex;

pub const T: i32 = 256;
const TF: f32 = T as f32;
/// Floor levels are multiples of this (a quarter tile).
pub const STEP: i32 = 64;
/// Highest floor change someone can step up or down.
pub const MAX_STEP: f32 = 80.0;

#[derive(Clone, Copy, Debug)]
pub struct Vert {
    pub p: [i32; 3],
    /// Texel coordinates inside the texture page.
    pub uv: [u8; 2],
    pub col: [u8; 3],
}

/// Wound so `(v1 - v0) x (v2 - v0)` points along the surface normal.
#[derive(Clone, Copy, Debug)]
pub struct Tri {
    pub v: [Vert; 3],
    pub tex: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slope {
    Flat,
    /// Straight incline from `y0` to `y1`.
    Ramp,
    /// Four steps from `y0` to `y1`.
    Stairs,
}

#[derive(Clone, Copy, Debug)]
pub enum Cell {
    /// Block height in tiles (only drawn outdoors) and its wall texture.
    Solid { height: i32, tex: u8 },
    /// `y0`/`y1` are the floor heights at the low/high edge along `axis`
    /// (0 = x, 1 = z); equal for flat cells. `height` is the ceiling clearance in tiles.
    Floor { room: usize, height: i32, y0: i32, y1: i32, axis: u8, slope: Slope },
}

impl Cell {
    fn flat(room: usize, height: i32, y: i32) -> Cell {
        Cell::Floor { room, height, y0: y, y1: y, axis: 0, slope: Slope::Flat }
    }
}

#[derive(Clone, Debug)]
pub struct Room {
    pub x: i32,
    pub z: i32,
    pub w: i32,
    pub d: i32,
    pub height: i32,
    /// Floor level (y of the floor, 0 or negative).
    pub floor: i32,
    pub floor_tex: u8,
    pub wall_tex: u8,
    pub ceil_tex: u8,
    pub tint: [f32; 3],
}

struct Light {
    pos: [f32; 3],
    col: [f32; 3],
    radius: f32,
}

pub struct Scene {
    pub w: i32,
    pub d: i32,
    pub cells: Vec<Cell>,
    pub rooms: Vec<Room>,
    pub outdoor: bool,
    pub tris: Vec<Tri>,
    /// Cells holding a prop, which the walker treats as solid.
    pub blocked: Vec<bool>,
    sun_dir: [f32; 3],
    sun_col: [f32; 3],
    ambient: [f32; 3],
    lights: Vec<Light>,
}

/// Position and orientation of one view. Angles use the GTE's 4096 = 360°.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: [i32; 3],
    pub yaw: i32,
    pub pitch: i32,
    pub roll: i32,
    /// GTE projection distance H in pixels (focal length; 160 ≈ 90° horizontal FOV).
    /// Geometry nearer than H/2 is not drawn.
    pub h: u16,
    pub fog_near: i32,
    pub fog_far: i32,
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

const FLOOR_TEX: [u8; 5] = [1, 2, 3, 4, 5];
const WALL_TEX: [u8; 6] = [0, 1, 2, 5, 6, 7];
const CEIL_TEX: [u8; 4] = [1, 2, 3, 5];

impl Scene {
    pub fn cell(&self, x: i32, z: i32) -> Cell {
        if x < 0 || z < 0 || x >= self.w || z >= self.d {
            Cell::Solid { height: 6, tex: 1 }
        } else {
            self.cells[(z * self.w + x) as usize]
        }
    }

    /// Position along a sloped cell's axis, 0 at its low edge and 1 at its high edge.
    fn along(cx: i32, cz: i32, axis: u8, p: [f32; 2]) -> f32 {
        let t = if axis == 0 { p[0] / TF - cx as f32 } else { p[1] / TF - cz as f32 };
        t.clamp(0.0, 1.0)
    }

    /// Floor y of cell (cx, cz) at world point `p` = (x, z); stairs are stepped.
    fn cell_floor(&self, cx: i32, cz: i32, p: [f32; 2]) -> Option<f32> {
        match self.cell(cx, cz) {
            Cell::Floor { y0, y1, axis, slope, .. } => {
                let t = Self::along(cx, cz, axis, p);
                let k = match slope {
                    Slope::Flat => 0.0,
                    Slope::Ramp => t,
                    Slope::Stairs => ((t * 4.0).floor().min(3.0) + 1.0) / 4.0,
                };
                Some(y0 as f32 + (y1 - y0) as f32 * k)
            }
            Cell::Solid { .. } => None,
        }
    }

    /// Ceiling y of cell (cx, cz) at `p`: a straight slope even over stairs.
    fn cell_ceiling(&self, cx: i32, cz: i32, p: [f32; 2]) -> Option<f32> {
        match self.cell(cx, cz) {
            Cell::Floor { y0, y1, axis, height, .. } => {
                let t = Self::along(cx, cz, axis, p);
                Some(y0 as f32 + (y1 - y0) as f32 * t - (height * T) as f32)
            }
            Cell::Solid { .. } => None,
        }
    }

    /// Floor y under world point (x, z), or `None` inside solid cells.
    pub fn floor_y(&self, x: f32, z: f32) -> Option<f32> {
        self.cell_floor((x / TF).floor() as i32, (z / TF).floor() as i32, [x, z])
    }

    /// Floor y under (x, z) if someone can stand there: not solid, no prop.
    pub fn walkable_y(&self, x: f32, z: f32) -> Option<f32> {
        let (cx, cz) = ((x / TF).floor() as i32, (z / TF).floor() as i32);
        if cx < 0 || cz < 0 || cx >= self.w || cz >= self.d || self.blocked[(cz * self.w + cx) as usize] {
            return None;
        }
        self.cell_floor(cx, cz, [x, z])
    }

    /// Can someone step from cell `a` to its neighbour `b`? Compares the two
    /// floors on either side of the shared edge, so stairs and ramps pass and ledges don't.
    pub fn can_step(&self, a: (i32, i32), b: (i32, i32)) -> bool {
        let (ax, az) = ((a.0 as f32 + 0.5) * TF, (a.1 as f32 + 0.5) * TF);
        let (dx, dz) = ((b.0 - a.0) as f32, (b.1 - a.1) as f32);
        let (ex, ez) = (ax + dx * TF / 2.0, az + dz * TF / 2.0);
        match (self.walkable_y(ex - dx, ez - dz), self.walkable_y(ex + dx, ez + dz)) {
            (Some(fa), Some(fb)) => (fa - fb).abs() <= MAX_STEP,
            _ => false,
        }
    }

    /// Breadth-first search from `start`: every reachable cell with its parent.
    pub fn reachable(&self, start: (i32, i32)) -> Vec<((i32, i32), (i32, i32))> {
        let mut seen = vec![false; (self.w * self.d) as usize];
        let mut out = vec![(start, start)];
        seen[(start.1 * self.w + start.0) as usize] = true;
        let mut i = 0;
        while i < out.len() {
            let c = out[i].0;
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let n = (c.0 + dx, c.1 + dz);
                if n.0 < 0 || n.1 < 0 || n.0 >= self.w || n.1 >= self.d {
                    continue;
                }
                let k = (n.1 * self.w + n.0) as usize;
                if !seen[k] && self.can_step(c, n) {
                    seen[k] = true;
                    out.push((n, c));
                }
            }
            i += 1;
        }
        out
    }

    pub fn floor_cells(&self) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        for z in 0..self.d {
            for x in 0..self.w {
                if let Cell::Floor { .. } = self.cell(x, z) {
                    out.push((x, z));
                }
            }
        }
        out
    }

    pub fn generate(rng: &mut Rng) -> Scene {
        let (w, d) = (rng.range(8, 16), rng.range(8, 16));
        let outdoor = rng.chance(0.25);
        let mut cells: Vec<Cell> =
            (0..w * d).map(|_| Cell::Solid { height: rng.range(2, 7), tex: rng.pick(&WALL_TEX) }).collect();
        let mut rooms: Vec<Room> = Vec::new();
        for _ in 0..rng.range(2, 6) {
            // Rooms never overlap, so a corridor's planned levels are never overwritten.
            let Some((x, z, rw, rd)) = (0..20).find_map(|_| {
                let rw = rng.range(2, 7).min(w - 2);
                let rd = rng.range(2, 7).min(d - 2);
                let (x, z) = (rng.range(1, w - rw), rng.range(1, d - rd));
                let clear = (z..z + rd)
                    .all(|cz| (x..x + rw).all(|cx| matches!(cells[(cz * w + cx) as usize], Cell::Solid { .. })));
                clear.then_some((x, z, rw, rd))
            }) else {
                break;
            };
            let base = rng.fr(0.65, 1.0);
            let mut room = Room {
                x,
                z,
                w: rw,
                d: rd,
                height: rng.range(2, 5),
                floor: if rng.chance(0.3) { 0 } else { -STEP * rng.range(0, 5) },
                floor_tex: rng.pick(&FLOOR_TEX),
                wall_tex: rng.pick(&WALL_TEX),
                ceil_tex: rng.pick(&CEIL_TEX),
                tint: [0, 1, 2].map(|_| (base + rng.fr(-0.2, 0.15)).clamp(0.35, 1.0)),
            };
            if let Some(prev) = rooms.last() {
                room.floor = Self::carve_corridor(&mut cells, w, rng, rooms.len() - 1, prev, &room);
            }
            for cz in z..z + rd {
                for cx in x..x + rw {
                    cells[(cz * w + cx) as usize] = Cell::flat(rooms.len(), room.height, room.floor);
                }
            }
            rooms.push(room);
        }

        let mut lights = Vec::new();
        for r in &rooms {
            if rng.chance(0.2) {
                continue;
            }
            let warm = rng.f();
            lights.push(Light {
                pos: [
                    (r.x as f32 + r.w as f32 * rng.fr(0.3, 0.7)) * TF,
                    r.floor as f32 - (r.height as f32 - 0.3) * TF,
                    (r.z as f32 + r.d as f32 * rng.fr(0.3, 0.7)) * TF,
                ],
                col: [1.0, 0.75 + 0.2 * warm, 0.5 + 0.4 * warm].map(|c| c * rng.fr(0.6, 1.3)),
                radius: rng.fr(1.5, 4.0) * TF,
            });
        }
        let sun = if outdoor { rng.fr(0.5, 0.9) } else { rng.fr(0.0, 0.25) };
        let amb = if outdoor { rng.fr(0.25, 0.45) } else { rng.fr(0.12, 0.35) };
        let mut scene = Scene {
            w,
            d,
            cells,
            rooms,
            outdoor,
            tris: Vec::new(),
            blocked: vec![false; (w * d) as usize],
            sun_dir: norm([rng.fr(-1.0, 1.0), rng.fr(0.6, 1.5), rng.fr(-1.0, 1.0)]),
            sun_col: [sun, sun * rng.fr(0.85, 1.0), sun * rng.fr(0.7, 1.0)],
            ambient: [amb * rng.fr(0.8, 1.0), amb * rng.fr(0.8, 1.0), amb],
            lights,
        };
        scene.build_level();
        scene.build_props(rng);
        scene
    }

    /// L-shaped corridor from room `a`'s centre toward room `b` (not yet written),
    /// in `a`'s style. Each run of new cells climbs (up to two steps per straight
    /// cell, by ramp or stairs) to the level of whatever it meets next; the last
    /// run aims for `b`'s floor as far as it can. Returns the floor `b` must use.
    fn carve_corridor(cells: &mut [Cell], w: i32, rng: &mut Rng, ai: usize, a: &Room, b: &Room) -> i32 {
        let (ax, az) = (a.x + a.w / 2, a.z + a.d / 2);
        let (bx, bz) = (b.x + b.w / 2, b.z + b.d / 2);
        let (sx, sz) = ((bx - ax).signum(), (bz - az).signum());
        // (x, z, travel direction, is the turn)
        let mut path = Vec::new();
        let mut cx = ax;
        loop {
            path.push((cx, az, (sx, 0), cx == bx && az != bz));
            if cx == bx {
                break;
            }
            cx += sx;
        }
        let mut cz = az;
        while cz != bz {
            cz += sz;
            path.push((bx, cz, (0, sz), false));
        }
        // The rest of the path is inside `b`, which is convex.
        let in_b = |x: i32, z: i32| x >= b.x && x < b.x + b.w && z >= b.z && z < b.z + b.d;
        if let Some(end) = path.iter().position(|&(x, z, ..)| in_b(x, z)) {
            path.truncate(end);
        }
        // Level of an existing floor cell, or None for a cell still to carve.
        let level = |c: Cell| match c {
            Cell::Floor { y0, y1, .. } => Some((y0 + y1) / 2 / STEP * STEP),
            Cell::Solid { .. } => None,
        };
        let climbs = |&(_, _, d, turn): &(i32, i32, (i32, i32), bool)| !turn && d != (0, 0);
        let height = a.height.min(2);
        let slope = if rng.chance(0.6) { Slope::Stairs } else { Slope::Ramp };
        let mut cur = a.floor;
        let mut i = 0;
        while i < path.len() {
            let (x, z, ..) = path[i];
            if let Some(y) = level(cells[(z * w + x) as usize]) {
                cur = y;
                i += 1;
                continue;
            }
            let j = (i..path.len()).find(|&k| level(cells[(path[k].1 * w + path[k].0) as usize]).is_some());
            let cap = path[i..j.unwrap_or(path.len())].iter().filter(|c| climbs(c)).count() as i32 * 2 * STEP;
            let target = match j {
                Some(k) => level(cells[(path[k].1 * w + path[k].0) as usize]).unwrap(),
                None => b.floor.clamp(cur - cap, cur + cap),
            };
            for c in &path[i..j.unwrap_or(path.len())] {
                let (x, z, (dx, dz), _) = *c;
                let rise = (target - cur).clamp(-2 * STEP, 2 * STEP);
                cells[(z * w + x) as usize] = if rise == 0 || !climbs(c) {
                    Cell::flat(ai, height, cur)
                } else {
                    let (y0, y1) = if dx + dz > 0 { (cur, cur + rise) } else { (cur + rise, cur) };
                    cur += rise;
                    Cell::Floor { room: ai, height, y0, y1, axis: if dx != 0 { 0 } else { 1 }, slope }
                };
            }
            i = j.unwrap_or(path.len());
        }
        cur
    }

    /// Baked vertex colour: ambient + sun + point lights, times tint, with a
    /// small per-position wobble so shared vertices stay consistent.
    fn shade(&self, p: [i32; 3], n: [f32; 3], tint: [f32; 3]) -> [u8; 3] {
        let pf = p.map(|c| c as f32);
        let lambert = dot(n, [-self.sun_dir[0], -self.sun_dir[1], -self.sun_dir[2]]).max(0.0);
        let mut light = [0, 1, 2].map(|c| self.ambient[c] + self.sun_col[c] * lambert);
        for l in &self.lights {
            let to = [l.pos[0] - pf[0], l.pos[1] - pf[1], l.pos[2] - pf[2]];
            let dist = dot(to, to).sqrt();
            let k = dot(n, norm(to)).max(0.0) / (1.0 + (dist / l.radius).powi(2));
            for c in 0..3 {
                light[c] += l.col[c] * k;
            }
        }
        let h = hash((p[0] as u64 & 0xFFFF) | (p[1] as u64 & 0xFFFF) << 16 | (p[2] as u64 & 0xFFFF) << 32);
        let wobble = 0.92 + 0.16 * (h & 0xFF) as f32 / 255.0;
        [0, 1, 2].map(|c| (tint[c] * light[c] * wobble * 128.0).clamp(0.0, 255.0) as u8)
    }

    /// `uv` is relative to the texture's origin (0..=63).
    fn tri(&mut self, mut v: [([i32; 3], [u8; 2]); 3], n: [f32; 3], tex: u8, tint: [f32; 3]) {
        let e1 = [0, 1, 2].map(|c| (v[1].0[c] - v[0].0[c]) as f32);
        let e2 = [0, 1, 2].map(|c| (v[2].0[c] - v[0].0[c]) as f32);
        let cross = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        if dot(cross, cross) < 1.0 {
            return;
        }
        if dot(cross, n) < 0.0 {
            v.swap(1, 2);
        }
        let (u0, v0) = tex::uv_origin(tex);
        let v = v.map(|(p, uv)| Vert { p, uv: [u0 + uv[0], v0 + uv[1]], col: self.shade(p, n, tint) });
        self.tris.push(Tri { v, tex });
    }

    /// Quad with corners in "Z" order (tl, tr, bl, br).
    fn quad_uv(&mut self, c: [[i32; 3]; 4], uv: [[u8; 2]; 4], n: [f32; 3], tex: u8, tint: [f32; 3]) {
        self.tri([(c[0], uv[0]), (c[1], uv[1]), (c[2], uv[2])], n, tex, tint);
        self.tri([(c[1], uv[1]), (c[3], uv[3]), (c[2], uv[2])], n, tex, tint);
    }

    /// Quad textured with one full tile.
    fn quad(&mut self, c: [[i32; 3]; 4], n: [f32; 3], tex: u8, tint: [f32; 3]) {
        self.quad_uv(c, [[0, 0], [63, 0], [0, 63], [63, 63]], n, tex, tint);
    }

    /// Vertical wall over the horizontal segment p→q, from `top` down to `bot`
    /// at each end. Split into one-tile bands so the texture stays
    /// world-aligned vertically; `u` is the texture u at p and q.
    #[allow(clippy::too_many_arguments)]
    fn wall(
        &mut self,
        p: [f32; 2],
        q: [f32; 2],
        top: [f32; 2],
        bot: [f32; 2],
        u: [u8; 2],
        n: [f32; 3],
        tex: u8,
        tint: [f32; 3],
    ) {
        if (bot[0] - top[0]).max(bot[1] - top[1]) < 1.0 {
            return;
        }
        let (hi, lo) = (top[0].min(top[1]), bot[0].max(bot[1]));
        let (kmin, kmax) = ((-lo / TF).floor() as i32, (-hi / TF).ceil() as i32 - 1);
        for k in kmin..=kmax {
            let (bt, bb) = (-((k + 1) * T) as f32, -(k * T) as f32);
            let clip = |e: usize| {
                let (t, b) = (top[e].max(bt), bot[e].min(bb));
                if t <= b {
                    (t, b)
                } else if bot[e] < bt {
                    (bt, bt)
                } else {
                    (bb, bb)
                }
            };
            let ((t0, b0), (t1, b1)) = (clip(0), clip(1));
            if t0 >= b0 && t1 >= b1 {
                continue;
            }
            let v = |y: f32| ((y - bt) / TF * 63.0).round().clamp(0.0, 63.0) as u8;
            let at = |s: [f32; 2], y: f32| [s[0].round() as i32, y.round() as i32, s[1].round() as i32];
            self.quad_uv(
                [at(p, t0), at(q, t1), at(p, b0), at(q, b1)],
                [[u[0], v(t0)], [u[1], v(t1)], [u[0], v(b0)], [u[1], v(b1)]],
                n,
                tex,
                tint,
            );
        }
    }

    fn build_level(&mut self) {
        for z in 0..self.d {
            for x in 0..self.w {
                match self.cell(x, z) {
                    Cell::Floor { room, .. } => self.build_floor_cell(x, z, room),
                    Cell::Solid { height, tex } if self.outdoor => {
                        let touches_floor = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                            .iter()
                            .any(|&(dx, dz)| matches!(self.cell(x + dx, z + dz), Cell::Floor { .. }));
                        if touches_floor {
                            let (x0, x1, z0, z1) = (x * T, (x + 1) * T, z * T, (z + 1) * T);
                            let y = -height * T;
                            self.quad(
                                [[x0, y, z0], [x1, y, z0], [x0, y, z1], [x1, y, z1]],
                                [0.0, -1.0, 0.0],
                                tex,
                                [0.8; 3],
                            );
                        }
                    }
                    Cell::Solid { .. } => {}
                }
            }
        }
    }

    fn build_floor_cell(&mut self, x: i32, z: i32, room: usize) {
        let r = self.rooms[room].clone();
        let Cell::Floor { y0, y1, axis, slope, .. } = self.cell(x, z) else { return };
        let (x0, x1, z0, z1) = ((x * T) as f32, ((x + 1) * T) as f32, (z * T) as f32, ((z + 1) * T) as f32);
        let floor = |s: &Self, p: [f32; 2]| s.cell_floor(x, z, p).unwrap();
        let ceil = |s: &Self, p: [f32; 2]| s.cell_ceiling(x, z, p).unwrap();
        let at = |p: [f32; 2], y: f32| [p[0].round() as i32, y.round() as i32, p[1].round() as i32];
        let corners = [[x0, z0], [x1, z0], [x0, z1], [x1, z1]];
        // Slope of the floor line along the axis, for normals.
        let grad = (y1 - y0) as f32 / TF;
        let axis_vec = |g: f32, y: f32| if axis == 0 { [g, y, 0.0] } else { [0.0, y, g] };

        if slope == Slope::Stairs {
            // Four treads with risers between them.
            for k in 0..4 {
                let (ta, tb) = (k as f32 / 4.0, (k + 1) as f32 / 4.0);
                let pt = |t: f32, side: f32| {
                    if axis == 0 {
                        [x0 + t * TF, z0 + side * TF]
                    } else {
                        [x0 + side * TF, z0 + t * TF]
                    }
                };
                let y = floor(self, pt((ta + tb) / 2.0, 0.5));
                let (va, vb) = ((ta * 63.0) as u8, (tb * 63.0) as u8);
                let c = [at(pt(ta, 0.0), y), at(pt(ta, 1.0), y), at(pt(tb, 0.0), y), at(pt(tb, 1.0), y)];
                let uv = if axis == 0 {
                    [[va, 0], [va, 63], [vb, 0], [vb, 63]]
                } else {
                    [[0, va], [63, va], [0, vb], [63, vb]]
                };
                self.quad_uv(c, uv, [0.0, -1.0, 0.0], r.floor_tex, r.tint);
                if k > 0 {
                    let below = floor(self, pt(ta - 0.01, 0.5));
                    let (top, bot) = (y.min(below), y.max(below));
                    // The riser faces whichever side is lower.
                    let n = if y < below { axis_vec(-1.0, 0.0) } else { axis_vec(1.0, 0.0) };
                    self.wall(pt(ta, 0.0), pt(ta, 1.0), [top; 2], [bot; 2], [0, 63], n, r.wall_tex, r.tint);
                }
            }
        } else {
            let c = corners.map(|p| at(p, floor(self, p)));
            self.quad(c, norm(axis_vec(grad, -1.0)), r.floor_tex, r.tint);
        }
        if !self.outdoor {
            let c = corners.map(|p| at(p, ceil(self, p)));
            self.quad(c, norm(axis_vec(-grad, 1.0)), r.ceil_tex, r.tint);
        }

        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (x + dx, z + dz);
            let neighbour = self.cell(nx, nz);
            // Shared edge, and the normal facing back into this cell.
            let (a, b) = match (dx, dz) {
                (1, _) => ([x1, z0], [x1, z1]),
                (-1, _) => ([x0, z1], [x0, z0]),
                (_, 1) => ([x1, z1], [x0, z1]),
                _ => ([x0, z0], [x1, z0]),
            };
            let n = [-dx as f32, 0.0, -dz as f32];
            let sloped = |c: Cell| matches!(c, Cell::Floor { slope, .. } if slope != Slope::Flat);
            let strips = if sloped(self.cell(x, z)) || sloped(neighbour) { 4 } else { 1 };
            for i in 0..strips {
                let (ta, tb) = (i as f32 / strips as f32, (i + 1) as f32 / strips as f32);
                let (p, q) = (lerp2(a, b, ta), lerp2(a, b, tb));
                // Sample just inside the strip so stair steps read as constant.
                let (pe, qe) = (lerp2(a, b, ta + 0.01), lerp2(a, b, tb - 0.01));
                let u = [(ta * 63.0) as u8, (tb * 63.0) as u8];
                let my_floor = [floor(self, pe), floor(self, qe)];
                let my_ceil = [ceil(self, pe), ceil(self, qe)];
                match neighbour {
                    Cell::Solid { height, tex } => {
                        let (top, tex) =
                            if self.outdoor { ([-(height * T) as f32; 2], tex) } else { (my_ceil, r.wall_tex) };
                        self.wall(p, q, top, my_floor, u, n, tex, r.tint);
                    }
                    Cell::Floor { .. } => {
                        let their_floor = [self.cell_floor(nx, nz, pe).unwrap(), self.cell_floor(nx, nz, qe).unwrap()];
                        // Riser where the neighbour's floor is higher (smaller y).
                        let top = [their_floor[0].min(my_floor[0]), their_floor[1].min(my_floor[1])];
                        self.wall(p, q, top, my_floor, u, n, r.wall_tex, r.tint);
                        if !self.outdoor {
                            // Lintel where the neighbour's ceiling is lower (larger y).
                            let their_ceil =
                                [self.cell_ceiling(nx, nz, pe).unwrap(), self.cell_ceiling(nx, nz, qe).unwrap()];
                            let bot = [their_ceil[0].max(my_ceil[0]), their_ceil[1].max(my_ceil[1])];
                            self.wall(p, q, my_ceil, bot, u, n, r.wall_tex, r.tint);
                        }
                    }
                }
            }
        }
    }

    /// Box standing at `base` (x, bottom y, z), yawed about the vertical axis.
    fn add_box(&mut self, base: [f32; 3], half: [f32; 2], height: f32, yaw: f32, tex: u8, tint: [f32; 3]) {
        let (s, c) = yaw.sin_cos();
        let p = |sx: usize, sy: usize, sz: usize| {
            let (lx, lz) = (if sx == 1 { half[0] } else { -half[0] }, if sz == 1 { half[1] } else { -half[1] });
            let y = base[1] - if sy == 1 { height } else { 0.0 };
            [(base[0] + lx * c - lz * s).round() as i32, y.round() as i32, (base[2] + lx * s + lz * c).round() as i32]
        };
        let rot = |n: [f32; 3]| [n[0] * c - n[2] * s, n[1], n[0] * s + n[2] * c];
        let faces: [([[usize; 3]; 4], [f32; 3]); 6] = [
            ([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1]], [1.0, 0.0, 0.0]),
            ([[0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]], [-1.0, 0.0, 0.0]),
            ([[1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1]], [0.0, 0.0, 1.0]),
            ([[0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]], [0.0, 0.0, -1.0]),
            ([[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]], [0.0, -1.0, 0.0]),
            ([[0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 0, 0]], [0.0, 1.0, 0.0]),
        ];
        for (corners, n) in faces {
            self.quad(corners.map(|[x, y, z]| p(x, y, z)), rot(n), tex, tint);
        }
    }

    fn add_pyramid(&mut self, base: [f32; 3], half: f32, height: f32, tex: u8, tint: [f32; 3]) {
        let at = |dx: f32, dz: f32| [(base[0] + dx).round() as i32, base[1] as i32, (base[2] + dz).round() as i32];
        let corners = [at(-half, -half), at(half, -half), at(half, half), at(-half, half)];
        let apex = [base[0] as i32, (base[1] - height) as i32, base[2] as i32];
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            let mid = [(a[0] + b[0]) as f32 / 2.0 - base[0], 0.0, (a[2] + b[2]) as f32 / 2.0 - base[2]];
            let n = norm([mid[0] * height / half, -half, mid[2] * height / half]);
            self.tri([(a, [0, 63]), (b, [63, 63]), (apex, [31, 0])], n, tex, tint);
        }
        self.quad([corners[0], corners[1], corners[3], corners[2]], [0.0, 1.0, 0.0], tex, tint);
    }

    fn build_props(&mut self, rng: &mut Rng) {
        for ri in 0..self.rooms.len() {
            let r = self.rooms[ri].clone();
            let count = rng.range(0, if self.outdoor { 7 } else { 5 });
            for _ in 0..count {
                let (cx, cz) = (r.x + rng.range(0, r.w), r.z + rng.range(0, r.d));
                // Keep the way through clear: no props in narrow rooms or beside a doorway.
                let inside = |x: i32, z: i32| x >= r.x && x < r.x + r.w && z >= r.z && z < r.z + r.d;
                let doorway = [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dz)| {
                    !inside(cx + dx, cz + dz) && matches!(self.cell(cx + dx, cz + dz), Cell::Floor { .. })
                });
                let k = (cz * self.w + cx) as usize;
                if r.w < 3 || r.d < 3 || doorway || self.blocked[k] {
                    continue;
                }
                // Only block the cell if every other floor cell stays reachable.
                let entry = (r.x + r.w / 2, r.z + r.d / 2);
                let before = self.reachable(entry).len();
                self.blocked[k] = true;
                if (cx, cz) == entry || self.reachable(entry).len() < before - 1 {
                    self.blocked[k] = false;
                    continue;
                }
                let (px, pz) = ((cx as f32 + rng.fr(0.25, 0.75)) * TF, (cz as f32 + rng.fr(0.25, 0.75)) * TF);
                let fy = r.floor as f32;
                let ceiling = if self.outdoor { 6.0 } else { r.height as f32 } * TF;
                let tex = rng.next() as u8 % tex::N as u8;
                let tint = [0, 1, 2].map(|_| rng.fr(0.5, 1.0));
                match rng.range(0, 20) {
                    0..=8 => {
                        let half = [rng.fr(0.12, 0.3) * TF, rng.fr(0.12, 0.3) * TF];
                        let h = (rng.fr(0.3, 1.3) * TF).min(ceiling - 8.0);
                        self.add_box([px, fy, pz], half, h, rng.fr(0.0, 6.283), tex, tint);
                    }
                    9..=12 if !self.outdoor => {
                        let s = rng.fr(0.12, 0.25) * TF;
                        self.add_box([px, fy, pz], [s, s], ceiling, 0.0, tex, tint);
                    }
                    9..=15 => {
                        let half = rng.fr(0.2, 0.3) * TF;
                        let h = (rng.fr(0.4, 1.5) * TF).min(ceiling - 8.0);
                        self.add_pyramid([px, fy, pz], half, h, tex, tint);
                    }
                    _ => {
                        let s = rng.fr(0.1, 0.3) * TF;
                        let lift = rng.fr(0.5, 1.6) * TF;
                        if lift + 2.0 * s < ceiling {
                            self.add_box([px, fy - lift, pz], [s, s], 2.0 * s, rng.fr(0.0, 6.283), tex, tint);
                        }
                    }
                }
            }
        }
    }

    /// A free-floating camera anywhere on the floor: varied height, tilt and lens.
    pub fn random_camera(&self, rng: &mut Rng) -> Camera {
        let cells = self.floor_cells();
        let (x, z) = cells[rng.next() as usize % cells.len()];
        let (px, pz) = ((x as f32 + rng.fr(0.4, 0.6)) * TF, (z as f32 + rng.fr(0.4, 0.6)) * TF);
        let fy = self.floor_y(px, pz).unwrap();
        let clearance = self.cell_ceiling(x, z, [px, pz]).map_or(1.6 * TF, |c| fy - c);
        let top = if self.outdoor { 1.6 } else { (clearance / TF - 0.3).min(1.4) };
        let fog_near = rng.fr(1.5, 4.0);
        Camera {
            pos: [px as i32, (fy - rng.fr(0.4, top) * TF) as i32, pz as i32],
            yaw: rng.range(0, 4096),
            pitch: rng.range(-280, 280),
            roll: if rng.chance(0.8) { 0 } else { rng.range(-150, 150) },
            h: rng.range(110, 180) as u16,
            fog_near: (fog_near * TF) as i32,
            fog_far: ((fog_near + rng.fr(3.0, 10.0)) * TF) as i32,
        }
    }
}
