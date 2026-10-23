//! Procedural grid levels: rooms joined by corridors (indoor), or open plazas
//! between extruded blocks (outdoor), plus a scatter of props. Rooms sit at
//! different floor levels and corridors climb between them by ramps or
//! stairs. Lighting is baked into vertex colours, as most PS1 games did.
//!
//! World units: one tile is `T` = 256. The y axis points down (the lowest
//! floor is y = 0, everything above it is negative y), as on the PS1.

use crate::world::rng::{hash, Rng};
use crate::world::{creature, skin, tex};

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
    /// Baked lighting (ambient, sun, room lights) times the surface tint; 128 is neutral.
    pub col: [u8; 3],
    /// Unlit surface tint on the same scale, for light added at render time.
    pub alb: [u8; 3],
}

/// Wound so `(v1 - v0) x (v2 - v0)` points along the surface normal.
#[derive(Clone, Copy, Debug)]
pub struct Tri {
    pub v: [Vert; 3],
    pub tex: u8,
    /// The normal and tint its colours were baked with, to bake them again.
    pub(crate) n: [f32; 3],
    pub(crate) tint: [f32; 3],
}

impl Tri {
    /// The way it faces (world axes, y down): into the open space it bounds.
    pub fn normal(&self) -> [f32; 3] {
        self.n
    }
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

/// One open run of a column of cells, from its floor up to its ceiling. A column
/// can hold several, one above another (a bridge over a pit, a room over a room).
/// Floor and ceiling are each a straight line along `axis` (0 = x, 1 = z): `y0`/`c0`
/// at the column's low edge, `y1`/`c1` at its high edge; equal when flat. Stairs
/// step the floor; the ceiling stays a straight slope.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub room: usize,
    pub y0: i32,
    pub y1: i32,
    pub c0: i32,
    pub c1: i32,
    pub axis: u8,
    pub slope: Slope,
    /// Open to the sky: no ceiling drawn, and the space goes on up.
    pub sky: bool,
    /// Holds a prop or creature, which the walker treats as solid.
    pub blocked: bool,
    /// Kept clear of props and creatures (beside doors).
    pub(crate) reserved: bool,
}

/// A span by column and index: (x, z, k).
pub type Node = (i32, i32, usize);

/// What a column with no open space looks like from outside: block height in
/// tiles (drawn outdoors) and wall texture.
#[derive(Clone, Copy, Debug)]
pub struct Rock {
    pub height: i32,
    pub tex: u8,
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

pub(crate) struct Light {
    pub(crate) pos: [f32; 3],
    pub(crate) col: [f32; 3],
    /// What `col` turns into where the dungeon is strange.
    pub(crate) weird: [f32; 3],
    pub(crate) radius: f32,
}

/// How strange the dungeon is from place to place, 0 (grounded) to 1 (deep): a
/// level-wide base, a slope across the level, a few rooms infected (or spared),
/// and a little noise. Textures morph with it, and lights and fog turn lurid.
#[derive(Clone, Debug)]
pub struct Strange {
    pub base: f32,
    /// Change per world unit along x and z, from `centre`.
    pub(crate) grad: [f32; 2],
    pub(crate) centre: [f32; 2],
    /// Centre (x, z), radius and amount of each local rise or fall.
    pub(crate) bumps: Vec<([f32; 2], f32, f32)>,
    /// Hand-built levels: (layer, value) points of a curve over depth, smooth between them.
    pub(crate) depth: Vec<(f32, f32)>,
    /// Hand-built levels: painted offsets on a lattice one tile apart in x, up and z
    /// (lattice point [i, j, k] sits at x = i·T, y = -j·T, z = k·T).
    pub(crate) points: std::collections::HashMap<[i32; 3], f32>,
    /// Value noise added on top: strength, and features per tile.
    pub(crate) noise: (f32, f32),
    pub(crate) seed: u64,
    /// Fog colour where it is fully strange; RGB 0..1.
    pub(crate) fog: [f32; 3],
    pub(crate) ambient: [f32; 3],
    pub(crate) sun: [f32; 3],
    /// Strangeness where the textures start and finish morphing.
    pub(crate) morph: (f32, f32),
    /// Where the lights, ambient and sun start and finish turning lurid.
    pub(crate) lurid: (f32, f32),
    /// Where the fog starts and finishes taking on `fog`.
    pub(crate) fog_ramp: (f32, f32),
    /// Hand-built levels: areas that add to or pin strangeness inside them.
    pub(crate) areas: Vec<StrangeArea>,
}

/// An area's say in strangeness: its box in world units (y down, so `lo` is the
/// top), fading out over a tile outside it.
#[derive(Clone, Debug)]
pub(crate) struct StrangeArea {
    pub(crate) lo: [f32; 3],
    pub(crate) hi: [f32; 3],
    pub(crate) mode: crate::world::level::AreaStrange,
}

impl StrangeArea {
    /// 1 inside the box and on its faces, falling to 0 a tile outside.
    fn weight(&self, p: [f32; 3]) -> f32 {
        (0..3)
            .map(|a| {
                let out = (self.lo[a] - p[a]).max(p[a] - self.hi[a]).max(0.0);
                1.0 - smoothstep(0.0, TF, out)
            })
            .product()
    }
}

/// Defaults of the look ramps, as generated levels have always used them.
pub(crate) const MORPH: (f32, f32) = (0.0, 1.0);
pub(crate) const LURID: (f32, f32) = (0.15, 0.85);
pub(crate) const FOG_RAMP: (f32, f32) = (0.2, 0.9);

/// Strangeness base forced on every scene generated from now on (the game's depth), as f32 bits; NaN = none.
static FORCE_STRANGE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x7FC0_0000);

/// Make every scene generated from now on this strange at its base (None: random, for training).
pub fn force_strangeness(base: Option<f32>) {
    FORCE_STRANGE.store(base.unwrap_or(f32::NAN).to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Smoothstep over a (start, end) range, or a step where the two meet.
fn ramp((a, b): (f32, f32), x: f32) -> f32 {
    if b - a < 1e-3 {
        return if x >= a { 1.0 } else { 0.0 };
    }
    smoothstep(a, b, x)
}

pub(crate) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A saturated colour of hue `h` (turns) at brightness `v`.
pub fn hue(h: f32, sat: f32, v: f32) -> [f32; 3] {
    [0.0, 1.0 / 3.0, 2.0 / 3.0].map(|o| {
        let k = (0.5 + 0.5 * ((h - o) * std::f32::consts::TAU).cos()).powf(1.5);
        v * (1.0 - sat + sat * k * 1.6)
    })
}

impl Strange {
    fn generate(rng: &mut Rng, w: i32, d: i32, rooms: &[Room], ambient: [f32; 3], sun: [f32; 3]) -> Strange {
        let forced = f32::from_bits(FORCE_STRANGE.load(std::sync::atomic::Ordering::Relaxed));
        let base = if !forced.is_nan() {
            forced
        } else if rng.chance(0.3) {
            rng.fr(0.0, 0.08)
        } else {
            rng.f()
        };
        // A forced base is a depth the player is at: keep the place mostly that strange.
        let vary = if forced.is_nan() { 1.0 } else { 0.5 };
        let a = rng.fr(0.0, std::f32::consts::TAU);
        let slope = vary * rng.fr(0.0, 0.5) / (w.max(d) as f32 * TF);
        let mut bumps = Vec::new();
        if rng.chance(0.45) && !rooms.is_empty() {
            for _ in 0..rng.range(1, 3) {
                let r = &rooms[rng.range(0, rooms.len() as i32) as usize];
                let c = [(r.x as f32 + r.w as f32 / 2.0) * TF, (r.z as f32 + r.d as f32 / 2.0) * TF];
                let amount = if rng.chance(0.25) { -rng.fr(0.3, 0.7) } else { rng.fr(0.3, 0.9) };
                bumps.push((c, rng.fr(1.5, 4.0) * TF, vary * amount));
            }
        }
        let light = |c: [f32; 3]| (c[0] + c[1] + c[2]) / 3.0;
        Strange {
            base,
            grad: [a.cos() * slope, a.sin() * slope],
            centre: [w as f32 * TF / 2.0, d as f32 * TF / 2.0],
            bumps,
            depth: Vec::new(),
            points: Default::default(),
            noise: (0.12, 0.8),
            seed: rng.next() as u64,
            fog: hue(rng.f(), rng.fr(0.7, 1.0), rng.fr(0.35, 0.75)).map(|c| c.min(1.0)),
            ambient: hue(rng.f(), rng.fr(0.6, 0.9), light(ambient).max(0.15) * rng.fr(1.1, 1.6)),
            sun: hue(rng.f(), rng.fr(0.6, 0.9), light(sun) * 1.2),
            morph: MORPH,
            lurid: LURID,
            fog_ramp: FOG_RAMP,
            areas: Vec::new(),
        }
    }

    pub fn at(&self, p: [f32; 3]) -> f32 {
        let (x, z) = (p[0], p[2]);
        let mut s = self.base + self.grad[0] * (x - self.centre[0]) + self.grad[1] * (z - self.centre[1]);
        for &(c, r, k) in &self.bumps {
            let d2 = ((x - c[0]).powi(2) + (z - c[1]).powi(2)) / (r * r);
            s += k * (-d2).exp();
        }
        let (strength, scale) = self.noise;
        s += strength * (tex::value_noise(x / TF * scale, z / TF * scale, 64, self.seed) - 0.5);
        if !self.depth.is_empty() {
            s += curve(&self.depth, -p[1] / STEP as f32);
        }
        if !self.points.is_empty() {
            s += self.lattice([x / TF, -p[1] / TF, z / TF]);
        }
        use crate::world::level::AreaStrange;
        for a in &self.areas {
            if let AreaStrange::Add(v) = a.mode {
                s += v * a.weight(p);
            }
        }
        for a in &self.areas {
            if let AreaStrange::Pin(v) = a.mode {
                s += (v - s) * a.weight(p);
            }
        }
        s.clamp(0.0, 1.0)
    }

    /// The painted lattice at lattice coordinates `q`, smooth between points.
    fn lattice(&self, q: [f32; 3]) -> f32 {
        let i = q.map(|c| c.floor() as i32);
        let f = q.map(|c| smoothstep(0.0, 1.0, c - c.floor()));
        let mut s = 0.0;
        for corner in 0..8 {
            let o = [corner & 1, corner >> 1 & 1, corner >> 2 & 1];
            let w: f32 = (0..3).map(|a| if o[a] == 1 { f[a] } else { 1.0 - f[a] }).product();
            if w > 0.0 {
                s += w * self.points.get(&[i[0] + o[0], i[1] + o[1], i[2] + o[2]]).copied().unwrap_or(0.0);
            }
        }
        s
    }
}

/// Piecewise curve through `(x, y)` points sorted by x, smooth between them and flat beyond.
pub(crate) fn curve(points: &[(f32, f32)], x: f32) -> f32 {
    let Some(i) = points.iter().position(|p| p.0 > x) else { return points.last().map_or(0.0, |p| p.1) };
    if i == 0 {
        return points[0].1;
    }
    let (a, b) = (points[i - 1], points[i]);
    a.1 + (b.1 - a.1) * smoothstep(a.0, b.0, x)
}

pub struct Scene {
    pub w: i32,
    pub d: i32,
    /// Open spans of each column (index z * w + x), top first.
    pub cols: Vec<Vec<Span>>,
    pub rock: Vec<Rock>,
    pub rooms: Vec<Room>,
    pub outdoor: bool,
    /// The level and its props; creatures are kept apart in `creature_tris`.
    pub tris: Vec<Tri>,
    pub creatures: Vec<creature::Creature>,
    /// Dense creature meshes, drawn with the level but never given to the network.
    pub creature_tris: Vec<Tri>,
    /// Each creature's share of `creature_tris`.
    pub creature_spans: Vec<std::ops::Range<usize>>,
    /// Creature skins made for this scene, texture ids from `tex::SKIN_BASE`.
    pub skins: Vec<tex::Texture>,
    pub(crate) skin_kinds: Vec<(skin::Motif, f32)>,
    /// The creatures again for the full-colour pass that draws them over the PS1 frame.
    pub flesh: crate::world::flesh::Flesh,
    pub(crate) sun_dir: [f32; 3],
    pub(crate) sun_col: [f32; 3],
    pub(crate) ambient: [f32; 3],
    pub(crate) lights: Vec<Light>,
    pub strange: Strange,
    /// Hand-built levels: texture slot of single faces, by cell (x, layer, z).
    pub(crate) paint: std::collections::HashMap<([i32; 3], crate::world::level::Face), u8>,
    /// Hand-built levels: doors, keys and levers, drawn from their own parts.
    pub things: crate::world::things::Things,
}

/// Position and orientation of one view. Angles use the GTE's 4096 = 360°.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// Light carried with the camera: RGB strength (0 = none) and reach in world units.
    pub torch: [f32; 3],
    pub torch_radius: i32,
}

pub(crate) fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// 3x3 rotation, row-major: `apply(m, v)` = m · v.
pub(crate) type M3 = [[f32; 3]; 3];

pub(crate) fn apply(m: M3, v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

pub(crate) fn mul(a: M3, b: M3) -> M3 {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

/// Yaw about the vertical axis: local +x turns toward +z as `a` grows.
pub(crate) fn rot_y(a: f32) -> M3 {
    let (s, c) = a.sin_cos();
    [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
}

/// Pitch about local x: swings local +y (down) toward +z (forward) as `a` grows.
pub(crate) fn rot_x(a: f32) -> M3 {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

/// Roll about local z: swings local +y (down) toward -x as `a` grows.
pub(crate) fn rot_z(a: f32) -> M3 {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// A warm, flickery torch: RGB strength around 1.
pub fn torch_colour(rng: &mut Rng) -> [f32; 3] {
    let k = rng.fr(0.8, 1.6);
    [k, k * rng.fr(0.7, 0.85), k * rng.fr(0.45, 0.6)]
}

pub(crate) fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

const FLOOR_TEX: [u8; 5] = [1, 2, 3, 4, 5];
const WALL_TEX: [u8; 6] = [0, 1, 2, 5, 6, 7];
const CEIL_TEX: [u8; 4] = [1, 2, 3, 5];

impl Span {
    /// Position along the span's axis at `p` = (x, z) in column (cx, cz): 0 at its low edge, 1 at its high edge.
    fn along(&self, cx: i32, cz: i32, p: [f32; 2]) -> f32 {
        let t = if self.axis == 0 { p[0] / TF - cx as f32 } else { p[1] / TF - cz as f32 };
        t.clamp(0.0, 1.0)
    }

    /// Floor y at `p` in column (cx, cz); stairs are stepped.
    pub fn floor(&self, cx: i32, cz: i32, p: [f32; 2]) -> f32 {
        let t = self.along(cx, cz, p);
        let k = match self.slope {
            Slope::Flat => 0.0,
            Slope::Ramp => t,
            Slope::Stairs => ((t * 4.0).floor().min(3.0) + 1.0) / 4.0,
        };
        self.y0 as f32 + (self.y1 - self.y0) as f32 * k
    }

    /// Ceiling y at `p` in column (cx, cz): a straight slope even over stairs.
    pub fn ceiling(&self, cx: i32, cz: i32, p: [f32; 2]) -> f32 {
        let t = self.along(cx, cz, p);
        self.c0 as f32 + (self.c1 - self.c0) as f32 * t
    }
}

impl Scene {
    pub fn spans(&self, x: i32, z: i32) -> &[Span] {
        if x < 0 || z < 0 || x >= self.w || z >= self.d {
            &[]
        } else {
            &self.cols[(z * self.w + x) as usize]
        }
    }

    pub fn span(&self, n: Node) -> &Span {
        &self.cols[(n.1 * self.w + n.0) as usize][n.2]
    }

    pub(crate) fn span_mut(&mut self, n: Node) -> &mut Span {
        &mut self.cols[(n.1 * self.w + n.0) as usize][n.2]
    }

    pub fn rock(&self, x: i32, z: i32) -> Rock {
        if x < 0 || z < 0 || x >= self.w || z >= self.d {
            Rock { height: 6, tex: 1 }
        } else {
            self.rock[(z * self.w + x) as usize]
        }
    }

    /// Floor y of span `n` at world point `p` = (x, z).
    pub fn node_floor(&self, n: Node, p: [f32; 2]) -> f32 {
        self.span(n).floor(n.0, n.1, p)
    }

    pub fn node_ceiling(&self, n: Node, p: [f32; 2]) -> f32 {
        self.span(n).ceiling(n.0, n.1, p)
    }

    /// The span someone at height `y` above (x, z) is in: the one whose open space
    /// holds `y` (or whose floor is at most a step above it), nearest floor first,
    /// else the nearest. `None` only where the column is solid.
    pub fn node_at(&self, x: f32, y: f32, z: f32) -> Option<Node> {
        let (cx, cz) = ((x / TF).floor() as i32, (z / TF).floor() as i32);
        let spans = self.spans(cx, cz);
        let miss = |s: &Span| {
            let (f, c) = (s.floor(cx, cz, [x, z]), s.ceiling(cx, cz, [x, z]));
            if s.sky || y >= c - 1.0 {
                if y <= f + MAX_STEP { (f - y).abs() } else { 1e6 + y - f }
            } else {
                1e6 + c - y
            }
        };
        let k = (0..spans.len()).min_by(|&a, &b| miss(&spans[a]).total_cmp(&miss(&spans[b])))?;
        Some((cx, cz, k))
    }

    /// Floor y under (x, z) for someone at height `y`, or `None` inside solid rock.
    pub fn floor_y(&self, x: f32, y: f32, z: f32) -> Option<f32> {
        self.node_at(x, y, z).map(|n| self.node_floor(n, [x, z]))
    }

    /// Ceiling y over (x, z) for someone at height `y` (`None` in rock or under sky).
    pub fn ceiling_y(&self, x: f32, y: f32, z: f32) -> Option<f32> {
        let n = self.node_at(x, y, z)?;
        (!self.span(n).sky).then(|| self.node_ceiling(n, [x, z]))
    }

    /// Floor y under (x, z) if someone at height `y` can stand there: not rock, no prop.
    pub fn walkable_y(&self, x: f32, y: f32, z: f32) -> Option<f32> {
        let n = self.node_at(x, y, z)?;
        (!self.span(n).blocked).then(|| self.node_floor(n, [x, z]))
    }

    /// Can someone step from span `a` to span `b` in the next column? Compares the two
    /// floors on either side of the shared edge, so stairs and ramps pass and ledges
    /// don't, and needs headroom across it.
    pub fn can_step(&self, a: Node, b: Node) -> bool {
        self.rise(a, b).is_some_and(|r| r <= MAX_STEP)
    }

    /// How far the floor climbs or drops across the edge from span `a` to span `b`
    /// in the next column, where there is headroom to cross it at all; `None` where
    /// a prop or a low ceiling is in the way.
    pub fn rise(&self, a: Node, b: Node) -> Option<f32> {
        let (sa, sb) = (self.span(a), self.span(b));
        if sa.blocked || sb.blocked {
            return None;
        }
        let (ax, az) = ((a.0 as f32 + 0.5) * TF, (a.1 as f32 + 0.5) * TF);
        let (dx, dz) = ((b.0 - a.0) as f32, (b.1 - a.1) as f32);
        let (ex, ez) = (ax + dx * TF / 2.0, az + dz * TF / 2.0);
        let (pa, pb) = ([ex - dx, ez - dz], [ex + dx, ez + dz]);
        let (fa, fb) = (sa.floor(a.0, a.1, pa), sb.floor(b.0, b.1, pb));
        let top = |s: &Span, n: Node, p| if s.sky { f32::MIN } else { s.ceiling(n.0, n.1, p) };
        (fa.min(fb) - top(sa, a, pa).max(top(sb, b, pb)) >= 0.75 * TF).then(|| (fa - fb).abs())
    }

    /// Spans one step away from `n`.
    pub fn neighbours(&self, n: Node) -> impl Iterator<Item = Node> + '_ {
        [(1, 0), (-1, 0), (0, 1), (0, -1)].into_iter().flat_map(move |(dx, dz)| {
            let (x, z) = (n.0 + dx, n.1 + dz);
            (0..self.spans(x, z).len()).map(move |k| (x, z, k)).filter(move |&m| self.can_step(n, m))
        })
    }

    /// Breadth-first search from `start`: every reachable span with its parent.
    pub fn reachable(&self, start: Node) -> Vec<(Node, Node)> {
        let mut seen = std::collections::HashSet::from([start]);
        let mut out = vec![(start, start)];
        let mut i = 0;
        while i < out.len() {
            let c = out[i].0;
            for m in self.neighbours(c) {
                if seen.insert(m) {
                    out.push((m, c));
                }
            }
            i += 1;
        }
        out
    }

    /// Every span with a floor, column by column.
    pub fn floor_cells(&self) -> Vec<Node> {
        let mut out = Vec::new();
        for z in 0..self.d {
            for x in 0..self.w {
                out.extend((0..self.spans(x, z).len()).map(|k| (x, z, k)));
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
                weird: [0.0; 3],
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
        let sun_col = [sun, sun * rng.fr(0.85, 1.0), sun * rng.fr(0.7, 1.0)];
        let ambient = [amb * rng.fr(0.8, 1.0), amb * rng.fr(0.8, 1.0), amb];
        // From a stream of its own, so levels keep their layouts for a given seed.
        let mut srng = Rng::new(rng.clone().next() as u64 ^ 0x57A4_6E55);
        let strange = Strange::generate(&mut srng, w, d, &rooms, ambient, sun_col);
        for l in &mut lights {
            l.weird = hue(srng.f(), srng.fr(0.75, 1.0), srng.fr(0.8, 1.5));
        }
        let (cols, rock) = cells
            .iter()
            .map(|&c| match c {
                Cell::Floor { room, height, y0, y1, axis, slope } => {
                    let (c0, c1) = (y0 - height * T, y1 - height * T);
                    let span = Span { room, y0, y1, c0, c1, axis, slope, sky: outdoor, blocked: false, reserved: false };
                    (vec![span], Rock { height: 0, tex: 0 })
                }
                Cell::Solid { height, tex } => (Vec::new(), Rock { height, tex }),
            })
            .unzip();
        let mut scene = Scene {
            w,
            d,
            cols,
            rock,
            rooms,
            outdoor,
            tris: Vec::new(),
            creatures: Vec::new(),
            creature_tris: Vec::new(),
            creature_spans: Vec::new(),
            skins: Vec::new(),
            skin_kinds: Vec::new(),
            flesh: Default::default(),
            sun_dir: norm([rng.fr(-1.0, 1.0), rng.fr(0.6, 1.5), rng.fr(-1.0, 1.0)]),
            sun_col,
            ambient,
            lights,
            strange,
            paint: Default::default(),
            things: Default::default(),
        };
        scene.build_level();
        if !scene.outdoor {
            scene.build_doors(rng);
        }
        scene.build_props(rng);
        scene.build_creatures(rng);
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
    pub(crate) fn shade(&self, p: [i32; 3], n: [f32; 3], tint: [f32; 3]) -> [u8; 3] {
        let pf = p.map(|c| c as f32);
        // Colours turn lurid with strangeness.
        let w = ramp(self.strange.lurid, self.strange.at(pf));
        let mix = |a: [f32; 3], b: [f32; 3]| [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * w);
        let (ambient, sun) = (mix(self.ambient, self.strange.ambient), mix(self.sun_col, self.strange.sun));
        let lambert = dot(n, [-self.sun_dir[0], -self.sun_dir[1], -self.sun_dir[2]]).max(0.0);
        let mut light = [0, 1, 2].map(|c| ambient[c] + sun[c] * lambert);
        for l in &self.lights {
            let to = [l.pos[0] - pf[0], l.pos[1] - pf[1], l.pos[2] - pf[2]];
            let dist = dot(to, to).sqrt();
            let k = dot(n, norm(to)).max(0.0) / (1.0 + (dist / l.radius).powi(2));
            let col = mix(l.col, l.weird);
            for c in 0..3 {
                light[c] += col[c] * k;
            }
        }
        let h = hash((p[0] as u64 & 0xFFFF) | (p[1] as u64 & 0xFFFF) << 16 | (p[2] as u64 & 0xFFFF) << 32);
        let wobble = 0.92 + 0.16 * (h & 0xFF) as f32 / 255.0;
        [0, 1, 2].map(|c| (tint[c] * light[c] * wobble * 128.0).clamp(0.0, 255.0) as u8)
    }

    /// Change how strange the level is at its base, and bake its lights again to match:
    /// the level's vertex colours and the full-colour creatures' part lights. (Creatures
    /// drawn through the PS1 pipeline keep the colours they were made with.)
    pub fn set_strangeness(&mut self, base: f32) {
        self.strange.base = base;
        let mut tris = std::mem::take(&mut self.tris);
        for t in &mut tris {
            for v in &mut t.v {
                v.col = self.shade(v.p, t.n, t.tint);
            }
        }
        self.tris = tris;
        let mut parts = std::mem::take(&mut self.things.parts);
        for p in &mut parts {
            for t in &mut p.tris {
                for v in &mut t.v {
                    v.col = self.shade(v.p, t.n, t.tint);
                }
            }
            p.version += 1;
        }
        self.things.parts = parts;
        let lights: Vec<_> = self.flesh.light_at.iter().map(|&c| self.light_sh(c)).collect();
        self.flesh.lights = lights;
    }

    /// Strangeness at a world point (see `Strange`).
    pub fn strangeness(&self, p: [f32; 3]) -> f32 {
        self.strange.at(p)
    }

    /// How far the textures have morphed at a world point, 0..1 (strangeness
    /// through the level's morph range).
    pub fn morph(&self, p: [f32; 3]) -> f32 {
        let (a, b) = self.strange.morph;
        if (a, b) == MORPH {
            return self.strange.at(p);
        }
        ((self.strange.at(p) - a) / (b - a).max(1e-3)).clamp(0.0, 1.0)
    }

    /// What fog fades to for a camera at `pos`, RGB 0..1: black while grounded,
    /// the level's lurid fog colour where it is strange.
    pub fn fog_colour(&self, pos: [i32; 3]) -> [f32; 3] {
        let w = ramp(self.strange.fog_ramp, self.strange.at(pos.map(|c| c as f32)));
        self.strange.fog.map(|c| c * w)
    }

    /// Baked light around `centre` as linear spherical harmonics, 1 = neutral: the mean
    /// over the six axis directions, then the gradient along world x, y and z.
    pub fn light_sh(&self, centre: [f32; 3]) -> [[f32; 3]; 4] {
        let centre = centre.map(|x| x.round() as i32);
        let light = |n: [f32; 3]| self.shade(centre, n, [1.0; 3]).map(|c| c as f32 / 128.0);
        let axis = |k: usize, s: f32| [0, 1, 2].map(|c| if c == k { s } else { 0.0 });
        let pairs: Vec<_> = (0..3).map(|k| (light(axis(k, 1.0)), light(axis(k, -1.0)))).collect();
        let mut sh = [[0.0; 3]; 4];
        sh[0] = [0, 1, 2].map(|ch| pairs.iter().map(|(a, b)| a[ch] + b[ch]).sum::<f32>() / 6.0);
        for (k, (a, b)) in pairs.iter().enumerate() {
            sh[1 + k] = [0, 1, 2].map(|ch| (a[ch] - b[ch]) / 2.0);
        }
        sh
    }

    /// `uv` is relative to the texture's origin (0..=63).
    pub(crate) fn tri(&mut self, mut v: [([i32; 3], [u8; 2]); 3], n: [f32; 3], tex: u8, tint: [f32; 3]) {
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
        let alb = tint.map(|c| (c * 128.0).clamp(0.0, 255.0) as u8);
        let v = v.map(|(p, uv)| Vert { p, uv: [u0 + uv[0], v0 + uv[1]], col: self.shade(p, n, tint), alb });
        self.tris.push(Tri { v, tex, n, tint });
    }

    /// Quad with corners in "Z" order (tl, tr, bl, br).
    pub(crate) fn quad_uv(&mut self, c: [[i32; 3]; 4], uv: [[u8; 2]; 4], n: [f32; 3], tex: u8, tint: [f32; 3]) {
        self.tri([(c[0], uv[0]), (c[1], uv[1]), (c[2], uv[2])], n, tex, tint);
        self.tri([(c[1], uv[1]), (c[3], uv[3]), (c[2], uv[2])], n, tex, tint);
    }

    /// Quad textured with one full tile.
    pub(crate) fn quad(&mut self, c: [[i32; 3]; 4], n: [f32; 3], tex: u8, tint: [f32; 3]) {
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
        face: Option<((i32, i32), crate::world::level::Face)>,
    ) {
        if (bot[0] - top[0]).max(bot[1] - top[1]) < 1.0 {
            return;
        }
        let (hi, lo) = (top[0].min(top[1]), bot[0].max(bot[1]));
        let (kmin, kmax) = ((-lo / TF).floor() as i32, (-hi / TF).ceil() as i32 - 1);
        for k in kmin..=kmax {
            let (bt, bb) = (-((k + 1) * T) as f32, -(k * T) as f32);
            // Where faces are painted layer by layer, a band splits where the paint changes.
            let mut segs = vec![(bt, bb, tex)];
            if let (Some(((x, z), f)), false) = (face, self.paint.is_empty()) {
                segs.clear();
                let per = T / STEP;
                for l in (k * per..(k + 1) * per).rev() {
                    let t = self.painted([x, l, z], f, tex);
                    let (st, sb) = (-((l + 1) * STEP) as f32, -(l * STEP) as f32);
                    match segs.last_mut() {
                        Some(last) if last.2 == t => last.1 = sb,
                        _ => segs.push((st, sb, t)),
                    }
                }
            }
            for (st, sb, tex) in segs {
                let clip = |e: usize| {
                    let (t, b) = (top[e].max(st), bot[e].min(sb));
                    if t <= b {
                        (t, b)
                    } else if bot[e] < st {
                        (st, st)
                    } else {
                        (sb, sb)
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
    }

    /// A scene with these columns and nothing built yet: no lights, no strangeness.
    pub(crate) fn empty(w: i32, d: i32, rooms: Vec<Room>, cols: Vec<Vec<Span>>, rock: Rock) -> Scene {
        Scene {
            w,
            d,
            cols,
            rock: vec![rock; (w * d) as usize],
            rooms,
            outdoor: false,
            tris: Vec::new(),
            creatures: Vec::new(),
            creature_tris: Vec::new(),
            creature_spans: Vec::new(),
            skins: Vec::new(),
            skin_kinds: Vec::new(),
            flesh: Default::default(),
            sun_dir: [0.0, 1.0, 0.0],
            sun_col: [0.0; 3],
            ambient: [0.2; 3],
            lights: Vec::new(),
            strange: Strange {
                base: 0.0,
                grad: [0.0; 2],
                centre: [0.0; 2],
                bumps: Vec::new(),
                depth: Vec::new(),
                points: Default::default(),
                noise: (0.0, 1.0),
                seed: 0,
                fog: [0.0; 3],
                ambient: [0.2; 3],
                sun: [0.0; 3],
                morph: MORPH,
                lurid: LURID,
                fog_ramp: FOG_RAMP,
                areas: Vec::new(),
            },
            paint: Default::default(),
            things: Default::default(),
        }
    }

    /// Paint of one face, or `default`.
    fn painted(&self, cell: [i32; 3], face: crate::world::level::Face, default: u8) -> u8 {
        if self.paint.is_empty() {
            return default;
        }
        self.paint.get(&(cell, face)).copied().unwrap_or(default)
    }

    pub(crate) fn build_level(&mut self) {
        for z in 0..self.d {
            for x in 0..self.w {
                let n = self.spans(x, z).len();
                if n == 0 && self.outdoor {
                    let touches_floor =
                        [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dz)| !self.spans(x + dx, z + dz).is_empty());
                    if touches_floor {
                        let Rock { height, tex } = self.rock(x, z);
                        let (x0, x1, z0, z1) = (x * T, (x + 1) * T, z * T, (z + 1) * T);
                        let y = -height * T;
                        self.quad([[x0, y, z0], [x1, y, z0], [x0, y, z1], [x1, y, z1]], [0.0, -1.0, 0.0], tex, [0.8; 3]);
                    }
                }
                for k in 0..n {
                    self.build_span((x, z, k));
                }
            }
        }
    }

    fn build_span(&mut self, node: Node) {
        let (x, z, _) = node;
        let span = *self.span(node);
        let mut r = self.rooms[span.room].clone();
        let Span { y0, y1, c0, c1, axis, slope, .. } = span;
        use crate::world::level::Face;
        r.floor_tex = self.painted([x, (-y0.max(y1)).div_euclid(STEP), z], Face::Floor, r.floor_tex);
        r.ceil_tex = self.painted([x, (-c0.max(c1)).div_euclid(STEP) - 1, z], Face::Ceiling, r.ceil_tex);
        let (x0, x1, z0, z1) = ((x * T) as f32, ((x + 1) * T) as f32, (z * T) as f32, ((z + 1) * T) as f32);
        let floor = |p: [f32; 2]| span.floor(x, z, p);
        // Open to the sky reads as a ceiling far overhead.
        const SKY: f32 = -1e9;
        let ceil = |p: [f32; 2]| if span.sky { SKY } else { span.ceiling(x, z, p) };
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
                let y = floor(pt((ta + tb) / 2.0, 0.5));
                let (va, vb) = ((ta * 63.0) as u8, (tb * 63.0) as u8);
                let c = [at(pt(ta, 0.0), y), at(pt(ta, 1.0), y), at(pt(tb, 0.0), y), at(pt(tb, 1.0), y)];
                let uv = if axis == 0 {
                    [[va, 0], [va, 63], [vb, 0], [vb, 63]]
                } else {
                    [[0, va], [63, va], [0, vb], [63, vb]]
                };
                self.quad_uv(c, uv, [0.0, -1.0, 0.0], r.floor_tex, r.tint);
                if k > 0 {
                    let below = floor(pt(ta - 0.01, 0.5));
                    let (top, bot) = (y.min(below), y.max(below));
                    // The riser faces whichever side is lower.
                    let n = if y < below { axis_vec(-1.0, 0.0) } else { axis_vec(1.0, 0.0) };
                    self.wall(pt(ta, 0.0), pt(ta, 1.0), [top; 2], [bot; 2], [0, 63], n, r.wall_tex, r.tint, None);
                }
            }
        } else {
            let c = corners.map(|p| at(p, floor(p)));
            self.quad(c, norm(axis_vec(grad, -1.0)), r.floor_tex, r.tint);
        }
        if !span.sky {
            let c = corners.map(|p| at(p, ceil(p)));
            self.quad(c, norm(axis_vec(-grad, 1.0)), r.ceil_tex, r.tint);
        }

        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (x + dx, z + dz);
            let theirs: Vec<Span> = self.spans(nx, nz).to_vec();
            // Shared edge, and the normal facing back into this cell.
            let (a, b) = match (dx, dz) {
                (1, _) => ([x1, z0], [x1, z1]),
                (-1, _) => ([x0, z1], [x0, z0]),
                (_, 1) => ([x1, z1], [x0, z1]),
                _ => ([x0, z0], [x1, z0]),
            };
            let n = [-dx as f32, 0.0, -dz as f32];
            let face = Some(((x, z), Face::Wall(crate::world::level::Dir::of_delta((dx, dz)))));
            let sloped = span.slope != Slope::Flat || theirs.iter().any(|t| t.slope != Slope::Flat);
            let strips = if sloped { 4 } else { 1 };
            for i in 0..strips {
                let (ta, tb) = (i as f32 / strips as f32, (i + 1) as f32 / strips as f32);
                let (p, q) = (lerp2(a, b, ta), lerp2(a, b, tb));
                // Sample just inside the strip so stair steps read as constant.
                let (pe, qe) = (lerp2(a, b, ta + 0.01), lerp2(a, b, tb - 0.01));
                let u = [(ta * 63.0) as u8, (tb * 63.0) as u8];
                let my_floor = [floor(pe), floor(qe)];
                let my_ceil = [ceil(pe), ceil(qe)];
                let their = |t: &Span| {
                    let c = |p| if t.sky { SKY } else { t.ceiling(nx, nz, p) };
                    ([t.floor(nx, nz, pe), t.floor(nx, nz, qe)], [c(pe), c(qe)])
                };
                // The neighbour's open spans that face ours; rock everywhere else.
                let open: Vec<_> = theirs
                    .iter()
                    .map(their)
                    .filter(|(f, c)| (0..2).any(|e| c[e] < my_floor[e] && f[e] > my_ceil[e]))
                    .collect();
                if open.is_empty() {
                    let Rock { height, tex } = self.rock(nx, nz);
                    let (top, tex) = if span.sky { ([-(height * T) as f32; 2], tex) } else { (my_ceil, r.wall_tex) };
                    self.wall(p, q, top, my_floor, u, n, tex, r.tint, face);
                    continue;
                }
                // Wall wherever theirs is closed: above each open span (a lintel, or the
                // slab between two), and below the last (a riser). Built bottom up.
                // Under the sky, the rock face goes up to the neighbour's rock height.
                let mut pieces = Vec::new();
                let mut cur = if span.sky { [-(self.rock(nx, nz).height * T) as f32; 2] } else { my_ceil };
                for (f, c) in &open {
                    let bot = [0, 1].map(|e| c[e].min(my_floor[e]).max(cur[e]));
                    pieces.push((cur, bot));
                    cur = *f;
                }
                pieces.push(([0, 1].map(|e| cur[e].min(my_floor[e])), my_floor));
                for (top, bot) in pieces.into_iter().rev() {
                    self.wall(p, q, top, bot, u, n, r.wall_tex, r.tint, face);
                }
            }
        }
    }

    /// Box standing at `base` (x, bottom y, z), yawed about the vertical axis.
    pub(crate) fn add_box(&mut self, base: [f32; 3], half: [f32; 2], height: f32, yaw: f32, tex: u8, tint: [f32; 3]) {
        let centre = [base[0], base[1] - height / 2.0, base[2]];
        self.add_obox(centre, rot_y(yaw), [half[0], height / 2.0, half[1]], tex, tint);
    }

    /// Box centred at `centre` with half extents `half` along the columns of `m`.
    pub(crate) fn add_obox(&mut self, centre: [f32; 3], m: M3, half: [f32; 3], tex: u8, tint: [f32; 3]) {
        // Local y = -half is the top (y points down), matching the face table.
        let p = |sx: usize, sy: usize, sz: usize| {
            let l = [
                if sx == 1 { half[0] } else { -half[0] },
                if sy == 1 { -half[1] } else { half[1] },
                if sz == 1 { half[2] } else { -half[2] },
            ];
            let w = apply(m, l);
            [0, 1, 2].map(|c| (centre[c] + w[c]).round() as i32)
        };
        let faces: [([[usize; 3]; 4], [f32; 3]); 6] = [
            ([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1]], [1.0, 0.0, 0.0]),
            ([[0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]], [-1.0, 0.0, 0.0]),
            ([[1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1]], [0.0, 0.0, 1.0]),
            ([[0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]], [0.0, 0.0, -1.0]),
            ([[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]], [0.0, -1.0, 0.0]),
            ([[0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 0, 0]], [0.0, 1.0, 0.0]),
        ];
        for (corners, n) in faces {
            self.quad(corners.map(|[x, y, z]| p(x, y, z)), apply(m, n), tex, tint);
        }
    }

    pub(crate) fn add_pyramid(&mut self, base: [f32; 3], half: f32, height: f32, tex: u8, tint: [f32; 3]) {
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

    /// Reserve cell (cx, cz) of room `ri` for an obstacle, if that keeps the way
    /// through clear: not in a narrow room, not beside a doorway or door, and
    /// every other floor cell still reachable. Marks the cell blocked on success.
    pub(crate) fn claim_cell(&mut self, ri: usize, cx: i32, cz: i32) -> bool {
        let r = &self.rooms[ri];
        let inside = |x: i32, z: i32| x >= r.x && x < r.x + r.w && z >= r.z && z < r.z + r.d;
        let doorway = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            .iter()
            .any(|&(dx, dz)| !inside(cx + dx, cz + dz) && !self.spans(cx + dx, cz + dz).is_empty());
        // Generated levels have one span per column.
        let k = (cx, cz, 0);
        let s = self.span(k);
        if r.w < 3 || r.d < 3 || doorway || s.blocked || s.reserved {
            return false;
        }
        let entry = (r.x + r.w / 2, r.z + r.d / 2, 0);
        let before = self.reachable(entry).len();
        self.span_mut(k).blocked = true;
        if k == entry || self.reachable(entry).len() < before - 1 {
            self.span_mut(k).blocked = false;
            return false;
        }
        true
    }

    fn build_props(&mut self, rng: &mut Rng) {
        for ri in 0..self.rooms.len() {
            let r = self.rooms[ri].clone();
            let count = rng.range(0, if self.outdoor { 7 } else { 5 });
            for _ in 0..count {
                let (cx, cz) = (r.x + rng.range(0, r.w), r.z + rng.range(0, r.d));
                if !self.claim_cell(ri, cx, cz) {
                    continue;
                }
                let (px, pz) = ((cx as f32 + rng.fr(0.25, 0.75)) * TF, (cz as f32 + rng.fr(0.25, 0.75)) * TF);
                let fy = r.floor as f32;
                let ceiling = if self.outdoor { 6.0 } else { r.height as f32 } * TF;
                let tex = rng.next() as u8 % 8;
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
        let n = cells[rng.next() as usize % cells.len()];
        let (px, pz) = ((n.0 as f32 + rng.fr(0.4, 0.6)) * TF, (n.1 as f32 + rng.fr(0.4, 0.6)) * TF);
        let fy = self.node_floor(n, [px, pz]);
        let clearance = if self.span(n).sky { 1.6 * TF } else { fy - self.node_ceiling(n, [px, pz]) };
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
            torch: if rng.chance(0.5) { torch_colour(rng) } else { [0.0; 3] },
            torch_radius: (rng.fr(2.0, 4.0) * TF) as i32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baking the lights again at the same strangeness gives back the colours the level was made with.
    #[test]
    fn set_strangeness_rebakes_the_same_colours() {
        crate::world::flesh::set_psx(true);
        for seed in 0..20 {
            let mut scene = Scene::generate(&mut Rng::new(seed));
            let before: Vec<[u8; 3]> = scene.tris.iter().flat_map(|t| t.v.map(|v| v.col)).collect();
            scene.set_strangeness(scene.strange.base);
            let after: Vec<[u8; 3]> = scene.tris.iter().flat_map(|t| t.v.map(|v| v.col)).collect();
            let worst = before.iter().zip(&after).flat_map(|(a, b)| (0..3).map(move |c| a[c].abs_diff(b[c]))).max().unwrap_or(0);
            assert!(worst == 0, "seed {seed}: colours moved by up to {worst}");
        }
    }
}
