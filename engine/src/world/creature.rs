//! Creatures as trees of parts: bodies, limbs, heads and whatever else grows
//! on them. Each part is a closed surface around its local y axis (ellipsoid,
//! capsule, cone or tentacle) with a size, a few shape dials and a skin (a
//! `skin::Motif` with a density, scale and tint). Datagen draws them as dense
//! meshes with freshly made skins; the network only gets one token per part
//! (`part_features` in lib.rs), so whatever detail it paints is its own.
//!
//! World units (256 per tile), y down. A part's local -y end is usually where
//! it joins its parent and +y its far end, like a hanging limb. A creature's
//! own frame faces +z.

use crate::world::rng::Rng;
use crate::world::scene::{apply, dot, mul, norm, rot_x, rot_y, rot_z, Scene, Tri, Vert, M3, T};
use crate::world::skin::{self, Motif};
use crate::world::tex;
use std::cell::RefCell;
use std::f32::consts::{PI, TAU};

const TF: f32 = T as f32;
const IDENT: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// Target mesh edge length in world units.
const EDGE: f32 = 5.0;

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f32; 3], k: f32) -> [f32; 3] {
    a.map(|c| c * k)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Ellipsoid,
    Capsule,
    Cone,
    Tentacle,
}

pub const SHAPES: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct Skin {
    pub motif: Motif,
    /// 0..1: how many eyes, teeth, scales... per texture repeat.
    pub density: f32,
    /// World units per texture repeat.
    pub scale: f32,
    /// Surface colour multiplier, 1 = neutral.
    pub tint: [f32; 3],
    /// Picks the texture's layout; never shown to the network.
    pub seed: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Part {
    /// An earlier part, or None for the root.
    pub parent: Option<usize>,
    /// Where the joint sits on the parent: along its axis (-1..1), around it in
    /// turns (0 = its +x, 0.25 = +z) and how far out toward its surface (0..1).
    /// For the root, an offset from the creature's base in the creature's frame.
    pub anchor: [f32; 3],
    /// Orientation relative to the parent's frame (the creature's for the root).
    pub rot: M3,
    /// The joint in this part's own frame, as fractions of `half`: [0, -1, 0] is its -y end.
    pub pivot: [f32; 3],
    /// Radii along local x and z, and half the length along y.
    pub half: [f32; 3],
    pub shape: Shape,
    /// Radius at the +y end relative to the -y end.
    pub taper: f32,
    /// Sideways curl toward local +x at the far end, in half-lengths.
    pub bend: f32,
    /// Extra girth in the middle, as a fraction.
    pub bulge: f32,
    /// Height of random bumps as a fraction of the radius; their layout is never shown to the network.
    pub lumps: f32,
    /// Index into the creature's skins.
    pub skin: usize,
    pub seed: u32,
}

impl Part {
    fn new(shape: Shape, half: [f32; 3], skin: usize) -> Part {
        Part {
            parent: None,
            anchor: [0.0; 3],
            rot: IDENT,
            pivot: [0.0, -1.0, 0.0],
            half,
            shape,
            taper: 1.0,
            bend: 0.0,
            bulge: 0.0,
            lumps: 0.0,
            skin,
            seed: 0,
        }
    }

    fn on(mut self, parent: usize, t: f32, around: f32, out: f32) -> Part {
        self.parent = Some(parent);
        self.anchor = [t, around, out];
        self
    }

    fn root(mut self, offset: [f32; 3]) -> Part {
        self.anchor = offset;
        self
    }

    fn rot(mut self, m: M3) -> Part {
        self.rot = m;
        self
    }

    fn pivot(mut self, p: [f32; 3]) -> Part {
        self.pivot = p;
        self
    }

    fn taper(mut self, k: f32) -> Part {
        self.taper = k;
        self
    }

    fn bend(mut self, k: f32) -> Part {
        self.bend = k;
        self
    }

    fn bulge(mut self, k: f32) -> Part {
        self.bulge = k;
        self
    }

    fn lumps(mut self, k: f32) -> Part {
        self.lumps = k;
        self
    }

    /// Radius along the axis (1 = `half`) at t in -1..1, before lumps. Every shape closes at both ends.
    pub(crate) fn profile(&self, t: f32) -> f32 {
        // Rounds an end off over the last `k` of the length.
        let cap = |e: f32, k: f32| (e / k).clamp(0.0, 1.0).sqrt();
        let r = match self.shape {
            Shape::Ellipsoid => (1.0 - t * t).max(0.0).sqrt(),
            Shape::Capsule => {
                let e = (t.abs() - 0.7) / 0.3;
                if e <= 0.0 {
                    1.0
                } else {
                    (1.0 - e * e).max(0.0).sqrt()
                }
            }
            Shape::Cone => (1.0 - t) / 2.0 * cap(t + 1.0, 0.15),
            Shape::Tentacle => {
                (1.0 - 0.8 * (t + 1.0) / 2.0) * (1.0 + 0.12 * (t * 6.0 * PI).sin()) * cap(t + 1.0, 0.1) * cap(1.0 - t, 0.1)
            }
        };
        r * (1.0 + (self.taper - 1.0) * (t + 1.0) / 2.0) * (1.0 + self.bulge * (1.0 - t * t))
    }

    /// The largest radius `profile` reaches.
    pub(crate) fn widest(&self) -> f32 {
        (0..=32).map(|k| self.profile(k as f32 / 16.0 - 1.0)).fold(1e-3, f32::max)
    }

    /// Point at `t` along the axis and angle `a` (radians) around it, `out` of the
    /// way from the axis to the surface, in the part's frame.
    fn local(&self, t: f32, a: f32, out: f32) -> [f32; 3] {
        let r = self.profile(t) * out;
        let s = (t + 1.0) / 2.0;
        [r * self.half[0] * a.cos() + self.bend * self.half[1] * s * s, t * self.half[1], r * self.half[2] * a.sin()]
    }

    /// The axis point at `t` (where the surface's radius is measured from).
    fn axis(&self, t: f32) -> [f32; 3] {
        self.local(t, 0.0, 0.0)
    }

    /// Outward surface normal at (t, a), in the part's frame.
    fn normal(&self, t: f32, a: f32) -> [f32; 3] {
        let (e, t) = (1e-3, t.clamp(-0.98, 0.98));
        let dt = sub(self.local(t + e, a, 1.0), self.local(t - e, a, 1.0));
        let da = sub(self.local(t, a + e, 1.0), self.local(t, a - e, 1.0));
        let n = cross(da, dt);
        if dot(n, n) < 1e-12 {
            return [0.0, t.signum(), 0.0];
        }
        let n = norm(n);
        if dot(n, sub(self.local(t, a, 1.0), self.axis(t))) < 0.0 {
            scale(n, -1.0)
        } else {
            n
        }
    }

    /// Smooth bumps in -1..1, periodic around the axis.
    pub(crate) fn lump_waves(&self) -> [[f32; 4]; 3] {
        let mut rng = Rng::new(self.seed as u64);
        [0; 3].map(|_| [rng.range(1, 6) as f32, rng.fr(1.0, 4.0), rng.fr(0.0, TAU), rng.fr(0.0, TAU)])
    }

    /// Box around the part in its own frame (lumps included): (lo, hi).
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for k in 0..=16 {
            for j in 0..16 {
                let p = self.local(k as f32 / 8.0 - 1.0, j as f32 / 16.0 * TAU, 1.0 + self.lumps);
                for c in 0..3 {
                    lo[c] = lo[c].min(p[c]);
                    hi[c] = hi[c].max(p[c]);
                }
            }
        }
        (lo, hi)
    }
}

/// The body plans a creature can grow from, numbered as the generator rolls them (0..20).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Kind {
    Humanoid = 0,
    Beast = 6,
    Blob = 9,
    Floater = 11,
    Mass = 13,
}

#[derive(Clone)]
pub struct Creature {
    /// Floor point under the creature.
    pub base: [f32; 3],
    /// Heading in radians (rot_y).
    pub yaw: f32,
    pub skins: Vec<Skin>,
    pub parts: Vec<Part>,
    /// Texture id of each skin in its scene.
    pub slots: Vec<u8>,
}

/// A part's place in the world after posing.
#[derive(Clone, Copy, Debug)]
pub struct Placed {
    pub joint: [f32; 3],
    pub centre: [f32; 3],
    /// Columns are the part's local axes in world space.
    pub axes: M3,
}

impl Placed {
    fn world(&self, local: [f32; 3]) -> [f32; 3] {
        add(self.centre, apply(self.axes, local))
    }
}

impl Creature {
    /// The creature on its own for the full-colour pass, meshed as its scene would
    /// mesh it. Slow (a sculpt), so worth doing off the main thread. Its lights are
    /// neutral: bake them with `Scene::light_sh` at `Flesh::light_at`.
    pub fn sculpt(&self) -> crate::world::flesh::Flesh {
        let mut scene = Scene::empty(0, 0, Vec::new(), Vec::new(), crate::world::scene::Rock { height: 0, tex: 0 });
        let placed = self.pose();
        let neutral = vec![[[1.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]]; placed.len()];
        scene.mesh_flesh(self, &placed, &neutral);
        scene.flesh
    }

    /// A hash of everything that shapes the creature's meshes (not its texture slots).
    pub fn shape_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{:?} {:?} {:?} {:?}", self.base, self.yaw, self.skins, self.parts).hash(&mut h);
        h.finish()
    }

    /// Every part placed in the world: forward kinematics down the tree.
    pub fn pose(&self) -> Vec<Placed> {
        let frame = rot_y(self.yaw);
        let mut out: Vec<Placed> = Vec::with_capacity(self.parts.len());
        for p in &self.parts {
            let (joint, f) = match p.parent {
                None => (add(self.base, apply(frame, p.anchor)), frame),
                Some(i) => {
                    let [t, around, o] = p.anchor;
                    (out[i].world(self.parts[i].local(t, around * TAU, o)), out[i].axes)
                }
            };
            let axes = mul(f, p.rot);
            let pivot = [0, 1, 2].map(|c| p.pivot[c] * p.half[c]);
            out.push(Placed { joint, centre: sub(joint, apply(axes, pivot)), axes });
        }
        out
    }

    /// Points spread over every part's surface.
    fn samples(&self) -> Vec<[f32; 3]> {
        let placed = self.pose();
        let mut pts = Vec::new();
        for (p, at) in self.parts.iter().zip(&placed) {
            for k in 0..=8 {
                for j in 0..8 {
                    pts.push(at.world(p.local(k as f32 / 4.0 - 1.0, j as f32 / 8.0 * TAU, 1.0 + p.lumps)));
                }
            }
        }
        pts
    }

    fn rescale(&mut self, k: f32) {
        for p in &mut self.parts {
            p.half = scale(p.half, k);
            if p.parent.is_none() {
                p.anchor = scale(p.anchor, k);
            }
        }
        for s in &mut self.skins {
            s.scale *= k;
        }
    }

    /// Shrink to fit within `max_r` of the base and below `ceiling`, then stand on
    /// `floor` (grounded) or just keep out of it.
    /// How far the creature reaches from its base in x/z, and its top and bottom y.
    pub fn extent(&self) -> (f32, f32, f32) {
        let pts = self.samples();
        let r = pts.iter().map(|p| (p[0] - self.base[0]).hypot(p[2] - self.base[2])).fold(0.0, f32::max);
        let top = pts.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let bottom = pts.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        (r, top, bottom)
    }

    fn fit(&mut self, floor: f32, ceiling: f32, max_r: f32, grounded: bool) {
        let extent = Creature::extent;
        let (r, top, bottom) = extent(self);
        let room = floor - ceiling - 24.0;
        let k = (max_r / r).min(room / (bottom - top)).min(1.0);
        if k < 1.0 {
            self.rescale(k);
        }
        let (_, top, bottom) = extent(self);
        if grounded || bottom > floor {
            self.base[1] += floor - bottom;
        } else if top < ceiling + 12.0 {
            self.base[1] += ceiling + 12.0 - top;
        }
    }
}

/// Grows one creature. The Rng sits in a RefCell so rolls need only `&self`
/// and can go straight into the parts being built.
struct Grow {
    c: Creature,
    rng: RefCell<Rng>,
}

fn body_tint(rng: &mut Rng) -> [f32; 3] {
    let base = rng.pick(&[
        [1.0, 1.0, 1.0],
        [0.8, 1.0, 0.75],
        [1.0, 0.8, 0.9],
        [0.8, 0.75, 1.0],
        [1.0, 0.95, 0.7],
        [0.7, 0.7, 0.75],
    ]);
    base.map(|c: f32| (c * rng.fr(0.85, 1.1)).clamp(0.3, 1.2))
}

/// Frame whose y axis is `n`.
fn frame_y(n: [f32; 3]) -> M3 {
    let h = if n[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let x = norm(cross(h, n));
    let z = cross(x, n);
    [0, 1, 2].map(|r| [x[r], n[r], z[r]])
}

impl Grow {
    fn f(&self) -> f32 {
        self.rng.borrow_mut().f()
    }

    fn fr(&self, lo: f32, hi: f32) -> f32 {
        self.rng.borrow_mut().fr(lo, hi)
    }

    fn range(&self, lo: i32, hi: i32) -> i32 {
        self.rng.borrow_mut().range(lo, hi)
    }

    fn chance(&self, p: f32) -> bool {
        self.rng.borrow_mut().chance(p)
    }

    fn pick<T: Copy>(&self, xs: &[T]) -> T {
        self.rng.borrow_mut().pick(xs)
    }

    fn next(&self) -> u32 {
        self.rng.borrow_mut().next()
    }

    fn tint(&self) -> [f32; 3] {
        body_tint(&mut self.rng.borrow_mut())
    }

    fn add(&mut self, mut p: Part) -> usize {
        p.seed = self.next();
        self.c.parts.push(p);
        self.c.parts.len() - 1
    }

    fn new_skin(&mut self, motif: Motif, tint: [f32; 3]) -> usize {
        let s = Skin {
            motif,
            density: self.f(),
            scale: self.fr(40.0, 110.0),
            tint,
            seed: self.next(),
        };
        self.c.skins.push(s);
        self.c.skins.len() - 1
    }

    /// The creature's skin of `motif`, made if it has none.
    fn skin_of(&mut self, motif: Motif) -> usize {
        match self.c.skins.iter().position(|s| s.motif == motif) {
            Some(i) => i,
            None => {
                let tint = if motif == Motif::Eyes || motif == Motif::Bone { [1.0; 3] } else { self.tint() };
                self.new_skin(motif, tint)
            }
        }
    }

    fn any_skin(&mut self) -> usize {
        self.range(0, self.c.skins.len() as i32) as usize
    }

    /// Orientation (relative to `parent`) pointing out of its surface at (t, around), twisted at random and tilted up to `tilt`.
    fn outward(&mut self, parent: usize, t: f32, around: f32, tilt: f32) -> M3 {
        let n = self.c.parts[parent].normal(t, around * TAU);
        mul(frame_y(n), mul(rot_y(self.fr(0.0, TAU)), rot_x(self.fr(-tilt, tilt))))
    }

    /// Girth of a part, for sizing what grows on it.
    fn girth(&self, i: usize) -> f32 {
        let h = self.c.parts[i].half;
        ((h[0] + h[2]) / 2.0).min(h[1] * 1.5)
    }

    /// A part to grow something on, larger ones likelier.
    fn pick_part(&mut self) -> usize {
        let w: Vec<f32> = (0..self.c.parts.len()).map(|i| self.girth(i)).collect();
        let mut x = self.f() * w.iter().sum::<f32>();
        for (i, wi) in w.iter().enumerate() {
            x -= wi;
            if x <= 0.0 {
                return i;
            }
        }
        0
    }

    fn eye(&mut self, parent: usize, t: f32, around: f32, r: f32) -> usize {
        let skin = self.skin_of(Motif::Eyes);
        let squash = self.fr(0.8, 1.1);
        let rot = self.outward(parent, t, around, 0.3);
        self.add(Part::new(Shape::Ellipsoid, [r, r * squash, r], skin).on(parent, t, around, 0.9).rot(rot).pivot([0.0, -0.3, 0.0]))
    }

    /// Two- or three-fingered claws at the end of `limb`.
    fn claws(&mut self, limb: usize, s: f32) {
        let skin = self.skin_of(Motif::Bone);
        let n = self.range(2, 5);
        for i in 0..n {
            let around = i as f32 / n as f32;
            let rot = mul(rot_y(-around * TAU), rot_z(-self.fr(0.2, 0.6)));
            self.add(Part::new(Shape::Cone, [0.3 * s, s * self.fr(0.8, 1.4), 0.3 * s], skin).on(limb, 0.95, around, 0.5).rot(rot).bend(self.fr(0.0, 0.8)));
        }
    }

    /// Something growing out of `parent` at a random spot; `s` sets its size.
    fn sprout(&mut self, parent: usize, depth: u32) {
        let s = self.girth(parent);
        let (t, around) = (self.fr(-0.8, 0.8), self.f());
        let rot = self.outward(parent, t, around, 0.6);
        match self.range(0, 7) {
            0 => {
                let skin = if self.chance(0.5) { self.skin_of(Motif::Suckers) } else { self.any_skin() };
                let p = Part::new(Shape::Tentacle, [0.28 * s, s * self.fr(1.0, 2.6), 0.28 * s], skin)
                    .on(parent, t, around, 0.85)
                    .rot(rot)
                    .bend(self.fr(-1.6, 1.6))
                    .lumps(self.fr(0.0, 0.15));
                self.add(p);
            }
            1 => {
                let skin = self.any_skin();
                let len = s * self.fr(0.5, 1.3);
                let stalk = self.add(Part::new(Shape::Capsule, [0.12 * s, len, 0.12 * s], skin).on(parent, t, around, 0.85).rot(rot).bend(self.fr(-0.6, 0.6)));
                let skin = self.skin_of(Motif::Eyes);
                let r = s * self.fr(0.25, 0.45);
                self.add(Part::new(Shape::Ellipsoid, [r; 3], skin).on(stalk, 1.0, 0.0, 0.0).pivot([0.0, -0.6, 0.0]));
            }
            2 => {
                let skin = self.skin_of(if self.chance(0.7) { Motif::Bone } else { Motif::Chitin });
                let p = Part::new(Shape::Cone, [0.2 * s, s * self.fr(0.4, 1.1), 0.2 * s], skin).on(parent, t, around, 0.9).rot(rot).bend(self.fr(-0.9, 0.9));
                self.add(p);
            }
            3 => {
                let skin = self.any_skin();
                let upper = Part::new(Shape::Capsule, [0.16 * s, 0.6 * s, 0.16 * s], skin).on(parent, t, around, 0.8).rot(rot).taper(self.fr(0.6, 1.0));
                let elbow = self.fr(0.3, 1.4);
                let upper = self.add(upper);
                let fore = self.add(Part::new(Shape::Capsule, [0.13 * s, 0.6 * s, 0.13 * s], skin).on(upper, 1.0, 0.0, 0.0).rot(rot_x(elbow)).taper(0.7));
                self.claws(fore, 0.14 * s);
            }
            4 => {
                let skin = self.any_skin();
                let k = self.fr(0.35, 0.7);
                let p = Part::new(Shape::Ellipsoid, [k * s, k * s * self.fr(0.7, 1.2), k * s], skin)
                    .on(parent, t, around, 0.8)
                    .rot(rot)
                    .pivot([0.0, -0.6, 0.0])
                    .lumps(self.fr(0.05, 0.3))
                    .bulge(self.fr(0.0, 0.3));
                let bulb = self.add(p);
                if depth < 2 {
                    for _ in 0..self.range(0, 3) {
                        self.sprout(bulb, depth + 1);
                    }
                }
            }
            5 => {
                let skin = self.skin_of(if self.chance(0.5) { Motif::Teeth } else { Motif::Mouths });
                let k = self.fr(0.35, 0.6);
                let maw = self.add(Part::new(Shape::Ellipsoid, [k * s, 0.75 * k * s, k * s], skin).on(parent, t, around, 0.85).rot(rot).pivot([0.0, -0.5, 0.0]));
                // A ring of fangs round its tip.
                let skin = self.skin_of(Motif::Bone);
                let n = self.range(5, 11);
                let ring = self.fr(0.2, 0.6);
                for i in 0..n {
                    let a = i as f32 / n as f32;
                    let rot = self.outward(maw, ring, a, 0.2);
                    self.add(Part::new(Shape::Cone, [0.08 * s, k * s * self.fr(0.25, 0.5), 0.08 * s], skin).on(maw, ring, a, 0.9).rot(rot).bend(0.4));
                }
            }
            _ => {
                for _ in 0..self.range(2, 6) {
                    let (dt, da) = (self.fr(-0.25, 0.25), self.fr(-0.08, 0.08));
                    let r = s * self.fr(0.12, 0.3);
                    self.eye(parent, (t + dt).clamp(-0.9, 0.9), around + da, r);
                }
            }
        }
    }

    fn humanoid(&mut self, h: f32) {
        let body = self.new_skin(self.pick(&skin::BODY), self.tint());
        let accent = self.new_skin(self.pick(&skin::ACCENT), self.tint());
        let thin = self.fr(0.55, 1.2);
        let pelvis = self.add(Part::new(Shape::Ellipsoid, [0.12 * h, 0.07 * h, 0.08 * h * thin], body).root([0.0, -0.5 * h, 0.0]).pivot([0.0; 3]));
        for (around, side) in [(0.0, 1.0), (0.5, -1.0)] {
            let (swing, spread, knee) = (self.fr(-0.5, 0.5), self.fr(0.0, 0.15), self.fr(0.0, 0.7));
            let thigh = Part::new(Shape::Capsule, [0.055 * h * thin, 0.13 * h, 0.055 * h * thin], body)
                .on(pelvis, 0.3, around, 0.55)
                .rot(mul(rot_x(swing), rot_z(-side * spread)))
                .taper(0.8);
            let thigh = self.add(thigh);
            let shin = self.add(Part::new(Shape::Capsule, [0.042 * h * thin, 0.13 * h, 0.042 * h * thin], body).on(thigh, 1.0, 0.0, 0.0).rot(rot_x(-knee)).taper(0.7));
            self.add(Part::new(Shape::Ellipsoid, [0.04 * h, 0.025 * h, 0.07 * h], body).on(shin, 1.0, 0.0, 0.0).rot(rot_x(knee - swing)).pivot([0.0, -0.5, -0.5]));
        }
        let torso = Part::new(Shape::Capsule, [0.13 * h, 0.17 * h, 0.08 * h * thin], body)
            .on(pelvis, -0.6, 0.0, 0.0)
            .rot(mul(rot_x(-self.fr(-0.15, 0.35)), rot_z(PI)))
            .taper(self.fr(0.85, 1.35))
            .bulge(self.fr(0.0, 0.3))
            .lumps(self.fr(0.0, 0.12));
        let torso = self.add(torso);
        let pairs = if self.chance(0.25) { 2 } else { 1 };
        for pair in 0..pairs {
            for (around, side) in [(0.0, 1.0), (0.5, -1.0)] {
                let (spread, reach, elbow) = (self.fr(0.05, 0.5), self.fr(-0.4, 1.0), self.fr(0.0, 1.2));
                let upper = Part::new(Shape::Capsule, [0.04 * h * thin, 0.12 * h, 0.04 * h * thin], body)
                    .on(torso, 0.75 - 0.4 * pair as f32, around, 0.85)
                    .rot(mul(rot_z(PI - side * spread), rot_x(reach)))
                    .taper(0.85);
                let upper = self.add(upper);
                let fore = self.add(Part::new(Shape::Capsule, [0.034 * h * thin, 0.12 * h, 0.034 * h * thin], body).on(upper, 1.0, 0.0, 0.0).rot(rot_x(elbow)).taper(0.8));
                self.claws(fore, 0.03 * h);
            }
        }
        let head_skin = if self.chance(0.35) { accent } else { body };
        let head = Part::new(Shape::Ellipsoid, [0.075 * h, 0.09 * h, 0.08 * h], head_skin)
            .on(torso, 1.0, 0.0, 0.0)
            .rot(mul(rot_y(self.fr(-0.6, 0.6)), rot_x(self.fr(-0.3, 0.3))))
            .lumps(self.fr(0.0, 0.2));
        let head = self.add(head);
        if self.chance(0.6) {
            let r = 0.022 * h * self.fr(0.8, 1.6);
            for da in [-0.06, 0.06] {
                self.eye(head, 0.15, 0.25 + da, r);
            }
        }
        if self.chance(0.4) {
            let skin = self.skin_of(if self.chance(0.5) { Motif::Teeth } else { Motif::Mouths });
            self.add(Part::new(Shape::Ellipsoid, [0.045 * h, 0.025 * h, 0.03 * h], skin).on(head, -0.45, 0.25, 0.85).pivot([0.0; 3]));
        }
    }

    fn beast(&mut self, len: f32) {
        let body = self.new_skin(self.pick(&skin::BODY), self.tint());
        self.new_skin(self.pick(&skin::ACCENT), self.tint());
        let leg = 0.3 * len;
        // Local y runs forward (+z) and local z up.
        let trunk = Part::new(Shape::Capsule, [0.16 * len, 0.42 * len, 0.13 * len], body)
            .root([0.0, -(2.0 * leg + 0.1 * len), 0.0])
            .rot(rot_x(PI / 2.0))
            .pivot([0.0; 3])
            .bulge(self.fr(0.0, 0.35))
            .lumps(self.fr(0.0, 0.15));
        let trunk = self.add(trunk);
        for (t, around) in [(0.65, 0.87), (0.65, 0.63), (-0.65, 0.87), (-0.65, 0.63)] {
            let (swing, knee) = (self.fr(-0.45, 0.45), self.fr(-0.6, 0.6));
            let thigh = self.add(Part::new(Shape::Capsule, [0.055 * len, leg / 2.0, 0.055 * len], body).on(trunk, t, around, 0.7).rot(rot_x(-PI / 2.0 + swing)).taper(0.8));
            let shin = self.add(Part::new(Shape::Capsule, [0.04 * len, leg / 2.0, 0.04 * len], body).on(thigh, 1.0, 0.0, 0.0).rot(rot_x(knee)).taper(0.7));
            self.claws(shin, 0.03 * len);
        }
        let neck = self.add(Part::new(Shape::Capsule, [0.07 * len, 0.13 * len, 0.07 * len], body).on(trunk, 0.85, 0.25, 0.4).rot(rot_x(self.fr(0.2, 0.9))));
        let head = Part::new(Shape::Capsule, [0.09 * len, 0.14 * len, 0.08 * len], body)
            .on(neck, 1.0, 0.0, 0.0)
            .rot(mul(rot_x(-self.fr(0.3, 0.9)), rot_z(self.fr(-0.4, 0.4))))
            .taper(self.fr(0.5, 1.0))
            .lumps(self.fr(0.0, 0.15));
        let head = self.add(head);
        let skin = self.skin_of(if self.chance(0.6) { Motif::Teeth } else { Motif::Mouths });
        self.add(Part::new(Shape::Ellipsoid, [0.06 * len, 0.08 * len, 0.04 * len], skin).on(head, 0.6, 0.75, 0.6).pivot([0.0; 3]));
        if self.chance(0.7) {
            let r = 0.025 * len * self.fr(0.8, 1.8);
            for a in [0.1, 0.4] {
                self.eye(head, 0.2, a, r);
            }
        }
        let tail = Part::new(Shape::Tentacle, [0.05 * len, 0.3 * len * self.fr(0.7, 1.5), 0.05 * len], body)
            .on(trunk, -0.95, 0.0, 0.0)
            .rot(rot_x(PI - self.fr(0.2, 1.0)))
            .bend(self.fr(-1.2, 1.2));
        self.add(tail);
        if self.chance(0.5) {
            let skin = self.skin_of(Motif::Bone);
            let n = self.range(4, 9);
            for i in 0..n {
                let t = -0.7 + 1.4 * i as f32 / (n - 1) as f32;
                let rot = self.outward(trunk, t, 0.25, 0.2);
                let s = 0.05 * len * self.fr(0.6, 1.4);
                self.add(Part::new(Shape::Cone, [s, 2.5 * s, s * 0.5], skin).on(trunk, t, 0.25, 0.9).rot(rot).bend(-0.5));
            }
        }
    }

    fn blob(&mut self, r: f32) {
        let skin = self.new_skin(self.pick(&[Motif::Flesh, Motif::Veins, Motif::Suckers, Motif::Wrinkles]), self.tint());
        let squash = self.fr(0.5, 0.8);
        let p = Part::new(Shape::Ellipsoid, [r, squash * r, r * self.fr(0.8, 1.2)], skin)
            .root([0.0, -squash * r, 0.0])
            .pivot([0.0; 3])
            .lumps(self.fr(0.12, 0.35))
            .bulge(self.fr(0.0, 0.3));
        let root = self.add(p);
        for _ in 0..self.range(1, 6) {
            let (t, a) = (self.fr(-0.9, 0.1), self.f());
            let er = r * self.fr(0.1, 0.3);
            self.eye(root, t, a, er);
        }
        for _ in 0..self.range(0, 5) {
            self.sprout(root, 1);
        }
    }

    fn floater(&mut self, r: f32, lift: f32) {
        let eye = self.new_skin(Motif::Eyes, [1.0; 3]);
        self.c.skins[eye].density = 0.0;
        self.c.skins[eye].scale = PI * r;
        let limbs = self.new_skin(self.pick(&[Motif::Suckers, Motif::Flesh, Motif::Veins, Motif::Scales]), self.tint());
        let root = self.add(Part::new(Shape::Ellipsoid, [r, r * self.fr(0.85, 1.1), r], eye).root([0.0, -lift, 0.0]).pivot([0.0; 3]).lumps(self.fr(0.0, 0.1)));
        let n = self.range(3, 9);
        for i in 0..n {
            let around = i as f32 / n as f32;
            let t = self.fr(0.3, 0.7);
            let rot = self.outward(root, t, around, 0.2);
            let p = Part::new(Shape::Tentacle, [0.12 * r, r * self.fr(1.2, 2.6), 0.12 * r], limbs).on(root, t, around, 0.9).rot(rot).bend(self.fr(-1.5, 1.5));
            self.add(p);
        }
    }

    /// A heap of parts sprouting from parts: the dream-creature.
    fn mass(&mut self, size: f32) {
        for _ in 0..self.range(2, 5) {
            let m = if self.c.skins.is_empty() { self.pick(&skin::BODY) } else { skin::ALL[self.range(0, skin::MOTIFS as i32) as usize] };
            self.new_skin(m, self.tint());
        }
        let root = Part::new(Shape::Ellipsoid, [size, size * self.fr(0.6, 1.0), size * self.fr(0.8, 1.2)], 0)
            .root([0.0, -size, 0.0])
            .pivot([0.0; 3])
            .lumps(self.fr(0.1, 0.35))
            .bulge(self.fr(0.0, 0.4));
        self.add(root);
        for _ in 0..self.range(5, 16) {
            let p = self.pick_part();
            self.sprout(p, 0);
        }
    }
}

impl Scene {
    pub(crate) fn build_creatures(&mut self, rng: &mut Rng) {
        for ri in 0..self.rooms.len() {
            let r = self.rooms[ri].clone();
            for _ in 0..rng.range(0, 3) {
                let (cx, cz) = (r.x + rng.range(0, r.w), r.z + rng.range(0, r.d));
                if !self.claim_cell(ri, cx, cz) {
                    continue;
                }
                let base = [(cx as f32 + rng.fr(0.4, 0.6)) * TF, r.floor as f32, (cz as f32 + rng.fr(0.4, 0.6)) * TF];
                let clearance = if self.outdoor { 6.0 * TF } else { r.height as f32 * TF };
                self.grow_creature(rng, base, clearance, None, 1.0, None);
            }
        }
    }

    /// Grow a creature standing on the floor at `base` under `clearance` of headroom,
    /// mesh it and add it: of `kind` (or a random one), `scale` times its usual size.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn grow_creature(
        &mut self,
        rng: &mut Rng,
        base: [f32; 3],
        clearance: f32,
        kind: Option<Kind>,
        scale: f32,
        yaw: Option<f32>,
    ) {
        let floor = base[1];
        let yaw = match yaw {
            Some(y) => y,
            None => rng.fr(0.0, TAU),
        };
        let c = Creature { base, yaw, skins: Vec::new(), parts: Vec::new(), slots: Vec::new() };
        let mut g = Grow { c, rng: RefCell::new(rng.clone()) };
        let kind = match kind {
            Some(k) => k as i32,
            None => g.range(0, 20),
        };
        let grounded = match kind {
            0..=5 => {
                let h = if g.chance(0.15) { g.fr(1.4, 1.9) } else { g.fr(0.75, 1.1) } * TF;
                g.humanoid(h.min(clearance - 30.0));
                true
            }
            6..=8 => {
                let len = g.fr(0.55, 0.85) * TF;
                g.beast(len);
                true
            }
            9..=10 => {
                let r = g.fr(0.25, 0.42) * TF;
                g.blob(r);
                true
            }
            11..=12 => {
                let (r, lift) = (g.fr(0.15, 0.28) * TF, g.fr(0.8, 1.4) * TF);
                g.floater(r, lift);
                false
            }
            _ => {
                let size = g.fr(0.18, 0.32) * TF;
                g.mass(size);
                true
            }
        };
        // Most also grow something they shouldn't.
        if kind <= 10 && g.chance(0.6) {
            for _ in 0..g.range(1, 6) {
                let p = g.pick_part();
                g.sprout(p, 1);
            }
        }
        *rng = g.rng.into_inner();
        let mut c = g.c;
        if scale != 1.0 {
            c.rescale(scale);
        }
        c.fit(floor, floor - clearance, 0.95 * TF * scale.max(1.0), grounded);
        // The full-colour pass has skins of its own; PS1 ones are only for the PS1 pass.
        c.slots = if crate::world::flesh::enabled() { vec![0; c.skins.len()] } else { c.skins.iter().map(|s| self.skin_slot(s)).collect() };
        let start = self.creature_tris.len();
        self.mesh_creature(&c);
        self.creature_spans.push(start..self.creature_tris.len());
        self.creatures.push(c);
    }

    /// Texture id for a creature skin: a fresh slot while there are any, else the
    /// closest one of the same motif (or any).
    fn skin_slot(&mut self, s: &Skin) -> u8 {
        if self.skins.len() < tex::SKIN_SLOTS {
            self.skins.push(skin::make(s.motif, s.density, s.seed));
            self.skin_kinds.push((s.motif, s.density));
            return tex::SKIN_BASE + (self.skins.len() - 1) as u8;
        }
        let cost = |&(m, d): &(Motif, f32)| if m == s.motif { (d - s.density).abs() } else { 2.0 };
        let best = (0..self.skin_kinds.len()).min_by(|&a, &b| cost(&self.skin_kinds[a]).total_cmp(&cost(&self.skin_kinds[b])));
        tex::SKIN_BASE + best.unwrap_or(0) as u8
    }

    /// The creature's meshes: through the PS1 pipeline, or for the full-colour pass
    /// as one sculpted surface (see `sculpt`) plus whatever parts are too thin for it.
    fn mesh_creature(&mut self, c: &Creature) {
        let placed = c.pose();
        if !crate::world::flesh::enabled() {
            for (p, at) in c.parts.iter().zip(&placed) {
                self.mesh_part(p, at, &c.skins[p.skin], c.slots[p.skin], None);
            }
            return;
        }
        let lights: Vec<_> = placed.iter().map(|at| self.light_sh(at.centre)).collect();
        self.mesh_flesh(c, &placed, &lights);
    }

    /// The creature for the full-colour pass, `lights` being each part's baked light.
    fn mesh_flesh(&mut self, c: &Creature, placed: &[Placed], part_lights: &[[[f32; 3]; 4]]) {
        let base = self.flesh.skins.len();
        for s in &c.skins {
            self.flesh.skins.push(crate::world::flesh::Skin::make(s.motif, s.density, s.seed, s.tint));
        }
        let skins: Vec<u16> = c.parts.iter().map(|p| (base + p.skin) as u16).collect();
        let lights: Vec<u16> = placed
            .iter()
            .zip(part_lights)
            .map(|(at, &light)| {
                self.flesh.lights.push(light);
                self.flesh.light_at.push(at.centre);
                (self.flesh.lights.len() - 1) as u16
            })
            .collect();
        let (tris, thin) = crate::world::sculpt::mesh(c, placed, &skins, &lights);
        self.flesh.tris.extend(tris);
        for i in thin {
            let p = &c.parts[i];
            self.mesh_part(p, &placed[i], &c.skins[p.skin], 0, Some((skins[i], lights[i])));
        }
    }

    /// A dense latitude-longitude mesh of one part, smooth-shaded, its skin
    /// repeated a whole number of times around and along: for the PS1 pass, or with
    /// `full` (a full-colour skin and light index) for the full-colour one.
    fn mesh_part(&mut self, p: &Part, at: &Placed, skin: &Skin, tex: u8, full: Option<(u16, u16)>) {
        let circ = PI * (p.half[0] + p.half[2]) * (1.0 + p.bulge) * p.taper.max(1.0);
        let len = 2.0 * p.half[1] * (1.0 + 0.5 * p.bend.abs());
        let reps = |l: f32| (l / skin.scale).round().clamp(1.0, 8.0) as usize;
        let (ta, tl) = (reps(circ), reps(len));
        let per = |l: f32, reps: usize, lo: f32, hi: f32| ((l / EDGE).clamp(lo, hi) / reps as f32).ceil().max(1.0) as usize;
        let (sa, sl) = (per(circ, ta, 8.0, 28.0), per(len, tl, 6.0, 28.0));
        let (na, nl) = (ta * sa, tl * sl);
        let waves = p.lump_waves();
        let lump = |t: f32, a: f32| waves.iter().map(|w| (w[0] * a + w[2]).sin() * (w[1] * t * PI + w[3]).sin()).sum::<f32>() / 3.0;

        let mut pos = Vec::with_capacity((nl + 1) * (na + 1));
        for k in 0..=nl {
            let t = k as f32 / nl as f32 * 2.0 - 1.0;
            for j in 0..=na {
                let a = j as f32 / na as f32 * TAU;
                pos.push(at.world(p.local(t, a, 1.0 + p.lumps * lump(t, a))));
            }
        }
        let at_grid = |k: usize, j: usize| pos[k * (na + 1) + j % na];
        let yaxis = [at.axes[0][1], at.axes[1][1], at.axes[2][1]];
        let mut nrm = Vec::with_capacity(pos.len());
        // Directions of increasing u (around) and v (along), for normal maps; zero at the poles.
        let mut tan = Vec::with_capacity(pos.len());
        for k in 0..=nl {
            let t = k as f32 / nl as f32 * 2.0 - 1.0;
            let mid = at.world(p.axis(t));
            for j in 0..=na {
                let (da, dt) = (sub(at_grid(k, j + 1), at_grid(k, j + na - 1)), sub(at_grid((k + 1).min(nl), j), at_grid(k.max(1) - 1, j)));
                tan.push((if dot(da, da) > 1e-9 { norm(da) } else { [0.0; 3] }, if dot(dt, dt) > 1e-9 { norm(dt) } else { [0.0; 3] }));
                let n = if k == 0 {
                    scale(yaxis, -1.0)
                } else if k == nl {
                    yaxis
                } else {
                    let da = sub(at_grid(k, j + 1), at_grid(k, j + na - 1));
                    let dt = sub(at_grid(k + 1, j), at_grid(k - 1, j));
                    let n = cross(da, dt);
                    if dot(n, n) < 1e-9 {
                        norm(sub(at_grid(k, j), mid))
                    } else if dot(n, sub(at_grid(k, j), mid)) < 0.0 {
                        norm(scale(n, -1.0))
                    } else {
                        norm(n)
                    }
                };
                nrm.push(n);
            }
        }

        let widest = p.widest();
        let cap_axes = [0, 2].map(|c| [at.axes[0][c], at.axes[1][c], at.axes[2][c]]);
        let (u0, v0) = tex::uv_origin(tex);
        let alb = skin.tint.map(|c| (c * 128.0).clamp(0.0, 255.0) as u8);
        let coord = |i: usize, sub: usize| (63 * i / sub) as u8;
        for k in 0..nl {
            let (va, vb) = (coord(k % sl, sl), coord(k % sl + 1, sl));
            for j in 0..na {
                let (ua, ub) = (coord(j % sa, sa), coord(j % sa + 1, sa));
                let v = |kk: usize, jj: usize, uv: [u8; 2]| (kk * (na + 1) + jj, [u0 + uv[0], v0 + uv[1]]);
                let quad = [v(k, j, [ua, va]), v(k, j + 1, [ub, va]), v(k + 1, j, [ua, vb]), v(k + 1, j + 1, [ub, vb])];
                if full.is_none() {
                    for [a, b, c] in [[0, 1, 2], [1, 3, 2]] {
                        self.creature_tri([quad[a], quad[b], quad[c]], &pos, &nrm, tex, skin.tint, alb);
                    }
                }
                if let Some((skin_id, light)) = full {
                    let fv = |kk: usize, jj: usize| {
                        let i = kk * (na + 1) + jj;
                        let g = kk * (na + 1) + jj % na;
                        let uv = [jj as f32 / sa as f32, kk as f32 / sl as f32];
                        let t = kk as f32 / nl as f32 * 2.0 - 1.0;
                        let l = p.local(t, jj as f32 / na as f32 * TAU, 1.0);
                        let cap = [l[0] / skin.scale, l[2] / skin.scale];
                        let wrap = crate::world::flesh::wrap_weight(p.profile(t) / widest);
                        crate::world::flesh::Vertex {
                            p: pos[g],
                            n: nrm[i],
                            w: 1.0,
                            uv: [uv; 2],
                            t: [tan[i].0; 2],
                            b: [tan[i].1; 2],
                            cap: [cap; 2],
                            wrap: [wrap; 2],
                        }
                    };
                    let q = [fv(k, j), fv(k, j + 1), fv(k + 1, j), fv(k + 1, j + 1)];
                    for [a, b, c] in [[0, 1, 2], [1, 3, 2]] {
                        self.flesh.tris.push(crate::world::flesh::Tri { v: [q[a], q[b], q[c]], skin: [skin_id; 2], light: [light; 2], cap_axes: [cap_axes; 2] });
                    }
                }
            }
        }
    }

    fn creature_tri(&mut self, v: [(usize, [u8; 2]); 3], pos: &[[f32; 3]], nrm: &[[f32; 3]], tex: u8, tint: [f32; 3], alb: [u8; 3]) {
        let p = v.map(|(i, _)| pos[i].map(|c| c.round() as i32));
        let e1 = [0, 1, 2].map(|c| (p[1][c] - p[0][c]) as f32);
        let e2 = [0, 1, 2].map(|c| (p[2][c] - p[0][c]) as f32);
        let cr = cross(e1, e2);
        if dot(cr, cr) < 1.0 {
            return;
        }
        let n_sum = add(add(nrm[v[0].0], nrm[v[1].0]), nrm[v[2].0]);
        let mut verts = [0, 1, 2].map(|i| Vert { p: p[i], uv: v[i].1, col: self.shade(p[i], nrm[v[i].0], tint), alb });
        if dot(cr, n_sum) < 0.0 {
            verts.swap(1, 2);
        }
        self.creature_tris.push(Tri { v: verts, tex, n: norm(n_sum), tint });
    }
}
