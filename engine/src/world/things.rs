//! What moves in a hand-built level: doors that open and close, keys lying
//! about to pick up, and levers that work doors. Each has its own share of
//! triangles (a `Part`) apart from `Scene::tris`, so it can be rebuilt on its
//! own as it moves; the renderers and the network draw the parts with the rest.

use crate::world::level::{Entity, EntityKind, Hinge};
use crate::world::scene::{rot_x, rot_y, smoothstep, mul, Scene, Tri, MAX_STEP, STEP, T};
use crate::world::tex;

const TF: f32 = T as f32;
/// Door frame posts: width along the doorway and depth across it.
const POST: f32 = 16.0;
const POST_D: f32 = 22.0;
/// Half the thickness of a door leaf.
const LEAF: f32 = 5.0;
/// Seconds a door takes to open or close.
const SWING: f32 = 0.5;
/// How far away (world units) a door's centre or a lever can be used from.
pub const REACH: f32 = 1.6 * TF;
/// How near (world units) the feet must come to a key to pick it up.
const PICK_UP: f32 = 0.5 * TF;
const KEY_TINT: [f32; 3] = [1.0, 0.85, 0.4];

/// Triangles that move or vanish apart from the rest of the level.
#[derive(Clone, Debug, Default)]
pub struct Part {
    pub tris: Vec<Tri>,
    /// Bumped whenever `tris` changes, so whatever draws them knows to again.
    pub version: u32,
}

/// A door in a doorway, from a `Door` entity.
#[derive(Clone, Debug)]
pub struct Door {
    pub entity: u32,
    part: usize,
    /// The middle of the doorway at the bottom of the leaf, world units.
    centre: [f32; 3],
    yaw: f32,
    /// Half the leaf's width.
    half: f32,
    /// The top of the opening, and the bottom of the leaf.
    top: f32,
    sill: f32,
    hinge: Hinge,
    /// 0 shut to 1 open, and where it is heading.
    pub open: f32,
    pub opening: bool,
    pub locked: bool,
    pub key: Option<u32>,
}

/// A key on the floor, until it is picked up.
#[derive(Clone, Debug)]
pub struct Pickup {
    pub entity: u32,
    pub name: String,
    pos: [f32; 3],
    part: usize,
    pub taken: bool,
}

/// A lever on the floor that toggles its target doors.
#[derive(Clone, Debug)]
pub struct Lever {
    pub entity: u32,
    pos: [f32; 3],
    yaw: f32,
    pub targets: Vec<u32>,
    part: usize,
    pub pulled: bool,
}

/// The moving things of a scene.
#[derive(Clone, Debug, Default)]
pub struct Things {
    pub parts: Vec<Part>,
    pub doors: Vec<Door>,
    pub pickups: Vec<Pickup>,
    pub levers: Vec<Lever>,
}

impl Door {
    /// Unit vectors along the doorway (local x) and across it (local z, the way it opens).
    fn axes(&self) -> ([f32; 2], [f32; 2]) {
        let (s, c) = self.yaw.sin_cos();
        ([c, s], [-s, c])
    }

    /// Where a point is relative to the doorway: along it and across it.
    fn local(&self, x: f32, z: f32) -> (f32, f32) {
        let (a, f) = self.axes();
        let (dx, dz) = (x - self.centre[0], z - self.centre[2]);
        (dx * a[0] + dz * a[1], dx * f[0] + dz * f[1])
    }

    pub fn moving(&self) -> bool {
        self.open != if self.opening { 1.0 } else { 0.0 }
    }
}

impl Scene {
    /// Triangles of the level and of every part: everything there is to draw.
    pub fn all_tris(&self) -> impl Iterator<Item = &Tri> {
        self.tris.iter().chain(self.part_tris())
    }

    pub fn part_tris(&self) -> impl Iterator<Item = &Tri> {
        self.things.parts.iter().flat_map(|p| p.tris.iter())
    }

    /// Doors, keys and levers from the level's entities, each starting as the level says.
    pub(crate) fn build_things(&mut self, entities: &[Entity]) {
        for e in entities {
            let [x, y, z] = e.pos;
            match &e.kind {
                EntityKind::Door { width, height, hinge, open, locked, key } => {
                    self.build_door(e, *width, *height, *hinge, *open, *locked, *key);
                }
                EntityKind::Key { name } => {
                    let Some(floor) = self.floor_y(x, y, z) else { continue };
                    let part = self.new_part();
                    self.things.pickups.push(Pickup { entity: e.id, name: name.clone(), pos: [x, floor, z], part, taken: false });
                    let i = self.things.pickups.len() - 1;
                    self.shape_pickup(i);
                }
                EntityKind::Lever { targets } => {
                    let Some(floor) = self.floor_y(x, y, z) else { continue };
                    let part = self.new_part();
                    let lever = Lever { entity: e.id, pos: [x, floor, z], yaw: e.yaw, targets: targets.clone(), part, pulled: false };
                    self.things.levers.push(lever);
                    let i = self.things.levers.len() - 1;
                    self.shape_lever(i);
                }
                _ => {}
            }
        }
    }

    fn new_part(&mut self) -> usize {
        self.things.parts.push(Part::default());
        self.things.parts.len() - 1
    }

    /// Whatever `draw` adds to the level, taken out again as part `i`'s triangles.
    fn set_part(&mut self, i: usize, draw: impl FnOnce(&mut Scene)) {
        let start = self.tris.len();
        draw(self);
        let tris = self.tris.split_off(start);
        let p = &mut self.things.parts[i];
        p.tris = tris;
        p.version += 1;
    }

    #[allow(clippy::too_many_arguments)]
    fn build_door(&mut self, e: &Entity, width: i32, height: i32, hinge: Hinge, open: bool, locked: bool, key: Option<u32>) {
        let [x, y, z] = e.pos;
        let (s, c) = e.yaw.sin_cos();
        let (fwd, along) = ([-s, c], [c, s]);
        // The floors and ceilings just in front of and behind the doorway.
        let side = |k: f32| (x + fwd[0] * k * 2.0, z + fwd[1] * k * 2.0);
        let mut floors = Vec::new();
        let mut ceilings = Vec::new();
        let mut front = None;
        for k in [-1.0, 1.0] {
            let (sx, sz) = side(k);
            let Some(n) = self.node_at(sx, y, sz) else { continue };
            floors.push(self.node_floor(n, [sx, sz]));
            if k > 0.0 {
                front = floors.last().copied();
            }
            if !self.span(n).sky {
                ceilings.push(self.node_ceiling(n, [sx, sz]));
            }
        }
        if floors.is_empty() {
            return;
        }
        // y points down: the higher floor has the smaller y, the lower ceiling the larger.
        let sill = floors.iter().copied().fold(f32::INFINITY, f32::min);
        let bottom = floors.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ceiling = ceilings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ceiling = if ceilings.is_empty() { sill - 6.0 * TF } else { ceiling };
        let top = (sill - (height.max(1) * STEP) as f32).max(ceiling);
        // A swinging leaf reaches down to the floor it swings over; one that slides
        // up closes right down to the lower floor.
        let sill = match hinge {
            Hinge::Up => bottom,
            _ => front.unwrap_or(sill),
        };
        let half_w = width.max(1) as f32 * TF / 2.0;
        let m = rot_y(e.yaw);
        let at = |l: f32, y: f32| [x + along[0] * l, y, z + along[1] * l];
        // The frame and the wall over it are part of the level.
        let frame = [0, 1, 2][e.id as usize % 3];
        let tint = [0.85, 0.8, 0.75];
        for sd in [-1.0, 1.0] {
            let c = at(sd * (half_w - POST / 2.0), (bottom + top) / 2.0);
            self.add_obox(c, m, [POST / 2.0, (bottom - top) / 2.0, POST_D], frame, tint);
        }
        let beam = 12.0_f32.min((top - ceiling) / 2.0);
        if beam > 1.0 {
            self.add_obox(at(0.0, top - beam), m, [half_w, beam, POST_D], frame, tint);
            let fill = top - 2.0 * beam;
            if fill - ceiling > 1.0 {
                let wall = self.node_at(x, y, z).map_or(0, |n| self.rooms[self.span(n).room].wall_tex);
                self.add_obox(at(0.0, (ceiling + fill) / 2.0), m, [half_w, (fill - ceiling) / 2.0, POST_D - 6.0], wall, tint);
            }
        }
        let part = self.new_part();
        let open = if open { 1.0 } else { 0.0 };
        self.things.doors.push(Door {
            entity: e.id,
            part,
            centre: [x, sill, z],
            yaw: e.yaw,
            half: half_w - POST,
            top,
            sill,
            hinge,
            open,
            opening: open > 0.5,
            locked,
            key,
        });
        let i = self.things.doors.len() - 1;
        self.shape_door(i);
    }

    /// Door `i`'s leaf as it is now.
    fn shape_door(&mut self, i: usize) {
        let d = self.things.doors[i].clone();
        let t = smoothstep(0.0, 1.0, d.open);
        let (along, _) = d.axes();
        let h = d.sill - d.top - 2.0;
        self.set_part(d.part, |s| match d.hinge {
            Hinge::Up => {
                // Slides up into the frame: the bottom edge rises.
                let bottom = d.sill - 2.0 - h * t;
                if bottom - d.top > 2.0 {
                    let c = [d.centre[0], (bottom + d.top) / 2.0, d.centre[2]];
                    s.add_obox(c, rot_y(d.yaw), [d.half, (bottom - d.top) / 2.0, LEAF], tex::DOOR, [0.9; 3]);
                }
            }
            Hinge::Left | Hinge::Right => {
                // Swings a quarter turn about the hinge post, toward the way it faces.
                let sd = if d.hinge == Hinge::Left { -1.0 } else { 1.0 };
                let hinge = [d.centre[0] + along[0] * sd * d.half, d.centre[2] + along[1] * sd * d.half];
                let a = d.yaw - sd * t * std::f32::consts::FRAC_PI_2;
                let (sa, ca) = a.sin_cos();
                let c = [hinge[0] - sd * ca * d.half, d.sill - 2.0 - h / 2.0, hinge[1] - sd * sa * d.half];
                s.add_obox(c, rot_y(a), [d.half, h / 2.0, LEAF], tex::DOOR, [0.9; 3]);
            }
        });
    }

    fn shape_pickup(&mut self, i: usize) {
        let k = self.things.pickups[i].clone();
        self.set_part(k.part, |s| {
            if k.taken {
                return;
            }
            let [x, floor, z] = k.pos;
            // A small stone plinth with the key lying across it.
            s.add_box([x, floor, z], [22.0, 22.0], 40.0, 0.0, 2, [0.7, 0.68, 0.65]);
            let top = floor - 40.0;
            s.add_box([x - 4.0, top, z], [26.0, 3.0], 6.0, 0.0, 5, KEY_TINT);
            s.add_box([x - 34.0, top, z], [8.0, 9.0], 6.0, 0.0, 5, KEY_TINT);
            s.add_box([x + 16.0, top, z + 7.0], [3.0, 5.0], 6.0, 0.0, 5, KEY_TINT);
        });
    }

    fn shape_lever(&mut self, i: usize) {
        let l = self.things.levers[i].clone();
        self.set_part(l.part, |s| {
            let [x, floor, z] = l.pos;
            s.add_box([x, floor, z], [26.0, 18.0], 30.0, l.yaw, 2, [0.6, 0.58, 0.55]);
            // The handle leans forward or back about the pivot on top of the base.
            let m = mul(rot_y(l.yaw), rot_x(if l.pulled { -0.6 } else { 0.6 }));
            let pivot = [x, floor - 30.0, z];
            let at = |k: f32| {
                let v = crate::world::scene::apply(m, [0.0, -k, 0.0]);
                [pivot[0] + v[0], pivot[1] + v[1], pivot[2] + v[2]]
            };
            s.add_obox(at(40.0), m, [4.0, 40.0, 4.0], 2, [0.5, 0.45, 0.4]);
            s.add_obox(at(84.0), m, [9.0, 9.0, 9.0], 5, KEY_TINT);
        });
    }

    /// Move the doors on by `dt` seconds. True when any part changed.
    pub fn tick_things(&mut self, dt: f32) -> bool {
        let mut changed = false;
        for i in 0..self.things.doors.len() {
            let d = &mut self.things.doors[i];
            if !d.moving() {
                continue;
            }
            let step = dt / SWING;
            d.open = if d.opening { (d.open + step).min(1.0) } else { (d.open - step).max(0.0) };
            self.shape_door(i);
            changed = true;
        }
        changed
    }

    /// Does a closed or moving door stand in the way of a body of radius `r`
    /// with its feet at `feet`, at (x, z)?
    pub fn door_blocks(&self, x: f32, z: f32, feet: f32, r: f32) -> bool {
        self.things.doors.iter().any(|d| {
            if d.open >= 1.0 || feet < d.top || feet > d.sill + MAX_STEP {
                return false;
            }
            let (u, v) = d.local(x, z);
            u.abs() < d.half + POST + r && v.abs() < LEAF + r
        })
    }

    /// Pick up any key within reach of the feet at `pos`: their ids and names.
    pub fn take_pickups(&mut self, pos: [f32; 3]) -> Vec<(u32, String)> {
        let mut got = Vec::new();
        for i in 0..self.things.pickups.len() {
            let k = &mut self.things.pickups[i];
            let d = [k.pos[0] - pos[0], k.pos[2] - pos[2]];
            if k.taken || (d[0] * d[0] + d[1] * d[1]).sqrt() > PICK_UP || (k.pos[1] - pos[1]).abs() > TF {
                continue;
            }
            k.taken = true;
            got.push((k.entity, k.name.clone()));
            self.shape_pickup(i);
        }
        got
    }

    /// Use whatever is in front of someone with their feet at `pos`, facing
    /// `yaw` (radians), holding the keys `held`: the nearest door or lever in
    /// reach. Says what happened, or `None` when nothing is there.
    pub fn use_thing(&mut self, pos: [f32; 3], yaw: f32, held: &[(u32, String)]) -> Option<String> {
        let (s, c) = yaw.sin_cos();
        let fwd = [-s, c];
        // How far ahead a point is, if it is ahead, near and on this floor.
        let ahead = |p: [f32; 3]| {
            let (dx, dz) = (p[0] - pos[0], p[2] - pos[2]);
            let dist = (dx * dx + dz * dz).sqrt();
            let facing = dist < 1.0 || (dx * fwd[0] + dz * fwd[1]) / dist > 0.4;
            (dist < REACH && facing && (p[1] - pos[1]).abs() < TF).then_some(dist)
        };
        let door = (0..self.things.doors.len())
            .filter_map(|i| ahead(self.things.doors[i].centre).map(|d| (d, i)))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let lever = (0..self.things.levers.len())
            .filter_map(|i| ahead(self.things.levers[i].pos).map(|d| (d, i)))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match (door, lever) {
            (Some((dd, i)), l) if l.is_none_or(|(ld, _)| dd <= ld) => Some(self.use_door(i, held)),
            (_, Some((_, i))) => Some(self.pull_lever(i)),
            _ => None,
        }
    }

    fn use_door(&mut self, i: usize, held: &[(u32, String)]) -> String {
        let d = &mut self.things.doors[i];
        if d.locked && !d.opening {
            match held.iter().find(|k| Some(k.0) == d.key) {
                Some(k) => {
                    d.locked = false;
                    d.opening = true;
                    return format!("unlocked with the {}", k.1);
                }
                None => return "locked".into(),
            }
        }
        d.opening = !d.opening;
        if d.opening { "opens" } else { "closes" }.into()
    }

    fn pull_lever(&mut self, i: usize) -> String {
        let l = &mut self.things.levers[i];
        l.pulled = !l.pulled;
        let targets = l.targets.clone();
        self.shape_lever(i);
        let mut n = 0;
        for d in self.things.doors.iter_mut().filter(|d| targets.contains(&d.entity)) {
            d.opening = !d.opening;
            n += 1;
        }
        match n {
            0 => "the lever moves, and nothing happens".into(),
            _ => "the lever grinds, and somewhere a door moves".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::world::level::{Entity, EntityKind, Hinge, Level};
    use crate::world::scene::T;

    const TF: f32 = T as f32;

    /// A corridor two tiles long with a door across the middle, facing +z.
    fn corridor(locked: bool) -> Level {
        let mut l = Level::new("door");
        l.open(0, 0, 0, 12);
        l.open(0, 1, 0, 12);
        l.entities.push(Entity { id: 1, kind: EntityKind::PlayerStart, pos: [0.5 * TF, 0.0, 0.5 * TF], yaw: 0.0 });
        l.entities.push(Entity {
            id: 2,
            kind: EntityKind::Door { width: 1, height: 8, hinge: Hinge::Left, open: false, locked, key: Some(3) },
            pos: [0.5 * TF, 0.0, TF],
            yaw: 0.0,
        });
        l.entities.push(Entity { id: 3, kind: EntityKind::Key { name: "iron key".into() }, pos: [0.5 * TF, 0.0, 0.3 * TF], yaw: 0.0 });
        l
    }

    #[test]
    fn doors_block_until_open() {
        let mut s = corridor(false).to_scene();
        let feet = 0.0;
        assert!(s.door_blocks(0.5 * TF, 0.9 * TF, feet, 30.0));
        assert!(!s.door_blocks(0.5 * TF, 0.5 * TF, feet, 30.0));
        let before = s.things.parts[s.things.doors[0].part].tris[0].v[0].p;
        assert_eq!(s.use_thing([0.5 * TF, 0.0, 0.6 * TF], 0.0, &[]).as_deref(), Some("opens"));
        // Moving still blocks; open doesn't.
        assert!(s.tick_things(0.1));
        assert!(s.door_blocks(0.5 * TF, 0.9 * TF, feet, 30.0));
        assert_ne!(s.things.parts[s.things.doors[0].part].tris[0].v[0].p, before);
        for _ in 0..10 {
            s.tick_things(0.1);
        }
        assert!(!s.tick_things(0.1));
        assert!(!s.door_blocks(0.5 * TF, 0.9 * TF, feet, 30.0));
    }

    #[test]
    fn locked_doors_need_their_key() {
        let mut s = corridor(true).to_scene();
        let at = [0.5 * TF, 0.0, 0.6 * TF];
        assert_eq!(s.use_thing(at, 0.0, &[]).as_deref(), Some("locked"));
        assert!(!s.tick_things(0.1));
        // Facing away, nothing is in reach.
        assert_eq!(s.use_thing(at, std::f32::consts::PI, &[]), None);
        let keys = s.take_pickups([0.5 * TF, 0.0, 0.4 * TF]);
        assert_eq!(keys, vec![(3, "iron key".to_string())]);
        assert!(s.things.parts[s.things.pickups[0].part].tris.is_empty());
        assert!(s.take_pickups([0.5 * TF, 0.0, 0.4 * TF]).is_empty());
        assert_eq!(s.use_thing(at, 0.0, &keys).as_deref(), Some("unlocked with the iron key"));
        assert!(s.tick_things(0.1));
    }

    #[test]
    fn levers_work_their_doors() {
        let mut l = corridor(true);
        l.entities.push(Entity { id: 4, kind: EntityKind::Lever { targets: vec![2] }, pos: [0.5 * TF, 0.0, 0.2 * TF], yaw: 0.0 });
        let mut s = l.to_scene();
        let v = s.things.parts[s.things.levers[0].part].version;
        // Standing beside the lever, facing it (toward -z), with the door behind.
        let msg = s.use_thing([0.5 * TF, 0.0, 0.6 * TF], std::f32::consts::PI, &[]).unwrap();
        assert!(msg.contains("lever"), "{msg}");
        assert_ne!(s.things.parts[s.things.levers[0].part].version, v);
        assert!(s.things.doors[0].opening);
    }
}
