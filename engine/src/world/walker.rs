//! A King's Field-style first-person walker: fixed eye height and lens, no
//! roll, gaze mostly level. It picks somewhere to go, path-finds over the
//! cell grid (stairs and ramps yes, ledges no), walks cell centres, cutting
//! straight across open floor, and now and then stops to look around.

use crate::world::creature::Creature;
use crate::world::rng::Rng;
use crate::world::scene::{torch_colour, Camera, Node, Scene, MAX_STEP, T};
use std::collections::VecDeque;

const TF: f32 = T as f32;
/// Eye height above the floor.
pub const EYE: f32 = 0.62 * TF;
/// Focal length for a ~83° horizontal field of view. The near plane is H/2 = 90.
pub const H: u16 = 180;
/// Body radius for shortcuts: kept above H/2 so nearby walls are not cut by the near plane.
pub const RADIUS: f32 = 100.0;
/// A creature further above or below the feet than this is on another floor.
const CREATURE_FLOOR: f32 = 1.5 * TF;
const SPEED: f32 = 14.0;
/// Turn rate cap per frame (4096 = 360°).
const MAX_TURN: f32 = 40.0;
/// Furthest the gaze tilts up or down (512 = 45°).
pub const MAX_PITCH: f32 = 512.0;
/// Sprinting, as a multiple of walking speed; it ends when the view turns faster than
/// `SPRINT_TURN` (yaw units per frame), so a very slow turn is still allowed.
const SPRINT: f32 = 1.7;
const SPRINT_TURN: f32 = 12.0;
/// A jump: speed off the ground and the pull down, per frame (about 0.4 tiles high).
const JUMP: f32 = 23.0;
const GRAVITY: f32 = 2.5;
const FASTEST_FALL: f32 = 40.0;
/// How near the eye comes to a ceiling in a jump.
const HEAD: f32 = 0.15 * TF;
/// World units walked per bob of the camera, and how far it bobs.
const STRIDE: f32 = 0.9 * TF;
const BOB: f32 = 6.0;

/// What only a player's walk has (`Walker::drive`). `step` leaves it at rest, so the
/// generated walks' cameras have no bob.
#[derive(Default)]
struct Body {
    /// Falling speed (y is down), the floor jumped from (while in the air) and the
    /// last frame's yaw, for how fast the view turned.
    vy: f32,
    took_off: Option<f32>,
    sprinting: bool,
    last_yaw: Option<f32>,
    /// Radians along the bob's cycle, and how much of it shows (0 standing, 1 walking).
    bob: f32,
    bob_amount: f32,
}

/// A knock to the view, on top of where the walker is and looks: pushed `back` (world
/// units) and turned by `yaw`, `pitch` and `roll` (4096 units a turn). The game's own; the
/// generated walks' cameras have none.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Kick {
    pub back: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

pub struct Walker {
    pub x: f32,
    pub z: f32,
    /// Eye y, eased toward floor - EYE so stairs read as a smooth climb.
    pub y: f32,
    pub yaw: f32,
    pub pitch: f32,
    /// Remaining waypoints, nearest first.
    path: VecDeque<Node>,
    /// Frames left in a stop-and-look-around, and where it is looking.
    pause: u32,
    look_yaw: f32,
    target_pitch: f32,
    fog_near: i32,
    fog_far: i32,
    torch: [f32; 3],
    torch_radius: i32,
    body: Body,
    pub kick: Kick,
}

fn centre(c: Node) -> (f32, f32) {
    ((c.0 as f32 + 0.5) * TF, (c.1 as f32 + 0.5) * TF)
}

/// Yaw that faces along (dx, dz); the inverse of the GTE rot_y forward (-sin, cos).
pub fn yaw_towards(dx: f32, dz: f32) -> f32 {
    (-dx).atan2(dz) / std::f32::consts::TAU * 4096.0
}

/// Creatures on the same floor as feet at `at` and how near each lets the body come: its
/// base, and its girth and the body's together.
fn creatures_near(scene: &Scene, at: [f32; 3]) -> impl Iterator<Item = ([f32; 3], f32)> + '_ {
    scene
        .creatures
        .iter()
        .filter(move |c| c.girth > 0.0 && (c.base[1] - at[1]).abs() < CREATURE_FLOOR)
        .map(|c| (c.base, c.girth + RADIUS))
}

/// Would moving from `at` to (x, z) take the body into a creature? Only a move that comes
/// nearer counts, so a body a creature has walked into can always back out of it.
fn creature_in_way(scene: &Scene, at: [f32; 3], x: f32, z: f32) -> bool {
    creatures_near(scene, at).any(|(b, near)| {
        let to = (x - b[0]).hypot(z - b[2]);
        to < near && to < (at[0] - b[0]).hypot(at[2] - b[2])
    })
}

/// The move (dx, dz) from `at` with whatever would push into a creature taken out, so the
/// body slides round it.
fn slide_round_creatures(scene: &Scene, at: [f32; 3], mut dx: f32, mut dz: f32) -> (f32, f32) {
    for (b, near) in creatures_near(scene, at) {
        let (ox, oz) = (at[0] + dx - b[0], at[2] + dz - b[2]);
        let to = ox.hypot(oz);
        if to < near && to > 1e-3 {
            let (nx, nz) = (ox / to, oz / to);
            let inward = (dx * nx + dz * nz).min(0.0);
            (dx, dz) = (dx - inward * nx, dz - inward * nz);
        }
    }
    (dx, dz)
}

/// `target` shifted by whole turns to be nearest `from`.
fn unwrap(target: f32, from: f32) -> f32 {
    target - ((target - from) / 4096.0).round() * 4096.0
}

impl Walker {
    pub fn new(scene: &Scene, rng: &mut Rng) -> Option<Walker> {
        let cells = scene.floor_cells();
        let start = cells[rng.next() as usize % cells.len()];
        // Somewhere worth walking: at least a few cells reachable.
        if scene.reachable(start).len() < 6 {
            return None;
        }
        let (x, z) = centre(start);
        if scene.span(start).blocked {
            return None;
        }
        let fy = scene.node_floor(start, [x, z]);
        let fog_near = rng.fr(1.0, 2.5) * TF;
        Some(Walker {
            x,
            z,
            y: fy - EYE,
            yaw: rng.range(0, 4) as f32 * 1024.0,
            pitch: 0.0,
            path: VecDeque::new(),
            pause: 0,
            look_yaw: 0.0,
            target_pitch: 0.0,
            fog_near: fog_near as i32,
            fog_far: (fog_near + rng.fr(3.0, 6.0) * TF) as i32,
            torch: if rng.chance(0.75) { torch_colour(rng) } else { [0.0; 3] },
            torch_radius: (rng.fr(2.0, 4.0) * TF) as i32,
            body: Body::default(),
            kick: Kick::default(),
        })
    }

    /// A walker standing at `pos` (world units, on or above a floor) facing `yaw` radians.
    pub fn at(scene: &Scene, pos: [f32; 3], yaw: f32, rng: &mut Rng) -> Option<Walker> {
        let n = scene.node_at(pos[0], pos[1], pos[2])?;
        let fy = scene.node_floor(n, [pos[0], pos[2]]);
        let fog_near = rng.fr(1.5, 2.5) * TF;
        Some(Walker {
            x: pos[0],
            z: pos[2],
            y: fy - EYE,
            yaw: yaw / std::f32::consts::TAU * 4096.0,
            pitch: 0.0,
            path: VecDeque::new(),
            pause: 0,
            look_yaw: 0.0,
            target_pitch: 0.0,
            fog_near: fog_near as i32,
            fog_far: (fog_near + rng.fr(4.0, 6.0) * TF) as i32,
            torch: torch_colour(rng),
            torch_radius: (3.0 * TF) as i32,
            body: Body::default(),
            kick: Kick::default(),
        })
    }

    /// Choose a destination (biased toward far cells) and the cell path to it.
    fn plan(&mut self, scene: &Scene, rng: &mut Rng) {
        let Some(here) = scene.node_at(self.x, self.feet(), self.z) else { return };
        let tree = scene.reachable(here);
        // BFS order is distance order, so the back half is the far half.
        let pick = tree.len() / 2 + rng.next() as usize % tree.len().div_ceil(2);
        let mut c = tree[pick.min(tree.len() - 1)].0;
        let mut path = VecDeque::new();
        while c != here {
            path.push_front(c);
            c = tree.iter().find(|e| e.0 == c).unwrap().1;
        }
        self.path = path;
    }

    /// Height of the feet: the eye eases toward the floor, so this is near it.
    fn feet(&self) -> f32 {
        self.y + EYE
    }

    /// Can the body stand at (x, z) coming from floor height `from`? The centre may
    /// move at most a step up or down; the corners of the body only need floor that
    /// isn't steeper than 45° from it, so steep stairs pass but walls and pits don't.
    /// Closed doors are in the way.
    fn free(scene: &Scene, x: f32, z: f32, from: f32) -> Option<f32> {
        let fy = scene.walkable_y(x, from, z)?;
        let reach = MAX_STEP + RADIUS * std::f32::consts::SQRT_2;
        let ok = (fy - from).abs() <= MAX_STEP
            && !scene.door_blocks(x, z, fy, RADIUS)
            && [(-RADIUS, -RADIUS), (RADIUS, -RADIUS), (-RADIUS, RADIUS), (RADIUS, RADIUS)]
                .iter()
                .all(|&(dx, dz)| scene.walkable_y(x + dx, fy, z + dz).is_some_and(|f| (f - fy).abs() <= reach));
        ok.then_some(fy)
    }

    /// Can a body in the air with its feet at `feet` move to (x, z)? It can land on floor
    /// up to a step above its feet, but not over one lower than where it jumped from can
    /// step down to: nothing it jumps into is deeper than it could walk back out of.
    fn free_in_air(scene: &Scene, x: f32, z: f32, feet: f32, took_off: f32) -> bool {
        let Some(fy) = scene.walkable_y(x, feet, z) else { return false };
        let reach = MAX_STEP + RADIUS * std::f32::consts::SQRT_2;
        fy >= feet - MAX_STEP
            && fy <= took_off + MAX_STEP
            && !scene.door_blocks(x, z, feet.min(fy), RADIUS)
            && [(-RADIUS, -RADIUS), (RADIUS, -RADIUS), (-RADIUS, RADIUS), (RADIUS, RADIUS)]
                .iter()
                .all(|&(dx, dz)| scene.walkable_y(x + dx, feet, z + dz).is_some_and(|f| (f - fy).abs() <= reach))
    }

    /// Straight walk from here to (tx, tz) without touching walls or ledges?
    fn clear_line(&self, scene: &Scene, tx: f32, tz: f32) -> bool {
        let (dx, dz) = (tx - self.x, tz - self.z);
        let n = ((dx * dx + dz * dz).sqrt() / 16.0).ceil().max(1.0) as i32;
        let mut from = scene.floor_y(self.x, self.feet(), self.z).unwrap_or(0.0);
        (1..=n).all(|i| {
            let t = i as f32 / n as f32;
            Self::free(scene, self.x + dx * t, self.z + dz * t, from).map(|f| from = f).is_some()
        })
    }

    pub fn camera(&self) -> Camera {
        let k = self.kick;
        let (s, c) = (self.yaw / 4096.0 * std::f32::consts::TAU).sin_cos();
        // Forward is (-sin, cos).
        let (x, z) = (self.x + s * k.back, self.z - c * k.back);
        Camera {
            pos: [x as i32, (self.y + self.bob()) as i32, z as i32],
            yaw: (self.yaw + k.yaw).round() as i32,
            pitch: (self.pitch + k.pitch).round() as i32,
            roll: k.roll.round() as i32,
            h: H,
            fog_near: self.fog_near,
            fog_far: self.fog_far,
            torch: self.torch,
            torch_radius: self.torch_radius,
        }
    }

    /// A camera like this walker's, looking at somewhere on creature `c` from any
    /// side, mostly close: 0.4-2.4 tiles out from its outer edge, biased near. The
    /// eye is usually at walking height but often anywhere from a crouch to
    /// overhead, the gaze can tilt steeply, and sometimes the head rolls a little.
    pub fn view_of(&self, scene: &Scene, c: &Creature, rng: &mut Rng) -> Option<Camera> {
        let (reach, top, bottom) = c.extent();
        let own = ((c.base[0] / TF).floor(), (c.base[2] / TF).floor());
        let in_own = |x: f32, z: f32| ((x / TF).floor(), (z / TF).floor()) == own;
        for _ in 0..40 {
            let target = [c.base[0], bottom + (top - bottom) * rng.fr(0.15, 0.85), c.base[2]];
            let angle = rng.fr(0.0, std::f32::consts::TAU);
            let dist = reach + (0.4 + 2.0 * rng.f() * rng.f()) * TF;
            let (x, z) = (target[0] + angle.cos() * dist, target[2] + angle.sin() * dist);
            // Open floor under the eye and all round it, so the near plane cuts no wall.
            let Some(fy) = scene.floor_y(x, c.base[1], z) else { continue };
            let open = |px: f32, pz: f32| in_own(px, pz) || scene.walkable_y(px, fy, pz).is_some();
            if ![(0.0, 0.0), (RADIUS, 0.0), (-RADIUS, 0.0), (0.0, RADIUS), (0.0, -RADIUS)]
                .iter()
                .all(|&(ox, oz)| open(x + ox, z + oz))
            {
                continue;
            }
            let ceiling = scene.ceiling_y(x, fy, z).unwrap_or(fy - 2.0 * TF);
            let lift = if rng.chance(0.5) { EYE + rng.fr(-0.08, 0.08) * TF } else { rng.fr(0.15, 1.5) * TF };
            let lift = lift.min(fy - ceiling - 0.2 * TF);
            if lift < 0.1 * TF {
                continue;
            }
            let y = fy - lift;
            // Nothing solid along the line of sight, skipping the creature's own cell.
            let (dx, dy, dz) = (target[0] - x, target[1] - y, target[2] - z);
            let n = (dist / 32.0).ceil() as i32;
            let clear = (1..n).all(|i| {
                let t = i as f32 / n as f32;
                let (px, py, pz) = (x + dx * t, y + dy * t, z + dz * t);
                in_own(px, pz)
                    || scene.node_at(px, py, pz).is_some_and(|n| {
                        let s = scene.span(n);
                        s.floor(n.0, n.1, [px, pz]) > py && (s.sky || s.ceiling(n.0, n.1, [px, pz]) < py)
                    })
            });
            if !clear {
                continue;
            }
            let pitch = dy.atan2(dist) / std::f32::consts::TAU * 4096.0;
            return Some(Camera {
                pos: [x as i32, y as i32, z as i32],
                yaw: (yaw_towards(dx, dz) + rng.fr(-120.0, 120.0)).round() as i32,
                pitch: (pitch + rng.fr(-60.0, 60.0)).clamp(-900.0, 900.0).round() as i32,
                roll: if rng.chance(0.2) { rng.range(-120, 120) } else { 0 },
                ..self.camera()
            });
        }
        None
    }

    /// A camera like this walker's, close in front of something at `at` on a wall facing `n`,
    /// looking at it from up to about 50 degrees off square. None if there's nowhere to stand.
    pub fn view_of_wall(&self, scene: &Scene, at: [f32; 3], n: [f32; 3], rng: &mut Rng) -> Option<Camera> {
        for _ in 0..40 {
            let dist = (0.7 + 2.6 * rng.f() * rng.f()) * TF;
            let out = n[0].atan2(n[2]);
            // Off a ceiling, the way out is any way at all.
            let a = if n[1].abs() > 0.8 { rng.fr(0.0, std::f32::consts::TAU) } else { out + rng.fr(-0.9, 0.9) };
            let (x, z) = (at[0] + a.sin() * dist, at[2] + a.cos() * dist);
            let Some(fy) = scene.floor_y(x, at[1], z) else { continue };
            if ![(0.0, 0.0), (RADIUS, 0.0), (-RADIUS, 0.0), (0.0, RADIUS), (0.0, -RADIUS)].iter().all(|&(ox, oz)| scene.walkable_y(x + ox, fy, z + oz).is_some()) {
                continue;
            }
            let ceiling = scene.ceiling_y(x, fy, z).unwrap_or(fy - 2.0 * TF);
            let lift = (EYE + rng.fr(-0.08, 0.08) * TF).min(fy - ceiling - 0.2 * TF);
            if lift < 0.1 * TF {
                continue;
            }
            let y = fy - lift;
            let (dx, dy, dz) = (at[0] - x, at[1] - y, at[2] - z);
            // The last stretch is the flesh standing out of the wall.
            let n = (dist / 32.0).ceil() as i32;
            let clear = (1..n).filter(|&i| i as f32 / n as f32 <= 1.0 - 0.3 * TF / dist).all(|i| {
                let t = i as f32 / n as f32;
                let (px, py, pz) = (x + dx * t, y + dy * t, z + dz * t);
                scene.node_at(px, py, pz).is_some_and(|n| {
                    let s = scene.span(n);
                    s.floor(n.0, n.1, [px, pz]) > py && (s.sky || s.ceiling(n.0, n.1, [px, pz]) < py)
                })
            });
            if !clear {
                continue;
            }
            let pitch = dy.atan2(dist) / std::f32::consts::TAU * 4096.0;
            return Some(Camera {
                pos: [x as i32, y as i32, z as i32],
                yaw: (yaw_towards(dx, dz) + rng.fr(-100.0, 100.0)).round() as i32,
                pitch: (pitch + rng.fr(-60.0, 60.0)).clamp(-900.0, 900.0).round() as i32,
                roll: 0,
                ..self.camera()
            });
        }
        None
    }

    /// How far the walking bob moves the camera now (world units, down is positive).
    pub fn bob(&self) -> f32 {
        self.body.bob.sin() * BOB * self.body.bob_amount
    }

    pub fn airborne(&self) -> bool {
        self.body.took_off.is_some()
    }

    pub fn sprinting(&self) -> bool {
        self.body.sprinting
    }

    /// Leave the ground, if standing on it; `drive` carries the jump through.
    pub fn jump(&mut self, scene: &Scene) {
        if self.body.took_off.is_none() {
            self.body.took_off = Some(scene.floor_y(self.x, self.feet(), self.z).unwrap_or(self.feet()));
            self.body.vy = -JUMP;
        }
    }

    /// Run until the walk forward stops or the view turns too fast (see `SPRINT_TURN`).
    pub fn start_sprint(&mut self) {
        self.body.sprinting = true;
    }

    pub fn step(&mut self, scene: &Scene, rng: &mut Rng) {
        // Gaze: usually level, sometimes a glance up or down.
        if rng.chance(0.012) {
            self.target_pitch = rng.fr(-MAX_PITCH, MAX_PITCH);
        } else if rng.chance(0.02) {
            self.target_pitch = 0.0;
        }
        self.pitch += (self.target_pitch - self.pitch) * 0.06;

        if self.pause > 0 {
            self.pause -= 1;
            let want = unwrap(self.look_yaw, self.yaw);
            self.yaw += ((want - self.yaw) * 0.06).clamp(-MAX_TURN, MAX_TURN);
            return;
        }
        if self.path.is_empty() {
            self.plan(scene, rng);
            if self.path.is_empty() {
                // Nowhere to go: look around instead.
                self.pause = 60;
                self.look_yaw = self.yaw + rng.fr(-2048.0, 2048.0);
                return;
            }
        }
        // Cut corners: aim at the furthest waypoint in a clear straight line.
        while self.path.len() > 1 {
            let (tx, tz) = centre(self.path[1]);
            if !self.clear_line(scene, tx, tz) {
                break;
            }
            self.path.pop_front();
        }
        let (tx, tz) = centre(self.path[0]);
        let (dx, dz) = (tx - self.x, tz - self.z);
        let dist = (dx * dx + dz * dz).sqrt();
        if dist > 1.0 {
            let want = unwrap(yaw_towards(dx, dz), self.yaw);
            let diff = want - self.yaw;
            self.yaw += (diff * 0.15).clamp(-MAX_TURN, MAX_TURN);
            // Turn first, then walk, as in King's Field.
            if diff.abs() < 300.0 {
                let s = SPEED.min(dist);
                self.x += dx / dist * s;
                self.z += dz / dist * s;
            }
        }
        if dist <= SPEED {
            self.path.pop_front();
            if rng.chance(0.06) {
                self.pause = rng.range(40, 90) as u32;
                let sweep = rng.fr(500.0, 2048.0);
                self.look_yaw = self.yaw + if rng.chance(0.5) { sweep } else { -sweep };
            }
        }
        if let Some(fy) = scene.floor_y(self.x, self.feet(), self.z) {
            self.y += (fy - EYE - self.y) * 0.35;
        }
    }

    /// One frame of player control instead of `step`: `forward` and `strafe`
    /// in -1..1 of walking speed, `turn` in -1..1 of the turn rate (positive =
    /// left: yaw grows anticlockwise seen from above), `look` the pitch to ease toward. Slides along walls; drops any
    /// planned walk, so `step` replans from wherever this leaves the body.
    pub fn drive(&mut self, scene: &Scene, forward: f32, strafe: f32, turn: f32, look: f32) {
        self.path.clear();
        self.pause = 0;
        self.target_pitch = look;
        self.pitch += (look - self.pitch) * 0.15;
        self.yaw += turn * MAX_TURN;
        let turned = self.body.last_yaw.map_or(0.0, |y| (self.yaw - y).abs());
        self.body.last_yaw = Some(self.yaw);
        if forward < 0.3 || turned > SPRINT_TURN {
            self.body.sprinting = false;
        }
        let speed = if self.body.sprinting { SPEED * SPRINT } else { SPEED };
        let a = self.yaw / 4096.0 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        // Forward is (-sin, cos) as in `yaw_towards`; right is down x forward.
        let dx = (-s * forward + c * strafe) * speed;
        let dz = (c * forward + s * strafe) * speed;
        let from = scene.floor_y(self.x, self.feet(), self.z).unwrap_or(0.0);
        let feet = self.feet();
        let took_off = self.body.took_off;
        let (x0, z0) = (self.x, self.z);
        let (dx, dz) = slide_round_creatures(scene, [x0, feet, z0], dx, dz);
        for (mx, mz) in [(dx, dz), (dx, 0.0), (0.0, dz)] {
            let (x, z) = (self.x + mx, self.z + mz);
            let free = match took_off {
                Some(t) => Self::free_in_air(scene, x, z, feet, t),
                None => Self::free(scene, x, z, from).is_some(),
            } && !creature_in_way(scene, [x0, feet, z0], x, z);
            if (mx != 0.0 || mz != 0.0) && free {
                self.x = x;
                self.z = z;
                break;
            }
        }
        if took_off.is_some() {
            self.fall(scene);
        } else if let Some(fy) = scene.floor_y(self.x, self.feet(), self.z) {
            self.y += (fy - EYE - self.y) * 0.35;
        }
        let moved = (self.x - x0).hypot(self.z - z0);
        let walking = (moved > 0.5 && self.body.took_off.is_none()) as i32 as f32;
        self.body.bob = (self.body.bob + moved / STRIDE * std::f32::consts::TAU) % std::f32::consts::TAU;
        self.body.bob_amount += (walking - self.body.bob_amount) * 0.2;
    }

    /// One frame of a jump: rise and fall, stopping at the ceiling and landing on the floor.
    fn fall(&mut self, scene: &Scene) {
        let before = self.feet();
        self.body.vy = (self.body.vy + GRAVITY).min(FASTEST_FALL);
        self.y += self.body.vy;
        if let Some(c) = scene.ceiling_y(self.x, before, self.z) {
            if self.y < c + HEAD {
                self.y = c + HEAD;
                self.body.vy = self.body.vy.max(0.0);
            }
        }
        match scene.floor_y(self.x, before, self.z) {
            Some(fy) if self.feet() >= fy && self.body.vy > 0.0 => {
                self.y = fy - EYE;
                self.body.vy = 0.0;
                self.body.took_off = None;
            }
            // Pushed into rock somehow: stand where it was.
            None => {
                self.body.vy = 0.0;
                self.body.took_off = None;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::level::{Entity, EntityKind, Level};

    /// A long room, 12 layers high, with the player at one end facing along it.
    fn hall() -> (Scene, Walker) {
        let mut l = Level::new("hall");
        for z in 0..20 {
            for x in 0..3 {
                l.open(x, z, 0, 12);
            }
        }
        l.entities.push(Entity { id: 1, kind: EntityKind::PlayerStart, pos: [1.5 * TF, 0.0, 1.5 * TF], yaw: 0.0 });
        let scene = l.to_scene();
        let w = Walker::at(&scene, [1.5 * TF, 0.0, 1.5 * TF], 0.0, &mut Rng::new(1)).expect("standing");
        (scene, w)
    }

    /// The hall with a creature standing in the way, five tiles ahead, free to move.
    fn hall_with_creature() -> (Scene, Walker) {
        let (_, w) = hall();
        let mut l = Level::new("hall");
        for z in 0..20 {
            for x in 0..3 {
                l.open(x, z, 0, 12);
            }
        }
        let kind = crate::world::creature::Kind::Humanoid;
        l.entities.push(Entity { id: 2, kind: EntityKind::Creature { kind, size: 1.0, seed: 3 }, pos: [1.5 * TF, 0.0, 6.5 * TF], yaw: 0.0 });
        let mut scene = l.to_scene();
        scene.set_creatures_move(true);
        (scene, w)
    }

    #[test]
    fn creatures_stop_the_body_wherever_they_stand_and_the_dead_do_not() {
        let (mut scene, mut w) = hall_with_creature();
        let c = scene.creatures[0].clone();
        assert!(c.girth > 0.0);
        for _ in 0..200 {
            w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        }
        let gap = (c.base[2] - w.z).hypot(c.base[0] - w.x);
        assert!(gap >= c.girth + RADIUS - 1.0 && gap < c.girth + RADIUS + 2.0 * SPEED, "stopped {gap} from it");
        // Walked off, it leaves nothing behind where it stood; and where it went, it blocks.
        scene.creatures[0].base[2] = 15.5 * TF;
        for _ in 0..200 {
            w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        }
        assert!(w.z > 8.0 * TF && w.z < 15.5 * TF - c.girth - RADIUS + 1.0, "at {}", w.z / TF);
        scene.creatures[0].girth = 0.0;
        for _ in 0..200 {
            w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        }
        assert!(w.z > 16.0 * TF, "the dead are in the way at {}", w.z / TF);
    }

    #[test]
    fn a_body_slides_round_a_creature_and_backs_out_of_one() {
        let (scene, mut w) = hall_with_creature();
        let c = scene.creatures[0].clone();
        // Walking at it a little to one side slides round its edge rather than sticking.
        let near = c.girth + RADIUS;
        w.x = 1.5 * TF + 0.3 * near;
        let gap = |w: &Walker| (c.base[2] - w.z).hypot(c.base[0] - w.x);
        for _ in 0..200 {
            if gap(&w) < near + SPEED {
                break;
            }
            w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        }
        let before = w.x;
        for _ in 0..3 {
            w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
            assert!(gap(&w) >= near - 1.0);
        }
        assert!(w.x > before + 1.0, "stuck on its front");
        // Caught inside one (it walked into the body), the body can still step away.
        w.x = c.base[0];
        w.z = c.base[2] - 0.5 * c.girth;
        let z = w.z;
        w.drive(&scene, -1.0, 0.0, 0.0, 0.0);
        assert!(w.z < z);
    }

    #[test]
    fn a_jump_rises_and_lands_where_it_started() {
        let (scene, mut w) = hall();
        let (standing, mut top) = (w.y, w.y);
        w.jump(&scene);
        assert!(w.airborne());
        for _ in 0..40 {
            w.drive(&scene, 0.0, 0.0, 0.0, 0.0);
            top = top.min(w.y);
        }
        assert!(!w.airborne(), "still in the air");
        assert_eq!(w.y, standing);
        let height = (standing - top) / TF;
        assert!((0.3..0.55).contains(&height), "jumped {height} tiles");
    }

    #[test]
    fn no_second_jump_in_the_air() {
        let (scene, mut w) = hall();
        w.jump(&scene);
        for _ in 0..3 {
            w.drive(&scene, 0.0, 0.0, 0.0, 0.0);
        }
        let v = w.body.vy;
        w.jump(&scene);
        assert_eq!(w.body.vy, v);
    }

    #[test]
    fn sprint_is_faster_and_stops_when_turning_fast() {
        let (scene, mut w) = hall();
        let z0 = w.z;
        w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        let walked = w.z - z0;
        w.start_sprint();
        let z1 = w.z;
        w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        assert!((w.z - z1) > walked * 1.5, "{} against {walked}", w.z - z1);
        // Turning slowly keeps it going, and a fast turn ends it.
        w.yaw += SPRINT_TURN * 0.5;
        w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        assert!(w.sprinting());
        w.yaw += SPRINT_TURN * 2.0;
        w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        assert!(!w.sprinting());
    }

    #[test]
    fn sprint_stops_with_the_walk() {
        let (scene, mut w) = hall();
        w.start_sprint();
        w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        assert!(w.sprinting());
        w.drive(&scene, 0.0, 0.0, 0.0, 0.0);
        assert!(!w.sprinting());
    }

    #[test]
    fn the_camera_bobs_only_while_walking() {
        let (scene, mut w) = hall();
        let standing = w.y;
        let mut most = 0f32;
        for _ in 0..30 {
            w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
            most = most.max(w.bob().abs());
        }
        assert!(most > BOB * 0.5, "bobbed {most}");
        assert_eq!(w.y, standing, "the bob is the camera's, not the body's");
        for _ in 0..60 {
            w.drive(&scene, 0.0, 0.0, 0.0, 0.0);
        }
        assert!(w.bob().abs() < 0.1);
    }

    #[test]
    fn step_keeps_the_camera_as_it_was() {
        let (scene, mut w) = hall();
        let mut rng = Rng::new(3);
        for _ in 0..60 {
            w.step(&scene, &mut rng);
            assert_eq!(w.bob(), 0.0);
        }
    }

    /// Corridors plan their climbs, so every room should be walkable from the first.
    #[test]
    fn rooms_are_connected() {
        // Only the layout matters here; skip sculpting the creatures.
        crate::world::flesh::set_psx(true);
        let (mut connected, mut total) = (0, 0);
        for seed in 0..300 {
            let mut rng = Rng::new(seed);
            let scene = Scene::generate(&mut rng);
            let first = &scene.rooms[0];
            let start = (first.x + first.w / 2, first.z + first.d / 2, 0);
            let reach: Vec<_> = scene.reachable(start).into_iter().map(|e| e.0).collect();
            for r in &scene.rooms[1..] {
                total += 1;
                let inside = |c: &Node| c.0 >= r.x && c.0 < r.x + r.w && c.1 >= r.z && c.1 < r.z + r.d;
                connected += reach.iter().any(inside) as usize;
            }
        }
        eprintln!("{connected}/{total} rooms reachable from room 0");
        assert!(connected * 100 >= total * 97, "{connected}/{total}");
    }
}
