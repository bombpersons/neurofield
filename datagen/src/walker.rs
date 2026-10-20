//! A King's Field-style first-person walker: fixed eye height and lens, no
//! roll, gaze mostly level. It picks somewhere to go, path-finds over the
//! cell grid (stairs and ramps yes, ledges no), walks cell centres, cutting
//! straight across open floor, and now and then stops to look around.

use crate::rng::Rng;
use crate::scene::{torch_colour, Camera, Scene, MAX_STEP, T};
use std::collections::VecDeque;

const TF: f32 = T as f32;
/// Eye height above the floor.
pub const EYE: f32 = 0.62 * TF;
/// Focal length for a ~83° horizontal field of view. The near plane is H/2 = 90.
pub const H: u16 = 180;
/// Body radius for shortcuts: kept above H/2 so nearby walls are not cut by the near plane.
const RADIUS: f32 = 100.0;
const SPEED: f32 = 14.0;
/// Turn rate cap per frame (4096 = 360°).
const MAX_TURN: f32 = 40.0;
/// Furthest the gaze tilts up or down (512 = 45°).
pub const MAX_PITCH: f32 = 512.0;

pub struct Walker {
    pub x: f32,
    pub z: f32,
    /// Eye y, eased toward floor - EYE so stairs read as a smooth climb.
    pub y: f32,
    pub yaw: f32,
    pub pitch: f32,
    /// Remaining waypoints (cell coordinates), nearest first.
    path: VecDeque<(i32, i32)>,
    /// Frames left in a stop-and-look-around, and where it is looking.
    pause: u32,
    look_yaw: f32,
    target_pitch: f32,
    fog_near: i32,
    fog_far: i32,
    torch: [f32; 3],
    torch_radius: i32,
}

fn centre(c: (i32, i32)) -> (f32, f32) {
    ((c.0 as f32 + 0.5) * TF, (c.1 as f32 + 0.5) * TF)
}

/// Yaw that faces along (dx, dz); the inverse of the GTE rot_y forward (-sin, cos).
fn yaw_towards(dx: f32, dz: f32) -> f32 {
    (-dx).atan2(dz) / std::f32::consts::TAU * 4096.0
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
        let fy = scene.walkable_y(x, z)?;
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
        })
    }

    /// Choose a destination (biased toward far cells) and the cell path to it.
    fn plan(&mut self, scene: &Scene, rng: &mut Rng) {
        let here = ((self.x / TF).floor() as i32, (self.z / TF).floor() as i32);
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

    /// Can the body stand at (x, z) coming from floor height `from`?
    fn free(scene: &Scene, x: f32, z: f32, from: f32) -> Option<f32> {
        let fy = scene.walkable_y(x, z)?;
        let ok = (fy - from).abs() <= MAX_STEP
            && [(-RADIUS, -RADIUS), (RADIUS, -RADIUS), (-RADIUS, RADIUS), (RADIUS, RADIUS)]
                .iter()
                .all(|&(dx, dz)| scene.walkable_y(x + dx, z + dz).is_some_and(|f| (f - fy).abs() <= MAX_STEP));
        ok.then_some(fy)
    }

    /// Straight walk from here to (tx, tz) without touching walls or ledges?
    fn clear_line(&self, scene: &Scene, tx: f32, tz: f32) -> bool {
        let (dx, dz) = (tx - self.x, tz - self.z);
        let n = ((dx * dx + dz * dz).sqrt() / 16.0).ceil().max(1.0) as i32;
        let mut from = scene.floor_y(self.x, self.z).unwrap_or(0.0);
        (1..=n).all(|i| {
            let t = i as f32 / n as f32;
            Self::free(scene, self.x + dx * t, self.z + dz * t, from).map(|f| from = f).is_some()
        })
    }

    pub fn camera(&self) -> Camera {
        Camera {
            pos: [self.x as i32, self.y as i32, self.z as i32],
            yaw: self.yaw.round() as i32,
            pitch: self.pitch.round() as i32,
            roll: 0,
            h: H,
            fog_near: self.fog_near,
            fog_far: self.fog_far,
            torch: self.torch,
            torch_radius: self.torch_radius,
        }
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
        if let Some(fy) = scene.floor_y(self.x, self.z) {
            self.y += (fy - EYE - self.y) * 0.35;
        }
    }

    /// One frame of player control instead of `step`: `forward` and `strafe`
    /// in -1..1 of walking speed, `turn` in -1..1 of the turn rate (positive =
    /// right), `look` the pitch to ease toward. Slides along walls; drops any
    /// planned walk, so `step` replans from wherever this leaves the body.
    pub fn drive(&mut self, scene: &Scene, forward: f32, strafe: f32, turn: f32, look: f32) {
        self.path.clear();
        self.pause = 0;
        self.target_pitch = look;
        self.pitch += (look - self.pitch) * 0.15;
        self.yaw += turn * MAX_TURN;
        let a = self.yaw / 4096.0 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        // Forward is (-sin, cos) as in `yaw_towards`; right is down x forward.
        let dx = (-s * forward + c * strafe) * SPEED;
        let dz = (c * forward + s * strafe) * SPEED;
        let from = scene.floor_y(self.x, self.z).unwrap_or(0.0);
        for (mx, mz) in [(dx, dz), (dx, 0.0), (0.0, dz)] {
            if (mx != 0.0 || mz != 0.0) && Self::free(scene, self.x + mx, self.z + mz, from).is_some() {
                self.x += mx;
                self.z += mz;
                break;
            }
        }
        if let Some(fy) = scene.floor_y(self.x, self.z) {
            self.y += (fy - EYE - self.y) * 0.35;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corridors plan their climbs, so every room should be walkable from the first.
    #[test]
    fn rooms_are_connected() {
        let (mut connected, mut total) = (0, 0);
        for seed in 0..300 {
            let mut rng = Rng::new(seed);
            let scene = Scene::generate(&mut rng);
            let first = &scene.rooms[0];
            let start = (first.x + first.w / 2, first.z + first.d / 2);
            let reach: Vec<_> = scene.reachable(start).into_iter().map(|e| e.0).collect();
            for r in &scene.rooms[1..] {
                total += 1;
                let inside = |c: &(i32, i32)| c.0 >= r.x && c.0 < r.x + r.w && c.1 >= r.z && c.1 < r.z + r.d;
                connected += reach.iter().any(inside) as usize;
            }
        }
        eprintln!("{connected}/{total} rooms reachable from room 0");
        assert!(connected * 100 >= total * 97, "{connected}/{total}");
    }
}
