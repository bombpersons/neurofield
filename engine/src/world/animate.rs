//! Creatures moving by themselves, whatever they grew into. Each one's body plan is read off
//! its rest pose: chains of parts that reach the floor are legs, which plant their feet and
//! step in turn, the rest of the body following; without legs it creeps or floats. Motion is
//! written to the creature itself (`base`, `yaw` and each part's `rot`, `bend` and `half`), so
//! whatever reads `Creature::pose` sees it. Other motions (attacks, spells) can be laid over
//! the same joints after `Animator::update`.

use super::creature::{add, scale, sub, Creature, Shape};
use super::rng::Rng;
use super::scene::{apply, dot, mul, norm, rot_x, rot_y, rot_z, Scene, M3, MAX_STEP, T};
use std::f32::consts::{PI, TAU};

const TF: f32 = T as f32;
/// A part this near the floor (as a fraction of the creature's height) can stand on it.
const CONTACT: f32 = 0.12;
/// Furthest a creature strays from where it was grown.
const LEASH: f32 = 4.0 * TF;
const TURN_RATE: f32 = 1.8;
const CREEP_SPEED: f32 = 0.25 * TF;
const FLOAT_SPEED: f32 = 0.4 * TF;

/// How a creature gets about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    Walk,
    Creep,
    Float,
}

fn length(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    add(a, scale(sub(b, a), t))
}

fn transpose(m: M3) -> M3 {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| m[c][r]))
}

/// The smallest rotation taking unit vector `a` to unit vector `b`.
fn arc(a: [f32; 3], b: [f32; 3]) -> M3 {
    let [x, y, z] = super::creature::cross(a, b);
    let c = dot(a, b).max(-0.999);
    let k = 1.0 / (1.0 + c);
    [
        [1.0 - k * (y * y + z * z), k * x * y - z, k * x * z + y],
        [k * x * y + z, 1.0 - k * (x * x + z * z), k * y * z - x],
        [k * x * z - y, k * y * z + x, 1.0 - k * (x * x + y * y)],
    ]
}

/// Moves `p`'s joints so the chain's end reaches `target`, keeping bone `lengths` and the root.
fn reach(p: &mut [[f32; 3]], lengths: &[f32], target: [f32; 3]) {
    let root = p[0];
    let n = lengths.len();
    if length(sub(target, root)) >= lengths.iter().sum::<f32>() {
        let dir = norm(sub(target, root));
        for i in 0..n {
            p[i + 1] = add(p[i], scale(dir, lengths[i]));
        }
        return;
    }
    for _ in 0..10 {
        p[n] = target;
        for i in (0..n).rev() {
            p[i] = add(p[i + 1], scale(norm(sub(p[i], p[i + 1])), lengths[i]));
        }
        p[0] = root;
        for i in 0..n {
            p[i + 1] = add(p[i], scale(norm(sub(p[i + 1], p[i])), lengths[i]));
        }
    }
}

struct Leg {
    /// Parts from the hip down, the last possibly a foot.
    chain: Vec<usize>,
    /// The chain's last part when it is a foot, which stays level instead of bending.
    foot: Option<usize>,
    /// Each bone's direction in its part's frame, and its length.
    dirs: Vec<[f32; 3]>,
    lengths: Vec<f32>,
    /// The chain's end as a point on the last bone's part (when there is no foot).
    tip: [f32; 3],
    /// Where the foot rests in the creature's frame (x, z), and how far the chain's end
    /// stays above the floor.
    home: [f32; 2],
    lift: f32,
    /// The foot's orientation in the creature's frame.
    foot_axes: M3,
    /// Legs of one group step while the other stands.
    group: usize,
    /// The chain's end in the world.
    at: [f32; 3],
    /// A step in progress: where it left from and how far along (0..1).
    step: Option<([f32; 3], f32)>,
}

/// A part that isn't a leg, and how it sways.
struct Free {
    part: usize,
    sway: f32,
    bend: f32,
    speed: f32,
    phase: f32,
    /// Hanging from the body: swings against the legs as it walks (with the side's sign).
    arm: Option<f32>,
}

pub struct Animator {
    pub plan: Plan,
    rest: Vec<(M3, f32, [f32; 3])>,
    legs: Vec<Leg>,
    free: Vec<Free>,
    leg_length: f32,
    step_time: f32,
    threshold: f32,
    top_speed: f32,
    radius: f32,
    spawn: [f32; 3],
    /// Lift above the smoothed floor is carried by the root's anchor; this is that floor.
    floor: f32,
    rng: Rng,
    time: f32,
    speed: f32,
    going: bool,
    wait: f32,
    want_yaw: f32,
    pitch: f32,
}

impl Animator {
    pub fn new(c: &Creature) -> Animator {
        let placed = c.pose();
        let n = c.parts.len();
        let floor = c.base[1];
        let frame = rot_y(c.yaw);
        let to_body = transpose(frame);
        let (mut top, mut bottoms) = (f32::MAX, vec![f32::MIN; n]);
        for (i, (p, at)) in c.parts.iter().zip(&placed).enumerate() {
            for k in 0..=8 {
                for j in 0..8 {
                    let y = at.world(p.local(k as f32 / 4.0 - 1.0, j as f32 / 8.0 * TAU, 1.0))[1];
                    top = top.min(y);
                    bottoms[i] = bottoms[i].max(y);
                }
            }
        }
        let height = (floor - top).max(1.0);
        let mut children = vec![Vec::new(); n];
        for (i, p) in c.parts.iter().enumerate() {
            if let Some(a) = p.parent {
                children[a].push(i);
            }
        }
        let volume = |i: usize| c.parts[i].half.iter().product::<f32>();
        // The trunk: from the root, on through each part's far biggest child.
        let mut trunk = vec![false; n];
        let mut at = 0;
        trunk[0] = true;
        while let Some(&big) = children[at].iter().max_by(|&&a, &&b| volume(a).total_cmp(&volume(b))) {
            if volume(big) < 0.45 * volume(at) {
                break;
            }
            trunk[big] = true;
            at = big;
        }
        // Whatever branches off the trunk and gets down to the floor is a leg.
        let mut legs = Vec::new();
        let mut in_leg = vec![false; n];
        for t in (0..n).filter(|&t| trunk[t]) {
            for &start in children[t].iter().filter(|&&b| !trunk[b]) {
                let (mut stack, mut lowest) = (vec![start], start);
                while let Some(i) = stack.pop() {
                    if bottoms[i] > bottoms[lowest] {
                        lowest = i;
                    }
                    stack.extend(&children[i]);
                }
                if floor - bottoms[lowest] > CONTACT * height {
                    continue;
                }
                let mut chain = vec![lowest];
                while *chain.last().unwrap() != start {
                    chain.push(c.parts[*chain.last().unwrap()].parent.unwrap());
                }
                chain.reverse();
                if chain.iter().any(|&i| c.parts[i].shape == Shape::Tentacle) || !chain.iter().any(|&i| c.parts[i].shape == Shape::Capsule) {
                    continue;
                }
                let last = *chain.last().unwrap();
                let foot = (chain.len() >= 3 && c.parts[last].half[1] < 0.6 * c.parts[chain[chain.len() - 2]].half[1]).then_some(last);
                let bones = chain.len() - foot.is_some() as usize;
                let tip_part = chain[bones - 1];
                let tip = c.parts[tip_part].axis(1.0);
                let end = match foot {
                    Some(f) => placed[f].joint,
                    None => placed[tip_part].world(tip),
                };
                let mut joints: Vec<[f32; 3]> = chain[..bones].iter().map(|&i| placed[i].joint).collect();
                joints.push(end);
                let lengths: Vec<f32> = joints.windows(2).map(|w| length(sub(w[1], w[0])).max(1.0)).collect();
                let hip = joints[0];
                if floor - hip[1] < 0.15 * height || lengths.iter().sum::<f32>() < 0.7 * (floor - hip[1]) {
                    continue;
                }
                let dirs = (0..bones).map(|i| norm(apply(transpose(placed[chain[i]].axes), sub(joints[i + 1], joints[i])))).collect();
                let home = apply(to_body, sub(end, c.base));
                for &i in &chain {
                    in_leg[i] = true;
                }
                legs.push(Leg {
                    foot,
                    foot_axes: mul(to_body, placed[last].axes),
                    dirs,
                    lengths,
                    tip,
                    home: [home[0], home[2]],
                    lift: (bottoms[last] - end[1]).max(0.0),
                    chain,
                    group: 0,
                    at: end,
                    step: None,
                });
            }
        }
        if legs.len() < 2 {
            legs.clear();
            in_leg.fill(false);
        }
        // Take turns down each side, the two sides out of step.
        for side in [false, true] {
            let mut along: Vec<usize> = (0..legs.len()).filter(|&i| (legs[i].home[0] > 0.0) == side).collect();
            along.sort_by(|&a, &b| legs[a].home[1].total_cmp(&legs[b].home[1]));
            for (rank, i) in along.into_iter().enumerate() {
                legs[i].group = (rank + side as usize) % 2;
            }
        }
        let plan = if !legs.is_empty() {
            Plan::Walk
        } else if floor - bottoms[0] > 1.5 * c.parts[0].half[1] {
            Plan::Float
        } else {
            Plan::Creep
        };

        let mut rng = Rng::new(c.parts[0].seed as u64 + 1);
        // What rides on a leg moves with it, unswayed.
        let mut riding = in_leg.clone();
        for i in 0..n {
            if let Some(a) = c.parts[i].parent {
                riding[i] |= riding[a];
            }
        }
        let mut free = Vec::new();
        for (i, p) in c.parts.iter().enumerate() {
            if p.parent.is_none() || riding[i] || trunk[i] {
                continue;
            }
            let (sway, bend) = match p.shape {
                Shape::Tentacle => (0.2, 0.45),
                Shape::Capsule => (0.05, 0.0),
                Shape::Cone => (0.04, 0.1),
                Shape::Ellipsoid => (0.015, 0.0),
            };
            let down = placed[i].axes[1][1];
            let on_trunk = trunk[p.parent.unwrap()];
            let side = apply(to_body, sub(placed[i].joint, c.base))[0];
            let arm = (p.shape == Shape::Capsule && on_trunk && down > 0.5).then(|| if side > 0.0 { 1.0 } else { -1.0 });
            free.push(Free { part: i, sway, bend, speed: rng.fr(1.2, 2.6), phase: rng.fr(0.0, TAU), arm });
        }

        let leg_length = if legs.is_empty() { 0.5 * TF } else { legs.iter().map(|l| l.lengths.iter().sum::<f32>()).sum::<f32>() / legs.len() as f32 };
        let step_time = 0.34 * (leg_length / (0.5 * TF)).max(0.2).sqrt();
        let threshold = 0.42 * leg_length;
        let top_speed = match plan {
            Plan::Walk => 0.7 * threshold / step_time,
            Plan::Creep => CREEP_SPEED,
            Plan::Float => FLOAT_SPEED,
        };
        let radius = placed.iter().map(|p| (p.centre[0] - c.base[0]).hypot(p.centre[2] - c.base[2])).fold(0.0, f32::max);
        let wait = rng.fr(0.0, 2.0);
        Animator {
            plan,
            rest: c.parts.iter().map(|p| (p.rot, p.bend, p.half)).collect(),
            legs,
            free,
            leg_length,
            step_time,
            threshold,
            top_speed,
            radius,
            spawn: c.base,
            floor,
            rng,
            time: 0.0,
            speed: 0.0,
            going: false,
            wait,
            want_yaw: c.yaw,
            pitch: 0.0,
        }
    }

    fn ground(scene: &Scene, x: f32, y: f32, z: f32) -> f32 {
        scene.floor_y(x, y - 16.0, z).unwrap_or(y)
    }

    /// Can it take its next stride? Looks ahead and to either side of its way.
    fn clear(&self, c: &Creature, scene: &Scene, dir: [f32; 3]) -> bool {
        let ahead = 0.6 * self.radius + 0.5 * self.top_speed;
        let across = [dir[2], 0.0, -dir[0]];
        [0.0, 0.5, -0.5].iter().all(|&s| {
            let p = add(c.base, add(scale(dir, ahead), scale(across, s * self.radius)));
            scene.walkable_y(p[0], c.base[1] - 16.0, p[2]).is_some_and(|y| (y - c.base[1]).abs() <= MAX_STEP * 0.6)
        })
    }

    /// Decides where to go and moves the base; returns the velocity.
    fn wander(&mut self, c: &mut Creature, scene: &Scene, dt: f32) -> [f32; 3] {
        self.wait -= dt;
        if self.wait <= 0.0 {
            self.going = self.rng.chance(0.65);
            self.wait = if self.going { self.rng.fr(2.0, 6.0) } else { self.rng.fr(1.0, 3.5) };
            self.want_yaw = c.yaw + self.rng.fr(-1.5, 1.5);
        }
        let home = sub(self.spawn, c.base);
        if home[0].hypot(home[2]) > LEASH {
            self.want_yaw = (-home[0]).atan2(home[2]);
            self.going = true;
        }
        let dir = apply(rot_y(c.yaw), [0.0, 0.0, 1.0]);
        if self.going && !self.clear(c, scene, dir) {
            let turn = self.rng.fr(1.5, PI);
            self.want_yaw = c.yaw + if self.rng.chance(0.5) { turn } else { -turn };
            self.going = false;
            self.wait = 0.7;
        }
        let miss = (self.want_yaw - c.yaw).sin().atan2((self.want_yaw - c.yaw).cos());
        c.yaw += miss.clamp(-TURN_RATE * dt, TURN_RATE * dt);
        let want = if !self.going { 0.0 } else if miss.abs() > 0.5 { 0.3 * self.top_speed } else { self.top_speed };
        self.speed += (want - self.speed) * (6.0 * dt).min(1.0);
        let velocity = scale(apply(rot_y(c.yaw), [0.0, 0.0, 1.0]), self.speed);
        c.base[0] += velocity[0] * dt;
        c.base[2] += velocity[2] * dt;
        let floor = Self::ground(scene, c.base[0], self.floor, c.base[2]);
        self.floor += (floor - self.floor) * (10.0 * dt).min(1.0);
        velocity
    }

    fn step_legs(&mut self, c: &Creature, scene: &Scene, velocity: [f32; 3], dt: f32) {
        let frame = rot_y(c.yaw);
        let lead = 0.6 * self.step_time;
        let moving = self.speed > 0.1 * self.top_speed;
        let mut stepping = [false; 2];
        for l in &self.legs {
            stepping[l.group] |= l.step.is_some();
        }
        for l in &mut self.legs {
            let place = add(add(c.base, apply(frame, [l.home[0], 0.0, l.home[1]])), scale(velocity, lead));
            let target = [place[0], Self::ground(scene, place[0], self.floor, place[2]) - l.lift, place[2]];
            if let Some((from, t)) = &mut l.step {
                *t += dt / self.step_time;
                if *t >= 1.0 {
                    l.at = target;
                    l.step = None;
                } else {
                    let eased = *t * *t * (3.0 - 2.0 * *t);
                    l.at = lerp(*from, target, eased);
                    l.at[1] -= 0.22 * self.leg_length * (PI * *t).sin();
                }
                continue;
            }
            let off = (target[0] - l.at[0]).hypot(target[2] - l.at[2]);
            let wanted = off > if moving { self.threshold } else { 0.12 * self.leg_length };
            if (wanted && !stepping[1 - l.group]) || off > 2.2 * self.threshold {
                l.step = Some((l.at, 0.0));
                stepping[l.group] = true;
            }
        }
    }

    /// Bends each leg to reach where its foot is.
    fn solve_legs(&self, c: &mut Creature) {
        let placed = c.pose();
        let frame = rot_y(c.yaw);
        let forward = apply(frame, [0.0, 0.0, 1.0]);
        for l in &self.legs {
            let bones = l.dirs.len();
            let mut p: Vec<[f32; 3]> = l.chain[..bones].iter().map(|&i| placed[i].joint).collect();
            p.push(match l.foot {
                Some(f) => placed[f].joint,
                None => placed[l.chain[bones - 1]].world(l.tip),
            });
            // Knees bend forward.
            for joint in &mut p[1..bones] {
                *joint = add(*joint, scale(forward, 0.2 * self.leg_length));
            }
            reach(&mut p, &l.lengths, l.at);
            let mut above = placed[c.parts[l.chain[0]].parent.unwrap()].axes;
            for i in 0..bones {
                let part = l.chain[i];
                let was = mul(above, c.parts[part].rot);
                let now = mul(arc(norm(apply(was, l.dirs[i])), norm(sub(p[i + 1], p[i]))), was);
                c.parts[part].rot = mul(transpose(above), now);
                above = now;
            }
            if let Some(f) = l.foot {
                c.parts[f].rot = mul(transpose(above), mul(frame, l.foot_axes));
            }
        }
    }

    /// Moves `c` on by `dt` seconds.
    pub fn update(&mut self, c: &mut Creature, scene: &Scene, dt: f32) {
        let dt = dt.min(0.05);
        self.time += dt;
        let velocity = self.wander(c, scene, dt);
        let pace = (self.speed / self.top_speed).clamp(0.0, 1.0);
        for (p, &(rot, bend, half)) in c.parts.iter_mut().zip(&self.rest) {
            p.rot = rot;
            p.bend = bend;
            p.half = half;
        }
        let stride = self.time * PI / self.step_time;
        let breath = self.time * 1.7;
        for f in &self.free {
            let wave = self.time * f.speed + f.phase;
            let p = &mut c.parts[f.part];
            let swing = f.arm.map_or(0.0, |side| 0.4 * pace * side * stride.sin());
            p.rot = mul(p.rot, mul(rot_x(f.sway * wave.sin() + swing), rot_z(f.sway * 0.7 * (wave * 0.8).cos())));
            p.bend += f.bend * (wave * 1.3).sin();
        }
        match self.plan {
            Plan::Walk => {
                self.step_legs(c, scene, velocity, dt);
                let height = |front: bool| {
                    let ys: Vec<f32> = self.legs.iter().filter(|l| (l.home[1] > 0.0) == front).map(|l| l.at[1] + l.lift).collect();
                    (!ys.is_empty()).then(|| ys.iter().sum::<f32>() / ys.len() as f32)
                };
                let span = self.legs.iter().map(|l| l.home[1]).fold(f32::MIN, f32::max) - self.legs.iter().map(|l| l.home[1]).fold(f32::MAX, f32::min);
                let want = match (height(true), height(false)) {
                    (Some(a), Some(b)) if span > 1.0 => -((a - b) / span).atan().clamp(-0.35, 0.35),
                    _ => 0.0,
                };
                self.pitch += (want - self.pitch) * (8.0 * dt).min(1.0);
                let roll = 0.05 * pace * stride.sin() + 0.015 * breath.sin();
                c.base[1] = self.floor + 0.035 * self.leg_length * pace * (2.0 * stride).sin() + 0.01 * self.leg_length * breath.sin();
                c.parts[0].rot = mul(mul(rot_x(self.pitch), rot_z(roll)), self.rest[0].0);
                self.solve_legs(c);
            }
            Plan::Creep => {
                let squash = (0.04 + 0.1 * pace) * (self.time * (2.0 + 3.0 * pace)).sin();
                let h = &mut c.parts[0].half;
                *h = [h[0] * (1.0 - 0.5 * squash), h[1] * (1.0 + squash), h[2] * (1.0 - 0.5 * squash)];
                c.base[1] = self.floor;
            }
            Plan::Float => {
                c.base[1] = self.floor + 0.08 * TF * breath.sin();
                c.parts[0].rot = mul(rot_z(0.06 * (breath * 0.7).sin()), self.rest[0].0);
            }
        }
    }
}

/// Every creature of a scene moving; remade when the scene's `version` changes.
#[derive(Default)]
pub struct Herd {
    version: Option<u64>,
    all: Vec<Animator>,
}

impl Herd {
    pub fn update(&mut self, scene: &mut Scene, version: u64, dt: f32) {
        if self.version != Some(version) {
            self.all = scene.creatures.iter().map(Animator::new).collect();
            self.version = Some(version);
        }
        let mut creatures = std::mem::take(&mut scene.creatures);
        for (a, c) in self.all.iter_mut().zip(&mut creatures) {
            a.update(c, scene, dt);
        }
        scene.creatures = creatures;
    }

    pub fn plans(&self) -> Vec<Plan> {
        self.all.iter().map(|a| a.plan).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{walkable_scene, rng::Rng};

    #[test]
    fn feet_stay_where_they_are_put() {
        let mut walkers = 0;
        for seed in 1..7 {
            let (mut scene, _) = walkable_scene(&mut Rng::new(seed), 30);
            let mut herd = Herd::default();
            for frame in 0..300 {
                herd.update(&mut scene, 0, 1.0 / 30.0);
                for (a, c) in herd.all.iter().zip(&scene.creatures) {
                    let placed = c.pose();
                    assert!(placed.iter().all(|p| p.centre.iter().chain(p.axes.iter().flatten()).all(|v| v.is_finite())), "seed {seed} frame {frame}");
                    for l in &a.legs {
                        let bones = l.dirs.len();
                        let end = match l.foot {
                            Some(f) => placed[f].joint,
                            None => placed[l.chain[bones - 1]].world(l.tip),
                        };
                        let hip = placed[l.chain[0]].joint;
                        if length(sub(l.at, hip)) < 0.98 * l.lengths.iter().sum::<f32>() {
                            assert!(length(sub(end, l.at)) < 4.0, "seed {seed} frame {frame}: foot {} from its place", length(sub(end, l.at)));
                        }
                    }
                }
            }
            walkers += herd.plans().iter().filter(|&&p| p == Plan::Walk).count();
        }
        assert!(walkers > 0, "no creature walked");
    }
}
