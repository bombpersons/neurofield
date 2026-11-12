//! Tentacles that grow out of the walls and ceilings round a badly afflicted player, and
//! whip at them: each rears back, curling, then cracks through at where the player was, its
//! tip last. The player can block a lash (the game decides, and `recoil` throws it off) or cut
//! one down (`strike`): a dead one flops down limp, lies there a while and withers. They are
//! one creature for the dream net, with a fixed number of parts: a tentacle that isn't there
//! has no size, and the net doesn't draw it.

use crate::world::creature::{cross, sub, Creature, Part, Shape};
use crate::world::rng::Rng;
use crate::world::scene::{dot, mul, norm, rot_x, rot_z, Camera, Scene, T};
use crate::world::view::view_matrix;
use std::f32::consts::TAU;

const TF: f32 = T as f32;
/// How many tentacles there can be at once, and the parts of each.
const MOST: usize = 8;
const SEGMENTS: usize = 5;
/// Seconds to grow out, and to wither back.
const GROW_SECS: f32 = 3.0;
const WITHER_SECS: f32 = 1.2;
/// Seconds a tentacle lasts once it has started, and between new ones.
const LIFE: (f32, f32) = (14.0, 26.0);
const SPAWN_SECS: f32 = 1.5;
/// How far from the player one can start (tiles): never so close that it fills the view.
const NEAREST: f32 = 1.8;
const FURTHEST: f32 = 4.5;
/// Tiles between the roots of any two, and how far the player can leave one before it withers.
const APART: f32 = 1.0;
const LEFT_BEHIND: f32 = 8.0;
/// A lash: seconds turning on the player and rearing back, then cracking through (only this
/// part hurts: there is time to step away or raise a shield), then going slack, then resting
/// before the next. Thrown off (blocked or struck), it rests longer.
const WINDUP: f32 = 0.7;
const CRACK: f32 = 0.25;
const SLACK: f32 = 0.8;
const REST: f32 = 2.0;
const THROWN_OFF: f32 = 1.5;
/// How far each joint curls back as it rears (radians), and past straight as it cracks; and
/// how far behind the joint before it each one follows through, in shares of the crack.
const REAR: f32 = 0.45;
const FOLLOW: f32 = 0.35;
const LAG: f32 = 0.12;
/// How much longer a tentacle is at the end of its crack.
const STRETCH: f32 = 1.25;
/// How close the crack has to pass to the player's chest to touch them (tiles).
const TOUCH: f32 = 0.3;
/// Health: a couple of good blows.
const HP: i32 = 6;
/// Once dead: seconds to flop down, how far its joints sag (radians), and seconds it lies
/// there before it withers.
const WILT: f32 = 0.7;
const SAG: f32 = 0.3;
const LIE: f32 = 2.5;
/// Down (y is down).
const DOWN: [f32; 3] = [0.0, 1.0, 0.0];
/// The player's chest, below the eye (y is down), and the seconds between hits.
const CHEST: f32 = 0.25 * TF;
const HIT_EVERY: f32 = 1.2;

fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}

/// How close `p` is to the segment from `a` to `b`.
fn to_segment(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = sub(b, a);
    let u = (dot(sub(p, a), ab) / dot(ab, ab).max(1e-6)).clamp(0.0, 1.0);
    distance(p, [0, 1, 2].map(|c| a[c] + ab[c] * u))
}

/// Whether nothing solid lies between `a` and `b`, up to just short of `b` (a point on a wall).
fn sees(scene: &Scene, a: [f32; 3], b: [f32; 3]) -> bool {
    let d = sub(b, a);
    let length = dot(d, d).sqrt();
    let n = (length / 32.0).ceil().max(1.0) as i32;
    let end = (1.0 - 0.2 * TF / length.max(1.0)).max(0.0);
    (1..n).all(|i| {
        let t = end * i as f32 / n as f32;
        let (x, y, z) = (a[0] + d[0] * t, a[1] + d[1] * t, a[2] + d[2] * t);
        scene.node_at(x, y, z).is_some_and(|n| {
            let s = scene.span(n);
            s.floor(n.0, n.1, [x, z]) > y && (s.sky || s.ceiling(n.0, n.1, [x, z]) < y)
        })
    })
}

/// Whether a world point is in front of `cam` and on its screen, at any distance.
fn in_view(cam: &Camera, p: [f32; 3]) -> bool {
    let m = view_matrix(cam);
    let r = |i: usize, j: usize| m.m[i][j] as f32 / 4096.0;
    let d = [0, 1, 2].map(|a| (p[a] - cam.pos[a] as f32) / TF);
    let c = [0, 1, 2].map(|k| r(k, 0) * d[0] + r(k, 1) * d[1] + r(k, 2) * d[2]);
    c[2] > 0.0 && c[0].abs() < c[2] && c[1].abs() < c[2] * 0.8
}

#[derive(Clone, Copy)]
struct Tentacle {
    root: [f32; 3],
    /// The way it grows, and the way it is lashing.
    out: [f32; 3],
    aim: [f32; 3],
    /// Half the length of each segment, and the radius at the root (world units).
    half: f32,
    girth: f32,
    phase: f32,
    /// How far grown, 0..1 (0: this slot is free), and seconds left before it withers.
    grown: f32,
    life: f32,
    /// Seconds into a lash, if it is lashing, and seconds before it can.
    lash: Option<f32>,
    rest: f32,
    hp: i32,
    /// Seconds since it was killed.
    dead: Option<f32>,
}

/// A lash that reached the player: which tentacle, and where it grows from.
#[derive(Clone, Copy, Debug)]
pub struct Lash {
    pub which: usize,
    pub root: [f32; 3],
}

impl Tentacle {
    /// How far it has turned from `out` to `aim`, 0..1: on the player as it lashes, down as it dies.
    fn turned(&self) -> f32 {
        if let Some(t) = self.dead {
            return smooth(t / WILT);
        }
        self.lash.map_or(0.0, |t| {
            if t < WINDUP + CRACK { smooth(t / WINDUP) } else { 1.0 - smooth((t - WINDUP - CRACK) / SLACK) }
        })
    }

    /// How far through its crack it has stretched, 0..1.
    fn stretched(&self) -> f32 {
        self.lash.map_or(0.0, |t| {
            if t < WINDUP + CRACK { smooth((t - WINDUP) / CRACK) } else { 1.0 - smooth((t - WINDUP - CRACK) / SLACK) }
        })
    }

    fn cracking(&self) -> bool {
        self.dead.is_none() && self.lash.is_some_and(|t| (WINDUP..WINDUP + CRACK).contains(&t))
    }

    fn alive(&self) -> bool {
        self.dead.is_none() && self.life > 0.0
    }

    /// How far joint `s` (1..) is curled (radians): back as it rears, through past straight as
    /// it cracks (later the nearer the tip), easing back as it goes slack; sagging once dead.
    fn curl(&self, s: usize) -> f32 {
        if let Some(t) = self.dead {
            return SAG * smooth(t / WILT);
        }
        let Some(t) = self.lash else { return 0.0 };
        if t < WINDUP {
            -REAR * smooth(t / WINDUP)
        } else if t < WINDUP + CRACK {
            let lag = (s - 1) as f32 * LAG;
            let u = ((t - WINDUP) / CRACK - lag) / (1.0 - (SEGMENTS - 2) as f32 * LAG);
            -REAR + (REAR + FOLLOW) * smooth(u)
        } else {
            FOLLOW * (1.0 - smooth((t - WINDUP - CRACK) / SLACK))
        }
    }

    /// How far it can reach from its root.
    fn reach(&self) -> f32 {
        2.0 * self.half * SEGMENTS as f32 * STRETCH
    }

    /// Set this tentacle's parts for `clock` seconds.
    fn pose(&self, parts: &mut [Part], clock: f32) {
        let turned = self.turned();
        let long = self.half * self.grown * (1.0 + (STRETCH - 1.0) * self.stretched());
        let y = norm([0, 1, 2].map(|c| self.out[c] + (self.aim[c] - self.out[c]) * turned));
        let x = norm(cross(y, if y[1].abs() > 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] }));
        let z = cross(x, y);
        parts[0].anchor = self.root;
        parts[0].rot = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
        for (s, p) in parts.iter_mut().enumerate() {
            let girth = self.girth * 0.8f32.powi(s as i32) * self.grown.sqrt();
            p.half = if self.grown > 0.0 { [girth.max(0.2), long.max(0.2), girth.max(0.2)] } else { [0.0; 3] };
            if s > 0 {
                // Swaying while it waits; still while it lashes, and once dead.
                let w = clock * 1.4 + self.phase - s as f32 * 0.8;
                let k = if self.dead.is_some() { 0.0 } else { (0.15 + 0.1 * s as f32) * (1.0 - turned) };
                p.rot = mul(rot_x(k * w.sin() + self.curl(s)), rot_z(k * (w * 0.7 + 1.0).cos()));
            }
        }
    }
}

/// The tentacles of one level.
pub struct Growths {
    creature: Creature,
    all: [Option<Tentacle>; MOST],
    clock: f32,
    /// Seconds before the next can start, and before another can hurt.
    spawn: f32,
    hit: f32,
}

impl Growths {
    /// None will grow in a level with no creature to borrow skins from.
    pub fn new(scene: &Scene) -> Growths {
        let mut creature = Creature { base: [0.0; 3], yaw: 0.0, skins: Vec::new(), parts: Vec::new(), slots: Vec::new(), girth: 0.0 };
        if let Some(like) = scene.creatures.first() {
            creature.skins = like.skins.clone();
            creature.slots = like.slots.clone();
            for i in 0..MOST * SEGMENTS {
                let joint = i % SEGMENTS;
                creature.parts.push(Part {
                    parent: (joint > 0).then(|| i - 1),
                    anchor: if joint == 0 { [0.0; 3] } else { [1.0, 0.0, 0.0] },
                    rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    pivot: [0.0, -1.0, 0.0],
                    half: [0.0; 3],
                    shape: Shape::Tentacle,
                    taper: 0.8,
                    bend: 0.0,
                    bulge: 0.0,
                    lumps: 0.0,
                    skin: 0,
                    seed: 0,
                });
            }
        }
        Growths { creature, all: [None; MOST], clock: 0.0, spawn: 0.0, hit: 0.0 }
    }

    /// How many tentacles are out, dead ones lying about included.
    pub fn count(&self) -> usize {
        self.all.iter().flatten().count()
    }

    /// How many are alive.
    pub fn living(&self) -> usize {
        self.all.iter().flatten().filter(|a| a.alive()).count()
    }

    /// `dt` seconds pass with the player at `cam`, and `wanted` (0..1) of the most there can be
    /// called for; with none called for they wither. The lash that reached the player, if one did.
    pub fn update(&mut self, scene: &Scene, cam: &Camera, wanted: f32, dt: f32, rng: &mut Rng) -> Option<Lash> {
        if self.creature.parts.is_empty() {
            return None;
        }
        self.clock += dt;
        self.spawn -= dt;
        self.hit -= dt;
        let eye = cam.pos.map(|c| c as f32);
        let chest = [eye[0], eye[1] + CHEST, eye[2]];
        if self.spawn <= 0.0 && wanted > 0.0 && self.living() < (wanted * MOST as f32).ceil() as usize {
            self.spawn = SPAWN_SECS;
            self.sprout(scene, cam, rng);
        }
        for slot in &mut self.all {
            let Some(a) = slot else { continue };
            if let Some(t) = &mut a.dead {
                *t += dt;
                if *t > WILT + LIE {
                    a.grown = (a.grown - dt / WITHER_SECS).max(0.0);
                }
                continue;
            }
            a.life -= dt;
            if wanted <= 0.0 || distance(a.root, eye) > LEFT_BEHIND * TF {
                a.life = a.life.min(0.0);
            }
            if a.life <= 0.0 {
                a.lash = None;
                a.grown -= dt / WITHER_SECS;
                if a.grown <= 0.0 {
                    a.grown = 0.0;
                }
                continue;
            }
            a.grown = (a.grown + dt / GROW_SECS).min(1.0);
            a.rest -= dt;
            match &mut a.lash {
                Some(t) => {
                    *t += dt;
                    if *t >= WINDUP + CRACK + SLACK {
                        a.lash = None;
                        a.rest = a.rest.max(REST);
                    }
                }
                None if a.grown >= 1.0 && a.rest <= 0.0 => {
                    let to = sub(chest, a.root);
                    if distance(chest, a.root) < a.reach() && dot(a.out, to) > 0.0 && sees(scene, chest, a.root) {
                        a.lash = Some(0.0);
                    }
                }
                None => {}
            }
            // It follows the player until it lunges, and the lunge goes where they were.
            if a.lash.is_some_and(|t| t < WINDUP) {
                a.aim = norm(sub(chest, a.root));
            }
        }
        for (i, slot) in self.all.iter().enumerate() {
            let parts = &mut self.creature.parts[i * SEGMENTS..(i + 1) * SEGMENTS];
            match slot {
                Some(a) => a.pose(parts, self.clock),
                None => parts.iter_mut().for_each(|p| p.half = [0.0; 3]),
            }
        }
        let touched = if self.hit <= 0.0 { self.touches(chest) } else { None };
        if touched.is_some() {
            self.hit = HIT_EVERY;
        }
        for slot in &mut self.all {
            if slot.is_some_and(|a| (a.life <= 0.0 || a.dead.is_some()) && a.grown <= 0.0) {
                *slot = None;
            }
        }
        touched
    }

    /// The segments of tentacle `i` as they stand: (from, to) each.
    fn segments(&self, i: usize) -> Vec<([f32; 3], [f32; 3])> {
        let placed = self.creature.pose();
        (i * SEGMENTS..(i + 1) * SEGMENTS).map(|s| (placed[s].joint, [0, 1, 2].map(|c| 2.0 * placed[s].centre[c] - placed[s].joint[c]))).collect()
    }

    /// The cracking tentacle at `chest`, if one is.
    fn touches(&self, chest: [f32; 3]) -> Option<Lash> {
        self.all.iter().enumerate().find_map(|(i, slot)| {
            let a = slot.as_ref()?;
            let near = a.cracking() && self.segments(i).iter().any(|&(from, to)| to_segment(chest, from, to) < TOUCH * TF + a.girth);
            near.then_some(Lash { which: i, root: a.root })
        })
    }

    /// Tentacle `which` is thrown off its lash (a shield took it): it goes slack and rests.
    pub fn recoil(&mut self, which: usize) {
        if let Some(Some(a)) = self.all.get_mut(which) {
            if a.lash.is_some_and(|t| t < WINDUP + CRACK) {
                a.lash = Some(WINDUP + CRACK);
            }
            a.rest = REST + THROWN_OFF;
        }
    }

    /// A blow of `damage` from `chest`, facing `ahead` (x, z), reaching `reach` and `arc`
    /// radians either side: the nearest living tentacle it finds takes it, and is thrown off its
    /// lash or killed. None if it found none, else whether it killed it.
    pub fn strike(&mut self, chest: [f32; 3], ahead: [f32; 2], reach: f32, arc: f32, damage: i32) -> Option<bool> {
        let facing = |p: [f32; 3]| {
            let d = sub(p, chest);
            let flat = d[0].hypot(d[2]);
            flat < 0.15 * TF || ((d[0] * ahead[0] + d[2] * ahead[1]) / flat).clamp(-1.0, 1.0).acos() < arc
        };
        let (i, _) = (0..MOST)
            .filter(|&i| self.all[i].is_some_and(|a| a.alive() && a.grown > 0.5))
            .filter_map(|i| {
                let near = self.segments(i).iter().flat_map(|&(from, to)| [from, [0, 1, 2].map(|c| (from[c] + to[c]) / 2.0), to]).filter(|&p| facing(p)).map(|p| distance(p, chest)).fold(f32::INFINITY, f32::min);
                (near < reach).then_some((i, near))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        let a = self.all[i].as_mut()?;
        a.hp -= damage;
        if a.hp > 0 {
            self.recoil(i);
            return Some(false);
        }
        // It flops down from however it stood.
        let turned = a.turned();
        a.out = norm([0, 1, 2].map(|c| a.out[c] + (a.aim[c] - a.out[c]) * turned));
        a.aim = norm([0, 1, 2].map(|c| DOWN[c] + 0.15 * a.out[c]));
        a.lash = None;
        a.dead = Some(0.0);
        Some(true)
    }

    /// Start one in a free slot, on a wall or ceiling the player is looking at.
    fn sprout(&mut self, scene: &Scene, cam: &Camera, rng: &mut Rng) {
        let Some(free) = self.all.iter().position(Option::is_none) else { return };
        let eye = cam.pos.map(|c| c as f32);
        let places: Vec<([f32; 3], [f32; 3])> = scene
            .tris
            .iter()
            .filter_map(|t| {
                let n = t.normal();
                let c = [0, 1, 2].map(|a| t.v.iter().map(|v| v.p[a] as f32).sum::<f32>() / 3.0);
                let d = distance(c, eye) / TF;
                let wanted = (n[1].abs() < 0.3 || n[1] > 0.7) && (NEAREST..FURTHEST).contains(&d) && dot(n, sub(eye, c)) > 0.0;
                (wanted && in_view(cam, c)).then_some((c, n))
            })
            .collect();
        if places.is_empty() {
            return;
        }
        for _ in 0..12 {
            let (root, n) = places[rng.next() as usize % places.len()];
            let crowded = self.all.iter().flatten().any(|a| distance(a.root, root) < APART * TF);
            if crowded || !sees(scene, eye, root) {
                continue;
            }
            let lean = [rng.fr(-0.35, 0.35), rng.fr(-0.35, 0.35), rng.fr(-0.35, 0.35)];
            let out = norm([n[0] + lean[0], n[1] + lean[1], n[2] + lean[2]]);
            self.all[free] = Some(Tentacle {
                root,
                out,
                aim: out,
                half: rng.fr(0.13, 0.22) * TF,
                girth: rng.fr(0.05, 0.09) * TF,
                phase: rng.fr(0.0, TAU),
                grown: 1e-3,
                life: rng.fr(LIFE.0, LIFE.1),
                lash: None,
                rest: 0.0,
                hp: HP,
                dead: None,
            });
            return;
        }
    }

    /// The creature to draw.
    pub fn creature(&self) -> Creature {
        self.creature.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::walkable_scene;

    fn scene_and_camera(seed: u64) -> (Scene, Camera) {
        let (scene, walker) = walkable_scene(&mut Rng::new(seed), 30);
        let cam = walker.camera();
        (scene, cam)
    }

    /// One fully grown tentacle pointing at nothing, `away` tiles from the camera along +x.
    fn one_at(g: &mut Growths, cam: &Camera, away: f32) -> [f32; 3] {
        let root = [cam.pos[0] as f32 + away * TF, cam.pos[1] as f32, cam.pos[2] as f32];
        g.all[0] = Some(Tentacle { root, out: [-1.0, 0.0, 0.0], aim: [-1.0, 0.0, 0.0], half: 0.2 * TF, girth: 0.07 * TF, phase: 0.0, grown: 1.0, life: 100.0, lash: None, rest: 0.0, hp: HP, dead: None });
        root
    }

    #[test]
    fn they_only_start_in_view_and_not_too_close() {
        let mut started = 0;
        for seed in 1..6 {
            let (scene, cam) = scene_and_camera(seed);
            let mut g = Growths::new(&scene);
            let mut rng = Rng::new(seed);
            for _ in 0..8 {
                g.sprout(&scene, &cam, &mut rng);
                g.all.iter_mut().for_each(|s| {
                    if let Some(a) = s.take() {
                        let d = distance(a.root, cam.pos.map(|c| c as f32)) / TF;
                        assert!((NEAREST..FURTHEST).contains(&d), "{d}");
                        assert!(in_view(&cam, a.root));
                        started += 1;
                    }
                });
            }
        }
        assert!(started > 0, "none ever started");
    }

    #[test]
    fn a_lash_that_arrives_hurts_once_per_hit_time() {
        let (scene, cam) = scene_and_camera(1);
        let mut g = Growths::new(&scene);
        let mut rng = Rng::new(1);
        one_at(&mut g, &cam, 1.0);
        g.all[0].as_mut().unwrap().lash = Some(WINDUP - 0.001);
        g.all[0].as_mut().unwrap().aim = norm(sub([cam.pos[0] as f32, (cam.pos[1] as f32) + CHEST, cam.pos[2] as f32], g.all[0].unwrap().root));
        let mut hits = 0;
        for _ in 0..3 {
            hits += g.update(&scene, &cam, 1.0, 1.0 / 30.0, &mut rng).is_some() as i32;
        }
        assert_eq!(hits, 1);
    }

    #[test]
    fn nothing_hurts_while_it_is_only_growing_or_swaying() {
        let (scene, cam) = scene_and_camera(1);
        let mut g = Growths::new(&scene);
        let mut rng = Rng::new(1);
        one_at(&mut g, &cam, 0.5);
        g.all[0].as_mut().unwrap().rest = 100.0;
        for _ in 0..60 {
            assert!(g.update(&scene, &cam, 1.0, 1.0 / 30.0, &mut rng).is_none());
        }
    }

    #[test]
    fn with_none_called_for_they_wither_and_cannot_hurt() {
        let (scene, cam) = scene_and_camera(1);
        let mut g = Growths::new(&scene);
        let mut rng = Rng::new(1);
        one_at(&mut g, &cam, 1.0);
        g.all[0].as_mut().unwrap().lash = Some(WINDUP);
        for _ in 0..(WITHER_SECS * 30.0) as i32 + 2 {
            assert!(g.update(&scene, &cam, 0.0, 1.0 / 30.0, &mut rng).is_none());
        }
        assert_eq!(g.count(), 0);
        assert!(g.creature.parts.iter().all(|p| p.half == [0.0; 3]));
    }

    /// The camera's chest, and which way (x, z) it faces.
    fn chest_and_ahead(cam: &Camera) -> ([f32; 3], [f32; 2]) {
        let eye = cam.pos.map(|c| c as f32);
        let m = view_matrix(cam);
        let r = |i: usize, j: usize| m.m[i][j] as f32 / 4096.0;
        let f = [r(2, 0), r(2, 2)];
        let l = f[0].hypot(f[1]);
        ([eye[0], eye[1] + CHEST, eye[2]], [f[0] / l, f[1] / l])
    }

    /// A tentacle a tile ahead of the camera, pointing back at it.
    fn one_ahead(g: &mut Growths, cam: &Camera) {
        let (chest, ahead) = chest_and_ahead(cam);
        let root = [chest[0] + ahead[0] * TF, chest[1], chest[2] + ahead[1] * TF];
        let out = norm(sub(chest, root));
        g.all[0] = Some(Tentacle { root, out, aim: out, half: 0.2 * TF, girth: 0.07 * TF, phase: 0.0, grown: 1.0, life: 100.0, lash: None, rest: 100.0, hp: HP, dead: None });
    }

    #[test]
    fn a_whip_rears_back_then_cracks_through_tip_last() {
        let t = Tentacle { root: [0.0; 3], out: DOWN, aim: DOWN, half: 1.0, girth: 1.0, phase: 0.0, grown: 1.0, life: 1.0, lash: Some(WINDUP), rest: 0.0, hp: HP, dead: None };
        assert!((1..SEGMENTS).all(|s| (t.curl(s) + REAR).abs() < 1e-4), "reared back at the start of the crack");
        let mid = Tentacle { lash: Some(WINDUP + CRACK * 0.5), ..t };
        assert!((2..SEGMENTS).all(|s| mid.curl(s) < mid.curl(s - 1)), "the tip follows through last");
        let end = Tentacle { lash: Some(WINDUP + CRACK - 1e-4), ..t };
        assert!((1..SEGMENTS).all(|s| (end.curl(s) - FOLLOW).abs() < 0.02), "through past straight at the end");
    }

    #[test]
    fn a_thrown_off_lash_goes_slack_and_waits_longer() {
        let (scene, cam) = scene_and_camera(1);
        let mut g = Growths::new(&scene);
        one_ahead(&mut g, &cam);
        g.all[0].as_mut().unwrap().lash = Some(WINDUP + 0.05);
        g.recoil(0);
        let a = g.all[0].unwrap();
        assert!(!a.cracking());
        assert_eq!(a.rest, REST + THROWN_OFF);
    }

    #[test]
    fn blows_hurt_then_kill_and_the_dead_flop_down_and_wither() {
        let (scene, cam) = scene_and_camera(1);
        let mut g = Growths::new(&scene);
        let mut rng = Rng::new(1);
        one_ahead(&mut g, &cam);
        g.spawn = 1e9; // no others
        g.update(&scene, &cam, 1.0, 1.0 / 30.0, &mut rng);
        let (chest, ahead) = chest_and_ahead(&cam);
        let behind = [-ahead[0], -ahead[1]];
        assert_eq!(g.strike(chest, behind, 1.1 * TF, 1.0, 4), None, "a blow the other way misses");
        assert_eq!(g.strike(chest, ahead, 0.1 * TF, 1.0, 4), None, "out of reach");
        assert_eq!(g.strike(chest, ahead, 1.1 * TF, 1.0, 4), Some(false));
        assert_eq!(g.strike(chest, ahead, 1.1 * TF, 1.0, 4), Some(true));
        assert_eq!(g.living(), 0);
        assert_eq!(g.strike(chest, ahead, 1.1 * TF, 1.0, 4), None, "the dead take no more blows");
        let tip = |g: &Growths| g.segments(0)[SEGMENTS - 1].1;
        let before = tip(&g)[1];
        for _ in 0..(WILT * 30.0) as i32 + 1 {
            assert!(g.update(&scene, &cam, 1.0, 1.0 / 30.0, &mut rng).is_none(), "the dead don't lash");
        }
        assert!(tip(&g)[1] > before + 0.2 * TF, "the tip has fallen (y is down): {} -> {}", before, tip(&g)[1]);
        for _ in 0..((LIE + WITHER_SECS) * 30.0) as i32 + 2 {
            g.update(&scene, &cam, 1.0, 1.0 / 30.0, &mut rng);
        }
        assert!(g.all[0].is_none(), "withered away");
    }
}
