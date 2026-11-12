//! Creatures moving by themselves, whatever they grew into. Each one's body plan is read off
//! its rest pose: chains of parts that reach the floor are legs, which plant their feet and
//! step in turn, the rest of the body following; without legs it creeps or floats. Motion is
//! written to the creature itself (`base`, `yaw` and each part's `rot`, `bend` and `half`), so
//! whatever reads `Creature::pose` sees it.
//!
//! Left alone a creature wanders at a walk, and when it has stood still a while it fidgets.
//! Told to `chase` it runs. Over its gait go what the game asks of it: a strike (`attack`),
//! which can be `blocked` and rebound, a spell (`cast`), and its death (`die`), which leaves
//! it lying on the floor for good. `update` says when a strike lands or a spell goes off.

use super::creature::{add, scale, sub, Creature, Placed, Shape};
use super::rng::Rng;
use super::scene::{apply, dot, mul, norm, rot_x, rot_y, rot_z, Scene, M3, MAX_STEP, T};
use std::f32::consts::{PI, TAU};

const TF: f32 = T as f32;
/// A part this near the floor (as a fraction of the creature's height) can stand on it.
const CONTACT: f32 = 0.12;
/// Furthest a creature strays from where it was grown, unless it is chasing something.
const LEASH: f32 = 4.0 * TF;
const TURN_RATE: f32 = 1.8;
const CHASE_TURN_RATE: f32 = 3.2;
const CREEP_SPEED: f32 = 0.2 * TF;
const CREEP_RUN: f32 = 0.7 * TF;
const FLOAT_SPEED: f32 = 0.3 * TF;
const FLOAT_RUN: f32 = 1.0 * TF;
/// The fastest anything runs: about the player's walking pace.
const RUN_CAP: f32 = 1.6 * TF;
/// How fast the dead fall, world units per second per second.
const GRAVITY: f32 = 8.0 * TF;

/// How a creature gets about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    Walk,
    Creep,
    Float,
}

/// Ways to hit something.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strike {
    /// An arm or other limb drawn back and swept across.
    Swipe,
    /// A tentacle curled back and cracked like a whip.
    Lash,
    /// A horn, spike or slender limb drawn back and thrust.
    Stab,
    /// Rearing back and throwing the whole body (and head) forward.
    Lunge,
    /// Rising up with every limb raised and crashing down.
    Slam,
}

impl Strike {
    pub const ALL: [Strike; 5] = [Strike::Swipe, Strike::Lash, Strike::Stab, Strike::Lunge, Strike::Slam];
}

/// Ways to die. Each ends with the body lying still on the floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Death {
    /// Falls over sideways, stiff.
    Topple,
    /// The legs give way and splay; the body drops between them.
    Crumple,
    /// Rolls onto its back and draws its legs in, twitching, like a dead spider.
    Curl,
    /// Sags and spreads into a puddle.
    Melt,
    /// Shrinks and wrinkles into a husk, its limbs curling under.
    Shrivel,
    /// Swells, trembling, then bursts and collapses flat.
    Burst,
}

impl Death {
    pub const ALL: [Death; 6] = [Death::Topple, Death::Crumple, Death::Curl, Death::Melt, Death::Shrivel, Death::Burst];

    fn length(self) -> f32 {
        match self {
            Death::Topple => 1.3,
            Death::Crumple => 1.2,
            Death::Curl => 2.4,
            Death::Melt => 3.0,
            Death::Shrivel => 2.4,
            Death::Burst => 1.4,
        }
    }
}

/// What a creature does with itself when there's nothing else to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fidget {
    /// Its limbs' ends flex one after another, like drumming fingers.
    Drum,
    /// Its head lolls slowly over to one side, then snaps back upright.
    Tilt,
    /// A shiver runs through it.
    Shudder,
    /// A slow, deep swell, as if breathing in.
    Heave,
    /// Jerks round to look one way, then the other.
    Look,
    /// Picks at its own body with a limb.
    Pick,
}

impl Fidget {
    pub const ALL: [Fidget; 6] = [Fidget::Drum, Fidget::Tilt, Fidget::Shudder, Fidget::Heave, Fidget::Look, Fidget::Pick];
}

/// The moments the game hears about from `Animator::update`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cue {
    /// A strike has reached its target (or where the target was): the game decides whether
    /// it hit, and calls `Animator::blocked` if it was turned aside.
    Strike(Strike),
    /// A spell has gone off.
    Spell,
}

fn length(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    add(a, scale(sub(b, a), t))
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 0 below 0, 1 above 1, easing in and out between.
fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
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

/// The lowest point of the creature's surface (y is down, so the largest y); with `bulk`,
/// of its body and whatever else is a good part of its size, not thin limbs.
fn lowest(c: &Creature, bulk: bool) -> f32 {
    let volume = |h: [f32; 3]| h[0] * h[1] * h[2];
    let body = volume(c.parts[0].half);
    let mut bottom = f32::MIN;
    for (p, at) in c.parts.iter().zip(c.pose()) {
        if bulk && volume(p.half) < 0.2 * body {
            continue;
        }
        for k in 0..=4 {
            for j in 0..6 {
                bottom = bottom.max(at.world(p.local(k as f32 / 2.0 - 1.0, j as f32 / 6.0 * TAU, 1.0))[1]);
            }
        }
    }
    bottom
}

/// Which way a part is to point.
#[derive(Clone, Copy)]
enum Dir {
    /// A direction in the world.
    Along([f32; 3]),
    /// Toward a point in the world.
    At([f32; 3]),
    /// Toward the middle of the creature's body.
    Body,
    /// Away from the body across the floor, dipping by this much.
    Out(f32),
}

/// A part turned toward a direction blended between two (`u` of the way), `weight` of the
/// way from where it already points.
#[derive(Clone, Copy)]
struct Aim {
    part: usize,
    from: Dir,
    to: Dir,
    u: f32,
    weight: f32,
}

/// Points each part of `c` (parents first) as `want` says: part index → aim and weight.
fn aim_parts(c: &mut Creature, mut want: impl FnMut(usize) -> Option<(Dir, Dir, f32, f32)>) {
    let n = c.parts.len();
    let mut out: Vec<Placed> = Vec::with_capacity(n);
    for i in 0..n {
        let mut p = c.place(&out, i);
        if let Some((from, to, u, w)) = want(i).filter(|a| a.3 > 0.0) {
            let body = out.first().map_or(p.centre, |r| r.centre);
            let resolve = |d: Dir| match d {
                Dir::Along(v) => norm(v),
                Dir::At(at) => norm(sub(at, p.joint)),
                Dir::Body => norm(sub(body, p.joint)),
                Dir::Out(dip) => {
                    let o = sub(p.joint, body);
                    let flat = if o[0].hypot(o[2]) < 1e-3 { [0.0, 0.0, 1.0] } else { norm([o[0], 0.0, o[2]]) };
                    norm([flat[0], dip, flat[2]])
                }
            };
            let goal = norm(lerp(resolve(from), resolve(to), u));
            let now = apply(p.axes, [0.0, 1.0, 0.0]);
            let blend = lerp(now, goal, w.min(1.0));
            if dot(blend, blend) > 1e-6 && goal.iter().all(|v| v.is_finite()) {
                let axes = mul(arc(now, norm(blend)), p.axes);
                let (_, frame) = c.socket(&out, i);
                c.parts[i].rot = mul(transpose(frame), axes);
                p = c.place(&out, i);
            }
        }
        out.push(p);
    }
}

/// One pose of a strike or spell, laid over the gait.
#[derive(Clone, Copy)]
struct Key {
    /// Where the limbs it uses point, in the creature's frame with +x out to the limb's own
    /// side (y down, z ahead); None is straight at the target.
    dir: Option<[f32; 3]>,
    /// Added to those limbs' curl, and to their length as a fraction.
    bend: f32,
    stretch: f32,
    /// The body leaning back (radians), turning toward the limb's side, rising (a fraction of
    /// its height), thrown forward (a fraction of its reach) and swelling (a fraction).
    pitch: f32,
    twist: f32,
    rise: f32,
    shove: f32,
    swell: f32,
}

const fn key(dir: Option<[f32; 3]>, bend: f32, stretch: f32, pitch: f32, twist: f32, rise: f32, shove: f32, swell: f32) -> Key {
    Key { dir, bend, stretch, pitch, twist, rise, shove, swell }
}

/// A strike or spell: seconds spent winding up, striking, following through and recovering,
/// and the poses at the end of the first three.
struct Move {
    times: [f32; 4],
    keys: [Key; 3],
}

impl Move {
    fn hit(&self) -> f32 {
        self.times[0] + self.times[1]
    }

    fn end(&self) -> f32 {
        self.times.iter().sum()
    }

    /// The keys to blend between at `t`, how far between, and how much of the pose shows.
    fn at(&self, t: f32) -> (Key, Key, f32, f32) {
        let [wind, strike, follow, recover] = self.times;
        let k = &self.keys;
        if t < wind {
            (k[0], k[0], 0.0, smooth(t / wind))
        } else if t < wind + strike {
            // Fast and straight through: no easing out before the blow lands.
            let u = (t - wind) / strike;
            (k[0], k[1], u * u, 1.0)
        } else if t < wind + strike + follow {
            (k[1], k[2], smooth((t - wind - strike) / follow), 1.0)
        } else {
            (k[2], k[2], 0.0, 1.0 - smooth((t - wind - strike - follow) / recover))
        }
    }
}

fn strike_move(s: Strike) -> Move {
    match s {
        Strike::Swipe => Move {
            times: [0.45, 0.13, 0.18, 0.45],
            keys: [
                key(Some([0.7, -0.6, -0.4]), -0.3, 0.0, 0.12, -0.35, 0.0, -0.05, 0.0),
                key(None, 0.2, 0.1, -0.15, 0.3, 0.0, 0.15, 0.0),
                key(Some([-0.6, 0.5, 0.5]), 0.4, 0.0, -0.2, 0.45, 0.0, 0.1, 0.0),
            ],
        },
        Strike::Lash => Move {
            times: [0.5, 0.11, 0.4, 0.5],
            keys: [
                key(Some([0.3, -0.7, -0.6]), -1.3, -0.1, 0.12, -0.2, 0.0, -0.05, 0.0),
                key(None, 0.5, 0.25, -0.1, 0.2, 0.0, 0.1, 0.0),
                key(Some([-0.3, 0.6, 0.6]), -0.6, 0.1, -0.1, 0.25, 0.0, 0.05, 0.0),
            ],
        },
        Strike::Stab => Move {
            times: [0.4, 0.09, 0.22, 0.4],
            keys: [
                key(Some([0.15, -0.2, -0.9]), 0.0, -0.2, 0.18, -0.1, 0.0, -0.12, 0.0),
                key(None, 0.0, 0.5, -0.15, 0.05, 0.0, 0.25, 0.0),
                key(None, 0.0, 0.4, -0.12, 0.05, 0.0, 0.2, 0.0),
            ],
        },
        Strike::Lunge => Move {
            times: [0.5, 0.15, 0.2, 0.55],
            keys: [
                key(Some([0.0, -0.4, 0.5]), 0.0, 0.0, 0.3, 0.0, -0.05, -0.2, 0.08),
                key(None, 0.0, 0.15, -0.35, 0.0, 0.05, 0.6, -0.05),
                key(None, 0.0, 0.1, -0.3, 0.0, 0.03, 0.5, 0.0),
            ],
        },
        Strike::Slam => Move {
            times: [0.7, 0.17, 0.3, 0.6],
            keys: [
                key(Some([0.4, -0.9, 0.1]), -0.3, 0.1, 0.45, 0.0, 0.15, -0.05, 0.1),
                key(Some([0.2, 0.45, 0.85]), 0.2, 0.1, -0.4, 0.0, -0.1, 0.25, -0.05),
                key(Some([0.25, 0.5, 0.8]), 0.3, 0.0, -0.35, 0.0, -0.12, 0.25, -0.08),
            ],
        },
    }
}

/// Gathering the spell with limbs raised and spread, then flinging it at the target.
const CAST: Move = Move {
    times: [1.3, 0.15, 0.35, 0.6],
    keys: [
        key(Some([0.8, -0.6, 0.25]), -0.3, 0.05, 0.25, 0.0, 0.1, -0.05, 0.12),
        key(None, 0.2, 0.15, -0.25, 0.0, 0.0, 0.15, -0.04),
        key(None, 0.1, 0.1, -0.15, 0.0, 0.0, 0.1, 0.0),
    ],
};

/// Flung back up and away when a strike is turned aside, the body reeling back.
const REBOUND: Key = key(Some([0.5, -0.7, -0.6]), -0.7, -0.1, 0.4, -0.35, 0.04, -0.3, -0.05);
/// Seconds of the rebound, the stagger after it, and the recovery.
const REBOUND_TIMES: [f32; 3] = [0.18, 0.7, 0.6];

enum Act {
    Strike { strike: Strike, limbs: Vec<usize>, target: [f32; 3], t: f32, blocked: Option<f32>, cued: bool },
    Cast { target: [f32; 3], t: f32, cued: bool },
    Fidget { fidget: Fidget, t: f32, length: f32, side: f32, limb: Option<usize> },
}

/// The whole body's share of what's laid over the gait.
#[derive(Clone, Copy, Default)]
struct Body {
    pitch: f32,
    roll: f32,
    twist: f32,
    rise: f32,
    shove: f32,
    swell: f32,
}

/// Everything laid over the gait this frame.
#[derive(Default)]
struct Over {
    body: Body,
    aims: Vec<Aim>,
    /// Added curl and length (a fraction) per part.
    shapes: Vec<(usize, f32, f32)>,
    /// Turns in a part's own frame.
    turns: Vec<(usize, M3)>,
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
    /// Which side of the body it grows from (+1 or -1).
    side: f32,
    /// Grows straight from the trunk and is long enough to strike or gesture with.
    limb: bool,
    shape: Shape,
    /// Length over girth.
    slender: f32,
}

pub struct Animator {
    pub plan: Plan,
    rest: Vec<(M3, f32, [f32; 3], f32)>,
    legs: Vec<Leg>,
    free: Vec<Free>,
    /// The trunk's far end, when it isn't the root: a head, or something like one.
    head: Option<usize>,
    in_leg: Vec<bool>,
    /// Each part's side of the body.
    sides: Vec<f32>,
    leg_length: f32,
    step_time: f32,
    threshold: f32,
    walk_speed: f32,
    run_speed: f32,
    radius: f32,
    height: f32,
    spawn: [f32; 3],
    /// Lift above the smoothed floor is carried by the root's anchor; this is that floor.
    floor: f32,
    rng: Rng,
    time: f32,
    /// The walk's phase, in half strides.
    stride: f32,
    speed: f32,
    going: bool,
    wait: f32,
    want_yaw: f32,
    pitch: f32,
    chase: Option<[f32; 3]>,
    /// The player's feet, which its body keeps out of.
    pub player: Option<[f32; 3]>,
    /// Something to stop and face (a person turning to the player near them), unless chasing.
    pub attend: Option<[f32; 3]>,
    /// Furthest it strays from where it was grown, unless chasing.
    pub leash: f32,
    /// Stays where it was put, facing the way it was put, unless something draws it.
    pub stay: bool,
    home_yaw: f32,
    /// What it does when it's stood still a while.
    fidgets: Vec<Fidget>,
    act: Option<Act>,
    /// Seconds stood still, and how long until it fidgets.
    still: f32,
    restless: f32,
    /// Where the body was thrown this frame, taken back before the next.
    offset: [f32; 3],
    death: Option<(Death, f32, f32)>,
    fall: f32,
    settled: bool,
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
        let head = (at != 0).then_some(at);
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
        let sides: Vec<f32> = placed.iter().map(|p| if apply(to_body, sub(p.joint, c.base))[0] >= 0.0 { 1.0 } else { -1.0 }).collect();
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
            let side = sides[i];
            let arm = (p.shape == Shape::Capsule && on_trunk && down > 0.5).then_some(side);
            let limb = on_trunk && p.shape != Shape::Ellipsoid && 2.0 * p.half[1] > 0.12 * height;
            let slender = p.half[1] / p.half[0].max(p.half[2]).max(1e-3);
            free.push(Free { part: i, sway, bend, speed: rng.fr(1.2, 2.6), phase: rng.fr(0.0, TAU), arm, side, limb, shape: p.shape, slender });
        }

        let leg_length = if legs.is_empty() { 0.5 * TF } else { legs.iter().map(|l| l.lengths.iter().sum::<f32>()).sum::<f32>() / legs.len() as f32 };
        let step_time = 0.34 * (leg_length / (0.5 * TF)).max(0.2).sqrt();
        let threshold = 0.42 * leg_length;
        let (walk_speed, run_speed) = match plan {
            // A run takes longer strides more quickly (see `update`).
            Plan::Walk => (0.45 * threshold / step_time, (0.75 * 1.5 * threshold / (0.6 * step_time)).min(RUN_CAP)),
            Plan::Creep => (CREEP_SPEED, CREEP_RUN),
            Plan::Float => (FLOAT_SPEED, FLOAT_RUN),
        };
        let radius = placed.iter().map(|p| (p.centre[0] - c.base[0]).hypot(p.centre[2] - c.base[2])).fold(0.0, f32::max);
        let wait = rng.fr(0.0, 2.0);
        let restless = rng.fr(1.5, 5.0);
        Animator {
            plan,
            rest: c.parts.iter().map(|p| (p.rot, p.bend, p.half, p.lumps)).collect(),
            legs,
            free,
            head,
            in_leg,
            sides,
            leg_length,
            step_time,
            threshold,
            walk_speed,
            run_speed: run_speed.max(1.5 * walk_speed),
            radius,
            height,
            spawn: c.base,
            floor,
            rng,
            time: 0.0,
            stride: 0.0,
            speed: 0.0,
            going: false,
            wait,
            want_yaw: c.yaw,
            pitch: 0.0,
            chase: None,
            player: None,
            attend: None,
            leash: LEASH,
            stay: false,
            home_yaw: c.yaw,
            fidgets: Fidget::ALL.to_vec(),
            act: None,
            still: 0.0,
            restless,
            offset: [0.0; 3],
            death: None,
            fall: 0.0,
            settled: false,
        }
    }

    /// Go about at an easy walk, fidgeting only in the ways given (a person, not a creature).
    pub fn calm(&mut self, fidgets: &[Fidget]) {
        self.walk_speed *= 0.7;
        self.fidgets = fidgets.to_vec();
    }

    /// How many legs it walks on.
    pub fn legs(&self) -> usize {
        self.legs.len()
    }

    /// Run after `target` (a point in the world), or with None go back to wandering.
    pub fn chase(&mut self, target: Option<[f32; 3]>) {
        if target.is_some() && matches!(self.act, Some(Act::Fidget { .. })) {
            self.act = None;
        }
        self.chase = target;
    }

    pub fn dead(&self) -> bool {
        self.death.is_some()
    }

    /// Striking or casting, so not free to start anything else.
    pub fn busy(&self) -> bool {
        self.dead() || matches!(self.act, Some(Act::Strike { .. } | Act::Cast { .. }))
    }

    /// Running rather than walking.
    pub fn running(&self) -> bool {
        self.speed > 0.5 * (self.walk_speed + self.run_speed)
    }

    /// How far from its base a strike lands.
    pub fn reach(&self) -> f32 {
        1.1 * self.radius + 0.3 * TF
    }

    /// The strikes this body can make; a lunge always.
    pub fn strikes(&self) -> Vec<Strike> {
        Strike::ALL.into_iter().filter(|&s| s == Strike::Lunge || !self.limbs_for(s).is_empty() || (s == Strike::Slam && self.plan == Plan::Creep)).collect()
    }

    /// The deaths that suit this body.
    pub fn deaths(&self) -> Vec<Death> {
        match self.plan {
            Plan::Walk => Death::ALL.to_vec(),
            Plan::Creep => vec![Death::Crumple, Death::Melt, Death::Shrivel, Death::Burst],
            Plan::Float => vec![Death::Topple, Death::Melt, Death::Shrivel, Death::Burst],
        }
    }

    /// The parts a strike is made with.
    fn limbs_for(&self, s: Strike) -> Vec<usize> {
        let limbs = || self.free.iter().filter(|f| f.limb);
        let longest = |it: Vec<&Free>| it.iter().max_by(|a, b| a.slender.total_cmp(&b.slender)).map(|f| vec![f.part]).unwrap_or_default();
        match s {
            Strike::Swipe => longest(limbs().filter(|f| f.arm.is_some() || f.shape == Shape::Capsule).collect()),
            Strike::Lash => longest(limbs().filter(|f| f.shape == Shape::Tentacle).collect()),
            Strike::Stab => longest(limbs().filter(|f| f.shape == Shape::Cone || (f.shape == Shape::Capsule && f.slender > 2.5)).collect()),
            Strike::Lunge => self.head.into_iter().collect(),
            Strike::Slam => limbs().filter(|f| f.shape != Shape::Cone).map(|f| f.part).collect(),
        }
    }

    /// Start a strike at `target` (a point in the world), unless already busy.
    pub fn attack(&mut self, strike: Strike, target: [f32; 3]) -> bool {
        if self.busy() {
            return false;
        }
        let mut limbs = self.limbs_for(strike);
        // Sometimes the other arm, for a swipe.
        if strike == Strike::Swipe && self.rng.chance(0.5) {
            let other = self.free.iter().filter(|f| f.limb && f.arm.is_some() && !limbs.contains(&f.part)).map(|f| f.part).next();
            limbs = other.map_or(limbs, |o| vec![o]);
        }
        self.act = Some(Act::Strike { strike, limbs, target, t: 0.0, blocked: None, cued: false });
        true
    }

    /// The strike that just landed was turned aside: it rebounds and the creature staggers.
    pub fn blocked(&mut self) -> bool {
        match &mut self.act {
            Some(Act::Strike { t, blocked: b @ None, cued: true, .. }) => {
                *b = Some(*t);
                true
            }
            _ => false,
        }
    }

    /// Start a spell at `target`, unless already busy.
    pub fn cast(&mut self, target: [f32; 3]) -> bool {
        if self.busy() {
            return false;
        }
        self.act = Some(Act::Cast { target, t: 0.0, cued: false });
        true
    }

    /// Start fidgeting now (the herd does this by itself when a creature stands about).
    pub fn fidget(&mut self, fidget: Fidget) {
        if self.busy() {
            return;
        }
        let side = if self.rng.chance(0.5) { 1.0 } else { -1.0 };
        let limbs: Vec<usize> = self.free.iter().filter(|f| f.limb).map(|f| f.part).collect();
        let limb = (!limbs.is_empty()).then(|| self.rng.pick(&limbs));
        let length = match fidget {
            Fidget::Tilt => self.rng.fr(2.5, 4.0),
            Fidget::Look => 3.2,
            Fidget::Heave => 3.2,
            _ => self.rng.fr(1.8, 3.2),
        };
        self.act = Some(Act::Fidget { fidget, t: 0.0, length, side, limb });
        // It stands about until it's done.
        self.going = false;
        self.wait = self.wait.max(length);
    }

    /// Die, falling to the floor and lying there for good.
    pub fn die(&mut self, death: Death) {
        if self.death.is_none() {
            let side = if self.rng.chance(0.5) { 1.0 } else { -1.0 };
            self.death = Some((death, 0.0, side));
            self.act = None;
            self.chase = None;
        }
    }

    fn ground(scene: &Scene, x: f32, y: f32, z: f32) -> f32 {
        scene.floor_y(x, y - 16.0, z).unwrap_or(y)
    }

    /// Can it take its next stride? Looks ahead and to either side of its way.
    fn clear(&self, c: &Creature, scene: &Scene, dir: [f32; 3]) -> bool {
        let ahead = 0.6 * self.radius + 0.5 * self.speed.max(self.walk_speed);
        let across = [dir[2], 0.0, -dir[0]];
        [0.0, 0.5, -0.5].iter().all(|&s| {
            let p = add(c.base, add(scale(dir, ahead), scale(across, s * self.radius)));
            scene.walkable_y(p[0], c.base[1] - 16.0, p[2]).is_some_and(|y| (y - c.base[1]).abs() <= MAX_STEP * 0.6)
        })
    }

    /// Would moving at `velocity` for `dt` take its body into the player's, or nearer while
    /// they already touch?
    fn into_player(&self, c: &Creature, velocity: [f32; 3], dt: f32) -> bool {
        let Some(p) = self.player else { return false };
        let near = c.girth + super::walker::RADIUS;
        let gap = |b: [f32; 3]| (p[0] - b[0]).hypot(p[2] - b[2]);
        let next = add(c.base, scale(velocity, dt));
        c.girth > 0.0 && (p[1] - c.base[1]).abs() < 1.5 * TF && gap(next) < near && gap(next) < gap(c.base)
    }

    /// Decides where to go and moves the base; returns the velocity.
    fn wander(&mut self, c: &mut Creature, scene: &Scene, dt: f32) -> [f32; 3] {
        self.wait -= dt;
        if let Some(target) = self.chase {
            let d = sub(target, c.base);
            self.want_yaw = (-d[0]).atan2(d[2]);
            self.going = d[0].hypot(d[2]) > 0.8 * self.reach();
            self.wait = 0.5;
        } else if let Some(target) = self.attend {
            let d = sub(target, c.base);
            self.want_yaw = (-d[0]).atan2(d[2]);
            self.going = false;
            self.wait = self.rng.fr(0.5, 1.5);
        } else {
            if self.wait <= 0.0 {
                self.going = !self.stay && self.rng.chance(0.65);
                self.wait = if self.going { self.rng.fr(2.0, 6.0) } else { self.rng.fr(1.0, 3.5) };
                self.want_yaw = if self.stay { self.home_yaw } else { c.yaw + self.rng.fr(-1.5, 1.5) };
            }
            let home = sub(self.spawn, c.base);
            if home[0].hypot(home[2]) > self.leash.max(1.0) {
                self.want_yaw = (-home[0]).atan2(home[2]);
                self.going = true;
            } else if self.stay && home[0].hypot(home[2]) < 0.5 * self.leash {
                self.going = false;
            }
        }
        // Striking or casting, it turns to face its target but stands.
        match &self.act {
            Some(Act::Strike { target, .. } | Act::Cast { target, .. }) => {
                let d = sub(*target, c.base);
                self.want_yaw = (-d[0]).atan2(d[2]);
            }
            Some(Act::Fidget { .. }) if self.going && self.chase.is_none() => self.act = None,
            _ => {}
        }
        let dir = apply(rot_y(c.yaw), [0.0, 0.0, 1.0]);
        if self.going && !self.clear(c, scene, dir) {
            if self.chase.is_none() {
                let turn = self.rng.fr(1.5, PI);
                self.want_yaw = c.yaw + if self.rng.chance(0.5) { turn } else { -turn };
                self.wait = 0.7;
            }
            self.going = false;
        }
        let miss = (self.want_yaw - c.yaw).sin().atan2((self.want_yaw - c.yaw).cos());
        let turn = if self.chase.is_some() { CHASE_TURN_RATE } else { TURN_RATE };
        c.yaw += miss.clamp(-turn * dt, turn * dt);
        let top = if self.chase.is_some() { self.run_speed } else { self.walk_speed };
        let want = if !self.going || self.act.is_some() { 0.0 } else if miss.abs() > 0.5 { 0.3 * top } else { top };
        self.speed += (want - self.speed) * (6.0 * dt).min(1.0);
        let mut velocity = scale(apply(rot_y(c.yaw), [0.0, 0.0, 1.0]), self.speed);
        if self.into_player(c, velocity, dt) {
            self.speed = 0.0;
            velocity = [0.0; 3];
        }
        c.base[0] += velocity[0] * dt;
        c.base[2] += velocity[2] * dt;
        let floor = Self::ground(scene, c.base[0], self.floor, c.base[2]);
        self.floor += (floor - self.floor) * (10.0 * dt).min(1.0);
        velocity
    }

    fn step_legs(&mut self, c: &Creature, scene: &Scene, velocity: [f32; 3], step_time: f32, threshold: f32, dt: f32) {
        let frame = rot_y(c.yaw);
        let lead = 0.6 * step_time;
        let moving = self.speed > 0.1 * self.walk_speed;
        let mut stepping = [false; 2];
        for l in &self.legs {
            stepping[l.group] |= l.step.is_some();
        }
        for l in &mut self.legs {
            let place = add(add(c.base, apply(frame, [l.home[0], 0.0, l.home[1]])), scale(velocity, lead));
            let target = [place[0], Self::ground(scene, place[0], self.floor, place[2]) - l.lift, place[2]];
            if let Some((from, t)) = &mut l.step {
                *t += dt / step_time;
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
            let wanted = off > if moving { threshold } else { 0.12 * self.leg_length };
            if (wanted && !stepping[1 - l.group]) || off > 2.2 * threshold {
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

    /// Moves the current strike, spell or fidget on by `dt` and says how it bends the body;
    /// also whether a strike landed or a spell went off just now.
    fn act(&mut self, c: &Creature, dt: f32) -> (Over, Option<Cue>) {
        let mut over = Over::default();
        let mut cue = None;
        let frame = rot_y(c.yaw);
        let mut done = false;
        match &mut self.act {
            None => {}
            Some(Act::Strike { strike, limbs, target, t, blocked, cued }) => {
                *t += dt;
                let mv = strike_move(*strike);
                if !*cued && *t >= mv.hit() {
                    *cued = true;
                    cue = Some(Cue::Strike(*strike));
                }
                let side = limbs.first().map_or(1.0, |&l| self.sides[l]);
                let (a, b, u, w, shake) = match *blocked {
                    None => {
                        done = *t >= mv.end();
                        let (a, b, u, w) = mv.at(*t);
                        (a, b, u, w, 0.0)
                    }
                    Some(at) => {
                        let e = *t - at;
                        let [hit, reel, recover] = REBOUND_TIMES;
                        done = e >= hit + reel + recover;
                        let shake = if e > hit { (1.0 - (e - hit) / reel).max(0.0) } else { 0.0 };
                        if e < hit {
                            (mv.keys[1], REBOUND, smooth(e / hit), 1.0, shake)
                        } else {
                            (REBOUND, REBOUND, 0.0, 1.0 - smooth((e - hit - reel) / recover), shake)
                        }
                    }
                };
                let mut body = blend(a, b, u, w, side);
                if blocked.is_some() {
                    // Staggering back, shaken.
                    body.roll += 0.1 * shake * (28.0 * *t).sin();
                    body.shove -= 0.25 * w * (1.0 - shake);
                }
                if self.plan == Plan::Creep && *strike == Strike::Slam {
                    body.swell += 0.15 * w * (1.0 - u);
                }
                over.body = body;
                for &l in limbs.iter() {
                    let s = self.sides[l];
                    over.shapes.push((l, w * (mix(a.bend, b.bend, u) + 0.3 * shake * (40.0 * *t).sin()), w * mix(a.stretch, b.stretch, u)));
                    let to = |k: Key| k.dir.map_or(Dir::At(*target), |d| Dir::Along(apply(frame, [d[0] * s, d[1], d[2]])));
                    // A head lunging only nods at its target; a limb points.
                    let reach = if *strike == Strike::Lunge { 0.5 } else { 1.0 };
                    over.aims.push(Aim { part: l, from: to(a), to: to(b), u, weight: w * reach });
                }
                // The lash's crack runs on down the tentacle.
                if *strike == Strike::Lash && *t > mv.hit() && blocked.is_none() {
                    let e = *t - mv.hit();
                    for &l in limbs.iter() {
                        over.shapes.push((l, 0.9 * (-4.0 * e).exp() * (30.0 * e).sin(), 0.0));
                    }
                }
            }
            Some(Act::Cast { target, t, cued }) => {
                *t += dt;
                if !*cued && *t >= CAST.hit() {
                    *cued = true;
                    cue = Some(Cue::Spell);
                }
                done = *t >= CAST.end();
                let (a, b, u, w) = CAST.at(*t);
                let gathering = smooth(*t / CAST.times[0]) * (*t < CAST.hit()) as i32 as f32;
                over.body = blend(a, b, u, w, 1.0);
                over.body.swell += 0.05 * gathering * (12.0 * *t).sin();
                over.body.roll += 0.03 * gathering * (31.0 * *t).sin();
                for (k, f) in self.free.iter().filter(|f| f.limb).enumerate() {
                    // Each limb trembles with the gathering.
                    let jitter = [0.0, 1.0, 2.0].map(|o| 0.25 * gathering * ((23.0 + 7.0 * o) * *t + k as f32 * 1.7 + o).sin());
                    let to = |key: Key| {
                        key.dir.map_or(Dir::At(*target), |d| Dir::Along(apply(frame, add([d[0] * f.side, d[1], d[2]], jitter))))
                    };
                    over.shapes.push((f.part, w * mix(a.bend, b.bend, u), w * mix(a.stretch, b.stretch, u)));
                    over.aims.push(Aim { part: f.part, from: to(a), to: to(b), u, weight: w });
                }
            }
            Some(Act::Fidget { fidget, t, length, side, limb }) => {
                *t += dt;
                let (t, length, side) = (*t, *length, *side);
                done = t >= length;
                let env = smooth(t / 0.4) * smooth((length - t) / 0.4);
                let b = &mut over.body;
                match fidget {
                    Fidget::Drum => {
                        let limbs: Vec<usize> = self.free.iter().filter(|f| f.limb || f.shape == Shape::Tentacle).map(|f| f.part).take(6).collect();
                        for (k, &p) in limbs.iter().enumerate() {
                            over.shapes.push((p, 0.45 * env * (16.0 * t - 0.9 * k as f32).sin().max(0.0), 0.0));
                        }
                        if limbs.is_empty() {
                            b.roll += 0.04 * env * (16.0 * t).sin();
                        }
                    }
                    Fidget::Tilt => {
                        // Lolls over slowly, holds a moment, then snaps back.
                        let snap = 0.15;
                        let v = if t < length - snap { smooth(t / 1.4) } else { 1.0 - (t - length + snap) / snap };
                        let angle = side * (0.75 * v.max(0.0) + 0.03 * v.max(0.0) * (40.0 * t).sin());
                        match self.head {
                            Some(h) => over.turns.push((h, rot_z(angle))),
                            None => b.roll += 0.4 * angle,
                        }
                    }
                    Fidget::Shudder => {
                        b.roll += 0.06 * env * (37.0 * t).sin();
                        b.swell += 0.03 * env * (23.0 * t).sin();
                        for (k, f) in self.free.iter().enumerate().take(8) {
                            over.shapes.push((f.part, 0.2 * env * (41.0 * t + k as f32).sin(), 0.0));
                        }
                    }
                    Fidget::Heave => {
                        let h = (PI * t / length * 2.0).sin().powi(2) * env;
                        b.swell += 0.14 * h;
                        b.pitch += 0.12 * h;
                        b.rise += 0.03 * h;
                    }
                    Fidget::Look => {
                        // Still, then a jerk to one side, then the other, then back.
                        let jerk = |at: f32| smooth((t - at) / 0.08);
                        let v = jerk(0.4) - 2.0 * jerk(1.5) + jerk(2.6);
                        b.twist += 0.45 * side * v;
                        if let Some(h) = self.head {
                            over.turns.push((h, rot_z(0.25 * side * v)));
                        }
                    }
                    Fidget::Pick => {
                        if let Some(l) = limb {
                            over.aims.push(Aim { part: *l, from: Dir::Body, to: Dir::Body, u: 0.0, weight: 0.7 * env });
                            over.shapes.push((*l, 0.3 * env * (12.0 * t).sin(), 0.0));
                        } else {
                            b.swell += 0.04 * env * (12.0 * t).sin();
                        }
                    }
                }
            }
        }
        if done {
            self.act = None;
        }
        (over, cue)
    }

    /// Lays the dying (or dead) body out as its death has got to.
    fn dying(&mut self, c: &mut Creature, dt: f32) {
        let Some((death, t, side)) = &mut self.death else { return };
        *t += dt;
        let (death, t, side) = (*death, *t, *side);
        let u = smooth(t / death.length());
        let n = c.parts.len();
        let walker = self.plan == Plan::Walk;
        let frame = rot_y(c.yaw);
        let shape = |c: &mut Creature, i: usize, along: f32, across: f32| {
            let h = &mut c.parts[i].half;
            *h = [h[0] * across, h[1] * along, h[2] * across];
        };
        let mut roll = 0.0;
        let mut pitch = 0.0;
        // Each part's aim, set below.
        let mut aims: Vec<Option<(Dir, f32)>> = vec![None; n];
        match death {
            Death::Topple => {
                // Faster and faster as it goes over, a little bounce as it lands.
                let fall = (t / death.length()).min(1.0).powi(2);
                let bounce = if t > death.length() { 0.06 * (-(t - death.length()) * 6.0).exp() * ((t - death.length()) * 18.0).sin() } else { 0.0 };
                roll = side * (1.45 * fall - bounce);
                for i in 1..n {
                    if !self.in_leg[i] {
                        aims[i] = Some((Dir::Out(0.35), 0.5 * u));
                    }
                }
            }
            Death::Crumple => {
                pitch = -0.35 * u;
                roll = side * 0.15 * u;
                for i in 1..n {
                    let s = self.sides[i];
                    let hip = self.legs.iter().any(|l| l.chain[0] == i);
                    aims[i] = Some(if hip {
                        (Dir::Along(apply(frame, [s, 0.25, 0.15])), u)
                    } else if self.in_leg[i] {
                        (Dir::Along(apply(frame, [0.6 * s, 1.0, 0.1])), 0.8 * u)
                    } else {
                        (Dir::Out(0.4), 0.6 * u)
                    });
                }
                if !walker {
                    shape(c, 0, 1.0 - 0.35 * u, 1.0 + 0.2 * u);
                }
            }
            Death::Curl => {
                // Over onto its back, then the legs draw in, twitching less and less.
                roll = side * PI * smooth(t / 1.0) * 0.97;
                let draw = smooth((t - 0.6) / 1.6);
                let twitch = 1.0 - 0.3 * (18.0 * t).sin().powi(2) * (1.0 - draw);
                for i in 1..n {
                    aims[i] = Some((Dir::Body, 0.7 * draw * twitch));
                    c.parts[i].bend += 0.8 * draw * self.sides[i];
                }
            }
            Death::Melt => {
                for i in 0..n {
                    if i == 0 {
                        shape(c, i, 1.0 - 0.7 * u, 1.0 + 0.8 * u);
                        c.parts[i].half[0] *= 1.0 + 0.05 * (1.0 - u) * (7.0 * t).sin();
                    } else {
                        shape(c, i, 1.0 - 0.5 * u, 1.0 + 0.25 * u);
                        aims[i] = Some((Dir::Out(0.25), 0.9 * u));
                    }
                    c.parts[i].lumps += 0.4 * u;
                }
            }
            Death::Shrivel => {
                roll = 0.07 * (1.0 - u) * (33.0 * t).sin() + side * 0.5 * u;
                for i in 0..n {
                    shape(c, i, 1.0 - 0.25 * u, 1.0 - 0.55 * u);
                    c.parts[i].lumps += 0.3 * u;
                    if i > 0 {
                        aims[i] = Some((Dir::Body, 0.6 * u));
                        c.parts[i].bend += 0.9 * u * self.sides[i];
                    }
                }
            }
            Death::Burst => {
                let swell = smooth(t / 0.9);
                let pop = smooth((t - 0.9) / 0.25);
                roll = 0.05 * swell * (1.0 - pop) * (45.0 * t).sin();
                for i in 0..n {
                    let grown = 1.0 + 0.4 * swell;
                    let (along, across) = if i == 0 { (grown - 1.0 * pop, grown + 0.2 * pop) } else { (grown - 0.8 * pop, grown - 0.7 * pop) };
                    shape(c, i, along, across);
                    if i > 0 {
                        aims[i] = Some((Dir::Out(0.1), pop));
                    }
                }
            }
        }
        c.parts[0].rot = mul(mul(rot_x(pitch), rot_z(roll)), self.rest[0].0);
        aim_parts(c, |i| aims[i].map(|(d, w)| (d, d, 0.0, w)));
    }

    /// The dead fall to the floor and lie on it. Gone soft, they lie on their bodies, their
    /// limbs flat on the floor whatever way they lie.
    fn settle(&mut self, c: &mut Creature, dt: f32) {
        let soft = self.death.is_some_and(|(d, ..)| matches!(d, Death::Melt | Death::Shrivel | Death::Burst));
        let gap = self.floor - lowest(c, soft);
        if gap > 1.0 {
            self.fall += GRAVITY * dt;
            let drop = (self.fall * dt).min(gap);
            c.base[1] += drop;
            if drop >= gap {
                self.fall = 0.0;
            }
        } else {
            c.base[1] += gap;
            self.fall = 0.0;
        }
        if self.death.is_some_and(|(d, t, _)| t > d.length() + 0.8) && self.fall == 0.0 {
            self.settled = true;
        }
    }

    /// Moves `c` on by `dt` seconds; says when a strike lands or a spell goes off.
    pub fn update(&mut self, c: &mut Creature, scene: &Scene, dt: f32) -> Option<Cue> {
        if self.settled {
            return None;
        }
        let dt = dt.min(0.05);
        self.time += dt;
        c.base = sub(c.base, self.offset);
        self.offset = [0.0; 3];
        for (p, &(rot, bend, half, lumps)) in c.parts.iter_mut().zip(&self.rest) {
            p.rot = rot;
            p.bend = bend;
            p.half = half;
            p.lumps = lumps;
        }
        if self.death.is_some() {
            self.speed = 0.0;
            self.dying(c, dt);
            self.settle(c, dt);
            return None;
        }

        let velocity = self.wander(c, scene, dt);
        if self.act.is_none() && self.chase.is_none() && self.speed < 0.05 * self.walk_speed {
            self.still += dt;
            if self.still > self.restless {
                let f = self.rng.pick(&self.fidgets);
                self.fidget(f);
                self.still = 0.0;
                self.restless = self.rng.fr(2.0, 6.0);
            }
        } else {
            self.still = 0.0;
        }
        let (over, cue) = self.act(c, dt);
        let pace = (self.speed / self.walk_speed).clamp(0.0, 1.0);
        let run = ((self.speed - self.walk_speed) / (self.run_speed - self.walk_speed)).clamp(0.0, 1.0);
        // Running, it takes longer strides more often.
        let step_time = self.step_time * (1.0 - 0.4 * run);
        let threshold = self.threshold * (1.0 + 0.5 * run);
        self.stride += dt * PI / step_time;
        let stride = self.stride;
        let breath = self.time * 1.7;
        for f in &self.free {
            let wave = self.time * f.speed + f.phase;
            let p = &mut c.parts[f.part];
            let swing = f.arm.map_or(0.0, |side| (0.4 + 0.5 * run) * pace * side * stride.sin());
            p.rot = mul(p.rot, mul(rot_x(f.sway * wave.sin() + swing), rot_z(f.sway * 0.7 * (wave * 0.8).cos())));
            p.bend += f.bend * (wave * 1.3).sin();
        }
        let (mut pitch, mut roll) = (0.0, 0.0);
        match self.plan {
            Plan::Walk => {
                self.step_legs(c, scene, velocity, step_time, threshold, dt);
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
                // Leaning into a run.
                pitch = self.pitch - 0.18 * run;
                roll = (0.05 + 0.04 * run) * pace * stride.sin() + 0.015 * breath.sin();
                let bob = 0.035 + 0.05 * run;
                c.base[1] = self.floor + bob * self.leg_length * pace * (2.0 * stride).sin() + 0.01 * self.leg_length * breath.sin();
            }
            Plan::Creep => {
                let squash = (0.04 + 0.1 * pace + 0.1 * run) * (self.time * (2.0 + 3.0 * pace + 5.0 * run)).sin();
                let h = &mut c.parts[0].half;
                // Running, it stretches out along its way.
                *h = [h[0] * (1.0 - 0.5 * squash), h[1] * (1.0 + squash - 0.1 * run), h[2] * (1.0 - 0.5 * squash + 0.25 * run)];
                c.base[1] = self.floor;
            }
            Plan::Float => {
                c.base[1] = self.floor + 0.08 * TF * (breath * (1.0 + run)).sin();
                pitch = -0.3 * run;
                roll = 0.06 * (breath * 0.7).sin();
            }
        }
        let b = over.body;
        c.base[1] -= b.rise * self.height;
        c.parts[0].rot = mul(mul(rot_y(b.twist), mul(rot_x(pitch + b.pitch), rot_z(roll + b.roll))), self.rest[0].0);
        c.parts[0].half = scale(c.parts[0].half, 1.0 + b.swell);
        self.offset = scale(apply(rot_y(c.yaw), [0.0, 0.0, 1.0]), b.shove * self.reach());
        c.base = add(c.base, self.offset);
        if self.plan == Plan::Walk {
            self.solve_legs(c);
        }
        for &(i, bend, stretch) in &over.shapes {
            c.parts[i].bend += bend;
            c.parts[i].half[1] *= (1.0 + stretch).max(0.3);
        }
        for &(i, m) in &over.turns {
            c.parts[i].rot = mul(c.parts[i].rot, m);
        }
        if !over.aims.is_empty() {
            aim_parts(c, |i| over.aims.iter().find(|a| a.part == i).map(|a| (a.from, a.to, a.u, a.weight)));
        }
        cue
    }
}

/// The body's share of a blend `u` of the way from key `a` to `b`, `w` of it showing.
fn blend(a: Key, b: Key, u: f32, w: f32, side: f32) -> Body {
    let m = |x: f32, y: f32| w * mix(x, y, u);
    Body {
        pitch: m(a.pitch, b.pitch),
        roll: 0.0,
        twist: side * m(a.twist, b.twist),
        rise: m(a.rise, b.rise),
        shove: m(a.shove, b.shove),
        swell: m(a.swell, b.swell),
    }
}

/// Every creature of a scene moving; remade when the scene's `version` changes.
#[derive(Default)]
pub struct Herd {
    version: Option<u64>,
    all: Vec<Animator>,
    /// The player's feet, which the creatures don't walk into.
    pub player: Option<[f32; 3]>,
}

impl Herd {
    /// Moves every creature on by `dt`; returns which creatures' strikes landed or spells
    /// went off.
    pub fn update(&mut self, scene: &mut Scene, version: u64, dt: f32) -> Vec<(usize, Cue)> {
        if self.version != Some(version) {
            self.all = scene.creatures.iter().map(Animator::new).collect();
            self.version = Some(version);
        }
        let mut creatures = std::mem::take(&mut scene.creatures);
        let mut cues = Vec::new();
        for (i, (a, c)) in self.all.iter_mut().zip(&mut creatures).enumerate() {
            a.player = self.player;
            if let Some(cue) = a.update(c, scene, dt) {
                cues.push((i, cue));
            }
            // The dead can be walked over.
            if a.dead() {
                c.girth = 0.0;
            }
        }
        scene.creatures = creatures;
        cues
    }

    /// The creatures' animators, in the order of the scene's creatures (empty before the
    /// first `update` of a level).
    pub fn animators(&mut self) -> &mut [Animator] {
        &mut self.all
    }

    pub fn plans(&self) -> Vec<Plan> {
        self.all.iter().map(|a| a.plan).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{rng::Rng, walkable_scene};

    fn finite(c: &Creature) -> bool {
        c.pose().iter().all(|p| p.centre.iter().chain(p.axes.iter().flatten()).all(|v| v.is_finite()))
            && c.parts.iter().all(|p| p.half.iter().all(|&h| h.is_finite() && h > 0.0))
    }

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
                    assert!(finite(c), "seed {seed} frame {frame}");
                    // A fidget can tip the body; the feet still stay put.
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

    /// Runs `frames` of every creature in a few scenes, `f` setting each going first.
    fn each(frames: usize, mut start: impl FnMut(&mut Animator, &Creature), mut check: impl FnMut(&Animator, &Creature, usize, Option<Cue>)) {
        for seed in 1..5 {
            let (mut scene, _) = walkable_scene(&mut Rng::new(seed), 30);
            let mut herd = Herd::default();
            herd.update(&mut scene, 0, 1.0 / 30.0);
            for (a, c) in herd.all.iter_mut().zip(&scene.creatures) {
                start(a, c);
            }
            for frame in 0..frames {
                let mut cues = vec![None; scene.creatures.len()];
                for (i, cue) in herd.update(&mut scene, 0, 1.0 / 30.0) {
                    cues[i] = Some(cue);
                }
                for ((a, c), cue) in herd.all.iter().zip(&scene.creatures).zip(cues) {
                    assert!(finite(c), "seed {seed} frame {frame}");
                    check(a, c, frame, cue);
                }
            }
        }
    }

    #[test]
    fn runs_faster_than_it_walks() {
        // Some are boxed in where they grew: something somewhere must run.
        let mut ran = false;
        for seed in 1..5 {
            let (mut scene, _) = walkable_scene(&mut Rng::new(seed), 30);
            let a: Vec<Animator> = scene.creatures.iter().map(Animator::new).collect();
            for a in &a {
                assert!(a.run_speed > 1.4 * a.walk_speed, "{:?}: walks {} runs {}", a.plan, a.walk_speed, a.run_speed);
            }
            // Something chased far off gets going faster than it wanders.
            let mut herd = Herd::default();
            herd.update(&mut scene, 0, 1.0 / 30.0);
            for (a, c) in herd.all.iter_mut().zip(&scene.creatures) {
                // Somewhere it has room to get going toward.
                let open = (0..8).map(|k| apply(rot_y(k as f32 * TAU / 8.0), [0.0, 0.0, 1.0])).find(|&d| a.clear(c, &scene, d));
                a.chase(open.map(|d| add(c.base, scale(d, 20.0 * TF))));
            }
            for _ in 0..90 {
                herd.update(&mut scene, 0, 1.0 / 30.0);
                ran |= herd.all.iter().any(|a| a.running());
            }
        }
        assert!(ran, "nothing ran");
    }

    #[test]
    fn a_chase_stops_at_the_players_body() {
        let near = |c: &Creature| c.girth + crate::world::walker::RADIUS;
        let mut closed = 0;
        for seed in 1..6 {
            let (mut scene, _) = walkable_scene(&mut Rng::new(seed), 30);
            scene.set_creatures_move(true);
            let mut herd = Herd::default();
            herd.update(&mut scene, 0, 1.0 / 30.0);
            let Some((a, c)) = herd.all.first().zip(scene.creatures.first()) else { continue };
            let Some(open) = (0..8).map(|k| apply(rot_y(k as f32 * TAU / 8.0), [0.0, 0.0, 1.0])).find(|&d| a.clear(c, &scene, d)) else { continue };
            let player = add(c.base, scale(open, near(c) + 0.6 * TF));
            herd.player = Some(player);
            for _ in 0..150 {
                herd.all[0].chase(Some(player));
                herd.update(&mut scene, 0, 1.0 / 30.0);
                let c = &scene.creatures[0];
                let gap = (player[0] - c.base[0]).hypot(player[2] - c.base[2]);
                assert!(gap >= near(c) - 1.0, "seed {seed}: walked {gap} into a body {} wide", near(c));
            }
            let c = &scene.creatures[0];
            closed += ((player[0] - c.base[0]).hypot(player[2] - c.base[2]) < near(c) + 0.3 * TF) as usize;
        }
        assert!(closed > 0, "nothing came up to the player");
    }

    #[test]
    fn strikes_land_once_and_rebound_when_blocked() {
        for strike in Strike::ALL {
            let (mut started, mut landed) = (0, 0);
            each(
                90,
                |a, c| {
                    if a.strikes().contains(&strike) {
                        assert!(a.attack(strike, add(c.base, [0.0, -0.5 * TF, TF])));
                        started += 1;
                    }
                },
                |a, _, _, cue| {
                    if let Some(cue) = cue {
                        assert_eq!(cue, Cue::Strike(strike));
                        landed += 1;
                        assert!(a.busy());
                    }
                },
            );
            assert_eq!(started, landed, "{strike:?}");
            assert!(started > 0 || strike == Strike::Lash || strike == Strike::Stab, "nothing could {strike:?}");
        }
        // Blocked, the strike goes on longer (the stagger) but still ends.
        let mut blocked_some = false;
        for seed in 1..4 {
            let (mut scene, _) = walkable_scene(&mut Rng::new(seed), 30);
            let mut herd = Herd::default();
            herd.update(&mut scene, 0, 1.0 / 30.0);
            for (a, c) in herd.all.iter_mut().zip(&scene.creatures) {
                a.attack(Strike::Lunge, add(c.base, [0.0, 0.0, TF]));
            }
            for _ in 0..120 {
                for (i, _) in herd.update(&mut scene, 0, 1.0 / 30.0) {
                    blocked_some |= herd.all[i].blocked();
                    assert!(!herd.all[i].blocked(), "blocked twice");
                }
            }
            assert!(herd.all.iter().all(|a| !a.busy()), "a blocked strike never ended");
        }
        assert!(blocked_some);
    }

    #[test]
    fn spells_go_off() {
        let (mut cast, mut went) = (0, 0);
        each(
            90,
            |a, c| {
                assert!(a.cast(add(c.base, [0.0, 0.0, 3.0 * TF])));
                cast += 1;
            },
            |_, _, _, cue| {
                if cue == Some(Cue::Spell) {
                    went += 1;
                }
            },
        );
        assert!(cast > 0);
        assert_eq!(cast, went);
    }

    #[test]
    fn fidgets_end() {
        for fidget in Fidget::ALL {
            each(
                150,
                |a, _| a.fidget(fidget),
                |a, _, frame, _| {
                    if frame == 149 {
                        assert!(!matches!(a.act, Some(Act::Fidget { fidget: f, t, .. }) if f == fidget && t > 4.5));
                    }
                },
            );
        }
    }

    #[test]
    fn the_dead_lie_on_the_floor() {
        for death in Death::ALL {
            let mut died = 0;
            let mut floors = Vec::new();
            each(
                180,
                |a, _| {
                    if a.deaths().contains(&death) {
                        a.die(death);
                        died += 1;
                    }
                },
                |a, c, frame, cue| {
                    assert!(cue.is_none());
                    if frame == 179 && a.dead() {
                        assert!(a.settled, "{death:?}: still falling");
                        let soft = matches!(death, Death::Melt | Death::Shrivel | Death::Burst);
                        floors.push((a.floor - lowest(c, soft)).abs());
                    }
                },
            );
            assert!(died > 0, "nothing could die by {death:?}");
            assert!(floors.iter().all(|&g| g < 2.0), "{death:?}: off the floor by {floors:?}");
        }
    }
}
