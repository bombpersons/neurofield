//! The player's hands and what they hold: a sword or a shield in each, small models built in
//! code, their poses in view space, and what each is doing. A sword swings, or drags itself
//! across with no stamina left; a shield goes up while its button is held, shakes when a blow
//! lands on it, and is knocked aside when one breaks the guard, the view kicked back with it.
//! They are drawn by `hud` with a little software rasteriser: flat-lit triangles on whole
//! pixels, sorted back to front with no depth buffer, as the PS1 drew them. Every pose is
//! written for the right hand and mirrored for the left. Nothing here needs Bevy.

use crate::player::{Item, Side, SIDES};
use engine::world::walker::Kick;

/// Focal length in pixels at the frame's 240 height (`engine::world::walker::H`).
const FOCAL: f32 = 180.0;
const NEAR: f32 = 0.08;
const TF: f32 = engine::world::scene::T as f32;

/// A swing: drawing back, striking, holding at the end, and returning.
const SWING: [(f32, Pose); 5] = [
    (0.0, REST),
    (0.13, Pose { pos: [0.50, 0.26, 0.62], rot: [0.15, 0.20, 0.50] }),
    (0.26, Pose { pos: [-0.24, 0.42, 0.75], rot: [-1.20, -0.15, -0.90] }),
    (0.31, Pose { pos: [-0.30, 0.46, 0.75], rot: [-1.30, -0.15, -1.00] }),
    (0.58, REST),
];
/// A swing with no stamina left: hauled back low, dragged across, and let sag.
const LABOUR: [(f32, Pose); 5] = [
    (0.0, REST),
    (0.5, Pose { pos: [0.50, 0.30, 0.66], rot: [0.05, 0.25, 0.70] }),
    (1.05, Pose { pos: [-0.08, 0.62, 0.76], rot: [-0.80, -0.10, -1.25] }),
    (1.3, Pose { pos: [-0.12, 0.72, 0.74], rot: [-0.60, -0.10, -1.45] }),
    (1.85, REST),
];
/// Seconds into each when the blade is halfway across, where it meets what it hits.
const SWING_SWEEP: f32 = 0.2;
const LABOUR_SWEEP: f32 = 0.8;
/// A new swing can start this long before the last has quite finished.
const NEXT_SWING: f32 = 0.08;
/// Held at the right, the blade up, leaning forward and a little inward.
const REST: Pose = Pose { pos: [0.46, 0.46, 0.80], rot: [-0.38, -0.15, -0.26] };

/// The shield low at the side, its face turned inward, and held up before the eye.
const SHIELD_REST: Pose = Pose { pos: [0.44, 0.46, 0.74], rot: [0.30, -0.50, 0.15] };
const SHIELD_UP: Pose = Pose { pos: [0.17, 0.20, 0.55], rot: [0.05, -0.20, 0.05] };
/// How fast it goes up and comes down, in raises a second; past `GUARDED` raised it
/// blocks.
const RAISE: f32 = 7.0;
const LOWER: f32 = 4.0;
const GUARDED: f32 = 0.75;
/// How long a blow on the shield shakes it, and how long a broken guard stuns: the shield
/// flung aside, held there, and brought back.
const KNOCK: f32 = 0.45;
pub const STUN: f32 = 1.1;

/// How fast the hands are put away out of sight and brought back, in whole moves a second.
const STOW: f32 = 1.8;

/// Let go of (dying), what was held falls this fast (sword lengths a second squared),
/// spinning away, and is gone after this many seconds.
const FALL: f32 = 7.0;
const FALLEN: f32 = 0.9;

/// How far the walking bob moves the hands per world unit it moves the camera.
const BOB: f32 = 0.005;
/// Sideways lag per yaw unit a second of turning, its limit, and how fast it follows.
const SWAY: f32 = 2.5e-5;
const SWAY_MOST: f32 = 0.08;
const SWAY_RATE: f32 = 12.0;

/// Where a hand is: `pos` in view space (x right, y down, z ahead, in sword lengths), `rot`
/// radians about x, y and z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub pos: [f32; 3],
    pub rot: [f32; 3],
}

impl Pose {
    fn mix(a: Pose, b: Pose, t: f32) -> Pose {
        let m = |x: f32, y: f32| x + (y - x) * t;
        Pose { pos: [0, 1, 2].map(|i| m(a.pos[i], b.pos[i])), rot: [0, 1, 2].map(|i| m(a.rot[i], b.rot[i])) }
    }

    /// The same pose in the other hand: everything is mirrored left to right, and the
    /// models are their own mirror images.
    fn mirrored(self) -> Pose {
        Pose { pos: [-self.pos[0], self.pos[1], self.pos[2]], rot: [self.rot[0], -self.rot[1], -self.rot[2]] }
    }

    /// A point of the model (y along the blade towards its tip) in view space.
    fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        // Model y is up, view y is down.
        let [x, y, z] = [p[0], -p[1], p[2]];
        let (sx, cx) = self.rot[0].sin_cos();
        let (sy, cy) = self.rot[1].sin_cos();
        let (sz, cz) = self.rot[2].sin_cos();
        let (x, z) = (x * cy + z * sy, -x * sy + z * cy);
        let (y, z) = (y * cx - z * sx, y * sx + z * cx);
        let (x, y) = (x * cz - y * sz, x * sz + y * cz);
        [x + self.pos[0], y + self.pos[1], z + self.pos[2]]
    }

    /// A direction of the model in view space.
    fn turn(&self, d: [f32; 3]) -> [f32; 3] {
        let o = Pose { pos: [0.0; 3], rot: self.rot };
        o.apply(d)
    }
}

/// What a hand holds, as it's drawn and what its button does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Sword,
    Shield,
}

impl Kind {
    pub fn of(item: Item) -> Option<Kind> {
        match item {
            Item::Sword => Some(Kind::Sword),
            _ if item.is_shield() => Some(Kind::Shield),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Act {
    Swing,
    Labour,
    Knock,
    Break,
}

#[derive(Default)]
struct Hand {
    held: Option<Kind>,
    /// What it's doing, and seconds into it.
    act: Option<(Act, f32)>,
    /// How far a shield is up, 0..1.
    raise: f32,
    /// The shield gave out: it stays down until its button is let go.
    spent: bool,
}

impl Hand {
    fn swinging(&self) -> bool {
        matches!(self.act, Some((Act::Swing | Act::Labour, _)))
    }

    fn guarding(&self) -> bool {
        self.held == Some(Kind::Shield) && self.raise >= GUARDED && !matches!(self.act, Some((Act::Break, _)))
    }

    /// The pose for the right hand.
    fn pose(&self) -> Pose {
        let keyed = |keys: &[(f32, Pose)], t: f32| {
            let i = keys.iter().rposition(|k| k.0 <= t).unwrap_or(0).min(keys.len() - 2);
            let (a, b) = (keys[i], keys[i + 1]);
            Pose::mix(a.1, b.1, smooth((t - a.0) / (b.0 - a.0)))
        };
        match (self.held, self.act) {
            (Some(Kind::Sword), Some((Act::Swing, t))) => keyed(&SWING, t),
            (Some(Kind::Sword), Some((Act::Labour, t))) => keyed(&LABOUR, t),
            (Some(Kind::Sword), _) | (None, _) => REST,
            (Some(Kind::Shield), act) => {
                let mut p = Pose::mix(SHIELD_REST, SHIELD_UP, smooth(self.raise));
                match act {
                    Some((Act::Knock, t)) => {
                        let (shove, wobble) = (shove(t), wobble(t, 38.0, 7.0));
                        p.pos[2] -= 0.12 * shove;
                        p.pos[1] -= 0.03 * shove;
                        p.rot[0] += 0.25 * wobble;
                        p.rot[2] += 0.15 * wobble;
                    }
                    Some((Act::Break, t)) => {
                        // Flung aside and down from where the blow caught it, held there,
                        // then let down to the side.
                        let out = smooth(t / 0.18);
                        p.pos[0] += 0.18 * out;
                        p.pos[1] += 0.12 * out;
                        p.pos[2] += 0.06 * out;
                        p.rot[1] -= 0.8 * out;
                        p.rot[2] += 1.3 * out;
                        p.rot[0] += 0.2 * wobble(t, 24.0, 4.0);
                        p = Pose::mix(p, SHIELD_REST, smooth((t - 0.7) / (STUN - 0.7)));
                    }
                    _ => {}
                }
                p
            }
        }
    }
}

fn smooth(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// A sharp push that eases off, 0..1.
fn shove(t: f32) -> f32 {
    (t / 0.05).min(1.0) * (-(t - 0.05).max(0.0) * 9.0).exp()
}

/// A shake at `rate` radians a second dying away at `fade`.
fn wobble(t: f32, rate: f32, fade: f32) -> f32 {
    (t * rate).sin() * (-t * fade).exp()
}

/// Both hands, and the lag of turning they share.
#[derive(Default)]
pub struct Hands {
    hands: [Hand; 2],
    sway: f32,
    last_yaw: Option<f32>,
    /// Weapons put away (somewhere friendly), and how far: 0 in hand, 1 out of sight.
    stow: bool,
    away: f32,
    /// Let go of, and seconds since.
    dropped: Option<f32>,
}

impl Hands {
    /// Let go of everything (dying): it falls away out of sight, and nothing more is done
    /// with it.
    pub fn drop_all(&mut self) {
        if self.dropped.is_none() {
            self.dropped = Some(0.0);
            for h in &mut self.hands {
                h.act = None;
            }
        }
    }

    /// What each hand holds now; a hand given something else starts afresh.
    pub fn hold(&mut self, right: Option<Item>, left: Option<Item>) {
        for (hand, item) in self.hands.iter_mut().zip([right, left]) {
            let kind = item.and_then(Kind::of);
            if hand.held != kind {
                *hand = Hand { held: kind, ..Hand::default() };
            }
        }
    }

    /// Start a swing of the sword in hand `side`, laboured if `tired`: false if there's no
    /// sword there, it's still swinging, or the player is stunned.
    pub fn swing(&mut self, side: Side, tired: bool) -> bool {
        if self.stunned() || self.put_away() || self.dropped.is_some() {
            return false;
        }
        let hand = &mut self.hands[side as usize];
        let free = match hand.act {
            Some((Act::Swing, t)) => t >= SWING[SWING.len() - 1].0 - NEXT_SWING,
            Some((Act::Labour, t)) => t >= LABOUR[LABOUR.len() - 1].0 - NEXT_SWING,
            _ => true,
        };
        if hand.held != Some(Kind::Sword) || !free {
            return false;
        }
        hand.act = Some((if tired { Act::Labour } else { Act::Swing }, 0.0));
        true
    }

    pub fn swinging(&self) -> bool {
        self.hands.iter().any(Hand::swinging)
    }

    /// Whether a shield is up.
    pub fn guarding(&self) -> bool {
        self.hands.iter().any(Hand::guarding)
    }

    /// The hands whose shields are up.
    pub fn guards(&self) -> Vec<Side> {
        SIDES.into_iter().filter(|&s| self.hands[s as usize].guarding()).collect()
    }

    /// Stamina ran out: any shield up comes down until its button is pressed again.
    pub fn give_out(&mut self) {
        for h in self.hands.iter_mut().filter(|h| h.held == Some(Kind::Shield) && h.raise > 0.0) {
            h.spent = true;
        }
    }

    /// A blow landed on the shield.
    pub fn knock(&mut self) {
        for h in self.hands.iter_mut().filter(|h| h.guarding()) {
            h.act = Some((Act::Knock, 0.0));
        }
    }

    /// A blow broke the guard: the shield is knocked aside and the player stunned.
    pub fn break_guard(&mut self) {
        for h in self.hands.iter_mut().filter(|h| h.guarding()) {
            h.act = Some((Act::Break, 0.0));
            h.spent = true;
        }
    }

    /// Put the weapons away, or with false take them out again; true if that's a change.
    pub fn stow(&mut self, on: bool) -> bool {
        let changed = self.stow != on;
        self.stow = on;
        changed
    }

    /// Whether the weapons are put away, or on their way there or back: nothing can be done
    /// with them.
    pub fn put_away(&self) -> bool {
        self.stow || self.away > 0.0
    }

    pub fn stunned(&self) -> bool {
        self.hands.iter().any(|h| matches!(h.act, Some((Act::Break, _))))
    }

    /// How the view is knocked by a blow on the shield.
    pub fn kick(&self) -> Kick {
        let mut k = Kick::default();
        for h in &self.hands {
            match h.act {
                Some((Act::Knock, t)) => {
                    let w = wobble(t, 30.0, 7.0);
                    k.back += 0.1 * TF * shove(t);
                    k.pitch -= 30.0 * w;
                    k.roll += 18.0 * w;
                }
                Some((Act::Break, t)) => {
                    let w = wobble(t, 18.0, 3.5);
                    k.back += 0.16 * TF * shove(t * 0.6);
                    k.yaw += 45.0 * w;
                    k.pitch -= 40.0 * w;
                    k.roll += 70.0 * wobble(t, 11.0, 3.0);
                }
                _ => {}
            }
        }
        k
    }

    /// Move on by `dt` seconds, with the view now turned to `yaw` (4096 units a turn) and
    /// each hand's button `held` or not; gives the hands whose blades have just swept through
    /// what's in front, and whether it was a full swing (not a laboured one).
    pub fn update(&mut self, dt: f32, yaw: f32, held: [bool; 2]) -> Vec<(Side, bool)> {
        let mut swept = Vec::new();
        let stunned = self.stunned();
        self.away = if self.stow { (self.away + STOW * dt).min(1.0) } else { (self.away - STOW * dt).max(0.0) };
        if let Some(t) = &mut self.dropped {
            *t += dt;
        }
        let held = if self.put_away() || self.dropped.is_some() { [false; 2] } else { held };
        for ((hand, held), side) in self.hands.iter_mut().zip(held).zip(SIDES) {
            if !held {
                hand.spent = false;
            }
            if let Some((act, t)) = &mut hand.act {
                let (sweep, length) = match act {
                    Act::Swing => (SWING_SWEEP, SWING[SWING.len() - 1].0),
                    Act::Labour => (LABOUR_SWEEP, LABOUR[LABOUR.len() - 1].0),
                    Act::Knock => (f32::MAX, KNOCK),
                    Act::Break => (f32::MAX, STUN),
                };
                if *t < sweep && *t + dt >= sweep {
                    swept.push((side, *act == Act::Swing));
                }
                *t += dt;
                if *t >= length {
                    // A broken guard ends with the shield down at the side.
                    if *act == Act::Break {
                        hand.raise = 0.0;
                    }
                    hand.act = None;
                }
            }
            if matches!(hand.act, Some((Act::Break, _))) {
                // Where the blow caught it until it's let down.
            } else if hand.held == Some(Kind::Shield) {
                let up = held && !hand.spent && !stunned;
                hand.raise = if up { (hand.raise + RAISE * dt).min(1.0) } else { (hand.raise - LOWER * dt).max(0.0) };
            } else {
                hand.raise = 0.0;
            }
        }
        let rate = self.last_yaw.map_or(0.0, |y| (yaw - y) / dt.max(1e-3));
        self.last_yaw = Some(yaw);
        let want = (rate * SWAY).clamp(-SWAY_MOST, SWAY_MOST);
        self.sway += (want - self.sway) * (dt * SWAY_RATE).min(1.0);
        swept
    }

    /// What each hand holds and its pose now, with the camera's walking bob `bob` (world
    /// units).
    pub fn shown(&self, bob: f32) -> Vec<(Kind, Pose)> {
        if self.away >= 1.0 || self.dropped.is_some_and(|t| t >= FALLEN) {
            return Vec::new();
        }
        let fallen = self.dropped.unwrap_or(0.0);
        // Put away, they drop down out of the bottom of the view, tipping forward.
        let down = smooth(self.away);
        self.hands
            .iter()
            .zip(SIDES)
            .filter_map(|(h, side)| {
                let pose = h.pose();
                let mut pose = if side == Side::Left { pose.mirrored() } else { pose };
                pose.pos[0] += self.sway;
                pose.pos[1] += bob * BOB + 0.9 * down;
                pose.rot[0] += 0.9 * down;
                // Let go of, each drops away down its own side, turning over as it goes.
                let out = if side == Side::Left { -1.0 } else { 1.0 };
                pose.pos[0] += out * 0.35 * fallen;
                pose.pos[1] += 0.5 * FALL * fallen * fallen;
                pose.rot[0] += 2.2 * fallen;
                pose.rot[2] += out * 2.8 * fallen;
                Some((h.held?, pose))
            })
            .collect()
    }
}

const STEEL: [f32; 3] = [175.0, 185.0, 200.0];
const EDGE: [f32; 3] = [225.0, 232.0, 240.0];
const BRASS: [f32; 3] = [165.0, 125.0, 55.0];
const LEATHER: [f32; 3] = [85.0, 50.0, 30.0];
const WOOD: [f32; 3] = [125.0, 85.0, 48.0];
const WOOD_BACK: [f32; 3] = [98.0, 64.0, 38.0];
const IRON: [f32; 3] = [112.0, 112.0, 120.0];

struct ModelTri {
    v: [[f32; 3]; 3],
    colour: [[f32; 3]; 3],
    normal: [f32; 3],
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-6);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// A triangle whose normal points away from `mid`, so the order of corners doesn't matter.
fn tri(v: [[f32; 3]; 3], colour: [[f32; 3]; 3], mid: [f32; 3], out: &mut Vec<ModelTri>) {
    let mut normal = unit(cross(sub(v[1], v[0]), sub(v[2], v[0])));
    let centre = [0, 1, 2].map(|c| (v[0][c] + v[1][c] + v[2][c]) / 3.0);
    if dot(normal, sub(centre, mid)) < 0.0 {
        normal = normal.map(|c| -c);
    }
    out.push(ModelTri { v, colour, normal });
}

/// One closed piece along y, from rings of (y, half width, half depth, colour), its normals
/// pointing away from its middle.
fn piece(rings: &[(f32, f32, f32, [f32; 3])], diamond: bool, out: &mut Vec<ModelTri>) {
    let ring = |&(y, w, d, _): &(f32, f32, f32, [f32; 3])| -> [[f32; 3]; 4] {
        if diamond {
            [[w, y, 0.0], [0.0, y, d], [-w, y, 0.0], [0.0, y, -d]]
        } else {
            [[w, y, d], [-w, y, d], [-w, y, -d], [w, y, -d]]
        }
    };
    let mid = {
        let n = rings.len() as f32;
        [0.0, rings.iter().map(|r| r.0).sum::<f32>() / n, 0.0]
    };
    for w in rings.windows(2) {
        let (a, b) = (ring(&w[0]), ring(&w[1]));
        let (ca, cb) = (w[0].3, w[1].3);
        for i in 0..4 {
            let j = (i + 1) % 4;
            tri([a[i], a[j], b[i]], [ca, ca, cb], mid, out);
            tri([a[j], b[j], b[i]], [ca, cb, cb], mid, out);
        }
    }
    for (r, c) in [(ring(&rings[0]), rings[0].3), (ring(&rings[rings.len() - 1]), rings[rings.len() - 1].3)] {
        tri([r[0], r[1], r[2]], [c; 3], mid, out);
        tri([r[0], r[2], r[3]], [c; 3], mid, out);
    }
}

/// The whole sword, about one unit long, its hand at the origin.
fn sword() -> Vec<ModelTri> {
    let mut out = Vec::new();
    piece(&[(0.03, 0.055, 0.016, STEEL), (0.78, 0.047, 0.013, STEEL), (0.98, 0.002, 0.001, EDGE)], true, &mut out);
    piece(&[(0.0, 0.17, 0.03, BRASS), (0.035, 0.17, 0.03, BRASS)], false, &mut out);
    piece(&[(-0.2, 0.024, 0.024, LEATHER), (0.0, 0.024, 0.024, LEATHER)], false, &mut out);
    piece(&[(-0.26, 0.012, 0.012, BRASS), (-0.23, 0.045, 0.045, BRASS), (-0.2, 0.012, 0.012, BRASS)], true, &mut out);
    out
}

/// A round shield facing ahead (+z), its hand on the grip at the origin behind it: a disc of
/// wood rimmed in iron with a brass boss, the grip a strap of leather across the back.
fn shield() -> Vec<ModelTri> {
    const N: usize = 10;
    const R: f32 = 0.21;
    let (back, front) = (0.03, 0.06);
    let mut out = Vec::new();
    let at = |i: usize, r: f32, z: f32| {
        let a = i as f32 / N as f32 * std::f32::consts::TAU;
        [r * a.cos(), r * a.sin(), z]
    };
    let mid = [0.0, 0.0, (back + front) / 2.0];
    for i in 0..N {
        let j = (i + 1) % N;
        tri([[0.0, 0.0, back], at(i, R, back), at(j, R, back)], [WOOD_BACK; 3], mid, &mut out);
        tri([[0.0, 0.0, front], at(i, R, front), at(j, R, front)], [WOOD; 3], mid, &mut out);
        tri([at(i, R, back), at(j, R, back), at(j, R, front)], [IRON; 3], mid, &mut out);
        tri([at(i, R, back), at(j, R, front), at(i, R, front)], [IRON; 3], mid, &mut out);
    }
    let tip = [0.0, 0.0, front + 0.045];
    for i in (0..N).step_by(2) {
        let j = (i + 2) % N;
        tri([tip, at(i, 0.05, front), at(j, 0.05, front)], [BRASS; 3], [0.0, 0.0, front], &mut out);
    }
    piece(&[(-0.08, 0.02, 0.012, LEATHER), (0.08, 0.02, 0.012, LEATHER)], false, &mut out);
    out
}

fn model(kind: Kind) -> Vec<ModelTri> {
    match kind {
        Kind::Sword => sword(),
        Kind::Shield => shield(),
    }
}

/// A triangle on the frame: whole pixels and a colour at each corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenTri {
    pub p: [(i32, i32); 3],
    pub colour: [[u8; 3]; 3],
}

/// What the hands hold, at their poses, on a frame `width` by `height` pixels, in drawing
/// order.
pub fn triangles(held: &[(Kind, Pose)], width: u32, height: u32) -> Vec<ScreenTri> {
    let scale = height as f32 / 240.0;
    let (cx, cy, f) = (width as f32 / 2.0, height as f32 / 2.0, FOCAL * scale);
    let light = unit([-0.45, -0.75, -0.5]);
    let mut out: Vec<(f32, ScreenTri)> = Vec::new();
    for &(kind, pose) in held {
        for t in model(kind) {
            let v = t.v.map(|p| pose.apply(p));
            let n = pose.turn(t.normal);
            // Facing away from the eye: the far side, hidden by the near one.
            if dot(n, unit(v[0])) >= 0.0 || v.iter().any(|p| p[2] < NEAR) {
                continue;
            }
            let shade = 0.42 + 0.75 * dot(n, light).max(0.0);
            let colour = t.colour.map(|c| c.map(|x| (x * shade).clamp(0.0, 255.0) as u8));
            let p = v.map(|p| ((cx + f * p[0] / p[2]).round() as i32, (cy + f * p[1] / p[2]).round() as i32));
            out.push(((v[0][2] + v[1][2] + v[2][2]) / 3.0, ScreenTri { p, colour }));
        }
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.into_iter().map(|t| t.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn armed(right: Option<Item>, left: Option<Item>) -> Hands {
        let mut h = Hands::default();
        h.hold(right, left);
        h
    }

    fn step(h: &mut Hands, seconds: f32, held: [bool; 2]) -> Vec<(Side, bool)> {
        let mut swept = Vec::new();
        for _ in 0..(seconds / DT).round() as i32 {
            swept.extend(h.update(DT, 0.0, held));
        }
        swept
    }

    fn right(h: &Hands) -> Pose {
        h.shown(0.0)[0].1
    }

    #[test]
    fn put_away_the_weapons_go_out_of_sight_and_do_nothing() {
        let mut h = armed(Some(Item::Sword), Some(Item::Buckler));
        assert!(h.stow(true) && !h.stow(true));
        assert!(!h.swing(Side::Right, false), "no swinging while they're put away");
        step(&mut h, 0.2, [false, true]);
        assert!(!h.guarding() && !h.shown(0.0).is_empty(), "on their way down");
        step(&mut h, 1.0, [false, true]);
        assert!(h.shown(0.0).is_empty() && !h.guarding());
        assert!(h.stow(false));
        step(&mut h, 0.2, [false, false]);
        assert!(!h.shown(0.0).is_empty() && !h.swing(Side::Right, false), "still coming back");
        step(&mut h, 1.0, [false, false]);
        assert!(h.swing(Side::Right, false));
    }

    #[test]
    fn let_go_of_they_fall_away_out_of_sight() {
        let mut h = armed(Some(Item::Sword), Some(Item::Buckler));
        let rest = right(&h);
        h.drop_all();
        assert!(!h.swing(Side::Right, false), "nothing more is done with them");
        step(&mut h, 0.2, [true, true]);
        assert!(!h.guarding());
        let falling = right(&h);
        assert!(falling.pos[1] > rest.pos[1] + 0.1 && falling.pos[0] > rest.pos[0], "{falling:?}");
        step(&mut h, 1.0, [false; 2]);
        assert!(h.shown(0.0).is_empty());
    }

    #[test]
    fn a_swing_leaves_rest_and_comes_back() {
        let mut h = armed(Some(Item::Sword), None);
        assert_eq!(right(&h), REST);
        assert!(h.swing(Side::Right, false));
        assert!(h.swinging());
        assert_eq!(step(&mut h, 0.26, [false; 2]), vec![(Side::Right, true)]);
        assert!(right(&h).pos[0] < REST.pos[0] - 0.4, "{:?}", right(&h));
        step(&mut h, 0.4, [false; 2]);
        assert!(!h.swinging());
        assert_eq!(right(&h), REST);
    }

    #[test]
    fn clicking_mid_swing_does_not_restart_it() {
        let mut h = armed(Some(Item::Sword), None);
        h.swing(Side::Right, false);
        step(&mut h, 0.2, [false; 2]);
        let before = right(&h);
        assert!(!h.swing(Side::Right, false));
        assert_eq!(right(&h), before);
    }

    #[test]
    fn another_swing_can_start_as_one_ends() {
        let mut h = armed(Some(Item::Sword), None);
        h.swing(Side::Right, false);
        step(&mut h, SWING[4].0 - 0.05, [false; 2]);
        assert!(h.swinging());
        assert!(h.swing(Side::Right, false));
        step(&mut h, 0.1, [false; 2]);
        assert!(h.swinging(), "the second swing should be under way");
        assert!((right(&h).pos[0] - REST.pos[0]).abs() < 0.2);
    }

    #[test]
    fn a_tired_swing_is_slow_and_says_so() {
        let mut h = armed(Some(Item::Sword), None);
        h.swing(Side::Right, true);
        assert!(step(&mut h, 0.5, [false; 2]).is_empty(), "a laboured swing lands late");
        assert_eq!(step(&mut h, 0.5, [false; 2]), vec![(Side::Right, false)]);
        assert!(h.swinging());
        step(&mut h, 1.0, [false; 2]);
        assert!(!h.swinging());
    }

    #[test]
    fn a_shield_goes_up_while_held_and_does_not_swing() {
        let mut h = armed(Some(Item::Sword), Some(Item::Buckler));
        assert!(!h.swing(Side::Left, false));
        step(&mut h, 0.3, [false, true]);
        assert!(h.guarding());
        let up = h.shown(0.0)[1].1;
        assert!(up.pos[0] > -0.3 && up.pos[2] < 0.6, "{up:?}");
        step(&mut h, 0.5, [false, false]);
        assert!(!h.guarding());
    }

    #[test]
    fn a_spent_shield_stays_down_until_pressed_again() {
        let mut h = armed(None, Some(Item::Buckler));
        step(&mut h, 0.3, [false, true]);
        h.give_out();
        step(&mut h, 0.5, [false, true]);
        assert!(!h.guarding());
        step(&mut h, 0.1, [false, false]);
        step(&mut h, 0.3, [false, true]);
        assert!(h.guarding());
    }

    #[test]
    fn a_blow_on_the_shield_kicks_the_view_and_a_broken_guard_stuns() {
        let mut h = armed(Some(Item::Buckler), Some(Item::Sword));
        step(&mut h, 0.3, [true, false]);
        assert!(h.guarding());
        h.knock();
        step(&mut h, 0.06, [true, false]);
        assert!(h.kick().back > 0.0);
        assert!(h.guarding(), "a knock doesn't lower the shield");
        step(&mut h, KNOCK, [true, false]);
        assert_eq!(h.kick(), Kick::default());
        h.break_guard();
        assert!(h.stunned() && !h.guarding());
        assert!(!h.swing(Side::Left, false), "no swinging while stunned");
        step(&mut h, 0.2, [true, false]);
        assert!(h.kick() != Kick::default());
        step(&mut h, STUN, [true, false]);
        assert!(!h.stunned());
        assert!(!h.guarding(), "the button has to be pressed again");
    }

    #[test]
    fn the_sword_is_on_the_right_at_rest_and_mirrored_in_the_left() {
        for (h, right) in [(armed(Some(Item::Sword), None), true), (armed(None, Some(Item::Sword)), false)] {
            let t = triangles(&h.shown(0.0), 320, 240);
            assert!(!t.is_empty());
            let on_screen: Vec<_> = t.iter().flat_map(|t| t.p).filter(|p| (0..240).contains(&p.1)).collect();
            assert!(on_screen.iter().all(|p| (p.0 > 160) == right), "{on_screen:?}");
        }
    }

    #[test]
    fn they_stay_in_front_of_the_eye_whatever_they_do() {
        let near = |h: &Hands| {
            for (kind, p) in h.shown(0.0) {
                assert!(model(kind).iter().flat_map(|t| t.v).all(|v| p.apply(v)[2] > NEAR), "{kind:?} {p:?}");
            }
        };
        for tired in [false, true] {
            let mut h = armed(Some(Item::Sword), Some(Item::Sword));
            h.swing(Side::Right, tired);
            h.swing(Side::Left, tired);
            while h.swinging() {
                near(&h);
                h.update(DT, 0.0, [false; 2]);
            }
        }
        let mut h = armed(Some(Item::Buckler), Some(Item::Buckler));
        for _ in 0..30 {
            near(&h);
            h.update(DT, 0.0, [true; 2]);
        }
        h.break_guard();
        while h.stunned() {
            near(&h);
            h.update(DT, 0.0, [true; 2]);
        }
    }

    #[test]
    fn a_raised_shield_covers_the_middle_of_the_view() {
        let mut h = armed(None, Some(Item::Buckler));
        step(&mut h, 0.4, [false, true]);
        let t = triangles(&h.shown(0.0), 320, 240);
        let xs: Vec<i32> = t.iter().flat_map(|t| t.p).map(|p| p.0).collect();
        let (lo, hi) = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap());
        assert!(lo < 120 && hi > 150, "{lo}..{hi}");
    }

    #[test]
    fn turning_makes_them_lag_and_settle() {
        let mut h = armed(Some(Item::Sword), None);
        h.update(DT, 0.0, [false; 2]);
        for i in 1..30 {
            h.update(DT, i as f32 * 20.0, [false; 2]);
        }
        assert!(right(&h).pos[0] != REST.pos[0]);
        for _ in 0..120 {
            h.update(DT, 580.0, [false; 2]);
        }
        assert!((right(&h).pos[0] - REST.pos[0]).abs() < 1e-3);
    }

    #[test]
    fn it_scales_with_the_frame() {
        let shown = [(Kind::Sword, REST)];
        let (a, b) = (triangles(&shown, 320, 240), triangles(&shown, 640, 480));
        assert_eq!(a.len(), b.len());
        let (pa, pb) = (a[0].p[0], b[0].p[0]);
        assert!((pb.0 - pa.0 * 2).abs() <= 2 && (pb.1 - pa.1 * 2).abs() <= 2, "{pa:?} {pb:?}");
    }

    #[test]
    fn normals_are_whole() {
        for kind in [Kind::Sword, Kind::Shield] {
            for t in model(kind) {
                assert!(dot(t.normal, t.normal) > 0.9);
            }
        }
    }
}
