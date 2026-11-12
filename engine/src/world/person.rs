//! People: townsfolk and the like, always drawn as plain PS1 models whatever renders the
//! level around them. A person's body is a creature's tree of parts (`creature::Part`) built
//! to a fixed human plan from their `Look`, so the creature animator walks it; they are kept
//! apart from the creatures (`Scene::people`), so nothing grows on them, nothing fights them
//! and the dream net never draws them.
//!
//! The body is bare for now (clothes come later, hung on its `Bones`). The head is meshed on
//! its own, a painted face (`face`) over its front, which is redrawn as they blink, look about
//! and talk; their hair (`hair`) is made from its dials and worn on the head.

use crate::world::animate::{Animator, Fidget};
use crate::world::creature::{scale, sub, Creature, Part, Placed, Shape, Skin};
use crate::world::clothes::{self, Bone, Cover, Wear};
use crate::world::face::{self, Features, Mouth};
use crate::world::hair::{self, Hair, HairTri};
use crate::world::rng::Rng;
use crate::world::scene::{apply, dot, mul, norm, rot_x, rot_y, rot_z, Scene, Tri, M3, T};
use crate::world::skin::Motif;
use crate::world::tex::{self, Texture, SIZE};
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

const TF: f32 = T as f32;
/// How near the player's body can come to a person's middle (x/z).
const GIRTH: f32 = 0.16 * TF;
/// Within this a person stops and turns to face the player.
const FACE_PLAYER: f32 = 1.5 * TF;
/// Within this, and in front of them, a person watches the player.
const WATCH: f32 = 3.5 * TF;
/// Furthest a wanderer strays from where they were put, and a guard.
const WANDER_LEASH: f32 = 3.0 * TF;
const GUARD_LEASH: f32 = 0.6 * TF;
/// How far the head turns (radians) and how fast.
const TURN_MOST: f32 = 1.1;
const NOD_MOST: f32 = 0.45;
const TURN_RATE: f32 = 3.5;
/// Seconds between blinks, and a blink's length.
const BLINKS: (f32, f32) = (1.8, 5.5);
const BLINK_SECS: f32 = 0.16;
/// Seconds each mouth shape lasts while talking.
const SYLLABLE: (f32, f32) = (0.08, 0.17);
/// The body's parts' sides and rows: a little finer than a creature's, so limbs are round
/// rather than diamonds.
const BODY_SIDES: (usize, usize) = (8, 6);
/// The head's sides and rows, finer than the low-poly body's for the face.
const HEAD_AROUND: usize = 12;
const HEAD_ALONG: usize = 9;
/// Head triangles this far forward (as a share of its depth) wear the face.
const FACE_FROM: f32 = 0.2;

/// What someone looks like.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Look {
    /// Head to foot, in tiles (the player's eyes are at 0.62).
    pub height: f32,
    /// 0 slight .. 1 heavy.
    pub build: f32,
    /// 0 anime proportions .. 1 a chibi's (a big head on a small body).
    #[serde(default)]
    pub chibi: f32,
    /// The skin's colour.
    pub skin: [f32; 3],
    pub face: Features,
    pub hair: Hair,
    /// What they wear, over the bare body.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clothes: Vec<Wear>,
}

impl Look {
    pub fn random(seed: u64) -> Look {
        let mut rng = Rng::new(seed ^ 0x09E0_F1E5);
        let skin = rng.pick(&[
            [1.0, 0.88, 0.8],
            [0.97, 0.8, 0.7],
            [0.9, 0.7, 0.57],
            [0.76, 0.55, 0.42],
            [0.58, 0.4, 0.3],
            [0.44, 0.3, 0.22],
        ]);
        let hair = Hair::random(&mut rng);
        let face = Features::random(&mut rng, hair.colour);
        Look { height: rng.fr(0.62, 0.74), build: rng.f(), chibi: 0.0, skin, face, hair, clothes: clothes::outfit(&mut rng) }
    }
}

/// What a person does with themselves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// Strolls about near where they were put.
    #[default]
    Wander,
    /// Stays put, facing the way they were put (a stallholder).
    Stand,
    /// Holds a post, hardly leaving it.
    Guard,
}

/// Which part is which, for clothes and gear to hang on; each pair is [+x side, -x side] of
/// the body's own frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bones {
    pub pelvis: usize,
    pub torso: usize,
    pub neck: usize,
    pub head: usize,
    pub ears: [usize; 2],
    pub upper_arms: [usize; 2],
    pub forearms: [usize; 2],
    pub hands: [usize; 2],
    pub thighs: [usize; 2],
    pub shins: [usize; 2],
    pub feet: [usize; 2],
}

impl Bones {
    /// The parts that are `bone`.
    pub fn of(&self, bone: Bone) -> Vec<usize> {
        match bone {
            Bone::Pelvis => vec![self.pelvis],
            Bone::Torso => vec![self.torso],
            Bone::Neck => vec![self.neck],
            Bone::Head => vec![self.head],
            Bone::UpperArm => self.upper_arms.to_vec(),
            Bone::Forearm => self.forearms.to_vec(),
            Bone::Hand => self.hands.to_vec(),
            Bone::Thigh => self.thighs.to_vec(),
            Bone::Shin => self.shins.to_vec(),
            Bone::Foot => self.feet.to_vec(),
        }
    }
}

/// Someone in a scene.
pub struct Person {
    pub look: Look,
    pub role: Role,
    /// What they're called, if the player is to know, and what they say when spoken to.
    pub name: Option<String>,
    pub lines: Vec<String>,
    /// What made them up.
    pub seed: u64,
    pub body: Creature,
    pub bones: Bones,
    /// The hair's mesh, in the head's frame (see `hair`).
    pub hair: Vec<HairTri>,
    /// Whether they are talking (the mouth moves).
    pub talking: bool,
    pub face: face::State,
    /// The face's texture as it is now, and a count bumped each time it is redrawn.
    pub face_tex: Texture,
    pub face_version: u32,
    /// Where the eyes are looking (the world), if anywhere in particular.
    pub watching: Option<[f32; 3]>,
    /// The head's turn and nod now (radians).
    turn: [f32; 2],
    /// Seconds to the next blink (negative while blinking), and to the next syllable.
    blink: f32,
    syllable: f32,
    rng: Rng,
}

impl Person {
    /// `look` standing on the floor at `base` facing `yaw`; `seed` for what they do.
    pub fn new(look: Look, role: Role, base: [f32; 3], yaw: f32, seed: u64) -> Person {
        let (body, bones) = body(&look, base, yaw);
        let state = face::State::default();
        let mut rng = Rng::new(seed ^ 0x9E5);
        Person {
            hair: hair::mesh(&look.hair, seed),
            face_tex: face::compose(&look.face, look.skin, &state),
            face_version: 0,
            face: state,
            talking: false,
            watching: None,
            turn: [0.0; 2],
            blink: rng.fr(BLINKS.0, BLINKS.1),
            syllable: 0.0,
            rng,
            look,
            role,
            name: None,
            lines: Vec::new(),
            seed,
            body,
            bones,
        }
    }

    /// Where the eyes are, between them.
    pub fn eyes(&self) -> [f32; 3] {
        let placed = self.body.pose();
        let at = &placed[self.bones.head];
        let half = self.body.parts[self.bones.head].half;
        at.world([0.0, (1.0 - 2.0 * face::EYE_ROW / SIZE as f32) * half[1], 0.9 * half[2]])
    }

    /// The face as time passes: blinking, the mouth moving while they talk, the head and
    /// eyes turned toward `watching`. Call after the body has been posed for the frame.
    fn live(&mut self, dt: f32) {
        // Blinks.
        self.blink -= dt;
        if self.blink < -BLINK_SECS {
            self.blink = self.rng.fr(BLINKS.0, BLINKS.1);
        }
        let blink = if self.blink < 0.0 { 1.0 - ((-self.blink / BLINK_SECS) * 2.0 - 1.0).abs() } else { 0.0 };
        // Talking.
        let mouth = if self.talking {
            self.syllable -= dt;
            if self.syllable <= 0.0 {
                self.syllable = self.rng.fr(SYLLABLE.0, SYLLABLE.1);
                self.face.mouth = self.rng.pick(&[Mouth::Ajar, Mouth::Open, Mouth::Oo, Mouth::Closed, Mouth::Ajar]);
            }
            self.face.mouth
        } else {
            Mouth::Closed
        };
        // The head turns toward what they watch, as far as it goes; the eyes do the rest.
        let head = self.bones.head;
        let rest = self.body.pose()[head];
        let want = match self.watching {
            Some(t) => {
                let d = apply(transpose(rest.axes), sub(t, rest.centre));
                let flat = d[0].hypot(d[2]).max(1e-3);
                [(-d[0]).atan2(d[2]).clamp(-TURN_MOST, TURN_MOST), (-d[1]).atan2(flat).clamp(-NOD_MOST, NOD_MOST)]
            }
            None => [0.0; 2],
        };
        for (turn, want) in self.turn.iter_mut().zip(want) {
            *turn += (want - *turn).clamp(-TURN_RATE * dt, TURN_RATE * dt);
        }
        let p = &mut self.body.parts[head];
        p.rot = mul(p.rot, mul(rot_y(self.turn[0]), rot_x(self.turn[1])));
        let gaze = match self.watching {
            Some(t) => {
                let at = self.body.pose()[head];
                let d = apply(transpose(at.axes), sub(t, at.centre));
                let flat = d[0].hypot(d[2]).max(1e-3);
                [(d[0].atan2(d[2]) / 0.6).clamp(-1.0, 1.0), (d[1].atan2(flat) / 0.5).clamp(-1.0, 1.0)]
            }
            None => [0.0; 2],
        };
        // Steps the texture can show: no need to redraw it for less.
        let state = face::State {
            blink: (blink * 2.0).round() / 2.0,
            gaze: gaze.map(|g| (g * 4.0).round() / 4.0),
            mouth,
            brow: if self.talking && mouth == Mouth::Open { 0.5 } else { 0.0 },
        };
        if state != self.face {
            self.face = state;
            self.face_tex = face::compose(&self.look.face, self.look.skin, &state);
            self.face_version += 1;
        }
    }
}

fn transpose(m: M3) -> M3 {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| m[c][r]))
}

/// A bare human body to `look`'s height and build, its feet on the floor at `base`.
pub fn body(look: &Look, base: [f32; 3], yaw: f32) -> (Creature, Bones) {
    let h = look.height * TF;
    let w = 0.85 + 0.45 * look.build.clamp(0.0, 1.0);
    let skin = Skin { motif: Motif::Flesh, density: 0.0, scale: 64.0, tint: look.skin, seed: 0 };
    let mut c = Creature { base, yaw, skins: vec![skin], parts: Vec::new(), slots: vec![tex::PERSON_SKIN], girth: GIRTH };
    let mut add = |p: Part| {
        c.parts.push(p);
        c.parts.len() - 1
    };
    // Each size with anime proportions (k = 0: about six heads tall, slender, long in the
    // leg, the head a little large) and for a chibi (k = 1: about two and a half heads tall,
    // a big round head, short limbs, mitten hands and round feet).
    let k = look.chibi.clamp(0.0, 1.0);
    let m = |a: f32, b: f32| (a + (b - a) * k) * h;
    let pelvis = add(Part::new(Shape::Ellipsoid, [m(0.072, 0.075) * w, m(0.05, 0.05), m(0.052, 0.06) * w], 0).root([0.0, -m(0.53, 0.33), 0.0]).pivot([0.0; 3]));
    let (mut thighs, mut shins, mut feet) = ([0; 2], [0; 2], [0; 2]);
    for (i, (around, side)) in [(0.0, 1.0), (0.5, -1.0)].into_iter().enumerate() {
        thighs[i] = add(Part::new(Shape::Capsule, [m(0.036, 0.036) * w, m(0.125, 0.07), m(0.036, 0.036) * w], 0).on(pelvis, 0.3, around, 0.5).rot(rot_z(-side * 0.03)).taper(0.8));
        shins[i] = add(Part::new(Shape::Capsule, [m(0.027, 0.031) * w, m(0.13, 0.075), m(0.027, 0.031) * w], 0).on(thighs[i], 1.0, 0.0, 0.0).rot(rot_x(-0.04)).taper(0.75));
        feet[i] = add(Part::new(Shape::Ellipsoid, [m(0.022, 0.036), m(0.016, 0.026), m(0.048, 0.058)], 0).on(shins[i], 1.0, 0.0, 0.0).rot(rot_x(0.04)).pivot([0.0, -0.5, -0.5]));
    }
    // From inside the hips, narrow at the waist and broad at the shoulders (a chibi's is
    // more of a bean).
    let torso = Part::new(Shape::Capsule, [m(0.064, 0.075) * w, m(0.14, 0.1), m(0.047, 0.06) * w], 0)
        .on(pelvis, -0.2, 0.0, 0.0)
        .rot(rot_z(PI))
        .taper(1.3 - 0.25 * k)
        .bulge(0.06 * look.build + 0.12 * k);
    let torso = add(torso);
    let neck = add(Part::new(Shape::Capsule, [m(0.019, 0.024), m(0.035, 0.02), m(0.019, 0.024)], 0).on(torso, 0.88 + 0.04 * k, 0.0, 0.0));
    let head = add(Part::new(Shape::Ellipsoid, [m(0.078, 0.15), m(0.092, 0.15), m(0.083, 0.14)], 0).on(neck, 0.8, 0.0, 0.0).pivot([0.0, -0.75 - 0.1 * k, -0.08 + 0.03 * k]));
    let mut ears = [0; 2];
    for (i, around) in [0.0, 0.5].into_iter().enumerate() {
        ears[i] = add(Part::new(Shape::Ellipsoid, [m(0.008, 0.012), m(0.017, 0.024), m(0.011, 0.016)], 0).on(head, -0.1 * k, around, 0.95).pivot([0.0; 3]));
    }
    let (mut upper, mut fore, mut hands) = ([0; 2], [0; 2], [0; 2]);
    for (i, (around, side)) in [(0.0, 1.0), (0.5, -1.0)].into_iter().enumerate() {
        // The shoulder's joint stands out past the torso's side, so the arm hangs clear of it.
        let arm = Part::new(Shape::Capsule, [m(0.022, 0.026) * w, m(0.09, 0.055), m(0.022, 0.026) * w], 0)
            .on(torso, 0.74 - 0.08 * k, around, 1.3 - 0.1 * k)
            .rot(rot_z(PI + side * (0.1 + 0.1 * k)))
            .taper(0.85);
        upper[i] = add(arm);
        fore[i] = add(Part::new(Shape::Capsule, [m(0.019, 0.024) * w, m(0.085, 0.05), m(0.019, 0.024) * w], 0).on(upper[i], 1.0, 0.0, 0.0).rot(rot_x(0.12)).taper(0.85));
        hands[i] = add(Part::new(Shape::Ellipsoid, [m(0.017, 0.03), m(0.03, 0.03), m(0.01, 0.026)], 0).on(fore[i], 1.0, 0.0, 0.0).pivot([0.0, -0.8, 0.0]));
    }
    // To the height asked for, standing on the floor.
    let (_, top, bottom) = c.extent();
    c.rescale(h / (bottom - top).max(1.0));
    let floor = base[1];
    c.fit(floor, floor - 4.0 * TF, 4.0 * TF, true);
    let bones = Bones { pelvis, torso, neck, head, ears, upper_arms: upper, forearms: fore, hands, thighs, shins, feet };
    (c, bones)
}

/// The body's skin, to be tinted with the skin's colour: near white, faintly mottled, its
/// mean the face's `SKIN_SHADE` so the face meets it without a seam.
pub fn skin_texture() -> Texture {
    let mut pal = [[0u8; 3]; 16];
    for (i, c) in pal.iter_mut().enumerate() {
        let v = (255.0 * face::SKIN_SHADE * (0.94 + 0.12 * i as f32 / 15.0)).min(255.0) as u8;
        *c = [v, v, v];
    }
    let idx = (0..SIZE * SIZE).map(|i| (tex::value_noise((i % SIZE) as f32 / 8.0, (i / SIZE) as f32 / 8.0, 8, 23) * 15.0).round() as u8).collect();
    Texture { idx, pal }
}

/// Every person's animator, remade when the scene's `version` changes (as `animate::Herd`).
#[derive(Default)]
pub struct People {
    version: Option<u64>,
    all: Vec<Animator>,
}

impl People {
    /// Moves everyone on by `dt`. `player` is the player's feet and `eye` their eyes.
    pub fn update(&mut self, scene: &mut Scene, version: u64, player: Option<([f32; 3], [f32; 3])>, dt: f32) {
        if self.version != Some(version) {
            self.all = scene.people.iter().map(animator).collect();
            self.version = Some(version);
        }
        let mut people = std::mem::take(&mut scene.people);
        for (a, p) in self.all.iter_mut().zip(&mut people) {
            let feet = player.map(|(f, _)| f);
            a.player = feet;
            let base = p.body.base;
            let near = |t: [f32; 3]| (t[0] - base[0]).hypot(t[2] - base[2]);
            a.attend = feet.filter(|&f| near(f) < FACE_PLAYER && (f[1] - base[1]).abs() < TF);
            a.update(&mut p.body, scene, dt);
            p.watching = player.and_then(|(f, eye)| {
                let ahead = apply(rot_y(p.body.yaw), [0.0, 0.0, 1.0]);
                let to = sub(eye, p.body.base);
                let d = near(f);
                (d < WATCH && dot(norm([to[0], 0.0, to[2]]), ahead) > -0.2).then_some(eye)
            });
            p.live(dt);
        }
        scene.people = people;
    }
}

fn animator(p: &Person) -> Animator {
    let mut a = Animator::new(&p.body);
    a.calm(&[Fidget::Look]);
    match p.role {
        Role::Wander => a.leash = WANDER_LEASH,
        Role::Stand => a.stay = true,
        Role::Guard => {
            a.stay = true;
            a.leash = GUARD_LEASH;
        }
    }
    a
}

impl Scene {
    /// Add someone to the scene.
    pub fn add_person(&mut self, look: Look, role: Role, base: [f32; 3], yaw: f32, seed: u64) {
        let id = self.people.len();
        assert!(id < (255 - tex::FACE_BASE) as usize, "too many people");
        self.people.push(Person::new(look, role, base, yaw, seed));
    }

    /// The texture id of person `i`'s face.
    pub fn face_tex_id(i: usize) -> u8 {
        tex::FACE_BASE + i as u8
    }

    /// The people `wanted` as they stand now, as PS1 models: the body's parts as the
    /// low-poly creatures' (`tex::PERSON_SKIN`), the head finer with the face over its front
    /// (`face_tex_id`), and the hair (`tex::PERSON_HAIR`).
    pub fn people_tris(&self, wanted: impl Fn(&Person) -> bool) -> Vec<Tri> {
        let mut out = Vec::new();
        for (i, person) in self.people.iter().enumerate() {
            if !wanted(person) {
                continue;
            }
            let c = &person.body;
            let skin = &c.skins[0];
            let placed = c.pose();
            // The clothes, inside out; what they cover end to end isn't drawn, and a covered
            // head keeps only the hair that hangs below the cover's rim.
            let mut hidden = vec![false; c.parts.len()];
            let mut rim = f32::MAX;
            let mut worn: Vec<_> = person.look.clothes.iter().map(|w| (w, w.kind.garment())).collect();
            worn.sort_by_key(|(_, g)| g.layer);
            for (wear, g) in &worn {
                let tint = wear.colour.unwrap_or(g.colour);
                for cover in &g.covers {
                    for k in person.bones.of(cover.bone) {
                        let puff = cover.puff + 0.06 * g.layer as f32;
                        self.cover_tris(&c.parts[k], &placed[k], c.base[1], cover, puff, g.cloth.tex(), tint, &mut out);
                        hidden[k] |= cover.from <= -1.0 && cover.to >= 1.0 && cover.flare == 0.0 && k != person.bones.head;
                        if k == person.bones.head {
                            rim = rim.min(cover.from);
                        }
                    }
                }
            }
            for (k, (p, at)) in c.parts.iter().zip(&placed).enumerate() {
                if k == person.bones.head {
                    self.head_tris(p, at, skin, Scene::face_tex_id(i), &mut out);
                } else if !hidden[k] {
                    self.low_poly_part(p, at, skin, tex::PERSON_SKIN, BODY_SIDES, &mut out);
                }
            }
            let at = &placed[person.bones.head];
            let half = c.parts[person.bones.head].half;
            let tint = person.look.hair.colour;
            let alb = tint.map(|c| (c * 128.0).clamp(0.0, 255.0) as u8);
            for t in person.hair.iter().filter(|t| t.p.iter().all(|p| p[1] < rim)) {
                let pos = t.p.map(|q| at.world([q[0] * half[0], q[1] * half[1], q[2] * half[2]]));
                let nrm = t.n.map(|n| norm(apply(at.axes, [n[0] / half[0], n[1] / half[1], n[2] / half[2]])));
                let uv = |k: usize| t.uv[k].map(|c| (c.clamp(0.0, 1.0) * 63.0).round() as u8);
                out.extend(self.creature_tri_of([(0, uv(0)), (1, uv(1)), (2, uv(2))], &pos, &nrm, None, tex::PERSON_HAIR, tint, alb));
            }
        }
        out
    }

    /// A garment's shell over one part: rings round its axis from `cover.from` to `cover.to`,
    /// `puff` off its surface and flaring toward the far end, never down through the `floor`.
    #[allow(clippy::too_many_arguments)]
    fn cover_tris(&self, p: &Part, at: &Placed, floor: f32, cover: &Cover, puff: f32, tex: u8, tint: [f32; 3], out: &mut Vec<Tri>) {
        const SIDES: usize = 8;
        let widest = p.widest();
        // A cover running on past the part (a skirt) keeps to above the floor.
        let down = at.axes[1][1];
        let to = if cover.to > 1.0 && down > 0.3 { cover.to.min((floor - 3.0 - at.centre[1]) / (p.half[1] * down)) } else { cover.to };
        // Rings where the body's own rows are (so the shell stays outside it), at the ends,
        // and every half-length past the part's end.
        let mut ts = vec![cover.from];
        let body = (0..=BODY_SIDES.1).map(|k| k as f32 / BODY_SIDES.1 as f32 * 2.0 - 1.0);
        ts.extend(body.chain((1..20).map(|k| 1.0 + k as f32 * 0.5)).filter(|&t| t > cover.from + 0.05 && t < to - 0.05));
        ts.push(to);
        let rows = ts.len() - 1;
        let ring = |t: f32| {
            let past = cover.to > 1.0 && t > 0.0;
            let r = if past { widest } else { p.profile(t.clamp(-1.0, 1.0)) };
            r * (1.0 + puff) + cover.flare * widest * ((t - cover.from) / (to - cover.from)).max(0.0)
        };
        let (mut pos, mut nrm) = (Vec::with_capacity((rows + 1) * (SIDES + 1)), Vec::with_capacity((rows + 1) * (SIDES + 1)));
        for &t in &ts {
            let r = ring(t);
            for j in 0..=SIDES {
                let a = j as f32 / SIDES as f32 * std::f32::consts::TAU;
                pos.push(at.world([r * p.half[0] * a.cos(), t * p.half[1], r * p.half[2] * a.sin()]));
                nrm.push(norm(apply(at.axes, [a.cos() / p.half[0], 0.0, a.sin() / p.half[2]])));
            }
        }
        let cols: Vec<[u8; 3]> = pos.iter().zip(&nrm).map(|(q, &n)| self.shade(q.map(|c| c.round() as i32), n, tint)).collect();
        let alb = tint.map(|c| (c * 128.0).clamp(0.0, 255.0) as u8);
        // The cloth twice round, once down.
        const HALF: usize = SIDES / 2;
        for k in 0..rows {
            for j in 0..SIDES {
                let v = |kk: usize, jj: usize| (kk * (SIDES + 1) + jj, [(63 * (jj - j / HALF * HALF) / HALF) as u8, (63 * kk / rows) as u8]);
                let q = [v(k, j), v(k, j + 1), v(k + 1, j), v(k + 1, j + 1)];
                for [a, b, c] in [[0, 1, 2], [1, 3, 2]] {
                    out.extend(self.creature_tri_of([q[a], q[b], q[c]], &pos, &nrm, Some(&cols), tex, tint, alb));
                }
            }
        }
    }

    /// A head, `HEAD_AROUND` by `HEAD_ALONG`: its front wears the face (`face_tex`) laid flat
    /// across it, the rest the body's skin.
    fn head_tris(&self, p: &Part, at: &Placed, skin: &Skin, face_tex: u8, out: &mut Vec<Tri>) {
        let (na, nl) = (HEAD_AROUND, HEAD_ALONG);
        let mut local = Vec::with_capacity((nl + 1) * (na + 1));
        let mut pos = Vec::with_capacity(local.capacity());
        let mut nrm = Vec::with_capacity(local.capacity());
        for k in 0..=nl {
            let t = k as f32 / nl as f32 * 2.0 - 1.0;
            let mid = at.world(p.axis(t));
            for j in 0..=na {
                let l = p.local(t, j as f32 / na as f32 * TAU, 1.0);
                let q = at.world(l);
                let out_of = sub(q, mid);
                let yaxis = [at.axes[0][1], at.axes[1][1], at.axes[2][1]];
                nrm.push(if dot(out_of, out_of) > 1.0 { norm(out_of) } else { scale(yaxis, t.signum()) });
                pos.push(q);
                local.push(l);
            }
        }
        let face_alb = [128u8; 3];
        let alb = skin.tint.map(|c| (c * 128.0).clamp(0.0, 255.0) as u8);
        // Each vertex lit once: under the skin's colour, and plain for the face (whose
        // texture has the skin's colour in it).
        let lit = |tint: [f32; 3]| -> Vec<[u8; 3]> { pos.iter().zip(&nrm).map(|(q, &n)| self.shade(q.map(|c: f32| c.round() as i32), n, tint)).collect() };
        let (skin_cols, face_cols) = (lit(skin.tint), lit([1.0; 3]));
        let grid = |i: usize, n: usize| (63 * i / n) as u8;
        for k in 0..nl {
            for j in 0..na {
                let ids = [k * (na + 1) + j, k * (na + 1) + j + 1, (k + 1) * (na + 1) + j, (k + 1) * (na + 1) + j + 1];
                for [a, b, c] in [[0, 1, 2], [1, 3, 2]] {
                    let tri = [ids[a], ids[b], ids[c]];
                    let front = tri.iter().all(|&v| local[v][2] > FACE_FROM * p.half[2]);
                    let v = |n: usize| {
                        let l = local[tri[n]];
                        let uv = if front {
                            // Across toward +x, down from the crown.
                            let u = (l[0] / p.half[0] + 1.0) / 2.0;
                            let v = (1.0 - l[1] / p.half[1]) / 2.0;
                            [u, v].map(|c| (c.clamp(0.0, 1.0) * 63.0).round() as u8)
                        } else {
                            let (kk, jj) = (tri[n] / (na + 1), tri[n] % (na + 1));
                            [grid(jj, na), grid(kk, nl)]
                        };
                        (tri[n], uv)
                    };
                    let t = if front {
                        self.creature_tri_of([v(0), v(1), v(2)], &pos, &nrm, Some(&face_cols), face_tex, [1.0; 3], face_alb)
                    } else {
                        self.creature_tri_of([v(0), v(1), v(2)], &pos, &nrm, Some(&skin_cols), tex::PERSON_SKIN, skin.tint, alb)
                    };
                    out.extend(t);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::scene::Scene;

    fn open_floor() -> Scene {
        let mut l = crate::world::level::Level::new("t");
        for x in 0..10 {
            for z in 0..10 {
                l.columns.insert((x, z), vec![crate::world::level::Run { from: 0, to: 12, slope: None, sky: true }]);
            }
        }
        l.plain = Some(Default::default());
        l.to_scene()
    }

    #[test]
    fn bodies_stand_on_the_floor_and_walk_on_two_legs() {
        for seed in 0..40 {
            let look = Look::random(seed);
            let (c, b) = body(&look, [5.0 * TF, 0.0, 5.0 * TF], 0.3);
            let (_, top, bottom) = c.extent();
            assert!(bottom.abs() < 2.0, "seed {seed}: feet at {bottom}");
            let tall = (bottom - top) / TF;
            assert!((tall - look.height).abs() < 0.1 * look.height, "seed {seed}: {tall} tall for {}", look.height);
            // The eyes are near the top, the head above the shoulders.
            let placed = c.pose();
            assert!(placed[b.head].centre[1] < placed[b.torso].centre[1] - 0.2 * look.height * TF);
            let a = Animator::new(&c);
            assert_eq!(a.plan, crate::world::animate::Plan::Walk, "seed {seed}");
            assert_eq!(a.legs(), 2, "seed {seed}");
        }
    }

    #[test]
    fn people_walk_blink_and_watch_the_player() {
        let mut scene = open_floor();
        for (i, role) in [Role::Wander, Role::Stand, Role::Guard].into_iter().enumerate() {
            scene.add_person(Look::random(i as u64), role, [(2.5 + 2.0 * i as f32) * TF, 0.0, 5.0 * TF], 0.0, i as u64);
        }
        let starts: Vec<_> = scene.people.iter().map(|p| p.body.base).collect();
        let mut people = People::default();
        let (mut blinked, mut redrawn) = (false, 0);
        for _ in 0..600 {
            people.update(&mut scene, 0, None, 1.0 / 30.0);
            blinked |= scene.people.iter().any(|p| p.face.blink > 0.0);
            redrawn = scene.people[0].face_version;
            for p in &scene.people {
                assert!(p.body.pose().iter().all(|q| q.centre.iter().all(|v| v.is_finite())));
            }
        }
        assert!(blinked && redrawn > 0);
        let moved = |i: usize| (scene.people[i].body.base[0] - starts[i][0]).hypot(scene.people[i].body.base[2] - starts[i][2]);
        assert!(moved(0) > 0.3 * TF, "the wanderer went {}", moved(0));
        assert!(moved(1) < 0.1 * TF, "the stallholder went {}", moved(1));
        // The player comes and stands in front of the stallholder, a little to one side: they
        // turn their head and look.
        let p = &scene.people[1];
        let ahead = apply(rot_y(p.body.yaw), [0.0, 0.0, 1.0]);
        let feet = [p.body.base[0] + 1.0 * TF * ahead[0] + 0.8 * TF, 0.0, p.body.base[2] + 1.0 * TF * ahead[2]];
        let eye = [feet[0], -0.62 * TF, feet[2]];
        for _ in 0..60 {
            people.update(&mut scene, 0, Some((feet, eye)), 1.0 / 30.0);
        }
        let p = &scene.people[1];
        assert_eq!(p.watching, Some(eye));
        let at = p.body.pose()[p.bones.head];
        let looks = apply(at.axes, [0.0, 0.0, 1.0]);
        let to = norm(sub(eye, at.centre));
        assert!(dot(looks, to) > 0.85, "the head faces {looks:?}, the player is {to:?}");
    }

    #[test]
    fn the_castle_is_peopled_and_safe() {
        let l = crate::world::level::Level::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../levels/castle.ron"))).unwrap();
        let scene = l.to_scene();
        assert_eq!(scene.people.len(), 8);
        let guard = scene.people.iter().find(|p| p.role == Role::Guard).unwrap();
        assert_eq!(guard.name.as_deref(), Some("Gate guard"));
        assert!(guard.lines.len() >= 2 && guard.look.clothes.iter().any(|w| w.kind == clothes::Kind::LeatherCap));
        assert!(scene.friendly(guard.body.base), "the waking world is safe everywhere");
    }

    #[test]
    fn a_person_is_a_low_poly_model_with_a_face_and_hair() {
        let mut scene = open_floor();
        let mut look = Look::random(4);
        look.hair.style = crate::world::hair::Style::Bob;
        scene.add_person(look, Role::Stand, [5.0 * TF, 0.0, 5.0 * TF], 0.0, 4);
        let tris = scene.people_tris(|_| true);
        let face = tris.iter().filter(|t| t.tex == Scene::face_tex_id(0)).count();
        let hair = tris.iter().filter(|t| t.tex == tex::PERSON_HAIR).count();
        let skin = tris.iter().filter(|t| t.tex == tex::PERSON_SKIN).count();
        assert!(face >= 8 && hair > 50 && skin > 200, "{face} face, {hair} hair, {skin} skin");
        assert!(tris.len() < 2500, "{} triangles", tris.len());
        // The face is in front.
        let p = &scene.people[0];
        let ahead = apply(rot_y(p.body.yaw), [0.0, 0.0, 1.0]);
        for t in tris.iter().filter(|t| t.tex == Scene::face_tex_id(0)) {
            assert!(dot(t.n, ahead) > 0.0);
        }
    }
}
