//! Painted faces for people (`person`): one 64x64 16-colour texture a person, laid flat
//! over the front of the head, composited from a few kinds of each feature (eyes, brows,
//! nose, mouth) and redrawn as the face moves: the eyes blink and look about, the brows
//! rise, the mouth opens and shapes itself as they talk.
//!
//! The style is soft and cartoonish, against the dungeons: big glossy eyes set low with
//! heavy upper lashes, little or no nose, a small mouth, often a blush. The face is flat,
//! shaded only toward its edges.
//!
//! Texture coordinates: u across the face toward the head's local +x, v down it. The eyes'
//! middles sit on row `EYE_ROW`, the mouth's on `MOUTH_ROW`.

use crate::world::rng::Rng;
use crate::world::tex::{Texture, SIZE};
use serde::{Deserialize, Serialize};

pub const EYE_ROW: f32 = 36.0;
pub const MOUTH_ROW: f32 = 50.5;
const NOSE_ROW: f32 = 44.0;
const MID: f32 = SIZE as f32 / 2.0;
/// Half the distance between the eyes' middles.
const EYES_APART: f32 = 13.5;

/// Palette slots.
const OUTLINE: u8 = 0;
const SHADOW_DEEP: u8 = 1;
const SHADOW: u8 = 2;
const SKIN: u8 = 3;
const LIGHT: u8 = 4;
const WHITE: u8 = 5;
const IRIS_DARK: u8 = 6;
const IRIS: u8 = 7;
const IRIS_LIGHT: u8 = 8;
const PUPIL: u8 = 9;
const BROW: u8 = 10;
const MOUTH_LINE: u8 = 11;
const MOUTH: u8 = 12;
const TONGUE: u8 = 13;
const CHEEK: u8 = 14;
const SHINE: u8 = 15;

/// How the skin of the face is shaded (its plain colour is `SKIN`): the same factor the
/// body's texture averages, so the face meets the rest of the head without a seam.
pub const SKIN_SHADE: f32 = 0.9;

/// The kinds of each feature.
pub const EYES: u8 = 6;
pub const BROWS: u8 = 4;
pub const NOSES: u8 = 5;
pub const MOUTHS: u8 = 5;

/// What a face is made of: which kind of each feature, and their colours.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Features {
    pub eyes: u8,
    pub brows: u8,
    pub nose: u8,
    pub mouth: u8,
    pub iris: [f32; 3],
    /// The brows' colour, usually the hair's.
    pub brow: [f32; 3],
    /// The mouth's colour (the tongue, and the line of a closed mouth is a darker shade).
    pub lips: [f32; 3],
    #[serde(default)]
    pub blush: bool,
}

impl Features {
    pub fn random(rng: &mut Rng, hair: [f32; 3]) -> Features {
        let iris = rng.pick(&[
            [0.25, 0.45, 0.9],
            [0.2, 0.7, 0.55],
            [0.6, 0.35, 0.85],
            [0.85, 0.55, 0.2],
            [0.9, 0.4, 0.55],
            [0.45, 0.28, 0.15],
            [0.8, 0.2, 0.25],
            [0.35, 0.6, 0.85],
        ]);
        let lips = rng.pick(&[[0.9, 0.45, 0.5], [0.85, 0.4, 0.45], [0.95, 0.55, 0.55]]);
        Features {
            eyes: rng.range(0, EYES as i32) as u8,
            brows: rng.range(0, BROWS as i32) as u8,
            nose: rng.range(0, NOSES as i32) as u8,
            mouth: rng.range(0, MOUTHS as i32) as u8,
            iris,
            brow: hair.map(|c| c * 0.7),
            lips,
            blush: rng.chance(0.6),
        }
    }
}

/// The mouth's shapes: at rest, and the few talking cycles through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mouth {
    #[default]
    Closed,
    Ajar,
    Open,
    Oo,
    Smile,
}

/// How the face is moving now.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct State {
    /// 0 open .. 1 shut.
    pub blink: f32,
    /// Where the eyes look, -1..1: toward the head's local +x, and up.
    pub gaze: [f32; 2],
    pub mouth: Mouth,
    /// 0 at rest .. 1 raised.
    pub brow: f32,
}

struct EyeKind {
    /// Half width and half height in texels.
    w: f32,
    h: f32,
    /// How far each eye's outer corner sits above its inner one, in texels.
    slant: f32,
    /// The upper lid's resting droop, as a share of the eye's height.
    lid: f32,
    /// The iris's half width and half height.
    iris: [f32; 2],
    /// Just a dark oval with a glint, no white.
    dot: bool,
    /// A second, larger glint.
    sparkle: bool,
}

const EYE_KINDS: [EyeKind; EYES as usize] = [
    // Big and round.
    EyeKind { w: 7.2, h: 8.4, slant: 0.0, lid: 0.0, iris: [4.8, 6.6], dot: false, sparkle: false },
    // Tall and sparkling.
    EyeKind { w: 6.6, h: 9.6, slant: 0.5, lid: 0.0, iris: [4.8, 7.8], dot: false, sparkle: true },
    // Sleepy.
    EyeKind { w: 7.8, h: 7.2, slant: -0.5, lid: 0.4, iris: [4.8, 6.0], dot: false, sparkle: false },
    // Sharp, cat-like.
    EyeKind { w: 7.8, h: 6.0, slant: 2.0, lid: 0.1, iris: [3.6, 5.4], dot: false, sparkle: false },
    // Little dots.
    EyeKind { w: 2.6, h: 3.6, slant: 0.0, lid: 0.0, iris: [2.6, 3.6], dot: true, sparkle: false },
    // Wide and gentle.
    EyeKind { w: 8.4, h: 7.2, slant: -1.0, lid: 0.15, iris: [5.4, 6.0], dot: false, sparkle: true },
];

struct BrowKind {
    thick: f32,
    arch: f32,
    /// How far the inner end sits below the outer.
    tilt: f32,
    /// Half length, as a share of the eye's half width.
    len: f32,
}

const BROW_KINDS: [BrowKind; BROWS as usize] = [
    BrowKind { thick: 1.0, arch: 1.2, tilt: 0.0, len: 0.8 },
    BrowKind { thick: 1.5, arch: 0.5, tilt: 0.8, len: 0.9 },
    BrowKind { thick: 1.0, arch: 0.8, tilt: -1.0, len: 0.75 },
    BrowKind { thick: 2.0, arch: 0.3, tilt: 0.0, len: 0.6 },
];

#[derive(Clone, Copy)]
enum MouthKind {
    /// A small curve up at the ends.
    Smile(f32),
    /// ":3".
    Cat,
    /// A short dash.
    Dash(f32),
    /// Up at one end.
    Smirk,
}

const MOUTH_KINDS: [MouthKind; MOUTHS as usize] = [MouthKind::Smile(3.0), MouthKind::Cat, MouthKind::Dash(2.5), MouthKind::Smirk, MouthKind::Smile(4.5)];

/// The face's texture as it looks in `state`. `skin` is the skin's colour (as the body's tint).
pub fn compose(f: &Features, skin: [f32; 3], state: &State) -> Texture {
    let mut c = Canvas { idx: vec![SKIN; SIZE * SIZE] };
    c.shade();
    let eye = &EYE_KINDS[f.eyes as usize % EYE_KINDS.len()];
    if f.blush {
        for side in [-1.0, 1.0] {
            let (x, y) = (MID + side * (EYES_APART + 1.5), EYE_ROW + eye.h + 3.0);
            c.disc(x, y, 4.0, 1.8, |v| (v == SKIN || v == LIGHT).then_some(CHEEK));
            // A few strokes across it.
            for k in -1..=1 {
                let sx = (x + 2.0 * k as f32).floor() as i32;
                c.set(sx, y.floor() as i32, LIGHT);
                c.set(sx + 1, y.floor() as i32 - 1, LIGHT);
            }
        }
    }
    for side in [-1.0, 1.0] {
        c.eye(eye, side, state);
        c.brow(&BROW_KINDS[f.brows as usize % BROW_KINDS.len()], eye, side, state.brow);
    }
    c.nose(f.nose % NOSES);
    c.mouth(MOUTH_KINDS[f.mouth as usize % MOUTH_KINDS.len()], state.mouth);
    let k = |a: [f32; 3], s: f32| a.map(|x| ((x * s).clamp(0.0, 1.0) * 255.0).round() as u8);
    let mix = |a: [f32; 3], b: [f32; 3], t: f32| [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t);
    let mut pal = [[0u8; 3]; 16];
    pal[OUTLINE as usize] = k(mix(skin, [0.15, 0.08, 0.1], 0.8), 0.6);
    pal[SHADOW_DEEP as usize] = k(skin, 0.7 * SKIN_SHADE);
    pal[SHADOW as usize] = k(skin, 0.86 * SKIN_SHADE);
    pal[SKIN as usize] = k(skin, SKIN_SHADE);
    pal[LIGHT as usize] = k(skin, 1.05 * SKIN_SHADE);
    pal[WHITE as usize] = [246, 246, 250];
    pal[IRIS_DARK as usize] = k(f.iris, 0.45);
    pal[IRIS as usize] = k(f.iris, 0.85);
    pal[IRIS_LIGHT as usize] = k(mix(f.iris, [1.0; 3], 0.35), 1.0);
    pal[PUPIL as usize] = k(f.iris, 0.18);
    pal[BROW as usize] = k(f.brow, 1.0);
    pal[MOUTH_LINE as usize] = k(mix(f.lips, [0.3, 0.1, 0.12], 0.5), 0.8);
    pal[MOUTH as usize] = [110, 30, 45];
    pal[TONGUE as usize] = k(f.lips, 1.0);
    pal[CHEEK as usize] = k(mix(skin, [1.0, 0.45, 0.5], 0.35), SKIN_SHADE);
    pal[SHINE as usize] = [255, 255, 255];
    Texture { idx: c.idx, pal }
}

struct Canvas {
    idx: Vec<u8>,
}

impl Canvas {
    fn get(&self, x: i32, y: i32) -> Option<u8> {
        ((0..SIZE as i32).contains(&x) && (0..SIZE as i32).contains(&y)).then(|| self.idx[y as usize * SIZE + x as usize])
    }

    fn set(&mut self, x: i32, y: i32, v: u8) {
        if (0..SIZE as i32).contains(&x) && (0..SIZE as i32).contains(&y) {
            self.idx[y as usize * SIZE + x as usize] = v;
        }
    }

    /// Paint texels of an ellipse (radii `rx`, `ry` about the point (cx, cy)) as `paint`
    /// says, given what each was.
    fn disc(&mut self, cx: f32, cy: f32, rx: f32, ry: f32, paint: impl Fn(u8) -> Option<u8>) {
        for y in (cy - ry).floor() as i32..=(cy + ry).ceil() as i32 {
            for x in (cx - rx).floor() as i32..=(cx + rx).ceil() as i32 {
                let (dx, dy) = ((x as f32 + 0.5 - cx) / rx, (y as f32 + 0.5 - cy) / ry);
                if dx * dx + dy * dy <= 1.0 {
                    if let Some(v) = self.get(x, y).and_then(&paint) {
                        self.set(x, y, v);
                    }
                }
            }
        }
    }

    /// Flat, rounding away only at the very sides and under the chin.
    fn shade(&mut self) {
        for y in 0..SIZE as i32 {
            for x in 0..SIZE as i32 {
                let side = ((x as f32 + 0.5 - MID) / MID).abs();
                let v = if side > 0.93 {
                    SHADOW_DEEP
                } else if side > 0.82 {
                    SHADOW
                } else {
                    SKIN
                };
                self.set(x, y, v);
            }
        }
        for x in 0..SIZE as i32 {
            for y in 60..SIZE as i32 {
                self.set(x, y, SHADOW);
            }
        }
    }

    fn eye(&mut self, k: &EyeKind, side: f32, s: &State) {
        let (cx, cy) = (MID + side * EYES_APART, EYE_ROW);
        let shut = (k.lid + (1.0 - k.lid) * s.blink.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        if shut >= 0.95 {
            // Shut: a lash line bowed downward, flicking up at the outer corner.
            let n = (k.w * 4.0) as i32;
            for i in 0..=n {
                let t = i as f32 / n as f32 * 2.0 - 1.0;
                let x = cx + t * k.w;
                let y = cy + 0.15 * k.h + 0.3 * k.h * (1.0 - t * t);
                self.set(x.floor() as i32, y.floor() as i32, OUTLINE);
                if k.w > 3.0 {
                    self.set(x.floor() as i32, y.floor() as i32 - 1, OUTLINE);
                }
            }
            let tip = cx + side * k.w;
            self.set(tip.floor() as i32 + side as i32, (cy + 0.1 * k.h).floor() as i32, OUTLINE);
            return;
        }
        // The upper lid's edge, from the top of the eye down as it closes.
        let lid = -k.h + 2.0 * k.h * shut;
        let lash = if k.dot { 0.0 } else { 1.6 };
        let (gx, gy) = (s.gaze[0].clamp(-1.0, 1.0), s.gaze[1].clamp(-1.0, 1.0));
        let iris = [cx + gx * (k.w - k.iris[0]).max(1.2), cy + 0.6 - gy * (k.h - k.iris[1]).max(1.0)];
        for y in (cy - k.h - 2.0).floor() as i32..=(cy + k.h + 1.0).ceil() as i32 {
            for x in (cx - k.w - 2.0).floor() as i32..=(cx + k.w + 2.0).ceil() as i32 {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let dx = (px - cx) / k.w;
                // The outer corner rises by `slant`.
                let dy = py - cy + 0.5 * k.slant * dx * side;
                if dx * dx + (dy / k.h).powi(2) > 1.0 {
                    continue;
                }
                if dy < lid {
                    continue;
                }
                if dy < lid + lash {
                    self.set(x, y, OUTLINE);
                    continue;
                }
                let (ix, iy) = ((px - iris[0]) / k.iris[0], (py - iris[1]) / k.iris[1]);
                let r = ix * ix + iy * iy;
                let v = if k.dot {
                    PUPIL
                } else if r <= 1.0 {
                    if ix * ix * 6.0 + iy * iy * 4.5 < 1.0 {
                        PUPIL
                    } else if r > 0.72 || iy < -0.4 {
                        IRIS_DARK
                    } else if iy > 0.35 {
                        IRIS_LIGHT
                    } else {
                        IRIS
                    }
                } else if dy < lid + lash + 1.0 {
                    // The lid's shadow on the white.
                    LIGHT
                } else {
                    WHITE
                };
                self.set(x, y, v);
            }
        }
        // Glints: a big one up and to the side, a small one low.
        let glint = |c: &mut Canvas, x: f32, y: f32, big: bool| {
            let (x, y) = (x.floor() as i32, y.floor() as i32);
            let on = |c: &Canvas, x: i32, y: i32| c.get(x, y).is_some_and(|v| matches!(v, IRIS_DARK | IRIS | IRIS_LIGHT | PUPIL));
            for (ox, oy) in if big { &[(0, 0), (1, 0), (0, 1), (1, 1)][..] } else { &[(0, 0)][..] } {
                if on(c, x + ox, y + oy) {
                    c.set(x + ox, y + oy, SHINE);
                }
            }
        };
        let big = k.iris[0] > 3.0;
        glint(self, iris[0] - 0.45 * k.iris[0], iris[1] - 0.5 * k.iris[1], big);
        glint(self, iris[0] + 0.45 * k.iris[0], iris[1] + 0.4 * k.iris[1], false);
        if k.sparkle {
            glint(self, iris[0] + 0.2 * k.iris[0], iris[1] - 0.6 * k.iris[1], false);
        }
        if !k.dot {
            // The lash flicks out past the outer corner, and a touch of lower lash.
            let ty = cy + lid + 0.5 - k.slant * 0.5;
            let tx = cx + side * (k.w + 0.3);
            self.set(tx.floor() as i32, ty.floor() as i32, OUTLINE);
            self.set((tx + side).floor() as i32, ty.floor() as i32 - 1, OUTLINE);
            self.set((cx + side * (k.w - 1.0)).floor() as i32, (cy + 0.85 * k.h).floor() as i32, SHADOW_DEEP);
        }
    }

    fn brow(&mut self, k: &BrowKind, eye: &EyeKind, side: f32, raised: f32) {
        let half = k.len * eye.w.max(4.5);
        let cx = MID + side * EYES_APART;
        let base = EYE_ROW - eye.h.max(4.0) - 4.0 - 2.0 * raised.clamp(0.0, 1.0);
        let n = (half * 4.0) as i32;
        for i in 0..=n {
            // From the inner end (t = -1) to the outer (t = 1).
            let t = i as f32 / n as f32 * 2.0 - 1.0;
            let x = cx + side * t * half;
            let y = base - k.arch * (1.0 - t * t) + k.tilt * (1.0 - t) * 0.5;
            let thick = k.thick * (1.0 - 0.4 * t.max(0.0));
            for r in 0..thick.ceil() as i32 {
                self.set(x.floor() as i32, (y + r as f32).floor() as i32, BROW);
            }
        }
    }

    fn nose(&mut self, kind: u8) {
        let (x, y) = (MID as i32, NOSE_ROW as i32);
        match kind {
            // None at all.
            0 => {}
            // A dot.
            1 => self.set(x, y, SHADOW),
            // A little shadow under one side.
            2 => {
                self.set(x, y, SHADOW);
                self.set(x + 1, y, SHADOW);
                self.set(x + 1, y - 1, SHADOW);
            }
            // A small hooked line.
            3 => {
                for dy in -3..=0 {
                    self.set(x + 1, y + dy, SHADOW);
                }
                self.set(x, y, SHADOW);
            }
            // A pink tip.
            _ => self.disc(MID, NOSE_ROW - 0.5, 1.4, 1.0, |v| (v == SKIN).then_some(CHEEK)),
        }
    }

    fn mouth(&mut self, kind: MouthKind, m: Mouth) {
        let cy = MOUTH_ROW;
        // A line through texels.
        let line = |c: &mut Canvas, from: f32, to: f32, y: &dyn Fn(f32) -> f32| {
            let n = ((to - from) * 3.0).max(1.0) as i32;
            for i in 0..=n {
                let x = from + (to - from) * i as f32 / n as f32;
                c.set(x.floor() as i32, y(x).floor() as i32, MOUTH_LINE);
            }
        };
        match m {
            Mouth::Closed => match kind {
                MouthKind::Smile(w) => line(self, MID - w, MID + w, &|x| cy - 1.2 * ((x - MID) / w).powi(2)),
                MouthKind::Cat => {
                    // Two little bumps: a "w" on its side, ":3".
                    line(self, MID - 3.5, MID + 3.5, &|x| cy - 0.5 + 1.2 * ((x - MID).abs() / 1.75 * std::f32::consts::PI).sin().abs());
                }
                MouthKind::Dash(w) => line(self, MID - w, MID + w, &|_| cy),
                MouthKind::Smirk => line(self, MID - 3.0, MID + 3.5, &|x| cy - 1.5 * ((x - MID + 3.0) / 6.5).powi(2)),
            },
            Mouth::Smile => line(self, MID - 4.5, MID + 4.5, &|x| cy - 2.0 * ((x - MID) / 4.5).powi(2)),
            Mouth::Ajar | Mouth::Open | Mouth::Oo => {
                // An open "D", the tongue in its lower half; "o" is small and round.
                let (w, h) = match m {
                    Mouth::Ajar => (3.2, 1.8),
                    Mouth::Open => (4.2, 4.0),
                    _ => (2.0, 2.4),
                };
                let top = if m == Mouth::Oo { cy - h } else { cy - 0.8 };
                for y in (top.floor()) as i32..=(cy + h).ceil() as i32 {
                    for x in (MID - w - 1.0) as i32..=(MID + w + 1.0) as i32 {
                        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                        let dx = (px - MID) / w;
                        let inside = if m == Mouth::Oo {
                            dx * dx + ((py - cy) / h).powi(2) <= 1.0
                        } else {
                            py >= top && dx * dx + ((py - top) / (cy + h - top)).powi(2) <= 1.0
                        };
                        if inside {
                            let low = py > cy + 0.4 * h;
                            self.set(x, y, if low && m != Mouth::Oo { TONGUE } else { MOUTH });
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(t: &Texture, slot: u8) -> usize {
        t.idx.iter().filter(|&&i| i == slot).count()
    }

    fn eye_texels(t: &Texture) -> usize {
        [WHITE, IRIS_DARK, IRIS, IRIS_LIGHT, PUPIL].iter().map(|&s| count(t, s)).sum()
    }

    /// Mean column of the iris and pupil texels of the eye on `side`.
    fn iris_x(t: &Texture, side: f32) -> f32 {
        let (mut sum, mut n) = (0.0f32, 0.0f32);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let i = t.idx[y * SIZE + x];
                if matches!(i, IRIS_DARK | IRIS | IRIS_LIGHT | PUPIL) && (x as f32 - MID) * side > 0.0 {
                    sum += x as f32;
                    n += 1.0;
                }
            }
        }
        sum / n.max(1.0)
    }

    #[test]
    fn eyes_look_where_they_are_told_and_shut_when_they_blink() {
        let mut rng = Rng::new(3);
        for _ in 0..40 {
            let f = Features::random(&mut rng, [0.3, 0.2, 0.1]);
            let skin = [0.95, 0.8, 0.7];
            let ahead = compose(&f, skin, &State::default());
            assert!(count(&ahead, PUPIL) + count(&ahead, IRIS) > 10, "{f:?}");
            if !EYE_KINDS[f.eyes as usize].dot {
                for side in [-1.0, 1.0] {
                    let l = iris_x(&compose(&f, skin, &State { gaze: [-1.0, 0.0], ..Default::default() }), side);
                    let r = iris_x(&compose(&f, skin, &State { gaze: [1.0, 0.0], ..Default::default() }), side);
                    assert!(r > l + 1.5, "{f:?}: iris at {l} looking -x, {r} looking +x");
                }
            }
            let shut = compose(&f, skin, &State { blink: 1.0, ..Default::default() });
            assert_eq!(eye_texels(&shut), 0, "{f:?}");
            assert!(count(&shut, OUTLINE) > 6, "{f:?}: a shut eye still shows its lashes");
        }
    }

    #[test]
    fn the_mouth_opens_and_every_kind_differs() {
        let skin = [0.8, 0.6, 0.5];
        for mouth in 0..MOUTHS {
            let f = Features { eyes: 0, brows: 0, nose: 0, mouth, iris: [0.3; 3], brow: [0.2; 3], lips: [0.9, 0.45, 0.5], blush: false };
            let shut = compose(&f, skin, &State::default());
            let open = compose(&f, skin, &State { mouth: Mouth::Open, ..Default::default() });
            assert_eq!(count(&shut, MOUTH), 0);
            assert!(count(&shut, MOUTH_LINE) >= 4, "kind {mouth} at rest");
            assert!(count(&open, MOUTH) > 4 && count(&open, TONGUE) > 0, "kind {mouth}");
        }
        let face = |eyes, brows, nose, mouth| {
            let f = Features { eyes, brows, nose, mouth, iris: [0.3; 3], brow: [0.2; 3], lips: [0.9, 0.45, 0.5], blush: false };
            compose(&f, skin, &State::default()).idx
        };
        for k in 1..4 {
            assert_ne!(face(k, 0, 0, 0), face(0, 0, 0, 0), "eyes {k}");
            assert_ne!(face(0, k, 0, 0), face(0, 0, 0, 0), "brows {k}");
            assert_ne!(face(0, 0, k, 0), face(0, 0, 0, 0), "nose {k}");
            assert_ne!(face(0, 0, 0, k), face(0, 0, 0, 0), "mouth {k}");
        }
    }
}
