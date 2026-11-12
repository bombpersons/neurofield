//! Painted faces for people (`person`): one 64x64 16-colour texture a person, laid flat
//! over the front of the head, composited from a few kinds of each feature (eyes, brows,
//! nose, mouth) and redrawn as the face moves: the eyes blink and look about, the brows
//! rise, the mouth opens and shapes itself as they talk.
//!
//! Texture coordinates: u across the face toward the head's local +x, v down it. The eyes'
//! middles sit on row `EYE_ROW`, the mouth's on `MOUTH_ROW`.

use crate::world::rng::Rng;
use crate::world::tex::{Texture, SIZE};
use serde::{Deserialize, Serialize};

pub const EYE_ROW: f32 = 31.0;
pub const BROW_ROW: f32 = 25.0;
pub const NOSE_ROW: f32 = 41.0;
pub const MOUTH_ROW: f32 = 48.5;
const MID: f32 = SIZE as f32 / 2.0;

/// Palette slots.
const OUTLINE: u8 = 0;
const SHADOW_DEEP: u8 = 1;
const SHADOW: u8 = 2;
const SKIN: u8 = 3;
const LIGHT: u8 = 4;
const WHITE: u8 = 5;
const IRIS: u8 = 6;
const IRIS_DARK: u8 = 7;
const PUPIL: u8 = 8;
const BROW: u8 = 9;
const LIP: u8 = 10;
const LIP_DARK: u8 = 11;
const MOUTH: u8 = 12;
const TEETH: u8 = 13;
const CHEEK: u8 = 14;
const SHINE: u8 = 15;

/// How the skin of the face is shaded (its plain colour is `SKIN`): the same factor the
/// body's texture averages, so the face meets the rest of the head without a seam.
pub const SKIN_SHADE: f32 = 0.9;

/// The kinds of each feature.
pub const EYES: u8 = 6;
pub const BROWS: u8 = 6;
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
    pub lips: [f32; 3],
    #[serde(default)]
    pub blush: bool,
}

impl Features {
    pub fn random(rng: &mut Rng, hair: [f32; 3]) -> Features {
        let iris = rng.pick(&[[0.35, 0.22, 0.12], [0.2, 0.35, 0.6], [0.3, 0.45, 0.25], [0.45, 0.35, 0.2], [0.15, 0.12, 0.1], [0.4, 0.45, 0.5]]);
        let lips = rng.pick(&[[0.72, 0.42, 0.4], [0.62, 0.35, 0.33], [0.78, 0.5, 0.48], [0.55, 0.3, 0.3]]);
        Features {
            eyes: rng.range(0, EYES as i32) as u8,
            brows: rng.range(0, BROWS as i32) as u8,
            nose: rng.range(0, NOSES as i32) as u8,
            mouth: rng.range(0, MOUTHS as i32) as u8,
            iris,
            brow: hair.map(|c| c * 0.8),
            lips,
            blush: rng.chance(0.3),
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
    iris: f32,
    lashes: bool,
}

const EYE_KINDS: [EyeKind; EYES as usize] = [
    EyeKind { w: 5.0, h: 3.0, slant: 0.0, lid: 0.1, iris: 2.3, lashes: false },
    EyeKind { w: 5.5, h: 2.6, slant: 1.0, lid: 0.15, iris: 2.1, lashes: true },
    EyeKind { w: 4.5, h: 3.2, slant: -0.5, lid: 0.05, iris: 2.4, lashes: false },
    EyeKind { w: 6.0, h: 2.8, slant: 0.4, lid: 0.25, iris: 2.1, lashes: false },
    EyeKind { w: 5.0, h: 3.4, slant: 0.2, lid: 0.0, iris: 2.6, lashes: true },
    EyeKind { w: 5.5, h: 2.8, slant: -1.0, lid: 0.25, iris: 2.2, lashes: false },
];

/// Half the distance between the eyes' middles.
const EYES_APART: f32 = 11.5;

struct BrowKind {
    thick: f32,
    arch: f32,
    /// How far the inner end sits below the outer.
    tilt: f32,
    /// Half length, as a share of the eye's half width.
    len: f32,
}

const BROW_KINDS: [BrowKind; BROWS as usize] = [
    BrowKind { thick: 1.6, arch: 1.0, tilt: 0.0, len: 1.15 },
    BrowKind { thick: 2.4, arch: 0.4, tilt: 0.6, len: 1.2 },
    BrowKind { thick: 1.2, arch: 2.0, tilt: -0.4, len: 1.0 },
    BrowKind { thick: 2.0, arch: 0.0, tilt: 1.2, len: 1.1 },
    BrowKind { thick: 1.4, arch: 1.4, tilt: 0.3, len: 1.3 },
    BrowKind { thick: 2.8, arch: 0.8, tilt: -0.2, len: 1.25 },
];

struct NoseKind {
    /// Rows from the eyes' line down to the tip.
    len: f32,
    /// Half the nostrils' spread.
    width: f32,
    /// The bridge drawn as a shadow down one side.
    bridge: bool,
    /// A rounded tip, lit.
    bulb: bool,
}

const NOSE_KINDS: [NoseKind; NOSES as usize] = [
    NoseKind { len: 10.0, width: 2.5, bridge: true, bulb: false },
    NoseKind { len: 8.0, width: 3.0, bridge: false, bulb: true },
    NoseKind { len: 12.0, width: 2.0, bridge: true, bulb: true },
    NoseKind { len: 9.0, width: 3.5, bridge: false, bulb: false },
    NoseKind { len: 11.0, width: 3.0, bridge: true, bulb: false },
];

struct MouthKind {
    /// Half width.
    w: f32,
    /// The lower lip's thickness in rows.
    lip: f32,
    /// How far the corners sit above the middle at rest.
    curve: f32,
}

const MOUTH_KINDS: [MouthKind; MOUTHS as usize] = [
    MouthKind { w: 6.0, lip: 1.5, curve: 0.0 },
    MouthKind { w: 7.5, lip: 2.0, curve: 0.5 },
    MouthKind { w: 5.0, lip: 1.0, curve: -0.6 },
    MouthKind { w: 6.5, lip: 2.5, curve: 0.2 },
    MouthKind { w: 8.0, lip: 1.2, curve: -0.2 },
];

/// The face's texture as it looks in `state`. `skin` is the skin's colour (as the body's tint).
pub fn compose(f: &Features, skin: [f32; 3], state: &State) -> Texture {
    let mut c = Canvas { idx: vec![SKIN; SIZE * SIZE] };
    c.shade();
    let eye = &EYE_KINDS[f.eyes as usize % EYE_KINDS.len()];
    for side in [-1.0, 1.0] {
        c.eye(eye, side, state);
        c.brow(&BROW_KINDS[f.brows as usize % BROW_KINDS.len()], eye, side, state.brow);
    }
    c.nose(&NOSE_KINDS[f.nose as usize % NOSE_KINDS.len()]);
    c.mouth(&MOUTH_KINDS[f.mouth as usize % MOUTH_KINDS.len()], state.mouth);
    if f.blush {
        for side in [-1.0, 1.0] {
            c.disc(MID + side * 13.0, NOSE_ROW + 2.0, 2.6, 1.0, |was| if was == SKIN || was == LIGHT { Some(CHEEK) } else { None });
        }
    }
    let k = |a: [f32; 3], s: f32| a.map(|x| ((x * s).clamp(0.0, 1.0) * 255.0).round() as u8);
    let mut pal = [[0u8; 3]; 16];
    pal[OUTLINE as usize] = k(skin, 0.28);
    pal[SHADOW_DEEP as usize] = k(skin, 0.6 * SKIN_SHADE);
    pal[SHADOW as usize] = k(skin, 0.8 * SKIN_SHADE);
    pal[SKIN as usize] = k(skin, SKIN_SHADE);
    pal[LIGHT as usize] = k(skin, 1.08 * SKIN_SHADE);
    pal[WHITE as usize] = [232, 228, 218];
    pal[IRIS as usize] = k(f.iris, 1.0);
    pal[IRIS_DARK as usize] = k(f.iris, 0.6);
    pal[PUPIL as usize] = [16, 12, 12];
    pal[BROW as usize] = k(f.brow, 1.0);
    pal[LIP as usize] = k([0, 1, 2].map(|i| 0.5 * (f.lips[i] + skin[i])), 1.0);
    pal[LIP_DARK as usize] = k(f.lips, 0.6);
    pal[MOUTH as usize] = [70, 22, 24];
    pal[TEETH as usize] = [220, 214, 196];
    pal[CHEEK as usize] = k([0, 1, 2].map(|i| skin[i] * SKIN_SHADE * [1.08, 0.85, 0.85][i]), 1.0);
    pal[SHINE as usize] = [250, 250, 250];
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

    /// Paint texels of an ellipse (radii `rx`, `ry` about the texel centre (cx, cy)) as
    /// `paint` says, given what each was.
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

    /// The face rounding away toward its sides and under the brows and nose.
    fn shade(&mut self) {
        for y in 0..SIZE as i32 {
            for x in 0..SIZE as i32 {
                let side = ((x as f32 + 0.5 - MID) / MID).abs();
                let v = if side > 0.9 {
                    SHADOW_DEEP
                } else if side > 0.72 {
                    SHADOW
                } else {
                    SKIN
                };
                self.set(x, y, v);
            }
        }
        // The forehead and cheekbones catch the light.
        self.disc(MID, 16.0, 8.0, 3.0, |v| (v == SKIN).then_some(LIGHT));
        for side in [-1.0, 1.0] {
            self.disc(MID + side * 12.0, EYE_ROW + 6.0, 4.0, 1.6, |v| (v == SKIN).then_some(LIGHT));
            // Sockets, under the brow.
            self.disc(MID + side * EYES_APART, EYE_ROW - 2.5, 6.5, 2.6, |v| (v == SKIN || v == LIGHT).then_some(SHADOW));
        }
        // Under the chin.
        for x in 0..SIZE as i32 {
            for y in 58..SIZE as i32 {
                self.set(x, y, SHADOW);
            }
        }
    }

    fn eye(&mut self, k: &EyeKind, side: f32, s: &State) {
        let cx = MID + side * EYES_APART;
        let cy = EYE_ROW;
        // The upper lid's edge: from the top of the eye down to its bottom as it shuts.
        let shut = (k.lid + (1.0 - k.lid) * s.blink.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        // Shut, the lid's edge stays a line across the eye's lower part.
        let lid = (-k.h + 2.0 * k.h * shut).min(0.5 * k.h - 0.5);
        let (gx, gy) = (s.gaze[0].clamp(-1.0, 1.0), s.gaze[1].clamp(-1.0, 1.0));
        let iris = [cx + gx * (k.w - k.iris), cy - gy * (k.h - 0.6 * k.iris).max(0.5)];
        for y in (cy - k.h - 1.0).floor() as i32..=(cy + k.h + 1.0).ceil() as i32 {
            for x in (cx - k.w - 1.0).floor() as i32..=(cx + k.w + 1.0).ceil() as i32 {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let dx = (px - cx) / k.w;
                // The outer corner (away from the nose) rises by `slant`.
                let dy = py - cy + 0.5 * k.slant * dx * side;
                let inside = dx * dx + (dy / k.h).powi(2) <= 1.0;
                if !inside {
                    continue;
                }
                let edge = dx * dx + (dy / k.h).powi(2) > 0.55;
                if dy < lid {
                    // Lid over the eye.
                    self.set(x, y, if dy < -k.h + 1.0 { SHADOW } else { SKIN });
                } else if dy < lid + 1.0 {
                    self.set(x, y, OUTLINE);
                } else if shut >= 0.95 {
                    self.set(x, y, SKIN);
                } else {
                    let (ix, iy) = (px - iris[0], py - iris[1]);
                    let r = (ix * ix + iy * iy).sqrt();
                    let v = if r < 0.75 {
                        PUPIL
                    } else if r < k.iris - 0.7 {
                        IRIS
                    } else if r < k.iris {
                        IRIS_DARK
                    } else if edge && dy > 0.0 {
                        LIGHT
                    } else {
                        WHITE
                    };
                    self.set(x, y, v);
                }
            }
        }
        if shut < 0.95 {
            // A glint.
            let g = [(iris[0] - 0.8).floor() as i32, (iris[1] - 0.8).floor() as i32];
            if self.get(g[0], g[1]).is_some_and(|v| v == IRIS || v == IRIS_DARK || v == PUPIL) {
                self.set(g[0], g[1], SHINE);
            }
        }
        if k.lashes {
            // A flick out past the outer corner.
            let (x, y) = ((cx + side * (k.w + 0.5)).floor() as i32, (cy + lid - k.slant * 0.5).floor() as i32);
            self.set(x, y, OUTLINE);
            self.set(x + side as i32, y - 1, OUTLINE);
        }
    }

    fn brow(&mut self, k: &BrowKind, eye: &EyeKind, side: f32, raised: f32) {
        let half = k.len * eye.w;
        let cx = MID + side * EYES_APART;
        let base = BROW_ROW - 2.0 * raised.clamp(0.0, 1.0) - (eye.h - 3.0);
        let n = (half * 4.0) as i32;
        for i in 0..=n {
            // From the inner end (t = -1) to the outer (t = 1).
            let t = i as f32 / n as f32 * 2.0 - 1.0;
            let x = cx + side * t * half;
            let y = base - k.arch * (1.0 - t * t) + k.tilt * (1.0 - t) * 0.5 - eye.slant * 0.4 * (t + 1.0);
            let thick = k.thick * (1.0 - 0.35 * t.max(0.0));
            for r in 0..thick.ceil() as i32 {
                self.set(x.floor() as i32, (y + r as f32).floor() as i32, BROW);
            }
        }
    }

    fn nose(&mut self, k: &NoseKind) {
        let tip = EYE_ROW + k.len;
        if k.bridge {
            for y in (EYE_ROW + 2.0) as i32..tip as i32 - 1 {
                self.set(MID as i32 + 2, y, SHADOW);
            }
        }
        // The shadow under the tip, and the nostrils.
        for x in (MID - k.width) as i32..=(MID + k.width) as i32 {
            self.set(x, tip as i32 + 1, SHADOW);
        }
        for side in [-1.0, 1.0] {
            self.set((MID + side * k.width - 0.5).floor() as i32, tip as i32, SHADOW_DEEP);
            self.set((MID + side * (k.width + 1.0) - 0.5).floor() as i32, tip as i32 - 1, SHADOW);
        }
        if k.bulb {
            self.disc(MID, tip - 1.5, 1.6, 1.4, |v| (v == SKIN || v == SHADOW).then_some(LIGHT));
        } else {
            self.set(MID as i32, tip as i32 - 1, LIGHT);
        }
    }

    fn mouth(&mut self, k: &MouthKind, m: Mouth) {
        let cy = MOUTH_ROW;
        let (w, open, curve) = match m {
            Mouth::Closed => (k.w, 0.0, k.curve),
            Mouth::Ajar => (k.w * 0.9, 1.0, k.curve * 0.5),
            Mouth::Open => (k.w * 0.85, 3.0, 0.0),
            Mouth::Oo => (k.w * 0.45, 2.2, 0.0),
            Mouth::Smile => (k.w * 1.1, 0.0, k.curve + 1.8),
        };
        // The line between the lips, the corners `curve` above the middle.
        let line = |x: f32| cy - curve * ((x - MID) / w).powi(2);
        // Upper lip.
        for x in (MID - w + 1.0) as i32..(MID + w - 1.0) as i32 {
            let y = line(x as f32 + 0.5) - open / 2.0;
            self.set(x, (y - 1.0).floor() as i32, LIP);
        }
        // Lower lip.
        for x in (MID - w * 0.8) as i32..(MID + w * 0.8) as i32 {
            let y = line(x as f32 + 0.5) + open / 2.0;
            for r in 1..=k.lip.round() as i32 {
                self.set(x, (y + r as f32).floor() as i32, LIP);
            }
            self.set(x, (y + k.lip + 1.0).floor() as i32, SHADOW);
        }
        if open < 0.5 {
            for x in (MID - w) as i32..(MID + w) as i32 {
                self.set(x, line(x as f32 + 0.5).floor() as i32, LIP_DARK);
            }
            return;
        }
        // Open: the inside dark, the top teeth showing when it's wide.
        let ry = open / 2.0 + 0.5;
        self.disc(MID, cy, w, ry, |_| Some(MOUTH));
        if open >= 2.5 && m != Mouth::Oo {
            for x in (MID - w * 0.6) as i32..(MID + w * 0.6) as i32 {
                self.set(x, (cy - ry + 0.5).floor() as i32, TEETH);
            }
        }
        self.disc(MID, cy, w + 0.6, ry + 0.6, |v| (v != MOUTH && v != TEETH && v != LIP).then_some(LIP_DARK));
        self.disc(MID, cy, w, ry, |v| (v == LIP_DARK).then_some(MOUTH));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(t: &Texture, slot: u8) -> usize {
        t.idx.iter().filter(|&&i| i == slot).count()
    }

    /// Mean column of the iris and pupil texels of the eye on `side`.
    fn iris_x(t: &Texture, side: f32) -> f32 {
        let (mut sum, mut n) = (0.0f32, 0.0f32);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let i = t.idx[y * SIZE + x];
                if (i == IRIS || i == IRIS_DARK || i == PUPIL) && (x as f32 - MID) * side > 0.0 {
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
        for _ in 0..30 {
            let f = Features::random(&mut rng, [0.3, 0.2, 0.1]);
            let skin = [0.85, 0.66, 0.52];
            let ahead = compose(&f, skin, &State::default());
            assert!(count(&ahead, IRIS) + count(&ahead, IRIS_DARK) > 10, "{f:?}");
            for side in [-1.0, 1.0] {
                let l = iris_x(&compose(&f, skin, &State { gaze: [-1.0, 0.0], ..Default::default() }), side);
                let r = iris_x(&compose(&f, skin, &State { gaze: [1.0, 0.0], ..Default::default() }), side);
                assert!(r > l + 2.0, "{f:?}: iris at {l} looking -x, {r} looking +x");
            }
            let shut = compose(&f, skin, &State { blink: 1.0, ..Default::default() });
            assert_eq!(count(&shut, IRIS) + count(&shut, IRIS_DARK) + count(&shut, PUPIL) + count(&shut, WHITE), 0, "{f:?}");
        }
    }

    #[test]
    fn the_mouth_opens_and_every_kind_differs() {
        let skin = [0.6, 0.45, 0.35];
        for mouth in 0..MOUTHS {
            let f = Features { eyes: 0, brows: 0, nose: 0, mouth, iris: [0.3; 3], brow: [0.2; 3], lips: [0.7, 0.4, 0.4], blush: false };
            let shut = compose(&f, skin, &State::default());
            let open = compose(&f, skin, &State { mouth: Mouth::Open, ..Default::default() });
            assert_eq!(count(&shut, MOUTH), 0);
            assert!(count(&open, MOUTH) > 8 && count(&open, TEETH) > 0, "kind {mouth}");
        }
        let face = |eyes, brows, nose, mouth| {
            let f = Features { eyes, brows, nose, mouth, iris: [0.3; 3], brow: [0.2; 3], lips: [0.7, 0.4, 0.4], blush: false };
            compose(&f, skin, &State::default()).idx
        };
        for k in 1..5 {
            assert_ne!(face(k, 0, 0, 0), face(0, 0, 0, 0), "eyes {k}");
            assert_ne!(face(0, k, 0, 0), face(0, 0, 0, 0), "brows {k}");
            assert_ne!(face(0, 0, k, 0), face(0, 0, 0, 0), "nose {k}");
            assert_ne!(face(0, 0, 0, k), face(0, 0, 0, 0), "mouth {k}");
        }
    }
}
