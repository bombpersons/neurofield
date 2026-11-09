//! The HUD, drawn pixel by pixel into an image at the network frame's own resolution, so
//! it is as chunky as the picture under it and stays solid whichever renderer is on.

use crate::player::{Item, Player, SLOTS};
use crate::sword::{self, Pose, ScreenTri};
use psx_bevy::psx::font;

const MARGIN: i32 = 8;
const SLOT: i32 = 20;
const GAP: i32 = 2;
/// Bar length per point of health or mana, so a bigger pool is a longer bar.
const BAR_UNIT: i32 = 2;
const COMPASS_R: i32 = 14;
/// Headings the compass can show: it redraws only when the needle really moves.
pub const HEADINGS: i32 = 128;

type Rgba = [u8; 4];
const BACK: Rgba = [0, 0, 0, 170];
const FRAME: Rgba = [205, 190, 150, 255];
const DIM: Rgba = [205, 190, 150, 90];
const HP: Rgba = [205, 35, 30, 255];
const MP: Rgba = [50, 85, 225, 255];
const WHITE: Rgba = [240, 235, 220, 255];

/// All that the HUD shows; it is redrawn when this changes.
#[derive(Clone, PartialEq)]
pub struct HudState {
    pub width: u32,
    pub hp: (i32, i32),
    pub mp: (i32, i32),
    pub slots: [Option<(Item, u32)>; SLOTS],
    /// The hotbar as a diamond for the d-pad, or a column for keys 1-4.
    pub pad: bool,
    /// Which way the view faces, 0..HEADINGS, growing as it turns left.
    pub heading: i32,
    pub sword: Pose,
}

impl HudState {
    pub fn of(p: &Player, width: u32, pad: bool, yaw: f32, sword: Pose) -> HudState {
        HudState {
            width,
            hp: (p.hp, p.max_hp),
            mp: (p.mp, p.max_mp),
            slots: p.hotbar.map(|s| s.map(|i| (i, p.count(i)))),
            pad,
            heading: (yaw / std::f32::consts::TAU * HEADINGS as f32).round().rem_euclid(HEADINGS as f32) as i32,
            sword,
        }
    }
}

/// RGBA rows `stride` pixels wide, of which `w` by `h` are in use.
struct Canvas<'a> {
    data: &'a mut [u8],
    stride: i32,
    w: i32,
    h: i32,
}

impl Canvas<'_> {
    fn put(&mut self, x: i32, y: i32, c: Rgba) {
        if (0..self.w).contains(&x) && (0..self.h).contains(&y) {
            let i = ((y * self.stride + x) * 4) as usize;
            self.data[i..i + 4].copy_from_slice(&c);
        }
    }

    fn fill(&mut self, x: i32, y: i32, w: i32, h: i32, c: Rgba) {
        for py in y..y + h {
            for px in x..x + w {
                self.put(px, py, c);
            }
        }
    }

    fn outline(&mut self, x: i32, y: i32, w: i32, h: i32, c: Rgba) {
        self.fill(x, y, w, 1, c);
        self.fill(x, y + h - 1, w, 1, c);
        self.fill(x, y, 1, h, c);
        self.fill(x + w - 1, y, 1, h, c);
    }

    /// A solid triangle, its corners' colours blended across it.
    fn triangle(&mut self, t: &ScreenTri) {
        let [a, b, c] = t.p;
        let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if area == 0 {
            return;
        }
        let x = (a.0.min(b.0).min(c.0).max(0), a.0.max(b.0).max(c.0).min(self.w - 1));
        let y = (a.1.min(b.1).min(c.1).max(0), a.1.max(b.1).max(c.1).min(self.h - 1));
        for py in y.0..=y.1 {
            for px in x.0..=x.1 {
                let w0 = (b.0 - px) * (c.1 - py) - (b.1 - py) * (c.0 - px);
                let w1 = (c.0 - px) * (a.1 - py) - (c.1 - py) * (a.0 - px);
                let w2 = area - w0 - w1;
                if (area > 0 && w0.min(w1).min(w2) >= 0) || (area < 0 && w0.max(w1).max(w2) <= 0) {
                    let mix = |i: usize| {
                        let sum = w0 * t.colour[0][i] as i32 + w1 * t.colour[1][i] as i32 + w2 * t.colour[2][i] as i32;
                        (sum / area).clamp(0, 255) as u8
                    };
                    self.put(px, py, [mix(0), mix(1), mix(2), 255]);
                }
            }
        }
    }

    fn line(&mut self, from: (f32, f32), to: (f32, f32), c: Rgba) {
        let n = (to.0 - from.0).abs().max((to.1 - from.1).abs()).ceil().max(1.0) as i32;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            self.put((from.0 + (to.0 - from.0) * t).round() as i32, (from.1 + (to.1 - from.1) * t).round() as i32, c);
        }
    }

    /// `text` in psx-bevy's bitmap font, with a dark edge so it reads over anything.
    fn text(&mut self, x: i32, y: i32, text: &str, c: Rgba) {
        let image = font::text_image(text);
        let Some(data) = image.data.as_ref() else { return };
        let (w, h) = (image.width() as i32, image.height() as i32);
        for pass in [true, false] {
            for py in 0..h {
                for px in 0..w {
                    if data[((py * w + px) * 4 + 3) as usize] == 0 {
                        continue;
                    }
                    if pass {
                        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                            self.put(x + px + dx, y + py + dy, [0, 0, 0, 255]);
                        }
                    } else {
                        self.put(x + px, y + py, c);
                    }
                }
            }
        }
    }

    fn bar(&mut self, x: i32, y: i32, height: i32, (now, most): (i32, i32), c: Rgba) {
        let w = most * BAR_UNIT;
        self.fill(x - 1, y - 1, w + 2, height + 2, BACK);
        self.fill(x, y, now.clamp(0, most) * BAR_UNIT, height, c);
        self.outline(x - 1, y - 1, w + 2, height + 2, FRAME);
    }

    /// A flask of `liquid`, in the 16 x 16 square at (x, y).
    fn flask(&mut self, x: i32, y: i32, liquid: Rgba) {
        self.fill(x + 6, y + 1, 4, 2, [150, 105, 60, 255]); // the cork
        self.fill(x + 6, y + 3, 4, 3, DIM); // the neck
        self.fill(x + 3, y + 6, 10, 9, liquid);
        self.fill(x + 4, y + 5, 8, 1, liquid);
        self.fill(x + 4, y + 15, 8, 1, liquid);
        self.fill(x + 5, y + 7, 2, 4, [255, 255, 255, 170]); // the shine
    }

    fn icon(&mut self, x: i32, y: i32, item: Item) {
        match item {
            Item::HealingPotion => self.flask(x, y, HP),
            Item::ManaPotion => self.flask(x, y, MP),
        }
    }

    /// One hotbar slot at (x, y): its key's number on a column, and how many are left.
    fn slot(&mut self, x: i32, y: i32, key: Option<i32>, content: Option<(Item, u32)>) {
        self.fill(x, y, SLOT, SLOT, BACK);
        self.outline(x, y, SLOT, SLOT, if content.is_some() { FRAME } else { DIM });
        if let Some((item, count)) = content {
            self.icon(x + 2, y + 2, item);
            let n = count.to_string();
            self.text(x + SLOT - 1 - font::text_width(&n) as i32, y + SLOT - font::GLYPH_H as i32, &n, WHITE);
        }
        if let Some(k) = key {
            self.text(x + 2, y + 1, &k.to_string(), FRAME);
        }
    }

    fn compass(&mut self, heading: i32) {
        let (cx, cy) = ((self.w - MARGIN - COMPASS_R - 1) as f32, (MARGIN + COMPASS_R + 1) as f32);
        // Turning left swings north round to the right.
        let a = heading as f32 / HEADINGS as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        let r = COMPASS_R as f32;
        for i in 0..96 {
            let t = i as f32 / 96.0 * std::f32::consts::TAU;
            self.put((cx + t.cos() * r).round() as i32, (cy + t.sin() * r).round() as i32, FRAME);
        }
        let at = |d: f32, side: f32| (cx + s * d + c * side, cy - c * d + s * side);
        self.line(at(-9.0, 0.0), at(0.0, 0.0), WHITE);
        self.line(at(10.0, 0.0), at(0.0, 2.0), HP);
        self.line(at(10.0, 0.0), at(0.0, -2.0), HP);
        self.line(at(10.0, 0.0), at(0.0, 0.0), HP);
    }

    fn hotbar(&mut self, state: &HudState) {
        if state.pad {
            // Up, right, down, left, as the d-pad.
            let step = SLOT + GAP;
            let (cx, cy) = (MARGIN + step, self.h - MARGIN - 2 * step - SLOT + GAP);
            let at = [(cx, cy - step + GAP), (cx + step, cy + GAP), (cx, cy + step + GAP), (cx - step, cy + GAP)];
            for (i, (x, y)) in at.into_iter().enumerate() {
                self.slot(x, y, None, state.slots[i]);
            }
        } else {
            let bottom = self.h - MARGIN - SLOT;
            for i in 0..SLOTS as i32 {
                let y = bottom - (SLOTS as i32 - 1 - i) * (SLOT + GAP);
                self.slot(MARGIN, y, Some(i + 1), state.slots[i as usize]);
            }
        }
    }

    /// The selected spell's slot, empty until there are spells.
    fn spell(&mut self) {
        let size = SLOT + 8;
        let (x, y) = (self.w - MARGIN - size, self.h - MARGIN - size);
        self.fill(x, y, size, size, BACK);
        self.outline(x, y, size, size, DIM);
    }
}

/// Draw the HUD over a cleared image of `stride` by `h` RGBA pixels.
pub fn draw(data: &mut [u8], stride: u32, h: u32, state: &HudState) {
    data.fill(0);
    let mut c = Canvas { data, stride: stride as i32, w: state.width as i32, h: h as i32 };
    // The sword first, so the HUD stays readable over it.
    for t in sword::triangles(state.sword, state.width, h) {
        c.triangle(&t);
    }
    c.bar(MARGIN + 1, MARGIN + 1, 5, state.hp, HP);
    c.bar(MARGIN + 1, MARGIN + 9, 3, state.mp, MP);
    c.compass(state.heading);
    c.hotbar(state);
    c.spell();
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 320;
    const H: u32 = 240;

    fn at(data: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * W + x) * 4) as usize;
        [data[i], data[i + 1], data[i + 2], data[i + 3]]
    }

    fn drawn(state: &HudState) -> Vec<u8> {
        let mut data = vec![255; (W * H * 4) as usize];
        draw(&mut data, W, H, state);
        data
    }

    fn pose() -> Pose {
        crate::sword::Sword::default().pose(0.0)
    }

    fn state() -> HudState {
        HudState::of(&Player::new(), W, false, 0.0, pose())
    }

    #[test]
    fn the_sword_is_drawn_solid() {
        let d = drawn(&state());
        let solid = (160..W).filter(|&x| (120..H).any(|y| at(&d, x, y)[3] == 255)).count();
        assert!(solid > 20, "{solid} columns");
    }

    #[test]
    fn the_health_bar_shows_how_much_is_left() {
        let mut s = state();
        let full = drawn(&s);
        assert_eq!(at(&full, 9 + 70, 11), HP);
        s.hp = (20, 40);
        let half = drawn(&s);
        assert_eq!(at(&half, 9 + 30, 11), HP);
        assert_ne!(at(&half, 9 + 70, 11), HP);
    }

    #[test]
    fn the_mana_bar_is_as_long_as_the_pool() {
        let d = drawn(&state());
        assert_eq!(at(&d, 9 + 20, 18), MP);
        assert_ne!(at(&d, 9 + 60, 18), MP);
    }

    #[test]
    fn the_needle_points_north_ahead_and_swings_right_on_turning_left() {
        let needle = |heading: i32| {
            let mut s = state();
            s.heading = heading;
            let d = drawn(&s);
            let (cx, cy) = ((W as i32 - MARGIN - COMPASS_R - 1) as u32, (MARGIN + COMPASS_R + 1) as u32);
            (at(&d, cx, cy - 8), at(&d, cx + 8, cy))
        };
        assert_eq!(needle(0).0, HP);
        assert_eq!(needle(HEADINGS / 4).1, HP);
    }

    #[test]
    fn the_hotbar_changes_shape_with_the_controller() {
        let mut s = state();
        let column = drawn(&s);
        s.pad = true;
        let diamond = drawn(&s);
        assert_ne!(column, diamond);
    }

    #[test]
    fn nothing_is_drawn_past_the_frame_in_use() {
        let mut s = state();
        s.width = 200;
        let d = drawn(&s);
        assert!((200..W).all(|x| (0..H).all(|y| at(&d, x, y)[3] == 0)));
    }

    #[test]
    fn the_heading_wraps_round() {
        let p = Player::new();
        let s = |yaw: f32| HudState::of(&p, W, false, yaw, pose()).heading;
        assert_eq!(s(std::f32::consts::TAU), s(0.0));
        assert_eq!(s(-std::f32::consts::FRAC_PI_2), HEADINGS * 3 / 4);
    }
}
