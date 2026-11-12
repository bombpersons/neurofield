//! The HUD, drawn pixel by pixel into an image at the network frame's own resolution, so
//! it is as chunky as the picture under it and stays solid whichever renderer is on.

use crate::menu::{self, Choice, GearWindow, ItemWindow, Menu, Screen, Window, PAUSE_ROWS};
use crate::hands::{self, Kind, Pose, ScreenTri};
use crate::player::{Item, Piece, Player, Stats, CLARITY_SECS, EQUIPMENT, SLOTS, STAMINA};
use psx_bevy::psx::font;

const MARGIN: i32 = 8;
const SLOT: i32 = 20;
const GAP: i32 = 2;
/// Bar length per point of health or mana, so a bigger pool is a longer bar.
const BAR_UNIT: i32 = 2;
const COMPASS_R: i32 = 14;
/// Points in the bar of clarity left, each `BAR_UNIT` long.
const CLARITY_POINTS: i32 = 24;
/// Headings the compass can show: it redraws only when the needle really moves.
pub const HEADINGS: i32 = 128;

type Rgba = [u8; 4];
const BACK: Rgba = [0, 0, 0, 170];
const FRAME: Rgba = [205, 190, 150, 255];
const DIM: Rgba = [205, 190, 150, 90];
const HP: Rgba = [205, 35, 30, 255];
const MP: Rgba = [50, 85, 225, 255];
const CLARITY: Rgba = [205, 235, 230, 255];
const STAMINA_BAR: Rgba = [110, 190, 60, 255];
const CALMING: Rgba = [165, 115, 225, 255];
const WHITE: Rgba = [240, 235, 220, 255];
/// Health and mana as text, lighter than their bars so they read on a dark panel.
const HP_TEXT: Rgba = [240, 95, 85, 255];
const MP_TEXT: Rgba = [120, 150, 255, 255];
/// A stat going up or down with what might be worn.
const BETTER: Rgba = [130, 225, 120, 255];
const WORSE: Rgba = [240, 95, 85, 255];
/// What is broken: its name, and the crack across its icon.
const BROKEN: Rgba = [200, 90, 70, 255];
/// Gear's colours: steel, wood, leather and copper.
const STEEL: Rgba = [200, 205, 215, 255];
const WOOD: Rgba = [150, 105, 60, 255];
const LEATHER: Rgba = [120, 80, 45, 255];
const COPPER: Rgba = [215, 130, 70, 255];
/// Height of an equipment slot's row.
const GEAR_ROW: i32 = 12;
/// The menus' panels, darker than the HUD's so text reads over a bright frame.
const PANEL: Rgba = [0, 0, 0, 215];
const PICKED: Rgba = [95, 80, 50, 235];
/// Height of a row of text in a menu, and of an item's row with its icon.
const ROW: i32 = 14;
const ITEM_ROW: i32 = 18;
/// Width of the pause menu and of an item's window.
const PAUSE_W: i32 = 120;
const WINDOW_W: i32 = 150;

/// All that the HUD shows; it is redrawn when this changes.
#[derive(Clone, PartialEq)]
pub struct HudState {
    pub width: u32,
    pub hp: (i32, i32),
    pub mp: (i32, i32),
    pub stamina: (i32, i32),
    /// Points of clarity left, out of `CLARITY_POINTS`; none and the bar is gone.
    pub clarity: i32,
    pub slots: [Option<(Item, u32)>; SLOTS],
    /// The hotbar as a diamond for the d-pad, or a column for keys 1-4.
    pub pad: bool,
    /// Which way the view faces, 0..HEADINGS, growing as it turns left.
    pub heading: i32,
    /// What the hands hold, and their poses.
    pub hands: Vec<(Kind, Pose)>,
    /// Dropped things in view (`drops::triangles`), under the hands and the HUD.
    pub drops: Vec<ScreenTri>,
    pub menu: Option<MenuShown>,
    /// A notice that something happened (gear breaking), shown for a moment.
    pub toast: Option<String>,
}

/// The menu as it is drawn, with what it shows of the player.
#[derive(Clone, PartialEq)]
pub struct MenuShown {
    pub menu: Menu,
    pub items: Vec<(Piece, u32)>,
    pub hotbar: [Option<Item>; SLOTS],
    pub equipped: [Option<Piece>; EQUIPMENT.len()],
    pub stats: Stats,
    /// The open item window's choices, or the equipment window's (`None` takes off).
    pub choices: Vec<Choice>,
    pub wears: Vec<Option<Piece>>,
    /// The stats with what the equipment window has picked worn.
    pub preview: Option<Stats>,
}

/// A selectable row as drawn, `[x, y, w, h]` in frame pixels, and its place in the list
/// taking presses: where the mouse can point.
pub type Hit = ([i32; 4], usize);

impl HudState {
    pub fn of(p: &Player, menu: Option<&Menu>, width: u32, pad: bool, yaw: f32, hands: Vec<(Kind, Pose)>) -> HudState {
        HudState {
            width,
            hp: (p.hp, p.max_hp),
            mp: (p.mp, p.max_mp),
            stamina: (p.stamina.ceil() as i32, STAMINA as i32),
            clarity: (p.clarity / CLARITY_SECS * CLARITY_POINTS as f32).ceil() as i32,
            slots: p.hotbar.map(|s| s.map(|i| (i, p.count(i)))),
            pad,
            heading: (yaw / std::f32::consts::TAU * HEADINGS as f32).round().rem_euclid(HEADINGS as f32) as i32,
            hands,
            drops: Vec::new(),
            toast: None,
            menu: menu.map(|m| MenuShown {
                menu: m.clone(),
                items: p.items.clone(),
                hotbar: p.hotbar,
                equipped: p.equipped,
                stats: p.stats(),
                choices: match m.screen {
                    Screen::Inventory { window: Some(Window::Item(w)), .. } => menu::choices(p, w.item),
                    _ => Vec::new(),
                },
                wears: match m.screen {
                    Screen::Inventory { window: Some(Window::Gear(w)), .. } => menu::gear_choices(p, w.at),
                    _ => Vec::new(),
                },
                preview: match m.screen {
                    Screen::Inventory { window: Some(Window::Gear(w)), .. } => {
                        menu::gear_choices(p, w.at).get(w.row).map(|&wear| p.stats_with(w.at, wear.map(|w| w.item)))
                    }
                    _ => None,
                },
            }),
        }
    }
}

/// How many of a piece are carried, and for what wears how much wear it has left:
/// "x3", "31/45", "x2 45/45"; a broken one's name says so.
fn amount(piece: Piece, n: u32) -> String {
    match piece.wear() {
        Some(_) if piece.broken() => if n > 1 { format!("x{n}") } else { String::new() },
        Some((left, of)) if n > 1 => format!("x{n} {left}/{of}"),
        Some((left, of)) => format!("{left}/{of}"),
        None => format!("x{n}"),
    }
}

/// The colour a piece's name is written in.
fn named(piece: Piece) -> Rgba {
    if piece.broken() { BROKEN } else { WHITE }
}

/// An item's name as a title: "Healing potion".
fn title(name: &str) -> String {
    let mut c = name.chars();
    c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
}

/// `text` broken at spaces into lines of at most `width` pixels.
fn wrap(text: &str, width: i32) -> Vec<String> {
    let most = (width / font::GLYPH_W as i32).max(1) as usize;
    let mut lines: Vec<String> = Vec::new();
    for word in text.split(' ') {
        match lines.last_mut() {
            Some(l) if l.len() + 1 + word.len() <= most => {
                l.push(' ');
                l.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

/// RGBA rows `stride` pixels wide, of which `w` by `h` are in use.
struct Canvas<'a> {
    data: &'a mut [u8],
    stride: i32,
    w: i32,
    h: i32,
    /// The rows of the list taking presses, as they're drawn.
    hits: Vec<Hit>,
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

    /// A piece's icon, cracked across if it's broken.
    fn piece_icon(&mut self, x: i32, y: i32, piece: Piece) {
        self.icon(x, y, piece.item);
        if piece.broken() {
            self.line((x as f32 + 2.0, y as f32 + 4.0), (x as f32 + 14.0, y as f32 + 12.0), BROKEN);
            self.line((x as f32 + 2.0, y as f32 + 5.0), (x as f32 + 13.0, y as f32 + 13.0), BROKEN);
        }
    }

    fn icon(&mut self, x: i32, y: i32, item: Item) {
        match item {
            Item::HealingPotion => self.flask(x, y, HP),
            Item::ManaPotion => self.flask(x, y, MP),
            Item::ClearSeeingPotion => self.flask(x, y, CLARITY),
            Item::CalmingPotion => self.flask(x, y, CALMING),
            Item::Sword => {
                self.line((x as f32 + 3.0, y as f32 + 12.0), (x as f32 + 13.0, y as f32 + 2.0), STEEL);
                self.line((x as f32 + 4.0, y as f32 + 12.0), (x as f32 + 13.0, y as f32 + 3.0), STEEL);
                self.line((x as f32 + 2.0, y as f32 + 9.0), (x as f32 + 6.0, y as f32 + 13.0), WOOD);
                self.fill(x + 1, y + 13, 3, 2, WOOD);
            }
            Item::Buckler => {
                for i in 0..48 {
                    let t = i as f32 / 48.0 * std::f32::consts::TAU;
                    for r in [5.5, 6.5] {
                        self.put((x as f32 + 8.0 + t.cos() * r).round() as i32, (y as f32 + 8.0 + t.sin() * r).round() as i32, STEEL);
                    }
                }
                self.fill(x + 4, y + 4, 9, 9, WOOD);
                self.fill(x + 7, y + 7, 3, 3, STEEL);
            }
            Item::LeatherCap => {
                self.fill(x + 4, y + 5, 8, 3, LEATHER);
                self.fill(x + 3, y + 8, 10, 4, LEATHER);
                self.fill(x + 2, y + 12, 12, 2, [90, 60, 35, 255]);
            }
            Item::LeatherVest => {
                self.fill(x + 3, y + 3, 10, 12, LEATHER);
                self.fill(x + 1, y + 3, 2, 5, LEATHER);
                self.fill(x + 13, y + 3, 2, 5, LEATHER);
                self.fill(x + 7, y + 3, 2, 3, [0, 0, 0, 0]);
                self.fill(x + 7, y + 7, 2, 7, [90, 60, 35, 255]);
            }
            Item::WorkGloves => {
                self.fill(x + 4, y + 6, 8, 8, LEATHER);
                for f in 0..4 {
                    self.fill(x + 4 + f * 2, y + 2 + (f == 0 || f == 3) as i32, 1, 4, LEATHER);
                }
                self.fill(x + 12, y + 7, 2, 3, LEATHER);
            }
            Item::CopperRing => {
                for i in 0..40 {
                    let t = i as f32 / 40.0 * std::f32::consts::TAU;
                    self.put((x as f32 + 8.0 + t.cos() * 4.0).round() as i32, (y as f32 + 9.0 + t.sin() * 4.0).round() as i32, COPPER);
                }
                self.fill(x + 7, y + 3, 3, 2, CLARITY);
            }
        }
    }

    /// One hotbar slot at (x, y): its key's number on a column, and how many are left.
    fn slot(&mut self, x: i32, y: i32, key: Option<i32>, content: Option<(Item, u32)>) {
        self.slot_picked(x, y, key, content, false);
    }

    /// A slot, outlined in white when it's the one picked.
    fn slot_picked(&mut self, x: i32, y: i32, key: Option<i32>, content: Option<(Item, u32)>, picked: bool) {
        self.fill(x, y, SLOT, SLOT, if picked { PICKED } else { BACK });
        let edge = if picked { WHITE } else if content.is_some() { FRAME } else { DIM };
        self.outline(x, y, SLOT, SLOT, edge);
        if let Some((item, count)) = content {
            self.icon(x + 2, y + 2, item);
            let n = count.to_string();
            self.text(x + SLOT - 1 - font::text_width(&n) as i32, y + SLOT - font::GLYPH_H as i32, &n, WHITE);
        }
        if let Some(k) = key {
            self.text(x + 2, y + 1, &k.to_string(), FRAME);
        }
    }

    /// A short notice, in a panel across the top middle of the screen.
    fn toast(&mut self, text: &str) {
        let w = font::text_width(text) as i32 + 12;
        let x = (self.w - w) / 2;
        let y = MARGIN + 30;
        self.fill(x, y, w, 15, PANEL);
        self.outline(x, y, w, 15, FRAME);
        self.text(x + 6, y + 4, text, WHITE);
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

    /// A panel with a frame and a title.
    fn panel(&mut self, x: i32, y: i32, w: i32, h: i32, heading: &str) {
        self.fill(x, y, w, h, PANEL);
        self.outline(x, y, w, h, FRAME);
        if !heading.is_empty() {
            self.text(x + 6, y + 4, heading, FRAME);
        }
    }

    /// A menu row, marked when picked, and where the mouse can point at it if its list
    /// takes presses.
    fn row(&mut self, rect: [i32; 4], picked: bool, active: bool, index: usize) {
        let [x, y, w, h] = rect;
        if picked {
            self.fill(x, y, w, h, PICKED);
            self.outline(x, y, w, h, if active { FRAME } else { DIM });
        }
        if active {
            self.hits.push((rect, index));
        }
    }

    fn pause(&mut self, picked: usize) {
        let h = 18 + PAUSE_ROWS.len() as i32 * ROW + 4;
        let (x, y) = ((self.w - PAUSE_W) / 2, (self.h - h) / 2);
        self.panel(x, y, PAUSE_W, h, "Paused");
        for (i, label) in PAUSE_ROWS.iter().enumerate() {
            let ry = y + 18 + i as i32 * ROW;
            self.row([x + 3, ry, PAUSE_W - 6, ROW], i == picked, true, i);
            let ready = menu::ready(i);
            self.text(x + 10, ry + 3, label, if ready { WHITE } else { DIM });
            if !ready {
                self.text(x + PAUSE_W - 10 - font::text_width("not yet") as i32, ry + 3, "not yet", DIM);
            }
        }
    }

    fn inventory(&mut self, shown: &MenuShown, pad: bool, hp: (i32, i32), mp: (i32, i32)) {
        let Screen::Inventory { row, gear, window } = shown.menu.screen else { return };
        let (x, y, w, h) = (MARGIN, MARGIN, self.w - 2 * MARGIN, self.h - 2 * MARGIN);
        self.panel(x, y, w, h, "Items");
        let foot = y + h - 2 * font::LINE_H as i32 - 8;
        self.fill(x + 1, foot, w - 2, 1, DIM);

        // The items, on the left, wide enough at 320 for the longest name and its count.
        let split = x + w * 14 / 25;
        let (lx, top, lw) = (x + 4, y + 16, split - x - 8);
        let fits = ((foot - 2 - top) / ITEM_ROW).max(1) as usize;
        let first = (row + 1).saturating_sub(fits);
        if shown.items.is_empty() {
            self.text(lx + 6, top + 4, "Nothing carried", DIM);
        }
        for (i, &(piece, n)) in shown.items.iter().enumerate().skip(first).take(fits) {
            let ry = top + (i - first) as i32 * ITEM_ROW;
            // The mouse counts the items on from the equipment slots.
            self.row([lx, ry, lw, ITEM_ROW], i == row && gear.is_none(), window.is_none(), EQUIPMENT.len() + i);
            self.piece_icon(lx + 1, ry + 1, piece);
            if let Some(s) = shown.hotbar.iter().position(|s| *s == Some(piece.item)) {
                self.text(lx + 1, ry, &(s + 1).to_string(), FRAME);
            }
            self.text(lx + 20, ry + 5, &title(&piece.name()), named(piece));
            let count = amount(piece, n);
            self.text(lx + lw - 2 - font::text_width(&count) as i32, ry + 5, &count, WHITE);
        }

        // The player, what they have on and how they stand, on the right.
        let (rx, rw) = (split + 4, x + w - 8 - split);
        self.outline(rx, top, rw, 42, DIM);
        let model = "player model";
        self.text(rx + (rw - font::text_width(model) as i32) / 2, top + 17, model, DIM);
        let mut ry = top + 46;
        for (i, slot) in EQUIPMENT.iter().enumerate() {
            self.row([rx, ry, rw, GEAR_ROW], gear == Some(i), window.is_none(), i);
            self.text(rx + 2, ry + 2, slot.name(), DIM);
            let on = shown.equipped[i].map_or("-".to_string(), |p| title(p.item.name()));
            self.text(rx + 44, ry + 2, &on, WHITE);
            if let Some((left, of)) = shown.equipped[i].and_then(Piece::wear) {
                let wear = format!("{left}/{of}");
                self.text(rx + rw - 2 - font::text_width(&wear) as i32, ry + 2, &wear, DIM);
            }
            ry += GEAR_ROW;
        }
        ry += 3;
        self.fill(rx, ry, rw, 1, DIM);
        ry += 4;
        self.text(rx + 2, ry, &format!("HP {}/{}", hp.0, hp.1), HP_TEXT);
        self.text(rx + 2 + rw / 2, ry, &format!("MP {}/{}", mp.0, mp.1), MP_TEXT);
        let (now, then) = (shown.stats, shown.preview.unwrap_or(shown.stats));
        for (name, a, b) in [("Attack", now.attack, then.attack), ("Defence", now.defence, then.defence), ("Magic", now.magic, then.magic)] {
            ry += 11;
            self.text(rx + 2, ry, name, DIM);
            let (value, c) = match b.cmp(&a) {
                std::cmp::Ordering::Equal => (a.to_string(), WHITE),
                std::cmp::Ordering::Greater => (format!("{a} -> {b}"), BETTER),
                std::cmp::Ordering::Less => (format!("{a} -> {b}"), WORSE),
            };
            self.text(rx + rw - 2 - font::text_width(&value) as i32, ry, &value, c);
        }

        // What the item does, or what was last done.
        let picked: Option<Piece> = match (window, gear) {
            (Some(Window::Gear(w)), _) => shown.wears.get(w.row).copied().flatten(),
            (_, Some(g)) => shown.equipped[g],
            _ => shown.items.get(row).map(|s| s.0),
        };
        let about = |p: Piece| {
            if p.broken() { format!("{} Broken: the blacksmith can mend it.", p.item.about()) } else { p.item.about().to_string() }
        };
        let note = match (&shown.menu.note, window) {
            (_, Some(Window::Item(ItemWindow { picking: Some(_), .. }))) => Some(if pad {
                "Press the d-pad direction of a slot.".to_string()
            } else {
                "Press 1-4, or pick a slot.".to_string()
            }),
            (_, Some(Window::Gear(_))) => picked.map(about),
            (Some(note), _) => Some(title(note)),
            // How to put something usable on the hotbar.
            (None, None) if gear.is_none() && picked.is_some_and(|p| p.item.slot().is_none()) => {
                let how = if pad { "Square + d-pad" } else { "1-4" };
                picked.map(|p| format!("{} {how} puts it on the hotbar.", about(p)))
            }
            (None, _) => picked.map(about),
        };
        if let Some(note) = note {
            for (i, line) in wrap(&note, w - 12).iter().take(2).enumerate() {
                self.text(x + 6, foot + 5 + i as i32 * font::LINE_H as i32, line, WHITE);
            }
        }
        match window {
            Some(Window::Item(win)) => self.item_window(shown, win, pad),
            Some(Window::Gear(win)) => self.gear_window(shown, win),
            None => {}
        }
    }

    /// What could be worn in an equipment slot, with how many of each are carried.
    fn gear_window(&mut self, shown: &MenuShown, win: GearWindow) {
        let rows = shown.wears.len().max(1) as i32;
        let h = 18 + rows * ITEM_ROW + 4;
        let (x, y) = ((self.w - WINDOW_W) / 2, (self.h - h) / 2);
        self.panel(x, y, WINDOW_W, h, EQUIPMENT[win.at].title());
        if shown.wears.is_empty() {
            self.text(x + 10, y + 18 + 5, "Nothing to wear", DIM);
        }
        for (i, wear) in shown.wears.iter().enumerate() {
            let ry = y + 18 + i as i32 * ITEM_ROW;
            self.row([x + 3, ry, WINDOW_W - 6, ITEM_ROW], i == win.row, true, i);
            match wear {
                Some(piece) => {
                    self.piece_icon(x + 4, ry + 1, *piece);
                    self.text(x + 23, ry + 5, &title(&piece.name()), WHITE);
                    let n = shown.items.iter().find(|s| s.0 == *piece).map_or(0, |s| s.1);
                    let count = amount(*piece, n);
                    self.text(x + WINDOW_W - 6 - font::text_width(&count) as i32, ry + 5, &count, WHITE);
                }
                None => self.text(x + 23, ry + 5, "Take off", WHITE),
            }
        }
    }

    fn item_window(&mut self, shown: &MenuShown, win: ItemWindow, pad: bool) {
        let step = SLOT + GAP;
        let h = match win.picking {
            None => 18 + shown.choices.len() as i32 * ROW + 4,
            Some(_) if pad => 18 + 3 * step + 4,
            Some(_) => 18 + SLOT + 6,
        };
        let (x, y) = ((self.w - WINDOW_W) / 2, (self.h - h) / 2);
        self.panel(x, y, WINDOW_W, h, &title(&win.item.name()));
        let count = |item: Item| shown.items.iter().filter(|s| s.0.item == item).map(|s| s.1).sum::<u32>();
        let content = |s: usize| shown.hotbar[s].map(|i| (i, count(i)));
        match win.picking {
            None => {
                for (i, choice) in shown.choices.iter().enumerate() {
                    let ry = y + 18 + i as i32 * ROW;
                    self.row([x + 3, ry, WINDOW_W - 6, ROW], i == win.row, true, i);
                    self.text(x + 10, ry + 3, choice.label(), WHITE);
                }
            }
            Some(picked) => {
                let at: Vec<(i32, i32)> = if pad {
                    // Up, right, down, left, as on the d-pad and the HUD.
                    let (cx, cy) = (x + (WINDOW_W - SLOT) / 2, y + 18 + step);
                    vec![(cx, cy - step), (cx + step, cy), (cx, cy + step), (cx - step, cy)]
                } else {
                    let x0 = x + (WINDOW_W - (SLOTS as i32 * step - GAP)) / 2;
                    (0..SLOTS as i32).map(|i| (x0 + i * step, y + 18)).collect()
                };
                for (s, (sx, sy)) in at.into_iter().enumerate() {
                    let key = (!pad).then_some(s as i32 + 1);
                    self.slot_picked(sx, sy, key, content(s), s == picked);
                    self.hits.push(([sx, sy, SLOT, SLOT], s));
                }
            }
        }
    }
}

/// Draw the HUD over a cleared image of `stride` by `h` RGBA pixels, and any menu over it;
/// gives where the mouse can point in the menu.
pub fn draw(data: &mut [u8], stride: u32, h: u32, state: &HudState) -> Vec<Hit> {
    data.fill(0);
    let mut c = Canvas { data, stride: stride as i32, w: state.width as i32, h: h as i32, hits: Vec::new() };
    // What lies in the world, then the hands, so the HUD stays readable over them.
    for t in &state.drops {
        c.triangle(t);
    }
    for t in hands::triangles(&state.hands, state.width, h) {
        c.triangle(&t);
    }
    c.bar(MARGIN + 1, MARGIN + 1, 5, state.hp, HP);
    c.bar(MARGIN + 1, MARGIN + 9, 3, state.mp, MP);
    c.bar(MARGIN + 1, MARGIN + 15, 3, state.stamina, STAMINA_BAR);
    if state.clarity > 0 {
        c.bar(MARGIN + 1, MARGIN + 21, 2, (state.clarity, CLARITY_POINTS), CLARITY);
    }
    c.compass(state.heading);
    c.hotbar(state);
    c.spell();
    if let (Some(toast), None) = (&state.toast, &state.menu) {
        c.toast(toast);
    }
    if let Some(shown) = &state.menu {
        match shown.menu.screen {
            Screen::Pause { row } => c.pause(row),
            Screen::Inventory { .. } => c.inventory(shown, state.pad, state.hp, state.mp),
        }
    }
    c.hits
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

    fn held() -> Vec<(Kind, Pose)> {
        let mut h = crate::hands::Hands::default();
        h.hold(Some(Item::Sword), None);
        h.shown(0.0)
    }

    fn state() -> HudState {
        HudState::of(&Player::new(), None, W, false, 0.0, held())
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
    fn the_stamina_bar_runs_down() {
        let mut s = state();
        assert_eq!(at(&drawn(&s), 9 + 70, 24), STAMINA_BAR);
        s.stamina.0 = 10;
        let low = drawn(&s);
        assert_eq!(at(&low, 9 + 10, 24), STAMINA_BAR);
        assert_ne!(at(&low, 9 + 30, 24), STAMINA_BAR);
    }

    #[test]
    fn the_clarity_bar_shows_only_while_it_lasts() {
        let mut s = state();
        assert_ne!(at(&drawn(&s), 9 + 2, 30), CLARITY);
        s.clarity = CLARITY_POINTS / 2;
        let half = drawn(&s);
        assert_eq!(at(&half, 9 + 2, 30), CLARITY);
        assert_ne!(at(&half, 9 + CLARITY_POINTS as u32 + 4, 30), CLARITY);
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

    fn menu_state(screen: Screen, width: u32, pad: bool) -> HudState {
        let p = Player::new();
        let menu = Menu { screen, note: None };
        HudState::of(&p, Some(&menu), width, pad, 0.0, held())
    }

    fn hits(state: &HudState) -> Vec<Hit> {
        let mut data = vec![0; (MAX * H * 4) as usize];
        draw(&mut data, MAX, H, state)
    }

    const MAX: u32 = 576;

    #[test]
    fn the_pause_menu_rows_are_where_the_mouse_can_point() {
        let rows = hits(&menu_state(Screen::Pause { row: 0 }, W, false));
        assert_eq!(rows.len(), PAUSE_ROWS.len());
        assert!(rows.windows(2).all(|r| r[1].0[1] > r[0].0[1]));
    }

    #[test]
    fn only_the_list_on_top_takes_the_mouse() {
        let items = Player::new().items.len();
        let list = hits(&menu_state(Screen::Inventory { row: 0, gear: None, window: None }, W, false));
        assert_eq!(list.len(), EQUIPMENT.len() + items);
        assert!(list.iter().any(|h| h.1 == EQUIPMENT.len() + items - 1));
        let window = Some(Window::Item(ItemWindow { item: Item::HealingPotion.into(), row: 0, picking: None }));
        assert_eq!(hits(&menu_state(Screen::Inventory { row: 0, gear: None, window }, W, false)).len(), 4);
        for pad in [false, true] {
            let picking = Some(Window::Item(ItemWindow { item: Item::HealingPotion.into(), row: 1, picking: Some(0) }));
            assert_eq!(hits(&menu_state(Screen::Inventory { row: 0, gear: None, window: picking }, W, pad)).len(), SLOTS);
        }
        let gear = Some(Window::Gear(GearWindow { at: 5, row: 0 }));
        assert_eq!(hits(&menu_state(Screen::Inventory { row: 0, gear: Some(5), window: gear }, W, false)).len(), 1);
    }

    #[test]
    fn the_inventory_fits_the_narrowest_and_widest_frames() {
        for width in [W, MAX] {
            let rows = hits(&menu_state(Screen::Inventory { row: 0, gear: None, window: None }, width, false));
            let (gear, items): (Vec<&Hit>, Vec<&Hit>) = rows.iter().partition(|h| h.1 < EQUIPMENT.len());
            let inside = |([x, y, w, h], _): &&Hit| *x >= 0 && *y >= 0 && x + w <= width as i32 - MARGIN && y + h <= H as i32 - 30;
            assert!(rows.iter().all(|h| inside(&h)));
            let list_right = items.iter().map(|h| h.0[0] + h.0[2]).max().unwrap();
            assert!(gear.iter().all(|h| h.0[0] > list_right));
        }
    }

    #[test]
    fn a_stat_shows_how_it_would_change() {
        let p = Player::new();
        let menu = Menu { screen: Screen::Inventory { row: 0, gear: Some(0), window: Some(Window::Gear(GearWindow { at: 0, row: 0 })) }, note: None };
        let s = HudState::of(&p, Some(&menu), W, false, 0.0, held());
        // The right hand offers only taking the sword off.
        assert_eq!(s.menu.as_ref().unwrap().preview, Some(Stats { attack: 0, ..p.stats() }));
        let d = drawn(&s);
        assert!((0..W).any(|x| (0..H).any(|y| at(&d, x, y) == WORSE)));
    }

    #[test]
    fn long_text_wraps_at_spaces() {
        assert_eq!(wrap("see the dungeon as it is", 60), ["see the", "dungeon as", "it is"]);
        assert_eq!(title("healing potion"), "Healing potion");
    }

    #[test]
    fn the_heading_wraps_round() {
        let p = Player::new();
        let s = |yaw: f32| HudState::of(&p, None, W, false, yaw, held()).heading;
        assert_eq!(s(std::f32::consts::TAU), s(0.0));
        assert_eq!(s(-std::f32::consts::FRAC_PI_2), HEADINGS * 3 / 4);
    }

    #[test]
    fn amounts_say_how_many_and_how_worn() {
        assert_eq!(amount(Item::HealingPotion.into(), 3), "x3");
        assert_eq!(amount(Piece { item: Item::Sword, left: 31 }, 1), "31/45");
        assert_eq!(amount(Item::Sword.into(), 2), "x2 45/45");
        assert_eq!(amount(Piece { item: Item::Sword, left: 0 }, 1), "");
        assert_eq!(amount(Piece { item: Item::Sword, left: 0 }, 2), "x2");
    }
}
