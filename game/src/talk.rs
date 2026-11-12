//! Talking to people (`engine::world::person`). The player speaks to whoever is close in
//! front of them; what they say comes a box of speech at a time, typed out a letter at a
//! time (faster while the button is held) in a bubble over their head, their mouth moving
//! while it types. A press shows the rest of the line at once, or goes on to the next;
//! after the last the talk is over. Someone with nothing written for them makes small talk.

use engine::world::rng::Rng;
use engine::world::scene::{Camera, Scene, T};

const TF: f32 = T as f32;
/// How near someone must be to be spoken to, and how far off straight ahead (radians).
pub const REACH: f32 = 1.5 * TF;
const OFF_AHEAD: f32 = 0.7;
/// Letters typed a second, and how much faster while the button is held.
const LETTERS: f32 = 28.0;
const HURRY: f32 = 4.0;

/// A talk under way.
pub struct Talk {
    pub person: usize,
    pub name: Option<String>,
    lines: Vec<String>,
    line: usize,
    /// Letters of the line typed so far.
    typed: f32,
}

impl Talk {
    /// Speaking to the scene's person `i`.
    pub fn with(scene: &Scene, i: usize) -> Talk {
        let p = &scene.people[i];
        let lines = if p.lines.is_empty() { small_talk(p.seed) } else { p.lines.clone() };
        Talk { person: i, name: p.name.clone(), lines, line: 0, typed: 0.0 }
    }

    fn whole(&self) -> &str {
        &self.lines[self.line]
    }

    /// Type on for `dt` seconds, faster with `hurry`.
    pub fn update(&mut self, dt: f32, hurry: bool) {
        let most = self.whole().chars().count() as f32;
        self.typed = (self.typed + dt * LETTERS * if hurry { HURRY } else { 1.0 }).min(most);
    }

    /// Whether the line is still being typed (the speaker's mouth moves).
    pub fn typing(&self) -> bool {
        (self.typed as usize) < self.whole().chars().count()
    }

    /// The button: show the rest of the line, or go on to the next. False once there's no
    /// more to say.
    pub fn press(&mut self) -> bool {
        if self.typing() {
            self.typed = self.whole().chars().count() as f32;
            return true;
        }
        self.line += 1;
        self.typed = 0.0;
        self.line < self.lines.len()
    }

    /// The line as typed so far.
    #[cfg(test)]
    pub fn shown(&self) -> String {
        self.whole().chars().take(self.typed as usize).collect()
    }

    /// The bubble showing it, its tail toward `at` (the speaker's head on the screen, if
    /// it's in view).
    pub fn bubble(&self, at: Option<(i32, i32)>) -> crate::hud::Bubble {
        crate::hud::Bubble {
            name: self.name.clone(),
            line: self.whole().to_string(),
            typed: self.typed as usize,
            at,
            more: !self.typing() && self.more(),
        }
    }

    /// Whether there's another line after this one.
    pub fn more(&self) -> bool {
        self.line + 1 < self.lines.len()
    }
}

/// The person close in front of feet at `feet` facing `yaw` (radians), nearest first.
pub fn speaker(scene: &Scene, feet: [f32; 3], yaw: f32) -> Option<usize> {
    scene
        .people
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let b = p.body.base;
            let far = (b[0] - feet[0]).hypot(b[2] - feet[2]);
            let ok = far < REACH && (b[1] - feet[1]).abs() < TF && crate::foes::off_ahead(feet, yaw, b) < OFF_AHEAD;
            ok.then_some((i, far))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Where world point `p` falls on a frame `width` pixels wide (240 high) as `cam` sees it,
/// if it's in front of the eye.
pub fn on_screen(cam: &Camera, width: u32, p: [f32; 3]) -> Option<(i32, i32)> {
    let eye = cam.pos.map(|c| c as f32);
    let rows = engine::world::view::view_matrix(cam).m.map(|r| r.map(|e| e as f32 / 4096.0));
    let d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
    let v = rows.map(|r| r[0] * d[0] + r[1] * d[1] + r[2] * d[2]);
    if v[2] < 8.0 {
        return None;
    }
    let f = cam.h as f32;
    Some(((width as f32 / 2.0 + v[0] / v[2] * f).round() as i32, (120.0 + v[1] / v[2] * f).round() as i32))
}

/// Something to say to a stranger, picked by `seed`.
pub fn small_talk(seed: u64) -> Vec<String> {
    const SAID: [&[&str]; 8] = [
        &["Oh! Hello there.", "Lovely weather today. Not a cloud over the castle."],
        &["Have you come from the dungeon?", "You look tired. You should rest a while before you go down again."],
        &["The bread at the market is fresh this morning.", "Well, fresh-ish."],
        &["They say the queen hasn't slept since the thing was taken.", "I hope someone finds it soon."],
        &["Mind the well. My cousin dropped a boot down it once.", "We never got it back."],
        &["Hm? Sorry, I was miles away."],
        &["Strange dreams lately. Everyone's been having them.", "Walls that breathe. Eyes in the stone. Silly, isn't it?"],
        &["If you're going below, buy a potion or two first.", "The ones that make you see clearly. Trust me."],
    ];
    let mut rng = Rng::new(seed ^ 0x7A1C);
    rng.pick(&SAID).iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn talk(lines: &[&str]) -> Talk {
        Talk { person: 0, name: None, lines: lines.iter().map(|s| s.to_string()).collect(), line: 0, typed: 0.0 }
    }

    #[test]
    fn lines_type_out_and_a_press_finishes_one_then_goes_on() {
        let mut t = talk(&["Hello there.", "Bye."]);
        assert_eq!(t.shown(), "");
        t.update(0.1, false);
        let some = t.shown();
        assert!(!some.is_empty() && some.len() < 12 && t.typing() && t.more());
        t.update(0.05, true);
        assert!(t.shown().len() > some.len() * 3 && t.typing());
        // A press shows the rest, the next goes on.
        assert!(t.press());
        assert_eq!(t.shown(), "Hello there.");
        assert!(!t.typing());
        assert!(t.press());
        assert_eq!(t.shown(), "");
        assert!(!t.more());
        t.update(1.0, false);
        assert_eq!(t.shown(), "Bye.");
        assert!(!t.press(), "that was the last");
    }

    #[test]
    fn everyone_has_something_to_say() {
        for seed in 0..50 {
            let said = small_talk(seed);
            assert!(!said.is_empty() && said.iter().all(|l| !l.is_empty() && l.len() < 120));
        }
    }
}
