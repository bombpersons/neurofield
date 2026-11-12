//! Talking to people (`engine::world::person`). The player speaks to whoever is close in
//! front of them; what they say comes a box of speech at a time, typed out a letter at a
//! time (faster while the button is held) in a bubble over their head, their mouth moving
//! while it types. A press shows the rest of the line at once, or goes on to the next;
//! after the last the talk is over. Someone with nothing written for them makes small talk.
//! Whoever keeps the ledger ends by asking whether to save, the answers under the question
//! to pick from; what they say to the answer comes after it.

use engine::world::rng::Rng;
use engine::world::scene::{Camera, Scene, T};

const TF: f32 = T as f32;
/// How near someone must be to be spoken to, and how far off straight ahead (radians).
pub const REACH: f32 = 1.5 * TF;
const OFF_AHEAD: f32 = 0.7;
/// Letters typed a second, and how much faster while the button is held.
const LETTERS: f32 = 28.0;
const HURRY: f32 = 4.0;
/// What a keeper of the ledger asks, and the answers.
pub const SAVE_QUESTION: &str = "Do you want to save?";
const YES_NO: [&str; 2] = ["Yes", "No"];

/// What a press of the button did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Said {
    /// The line was shown whole, or the next begun.
    More,
    /// There's no more to say.
    Over,
    /// The question was answered with answer `n`; `reply` goes on.
    Answered(usize),
}

/// A talk under way.
pub struct Talk {
    pub person: usize,
    pub name: Option<String>,
    lines: Vec<String>,
    line: usize,
    /// Letters of the line typed so far.
    typed: f32,
    /// The answers to the last line when it's a question, and which is picked.
    answers: Vec<String>,
    pub picked: usize,
}

impl Talk {
    /// Speaking to the scene's person `i`.
    pub fn with(scene: &Scene, i: usize) -> Talk {
        let p = &scene.people[i];
        let mut lines = if p.lines.is_empty() { small_talk(p.seed) } else { p.lines.clone() };
        let mut answers = Vec::new();
        if p.saves {
            lines.push(SAVE_QUESTION.to_string());
            answers = YES_NO.map(String::from).to_vec();
        }
        Talk { person: i, name: p.name.clone(), lines, line: 0, typed: 0.0, answers, picked: 0 }
    }

    /// Whether the question is being asked: its answers are up to pick from once it's typed.
    pub fn asking(&self) -> bool {
        !self.answers.is_empty() && self.line + 1 == self.lines.len()
    }

    /// Move the pick `by` answers along, round from the end.
    pub fn pick(&mut self, by: i32) {
        if self.asking() && !self.typing() {
            let n = self.answers.len() as i32;
            self.picked = (self.picked as i32 + by).rem_euclid(n) as usize;
        }
    }

    /// Say `line` next, after the answer to the question.
    pub fn reply(&mut self, line: &str) {
        self.lines.push(line.to_string());
        self.line += 1;
        self.typed = 0.0;
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

    /// The button: show the rest of the line, answer the question, or go on to the next.
    pub fn press(&mut self) -> Said {
        if self.typing() {
            self.typed = self.whole().chars().count() as f32;
            return Said::More;
        }
        if self.asking() {
            self.answers.clear();
            return Said::Answered(self.picked);
        }
        self.line += 1;
        self.typed = 0.0;
        if self.line < self.lines.len() { Said::More } else { Said::Over }
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
            answers: if self.asking() && !self.typing() { self.answers.clone() } else { Vec::new() },
            picked: self.picked,
        }
    }

    /// Whether there's another line after this one.
    pub fn more(&self) -> bool {
        self.line + 1 < self.lines.len() && !self.asking()
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
        Talk { person: 0, name: None, lines: lines.iter().map(|s| s.to_string()).collect(), line: 0, typed: 0.0, answers: Vec::new(), picked: 0 }
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
        assert_eq!(t.press(), Said::More);
        assert_eq!(t.shown(), "Hello there.");
        assert!(!t.typing());
        assert_eq!(t.press(), Said::More);
        assert_eq!(t.shown(), "");
        assert!(!t.more());
        t.update(1.0, false);
        assert_eq!(t.shown(), "Bye.");
        assert_eq!(t.press(), Said::Over, "that was the last");
    }

    #[test]
    fn the_keeper_of_the_ledger_asks_to_save_and_answers_the_answer() {
        let mut t = talk(&["Hello.", SAVE_QUESTION]);
        t.answers = YES_NO.map(String::from).to_vec();
        t.update(1.0, false);
        assert!(!t.asking() && t.bubble(None).answers.is_empty());
        assert_eq!(t.press(), Said::More);
        assert!(t.asking() && !t.more());
        // No answers until the question's typed; picking waits for them.
        assert!(t.bubble(None).answers.is_empty());
        t.pick(1);
        assert_eq!(t.picked, 0);
        t.update(1.0, false);
        assert_eq!(t.bubble(None).answers, ["Yes", "No"]);
        t.pick(1);
        t.pick(1);
        t.pick(-1);
        assert_eq!(t.picked, 1);
        assert_eq!(t.press(), Said::Answered(1));
        t.reply("Another time, then.");
        assert!(!t.asking() && t.bubble(None).answers.is_empty());
        t.update(1.0, false);
        assert_eq!(t.shown(), "Another time, then.");
        assert_eq!(t.press(), Said::Over);
    }

    #[test]
    fn everyone_has_something_to_say() {
        for seed in 0..50 {
            let said = small_talk(seed);
            assert!(!said.is_empty() && said.iter().all(|l| !l.is_empty() && l.len() < 120));
        }
    }
}
