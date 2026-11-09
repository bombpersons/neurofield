//! The player's state: health, mana, what they carry and what is on the hotbar. No Bevy
//! here, so the rules can be tested alone.

/// Every kind of item. Its use is in `Player::use_item`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    HealingPotion,
    ManaPotion,
    ClarityPotion,
}

impl Item {
    pub fn name(self) -> &'static str {
        match self {
            Item::HealingPotion => "healing potion",
            Item::ManaPotion => "mana potion",
            Item::ClarityPotion => "clarity potion",
        }
    }
}

pub const SLOTS: usize = 4;
const HEAL: i32 = 15;
const RESTORE: i32 = 6;
/// How fast affliction rises by itself, per second: from nothing to all in ten minutes.
const AFFLICTION_RISE: f32 = 1.0 / 600.0;
/// How long a clarity potion shows the dungeon as it is, in seconds, and what it lifts.
pub const CLARITY_SECS: f32 = 20.0;
const CLARITY_EASE: f32 = 0.25;

pub struct Player {
    pub hp: i32,
    pub max_hp: i32,
    pub mp: i32,
    pub max_mp: i32,
    /// How far gone the player is, 0..1: it makes the dungeon stranger around them.
    pub affliction: f32,
    /// Seconds left of seeing the dungeon as it is.
    pub clarity: f32,
    /// What is carried and how many, in the order first picked up.
    pub items: Vec<(Item, u32)>,
    pub hotbar: [Option<Item>; SLOTS],
}

impl Player {
    pub fn new() -> Player {
        let mut p = Player { hp: 40, max_hp: 40, mp: 12, max_mp: 12, affliction: 0.0, clarity: 0.0, items: Vec::new(), hotbar: [None; SLOTS] };
        p.add(Item::HealingPotion, 3);
        p.add(Item::ManaPotion, 2);
        p.add(Item::ClarityPotion, 2);
        p
    }

    pub fn count(&self, item: Item) -> u32 {
        self.items.iter().find(|s| s.0 == item).map_or(0, |s| s.1)
    }

    /// Carry `n` more of `item`, on the first free hotbar slot if it isn't on one.
    pub fn add(&mut self, item: Item, n: u32) {
        match self.items.iter_mut().find(|s| s.0 == item) {
            Some(s) => s.1 += n,
            None => self.items.push((item, n)),
        }
        if !self.hotbar.contains(&Some(item)) {
            if let Some(free) = self.hotbar.iter_mut().find(|s| s.is_none()) {
                *free = Some(item);
            }
        }
    }

    /// Lose one of `item`, and its hotbar slot with the last of them.
    fn remove_one(&mut self, item: Item) {
        if let Some(i) = self.items.iter().position(|s| s.0 == item) {
            self.items[i].1 -= 1;
            if self.items[i].1 == 0 {
                self.items.remove(i);
                self.hotbar.iter_mut().filter(|s| **s == Some(item)).for_each(|s| *s = None);
            }
        }
    }

    pub fn hurt(&mut self, n: i32) {
        self.hp = (self.hp - n).max(0);
    }

    pub fn spend_mp(&mut self, n: i32) {
        self.mp = (self.mp - n).max(0);
    }

    pub fn afflict(&mut self, n: f32) {
        self.affliction = (self.affliction + n).clamp(0.0, 1.0);
    }

    /// `dt` seconds pass: clarity runs out, and while it lasts affliction doesn't rise.
    pub fn update(&mut self, dt: f32) {
        if self.clarity > 0.0 {
            self.clarity = (self.clarity - dt).max(0.0);
        } else {
            self.afflict(AFFLICTION_RISE * dt);
        }
    }

    /// Use the item on hotbar slot `slot`; says what happened, `None` for an empty slot.
    pub fn use_slot(&mut self, slot: usize) -> Option<String> {
        let item = self.hotbar[slot]?;
        Some(self.use_item(item))
    }

    /// Use one of `item`. One that would do nothing isn't used up.
    fn use_item(&mut self, item: Item) -> String {
        match item {
            Item::HealingPotion if self.hp >= self.max_hp => "you are not hurt".into(),
            Item::HealingPotion => {
                self.hp = (self.hp + HEAL).min(self.max_hp);
                self.remove_one(item);
                "you drink the healing potion".into()
            }
            Item::ManaPotion if self.mp >= self.max_mp => "your mind is already clear".into(),
            Item::ManaPotion => {
                self.mp = (self.mp + RESTORE).min(self.max_mp);
                self.remove_one(item);
                "you drink the mana potion".into()
            }
            Item::ClarityPotion if self.clarity > 0.0 => "you already see clearly".into(),
            Item::ClarityPotion => {
                self.clarity = CLARITY_SECS;
                self.afflict(-CLARITY_EASE);
                self.remove_one(item);
                "the dream lifts".into()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_potions_on_the_hotbar() {
        let p = Player::new();
        assert_eq!(p.hotbar, [Some(Item::HealingPotion), Some(Item::ManaPotion), Some(Item::ClarityPotion), None]);
        assert_eq!(p.count(Item::HealingPotion), 3);
    }

    #[test]
    fn a_potion_heals_up_to_the_most_and_is_used_up() {
        let mut p = Player::new();
        p.hurt(5);
        assert!(p.use_slot(0).is_some());
        assert_eq!(p.hp, p.max_hp);
        assert_eq!(p.count(Item::HealingPotion), 2);
    }

    #[test]
    fn a_potion_that_would_do_nothing_is_kept() {
        let mut p = Player::new();
        assert_eq!(p.use_slot(0).as_deref(), Some("you are not hurt"));
        assert_eq!(p.use_slot(1).as_deref(), Some("your mind is already clear"));
        assert_eq!(p.count(Item::HealingPotion), 3);
        assert_eq!(p.count(Item::ManaPotion), 2);
    }

    #[test]
    fn the_last_one_leaves_the_hotbar_and_picking_more_up_returns_it() {
        let mut p = Player::new();
        for _ in 0..2 {
            p.spend_mp(12);
            p.use_slot(1);
        }
        assert_eq!(p.count(Item::ManaPotion), 0);
        assert_eq!(p.hotbar[1], None);
        assert_eq!(p.use_slot(1), None);
        p.add(Item::ManaPotion, 1);
        assert_eq!(p.hotbar[1], Some(Item::ManaPotion));
    }

    #[test]
    fn a_full_hotbar_still_carries_more() {
        let mut p = Player::new();
        p.hotbar = [Some(Item::HealingPotion); SLOTS];
        p.add(Item::ManaPotion, 1);
        assert_eq!(p.count(Item::ManaPotion), 3);
    }

    #[test]
    fn affliction_creeps_up_and_stops_at_all() {
        let mut p = Player::new();
        p.update(60.0);
        assert!((p.affliction - 0.1).abs() < 1e-5);
        p.update(6000.0);
        assert_eq!(p.affliction, 1.0);
    }

    #[test]
    fn clarity_lifts_affliction_holds_it_and_runs_out() {
        let mut p = Player::new();
        p.afflict(0.5);
        assert_eq!(p.use_slot(2).as_deref(), Some("the dream lifts"));
        assert_eq!(p.clarity, CLARITY_SECS);
        assert_eq!(p.count(Item::ClarityPotion), 1);
        assert!((p.affliction - 0.25).abs() < 1e-5);
        p.update(CLARITY_SECS - 1.0);
        assert!((p.affliction - 0.25).abs() < 1e-5);
        assert_eq!(p.clarity, 1.0);
        p.update(1.0);
        assert_eq!(p.clarity, 0.0);
        p.update(6.0);
        assert!(p.affliction > 0.25);
    }

    #[test]
    fn a_second_clarity_potion_is_kept_while_the_first_lasts() {
        let mut p = Player::new();
        p.use_slot(2);
        assert_eq!(p.use_slot(2).as_deref(), Some("you already see clearly"));
        assert_eq!(p.count(Item::ClarityPotion), 1);
    }

    #[test]
    fn affliction_never_goes_below_nothing() {
        let mut p = Player::new();
        p.use_slot(2);
        assert_eq!(p.affliction, 0.0);
    }

    #[test]
    fn health_and_mana_stop_at_zero() {
        let mut p = Player::new();
        p.hurt(1000);
        p.spend_mp(1000);
        assert_eq!((p.hp, p.mp), (0, 0));
    }
}
