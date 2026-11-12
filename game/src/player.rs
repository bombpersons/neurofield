//! The player's state: health, mana, what they carry and wear, and what is on the hotbar.
//! No Bevy here, so the rules can be tested alone.

/// Every kind of item. Its use is in `Player::use_item`; what can be worn has a `Slot`.
/// The gear is a stand-in until the real items are designed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    HealingPotion,
    ManaPotion,
    ClearSeeingPotion,
    CalmingPotion,
    Sword,
    Buckler,
    LeatherCap,
    LeatherVest,
    WorkGloves,
    CopperRing,
}

/// Where something is worn. Either hand takes a weapon or a shield.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    RightHand,
    LeftHand,
    Helmet,
    Armour,
    Gloves,
    Ring,
}

/// The equipment slots in order, as the inventory lists them: the hands first, two for rings.
pub const EQUIPMENT: [Slot; 7] = [Slot::RightHand, Slot::LeftHand, Slot::Helmet, Slot::Armour, Slot::Gloves, Slot::Ring, Slot::Ring];

impl Slot {
    pub fn name(self) -> &'static str {
        match self {
            Slot::RightHand => "Right",
            Slot::LeftHand => "Left",
            Slot::Helmet => "Helmet",
            Slot::Armour => "Armour",
            Slot::Gloves => "Gloves",
            Slot::Ring => "Ring",
        }
    }

    /// The name heading its window.
    pub fn title(self) -> &'static str {
        match self {
            Slot::RightHand => "Right hand",
            Slot::LeftHand => "Left hand",
            _ => self.name(),
        }
    }

    fn hand(self) -> bool {
        matches!(self, Slot::RightHand | Slot::LeftHand)
    }
}

/// Which hand: its equipment slot is `EQUIPMENT[side as usize]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Right,
    Left,
}

pub const SIDES: [Side; 2] = [Side::Right, Side::Left];

/// Attack, defence and magic, the player's own or what a piece of gear adds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub attack: i32,
    pub defence: i32,
    pub magic: i32,
}

impl std::ops::Add for Stats {
    type Output = Stats;
    fn add(self, o: Stats) -> Stats {
        Stats { attack: self.attack + o.attack, defence: self.defence + o.defence, magic: self.magic + o.magic }
    }
}

impl Item {
    pub const ALL: [Item; 10] = [
        Item::HealingPotion,
        Item::ManaPotion,
        Item::ClearSeeingPotion,
        Item::CalmingPotion,
        Item::Sword,
        Item::Buckler,
        Item::LeatherCap,
        Item::LeatherVest,
        Item::WorkGloves,
        Item::CopperRing,
    ];

    /// The item a level names (`engine::world::level::ITEMS`).
    pub fn named(name: &str) -> Option<Item> {
        Item::ALL.into_iter().find(|i| i.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Item::HealingPotion => "healing potion",
            Item::ManaPotion => "mana potion",
            Item::ClearSeeingPotion => "clear seeing potion",
            Item::CalmingPotion => "calming potion",
            Item::Sword => "sword",
            Item::Buckler => "buckler",
            Item::LeatherCap => "leather cap",
            Item::LeatherVest => "leather vest",
            Item::WorkGloves => "work gloves",
            Item::CopperRing => "copper ring",
        }
    }

    /// What it does, in a line or two of the inventory screen.
    pub fn about(self) -> &'static str {
        match self {
            Item::HealingPotion => "Heals 15 HP.",
            Item::ManaPotion => "Restores 6 MP.",
            Item::ClearSeeingPotion => "See the dungeon as it really is, for a little while.",
            Item::CalmingPotion => "Takes away all affliction.",
            Item::Sword => "A plain sword, well kept.",
            Item::Buckler => "A small round shield.",
            Item::LeatherCap => "Better than nothing.",
            Item::LeatherVest => "Stiff leather over the chest.",
            Item::WorkGloves => "Rough gloves that grip well.",
            Item::CopperRing => "It hums faintly.",
        }
    }

    /// Where it's worn, for gear: for what's held, the hand it goes in first.
    pub fn slot(self) -> Option<Slot> {
        match self {
            Item::Sword => Some(Slot::RightHand),
            Item::Buckler => Some(Slot::LeftHand),
            Item::LeatherCap => Some(Slot::Helmet),
            Item::LeatherVest => Some(Slot::Armour),
            Item::WorkGloves => Some(Slot::Gloves),
            Item::CopperRing => Some(Slot::Ring),
            _ => None,
        }
    }

    /// Whether it can be worn in `slot`: what's held goes in either hand.
    pub fn fits(self, slot: Slot) -> bool {
        self.slot().is_some_and(|s| s == slot || s.hand() && slot.hand())
    }

    pub fn is_shield(self) -> bool {
        self == Item::Buckler
    }

    /// What wearing it adds.
    pub fn stats(self) -> Stats {
        let s = |attack, defence, magic| Stats { attack, defence, magic };
        match self {
            Item::Sword => s(5, 0, 0),
            Item::Buckler => s(0, 2, 0),
            Item::LeatherCap => s(0, 1, 0),
            Item::LeatherVest => s(0, 3, 0),
            Item::WorkGloves => s(1, 1, 0),
            Item::CopperRing => s(0, 0, 1),
            _ => Stats::default(),
        }
    }
}

pub const SLOTS: usize = 4;
/// The most of one item that can be carried.
pub const MOST: u32 = 99;
const HEAL: i32 = 15;
const RESTORE: i32 = 6;
/// How fast affliction rises by itself, per second: from nothing to all in ten minutes.
const AFFLICTION_RISE: f32 = 1.0 / 600.0;
/// Stamina: the most, and how much comes back a second while no shield is held up.
pub const STAMINA: f32 = 40.0;
const RECOVER: f32 = 10.0;
/// How long a clear seeing potion lasts, in seconds: the plain picture shows for about ten
/// of them, the rest being the fades.
pub const CLARITY_SECS: f32 = 12.0;

pub struct Player {
    pub hp: i32,
    pub max_hp: i32,
    pub mp: i32,
    pub max_mp: i32,
    /// How far gone the player is, 0..1: it makes the dungeon stranger around them.
    pub affliction: f32,
    /// Seconds left of seeing the dungeon as it is.
    pub clarity: f32,
    /// What holding up a shield, being hit and attacking use up, 0..`STAMINA`.
    pub stamina: f32,
    /// What is carried and how many, in the order first picked up.
    pub items: Vec<(Item, u32)>,
    pub hotbar: [Option<Item>; SLOTS],
    /// What is worn, by `EQUIPMENT` slot; worn things aren't in `items`.
    pub equipped: [Option<Item>; EQUIPMENT.len()],
}

impl Player {
    pub fn new() -> Player {
        let mut p = Player {
            hp: 40,
            max_hp: 40,
            mp: 12,
            max_mp: 12,
            affliction: 0.0,
            clarity: 0.0,
            stamina: STAMINA,
            items: Vec::new(),
            hotbar: [None; SLOTS],
            equipped: [None; EQUIPMENT.len()],
        };
        p.equipped[0] = Some(Item::Sword);
        p.equipped[1] = Some(Item::Buckler);
        p.add(Item::HealingPotion, 3);
        p.add(Item::ManaPotion, 2);
        p.add(Item::ClearSeeingPotion, 2);
        p.add(Item::CalmingPotion, 1);
        p.add(Item::LeatherCap, 1);
        p.add(Item::LeatherVest, 1);
        p.add(Item::WorkGloves, 1);
        p.add(Item::CopperRing, 2);
        p
    }

    /// The player's own stats, with all they wear.
    pub fn stats(&self) -> Stats {
        self.equipped.iter().flatten().fold(Stats::default(), |s, i| s + i.stats())
    }

    /// The stats with `item` worn in equipment slot `at` instead of what is there.
    pub fn stats_with(&self, at: usize, item: Option<Item>) -> Stats {
        let mut p = self.equipped;
        p[at] = item;
        p.iter().flatten().fold(Stats::default(), |s, i| s + i.stats())
    }

    /// Wear one of `item` in equipment slot `at`, the one there going back into the bag.
    /// False if it doesn't go there, none is carried, or there's no room for the old one.
    pub fn equip(&mut self, item: Item, at: usize) -> bool {
        if !item.fits(EQUIPMENT[at]) || self.count(item) == 0 {
            return false;
        }
        if let Some(old) = self.equipped[at] {
            if old != item && self.count(old) >= MOST {
                return false;
            }
        }
        self.remove_one(item);
        if let Some(old) = self.equipped[at].replace(item) {
            self.add(old, 1);
        }
        true
    }

    /// Where `item` would go on: the first free slot it fits, its own first (a weapon's the
    /// right hand, a shield's the left), else its own.
    pub fn slot_for(&self, item: Item) -> Option<usize> {
        let mut fits: Vec<usize> = (0..EQUIPMENT.len()).filter(|&i| item.fits(EQUIPMENT[i])).collect();
        fits.sort_by_key(|&i| Some(EQUIPMENT[i]) != item.slot());
        fits.iter().copied().find(|&i| self.equipped[i].is_none()).or(fits.first().copied())
    }

    /// Take off what is in equipment slot `at`; false if there's nothing, or no room for it.
    pub fn unequip(&mut self, at: usize) -> bool {
        match self.equipped[at] {
            Some(item) if self.count(item) < MOST => {
                self.equipped[at] = None;
                self.add(item, 1);
                true
            }
            _ => false,
        }
    }

    /// Drop one of `item`: false if none is carried.
    pub fn drop_one(&mut self, item: Item) -> bool {
        let had = self.count(item) > 0;
        self.remove_one(item);
        had
    }

    /// What is held in a hand.
    pub fn hand(&self, side: Side) -> Option<Item> {
        self.equipped[side as usize]
    }

    /// The attack of a blow from the weapon in hand `side`: the other hand's weapon doesn't
    /// add to it.
    pub fn attack_with(&self, side: Side) -> i32 {
        let other = 1 - side as usize;
        self.stats_with(other, None).attack
    }

    /// Use up `n` stamina.
    pub fn tire(&mut self, n: f32) {
        self.stamina = (self.stamina - n).max(0.0);
    }

    pub fn count(&self, item: Item) -> u32 {
        self.items.iter().find(|s| s.0 == item).map_or(0, |s| s.1)
    }

    /// Carry up to `n` more of `item`, as many as there's room for up to `MOST`, and put
    /// what can be used on the first free hotbar slot if it isn't on one. Gives how many
    /// there was no room for.
    pub fn add(&mut self, item: Item, n: u32) -> u32 {
        let took = n.min(MOST - self.count(item));
        if took == 0 {
            return n;
        }
        match self.items.iter_mut().find(|s| s.0 == item) {
            Some(s) => s.1 += took,
            None => self.items.push((item, took)),
        }
        if item.slot().is_none() && !self.hotbar.contains(&Some(item)) {
            if let Some(free) = self.hotbar.iter_mut().find(|s| s.is_none()) {
                *free = Some(item);
            }
        }
        n - took
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

    /// `dt` seconds pass: clarity runs out, affliction creeps up and stamina comes back,
    /// unless a shield is held up or the player sprints (`straining`).
    pub fn update(&mut self, dt: f32, straining: bool) {
        self.clarity = (self.clarity - dt).max(0.0);
        if !straining {
            self.stamina = (self.stamina + RECOVER * dt).min(STAMINA);
        }
        self.afflict(AFFLICTION_RISE * dt);
    }

    /// Use the item on hotbar slot `slot`; says what happened, `None` for an empty slot.
    pub fn use_slot(&mut self, slot: usize) -> Option<String> {
        let item = self.hotbar[slot]?;
        Some(self.use_item(item))
    }

    /// The hotbar slot `item` is on, if any.
    pub fn slot_of(&self, item: Item) -> Option<usize> {
        self.hotbar.iter().position(|s| *s == Some(item))
    }

    /// Put `item` on hotbar slot `slot`, moving it from any other slot and taking off
    /// whatever was there.
    pub fn put_on_hotbar(&mut self, item: Item, slot: usize) {
        self.take_off_hotbar(item);
        self.hotbar[slot] = Some(item);
    }

    pub fn take_off_hotbar(&mut self, item: Item) {
        self.hotbar.iter_mut().filter(|s| **s == Some(item)).for_each(|s| *s = None);
    }

    /// Use one of `item`. One that would do nothing isn't used up.
    pub fn use_item(&mut self, item: Item) -> String {
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
            Item::ClearSeeingPotion if self.clarity > 0.0 => "you already see clearly".into(),
            Item::ClearSeeingPotion => {
                self.clarity = CLARITY_SECS;
                self.remove_one(item);
                "the dream lifts".into()
            }
            Item::CalmingPotion if self.affliction <= 0.0 => "nothing troubles you".into(),
            Item::CalmingPotion => {
                self.affliction = 0.0;
                self.remove_one(item);
                "your head clears".into()
            }
            _ => format!("the {} can't be used", item.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_name_the_same_items() {
        let names: Vec<&str> = Item::ALL.iter().map(|i| i.name()).collect();
        assert_eq!(names, engine::world::level::ITEMS);
        assert_eq!(Item::named("copper ring"), Some(Item::CopperRing));
        assert_eq!(Item::named("a sandwich"), None);
    }

    #[test]
    fn starts_with_potions_on_the_hotbar() {
        let p = Player::new();
        assert_eq!(p.hotbar, [Some(Item::HealingPotion), Some(Item::ManaPotion), Some(Item::ClearSeeingPotion), Some(Item::CalmingPotion)]);
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
    fn an_item_moves_between_hotbar_slots_and_takes_the_place_of_another() {
        let mut p = Player::new();
        p.put_on_hotbar(Item::HealingPotion, 2);
        assert_eq!(p.hotbar, [None, Some(Item::ManaPotion), Some(Item::HealingPotion), Some(Item::CalmingPotion)]);
        assert_eq!(p.slot_of(Item::ClearSeeingPotion), None);
        assert_eq!(p.count(Item::ClearSeeingPotion), 2);
        p.take_off_hotbar(Item::ManaPotion);
        assert_eq!(p.slot_of(Item::ManaPotion), None);
        assert_eq!(p.use_slot(1), None);
    }

    #[test]
    fn no_more_than_the_most_is_carried() {
        let mut p = Player::new();
        assert_eq!(p.add(Item::HealingPotion, 100), 4);
        assert_eq!(p.count(Item::HealingPotion), MOST);
        assert_eq!(p.add(Item::HealingPotion, 1), 1);
    }

    #[test]
    fn gear_is_worn_in_its_slot_and_the_old_piece_comes_back() {
        let mut p = Player::new();
        assert_eq!(p.stats(), Stats { attack: 5, defence: 2, magic: 0 });
        assert!(p.unequip(1));
        assert_eq!(p.stats().defence, 0);
        assert!(!p.equip(Item::Buckler, 2));
        p.items.retain(|s| s.0 != Item::LeatherVest);
        assert!(!p.equip(Item::LeatherVest, 3));
        assert!(p.equip(Item::Buckler, 1));
        assert_eq!(p.count(Item::Buckler), 0);
        assert_eq!(p.stats().defence, 2);
        assert!(p.unequip(0));
        assert_eq!(p.count(Item::Sword), 1);
        assert_eq!(p.stats().attack, 0);
        assert_eq!(p.stats_with(0, Some(Item::Sword)).attack, 5);
        assert!(!p.unequip(0));
    }

    #[test]
    fn either_hand_holds_a_weapon_or_a_shield() {
        let mut p = Player::new();
        p.unequip(0);
        p.unequip(1);
        assert_eq!(p.slot_for(Item::Sword), Some(0));
        assert_eq!(p.slot_for(Item::Buckler), Some(1));
        assert!(p.equip(Item::Sword, 1));
        assert!(p.equip(Item::Buckler, 0));
        assert_eq!((p.hand(Side::Right), p.hand(Side::Left)), (Some(Item::Buckler), Some(Item::Sword)));
        assert!(!p.equip(Item::LeatherCap, 0));
        // A second sword: each hand's blow is its own.
        p.unequip(0);
        p.add(Item::Sword, 1);
        assert!(p.equip(Item::Sword, 0));
        assert_eq!(p.stats().attack, 10);
        assert_eq!(p.attack_with(Side::Left), 5);
    }

    #[test]
    fn stamina_comes_back_steadily_but_not_behind_a_shield() {
        let mut p = Player::new();
        p.tire(30.0);
        assert_eq!(p.stamina, STAMINA - 30.0);
        p.update(1.0, true);
        assert_eq!(p.stamina, STAMINA - 30.0);
        p.update(1.0, false);
        assert_eq!(p.stamina, STAMINA - 30.0 + RECOVER);
        p.update(10.0, false);
        assert_eq!(p.stamina, STAMINA);
        p.tire(99.0);
        assert_eq!(p.stamina, 0.0);
    }

    #[test]
    fn two_rings_go_in_the_two_ring_slots() {
        let mut p = Player::new();
        let first = p.slot_for(Item::CopperRing).unwrap();
        assert!(p.equip(Item::CopperRing, first));
        let second = p.slot_for(Item::CopperRing).unwrap();
        assert_ne!(first, second);
        assert!(p.equip(Item::CopperRing, second));
        assert_eq!(p.count(Item::CopperRing), 0);
        assert_eq!(p.stats().magic, 2);
        assert!(!p.items.iter().any(|s| s.0 == Item::CopperRing));
    }

    #[test]
    fn gear_doesnt_go_on_the_hotbar_by_itself_and_cant_be_used() {
        let mut p = Player::new();
        p.hotbar = [None; SLOTS];
        p.items.clear();
        p.add(Item::LeatherVest, 1);
        assert_eq!(p.hotbar, [None; SLOTS]);
        assert_eq!(p.use_item(Item::LeatherVest), "the leather vest can't be used");
        assert_eq!(p.count(Item::LeatherVest), 1);
    }

    #[test]
    fn dropping_takes_one_and_its_slot_with_the_last() {
        let mut p = Player::new();
        assert!(p.drop_one(Item::CalmingPotion));
        assert_eq!(p.slot_of(Item::CalmingPotion), None);
        assert!(!p.drop_one(Item::CalmingPotion));
    }

    #[test]
    fn affliction_creeps_up_and_stops_at_all() {
        let mut p = Player::new();
        p.update(60.0, false);
        assert!((p.affliction - 0.1).abs() < 1e-5);
        p.update(6000.0, false);
        assert_eq!(p.affliction, 1.0);
    }

    #[test]
    fn clear_seeing_runs_out_and_leaves_affliction_alone() {
        let mut p = Player::new();
        p.afflict(0.5);
        assert_eq!(p.use_slot(2).as_deref(), Some("the dream lifts"));
        assert_eq!(p.clarity, CLARITY_SECS);
        assert_eq!(p.count(Item::ClearSeeingPotion), 1);
        assert_eq!(p.affliction, 0.5);
        p.update(CLARITY_SECS, false);
        assert_eq!(p.clarity, 0.0);
        assert!(p.affliction > 0.5);
    }

    #[test]
    fn a_second_clarity_potion_is_kept_while_the_first_lasts() {
        let mut p = Player::new();
        p.use_slot(2);
        assert_eq!(p.use_slot(2).as_deref(), Some("you already see clearly"));
        assert_eq!(p.count(Item::ClearSeeingPotion), 1);
    }

    #[test]
    fn a_calming_potion_clears_affliction_and_is_kept_if_there_is_none() {
        let mut p = Player::new();
        assert_eq!(p.use_slot(3).as_deref(), Some("nothing troubles you"));
        assert_eq!(p.count(Item::CalmingPotion), 1);
        p.afflict(0.7);
        assert_eq!(p.use_slot(3).as_deref(), Some("your head clears"));
        assert_eq!(p.affliction, 0.0);
        assert_eq!(p.count(Item::CalmingPotion), 0);
    }

    #[test]
    fn health_and_mana_stop_at_zero() {
        let mut p = Player::new();
        p.hurt(1000);
        p.spend_mp(1000);
        assert_eq!((p.hp, p.mp), (0, 0));
    }
}
