//! The pause menu and the inventory screen it opens: what is selected and what a press
//! does. No Bevy and no drawing here (`hud` draws it), so the rules can be tested alone.

use crate::player::{Item, Player, EQUIPMENT, SLOTS};

/// One press, from the keyboard, the pad or the mouse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Press {
    Up,
    Down,
    Left,
    Right,
    Confirm,
    Back,
    /// A hotbar slot's own button: puts the item picked in the list on that slot, or picks
    /// the slot while choosing one.
    Slot(usize),
}

pub const PAUSE_ROWS: [&str; 5] = ["Resume", "Inventory", "Spells", "Options", "Quit game"];
const INVENTORY: usize = 1;

/// Whether a pause menu row does anything yet; those that don't are drawn dim.
pub fn ready(row: usize) -> bool {
    !matches!(row, 2 | 3)
}

/// What can be done with an item, in the window it opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    Use,
    PutOnHotbar,
    MoveOnHotbar,
    TakeOffHotbar,
    Equip,
    Drop,
}

impl Choice {
    pub fn label(self) -> &'static str {
        match self {
            Choice::Use => "Use",
            Choice::PutOnHotbar => "Put on hotbar",
            Choice::MoveOnHotbar => "Move on hotbar",
            Choice::TakeOffHotbar => "Take off hotbar",
            Choice::Equip => "Equip",
            Choice::Drop => "Drop",
        }
    }
}

pub fn choices(player: &Player, item: Item) -> Vec<Choice> {
    if item.slot().is_some() {
        return vec![Choice::Equip, Choice::Drop];
    }
    match player.slot_of(item) {
        Some(_) => vec![Choice::Use, Choice::MoveOnHotbar, Choice::TakeOffHotbar, Choice::Drop],
        None => vec![Choice::Use, Choice::PutOnHotbar, Choice::Drop],
    }
}

/// What an equipment slot's window offers: each kind of carried thing that goes there, then
/// taking off what is worn (`None`).
pub fn gear_choices(player: &Player, at: usize) -> Vec<Option<Item>> {
    let mut out: Vec<Option<Item>> =
        player.items.iter().filter(|s| s.0.slot() == Some(EQUIPMENT[at])).map(|s| Some(s.0)).collect();
    if player.equipped[at].is_some() {
        out.push(None);
    }
    out
}

/// The window an item in the list opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemWindow {
    pub item: Item,
    pub row: usize,
    /// The hotbar slot being picked, while putting the item on the hotbar.
    pub picking: Option<usize>,
}

/// The window an equipment slot opens: what could be worn there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GearWindow {
    pub at: usize,
    pub row: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Item(ItemWindow),
    Gear(GearWindow),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Screen {
    Pause { row: usize },
    /// `row` in the item list; `gear` is the equipment slot picked while the equipment
    /// column has the selection.
    Inventory { row: usize, gear: Option<usize>, window: Option<Window> },
}

/// What the game has to do after a press.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Nothing,
    Close,
    Quit,
    /// One of this was dropped, and has to be put on the floor.
    Drop(Item),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Menu {
    pub screen: Screen,
    /// What the last choice did, shown at the foot of the screen.
    pub note: Option<String>,
}

fn up(i: &mut usize, n: usize) {
    if n > 0 {
        *i = (*i + n - 1) % n;
    }
}

fn down(i: &mut usize, n: usize) {
    if n > 0 {
        *i = (*i + 1) % n;
    }
}

impl Menu {
    pub fn new() -> Menu {
        Menu { screen: Screen::Pause { row: 0 }, note: None }
    }

    pub fn inventory() -> Menu {
        Menu { screen: Screen::Inventory { row: 0, gear: None, window: None }, note: None }
    }

    /// Whether a hotbar slot is being picked, when the d-pad picks slots rather than moving.
    pub fn picking(&self) -> bool {
        matches!(self.screen, Screen::Inventory { window: Some(Window::Item(ItemWindow { picking: Some(_), .. })), .. })
    }

    /// The selection in whichever list takes presses now, for the mouse.
    pub fn point(&mut self, at: usize) {
        match &mut self.screen {
            Screen::Pause { row } => *row = at,
            Screen::Inventory { window: Some(Window::Item(ItemWindow { picking: Some(slot), .. })), .. } => *slot = at,
            Screen::Inventory { window: Some(Window::Item(w)), .. } => w.row = at,
            Screen::Inventory { window: Some(Window::Gear(w)), .. } => w.row = at,
            Screen::Inventory { row, gear, window: None } => match gear {
                // The item list counts on from the equipment slots.
                _ if at >= EQUIPMENT.len() => {
                    *row = at - EQUIPMENT.len();
                    *gear = None;
                }
                _ => *gear = Some(at),
            },
        }
    }

    pub fn press(&mut self, press: Press, player: &mut Player) -> Outcome {
        let Screen::Inventory { row, gear, window } = &mut self.screen else {
            return self.pause_press(press);
        };
        let items = player.items.len();
        let mut outcome = Outcome::Nothing;
        let mut note = None;
        match window {
            None => match (press, gear.as_mut()) {
                (Press::Up, None) => up(row, items),
                (Press::Down, None) => down(row, items),
                (Press::Up, Some(g)) => up(g, EQUIPMENT.len()),
                (Press::Down, Some(g)) => down(g, EQUIPMENT.len()),
                (Press::Right, None) => *gear = Some(0),
                (Press::Left, Some(_)) => *gear = None,
                (Press::Confirm, None) if *row < items => {
                    *window = Some(Window::Item(ItemWindow { item: player.items[*row].0, row: 0, picking: None }));
                }
                (Press::Confirm, Some(g)) => *window = Some(Window::Gear(GearWindow { at: *g, row: 0 })),
                (Press::Slot(s), None) if *row < items && s < SLOTS => {
                    let item = player.items[*row].0;
                    note = Some(match item.slot() {
                        Some(_) => "only what can be used goes on the hotbar".to_string(),
                        None => {
                            player.put_on_hotbar(item, s);
                            format!("the {} goes on slot {}", item.name(), s + 1)
                        }
                    });
                }
                (Press::Back, _) => self.screen = Screen::Pause { row: INVENTORY },
                _ => {}
            },
            Some(Window::Item(w)) => {
                let done = match window_press(w, press, player) {
                    Some((said, dropped)) => {
                        note = Some(said);
                        if let Some(item) = dropped {
                            outcome = Outcome::Drop(item);
                        }
                        true
                    }
                    None => press == Press::Back && w.picking.is_none(),
                };
                if press == Press::Back {
                    w.picking = None;
                }
                if done {
                    *window = None;
                }
            }
            Some(Window::Gear(w)) => {
                let wears = gear_choices(player, w.at);
                match press {
                    Press::Up => up(&mut w.row, wears.len()),
                    Press::Down => down(&mut w.row, wears.len()),
                    Press::Back => *window = None,
                    Press::Confirm => {
                        note = wears.get(w.row).map(|&wear| match wear {
                            Some(item) if player.equip(item, w.at) => format!("you put on the {}", item.name()),
                            None if player.unequip(w.at) => "you take it off".to_string(),
                            _ => "you have no room for what you'd take off".to_string(),
                        });
                        *window = None;
                    }
                    _ => {}
                }
            }
        }
        // What was used, dropped or put on leaves the list.
        if let Screen::Inventory { row, .. } = &mut self.screen {
            *row = (*row).min(player.items.len().saturating_sub(1));
        }
        if note.is_some() {
            self.note = note;
        }
        outcome
    }

    fn pause_press(&mut self, press: Press) -> Outcome {
        let Screen::Pause { row } = &mut self.screen else { return Outcome::Nothing };
        match press {
            Press::Up => up(row, PAUSE_ROWS.len()),
            Press::Down => down(row, PAUSE_ROWS.len()),
            Press::Back => return Outcome::Close,
            Press::Confirm => match *row {
                0 => return Outcome::Close,
                INVENTORY => *self = Menu::inventory(),
                4 => return Outcome::Quit,
                _ => {}
            },
            _ => {}
        }
        Outcome::Nothing
    }
}

/// A press in an item's window; once something was done, what to say and anything dropped.
fn window_press(w: &mut ItemWindow, press: Press, player: &mut Player) -> Option<(String, Option<Item>)> {
    let name = w.item.name();
    if let Some(slot) = &mut w.picking {
        let put = match press {
            Press::Up | Press::Left => {
                up(slot, SLOTS);
                None
            }
            Press::Down | Press::Right => {
                down(slot, SLOTS);
                None
            }
            Press::Confirm => Some(*slot),
            Press::Slot(s) if s < SLOTS => Some(s),
            _ => None,
        };
        return put.map(|s| {
            player.put_on_hotbar(w.item, s);
            (format!("the {name} goes on slot {}", s + 1), None)
        });
    }
    let choices = choices(player, w.item);
    match press {
        Press::Up => up(&mut w.row, choices.len()),
        Press::Down => down(&mut w.row, choices.len()),
        Press::Confirm => match choices[w.row.min(choices.len() - 1)] {
            Choice::Use => return Some((player.use_item(w.item), None)),
            Choice::PutOnHotbar | Choice::MoveOnHotbar => {
                let free = player.hotbar.iter().position(Option::is_none);
                w.picking = Some(player.slot_of(w.item).or(free).unwrap_or(0));
            }
            Choice::TakeOffHotbar => {
                player.take_off_hotbar(w.item);
                return Some((format!("the {name} is off the hotbar"), None));
            }
            Choice::Equip => {
                let worn = player.slot_for(w.item).is_some_and(|at| player.equip(w.item, at));
                let said = if worn { format!("you put on the {name}") } else { "you have no room for what you'd take off".to_string() };
                return Some((said, None));
            }
            Choice::Drop => {
                player.drop_one(w.item);
                return Some((format!("you drop the {name}"), Some(w.item)));
            }
        },
        _ => {}
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn presses(menu: &mut Menu, player: &mut Player, presses: &[Press]) -> Outcome {
        let mut last = Outcome::Nothing;
        for &p in presses {
            last = menu.press(p, player);
        }
        last
    }

    fn at(row: usize) -> Menu {
        Menu { screen: Screen::Inventory { row, gear: None, window: None }, note: None }
    }

    fn item_window(menu: &Menu) -> Option<ItemWindow> {
        match menu.screen {
            Screen::Inventory { window: Some(Window::Item(w)), .. } => Some(w),
            _ => None,
        }
    }

    #[test]
    fn resume_and_back_close_the_pause_menu_and_quit_quits() {
        let mut p = Player::new();
        assert_eq!(Menu::new().press(Press::Confirm, &mut p), Outcome::Close);
        assert_eq!(Menu::new().press(Press::Back, &mut p), Outcome::Close);
        assert_eq!(presses(&mut Menu::new(), &mut p, &[Press::Up, Press::Confirm]), Outcome::Quit);
    }

    #[test]
    fn rows_not_ready_do_nothing() {
        let mut p = Player::new();
        let mut m = Menu::new();
        assert_eq!(presses(&mut m, &mut p, &[Press::Down, Press::Down, Press::Confirm]), Outcome::Nothing);
        assert_eq!(m.screen, Screen::Pause { row: 2 });
        assert!(!ready(2) && !ready(3) && ready(1));
    }

    #[test]
    fn the_inventory_opens_and_backs_out_to_its_row() {
        let mut p = Player::new();
        let mut m = Menu::new();
        presses(&mut m, &mut p, &[Press::Down, Press::Confirm]);
        assert_eq!(m, Menu::inventory());
        presses(&mut m, &mut p, &[Press::Up]);
        assert_eq!(m.screen, Screen::Inventory { row: p.items.len() - 1, gear: None, window: None });
        presses(&mut m, &mut p, &[Press::Back]);
        assert_eq!(m.screen, Screen::Pause { row: 1 });
    }

    #[test]
    fn using_an_item_from_the_inventory_says_what_happened() {
        let mut p = Player::new();
        p.hurt(10);
        let mut m = Menu::inventory();
        presses(&mut m, &mut p, &[Press::Confirm, Press::Confirm]);
        assert_eq!(p.count(Item::HealingPotion), 2);
        assert_eq!(m.note.as_deref(), Some("you drink the healing potion"));
        assert_eq!(m, Menu { note: m.note.clone(), ..Menu::inventory() });
    }

    #[test]
    fn using_the_last_one_keeps_the_selection_on_the_list() {
        let mut p = Player::new();
        p.items.truncate(4);
        p.afflict(0.5);
        let mut m = at(3);
        presses(&mut m, &mut p, &[Press::Confirm, Press::Confirm]);
        assert_eq!(p.items.len(), 3);
        assert_eq!(m.screen, Screen::Inventory { row: 2, gear: None, window: None });
    }

    #[test]
    fn an_item_goes_on_the_hotbar_slot_picked() {
        let mut p = Player::new();
        p.take_off_hotbar(Item::ManaPotion);
        let mut m = at(1);
        presses(&mut m, &mut p, &[Press::Confirm, Press::Down]);
        assert!(!m.picking());
        presses(&mut m, &mut p, &[Press::Confirm]);
        // The free slot is offered first.
        assert!(m.picking());
        assert_eq!(item_window(&m), Some(ItemWindow { item: Item::ManaPotion, row: 1, picking: Some(1) }));
        presses(&mut m, &mut p, &[Press::Slot(3)]);
        assert_eq!(p.hotbar[3], Some(Item::ManaPotion));
        assert_eq!(p.slot_of(Item::CalmingPotion), None);
        assert_eq!(m.screen, Screen::Inventory { row: 1, gear: None, window: None });
    }

    #[test]
    fn a_hotbar_button_on_the_list_puts_the_item_there() {
        let mut p = Player::new();
        let mut m = at(2);
        presses(&mut m, &mut p, &[Press::Slot(0)]);
        assert_eq!(p.hotbar[0], Some(Item::ClearSeeingPotion));
        assert_eq!(p.hotbar[2], None);
        let buckler = p.items.iter().position(|s| s.0 == Item::Buckler).unwrap();
        let mut m = at(buckler);
        presses(&mut m, &mut p, &[Press::Slot(1)]);
        assert_eq!(p.hotbar[1], Some(Item::ManaPotion));
        assert_eq!(m.note.as_deref(), Some("only what can be used goes on the hotbar"));
    }

    #[test]
    fn back_while_picking_returns_to_the_choices() {
        let mut p = Player::new();
        let mut m = Menu::inventory();
        presses(&mut m, &mut p, &[Press::Confirm, Press::Down, Press::Confirm, Press::Back]);
        assert_eq!(item_window(&m), Some(ItemWindow { item: Item::HealingPotion, row: 1, picking: None }));
        presses(&mut m, &mut p, &[Press::Back]);
        assert_eq!(m, Menu::inventory());
        assert_eq!(p.hotbar[0], Some(Item::HealingPotion));
    }

    #[test]
    fn an_item_comes_off_the_hotbar() {
        let mut p = Player::new();
        let mut m = Menu::inventory();
        presses(&mut m, &mut p, &[Press::Confirm, Press::Down, Press::Down, Press::Confirm]);
        assert_eq!(p.hotbar[0], None);
        assert_eq!(choices(&p, Item::HealingPotion), [Choice::Use, Choice::PutOnHotbar, Choice::Drop]);
    }

    #[test]
    fn dropping_one_asks_the_game_to_put_it_down() {
        let mut p = Player::new();
        let mut m = Menu::inventory();
        let out = presses(&mut m, &mut p, &[Press::Confirm, Press::Up, Press::Confirm]);
        assert_eq!(out, Outcome::Drop(Item::HealingPotion));
        assert_eq!(p.count(Item::HealingPotion), 2);
    }

    #[test]
    fn gear_from_the_list_is_equipped_where_it_goes() {
        let mut p = Player::new();
        let ring = p.items.iter().position(|s| s.0 == Item::CopperRing).unwrap();
        let mut m = at(ring);
        presses(&mut m, &mut p, &[Press::Confirm, Press::Confirm, Press::Confirm, Press::Confirm]);
        assert_eq!(p.equipped[5..], [Some(Item::CopperRing); 2]);
        assert_eq!(p.stats().magic, 2);
    }

    #[test]
    fn the_equipment_column_swaps_and_takes_off_gear() {
        let mut p = Player::new();
        let mut m = Menu::inventory();
        presses(&mut m, &mut p, &[Press::Right, Press::Down]);
        assert_eq!(m.screen, Screen::Inventory { row: 0, gear: Some(1), window: None });
        // The shield slot offers the buckler and nothing to take off.
        assert_eq!(gear_choices(&p, 1), [Some(Item::Buckler)]);
        presses(&mut m, &mut p, &[Press::Confirm, Press::Confirm]);
        assert_eq!(p.equipped[1], Some(Item::Buckler));
        assert_eq!(gear_choices(&p, 1), [None]);
        presses(&mut m, &mut p, &[Press::Confirm, Press::Confirm]);
        assert_eq!(p.equipped[1], None);
        assert_eq!(p.count(Item::Buckler), 1);
        presses(&mut m, &mut p, &[Press::Left]);
        assert_eq!(m.screen, Screen::Inventory { row: 0, gear: None, window: None });
    }

    #[test]
    fn an_empty_equipment_slot_with_nothing_to_wear_opens_an_empty_window() {
        let mut p = Player::new();
        p.items.retain(|s| s.0 != Item::LeatherVest);
        let mut m = Menu::inventory();
        presses(&mut m, &mut p, &[Press::Right, Press::Down, Press::Down, Press::Down, Press::Confirm]);
        assert_eq!(gear_choices(&p, 3), []);
        presses(&mut m, &mut p, &[Press::Down, Press::Confirm]);
        assert_eq!(m.screen, Screen::Inventory { row: 0, gear: Some(3), window: None });
    }

    #[test]
    fn the_mouse_points_at_the_list_taking_presses() {
        let mut p = Player::new();
        let mut m = Menu::inventory();
        m.point(EQUIPMENT.len() + 2);
        presses(&mut m, &mut p, &[Press::Confirm]);
        m.point(1);
        presses(&mut m, &mut p, &[Press::Confirm]);
        m.point(3);
        presses(&mut m, &mut p, &[Press::Confirm]);
        assert_eq!(p.hotbar[3], Some(Item::ClearSeeingPotion));
        m.point(1);
        assert_eq!(m.screen, Screen::Inventory { row: 2, gear: Some(1), window: None });
    }

    #[test]
    fn an_empty_inventory_takes_presses_safely() {
        let mut p = Player::new();
        p.items.clear();
        p.hotbar = [None; SLOTS];
        let mut m = Menu::inventory();
        presses(&mut m, &mut p, &[Press::Up, Press::Down, Press::Confirm, Press::Slot(0)]);
        assert_eq!(m, Menu::inventory());
    }
}
