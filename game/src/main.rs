//! A first-person walk through a generated level, drawn by the trained network, in Bevy.
//!
//! `cargo run -p game [-- --seed 7]`
//!     The model, skins and textures come from `neurofield.ron` (`engine::world::config`);
//!     options of the same name override it: `--model x.safetensors`, `--photos DIR`,
//!     `--strange DIR`, `--level DIR` (level textures alone).
//!     `--dream x.safetensors` draws the creatures with that dream net (train/export_dream.py;
//!     `neurofield.ron` names the usual one); `--dream none` leaves them out of the network's
//!     picture and still. With it they walk about, and show as low-poly PS1 models when the
//!     rasteriser is on (a clear seeing potion, or F1).
//!     `--window 1280x720` sets the window's starting size; the frame follows its shape.
//!     `--sculpt data/eye` turns the level to folded flesh with filmed eyes, veins and mouths and
//!     draws it with the rasteriser, as the level net might one day (`engine::world::dreamed`);
//!     its textures default to data/painted. [ and ] turn it from plain stone to fully flesh
//!     and back (select + L1 / R1 on the pad).
//!     `--view raster` (or `both`) starts on the rasteriser instead of the network.
//!     `--strangeness 0.7` sets how strange each level is (otherwise random).
//!     `--affliction 0.7` starts the player that afflicted, to try how it looks.
//!     `--peaceful`: the creatures' blows and spells do nothing (for trying other things).
//!     `--level-file levels/x.ron` plays a hand-built level (see `engine::world::level`) from its
//!     player start, or from `--start x,y,z,yaw` (world units, yaw in radians); N reloads it.
//!     E (or circle) opens and closes the door in front, pulls the lever, or opens the chest
//!     (what it holds goes into the inventory); walking over a key or an item picks it up,
//!     and locked doors and chests open for their key.
//!     Mouse: look   Left / right click: the left / right hand's weapon swings, or shield goes
//!     up while held   WASD: walk and strafe   Arrows: walk and turn   Space: jump
//!     Shift: sprint   1-4: use the hotbar   Esc: the pause menu (`menu`), which frees the mouse
//!     Gamepad: left stick walk and strafe, right stick look, cross jump, L3 sprint,
//!     R2 / L2 the right / left hand (as the clicks), circle use, d-pad use the hotbar, start the pause menu
//!     In the menus: arrows or WASD (d-pad) move, Enter, Space or E (cross) choose, Esc or
//!     Backspace (circle) go back; the mouse points and clicks, right click goes back. In the
//!     inventory, 1-4 (square + d-pad) put the item picked on that hotbar slot.
//!     Debug keys: F1 network / rasteriser   F2 both side by side   F3 be hit   F4 free or
//!     capture the mouse   Tab auto-walk   H show the effects panel   N new level   P freeze
//!     Debug on the pad, holding select: d-pad walk and turn, L2 sprint, triangle
//!     network / rasteriser, square auto-walk, L1 / R1 strangeness down / up, circle
//!     hurt, cross new level (or reload)
//!
//! The dream effects panel (egui) turns effects on and off in any combination; free the
//! mouse with Esc to use it. The network runs in the main world each frame and its image, 240
//! high and from 320 to 576 wide to match the window's shape (the widths it was trained on;
//! beyond them there are bars), is uploaded to a texture, drawn with nearest sampling and
//! scaled to fill the window.
//! The rasteriser is psx-bevy drawing the level (`engine::render`) into its 320x240 screen.
//! `GAME_SHOT=out.png:5` saves a screenshot after 5 seconds and quits.
//!
//! Developer builds (`--features dev-mcp`) take `--mcp` for psx-bevy's developer MCP
//! on stdio (see psx-bevy's README, "Agent driving"), with `--headless` and `--software`
//! to run without a window. Under it the walk starts paused; `game_step` holds pad
//! buttons for a number of ticks (select with up/down walks, with left/right turns;
//! cross jumps, circle uses), `game_screenshot {"target":"screen"}` grabs the
//! rasteriser's frame, and `game_query {"resources":["Report"]}` reads where the walker
//! is and what happened.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::image::ImageSampler;
use bevy::input::gamepad::Gamepad;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_egui::input::EguiWantsInput;
use bevy_egui::{egui, EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext};
use engine::world::view::{H, W};
use engine::world::rng::Rng;
use engine::world::growths::Growths;
use engine::world::walker::{self, MAX_PITCH};
use psx_bevy::controller::{BindingDevice, Button, ControllerBindings, ControllerState};
use psx_bevy::psx::{fit_scale, PsxPlugin, PsxScreen, DISPLAY_LAYER};
use psx_bevy::shell::{GameSystems, ShellPlugin, TickRate};
use std::collections::HashMap;
use std::path::PathBuf;
use engine::neural::gpu_net::MAX_W;
use engine::neural::{effects, Level, Network, TICK};

mod hud;
mod drops;
mod foes;
mod menu;
mod player;
mod hands;
use hands::Hands;
use hud::HudState;
use menu::{Menu, Outcome, Press};
use player::{Player, Side, SIDES};

/// Stamina a swing uses, and holding up a shield uses a second.
const SWING_TIRE: f32 = 7.0;
const GUARD_TIRE: f32 = 4.0;
/// A laboured swing, with no stamina left, does this share of the attack (at least 1).
const LABOURED: i32 = 5;

/// Mouse sensitivity: yaw units (4096 per turn) per pixel of mouse movement.
const LOOK: f32 = 1.2;
/// Right stick look speed at full tilt: yaw and pitch units per second.
const PAD_YAW: f32 = 2600.0;
const PAD_PITCH: f32 = 1400.0;
/// Seconds for what the player feels to catch up with a whole bar of change in affliction.
const FEEL_SECS: f32 = 2.0;
/// Seconds the strangeness takes to drain away, or come back, under the network.
const CALM_SECS: f32 = 1.5;
/// Seconds the network's calm picture takes to fade to the plain one, and back.
const FADE_SECS: f32 = 0.8;
/// What a tentacle's lash does: health lost and affliction added, and the stamina it costs
/// to block.
const LASH_HP: i32 = 4;
const LASH_AFFLICTION: f32 = 0.01;
const LASH_BLOCK: f32 = 7.0;
/// Stamina sprinting uses a second (none comes back meanwhile).
const SPRINT_TIRE: f32 = 3.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Network,
    Raster,
    Both,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Network => "network",
            Mode::Raster => "rasteriser",
            Mode::Both => "rasteriser | network",
        }
    }
}

#[derive(Resource)]
struct Sim {
    level: Level,
    rng: Rng,
    auto_walk: bool,
    paused: bool,
    mode: Mode,
    /// Seconds since each dream effect was turned on, or None when it is off.
    effects: [Option<f32>; effects::EFFECTS.len()],
    panel: bool,
    net_ms: f32,
    herd: engine::world::animate::Herd,
    tris: usize,
    /// The network's frame width, following the window's shape (`layout`).
    width: u32,
    /// How strange the level is at its base, before affliction (the egui slider sets it).
    strangeness: f32,
    /// Affliction as the player feels it, easing after the real value so nothing jumps.
    felt: f32,
    /// The tentacles of near-full affliction; only the dream net can draw them.
    growths: Option<Growths>,
    /// With `--sculpt`: where the filmed eyes are, and what the level's been turned into.
    eyes: Option<PathBuf>,
    sculpted: Option<engine::world::dreamed::Dreamed>,
    /// How far the sculpted level has turned, 0..1; [ and ] move it.
    dial: f32,
    /// The level's triangles as affliction has swollen them, for this frame.
    swollen: Vec<f32>,
    /// Seconds of unpaused time, for the affliction's effects.
    dream_time: f32,
    /// How much of the strangeness has drained away, 0..1, and after that how far the
    /// network has faded to the plain picture, 0..1.
    calm: f32,
    veil: f32,
    /// A hand-built level being played, and where to start in it (else its player start).
    file: Option<PathBuf>,
    start: Option<([f32; 3], f32)>,
    /// Bumped for a new level: its meshes are rebuilt and its creatures start afresh.
    version: u64,
    /// Bumped when the level's lights are baked again, which its meshes will have to show.
    look: u64,
    /// Keys picked up in this level: entity ids and names.
    held: Vec<(u32, String)>,
    /// What last happened (a door, a key), and seconds left to show it.
    said: Option<(String, f32)>,
    player: Player,
    hands: Hands,
    /// Each hand's button held (right, left), on the mouse and on the pad.
    mouse_arms: [bool; 2],
    pad_arms: [bool; 2],
    /// The creatures as foes, the strikes and spells their animators said went off since the
    /// last tick, and the attack of the blades that swept through since then.
    foes: foes::Foes,
    cues: Vec<(usize, engine::world::animate::Cue)>,
    /// `--peaceful`: the creatures do no harm.
    peaceful: bool,
    struck: Option<i32>,
    /// Whether the controller was the last thing used, for the shape of the hotbar.
    on_pad: bool,
    /// The pause menu, while it's open; the game stands still under it.
    menu: Option<Menu>,
    /// What the player has dropped in this level.
    drops: Vec<drops::Dropped>,
    /// Quit game was chosen.
    quit: bool,
}

impl Sim {
    fn set_strangeness(&mut self, s: f32) {
        self.strangeness = s.clamp(0.0, 1.0);
        engine::world::scene::force_strangeness(Some(self.strangeness));
        self.relight();
    }

    /// Light the level for its strangeness.
    fn relight(&mut self) {
        self.level.set_strangeness(self.strangeness);
        self.look += 1;
    }

    /// What is on the screen, once any fade is over: with clarity, the dungeon as it is.
    fn shown(&self) -> Mode {
        if self.veil >= 1.0 && self.mode == Mode::Network { Mode::Raster } else { self.mode }
    }

    /// Whether the rasteriser is needed, showing through a fade as well.
    fn raster_on(&self) -> bool {
        self.shown() != Mode::Network || self.veil > 0.0
    }

    /// Whether a clear seeing potion has the dream draining, gone, or coming back: then the
    /// creatures are seen as they really are, as PS1 models.
    fn seeing_clearly(&self) -> bool {
        self.calm > 0.0
    }

    /// Clarity drains the strangeness with the network still drawing, then fades to the
    /// plain picture; when it runs out, the same backwards.
    fn fade(&mut self, dt: f32) {
        let calm = self.calm;
        if self.player.clarity > 0.0 {
            if self.calm < 1.0 {
                self.calm = (self.calm + dt / CALM_SECS).min(1.0);
            } else {
                self.veil = (self.veil + dt / FADE_SECS).min(1.0);
            }
        } else if self.veil > 0.0 {
            self.veil = (self.veil - dt / FADE_SECS).max(0.0);
        } else {
            self.calm = (self.calm - dt / CALM_SECS).max(0.0);
        }
        if self.calm != calm {
            self.level.set_strange_amount(1.0 - self.calm);
        }
    }

    fn toggle_renderer(&mut self) {
        self.mode = if self.mode == Mode::Network { Mode::Raster } else { Mode::Network };
    }

    /// Use whatever is in front: a door, a lever or a chest. What a chest holds goes into
    /// the inventory, or on the floor before it when there's no room.
    fn use_thing(&mut self) {
        let (feet, yaw) = self.feet();
        let held = self.held.clone();
        let Some(used) = self.level.scene.use_thing(feet, yaw, &held) else { return };
        let mut said = used.said;
        let mut found = Vec::new();
        let mut spilled = false;
        for name in &used.items {
            let Some(item) = player::Item::named(name) else {
                eprintln!("a chest holds {name:?}, which isn't an item");
                continue;
            };
            found.push(item.name());
            if self.player.add(item, 1) > 0 {
                self.drops.push(drops::lying(&self.level.scene, item, used.spill.unwrap_or(feet)));
                spilled = true;
            }
        }
        if !found.is_empty() {
            said = format!("{said}: {}", found.join(", "));
            if spilled {
                said += " (no room for some; they're on the floor)";
            }
        }
        self.say(said);
    }

    /// A new level, or the file again (it may have been edited), its own strangeness
    /// (a file keeps the slider's).
    fn reload(&mut self) {
        self.version += 1;
        self.held.clear();
        self.drops.clear();
        match &self.file {
            Some(f) => match load_level(f, self.start, &mut self.rng) {
                Ok((level, lying)) => {
                    self.level = level;
                    self.drops = lying;
                }
                Err(e) => eprintln!("{e}"),
            },
            None => {
                self.level = Level::new(&mut self.rng);
                self.strangeness = self.level.scene.strange.base;
            }
        }
        if let Some(dir) = &self.eyes {
            self.sculpted = Some(engine::world::dreamed::sculpt(&mut self.level.scene, self.rng.next() as u64, Some(dir)));
            self.strangeness = self.level.scene.strange.base;
        }
        let moving = self.growths.is_some();
        self.level.scene.set_creatures_move(moving);
        self.relight();
        self.level.set_strange_amount(1.0 - self.calm);
        if self.growths.is_some() {
            self.growths = Some(Growths::new(&self.level.scene));
        }
    }

    /// Affliction as it shows: drained away while the player sees clearly.
    fn showing(&self) -> f32 {
        self.felt * (1.0 - self.calm)
    }

    fn say(&mut self, what: String) {
        self.said = Some((what, 3.0));
    }

    /// Whether the game stands still: frozen by the debug key, or under the menu.
    fn frozen(&self) -> bool {
        self.paused || self.menu.is_some()
    }

    /// Whether the player's own actions count: not standing still, not on auto-walk.
    fn in_control(&self) -> bool {
        !self.frozen() && !self.auto_walk
    }

    fn toggle_menu(&mut self) {
        self.menu = if self.menu.is_some() { None } else { Some(Menu::new()) };
    }

    fn menu_press(&mut self, press: Press) {
        let Some(menu) = &mut self.menu else { return };
        match menu.press(press, &mut self.player) {
            Outcome::Nothing => {}
            Outcome::Close => self.menu = None,
            Outcome::Quit => self.quit = true,
            Outcome::Drop(item) => {
                let (feet, yaw) = self.feet();
                self.drops.push(drops::drop(&self.level.scene, item, feet, yaw));
            }
        }
    }

    /// Pick up whatever dropped thing the feet have come back to, if there's room for it.
    fn pick_up(&mut self) {
        let (feet, _) = self.feet();
        for i in drops::in_reach(&mut self.drops, feet).into_iter().rev() {
            let item = self.drops[i].item;
            if self.player.add(item, 1) == 0 {
                self.drops.remove(i);
                self.say(format!("picked up the {}", item.name()));
            } else {
                drops::disarm(&mut self.drops[i]);
                self.say(format!("you can't carry more of the {}", item.name()));
            }
        }
    }

    fn jump(&mut self) {
        if self.hands.stunned() {
            return;
        }
        let level = &mut self.level;
        level.walker.jump(&level.scene);
    }

    fn use_slot(&mut self, slot: usize) {
        if let Some(what) = self.player.use_slot(slot) {
            self.say(what);
        }
    }

    /// What a hit will do, so it can all be tried before anything can hurt.
    fn debug_hurt(&mut self) {
        self.player.hurt(5);
        self.player.spend_mp(2);
        self.player.afflict(0.1);
    }

    /// Hand `side`'s button went down: a weapon there swings, laboured with no stamina
    /// left. A shield goes up while the button is held (`update_hud`).
    fn press_arm(&mut self, side: Side) {
        if !self.in_control() {
            return;
        }
        let tired = self.player.stamina <= 0.0;
        if self.hands.swing(side, tired) {
            self.player.tire(if tired { 0.0 } else { SWING_TIRE });
        }
    }

    /// The creatures come for the player, and the sword finds them. Holding up a shield
    /// tires the player, and it comes down when they're spent.
    fn fight(&mut self) {
        let mut cues = std::mem::take(&mut self.cues);
        if self.peaceful {
            cues.clear();
        }
        let struck = std::mem::take(&mut self.struck);
        if self.auto_walk {
            return;
        }
        let dt = 1.0 / TICK as f32;
        if self.hands.guarding() {
            self.player.tire(GUARD_TIRE * dt);
            if self.player.stamina <= 0.0 {
                self.hands.give_out();
            }
        }
        let (feet, yaw) = self.feet();
        // A blow finds a tentacle in its way before any creature.
        let mut struck = struck;
        if let (Some(attack), Some(g)) = (struck, &mut self.growths) {
            let chest = [feet[0], feet[1] - 0.6 * walker::EYE, feet[2]];
            if let Some(killed) = g.strike(chest, [-yaw.sin(), yaw.cos()], foes::SWORD_REACH, foes::SWORD_ARC, attack) {
                self.say(if killed { "the tentacle falls limp" } else { "you cut the tentacle" }.to_string());
                struck = None;
            }
        }
        let stats = self.player.stats();
        let target = foes::Target { feet, yaw, guarding: self.hands.guarding(), stamina: self.player.stamina, struck, defence: stats.defence };
        let out = self.foes.tick(self.version, self.herd.animators(), &self.level.scene.creatures, &cues, &target, dt, &mut self.rng);
        self.player.hurt(out.hurt);
        self.player.afflict(out.afflict);
        self.player.tire(out.tired);
        if out.broken {
            self.hands.break_guard();
        } else if out.blocked {
            self.hands.knock();
        }
        if let Some(what) = out.said {
            self.say(what);
        }
    }

    /// Start running, if there is stamina to run on (`walk` uses it up).
    fn sprint(&mut self) {
        if self.player.stamina > 0.0 {
            self.level.walker.start_sprint();
        }
    }

    /// A tentacle's lash reached the player: a raised shield facing it takes the blow (and
    /// throws the tentacle off), as it does a creature's.
    fn lashed(&mut self, lash: engine::world::growths::Lash) {
        let (feet, yaw) = self.feet();
        if self.hands.guarding() && foes::off_ahead(feet, yaw, lash.root) < foes::SHIELD_ARC {
            self.player.tire(LASH_BLOCK);
            if let Some(g) = &mut self.growths {
                g.recoil(lash.which);
            }
            if self.player.stamina <= 0.0 {
                self.hands.break_guard();
                self.say("your guard is broken".to_string());
            } else {
                self.hands.knock();
                self.say("your shield takes the lash".to_string());
            }
        } else {
            self.player.hurt(LASH_HP);
            self.player.afflict(LASH_AFFLICTION + foes::HIT_AFFLICTION);
            self.player.tire(foes::HIT_TIRE);
            self.say("something lashes you".to_string());
        }
    }

    /// The walker's feet, and which way it faces in radians.
    fn feet(&self) -> ([f32; 3], f32) {
        let w = &self.level.walker;
        ([w.x, w.y + walker::EYE, w.z], w.yaw / 4096.0 * std::f32::consts::TAU)
    }
}

/// A hand-built level, and the items lying about in it.
fn load_level(file: &std::path::Path, start: Option<([f32; 3], f32)>, rng: &mut Rng) -> Result<(Level, Vec<drops::Dropped>), String> {
    let level = engine::world::level::Level::load(file)?;
    let built = Level::from_file(&level, start, rng).map_err(|e| format!("{}: {e}", file.display()))?;
    let lying = drops::of_level(&built.scene, &level);
    Ok((built, lying))
}

/// The network's 320x240 frame.
#[derive(Resource)]
struct Frame(Handle<Image>);

/// Shows the network's frame in the window, over psx-bevy's display of its screen.
#[derive(Component)]
struct NetworkSprite;

/// The dream creatures alone, laid into the rasteriser's picture when it shows without a
/// clear seeing potion (F1, `--view raster`, `--sculpt`), hidden by the level and its flesh
/// (`engine::render::DreamCreatures`): for now, until the level network can draw what the
/// rasteriser does, and perhaps for special cases after.
#[derive(Resource)]
struct Creatures {
    /// [H][W][4] and [H][W], from `DreamLayer::draw_alone`.
    rgba: Vec<f32>,
    near: Vec<f32>,
}

/// The scene's meshes in psx-bevy, and the versions of the scene they were built from,
/// and the version of each moving part's meshes (`Scene::things`).
#[derive(Resource, Default)]
struct Parts {
    entities: Vec<Entity>,
    /// `Sim::version` and `Sim::look`.
    version: Option<(u64, u64)>,
    things: Vec<u32>,
}

#[derive(Component)]
struct Status;

/// The HUD's image, and what it last showed.
#[derive(Resource)]
struct Hud {
    image: Handle<Image>,
    shown: Option<HudState>,
    /// Where the mouse can point in the menu, in the HUD's pixels.
    hits: Vec<hud::Hit>,
}

/// Shows the HUD, over the network's frame.
#[derive(Component)]
struct HudSprite;

/// What the walk is doing, for the developer MCP's `game_query` to read.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource)]
struct Report {
    /// The walker's feet (world units) and yaw (radians).
    feet: Vec3,
    yaw: f32,
    /// The last thing that happened, and the keys held.
    said: String,
    keys: Vec<String>,
    /// Each door's openness, 0 shut to 1 open, by entity id.
    doors: Vec<(u32, f32)>,
    /// The player: health and mana, what is carried, and the hotbar's items.
    hp: i32,
    mp: i32,
    items: Vec<(String, u32)>,
    hotbar: Vec<String>,
    airborne: bool,
    sprinting: bool,
    attacking: bool,
    /// A shield is up; a broken guard has the player stunned; and the stamina left.
    guarding: bool,
    stunned: bool,
    stamina: f32,
    /// The walking bob's offset of the camera, in world units.
    bob: f32,
    affliction: f32,
    felt: f32,
    clarity: f32,
    /// The level's strangeness at its base, affliction included, and the picture on show.
    strangeness: f32,
    shown: String,
    /// How much of the strangeness has drained, and how far the plain picture has faded in.
    calm: f32,
    veil: f32,
    /// How many tentacles are out.
    growths: usize,
    /// The menu's screen and selection, empty while it's closed.
    menu: String,
    /// What is worn, by equipment slot, and the stats it makes.
    equipped: Vec<String>,
    stats: [i32; 3],
    /// What lies dropped, and where.
    drops: Vec<(String, Vec3)>,
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Flags without values, for psx-bevy's developer MCP.
    let mut flag = |name: &str| {
        let had = args.iter().any(|a| a == name);
        args.retain(|a| a != name);
        had
    };
    let (mcp, headless, software) = (flag("--mcp"), flag("--headless"), flag("--software"));
    let peaceful = flag("--peaceful");
    if mcp && !cfg!(feature = "dev-mcp") {
        eprintln!("--mcp needs a developer build: cargo run -p game --features dev-mcp -- --mcp");
        std::process::exit(2);
    }
    let mut opts = HashMap::new();
    for pair in args.chunks(2) {
        match pair {
            [k, v] if k.starts_with("--") => {
                opts.insert(k[2..].to_string(), v.clone());
            }
            _ => panic!("expected --key value pairs, got {pair:?}"),
        }
    }
    let config = engine::world::config::Config::load();
    let model = engine::neural::model_path(&opts, &config);
    let seed: u64 = opts.get("seed").map_or(7, |s| s.parse().expect("--seed is a number"));
    if opts.contains_key("sculpt") && !opts.contains_key("strange") {
        opts.insert("strange".to_string(), "data/painted".to_string());
    }
    config.load_textures(&opts);
    if let Some(s) = opts.get("strangeness") {
        engine::world::scene::force_strangeness(Some(s.parse().expect("--strangeness 0..1")));
    }
    let net = Network::load(&model);
    // The dream creatures, drawn by their own network over the level network's frames.
    // One named on the command line must load; the config's may be missing on this machine.
    let dream_path = match opts.get("dream").map(String::as_str) {
        Some("none") => None,
        Some(d) => Some((PathBuf::from(d), true)),
        None => config.dream.clone().map(|d| (d, false)),
    };
    let dream = dream_path.and_then(|(d, asked)| {
        if !asked && !d.is_file() {
            eprintln!("no dream net at {}: the creatures stay still and the network leaves them out", d.display());
            return None;
        }
        let layer = engine::neural::dream_net::DreamLayer::load(&d, seed).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("dream creatures on {}", layer.adapter());
        Some(layer)
    });
    let mut rng = Rng::new(seed);
    let start = opts.get("start").map(|s| {
        let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().expect("--start x,y,z,yaw")).collect();
        assert_eq!(v.len(), 4, "--start x,y,z,yaw");
        ([v[0], v[1], v[2]], v[3])
    });
    let file = opts.get("level-file").map(PathBuf::from);
    let (mut level, lying) = match &file {
        Some(f) => load_level(f, start, &mut rng).unwrap_or_else(|e| panic!("{e}")),
        None => (Level::new(&mut rng), Vec::new()),
    };
    let eyes = opts.get("sculpt").map(PathBuf::from);
    let sculpted = eyes.as_deref().map(|dir| engine::world::dreamed::sculpt(&mut level.scene, seed, Some(dir)));
    let strangeness = level.scene.strange.base;
    let growths = dream.is_some().then(|| Growths::new(&level.scene));
    // The herd walks the dream creatures about, so the rasteriser draws them as they stand.
    level.scene.set_creatures_move(dream.is_some());

    let mut player = Player::new();
    if let Some(a) = opts.get("affliction") {
        player.afflict(a.parse().expect("--affliction 0..1"));
    }

    let window_size = opts.get("window").map_or((960, 720), |w| {
        let (x, y) = w.split_once('x').expect("--window WIDTHxHEIGHT");
        (x.parse::<u32>().expect("--window WIDTHxHEIGHT"), y.parse::<u32>().expect("--window WIDTHxHEIGHT"))
    });

    let mut app = App::new();
    let mut plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: (!headless)
                .then(|| Window { title: "neurofield".into(), resolution: window_size.into(), ..default() }),
            exit_condition: if headless {
                bevy::window::ExitCondition::DontExit
            } else {
                bevy::window::ExitCondition::OnAllClosed
            },
            ..default()
        })
        .set(ImagePlugin::default_nearest())
        .set(bevy::render::RenderPlugin {
            render_creation: bevy::render::settings::WgpuSettings { force_fallback_adapter: software, ..default() }
                .into(),
            synchronous_pipeline_compilation: headless,
            ..default()
        });
    if headless {
        plugins = plugins.disable::<bevy::winit::WinitPlugin>();
        app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(std::time::Duration::from_secs_f64(1.0 / 60.0)));
    }
    // Pad buttons (and the MCP's) come through the shell's controller; the keyboard
    // is read directly, so it has no bindings there.
    let mut bindings = ControllerBindings::default();
    for b in Button::ALL {
        bindings.clear(b, BindingDevice::Keyboard);
    }
    app.add_plugins(plugins)
    // The rasteriser: the network's own size and shape, so the two compare pixel for pixel.
    .add_plugins(PsxPlugin { screen: PsxScreen { width: W, height: H, ..default() } })
    .add_plugins(ShellPlugin { tick: TickRate { hz: TICK } })
    .add_plugins(engine::render::LevelGfxPlugin)
    .add_plugins(EguiPlugin::default())
        .insert_resource(bindings)
        .init_resource::<Report>()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Sim {
            level,
            rng,
            auto_walk: false,
            paused: false,
            mode: match opts.get("view").map(String::as_str) {
                None if eyes.is_some() => Mode::Raster,
                None | Some("network") => Mode::Network,
                Some("raster") => Mode::Raster,
                Some("both") => Mode::Both,
                Some(v) => panic!("--view network|raster|both, not {v}"),
            },
            effects: [None; effects::EFFECTS.len()],
            panel: false,
            net_ms: 0.0,
            herd: Default::default(),
            foes: Default::default(),
            cues: Vec::new(),
            peaceful,
            struck: None,
            tris: 0,
            width: W,
            strangeness,
            felt: player.affliction,
            growths,
            eyes,
            sculpted,
            dial: 1.0,
            swollen: Vec::new(),
            dream_time: 0.0,
            calm: 0.0,
            veil: 0.0,
            file,
            start,
            version: 0,
            look: 0,
            held: Vec::new(),
            said: None,
            player,
            hands: Hands::default(),
            mouse_arms: [false; 2],
            pad_arms: [false; 2],
            on_pad: false,
            menu: None,
            drops: lying,
            quit: false,
        })
        .insert_non_send(net)
        .init_resource::<Parts>()
        .add_systems(Startup, (setup, capture_mouse))
        .add_systems(Update, (keys, menu_mouse, menu_cursor, mouse, draw, build_scene, move_things, pose_sculpted, pose_creatures, raster_camera, status, update_hud, auto_screenshot).chain())
        .add_systems(Update, egui_on_display)
        // After psx-bevy fits its display to the window.
        .add_systems(PostUpdate, layout.before(TransformSystems::Propagate))
        .add_systems(FixedUpdate, walk.in_set(GameSystems::Game))
        .add_systems(EguiPrimaryContextPass, panel);
    if let Some(layer) = dream {
        app.insert_non_send(layer);
    }
    #[cfg(feature = "dev-mcp")]
    if mcp {
        app.add_plugins(psx_bevy::dev_mcp::GameMcpPlugin);
    }
    // egui draws on the window at full resolution, not into the PS1 screen.
    app.world_mut().resource_mut::<EguiGlobalSettings>().auto_create_primary_context = false;
    app.run();
}

/// As wide as the network can draw, whatever the window: the sprite shows the part in use
/// (changing the texture's size left the sprite stuck on the first frame of the new size).
fn frame_image() -> Image {
    let size = Extent3d { width: MAX_W, height: H, depth_or_array_layers: 1 };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    image
}

/// Transparent, and as wide as the frame can be, like `frame_image`.
fn hud_image() -> Image {
    let size = Extent3d { width: MAX_W, height: H, depth_or_array_layers: 1 };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    image
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let frame = images.add(frame_image());
    // Over psx-bevy's display quad (z 0), drawn by its window camera.
    commands.spawn((
        Sprite::from_image(frame.clone()),
        Transform::from_xyz(0.0, 0.0, 1.0),
        RenderLayers::layer(DISPLAY_LAYER),
        NetworkSprite,
    ));
    commands.insert_resource(Frame(frame));
    commands.insert_resource(Creatures { rgba: vec![0.0; (W * H * 4) as usize], near: vec![f32::MAX; (W * H) as usize] });
    let hud = images.add(hud_image());
    commands.spawn((
        Sprite::from_image(hud.clone()),
        Transform::from_xyz(0.0, 0.0, 2.0),
        RenderLayers::layer(DISPLAY_LAYER),
        HudSprite,
    ));
    commands.insert_resource(Hud { image: hud, shown: None, hits: Vec::new() });
    commands.spawn((
        Text::new(""),
        TextFont { font_size: bevy::text::FontSize::Px(14.0), ..default() },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(6.0),
            left: Val::Percent(22.0),
            max_width: Val::Percent(62.0),
            ..default()
        },
        Status,
    ));
}

fn capture_mouse(mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    // A scripted screenshot must leave the desktop's mouse alone.
    if std::env::var_os("GAME_SHOT").is_some() {
        return;
    }
    if let Ok(mut cursor) = cursor.single_mut() {
        set_captured(&mut cursor, true);
    }
}

fn set_captured(cursor: &mut CursorOptions, captured: bool) {
    cursor.grab_mode = if captured { CursorGrabMode::Locked } else { CursorGrabMode::None };
    cursor.visible = !captured;
}

fn keys(
    key: Res<ButtonInput<KeyCode>>,
    button: Res<ButtonInput<MouseButton>>,
    egui: Res<EguiWantsInput>,
    mut sim: ResMut<Sim>,
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
    gamepads: Query<&Gamepad>,
    mut exit: MessageWriter<AppExit>,
) {
    if key.get_just_pressed().next().is_some() {
        sim.on_pad = false;
    }
    if sim.quit {
        exit.write(AppExit::Success);
    }
    if sim.menu.is_some() {
        menu_keys(&key, &mut sim);
        return;
    }
    if key.just_pressed(KeyCode::Escape) {
        sim.toggle_menu();
        return;
    }
    if let Ok(mut cursor) = cursor.single_mut() {
        let captured = cursor.grab_mode != CursorGrabMode::None;
        // Each button is the hand on its side of the screen: right click the right hand.
        for (i, b) in [MouseButton::Right, MouseButton::Left].into_iter().enumerate() {
            sim.mouse_arms[i] = captured && button.pressed(b);
            if captured && button.just_pressed(b) {
                sim.press_arm(SIDES[i]);
            }
        }
        if button.just_pressed(MouseButton::Left) && !captured && !egui.is_pointer_over_area() {
            set_captured(&mut cursor, true);
        }
        if key.just_pressed(KeyCode::F4) {
            set_captured(&mut cursor, !captured);
        }
    }
    if key.just_pressed(KeyCode::F1) {
        sim.toggle_renderer();
    }
    if key.just_pressed(KeyCode::F2) {
        sim.mode = if sim.mode == Mode::Both { Mode::Network } else { Mode::Both };
    }
    if key.just_pressed(KeyCode::F3) {
        sim.debug_hurt();
    }
    // L3 isn't one of the shell's pad buttons, so it comes straight from the gamepad.
    let sprint = key.any_just_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight])
        || gamepads.iter().any(|g| g.just_pressed(GamepadButton::LeftThumb));
    if sim.in_control() {
        if key.just_pressed(KeyCode::Space) {
            sim.jump();
        }
        if sprint {
            sim.sprint();
        }
        for (slot, k) in [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4].into_iter().enumerate() {
            if key.just_pressed(k) {
                sim.use_slot(slot);
            }
        }
    }
    if key.just_pressed(KeyCode::Tab) {
        sim.auto_walk = !sim.auto_walk;
    }
    if key.just_pressed(KeyCode::KeyP) {
        sim.paused = !sim.paused;
    }
    if key.just_pressed(KeyCode::KeyE) {
        sim.use_thing();
    }
    if key.just_pressed(KeyCode::KeyN) {
        sim.reload();
    }
    if key.just_pressed(KeyCode::KeyH) {
        sim.panel = !sim.panel;
    }
}

/// The keyboard in the menus.
fn menu_keys(key: &ButtonInput<KeyCode>, sim: &mut Sim) {
    let mut moves = vec![
        (KeyCode::ArrowUp, Press::Up),
        (KeyCode::KeyW, Press::Up),
        (KeyCode::ArrowDown, Press::Down),
        (KeyCode::KeyS, Press::Down),
        (KeyCode::Enter, Press::Confirm),
        (KeyCode::NumpadEnter, Press::Confirm),
        (KeyCode::Space, Press::Confirm),
        (KeyCode::KeyE, Press::Confirm),
        (KeyCode::Escape, Press::Back),
        (KeyCode::Backspace, Press::Back),
    ];
    moves.extend([(KeyCode::ArrowLeft, Press::Left), (KeyCode::KeyA, Press::Left), (KeyCode::ArrowRight, Press::Right), (KeyCode::KeyD, Press::Right)]);
    for (slot, k) in [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4].into_iter().enumerate() {
        moves.push((k, Press::Slot(slot)));
    }
    for (k, press) in moves {
        if key.just_pressed(k) {
            sim.menu_press(press);
        }
    }
}

/// The pad in the menus. While a hotbar slot is being picked, or with square held, a d-pad
/// direction is its hotbar slot, as in play: square and a direction puts the item picked in
/// the list on that slot.
fn menu_pad(pad: &ControllerState, sim: &mut Sim) {
    if pad.pressed(Button::Start) {
        sim.menu = None;
        return;
    }
    let picking = sim.menu.as_ref().is_some_and(Menu::picking) || pad.held(Button::Square);
    let dpad = [Button::Up, Button::Right, Button::Down, Button::Left];
    for (i, b) in dpad.into_iter().enumerate() {
        if pad.pressed(b) {
            match (picking, b) {
                (true, _) => sim.menu_press(Press::Slot(i)),
                (false, Button::Up) => sim.menu_press(Press::Up),
                (false, Button::Down) => sim.menu_press(Press::Down),
                (false, Button::Left) => sim.menu_press(Press::Left),
                (false, _) => sim.menu_press(Press::Right),
            }
        }
    }
    if pad.pressed(Button::Cross) {
        sim.menu_press(Press::Confirm);
    }
    if pad.pressed(Button::Circle) {
        sim.menu_press(Press::Back);
    }
}

/// Pad buttons, sampled once per tick by psx-bevy's shell (where the developer
/// MCP's buttons arrive too).
fn pad_buttons(pad: &ControllerState, sim: &mut Sim) {
    sim.pad_arms = [pad.held(Button::R2), pad.held(Button::L2)];
    if pad.left_stick != Vec2::ZERO || pad.right_stick != Vec2::ZERO || !pad.buttons().is_empty() {
        sim.on_pad = true;
    }
    if sim.menu.is_some() {
        menu_pad(pad, sim);
        return;
    }
    if pad.held(Button::Select) {
        debug_chords(pad, sim);
        return;
    }
    if pad.pressed(Button::Start) {
        sim.toggle_menu();
        return;
    }
    if !sim.in_control() {
        return;
    }
    if pad.pressed(Button::Cross) {
        sim.jump();
    }
    if pad.pressed(Button::Circle) {
        sim.use_thing();
    }
    for (i, b) in [Button::R2, Button::L2].into_iter().enumerate() {
        if pad.pressed(b) {
            sim.press_arm(SIDES[i]);
        }
    }
    // Holding triangle will give the d-pad to the quick-change list.
    if !pad.held(Button::Triangle) {
        for (slot, b) in [Button::Up, Button::Right, Button::Down, Button::Left].into_iter().enumerate() {
            if pad.pressed(b) {
                sim.use_slot(slot);
            }
        }
    }
}

/// The developer's controls, on the pad while select is held; walking is in `walk`.
fn debug_chords(pad: &ControllerState, sim: &mut Sim) {
    if pad.pressed(Button::Triangle) {
        sim.toggle_renderer();
    }
    for (b, d) in [(Button::L1, -0.1), (Button::R1, 0.1)] {
        if pad.pressed(b) && sim.sculpted.is_some() {
            sim.dial = (sim.dial + d).clamp(0.0, 1.0);
        } else if pad.pressed(b) {
            let s = sim.strangeness + d;
            sim.set_strangeness(s);
        }
    }
    if pad.pressed(Button::Square) {
        sim.auto_walk = !sim.auto_walk;
    }
    if pad.pressed(Button::L2) {
        sim.sprint();
    }
    if pad.pressed(Button::Circle) {
        sim.debug_hurt();
    }
    if pad.pressed(Button::Cross) {
        sim.reload();
    }
}

/// The dream effects panel: a checkbox per effect, in any combination.
fn panel(mut contexts: EguiContexts, mut sim: ResMut<Sim>) -> Result {
    if !sim.panel {
        return Ok(());
    }
    let sim = &mut *sim;
    egui::Window::new("dream effects").default_pos([12.0, 60.0]).resizable(false).show(contexts.ctx_mut()?, |ui| {
        for (e, name) in effects::EFFECTS.iter().enumerate() {
            let mut on = sim.effects[e].is_some();
            if ui.checkbox(&mut on, *name).changed() {
                // Each effect's slow cycle starts again from calm when it is turned on.
                sim.effects[e] = on.then_some(0.0);
            }
        }
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("none").clicked() {
                sim.effects = [None; effects::EFFECTS.len()];
            }
            if ui.button("dream").clicked() {
                sim.effects = [None; effects::EFFECTS.len()];
                effects::DREAM.iter().for_each(|&e| sim.effects[e] = Some(0.0));
            }
        });
        ui.separator();
        let slider = egui::Slider::new(&mut sim.strangeness, 0.0..=1.0).text("strangeness");
        if ui.add(slider).changed() {
            let s = sim.strangeness;
            sim.set_strangeness(s);
        }
        ui.label("esc frees the mouse; click the view to look again");
    });
    Ok(())
}

/// The mouse in the menus: hovering picks a row, a click chooses it and right click goes back.
fn menu_mouse(
    button: Res<ButtonInput<MouseButton>>,
    egui: Res<EguiWantsInput>,
    hud: Res<Hud>,
    window: Query<&Window, With<PrimaryWindow>>,
    shown: Query<(&Sprite, &Transform), With<HudSprite>>,
    mut last: Local<Option<Vec2>>,
    mut sim: ResMut<Sim>,
) {
    if sim.menu.is_none() || egui.is_pointer_over_area() {
        return;
    }
    if button.just_pressed(MouseButton::Right) {
        sim.menu_press(Press::Back);
        return;
    }
    let (Ok(window), Ok((sprite, t))) = (window.single(), shown.single()) else { return };
    let (Some(cursor), Some(size), Some(rect)) = (window.cursor_position(), sprite.custom_size, sprite.rect) else { return };
    // The sprite is centred on its translation, with y up; the cursor is from the top left.
    let corner = window.size() / 2.0 + Vec2::new(t.translation.x, -t.translation.y) - size / 2.0;
    let at = (cursor - corner) / size * rect.size();
    let hit = hud.hits.iter().find(|([x, y, w, h], _)| {
        (*x as f32..(x + w) as f32).contains(&at.x) && (*y as f32..(y + h) as f32).contains(&at.y)
    });
    let moved = *last != Some(cursor);
    *last = Some(cursor);
    let Some(&(_, index)) = hit else { return };
    let clicked = button.just_pressed(MouseButton::Left);
    if moved || clicked {
        sim.menu.as_mut().unwrap().point(index);
    }
    if clicked {
        sim.menu_press(Press::Confirm);
    }
}

/// Frees the mouse while the menu is open, and takes it back after if it had it before.
fn menu_cursor(sim: Res<Sim>, mut was: Local<Option<bool>>, mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    let Ok(mut cursor) = cursor.single_mut() else { return };
    match (sim.menu.is_some(), *was) {
        (true, None) => {
            *was = Some(cursor.grab_mode != CursorGrabMode::None);
            set_captured(&mut cursor, false);
        }
        (false, Some(captured)) => {
            *was = None;
            if captured {
                set_captured(&mut cursor, true);
            }
        }
        _ => {}
    }
}

/// Mouse and right-stick look, every frame (not at the walk's 30 Hz) so it stays smooth.
fn mouse(
    motion: Res<AccumulatedMouseMotion>,
    pad: Res<ControllerState>,
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    cursor: Query<&CursorOptions, With<PrimaryWindow>>,
) {
    if sim.auto_walk || sim.frozen() {
        return;
    }
    let captured = cursor.single().is_ok_and(|c| c.grab_mode != CursorGrabMode::None);
    if captured && motion.delta != Vec2::ZERO {
        sim.on_pad = false;
    }
    let mut d = if captured { motion.delta * LOOK } else { Vec2::ZERO };
    // Stick up is +y; looking up is less pitch.
    let s = pad.right_stick;
    d += Vec2::new(s.x * PAD_YAW, -s.y * PAD_PITCH) * time.delta_secs();
    let w = &mut sim.level.walker;
    // Positive yaw turns left, so looking right takes yaw away.
    w.yaw -= d.x;
    w.pitch = (w.pitch + d.y).clamp(-MAX_PITCH, MAX_PITCH);
}

/// The walk, at the fixed rate the walker was made for.
fn walk(key: Res<ButtonInput<KeyCode>>, pad: Res<ControllerState>, mut sim: ResMut<Sim>, mut report: ResMut<Report>) {
    let sim = &mut *sim;
    pad_buttons(&pad, sim);
    let (feet, yaw) = sim.feet();
    *report = Report {
        feet: Vec3::from(feet),
        yaw,
        said: sim.said.as_ref().map_or(String::new(), |s| s.0.clone()),
        keys: sim.held.iter().map(|k| k.1.clone()).collect(),
        doors: sim.level.scene.things.doors.iter().map(|d| (d.entity, d.open)).collect(),
        hp: sim.player.hp,
        mp: sim.player.mp,
        items: sim.player.items.iter().map(|(i, n)| (i.name().to_string(), *n)).collect(),
        hotbar: sim.player.hotbar.iter().map(|s| s.map_or(String::new(), |i| i.name().to_string())).collect(),
        airborne: sim.level.walker.airborne(),
        sprinting: sim.level.walker.sprinting(),
        attacking: sim.hands.swinging(),
        guarding: sim.hands.guarding(),
        stunned: sim.hands.stunned(),
        stamina: sim.player.stamina,
        bob: sim.level.walker.bob(),
        affliction: sim.player.affliction,
        felt: sim.felt,
        clarity: sim.player.clarity,
        strangeness: if sim.sculpted.is_some() { sim.dial } else { sim.level.scene.strange.base },
        shown: sim.shown().name().to_string(),
        calm: sim.calm,
        veil: sim.veil,
        growths: sim.growths.as_ref().map_or(0, Growths::count),
        menu: sim.menu.as_ref().map_or(String::new(), |m| format!("{:?}", m.screen)),
        equipped: sim.player.equipped.iter().map(|s| s.map_or(String::new(), |i| i.name().to_string())).collect(),
        stats: { let s = sim.player.stats(); [s.attack, s.defence, s.magic] },
        drops: sim.drops.iter().map(|d| (d.item.name().to_string(), Vec3::from(d.at))).collect(),
    };
    if sim.frozen() {
        return;
    }
    let clear = sim.player.clarity > 0.0;
    let straining = sim.hands.guarding() || sim.level.walker.sprinting();
    sim.player.update(1.0 / TICK as f32, straining);
    if clear && sim.player.clarity == 0.0 {
        sim.say("the dream returns".to_string());
    }
    sim.fade(1.0 / TICK as f32);
    let most = 1.0 / TICK as f32 / FEEL_SECS;
    sim.felt += (sim.player.affliction - sim.felt).clamp(-most, most);
    let wanted = effects::growth(sim.showing());
    if let Some(growths) = &mut sim.growths {
        let cam = sim.level.walker.camera();
        if let Some(lash) = growths.update(&sim.level.scene, &cam, wanted, 1.0 / TICK as f32, &mut sim.rng) {
            if !sim.auto_walk {
                sim.lashed(lash);
            }
        }
    }
    sim.fight();
    if sim.level.scene.tick_things(1.0 / TICK as f32) {
        sim.level.things_moved();
    }
    let (feet, _) = sim.feet();
    sim.pick_up();
    for (id, name) in sim.level.scene.take_pickups(feet) {
        sim.said = Some((format!("picked up the {name}"), 3.0));
        sim.held.push((id, name));
        sim.level.things_moved();
    }
    if let Some((_, t)) = &mut sim.said {
        *t -= 1.0 / TICK as f32;
        if *t <= 0.0 {
            sim.said = None;
        }
    }
    let stunned = sim.hands.stunned();
    let level = &mut sim.level;
    if sim.auto_walk {
        level.walker.step(&level.scene, &mut sim.rng);
        return;
    }
    let down = |a: KeyCode, b: KeyCode| (key.pressed(a) || key.pressed(b)) as i32 as f32;
    let forward = down(KeyCode::KeyW, KeyCode::ArrowUp) - down(KeyCode::KeyS, KeyCode::ArrowDown);
    let strafe = down(KeyCode::KeyD, KeyCode::KeyD) - down(KeyCode::KeyA, KeyCode::KeyA);
    // Positive turn is to the left.
    let mut turn = down(KeyCode::ArrowLeft, KeyCode::ArrowLeft) - down(KeyCode::ArrowRight, KeyCode::ArrowRight);
    let (mut forward, mut strafe) = (forward, strafe);
    let held = |b: Button| pad.held(b) as i32 as f32;
    forward += pad.left_stick.y;
    strafe += pad.left_stick.x;
    if pad.held(Button::Select) {
        forward += held(Button::Up) - held(Button::Down);
        turn += held(Button::Left) - held(Button::Right);
    }
    let turn = turn.clamp(-1.0, 1.0);
    // Stunned, the player can only look about.
    let (forward, strafe, turn) = if stunned { (0.0, 0.0, 0.0) } else { (forward, strafe, turn) };
    let len = (forward * forward + strafe * strafe).sqrt().max(1.0);
    // The pitch is the mouse's: passing it as the target leaves it where it is.
    let look = level.walker.pitch;
    level.walker.drive(&level.scene, forward / len, strafe / len, turn, look);
    if level.walker.sprinting() {
        sim.player.tire(SPRINT_TIRE / TICK as f32);
        if sim.player.stamina <= 0.0 {
            level.walker.stop_sprint();
        }
    }
}

/// RGB floats (roughly 0..1, the network's) into an RGBA8 image.
fn upload_floats(image: &mut Image, rgb: &[f32], width: u32) {
    let stride = MAX_W as usize * 4;
    let data = image.data.get_or_insert_with(Vec::new);
    data.resize(stride * H as usize, 255);
    for (row, from) in data.chunks_exact_mut(stride).zip(rgb.chunks_exact(width as usize * 3)) {
        for (px, c) in row.chunks_exact_mut(4).zip(from.chunks_exact(3)) {
            for i in 0..3 {
                px[i] = (c[i].clamp(0.0, 1.0) * 255.0) as u8;
            }
            px[3] = 255;
        }
    }
}

fn draw(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut net: NonSendMut<Network>,
    mut dream: Option<NonSendMut<engine::neural::dream_net::DreamLayer>>,
    frame: Res<Frame>,
    mut creatures: ResMut<Creatures>,
    mut shown: ResMut<engine::render::DreamCreatures>,
    mut images: ResMut<Assets<Image>>,
    mut sprite: Query<&mut Sprite, With<NetworkSprite>>,
) {
    if !sim.frozen() {
        let dt = time.delta_secs();
        sim.effects.iter_mut().flatten().for_each(|t| *t += dt);
        sim.dream_time += dt;
        if dream.is_some() {
            let sim = &mut *sim;
            sim.herd.player = (!sim.auto_walk).then(|| sim.feet().0);
            let cues = sim.herd.update(&mut sim.level.scene, sim.version, dt);
            sim.cues.extend(cues);
        }
    }
    if sim.shown() != Mode::Raster {
        let t0 = std::time::Instant::now();
        let width = sim.width;
        let sim = &mut *sim;
        let cam = sim.level.walker.camera();
        // All this drains away with the strangeness when the player sees clearly.
        let swelling = effects::Swelling::of(sim.showing(), sim.dream_time, &cam);
        swelling.tris(&sim.level.tris, &mut sim.swollen);
        let mut tokens = sim.level.tokens_of(&sim.swollen, width);
        let disturbed = effects::apply_set(&sim.effects, &mut tokens, &sim.swollen);
        let mut rgb = net.render(&tokens, &disturbed);
        if let (Some(layer), Some(growths)) = (dream.as_mut(), &sim.growths) {
            if let Err(e) = layer.draw(&mut sim.level.scene, sim.version, &sim.swollen, growths.creature(), |p| swelling.tiles(p), &cam, &mut rgb, width) {
                eprintln!("dream layer: {e}");
            }
        }
        if let Some(mut image) = images.get_mut(&frame.0) {
            upload_floats(&mut image, &rgb, width);
        }
        if let Ok(mut sprite) = sprite.single_mut() {
            sprite.rect = Some(Rect::new(0.0, 0.0, width as f32, H as f32));
        }
        sim.tris = tokens.len();
        sim.net_ms = sim.net_ms * 0.9 + 0.1 * t0.elapsed().as_secs_f32() * 1000.0;
    }
    // The dream creatures in the rasteriser's picture, unless the player sees clearly. The
    // rasteriser doesn't swell the level, so neither are they; the flesh pass hides them
    // behind the level and its flesh.
    let mut on = false;
    if sim.shown() != Mode::Network && !sim.seeing_clearly() {
        let sim = &mut *sim;
        if let (Some(layer), Some(growths)) = (dream.as_mut(), &sim.growths) {
            let cam = sim.level.walker.camera();
            let Creatures { rgba, near } = &mut *creatures;
            match layer.draw_alone(&mut sim.level.scene, sim.version, &[], growths.creature(), |p| p, &cam, rgba, near, W) {
                Ok(()) => on = true,
                Err(e) => eprintln!("dream layer: {e}"),
            }
        }
        if on {
            for (image, floats) in [(&shown.colour, &creatures.rgba), (&shown.near, &creatures.near)] {
                if let Some(mut image) = images.get_mut(image) {
                    image.data = Some(floats.iter().flat_map(|f| f.to_ne_bytes()).collect());
                }
            }
        }
    }
    if shown.on != on {
        shown.on = on;
    }
}

/// The scene's meshes, rebuilt when it changes.
fn build_scene(mut commands: Commands, sim: Res<Sim>, mut parts: ResMut<Parts>, mut gfx: engine::render::Gfx) {
    // With no strangeness in force the lights no longer depend on `look`.
    let now = (sim.version, if sim.calm >= 1.0 { u64::MAX } else { sim.look });
    // New lights can wait until the rasteriser is on show.
    if parts.version == Some(now) || (parts.version.is_some_and(|v| v.0 == now.0) && !sim.raster_on()) {
        return;
    }
    for e in parts.entities.drain(..) {
        commands.entity(e).despawn();
    }
    parts.entities = engine::render::spawn(&mut commands, &mut gfx, &sim.level.scene);
    parts.version = Some(now);
    parts.things = sim.level.scene.things.parts.iter().map(|p| p.version).collect();
}

/// The meshes of each door, key or lever that moved since they were made.
fn move_things(
    mut commands: Commands,
    sim: Res<Sim>,
    mut parts: ResMut<Parts>,
    shown: Query<(Entity, &engine::render::ThingPart)>,
    mut gfx: engine::render::Gfx,
) {
    let scene = &sim.level.scene;
    for (i, p) in scene.things.parts.iter().enumerate() {
        if parts.things.get(i) == Some(&p.version) {
            continue;
        }
        for (e, t) in &shown {
            if t.0 == i {
                commands.entity(e).despawn();
                parts.entities.retain(|&x| x != e);
            }
        }
        let new = engine::render::spawn_thing(&mut commands, &mut gfx, scene, i);
        parts.entities.extend(new);
        if let Some(v) = parts.things.get_mut(i) {
            *v = p.version;
        }
    }
}

/// How many times a second, at most, the sculpted level's flesh is remade while the dial turns.
const REMAKE_RATE: f32 = 20.0;
/// How much of the sculpted level's dial [ and ] move a second.
const DIAL_SPEED: f32 = 0.25;

/// The sculpted level's flesh (with `--sculpt`): its walls and what moves on them (the eyes,
/// veins and sphincters), each a mesh of its own remade only when the dial turns, so that the
/// rest of the level's flesh stays put; the GPU moves the eyes and the rest every frame.
fn pose_sculpted(
    mut commands: Commands,
    mut sim: ResMut<Sim>,
    time: Res<Time>,
    key: Res<ButtonInput<KeyCode>>,
    mut moving: Local<Option<(Entity, Handle<bevy::render::storage::ShaderBuffer>, engine::world::dreamed::Moving)>>,
    mut walls: Local<Option<Entity>>,
    mut walls_of: Local<Option<(AssetId<engine::render::flesh::FleshMaterial>, f32)>>,
    mut since: Local<f32>,
    // The scene's own only: what's remade may still have the last level's skins.
    flesh: Query<&MeshMaterial3d<engine::render::flesh::FleshMaterial>, Without<engine::render::flesh::MovingFlesh>>,
    mut gfx: engine::render::Gfx,
) {
    let sim = &mut *sim;
    if sim.sculpted.is_none() {
        return;
    }
    let dt = time.delta_secs();
    let held = key.pressed(KeyCode::BracketRight) as i32 as f32 - key.pressed(KeyCode::BracketLeft) as i32 as f32;
    sim.dial = (sim.dial + held * DIAL_SPEED * dt).clamp(0.0, 1.0);
    *since += dt;
    let Some(sculpted) = &mut sim.sculpted else { return };
    // Clear seeing takes the flesh back into the stone as well.
    sculpted.amount = sim.dial * (1.0 - sim.calm);
    let Some(material) = flesh.iter().next() else { return };
    let scene = &mut sim.level.scene;
    let now = (material.id(), sculpted.amount);
    // Remaking the meshes is the slow part of a frame: not every frame while the dial turns.
    if *walls_of != Some(now) && *since >= 1.0 / REMAKE_RATE {
        *since = 0.0;
        for e in walls.take().into_iter().chain(moving.take().map(|m| m.0)) {
            commands.entity(e).despawn();
        }
        let tris = engine::world::dreamed::walls(sculpted);
        if !tris.is_empty() {
            *walls = Some(gfx.spawn_moving_flesh(&mut commands, material, &scene.flesh, &tris));
        }
        let parts = engine::world::dreamed::Moving::new(scene, sculpted);
        if !parts.tris.is_empty() {
            *moving = gfx.spawn_moving_parts(&mut commands, material, &scene.flesh, &parts).map(|(e, h)| (e, h, parts));
        }
        *walls_of = Some(now);
    }
    if let Some((_, handle, parts)) = &mut *moving {
        parts.pose(sculpted, &sim.level.walker.camera(), sim.dream_time);
        gfx.set_moving_parts(handle, parts);
    }
}

/// How far a creature can reach from where it stands, world units: two tiles.
const CREATURE_REACH: f32 = 512.0;

/// While a clear seeing potion has the rasteriser showing, the creatures as the herd has
/// walked them, as low-poly PS1 models made afresh each frame; otherwise the dream net
/// draws them, over the network's picture or the rasteriser's (`draw`).
fn pose_creatures(
    mut commands: Commands,
    sim: Res<Sim>,
    mut shown: Local<Vec<Entity>>,
    // Each skin's material, for the level `Sim::version`.
    mut skins: Local<(u64, HashMap<u8, Handle<psx_bevy::psx::PsxMaterial>>)>,
    mut gfx: engine::render::Gfx,
) {
    for e in shown.drain(..) {
        commands.entity(e).despawn();
    }
    let scene = &sim.level.scene;
    if !scene.creatures_move || !sim.raster_on() || !sim.seeing_clearly() {
        return;
    }
    if skins.0 != sim.version {
        *skins = (sim.version, HashMap::new());
    }
    // Only those that could be in view: ahead of the camera and short of the fog.
    let cam = sim.level.walker.camera();
    let eye = cam.pos.map(|c| c as f32);
    let ahead = engine::world::view::view_matrix(&cam).m[2].map(|e| e as f32 / 4096.0);
    let reach = cam.fog_far as f32 + CREATURE_REACH;
    let in_view = |c: &engine::world::creature::Creature| {
        let d = [c.base[0] - eye[0], c.base[1] - eye[1], c.base[2] - eye[2]];
        let far = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        far < reach && d[0] * ahead[0] + d[1] * ahead[1] + d[2] * ahead[2] > -CREATURE_REACH
    };
    let tris = scene.low_poly_creatures(in_view);
    *shown = gfx.spawn_creatures(&mut commands, scene, &tris, &mut skins.1);
}

/// The level camera follows the walker; it rests while only the network is shown.
fn raster_camera(sim: Res<Sim>, mut view: ResMut<engine::render::View>) {
    let cam = sim.level.walker.camera();
    let v = engine::render::View { hidden: !sim.raster_on(), ..engine::render::View::of(&sim.level.scene, &cam) };
    view.set_if_neq(v);
}

/// Give egui psx-bevy's window camera (the one showing the screen), once it exists.
fn egui_on_display(
    mut commands: Commands,
    cams: Query<(Entity, &RenderLayers), (With<Camera2d>, Without<PrimaryEguiContext>)>,
    done: Query<(), With<PrimaryEguiContext>>,
) {
    if !done.is_empty() {
        return;
    }
    for (e, layers) in &cams {
        if layers.intersects(&RenderLayers::layer(DISPLAY_LAYER)) {
            commands.entity(e).insert(PrimaryEguiContext);
        }
    }
}

/// Fit the shown frame(s) to the window, keeping 4:3: psx-bevy's display of its
/// screen for the rasteriser, the network's sprite over it. Both: rasteriser on
/// the left, network on the right.
/// The narrowest and widest frames the level network was trained on, at 240 high.
const NET_WIDTHS: (f32, f32) = (W as f32, MAX_W as f32);

fn layout(
    mut sim: ResMut<Sim>,
    screen: Res<PsxScreen>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut network: Query<(&mut Sprite, &mut Transform, &mut Visibility), With<NetworkSprite>>,
    mut hud: Query<(&mut Sprite, &mut Transform), (With<HudSprite>, Without<NetworkSprite>)>,
    mut display: Query<(&Sprite, &mut Transform), (Without<NetworkSprite>, Without<HudSprite>)>,
) {
    let Ok(window) = window.single() else { return };
    let mode = sim.shown();
    let frames = if mode == Mode::Both { 2.0 } else { 1.0 };
    let room = window.size() / Vec2::new(frames, 1.0);
    let half = if mode == Mode::Both { room.x / 2.0 } else { 0.0 };
    // The rasteriser stays 4:3; the network is as wide as the window asks, within what it knows.
    let raster = fit_scale(room, Vec2::new(W as f32, H as f32));
    for (sprite, mut t) in &mut display {
        if sprite.image == screen.image {
            t.scale = Vec3::splat(raster);
            t.translation.x = -half;
        }
    }
    let wanted = (room.x / room.y.max(1.0) * H as f32).clamp(NET_WIDTHS.0, NET_WIDTHS.1);
    // The frame is a multiple of 16 wide, and stretched by at most 2.5% to fill the window.
    let width = ((wanted / 16.0).round() as u32 * 16).clamp(W, MAX_W);
    if sim.width != width {
        sim.width = width;
    }
    let size = Vec2::new(wanted, H as f32);
    let scale = fit_scale(room, size);
    let Ok((mut ns, mut nt, mut nv)) = network.single_mut() else { return };
    ns.custom_size = Some(size * scale);
    nt.translation.x = half;
    *nv = if mode == Mode::Raster { Visibility::Hidden } else { Visibility::Visible };
    let opaque = if sim.mode == Mode::Network { 1.0 - sim.veil } else { 1.0 };
    if ns.color.alpha() != opaque {
        ns.color = Color::WHITE.with_alpha(opaque);
    }
    // The HUD sits on whichever picture is shown, at its resolution.
    if let Ok((mut hs, mut ht)) = hud.single_mut() {
        if mode == Mode::Raster {
            hs.rect = Some(Rect::new(0.0, 0.0, W as f32, H as f32));
            hs.custom_size = Some(Vec2::new(W as f32, H as f32) * raster);
            ht.translation.x = 0.0;
        } else {
            hs.rect = Some(Rect::new(0.0, 0.0, width as f32, H as f32));
            hs.custom_size = Some(size * scale);
            ht.translation.x = half;
        }
    }
}

/// Redraw the HUD's image when what it shows has changed.
fn update_hud(time: Res<Time>, mut sim: ResMut<Sim>, mut hud: ResMut<Hud>, mut images: ResMut<Assets<Image>>) {
    let sim = &mut *sim;
    if !sim.frozen() {
        sim.hands.hold(sim.player.hand(Side::Right), sim.player.hand(Side::Left));
        let control = sim.in_control();
        let held = [0, 1].map(|i| control && (sim.mouse_arms[i] || sim.pad_arms[i]));
        for (side, full) in sim.hands.update(time.delta_secs(), sim.level.walker.yaw, held) {
            let attack = sim.player.attack_with(side);
            let blow = if full { attack } else { (attack / LABOURED).max(1) };
            *sim.struck.get_or_insert(0) += blow;
        }
        sim.level.walker.kick = sim.hands.kick();
    }
    let width = if sim.shown() == Mode::Raster { W } else { sim.width };
    let yaw = sim.level.walker.yaw / 4096.0 * std::f32::consts::TAU;
    let mut state = HudState::of(&sim.player, sim.menu.as_ref(), width, sim.on_pad, yaw, sim.hands.shown(sim.level.walker.bob()));
    if !sim.drops.is_empty() {
        state.drops = drops::triangles(&sim.drops, &sim.level.scene, &sim.level.walker.camera(), width);
    }
    if hud.shown.as_ref() == Some(&state) {
        return;
    }
    let Some(mut image) = images.get_mut(&hud.image) else { return };
    if let Some(data) = image.data.as_mut() {
        hud.hits = hud::draw(data, MAX_W, H, &state);
        hud.shown = Some(state);
    }
}

/// `GAME_SHOT=out.png:5` saves a screenshot of the window after 5 seconds and
/// quits, for checking the game from scripts.
fn auto_screenshot(mut commands: Commands, time: Res<Time>, mut frames: Local<u32>, mut exit: MessageWriter<AppExit>) {
    use bevy::render::view::screenshot::{save_to_disk, Screenshot};
    let Ok(spec) = std::env::var("GAME_SHOT") else { return };
    let (path, secs) = spec.rsplit_once(':').unwrap_or((&spec, "5"));
    if *frames == 0 && time.elapsed_secs() > secs.parse().unwrap_or(5.0) {
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path.to_string()));
        *frames = 1;
    } else if *frames > 0 {
        // Give the capture a moment to reach the disk.
        *frames += 1;
        if *frames > 30 {
            exit.write(AppExit::Success);
        }
    }
}

/// How often the frame rate on show is counted afresh, seconds.
const FPS_EVERY: f32 = 0.5;

/// `counted`: the seconds and frames since the frame rate was last counted, and what it was.
fn status(sim: Res<Sim>, time: Res<Time>, mut text: Query<&mut Text, With<Status>>, mut counted: Local<(f32, u32, f32)>) {
    let Ok(mut text) = text.single_mut() else { return };
    // Counted over a while, not from the last frame alone: frames can come unevenly (the
    // sculpted level's flesh is remade while its dial turns).
    let (secs, frames, fps) = &mut *counted;
    *secs += time.delta_secs();
    *frames += 1;
    if *secs >= FPS_EVERY {
        *fps = *frames as f32 / *secs;
        (*secs, *frames) = (0.0, 0);
    }
    let fps = *fps;
    let mode = sim.shown().name();
    let walk = if sim.auto_walk { "auto-walk" } else { "walking" };
    let pause = if sim.paused { " | frozen" } else if sim.menu.is_some() { " | menu" } else { "" };
    let on = sim.effects.iter().flatten().count();
    let keys = match sim.held.len() {
        0 => String::new(),
        _ => format!(" | keys: {}", sim.held.iter().map(|k| k.1.as_str()).collect::<Vec<_>>().join(", ")),
    };
    let said = sim.said.as_ref().map_or(String::new(), |s| format!("\n{}", s.0));
    let dial = if sim.sculpted.is_some() { format!("flesh {:.0}% | ", sim.dial * 100.0) } else { String::new() };
    text.0 = format!(
        "{mode} | {on} effects | {fps:.0} fps | net {:.1} ms, {} tris | affliction {:.0}% | {dial}{walk}{pause}{keys}\n\
         mouse: look  lmb / rmb: left / right hand  wasd: move  space: jump  shift: sprint  e: use  1-4: hotbar  esc: menu  f1: switch  f2: both  f3: hurt  f4: free mouse  h: panel{said}",
        sim.net_ms, sim.tris, sim.player.affliction * 100.0
    );
}
