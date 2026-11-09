//! A first-person walk through a generated level, drawn by the trained network, in Bevy.
//!
//! `cargo run -p game [-- --seed 7]`
//!     The model, skins and textures come from `neurofield.ron` (`engine::world::config`);
//!     options of the same name override it: `--model x.safetensors`, `--photos DIR`,
//!     `--strange DIR`, `--level DIR` (level textures alone).
//!     `--dream x.safetensors` draws the creatures with the dream net (train/export_dream.py).
//!     `--window 1280x720` sets the window's starting size; the frame follows its shape.
//!     `--view raster` (or `both`) starts on the rasteriser instead of the network.
//!     `--strangeness 0.7` sets how strange each level is (otherwise random).
//!     `--level-file levels/x.ron` plays a hand-built level (see `engine::world::level`) from its
//!     player start, or from `--start x,y,z,yaw` (world units, yaw in radians); N reloads it.
//!     E (or circle) opens and closes the door in front, or pulls the lever; walking over a
//!     key picks it up, and locked doors open for their key.
//!     Mouse: look   Right click: attack   WASD: walk and strafe   Arrows: walk and turn   Space: jump
//!     Shift: sprint   1-4: use the hotbar   Esc: free the mouse (click to capture it
//!     again; Esc again quits)
//!     Gamepad: left stick walk and strafe, right stick look, cross jump, L3 sprint,
//!     R2 attack, circle use, d-pad use the hotbar, start pause
//!     Debug keys: F1 network / rasteriser   F2 both side by side   F3 be hit   Tab auto-walk
//!     H show the effects panel   N new level   P pause
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
use engine::world::walker::{self, MAX_PITCH};
use psx_bevy::controller::{BindingDevice, Button, ControllerBindings, ControllerState};
use psx_bevy::psx::{fit_scale, PsxPlugin, PsxScreen, DISPLAY_LAYER};
use psx_bevy::shell::{GameSystems, ShellPlugin, TickRate};
use std::collections::HashMap;
use std::path::PathBuf;
use engine::neural::gpu_net::MAX_W;
use engine::neural::{effects, Level, Network, TICK};

mod hud;
mod player;
mod sword;
use hud::HudState;
use player::Player;
use sword::Sword;

/// Mouse sensitivity: yaw units (4096 per turn) per pixel of mouse movement.
const LOOK: f32 = 1.2;
/// Right stick look speed at full tilt: yaw and pitch units per second.
const PAD_YAW: f32 = 2600.0;
const PAD_PITCH: f32 = 1400.0;
/// Affliction reaches the level in this many steps: each bakes the level's lights again.
const AFFLICTION_STEPS: f32 = 50.0;
/// Seconds the strangeness takes to drain away, or come back, under the network.
const CALM_SECS: f32 = 1.5;
/// Seconds the network's calm picture takes to fade to the plain one, and back.
const FADE_SECS: f32 = 0.8;

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
    /// The step of affliction the level was last lit for (`apply_affliction`).
    applied: Option<i32>,
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
    sword: Sword,
    /// Whether the controller was the last thing used, for the shape of the hotbar.
    on_pad: bool,
}

impl Sim {
    fn set_strangeness(&mut self, s: f32) {
        self.strangeness = s.clamp(0.0, 1.0);
        engine::world::scene::force_strangeness(Some(self.strangeness));
        self.applied = None;
        self.apply_affliction();
    }

    /// Light the level for its base strangeness and the player's affliction, which takes
    /// it from there towards fully strange. Only a new step of affliction does anything.
    fn apply_affliction(&mut self) {
        // A jump in the strangeness would show as it drains: it can wait until that's done.
        if self.player.clarity > 0.0 && self.calm < 1.0 {
            return;
        }
        let step = (self.player.affliction * AFFLICTION_STEPS).round() as i32;
        if self.applied == Some(step) {
            return;
        }
        self.applied = Some(step);
        let base = self.strangeness;
        self.level.set_strangeness(base + (1.0 - base) * step as f32 / AFFLICTION_STEPS);
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

    /// Use whatever is in front: a door or a lever.
    fn use_thing(&mut self) {
        let (feet, yaw) = self.feet();
        let held = self.held.clone();
        if let Some(what) = self.level.scene.use_thing(feet, yaw, &held) {
            self.say(what);
        }
    }

    /// A new level, or the file again (it may have been edited), its own strangeness
    /// (a file keeps the slider's) with the player's affliction on top.
    fn reload(&mut self) {
        self.version += 1;
        self.held.clear();
        match &self.file {
            Some(f) => match load_level(f, self.start, &mut self.rng) {
                Ok(level) => self.level = level,
                Err(e) => eprintln!("{e}"),
            },
            None => {
                self.level = Level::new(&mut self.rng);
                self.strangeness = self.level.scene.strange.base;
            }
        }
        self.level.set_strange_amount(1.0 - self.calm);
        self.applied = None;
        self.apply_affliction();
    }

    fn say(&mut self, what: String) {
        self.said = Some((what, 3.0));
    }

    /// Whether the player's own actions count: not paused, not on auto-walk.
    fn in_control(&self) -> bool {
        !self.paused && !self.auto_walk
    }

    fn jump(&mut self) {
        let level = &mut self.level;
        level.walker.jump(&level.scene);
    }

    fn use_slot(&mut self, slot: usize) {
        if let Some(what) = self.player.use_slot(slot) {
            self.say(what);
            self.apply_affliction();
        }
    }

    /// What a hit will do, so it can all be tried before anything can hurt.
    fn debug_hurt(&mut self) {
        self.player.hurt(5);
        self.player.spend_mp(2);
        self.player.afflict(0.1);
    }

    /// The walker's feet, and which way it faces in radians.
    fn feet(&self) -> ([f32; 3], f32) {
        let w = &self.level.walker;
        ([w.x, w.y + walker::EYE, w.z], w.yaw / 4096.0 * std::f32::consts::TAU)
    }
}

fn load_level(file: &std::path::Path, start: Option<([f32; 3], f32)>, rng: &mut Rng) -> Result<Level, String> {
    let level = engine::world::level::Level::load(file)?;
    Level::from_file(&level, start, rng).map_err(|e| format!("{}: {e}", file.display()))
}

/// The network's 320x240 frame.
#[derive(Resource)]
struct Frame(Handle<Image>);

/// Shows the network's frame in the window, over psx-bevy's display of its screen.
#[derive(Component)]
struct NetworkSprite;

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
    /// The walking bob's offset of the camera, in world units.
    bob: f32,
    affliction: f32,
    clarity: f32,
    /// The level's strangeness at its base, affliction included, and the picture on show.
    strangeness: f32,
    shown: String,
    /// How much of the strangeness has drained, and how far the plain picture has faded in.
    calm: f32,
    veil: f32,
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
    config.load_textures(&opts);
    if let Some(s) = opts.get("strangeness") {
        engine::world::scene::force_strangeness(Some(s.parse().expect("--strangeness 0..1")));
    }
    let net = Network::load(&model);
    // The dream creatures, drawn by their own network over the level network's frames.
    let dream = opts.get("dream").map(|d| {
        let layer = engine::neural::dream_net::DreamLayer::load(std::path::Path::new(d), seed).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("dream creatures on {}", layer.adapter());
        layer
    });
    let mut rng = Rng::new(seed);
    let start = opts.get("start").map(|s| {
        let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().expect("--start x,y,z,yaw")).collect();
        assert_eq!(v.len(), 4, "--start x,y,z,yaw");
        ([v[0], v[1], v[2]], v[3])
    });
    let file = opts.get("level-file").map(PathBuf::from);
    let level = match &file {
        Some(f) => load_level(f, start, &mut rng).unwrap_or_else(|e| panic!("{e}")),
        None => Level::new(&mut rng),
    };
    let strangeness = level.scene.strange.base;

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
                None | Some("network") => Mode::Network,
                Some("raster") => Mode::Raster,
                Some("both") => Mode::Both,
                Some(v) => panic!("--view network|raster|both, not {v}"),
            },
            effects: [None; effects::EFFECTS.len()],
            panel: false,
            net_ms: 0.0,
            herd: Default::default(),
            tris: 0,
            width: W,
            strangeness,
            applied: Some(0),
            calm: 0.0,
            veil: 0.0,
            file,
            start,
            version: 0,
            look: 0,
            held: Vec::new(),
            said: None,
            player: Player::new(),
            sword: Sword::default(),
            on_pad: false,
        })
        .insert_non_send(net)
        .init_resource::<Parts>()
        .add_systems(Startup, (setup, capture_mouse))
        .add_systems(Update, (keys, mouse, draw, build_scene, move_things, raster_camera, status, update_hud, auto_screenshot).chain())
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
    let hud = images.add(hud_image());
    commands.spawn((
        Sprite::from_image(hud.clone()),
        Transform::from_xyz(0.0, 0.0, 2.0),
        RenderLayers::layer(DISPLAY_LAYER),
        HudSprite,
    ));
    commands.insert_resource(Hud { image: hud, shown: None });
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
    if let Ok(mut cursor) = cursor.single_mut() {
        let captured = cursor.grab_mode != CursorGrabMode::None;
        if captured && button.just_pressed(MouseButton::Right) && sim.in_control() {
            sim.sword.attack();
        }
        if key.just_pressed(KeyCode::Escape) {
            if captured {
                set_captured(&mut cursor, false);
            } else {
                exit.write(AppExit::Success);
            }
        }
        if button.just_pressed(MouseButton::Left) && !captured && !egui.is_pointer_over_area() {
            set_captured(&mut cursor, true);
        }
    }
    if key.get_just_pressed().next().is_some() {
        sim.on_pad = false;
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
            sim.level.walker.start_sprint();
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

/// Pad buttons, sampled once per tick by psx-bevy's shell (where the developer
/// MCP's buttons arrive too).
fn pad_buttons(pad: &ControllerState, sim: &mut Sim) {
    if pad.left_stick != Vec2::ZERO || pad.right_stick != Vec2::ZERO || !pad.buttons().is_empty() {
        sim.on_pad = true;
    }
    if pad.held(Button::Select) {
        debug_chords(pad, sim);
        return;
    }
    if pad.pressed(Button::Start) {
        sim.paused = !sim.paused;
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
    if pad.pressed(Button::R2) {
        sim.sword.attack();
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
        if pad.pressed(b) {
            let s = sim.strangeness + d;
            sim.set_strangeness(s);
        }
    }
    if pad.pressed(Button::Square) {
        sim.auto_walk = !sim.auto_walk;
    }
    if pad.pressed(Button::L2) {
        sim.level.walker.start_sprint();
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

/// Mouse and right-stick look, every frame (not at the walk's 30 Hz) so it stays smooth.
fn mouse(
    motion: Res<AccumulatedMouseMotion>,
    pad: Res<ControllerState>,
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    cursor: Query<&CursorOptions, With<PrimaryWindow>>,
) {
    if sim.auto_walk || sim.paused {
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
        attacking: sim.sword.swinging(),
        bob: sim.level.walker.bob(),
        affliction: sim.player.affliction,
        clarity: sim.player.clarity,
        strangeness: sim.level.scene.strange.base,
        shown: sim.shown().name().to_string(),
        calm: sim.calm,
        veil: sim.veil,
    };
    if sim.paused {
        return;
    }
    let clear = sim.player.clarity > 0.0;
    sim.player.update(1.0 / TICK as f32);
    if clear && sim.player.clarity == 0.0 {
        sim.say("the dream returns".to_string());
    }
    sim.fade(1.0 / TICK as f32);
    sim.apply_affliction();
    if sim.level.scene.tick_things(1.0 / TICK as f32) {
        sim.level.things_moved();
    }
    let (feet, _) = sim.feet();
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
    let len = (forward * forward + strafe * strafe).sqrt().max(1.0);
    // The pitch is the mouse's: passing it as the target leaves it where it is.
    let look = level.walker.pitch;
    level.walker.drive(&level.scene, forward / len, strafe / len, turn, look);
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
    mut images: ResMut<Assets<Image>>,
    mut sprite: Query<&mut Sprite, With<NetworkSprite>>,
) {
    if !sim.paused {
        let dt = time.delta_secs();
        sim.effects.iter_mut().flatten().for_each(|t| *t += dt);
        if dream.is_some() {
            let sim = &mut *sim;
            sim.herd.update(&mut sim.level.scene, sim.version, dt);
        }
    }
    if sim.shown() != Mode::Raster {
        let t0 = std::time::Instant::now();
        let width = sim.width;
        let mut tokens = sim.level.tokens_at(width);
        let calm = effects::apply_set(&sim.effects, &mut tokens, &sim.level.tris);
        let mut rgb = net.render(&tokens, &calm);
        if let Some(layer) = dream.as_mut() {
            let cam = sim.level.walker.camera();
            if let Err(e) = layer.draw(&sim.level.scene, sim.version, &cam, &mut rgb, width) {
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
    if !sim.paused {
        sim.sword.update(time.delta_secs(), sim.level.walker.yaw);
    }
    let width = if sim.shown() == Mode::Raster { W } else { sim.width };
    let yaw = sim.level.walker.yaw / 4096.0 * std::f32::consts::TAU;
    let state = HudState::of(&sim.player, width, sim.on_pad, yaw, sim.sword.pose(sim.level.walker.bob()));
    if hud.shown.as_ref() == Some(&state) {
        return;
    }
    let Some(mut image) = images.get_mut(&hud.image) else { return };
    if let Some(data) = image.data.as_mut() {
        hud::draw(data, MAX_W, H, &state);
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

fn status(sim: Res<Sim>, time: Res<Time>, mut text: Query<&mut Text, With<Status>>) {
    let Ok(mut text) = text.single_mut() else { return };
    let fps = 1.0 / time.delta_secs().max(1e-6);
    let mode = sim.shown().name();
    let walk = if sim.auto_walk { "auto-walk" } else { "walking" };
    let pause = if sim.paused { " | paused" } else { "" };
    let on = sim.effects.iter().flatten().count();
    let keys = match sim.held.len() {
        0 => String::new(),
        _ => format!(" | keys: {}", sim.held.iter().map(|k| k.1.as_str()).collect::<Vec<_>>().join(", ")),
    };
    let said = sim.said.as_ref().map_or(String::new(), |s| format!("\n{}", s.0));
    text.0 = format!(
        "{mode} | {on} effects | {fps:.0} fps | net {:.1} ms, {} tris | affliction {:.0}% | {walk}{pause}{keys}\n\
         mouse: look  rmb: attack  wasd: move  space: jump  shift: sprint  e: use  1-4: hotbar  f1: switch  f2: both  f3: hurt  h: panel  esc: free mouse{said}",
        sim.net_ms, sim.tris, sim.player.affliction * 100.0
    );
}
