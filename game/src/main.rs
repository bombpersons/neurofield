//! A first-person walk through a generated level, drawn by the trained network, in Bevy.
//!
//! `cargo run -p game [-- --seed 7]`
//!     The model, skins and textures come from `neurofield.ron` (`engine::world::config`);
//!     options of the same name override it: `--model x.safetensors`, `--photos DIR`,
//!     `--strange DIR`, `--level DIR` (level textures alone).
//!     `--view raster` (or `both`) starts on the rasteriser instead of the network.
//!     `--strangeness 0.7` sets how strange each level is (otherwise random).
//!     `--level-file levels/x.ron` plays a hand-built level (see `engine::world::level`) from its
//!     player start, or from `--start x,y,z,yaw` (world units, yaw in radians); N reloads it.
//!     E (or cross) opens and closes the door in front, or pulls the lever; walking over a
//!     key picks it up, and locked doors open for their key.
//!     Mouse: look   WASD: walk and strafe   Arrows: walk and turn   Tab: auto-walk
//!     Space: network / rasteriser   B: both side by side   H: hide the effects panel
//!     N: new level   P: pause   Esc: free the mouse (click to capture it again; Esc again quits)
//!     Gamepad: left stick walk and strafe, right stick look, d-pad walk and turn,
//!     L2 / R2 strafe, triangle network / rasteriser, square auto-walk, L1 / R1
//!     strangeness down / up, select new level (or reload), start pause
//!
//! The dream effects panel (egui) turns effects on and off in any combination; free the
//! mouse with Esc to use it. The network runs in the main world each frame and its 320x240
//! image is uploaded to a texture, drawn with nearest sampling and scaled to fit the window.
//! The rasteriser is psx-bevy drawing the level (`engine::render`) into its 320x240 screen.
//! `GAME_SHOT=out.png:5` saves a screenshot after 5 seconds and quits.
//!
//! Developer builds (`--features dev-mcp`) take `--mcp` for psx-bevy's developer MCP
//! on stdio (see psx-bevy's README, "Agent driving"), with `--headless` and `--software`
//! to run without a window. Under it the walk starts paused; `game_step` holds pad
//! buttons for a number of ticks (up/down walk, left/right turn, l2/r2 strafe, cross
//! uses), `game_screenshot {"target":"screen"}` grabs the rasteriser's frame, and
//! `game_query {"resources":["Report"]}` reads where the walker is and what happened.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::image::ImageSampler;
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
use engine::neural::{effects, Level, Network, TICK};

/// Mouse sensitivity: yaw units (4096 per turn) per pixel of mouse movement.
const LOOK: f32 = 1.2;
/// Right stick look speed at full tilt: yaw and pitch units per second.
const PAD_YAW: f32 = 2600.0;
const PAD_PITCH: f32 = 1400.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Network,
    Raster,
    Both,
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
    tris: usize,
    /// How strange levels are at their base (the egui slider; new levels keep it).
    strangeness: f32,
    /// A hand-built level being played, and where to start in it (else its player start).
    file: Option<PathBuf>,
    start: Option<([f32; 3], f32)>,
    /// Bumped whenever the scene changes (new level, strangeness), so its meshes are rebuilt.
    version: u64,
    /// Keys picked up in this level: entity ids and names.
    held: Vec<(u32, String)>,
    /// What last happened (a door, a key), and seconds left to show it.
    said: Option<(String, f32)>,
}

impl Sim {
    fn set_strangeness(&mut self, s: f32) {
        self.strangeness = s.clamp(0.0, 1.0);
        engine::world::scene::force_strangeness(Some(self.strangeness));
        self.level.set_strangeness(self.strangeness);
        self.version += 1;
    }

    /// Use whatever is in front: a door or a lever.
    fn use_thing(&mut self) {
        let (feet, yaw) = self.feet();
        let held = self.held.clone();
        if let Some(what) = self.level.scene.use_thing(feet, yaw, &held) {
            self.say(what);
        }
    }

    /// A new level, or the file again (it may have been edited), keeping strangeness.
    fn reload(&mut self) {
        self.version += 1;
        self.held.clear();
        match &self.file {
            Some(f) => match load_level(f, self.start, &mut self.rng) {
                Ok(mut level) => {
                    level.set_strangeness(self.strangeness);
                    self.level = level;
                }
                Err(e) => eprintln!("{e}"),
            },
            None => self.level = Level::new(&mut self.rng),
        }
    }

    fn say(&mut self, what: String) {
        self.said = Some((what, 3.0));
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

/// The scene's meshes in psx-bevy, and the `Sim::version` they were built from,
/// and the version of each moving part's meshes (`Scene::things`).
#[derive(Resource, Default)]
struct Parts {
    entities: Vec<Entity>,
    version: Option<u64>,
    things: Vec<u32>,
}

#[derive(Component)]
struct Status;

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

    let mut app = App::new();
    let mut plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: (!headless)
                .then(|| Window { title: "neurofield".into(), resolution: (960, 720).into(), ..default() }),
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
            panel: true,
            net_ms: 0.0,
            tris: 0,
            strangeness,
            file,
            start,
            version: 0,
            held: Vec::new(),
            said: None,
        })
        .insert_non_send(net)
        .init_resource::<Parts>()
        .add_systems(Startup, (setup, capture_mouse))
        .add_systems(Update, (keys, mouse, draw, build_scene, move_things, raster_camera, status, auto_screenshot).chain())
        .add_systems(Update, egui_on_display)
        // After psx-bevy fits its display to the window.
        .add_systems(PostUpdate, layout.before(TransformSystems::Propagate))
        .add_systems(FixedUpdate, walk.in_set(GameSystems::Game))
        .add_systems(EguiPrimaryContextPass, panel);
    #[cfg(feature = "dev-mcp")]
    if mcp {
        app.add_plugins(psx_bevy::dev_mcp::GameMcpPlugin);
    }
    // egui draws on the window at full resolution, not into the PS1 screen.
    app.world_mut().resource_mut::<EguiGlobalSettings>().auto_create_primary_context = false;
    app.run();
}

fn frame_image() -> Image {
    let size = Extent3d { width: W, height: H, depth_or_array_layers: 1 };
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
    commands.spawn((
        Text::new(""),
        TextFont { font_size: bevy::text::FontSize::Px(14.0), ..default() },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
        Node { position_type: PositionType::Absolute, top: Val::Px(6.0), left: Val::Px(8.0), ..default() },
        Status,
    ));
}

fn capture_mouse(mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>) {
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
    mut exit: MessageWriter<AppExit>,
) {
    if let Ok(mut cursor) = cursor.single_mut() {
        let captured = cursor.grab_mode != CursorGrabMode::None;
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
    if key.just_pressed(KeyCode::Space) {
        sim.mode = if sim.mode == Mode::Network { Mode::Raster } else { Mode::Network };
    }
    if key.just_pressed(KeyCode::KeyB) {
        sim.mode = if sim.mode == Mode::Both { Mode::Network } else { Mode::Both };
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
    if pad.pressed(Button::Triangle) {
        sim.mode = if sim.mode == Mode::Network { Mode::Raster } else { Mode::Network };
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
    if pad.pressed(Button::Start) {
        sim.paused = !sim.paused;
    }
    if pad.pressed(Button::Cross) {
        sim.use_thing();
    }
    if pad.pressed(Button::Select) {
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
    };
    if sim.paused {
        return;
    }
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
    forward += pad.left_stick.y + held(Button::Up) - held(Button::Down);
    strafe += pad.left_stick.x + held(Button::R2) - held(Button::L2);
    turn += held(Button::Left) - held(Button::Right);
    let turn = turn.clamp(-1.0, 1.0);
    let len = (forward * forward + strafe * strafe).sqrt().max(1.0);
    // The pitch is the mouse's: passing it as the target leaves it where it is.
    let look = level.walker.pitch;
    level.walker.drive(&level.scene, forward / len, strafe / len, turn, look);
}

/// RGB floats (roughly 0..1, the network's) into an RGBA8 image.
fn upload_floats(image: &mut Image, rgb: &[f32]) {
    let data = image.data.get_or_insert_with(Vec::new);
    data.resize(rgb.len() / 3 * 4, 255);
    for (px, c) in data.chunks_exact_mut(4).zip(rgb.chunks_exact(3)) {
        for i in 0..3 {
            px[i] = (c[i].clamp(0.0, 1.0) * 255.0) as u8;
        }
        px[3] = 255;
    }
}

fn draw(time: Res<Time>, mut sim: ResMut<Sim>, mut net: NonSendMut<Network>, frame: Res<Frame>, mut images: ResMut<Assets<Image>>) {
    if !sim.paused {
        let dt = time.delta_secs();
        sim.effects.iter_mut().flatten().for_each(|t| *t += dt);
    }
    if sim.mode != Mode::Raster {
        let t0 = std::time::Instant::now();
        let mut tokens = sim.level.tokens();
        let dream = effects::apply_set(&sim.effects, &mut tokens, &sim.level.tris);
        let rgb = net.render(&tokens, &dream);
        if let Some(mut image) = images.get_mut(&frame.0) {
            upload_floats(&mut image, &rgb);
        }
        sim.tris = tokens.len();
        sim.net_ms = sim.net_ms * 0.9 + 0.1 * t0.elapsed().as_secs_f32() * 1000.0;
    }
}

/// The scene's meshes, rebuilt when it changes.
fn build_scene(mut commands: Commands, sim: Res<Sim>, mut parts: ResMut<Parts>, mut gfx: engine::render::Gfx) {
    if parts.version == Some(sim.version) {
        return;
    }
    for e in parts.entities.drain(..) {
        commands.entity(e).despawn();
    }
    parts.entities = engine::render::spawn(&mut commands, &mut gfx, &sim.level.scene);
    parts.version = Some(sim.version);
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
    let v = engine::render::View { hidden: sim.mode == Mode::Network, ..engine::render::View::of(&sim.level.scene, &cam) };
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
fn layout(
    sim: Res<Sim>,
    screen: Res<PsxScreen>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut network: Query<(&mut Sprite, &mut Transform, &mut Visibility), With<NetworkSprite>>,
    mut display: Query<(&Sprite, &mut Transform), Without<NetworkSprite>>,
) {
    let Ok(window) = window.single() else { return };
    let size = Vec2::new(W as f32, H as f32);
    let frames = if sim.mode == Mode::Both { 2.0 } else { 1.0 };
    let scale = fit_scale(window.size() / Vec2::new(frames, 1.0), size);
    let half = if sim.mode == Mode::Both { size.x * scale / 2.0 } else { 0.0 };
    for (sprite, mut t) in &mut display {
        if sprite.image == screen.image {
            t.scale = Vec3::splat(scale);
            t.translation.x = -half;
        }
    }
    let Ok((mut ns, mut nt, mut nv)) = network.single_mut() else { return };
    ns.custom_size = Some(size * scale);
    nt.translation.x = half;
    *nv = if sim.mode == Mode::Raster { Visibility::Hidden } else { Visibility::Visible };
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
    let mode = match sim.mode {
        Mode::Network => "network",
        Mode::Raster => "rasteriser",
        Mode::Both => "rasteriser | network",
    };
    let walk = if sim.auto_walk { "auto-walk" } else { "walking" };
    let pause = if sim.paused { " | paused" } else { "" };
    let on = sim.effects.iter().flatten().count();
    let keys = match sim.held.len() {
        0 => String::new(),
        _ => format!(" | keys: {}", sim.held.iter().map(|k| k.1.as_str()).collect::<Vec<_>>().join(", ")),
    };
    let said = sim.said.as_ref().map_or(String::new(), |s| format!("\n{}", s.0));
    text.0 = format!(
        "{mode} | {on} effects | {fps:.0} fps | net {:.1} ms, {} tris | {walk}{pause}{keys}\n\
         mouse: look  wasd: move  e: use  tab: auto-walk  space: switch  b: both  h: panel  n: new level  esc: free mouse{said}",
        sim.net_ms, sim.tris
    );
}
