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
//!     Mouse: look   WASD: walk and strafe   Arrows: walk and turn   Tab: auto-walk
//!     Space: network / rasteriser   B: both side by side   H: hide the effects panel
//!     N: new level   P: pause   Esc: free the mouse (click to capture it again; Esc again quits)
//!     Gamepad: left stick walk and strafe, right stick look, d-pad walk and turn,
//!     triangle network / rasteriser, square auto-walk, L1 / R1 strangeness down / up,
//!     select new level (or reload), start pause
//!
//! The dream effects panel (egui) turns effects on and off in any combination; free the
//! mouse with Esc to use it. The network runs in the main world each frame and its 320x240
//! image is uploaded to a texture, drawn with nearest sampling and scaled to fit the window.
//! The rasteriser is psx-bevy drawing the level (`engine::render`) into its 320x240 screen.
//! `GAME_SHOT=out.png:5` saves a screenshot after 5 seconds and quits.

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
use engine::world::walker::MAX_PITCH;
use psx_bevy::psx::{fit_scale, PsxPlugin, PsxScreen, DISPLAY_LAYER};
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
}

impl Sim {
    fn set_strangeness(&mut self, s: f32) {
        self.strangeness = s.clamp(0.0, 1.0);
        engine::world::scene::force_strangeness(Some(self.strangeness));
        self.level.set_strangeness(self.strangeness);
        self.version += 1;
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

/// The scene's meshes in psx-bevy, and the `Sim::version` they were built from.
#[derive(Resource, Default)]
struct Parts {
    entities: Vec<Entity>,
    version: Option<u64>,
}

#[derive(Component)]
struct Status;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window { title: "neurofield".into(), resolution: (960, 720).into(), ..default() }),
                ..default()
            })
            .set(ImagePlugin::default_nearest()),
    )
    // The rasteriser: the network's own size and shape, so the two compare pixel for pixel.
    .add_plugins(PsxPlugin { screen: PsxScreen { width: W, height: H, ..default() } })
    .add_plugins(engine::render::LevelGfxPlugin)
    .add_plugins(EguiPlugin::default())
        .insert_resource(Time::<Fixed>::from_hz(TICK))
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
        })
        .insert_non_send(net)
        .init_resource::<Parts>()
        .add_systems(Startup, (setup, capture_mouse))
        .add_systems(Update, (keys, mouse, draw, build_scene, raster_camera, status, auto_screenshot).chain())
        .add_systems(Update, egui_on_display)
        // After psx-bevy fits its display to the window.
        .add_systems(PostUpdate, layout.before(TransformSystems::Propagate))
        .add_systems(FixedUpdate, walk)
        .add_systems(EguiPrimaryContextPass, panel);
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
    pads: Query<&Gamepad>,
    button: Res<ButtonInput<MouseButton>>,
    egui: Res<EguiWantsInput>,
    mut sim: ResMut<Sim>,
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Ok(mut cursor) = cursor.single_mut() else { return };
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
    let pad = |b: GamepadButton| pads.iter().any(|p| p.just_pressed(b));
    if key.just_pressed(KeyCode::Space) || pad(GamepadButton::North) {
        sim.mode = if sim.mode == Mode::Network { Mode::Raster } else { Mode::Network };
    }
    for (b, d) in [(GamepadButton::LeftTrigger, -0.1), (GamepadButton::RightTrigger, 0.1)] {
        if pad(b) {
            let s = sim.strangeness + d;
            sim.set_strangeness(s);
        }
    }
    if key.just_pressed(KeyCode::KeyB) {
        sim.mode = if sim.mode == Mode::Both { Mode::Network } else { Mode::Both };
    }
    if key.just_pressed(KeyCode::Tab) || pad(GamepadButton::West) {
        sim.auto_walk = !sim.auto_walk;
    }
    if key.just_pressed(KeyCode::KeyP) || pad(GamepadButton::Start) {
        sim.paused = !sim.paused;
    }
    if key.just_pressed(KeyCode::KeyN) || pad(GamepadButton::Select) {
        let sim = &mut *sim;
        sim.version += 1;
        match &sim.file {
            // Reload the file (it may have been edited), keeping strangeness.
            Some(f) => match load_level(f, sim.start, &mut sim.rng) {
                Ok(mut level) => {
                    level.set_strangeness(sim.strangeness);
                    sim.level = level;
                }
                Err(e) => eprintln!("{e}"),
            },
            None => sim.level = Level::new(&mut sim.rng),
        }
    }
    if key.just_pressed(KeyCode::KeyH) {
        sim.panel = !sim.panel;
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
    pads: Query<&Gamepad>,
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    cursor: Query<&CursorOptions, With<PrimaryWindow>>,
) {
    if sim.auto_walk || sim.paused {
        return;
    }
    let captured = cursor.single().is_ok_and(|c| c.grab_mode != CursorGrabMode::None);
    let mut d = if captured { motion.delta * LOOK } else { Vec2::ZERO };
    for p in &pads {
        // Stick up is +y; looking up is less pitch.
        let s = p.right_stick();
        d += Vec2::new(s.x * PAD_YAW, -s.y * PAD_PITCH) * time.delta_secs();
    }
    let w = &mut sim.level.walker;
    // Positive yaw turns left, so looking right takes yaw away.
    w.yaw -= d.x;
    w.pitch = (w.pitch + d.y).clamp(-MAX_PITCH, MAX_PITCH);
}

/// The walk, at the fixed rate the walker was made for.
fn walk(key: Res<ButtonInput<KeyCode>>, pads: Query<&Gamepad>, mut sim: ResMut<Sim>) {
    if sim.paused {
        return;
    }
    let sim = &mut *sim;
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
    for p in &pads {
        let (stick, dpad) = (p.left_stick(), p.dpad());
        forward += stick.y + dpad.y;
        strafe += stick.x;
        turn -= dpad.x;
    }
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
    text.0 = format!(
        "{mode} | {on} effects | {fps:.0} fps | net {:.1} ms, {} tris | {walk}{pause}\n\
         mouse: look  wasd: move  tab: auto-walk  space: switch  b: both  h: panel  n: new level  esc: free mouse",
        sim.net_ms, sim.tris
    );
}
