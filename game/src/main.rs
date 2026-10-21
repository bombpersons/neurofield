//! A first-person walk through a generated level, drawn by the trained network, in Bevy.
//!
//! `game --model runs/x/out/model.safetensors [--seed 7]` (GPU; a .onnx model runs on the CPU)
//!     Mouse: look   WASD: walk and strafe   Arrows: walk and turn   Tab: auto-walk
//!     Space: network / rasteriser   B: both side by side   H: hide the effects panel
//!     N: new level   P: pause   Esc: free the mouse (click to capture it again; Esc again quits)
//!
//! The dream effects panel (egui) turns effects on and off in any combination; free the
//! mouse with Esc to use it. The network runs in the main world each frame and its 320x240
//! image is uploaded to a texture, drawn with nearest sampling and scaled to fit the window.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_egui::input::EguiWantsInput;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use datagen::render::{self, H, W};
use datagen::rng::Rng;
use datagen::walker::MAX_PITCH;
use std::collections::HashMap;
use std::path::PathBuf;
use viewer::{effects, Level, Network, TICK};

/// Mouse sensitivity: yaw units (4096 per turn) per pixel of mouse movement.
const LOOK: f32 = 1.2;

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
    /// Whether the model takes the dream inputs (old ONNX exports do not).
    dreams: bool,
    net_ms: f32,
    tris: usize,
}

/// The network and the PS1 rasteriser, which own GPU or plain-memory state.
struct Renderers {
    net: Network,
    raster: render::Gpu,
}

/// The two 320x240 frames and the sprites that show them.
#[derive(Resource)]
struct Frames {
    network: Handle<Image>,
    raster: Handle<Image>,
}

#[derive(Component)]
struct NetworkSprite;

#[derive(Component)]
struct RasterSprite;

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
    let model = PathBuf::from(opts.get("model").expect("--model runs/x/model.safetensors (or .onnx)"));
    let seed: u64 = opts.get("seed").map_or(7, |s| s.parse().expect("--seed is a number"));
    let net = Network::load(&model);
    let dreams = net.dreams();
    let textures = datagen::tex::make_all();
    let mut rng = Rng::new(seed);
    let level = Level::new(&mut rng);

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { title: "neurofield".into(), resolution: (960, 720).into(), ..default() }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .insert_resource(Time::<Fixed>::from_hz(TICK))
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Sim {
            level,
            rng,
            auto_walk: false,
            paused: false,
            mode: Mode::Network,
            effects: [None; effects::EFFECTS.len()],
            panel: true,
            dreams,
            net_ms: 0.0,
            tris: 0,
        })
        .insert_non_send(Renderers { net, raster: render::new_gpu(&textures) })
        .add_systems(Startup, (setup, capture_mouse))
        .add_systems(Update, (keys, mouse, draw, layout, status).chain())
        .add_systems(FixedUpdate, walk)
        .add_systems(EguiPrimaryContextPass, panel)
        .run();
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
    commands.spawn(Camera2d);
    let frames = Frames { network: images.add(frame_image()), raster: images.add(frame_image()) };
    commands.spawn((Sprite::from_image(frames.network.clone()), NetworkSprite));
    commands.spawn((Sprite::from_image(frames.raster.clone()), RasterSprite));
    commands.insert_resource(frames);
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
    if key.just_pressed(KeyCode::KeyN) {
        let sim = &mut *sim;
        sim.level = Level::new(&mut sim.rng);
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
        if !sim.dreams && sim.effects.iter().enumerate().any(|(e, t)| t.is_some() && effects::needs_dream_inputs(e)) {
            ui.colored_label(egui::Color32::YELLOW, "this model has no dream inputs: re-export it");
        }
        ui.label("esc frees the mouse; click the view to look again");
    });
    Ok(())
}

/// Mouse look, every frame (not at the walk's 30 Hz) so it stays smooth.
fn mouse(
    motion: Res<AccumulatedMouseMotion>,
    mut sim: ResMut<Sim>,
    cursor: Query<&CursorOptions, With<PrimaryWindow>>,
) {
    let captured = cursor.single().is_ok_and(|c| c.grab_mode != CursorGrabMode::None);
    if !captured || sim.auto_walk || sim.paused {
        return;
    }
    let w = &mut sim.level.walker;
    // Positive yaw turns left, so moving the mouse right takes yaw away.
    w.yaw -= motion.delta.x * LOOK;
    w.pitch = (w.pitch + motion.delta.y * LOOK).clamp(-MAX_PITCH, MAX_PITCH);
}

/// The walk, at the fixed rate the walker was made for.
fn walk(key: Res<ButtonInput<KeyCode>>, mut sim: ResMut<Sim>) {
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
    let turn = down(KeyCode::ArrowLeft, KeyCode::ArrowLeft) - down(KeyCode::ArrowRight, KeyCode::ArrowRight);
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

fn draw(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut renderers: NonSendMut<Renderers>,
    frames: Res<Frames>,
    mut images: ResMut<Assets<Image>>,
) {
    if !sim.paused {
        let dt = time.delta_secs();
        sim.effects.iter_mut().flatten().for_each(|t| *t += dt);
    }
    let cam = sim.level.walker.camera();
    if sim.mode != Mode::Network {
        let rgb = render::render(&mut renderers.raster, &sim.level.scene, &cam);
        if let Some(mut image) = images.get_mut(&frames.raster) {
            let data = image.data.get_or_insert_with(Vec::new);
            data.resize(rgb.len() / 3 * 4, 255);
            for (px, c) in data.chunks_exact_mut(4).zip(rgb.chunks_exact(3)) {
                px[..3].copy_from_slice(c);
            }
        }
    }
    if sim.mode != Mode::Raster {
        let t0 = std::time::Instant::now();
        let mut tokens = sim.level.tokens();
        let dream = effects::apply_set(&sim.effects, &mut tokens, &sim.level.tris);
        let rgb = renderers.net.render(&tokens, &dream);
        if let Some(mut image) = images.get_mut(&frames.network) {
            upload_floats(&mut image, &rgb);
        }
        sim.tris = tokens.len();
        sim.net_ms = sim.net_ms * 0.9 + 0.1 * t0.elapsed().as_secs_f32() * 1000.0;
    }
}

/// Fit the shown frame(s) to the window, keeping 4:3 and whole pixels sharp.
fn layout(
    sim: Res<Sim>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut network: Query<(&mut Sprite, &mut Transform, &mut Visibility), (With<NetworkSprite>, Without<RasterSprite>)>,
    mut raster: Query<(&mut Sprite, &mut Transform, &mut Visibility), (With<RasterSprite>, Without<NetworkSprite>)>,
) {
    let Ok(window) = window.single() else { return };
    let (ww, wh) = (window.width(), window.height());
    let frames = if sim.mode == Mode::Both { 2.0 } else { 1.0 };
    let scale = (ww / (W as f32 * frames)).min(wh / H as f32);
    let size = Vec2::new(W as f32, H as f32) * scale;
    let (Ok((mut ns, mut nt, mut nv)), Ok((mut rs, mut rt, mut rv))) = (network.single_mut(), raster.single_mut())
    else {
        return;
    };
    ns.custom_size = Some(size);
    rs.custom_size = Some(size);
    let half = if sim.mode == Mode::Both { size.x / 2.0 } else { 0.0 };
    // Both: rasteriser on the left, network on the right.
    rt.translation.x = -half;
    nt.translation.x = half;
    *nv = if sim.mode == Mode::Raster { Visibility::Hidden } else { Visibility::Visible };
    *rv = if sim.mode == Mode::Network { Visibility::Hidden } else { Visibility::Visible };
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
