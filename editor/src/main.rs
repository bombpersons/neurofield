//! Level editor for hand-built dungeons (see docs/editor.md).
//!
//! `cargo run -p editor [-- levels/x.ron]`
//!
//! The 3D view fills the window; the menus, status bar and dockable panels draw
//! over it. Data paths (textures, skins, levels) come from `neurofield.ron`.
//! Right mouse: fly (WASD, Q/E down/up, Shift faster)   Middle mouse or Alt+left: orbit
//! Wheel: dolly   F: frame the selection   PgUp/PgDn: move the height slice

mod doc;
mod flesh;
mod settings;
mod ui;
mod view3d;

use bevy::prelude::*;
use bevy::window::WindowCloseRequested;
use bevy::camera::visibility::RenderLayers;
use bevy_egui::{EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass, EguiTextureHandle, EguiUserTextures, PrimaryEguiContext};
use engine::world::config::Config;
use engine::world::level::Level;
use std::collections::HashMap;
use std::path::PathBuf;
use psx_bevy::psx::{PsxAspect, PsxPlugin, PsxScreen, DISPLAY_LAYER};
use view3d::{Atlas, Built, EditorCam, ViewInput, ViewMode};

/// The level being edited.
#[derive(Resource)]
pub struct Document(pub doc::Doc);

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut opts = HashMap::new();
    let mut file = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.strip_prefix("--") {
            Some(k) => {
                opts.insert(k.to_string(), it.next().cloned().unwrap_or_else(|| panic!("--{k} needs a value")));
            }
            None => file = Some(PathBuf::from(a)),
        }
    }
    let config = Config::load();
    config.load_textures(&opts);
    // Scenes get creatures through the PS1 pipeline, which is quick to build; the
    // full-colour ones are sculpted in the background (see `flesh`).
    engine::world::flesh::set_psx(true);
    let textures = engine::world::tex::make_all();

    let settings = settings::Settings::load();
    let file = file.or_else(|| settings.recent.first().cloned().filter(|p| p.is_file()));
    let doc = match &file {
        Some(f) => doc::Doc::open(f).unwrap_or_else(|e| {
            eprintln!("{e}");
            doc::Doc::new(Level::demo(), None)
        }),
        None => doc::Doc::new(Level::demo(), None),
    };
    // Start looking down at the level from above one corner.
    let mut cam = EditorCam { pos: Vec3::ZERO, yaw: -0.7, pitch: -0.75, focus: 9.0 };
    cam.frame(ui::level_centre(&doc.level), 9.0);

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window { title: "neurofield editor".into(), resolution: (1400, 900).into(), ..default() }),
                // Closing asks first when there are unsaved changes.
                close_when_requested: false,
                ..default()
            })
            .set(ImagePlugin::default_nearest()),
    )
    .add_plugins(PsxPlugin { screen: PsxScreen::default() })
    .insert_resource(PsxAspect::Window)
    .add_plugins(engine::render::LevelGfxPlugin)
    .add_plugins(EguiPlugin::default())
    .insert_resource(Document(doc))
    .insert_resource(ui::Ui::new(settings, config))
    .insert_resource(ViewInput::default())
    .insert_resource(cam)
    .insert_resource(Built {
        scene: Level::new("").to_scene(),
        version: 0,
        mode: ViewMode::Plain,
        slice: 0,
        show_grid: true,
        show_entities: true,
        full_colour: true,
        shapes: Vec::new(),
        changed_at: 0.0,
        parts: Vec::new(),
        hidden: Vec::new(),
    })
    .insert_resource(flesh::Sculpts::default())
    .add_systems(Startup, setup)
    .add_systems(Update, egui_on_display)
    .add_systems(EguiPrimaryContextPass, ui::draw)
    .add_systems(
        Update,
        (
            auto_screenshot,
            close_requested,
            flesh::sculpt,
            view3d::rebuild,
            view3d::camera_control,
            view3d::apply_camera,
            view3d::screen_mode,
            view3d::gizmos,
        )
            .chain(),
    );
    // egui draws on the window at full resolution, not into the PS1 screen.
    app.world_mut().resource_mut::<EguiGlobalSettings>().auto_create_primary_context = false;
    let atlas = view3d::atlas_image(&textures);
    let atlas = app.world_mut().resource_mut::<Assets<Image>>().add(atlas);
    app.insert_resource(Atlas(atlas));
    app.run();
}

fn setup(atlas: Res<Atlas>, mut egui_textures: ResMut<EguiUserTextures>) {
    egui_textures.add_image(EguiTextureHandle::Strong(atlas.0.clone()));
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

/// `EDITOR_SHOT=out.png:5` saves a screenshot of the window after 5 seconds and
/// quits, for checking the editor from scripts.
fn auto_screenshot(mut commands: Commands, time: Res<Time>, mut frames: Local<u32>, mut exit: MessageWriter<AppExit>) {
    use bevy::render::view::screenshot::{save_to_disk, Screenshot};
    let Ok(spec) = std::env::var("EDITOR_SHOT") else { return };
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

/// The window's close button: quit, or ask first if there are unsaved changes.
fn close_requested(mut requests: MessageReader<WindowCloseRequested>, mut ui: ResMut<ui::Ui>) {
    for _ in requests.read() {
        ui.quit_requested = true;
    }
}

/// Start the game on the level as it is now, from its player start or from `start`.
pub fn play(level: &Level, start: Option<([f32; 3], f32)>) -> Result<(), String> {
    let tmp = std::env::temp_dir().join("neurofield-play.ron");
    level.save(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    // Under `cargo run`, run the game the same way (so it is built and current);
    // otherwise it sits beside the editor.
    let mut cmd = match std::env::var_os("CARGO") {
        Some(cargo) => {
            let mut c = std::process::Command::new(cargo);
            c.args(["run", "-p", "game"]);
            if exe.components().any(|c| c.as_os_str() == "release") {
                c.arg("--release");
            }
            c.arg("--");
            c
        }
        None => std::process::Command::new(exe.with_file_name(if cfg!(windows) { "game.exe" } else { "game" })),
    };
    cmd.arg("--level-file").arg(&tmp);
    if let Some((p, yaw)) = start {
        cmd.arg("--start").arg(format!("{},{},{},{}", p[0], p[1], p[2], yaw));
    }
    cmd.spawn().map(|_| ()).map_err(|e| format!("starting the game: {e}"))
}
