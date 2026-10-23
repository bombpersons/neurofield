//! The editor's developer MCP server (`--mcp`, needs the `dev-mcp` feature): JSON-RPC
//! tools on stdio for driving the editor the way a person does, so agents can test it
//! without touching the desktop's real mouse and keyboard.
//!
//! Input goes in through the editor's own input path: egui events straight to its
//! context, and Bevy's keyboard and mouse messages for the 3D camera. The window's real
//! input no longer reaches egui, so someone moving the mouse over it changes nothing.
//! With `--headless` there is no OS window: the window is a plain Bevy entity and
//! everything egui and the 3D view draw lands in an image instead (see
//! `headless_display`).
//!
//! Tools: `editor_status`, `editor_input`, `editor_widgets`, `editor_locate`,
//! `editor_level`, `editor_setup` and `editor_screenshot` (see `tools()` and
//! docs/editor.md). Positions are window pixels (egui points), origin top-left.
//!
//! The server thread hands requests to the main thread over a channel and waits for
//! replies, as psx-bevy's game MCP does; this one is separate because the editor has no
//! ticks or pad, and its tools are about panels, widgets and the level.

use crate::tools::{entity_kinds, marker, Sel};
use crate::ui::Ui;
use crate::view3d::{to_view, Built, EditorCam};
use crate::Document;
use base64::Engine;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::input::keyboard::{Key as BevyKey, KeyboardInput};
use bevy::input::mouse::{MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::input::{ButtonState, InputSystems};
use bevy::prelude::*;
use bevy::render::render_resource::{CachedPipelineState, PipelineCache, TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::window::PrimaryWindow;
use bevy_egui::input::{EguiInputEvent, FocusedNonWindowEguiContext};
use bevy_egui::{EguiContext, EguiFullOutput, EguiGlobalSettings, EguiInputSet, EguiPostUpdateSet, PrimaryEguiContext};
use bevy_egui::egui;
use egui::accesskit;
use engine::world::level::{Dir, Face};
use engine::world::scene::{STEP, T};
use psx_bevy::psx::DISPLAY_LAYER;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ErrorData, ServerHandler, ServiceExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use tokio::sync::{oneshot, Semaphore};

const TF: f32 = T as f32;

pub struct EditorMcpPlugin {
    pub headless: bool,
}

type Reply = oneshot::Sender<Result<Response, String>>;

enum Response {
    Json(Value),
    Image { png: Vec<u8>, metadata: Value },
}

struct Request {
    command: Command,
    reply: Reply,
}

enum Command {
    Status,
    Input(Vec<Step>),
    Widgets(WidgetArgs),
    Locate(LocateArgs),
    Level(LevelArgs),
    Setup(SetupArgs),
    Screenshot(ShotArgs),
}

/// One step of `editor_input`. Each becomes one or more frames of events.
#[derive(Deserialize, Debug)]
#[serde(tag = "do", rename_all = "snake_case", deny_unknown_fields)]
enum Step {
    /// Move the pointer, over `frames` frames (1 jumps).
    Move {
        to: [f32; 2],
        #[serde(default = "one")]
        frames: u32,
    },
    Click {
        at: [f32; 2],
        #[serde(default)]
        button: Button,
        #[serde(default)]
        with: Vec<String>,
    },
    DoubleClick {
        at: [f32; 2],
        #[serde(default)]
        button: Button,
    },
    Press {
        #[serde(default)]
        button: Button,
        at: Option<[f32; 2]>,
    },
    Release {
        #[serde(default)]
        button: Button,
        at: Option<[f32; 2]>,
    },
    Drag {
        from: [f32; 2],
        to: [f32; 2],
        #[serde(default)]
        button: Button,
        #[serde(default = "ten")]
        frames: u32,
        #[serde(default)]
        with: Vec<String>,
    },
    /// Lines; positive scrolls up (zooms in).
    Wheel {
        lines: f32,
        at: Option<[f32; 2]>,
    },
    /// A key or chord ("Ctrl+Z", "Tab", "Shift+F5"), pressed and released.
    Key { key: String },
    KeyDown { key: String },
    KeyUp { key: String },
    /// Hold a key (or chord) down for `frames` frames, as when flying with WASD.
    Hold { key: String, frames: u32 },
    /// Typed text, for the focused text field.
    Text { text: String },
    Wait { frames: u32 },
}

fn one() -> u32 {
    1
}
fn ten() -> u32 {
    10
}

#[derive(Deserialize, Debug, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Button {
    #[default]
    Left,
    Right,
    Middle,
}

impl Button {
    fn bevy(self) -> MouseButton {
        match self {
            Button::Left => MouseButton::Left,
            Button::Right => MouseButton::Right,
            Button::Middle => MouseButton::Middle,
        }
    }
    fn egui(self) -> egui::PointerButton {
        match self {
            Button::Left => egui::PointerButton::Primary,
            Button::Right => egui::PointerButton::Secondary,
            Button::Middle => egui::PointerButton::Middle,
        }
    }
}

#[derive(Clone, Debug)]
struct KeyName {
    code: KeyCode,
    egui: Option<egui::Key>,
    logical: BevyKey,
    /// The text a press types, for keys that type one character.
    ch: Option<char>,
}

/// One key by name: a letter or digit, a modifier (Ctrl, Shift, Alt) or one of egui's
/// key names (Escape, Tab, Enter, Space, Backspace, Delete, Home, End, PageUp,
/// PageDown, ArrowUp..., F1...).
fn key_name(name: &str) -> Result<KeyName, String> {
    let named = |code, logical| Ok(KeyName { code, egui: None, logical, ch: None });
    match name.to_ascii_lowercase().as_str() {
        "ctrl" | "control" => return named(KeyCode::ControlLeft, BevyKey::Control),
        "shift" => return named(KeyCode::ShiftLeft, BevyKey::Shift),
        "alt" => return named(KeyCode::AltLeft, BevyKey::Alt),
        _ => {}
    }
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.clone().next()) {
        let code = match c.to_ascii_uppercase() {
            l @ 'A'..='Z' => format!("Key{l}"),
            d @ '0'..='9' => format!("Digit{d}"),
            _ => return Err(format!("unknown key {name}")),
        };
        let code: KeyCode = serde_json::from_value(json!(code)).map_err(|e| e.to_string())?;
        let ch = c.to_ascii_lowercase();
        return Ok(KeyName {
            code,
            egui: egui::Key::from_name(&c.to_ascii_uppercase().to_string()),
            logical: BevyKey::Character(ch.to_string().into()),
            ch: Some(ch),
        });
    }
    let key = egui::Key::from_name(name).ok_or_else(|| format!("unknown key {name}"))?;
    let canonical = key.name();
    // Bevy's key codes and logical keys mostly share egui's names.
    let code_name = match canonical {
        "Space" => "Space",
        other => other,
    };
    let code: KeyCode = serde_json::from_value(json!(code_name)).map_err(|_| format!("no key code for {name}"))?;
    let logical = match canonical {
        "Space" => BevyKey::Space,
        "Escape" => BevyKey::Escape,
        "Tab" => BevyKey::Tab,
        "Enter" => BevyKey::Enter,
        "Backspace" => BevyKey::Backspace,
        "Delete" => BevyKey::Delete,
        "Home" => BevyKey::Home,
        "End" => BevyKey::End,
        "PageUp" => BevyKey::PageUp,
        "PageDown" => BevyKey::PageDown,
        "ArrowUp" => BevyKey::ArrowUp,
        "ArrowDown" => BevyKey::ArrowDown,
        "ArrowLeft" => BevyKey::ArrowLeft,
        "ArrowRight" => BevyKey::ArrowRight,
        _ => serde_json::from_value(json!(canonical)).unwrap_or(BevyKey::Dead(None)),
    };
    Ok(KeyName { code, egui: Some(key), logical, ch: (canonical == "Space").then_some(' ') })
}

/// "Ctrl+Shift+Z": the modifiers, then the key.
fn chord(text: &str) -> Result<Vec<KeyName>, String> {
    text.split('+').map(str::trim).filter(|k| !k.is_empty()).map(key_name).collect::<Result<Vec<_>, _>>().and_then(|k| {
        if k.is_empty() {
            Err("empty key".into())
        } else {
            Ok(k)
        }
    })
}

/// One event of a frame of injected input.
#[derive(Clone, Debug)]
enum Ev {
    To(Vec2),
    Button(MouseButton, egui::PointerButton, bool),
    Wheel(f32),
    Key(KeyName, bool),
    Text(String),
}

/// The frames of events a step makes, starting from the pointer at `at`.
fn frames(step: &Step, at: &mut Vec2) -> Result<Vec<Vec<Ev>>, String> {
    let v = |p: [f32; 2]| Vec2::new(p[0], p[1]);
    let mods = |with: &[String]| with.iter().map(|m| key_name(m)).collect::<Result<Vec<_>, _>>();
    let down = |keys: &[KeyName]| keys.iter().map(|k| Ev::Key(k.clone(), true)).collect::<Vec<_>>();
    let up = |keys: &[KeyName]| keys.iter().rev().map(|k| Ev::Key(k.clone(), false)).collect::<Vec<_>>();
    let mut out = Vec::new();
    match step {
        Step::Move { to, frames } => {
            let (from, n) = (*at, (*frames).max(1));
            for i in 1..=n {
                out.push(vec![Ev::To(from.lerp(v(*to), i as f32 / n as f32))]);
            }
            *at = v(*to);
        }
        Step::Click { at: p, button, with } => {
            let m = mods(with)?;
            let mut first = down(&m);
            first.push(Ev::To(v(*p)));
            out.push(first);
            out.push(vec![Ev::Button(button.bevy(), button.egui(), true)]);
            out.push(vec![Ev::Button(button.bevy(), button.egui(), false)]);
            // Modifiers come up after the button, as they do by hand: a drag still
            // under way on the release frame sees them.
            if !m.is_empty() {
                out.push(up(&m));
            }
            *at = v(*p);
        }
        Step::DoubleClick { at: p, button } => {
            // All in one frame: egui's double-click window (0.3 s) is shorter than a few
            // frames of software rendering.
            out.push(vec![Ev::To(v(*p))]);
            let (b, e) = (button.bevy(), button.egui());
            out.push(vec![Ev::Button(b, e, true), Ev::Button(b, e, false), Ev::Button(b, e, true), Ev::Button(b, e, false)]);
            *at = v(*p);
        }
        Step::Press { button, at: p } | Step::Release { button, at: p } => {
            let mut f = Vec::new();
            if let Some(p) = p {
                f.push(Ev::To(v(*p)));
                *at = v(*p);
            }
            f.push(Ev::Button(button.bevy(), button.egui(), matches!(step, Step::Press { .. })));
            out.push(f);
        }
        Step::Drag { from, to, button, frames, with } => {
            let m = mods(with)?;
            let mut first = down(&m);
            first.push(Ev::To(v(*from)));
            out.push(first);
            out.push(vec![Ev::Button(button.bevy(), button.egui(), true)]);
            let n = (*frames).max(1);
            for i in 1..=n {
                out.push(vec![Ev::To(v(*from).lerp(v(*to), i as f32 / n as f32))]);
            }
            out.push(vec![Ev::Button(button.bevy(), button.egui(), false)]);
            if !m.is_empty() {
                out.push(up(&m));
            }
            *at = v(*to);
        }
        Step::Wheel { lines, at: p } => {
            if let Some(p) = p {
                out.push(vec![Ev::To(v(*p))]);
                *at = v(*p);
            }
            out.push(vec![Ev::Wheel(*lines)]);
        }
        Step::Key { key } => {
            let k = chord(key)?;
            out.push(down(&k));
            out.push(up(&k));
        }
        Step::KeyDown { key } => out.push(down(&chord(key)?)),
        Step::KeyUp { key } => out.push(up(&chord(key)?)),
        Step::Hold { key, frames } => {
            let k = chord(key)?;
            out.push(down(&k));
            out.extend((1..*frames).map(|_| Vec::new()));
            out.push(up(&k));
        }
        Step::Text { text } => out.push(vec![Ev::Text(text.clone())]),
        Step::Wait { frames } => out.extend((0..*frames).map(|_| Vec::new())),
    }
    Ok(out)
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct WidgetArgs {
    /// Labels, values or hints containing this (any case).
    text: Option<String>,
    /// Roles containing this (any case): button, checkbox, combobox, label...
    role: Option<String>,
    /// Only widgets inside this panel (its tab title).
    panel: Option<String>,
    limit: Option<usize>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct LocateArgs {
    entity: Option<u32>,
    light: Option<usize>,
    /// World units, y down.
    point: Option<[f32; 3]>,
    /// The face of cell [x, layer, z]: floor (default), ceiling, n, s, e or w.
    cell: Option<[i32; 3]>,
    face: Option<String>,
    /// A plan tile [x, z].
    tile: Option<[i32; 2]>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct LevelArgs {
    #[serde(default)]
    part: Part,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Part {
    #[default]
    Entities,
    Lights,
    Areas,
    Ron,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SetupArgs {
    /// Open this level file, dropping unsaved changes and the undo history.
    open: Option<PathBuf>,
    /// Start a new level (the demo when `demo` is true, else empty).
    new: Option<bool>,
    #[serde(default)]
    demo: bool,
    camera: Option<CameraArgs>,
    /// Window size in pixels.
    size: Option<[u32; 2]>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CameraArgs {
    /// Radians: yaw about the vertical (0 looks along world +z), pitch up.
    yaw: Option<f32>,
    pitch: Option<f32>,
    /// World point to look at, from `distance` tiles away.
    look_at: Option<[f32; 3]>,
    distance: Option<f32>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ShotArgs {
    /// Crop to a panel (its tab title).
    panel: Option<String>,
    /// Crop to [x, y, width, height] window pixels.
    rect: Option<[f32; 4]>,
}

#[derive(Resource)]
struct Bridge {
    receiver: Mutex<mpsc::Receiver<Request>>,
    closed: Arc<AtomicBool>,
    ready: RenderReady,
    /// Screenshots asked for before the first frames rendered.
    waiting: Vec<(ShotArgs, Reply)>,
}

#[derive(Resource, Clone)]
struct RenderReady(Arc<AtomicBool>);

/// Input being played into the editor, one frame of events at a time.
#[derive(Resource, Default)]
struct Script {
    queue: VecDeque<(VecDeque<Vec<Ev>>, Reply, usize)>,
    pointer: Vec2,
    keys: BTreeSet<KeyCode>,
    frame: u64,
}

/// The widgets egui drew last frame, from its accessibility tree.
#[derive(Resource, Default)]
struct Widgets(Vec<Widget>);

#[derive(Clone)]
struct Widget {
    role: String,
    label: String,
    value: String,
    /// A text field's hint, shown while it is empty.
    hint: String,
    /// The part on screen.
    rect: egui::Rect,
    /// Partly scrolled or cut out of sight.
    clipped: bool,
    disabled: bool,
    toggled: Option<bool>,
    clickable: bool,
}

/// `--headless`: the image the window's camera draws into.
#[derive(Resource, Default)]
struct HeadlessTarget(Option<Handle<Image>>);

impl Plugin for EditorMcpPlugin {
    fn build(&self, app: &mut App) {
        // Requests are served even while the window is unfocused.
        app.insert_resource(bevy::winit::WinitSettings::continuous());
        // Only injected input reaches egui; the modifiers still come from Bevy's
        // keyboard messages, which `inject` writes.
        let input = &mut app.world_mut().resource_mut::<EguiGlobalSettings>().input_system_settings;
        input.run_write_window_pointer_moved_messages = false;
        input.run_write_pointer_button_messages = false;
        input.run_write_mouse_wheel_messages_system = false;
        input.run_write_keyboard_input_messages_system = false;
        input.run_write_ime_messages_system = false;
        input.run_write_window_touch_messages_system = false;
        input.run_write_non_window_touch_messages_system = false;
        input.run_write_non_window_pointer_moved_messages_system = false;
        input.run_write_pinch_gesture_messages_system = false;

        let (sender, receiver) = mpsc::sync_channel(16);
        let closed = Arc::new(AtomicBool::new(false));
        let thread_closed = closed.clone();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
            runtime.block_on(async move {
                let server = EditorServer { sender, slots: Arc::new(Semaphore::new(1)) };
                match server.serve(rmcp::transport::stdio()).await {
                    Ok(service) => {
                        if let Err(e) = service.waiting().await {
                            eprintln!("editor MCP: {e}");
                        }
                    }
                    Err(e) => eprintln!("editor MCP startup: {e}"),
                }
                thread_closed.store(true, Ordering::Release);
            });
        });
        let ready = RenderReady(Arc::new(AtomicBool::new(false)));
        app.sub_app_mut(RenderApp)
            .insert_resource(ready.clone())
            .add_systems(Render, mark_render_ready.after(RenderSystems::Render).before(RenderSystems::Cleanup));
        app.insert_resource(Bridge { receiver: Mutex::new(receiver), closed, ready, waiting: Vec::new() })
            .init_resource::<Script>()
            .init_resource::<Widgets>()
            .add_systems(
                PreUpdate,
                (serve, inject).chain().before(InputSystems).before(EguiInputSet::InitReading),
            )
            .add_systems(PostUpdate, read_widgets.after(EguiPostUpdateSet::EndPass).before(EguiPostUpdateSet::ProcessOutput));
        if self.headless {
            app.init_resource::<HeadlessTarget>().add_systems(Update, headless_display);
        }
    }
}

fn mark_render_ready(
    ready: Res<RenderReady>,
    cache: Res<PipelineCache>,
    views: Query<&ViewTarget>,
    mut settled: Local<u32>,
) {
    // Wait for the views to exist and every queued pipeline to compile for a few
    // frames running, so the first screenshot isn't of an empty target.
    let compiled = cache.pipelines().next().is_some()
        && cache.pipelines().all(|p| matches!(&p.state, CachedPipelineState::Ok(_)));
    *settled = if views.iter().count() >= 2 && compiled { *settled + 1 } else { 0 };
    if *settled >= 3 {
        ready.0.store(true, Ordering::Release);
    }
}

/// Without an OS window the window's camera (psx-bevy's display camera, which egui
/// draws on) renders into an image the size of the window.
fn headless_display(
    mut target: ResMut<HeadlessTarget>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cams: Query<(&mut RenderTarget, &RenderLayers), With<Camera2d>>,
) {
    let Ok(window) = windows.single() else { return };
    let size = UVec2::new(window.physical_width().max(1), window.physical_height().max(1));
    if target.0.as_ref().and_then(|h| images.get(h)).is_none_or(|i| i.size() != size) {
        let mut image = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba8UnormSrgb, None);
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        target.0 = Some(images.add(image));
    }
    let handle = target.0.clone().unwrap();
    for (mut t, layers) in &mut cams {
        if !layers.intersects(&RenderLayers::layer(DISPLAY_LAYER)) {
            continue;
        }
        let wanted = RenderTarget::Image(handle.clone().into());
        if !matches!(&*t, RenderTarget::Image(i) if i.handle == handle) {
            *t = wanted;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn serve(
    mut commands: Commands,
    mut bridge: ResMut<Bridge>,
    mut script: ResMut<Script>,
    mut ui: ResMut<Ui>,
    mut doc: ResMut<Document>,
    mut cam: ResMut<EditorCam>,
    built: Res<Built>,
    widgets: Res<Widgets>,
    headless: Option<Res<HeadlessTarget>>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut contexts: Query<(Entity, &mut EguiContext), With<PrimaryEguiContext>>,
    mut exit: MessageWriter<AppExit>,
    mut enabled: Local<bool>,
) {
    if bridge.closed.load(Ordering::Acquire) {
        exit.write(AppExit::Success);
        return;
    }
    // egui gets the keyboard as if its window had focus, and fills in its widget tree.
    if let Ok((entity, mut ctx)) = contexts.single_mut() {
        commands.insert_resource(FocusedNonWindowEguiContext(entity));
        if !*enabled {
            ctx.get_mut().enable_accesskit();
            *enabled = true;
        }
    }
    let scale = windows.single().map_or(1.0, |w| w.scale_factor());
    let ready = bridge.ready.0.load(Ordering::Acquire);
    let mut requests: Vec<Request> = Vec::new();
    if ready {
        for (args, reply) in std::mem::take(&mut bridge.waiting) {
            screenshot(&mut commands, &ui, headless.as_deref(), scale, args, reply);
        }
    }
    requests.extend(bridge.receiver.lock().unwrap().try_iter());
    for Request { command, reply } in requests {
        if reply.is_closed() {
            continue;
        }
        let result = match command {
            Command::Status => {
                Ok(status(&ui, &doc.0, &built, &cam, windows.single().ok(), headless.is_some(), script.frame))
            }
            Command::Input(steps) => {
                // One request at a time (see `EditorServer::slots`), so nothing is queued ahead.
                let mut at = script.pointer;
                let mut all = VecDeque::new();
                let mut err = None;
                for s in &steps {
                    match frames(s, &mut at) {
                        Ok(f) => all.extend(f),
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                if let Some(e) = err {
                    Err(e)
                } else {
                    let n = all.len();
                    script.queue.push_back((all, reply, n));
                    continue;
                }
            }
            Command::Widgets(args) => widget_list(&widgets.0, &ui, &args),
            Command::Locate(args) => locate(&ui, &doc, &built, &cam, &args),
            Command::Level(args) => level(&doc, args.part),
            Command::Setup(args) => match setup(&mut ui, &mut doc, &mut cam, windows.single_mut().ok(), args) {
                // Answer once the editor has drawn the result, so panel and plan
                // positions are current.
                Ok(()) => {
                    script.queue.push_back((VecDeque::from([Vec::new(), Vec::new()]), reply, 0));
                    continue;
                }
                Err(e) => Err(e),
            },
            Command::Screenshot(args) => {
                if ready {
                    screenshot(&mut commands, &ui, headless.as_deref(), scale, args, reply);
                } else {
                    bridge.waiting.push((args, reply));
                }
                continue;
            }
        };
        let _ = reply.send(result.map(Response::Json));
    }
}

/// Plays a frame of the input script into egui and Bevy's input, and answers the
/// script's request on the frame after its last events (once the editor has seen them).
#[allow(clippy::too_many_arguments)]
fn inject(
    mut script: ResMut<Script>,
    ui: Res<Ui>,
    doc: Res<Document>,
    built: Res<Built>,
    cam: Res<EditorCam>,
    windows: Query<&Window, With<PrimaryWindow>>,
    headless: Option<Res<HeadlessTarget>>,
    contexts: Query<Entity, With<PrimaryEguiContext>>,
    mut egui_events: MessageWriter<EguiInputEvent>,
    mut keys: MessageWriter<KeyboardInput>,
    mut buttons: MessageWriter<MouseButtonInput>,
    mut motion: MessageWriter<MouseMotion>,
    mut wheel: MessageWriter<MouseWheel>,
) {
    script.frame += 1;
    let Ok(context) = contexts.single() else { return };
    let Some((frames, _, _)) = script.queue.front_mut() else { return };
    let Some(events) = frames.pop_front() else {
        let (_, reply, n) = script.queue.pop_front().unwrap();
        let mut state = status(&ui, &doc.0, &built, &cam, windows.single().ok(), headless.is_some(), script.frame);
        if n > 0 {
            // After input, what it changed; editor_status has the rest.
            let o = state.as_object_mut().unwrap();
            o.retain(|k, _| ["frame", "dirty", "undo", "redo", "tool", "op", "selection", "faces_selected", "hover", "link", "message", "counts"].contains(&k.as_str()));
            state["frames"] = json!(n);
        }
        let _ = reply.send(Ok(Response::Json(state)));
        return;
    };
    // Bevy's messages carry a window; none that egui maps to a context, so egui only
    // sees what it is sent directly.
    let window = Entity::PLACEHOLDER;
    for ev in events {
        let modifiers = egui_modifiers(&script.keys);
        let event = match ev {
            Ev::To(p) => {
                motion.write(MouseMotion { delta: p - script.pointer });
                script.pointer = p;
                Some(egui::Event::PointerMoved(egui::pos2(p.x, p.y)))
            }
            Ev::Button(b, e, pressed) => {
                let state = if pressed { ButtonState::Pressed } else { ButtonState::Released };
                buttons.write(MouseButtonInput { button: b, state, window });
                let p = script.pointer;
                Some(egui::Event::PointerButton { pos: egui::pos2(p.x, p.y), button: e, pressed, modifiers })
            }
            Ev::Wheel(lines) => {
                wheel.write(MouseWheel { unit: MouseScrollUnit::Line, x: 0.0, y: lines, window, phase: TouchPhase::Moved });
                Some(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: egui::vec2(0.0, lines),
                    phase: egui::TouchPhase::Move,
                    modifiers,
                })
            }
            Ev::Key(k, pressed) => {
                if pressed {
                    script.keys.insert(k.code);
                } else {
                    script.keys.remove(&k.code);
                }
                let state = if pressed { ButtonState::Pressed } else { ButtonState::Released };
                let text = k.ch.filter(|_| pressed).map(|c| c.to_string().into());
                keys.write(KeyboardInput { key_code: k.code, logical_key: k.logical.clone(), state, text, repeat: false, window });
                let modifiers = egui_modifiers(&script.keys);
                if let Some(key) = k.egui {
                    egui_events.write(EguiInputEvent {
                        context,
                        event: egui::Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers },
                    });
                }
                // A plain key press also types its character, as bevy_egui does.
                k.ch.filter(|_| pressed && !modifiers.ctrl && !modifiers.alt && !modifiers.command).map(|c| {
                    let c = if modifiers.shift { c.to_ascii_uppercase() } else { c };
                    egui::Event::Text(c.to_string())
                })
            }
            Ev::Text(t) => Some(egui::Event::Text(t)),
        };
        if let Some(event) = event {
            egui_events.write(EguiInputEvent { context, event });
        }
    }
}

fn egui_modifiers(keys: &BTreeSet<KeyCode>) -> egui::Modifiers {
    let has = |a, b| keys.contains(&a) || keys.contains(&b);
    let ctrl = has(KeyCode::ControlLeft, KeyCode::ControlRight);
    egui::Modifiers {
        alt: has(KeyCode::AltLeft, KeyCode::AltRight),
        ctrl,
        shift: has(KeyCode::ShiftLeft, KeyCode::ShiftRight),
        mac_cmd: false,
        command: ctrl,
    }
}

/// Copies egui's accessibility tree, built during the pass that just ended. Scroll
/// areas lay out all their contents, so each widget is cut to the part egui would
/// let you click (its clipped interaction rectangle), and ones scrolled out of sight
/// are left out.
fn read_widgets(mut widgets: ResMut<Widgets>, mut outputs: Query<(&EguiFullOutput, &mut EguiContext), With<PrimaryEguiContext>>) {
    let Ok((EguiFullOutput(Some(output)), mut ctx)) = outputs.single_mut() else { return };
    let Some(update) = &output.platform_output.accesskit_update else { return };
    let ctx = ctx.get_mut().clone();
    widgets.0 = update
        .nodes
        .iter()
        // Text runs repeat their label's text.
        .filter(|(_, n)| n.role() != accesskit::Role::TextRun)
        .filter_map(|(id, n)| {
            let b = n.bounds()?;
            let full = egui::Rect::from_min_max(egui::pos2(b.x0 as f32, b.y0 as f32), egui::pos2(b.x1 as f32, b.y1 as f32));
            // SAFETY: not a memory-safety matter; the value came from an egui Id, so it
            // has the entropy egui's id maps expect.
            let egui_id = (id.0 != 0).then(|| unsafe { egui::Id::from_high_entropy_bits(id.0) });
            let shown = match egui_id.and_then(|i| ctx.read_response(i)) {
                Some(r) => full.intersect(r.interact_rect),
                None => full,
            };
            if shown.width() < 1.0 || shown.height() < 1.0 {
                return None;
            }
            Some(Widget {
                role: format!("{:?}", n.role()),
                label: n.label().unwrap_or_default().to_string(),
                value: n.value().unwrap_or_default().to_string(),
                hint: n.placeholder().unwrap_or_default().to_string(),
                rect: shown,
                clipped: shown != full,
                disabled: n.is_disabled(),
                toggled: n.toggled().map(|t| t != accesskit::Toggled::False),
                clickable: n.supports_action(accesskit::Action::Click),
            })
        })
        .collect();
}

fn rect_json(r: egui::Rect) -> Value {
    json!([r.left(), r.top(), r.width(), r.height()])
}

fn panel_rect(ui: &Ui, name: &str) -> Result<egui::Rect, String> {
    ui.panels.iter().find(|(t, _)| t.title().eq_ignore_ascii_case(name)).map(|(_, r)| *r).ok_or_else(|| {
        let shown: Vec<_> = ui.panels.iter().map(|(t, _)| t.title()).collect();
        format!("panel {name} isn't shown (shown: {})", shown.join(", "))
    })
}

fn widget_list(widgets: &[Widget], ui: &Ui, args: &WidgetArgs) -> Result<Value, String> {
    let within = args.panel.as_deref().map(|p| panel_rect(ui, p)).transpose()?;
    let text = args.text.as_deref().map(str::to_lowercase);
    let role = args.role.as_deref().map(str::to_lowercase);
    let mut found: Vec<&Widget> = widgets
        .iter()
        .filter(|w| !w.label.is_empty() || !w.value.is_empty() || !w.hint.is_empty() || w.clickable)
        .filter(|w| within.is_none_or(|r| r.contains(w.rect.center())))
        .filter(|w| text.as_ref().is_none_or(|t| [&w.label, &w.value, &w.hint].iter().any(|s| s.to_lowercase().contains(t))))
        .filter(|w| role.as_ref().is_none_or(|r| w.role.to_lowercase().contains(r)))
        .collect();
    found.sort_by(|a, b| a.rect.top().total_cmp(&b.rect.top()).then(a.rect.left().total_cmp(&b.rect.left())));
    let total = found.len();
    let list: Vec<Value> = found
        .into_iter()
        .take(args.limit.unwrap_or(60))
        .map(|w| {
            let panel = ui.panels.iter().find(|(_, r)| r.contains(w.rect.center())).map(|(t, _)| t.title());
            let c = w.rect.center();
            let mut o = json!({"role": w.role, "rect": rect_json(w.rect), "centre": [c.x, c.y]});
            for (k, v) in [("label", &w.label), ("value", &w.value), ("hint", &w.hint)] {
                if !v.is_empty() {
                    o[k] = json!(v);
                }
            }
            if let Some(p) = panel {
                o["panel"] = json!(p);
            }
            if w.disabled {
                o["disabled"] = json!(true);
            }
            if w.clipped {
                o["clipped"] = json!(true);
            }
            if let Some(t) = w.toggled {
                o["toggled"] = json!(t);
            }
            o
        })
        .collect();
    Ok(json!({"widgets": list, "total": total}))
}

fn sel_json(sel: Sel) -> Value {
    match sel {
        Sel::None => Value::Null,
        Sel::Entity(id) => json!({"entity": id}),
        Sel::Light(i) => json!({"light": i}),
        Sel::Area(i) => json!({"area": i}),
    }
}

fn status(ui: &Ui, doc: &crate::doc::Doc, built: &Built, cam: &EditorCam, window: Option<&Window>, headless: bool, frame: u64) -> Value {
    let t = &ui.tools;
    let (undo, redo) = doc.history();
    let p = crate::view3d::to_world(cam.pos);
    let mut v = json!({
        "frame": frame,
        "file": doc.path,
        "dirty": doc.dirty(),
        "undo": undo,
        "redo": redo,
        "tool": t.tool.name(),
        "op": t.op.name(),
        "selection": sel_json(ui.sel),
        "faces_selected": t.faces.len(),
        "hover": t.hover.map(|h| json!({"cell": h.cell, "face": format!("{:?}", h.face)})),
        "link": t.link.map(|l| format!("{l:?}")),
        "place": {"kind": entity_kinds().get(t.place).map(|k| k.0), "yaw": t.yaw},
        "message": ui.message,
        "camera": {"pos": p, "yaw": cam.yaw, "pitch": cam.pitch},
        "view": {"mode": built.mode, "slice": built.slice, "cut": built.cut, "grid": built.show_grid, "markers": built.show_entities},
        "counts": {"entities": doc.level.entities.len(), "lights": doc.level.lights.len(), "areas": doc.level.areas.len(), "columns": doc.level.columns.len()},
        "panels": ui.panels.iter().map(|(t, r)| json!({"name": t.title(), "rect": rect_json(*r)})).collect::<Vec<_>>(),
        "plan": ui.plan.rect.map(|r| json!({"rect": rect_json(r), "centre": [ui.plan.centre.x, ui.plan.centre.y], "zoom": ui.plan.zoom})),
    });
    if let Some(w) = window {
        v["window"] = json!({"width": w.width(), "height": w.height(), "scale": w.scale_factor(), "headless": headless});
    }
    v
}

/// The view-space point a face of a cell is clicked at: its middle.
fn face_point(cell: [i32; 3], face: &str) -> Result<Vec3, String> {
    let [x, l, z] = cell.map(|c| c as f32);
    let (step, mid) = (STEP as f32, |c: f32| (c + 0.5) * TF);
    let w = match face.to_ascii_lowercase().as_str() {
        "floor" => [mid(x), -l * step, mid(z)],
        "ceiling" => [mid(x), -(l + 1.0) * step, mid(z)],
        "e" => [(x + 1.0) * TF, -(l + 0.5) * step, mid(z)],
        "w" => [x * TF, -(l + 0.5) * step, mid(z)],
        "s" => [mid(x), -(l + 0.5) * step, (z + 1.0) * TF],
        "n" => [mid(x), -(l + 0.5) * step, z * TF],
        f => return Err(format!("face {f}: floor, ceiling, n, s, e or w")),
    };
    Ok(to_view(w))
}

fn face_name(f: Face) -> &'static str {
    match f {
        Face::Floor => "floor",
        Face::Ceiling => "ceiling",
        Face::Wall(Dir::N) => "n",
        Face::Wall(Dir::S) => "s",
        Face::Wall(Dir::E) => "e",
        Face::Wall(Dir::W) => "w",
    }
}

/// Where to point at something: in the 3D view (and what the pointer would be over
/// there) and in the plan.
fn locate(ui: &Ui, doc: &Document, built: &Built, cam: &EditorCam, a: &LocateArgs) -> Result<Value, String> {
    let level = &doc.0.level;
    let (view, plan) = if let Some(id) = a.entity {
        let e = level.entities.iter().find(|e| e.id == id).ok_or(format!("no entity {id}"))?;
        (marker(e.pos), [e.pos[0] / TF, e.pos[2] / TF])
    } else if let Some(i) = a.light {
        let l = level.lights.get(i).ok_or(format!("no light {i}"))?;
        (to_view(l.pos), [l.pos[0] / TF, l.pos[2] / TF])
    } else if let Some(p) = a.point {
        (to_view(p), [p[0] / TF, p[2] / TF])
    } else if let Some(c) = a.cell {
        (face_point(c, a.face.as_deref().unwrap_or("floor"))?, [c[0] as f32 + 0.5, c[2] as f32 + 0.5])
    } else if let Some([x, z]) = a.tile {
        let v = to_view([(x as f32 + 0.5) * TF, -(built.slice as f32) * STEP as f32, (z as f32 + 0.5) * TF]);
        (v, [x as f32 + 0.5, z as f32 + 0.5])
    } else {
        return Err("give entity, light, point, cell or tile".into());
    };
    let mut out = json!({});
    // The 3D view fills the window; panels sit over it.
    if let Some(screen) = ui.screen {
        if let Some(p) = crate::pick::project(cam, screen, view) {
            let mut o = json!({"at": [p.x, p.y]});
            // What a click there would land on: a panel, or a face of the level.
            if let Some((t, _)) = ui.panels.iter().find(|(t, r)| *t != crate::ui::Tab::Viewport && r.contains(p)) {
                o["covered_by"] = json!(t.title());
            }
            if !screen.contains(p) {
                o["off_screen"] = json!(true);
            }
            if let Some(h) = crate::pick::pick(&built.scene, level, crate::pick::ray(cam, screen, p)) {
                o["hits"] = json!({"cell": h.cell, "face": face_name(h.face)});
            }
            out["viewport"] = o;
        }
    }
    if let Some(r) = ui.plan.rect {
        let p = r.center() + (egui::vec2(plan[0], plan[1]) - ui.plan.centre) * ui.plan.zoom;
        out["plan"] = json!({"at": [p.x, p.y], "visible": r.contains(p)});
    }
    Ok(out)
}

fn level(doc: &Document, part: Part) -> Result<Value, String> {
    let l = &doc.0.level;
    let j = |v: Result<Value, serde_json::Error>| v.map_err(|e| e.to_string());
    Ok(match part {
        Part::Entities => json!({"entities": j(serde_json::to_value(&l.entities))?}),
        Part::Lights => json!({"lights": j(serde_json::to_value(&l.lights))?}),
        Part::Areas => json!({"areas": j(serde_json::to_value(&l.areas))?}),
        Part::Ron => json!({"ron": ron::ser::to_string_pretty(l, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?}),
    })
}

fn setup(ui: &mut Ui, doc: &mut Document, cam: &mut EditorCam, window: Option<Mut<Window>>, a: SetupArgs) -> Result<(), String> {
    use engine::world::level::Level;
    if let Some(path) = &a.open {
        doc.0 = crate::doc::Doc::open(path)?;
        ui.reset();
        cam.frame(crate::ui::level_centre(&doc.0.level), 9.0);
    } else if a.new == Some(true) {
        doc.0 = crate::doc::Doc::new(if a.demo { Level::demo() } else { Level::new("") }, None);
        ui.reset();
        cam.frame(crate::ui::level_centre(&doc.0.level), 9.0);
    }
    if let Some(c) = a.camera {
        if let Some(y) = c.yaw {
            // EditorCam's yaw of 0 looks along view -z, which is world +z.
            cam.yaw = y;
        }
        if let Some(p) = c.pitch {
            cam.pitch = p;
        }
        if let Some(at) = c.look_at {
            cam.frame(to_view(at), c.distance.unwrap_or(cam.focus));
        }
    }
    if let Some([w, h]) = a.size {
        let mut window = window.ok_or("no window")?;
        window.resolution.set(w as f32, h as f32);
    }
    Ok(())
}

fn screenshot(commands: &mut Commands, ui: &Ui, headless: Option<&HeadlessTarget>, scale: f32, a: ShotArgs, reply: Reply) {
    let crop = match (&a.panel, a.rect) {
        (Some(p), _) => match panel_rect(ui, p) {
            Ok(r) => Some(r),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        },
        (None, Some([x, y, w, h])) => Some(egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h))),
        (None, None) => None,
    };
    let shot = match headless.and_then(|h| h.0.clone()) {
        Some(image) => Screenshot::image(image),
        None => Screenshot::primary_window(),
    };
    let mut reply = Some(reply);
    commands.spawn(shot).observe(move |capture: On<ScreenshotCaptured>| {
        let Some(reply) = reply.take() else { return };
        let result = (|| {
            let mut image = capture.image.clone().try_into_dynamic().map_err(|e| e.to_string())?;
            if let Some(r) = crop {
                let (x, y) = ((r.left() * scale).max(0.0) as u32, (r.top() * scale).max(0.0) as u32);
                let w = ((r.width() * scale) as u32).min(image.width().saturating_sub(x));
                let h = ((r.height() * scale) as u32).min(image.height().saturating_sub(y));
                image = image.crop_imm(x, y, w.max(1), h.max(1));
            }
            let (width, height) = (image.width(), image.height());
            let mut png = std::io::Cursor::new(Vec::new());
            image.write_to(&mut png, image::ImageFormat::Png).map_err(|e| e.to_string())?;
            Ok(Response::Image { png: png.into_inner(), metadata: json!({"width": width, "height": height, "scale": scale}) })
        })();
        let _ = reply.send(result);
    });
}

struct EditorServer {
    sender: mpsc::SyncSender<Request>,
    slots: Arc<Semaphore>,
}

fn tools() -> Vec<Tool> {
    let pair = json!({"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2});
    let button = json!({"type": "string", "enum": ["left", "right", "middle"], "default": "left"});
    let mods = json!({"type": "array", "items": {"type": "string", "enum": ["Ctrl", "Shift", "Alt"]}, "description": "Modifiers held throughout"});
    let step = json!({"type": "object", "required": ["do"], "properties": {
        "do": {"type": "string", "enum": ["move", "click", "double_click", "press", "release", "drag", "wheel", "key", "key_down", "key_up", "hold", "text", "wait"]},
        "to": pair, "at": pair, "from": pair, "button": button, "with": mods,
        "frames": {"type": "integer", "minimum": 0, "maximum": 3600},
        "lines": {"type": "number", "description": "wheel: positive scrolls up / zooms in"},
        "key": {"type": "string", "description": "A key or chord: \"A\", \"Tab\", \"Escape\", \"PageUp\", \"F5\", \"Delete\", \"Ctrl+Z\", \"Shift+F5\""},
        "text": {"type": "string"}
    }, "description": "move {to, frames=1} | click {at, button, with} | double_click {at, button} | press/release {button, at?} | drag {from, to, button, frames=10, with} | wheel {lines, at?} | key {key} | key_down/key_up {key} | hold {key, frames} | text {text} | wait {frames}"});
    [
        ("editor_status", "The editor's state: file, unsaved changes, undo depth, tool and drag under way, selection, the face under the pointer, the status message, camera, view toggles, counts, where each panel and the plan map are, and the window.", json!({"type": "object", "properties": {}, "additionalProperties": false})),
        ("editor_input", "Play mouse and keyboard input into the editor, a frame at a time, through the same path real input takes (egui and the 3D camera). Positions are window pixels, origin top-left. Returns editor_status once the editor has seen the last step. Only injected input reaches egui while the MCP runs.", json!({"type": "object", "properties": {"steps": {"type": "array", "items": step, "minItems": 1, "maxItems": 256}}, "required": ["steps"], "additionalProperties": false})),
        ("editor_widgets", "Find widgets egui drew last frame (buttons, labels, check boxes, combo boxes, text fields and their hints, menu items) by text, role or panel, with their rectangles [x, y, w, h] and centres to click.", json!({"type": "object", "properties": {"text": {"type": "string", "description": "Label, value or hint contains this (any case)"}, "role": {"type": "string", "description": "Role contains this: button, checkbox, combobox, label, textinput..."}, "panel": {"type": "string", "description": "Tab title: Viewport, Plan, Tools, Properties, Entities, Textures, Areas, Issues"}, "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 60}}, "additionalProperties": false})),
        ("editor_locate", "Where something is on screen: in the 3D view (with what a click there would hit, and the panel covering it, if any) and in the plan. Give one of entity (id), light (index), point (world units, y down), cell [x, layer, z] with face (floor, ceiling, n, s, e, w), or tile [x, z] (at the plan's slice).", json!({"type": "object", "properties": {"entity": {"type": "integer"}, "light": {"type": "integer"}, "point": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "cell": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}, "face": {"type": "string", "enum": ["floor", "ceiling", "n", "s", "e", "w"]}, "tile": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}, "additionalProperties": false})),
        ("editor_level", "Read the level being edited: its entities, lights or areas as JSON, or the whole level as RON (part: ron).", json!({"type": "object", "properties": {"part": {"type": "string", "enum": ["entities", "lights", "areas", "ron"], "default": "entities"}}, "additionalProperties": false})),
        ("editor_setup", "Set up a test: open a level file or start a new one (dropping unsaved changes and history), point the camera, or resize the window. Returns editor_status.", json!({"type": "object", "properties": {"open": {"type": "string"}, "new": {"type": "boolean"}, "demo": {"type": "boolean", "description": "with new: the demo level"}, "camera": {"type": "object", "properties": {"yaw": {"type": "number"}, "pitch": {"type": "number"}, "look_at": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "distance": {"type": "number"}}, "additionalProperties": false}, "size": {"type": "array", "items": {"type": "integer", "minimum": 200, "maximum": 4096}, "minItems": 2, "maxItems": 2}}, "additionalProperties": false})),
        ("editor_screenshot", "PNG of the editor as rendered: the whole window, one panel, or a rectangle [x, y, w, h].", json!({"type": "object", "properties": {"panel": {"type": "string"}, "rect": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}, "additionalProperties": false})),
    ]
    .into_iter()
    .map(|(name, description, schema)| Tool::new(name, description, Arc::new(schema.as_object().unwrap().clone())))
    .collect()
}

fn parse<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, ErrorData> {
    serde_json::from_value(value).map_err(|e| ErrorData::invalid_params(e.to_string(), None))
}

impl ServerHandler for EditorServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")))
            .with_instructions("neurofield level editor developer tools. Drive it with editor_input (window pixels); find where to click with editor_widgets and editor_locate; check results with editor_status, editor_level and editor_screenshot. stdout is MCP only.")
    }
    fn get_tool(&self, name: &str) -> Option<Tool> {
        tools().into_iter().find(|t| t.name == name)
    }
    // ListToolsResult is non-exhaustive, so it cannot be built with a literal.
    #[allow(clippy::field_reassign_with_default)]
    async fn list_tools(&self, _: Option<PaginatedRequestParams>, _: RequestContext<RoleServer>) -> Result<ListToolsResult, ErrorData> {
        let mut result = ListToolsResult::default();
        result.tools = tools();
        Ok(result)
    }
    async fn call_tool(&self, request: CallToolRequestParams, context: RequestContext<RoleServer>) -> Result<CallToolResponse, ErrorData> {
        let value = Value::Object(request.arguments.unwrap_or_default());
        let command = match request.name.as_ref() {
            "editor_status" => Command::Status,
            "editor_input" => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Args {
                    steps: Vec<Step>,
                }
                let steps = parse::<Args>(value)?.steps;
                if steps.is_empty() || steps.len() > 256 {
                    return Err(ErrorData::invalid_params("give 1..256 steps", None));
                }
                Command::Input(steps)
            }
            "editor_widgets" => Command::Widgets(parse(value)?),
            "editor_locate" => Command::Locate(parse(value)?),
            "editor_level" => Command::Level(parse(value)?),
            "editor_setup" => Command::Setup(parse(value)?),
            "editor_screenshot" => Command::Screenshot(parse(value)?),
            _ => return Err(ErrorData::invalid_params("unknown editor tool", None)),
        };
        let _permit = tokio::select! {
            permit = self.slots.acquire() => permit.map_err(|_| ErrorData::internal_error("editor queue closed", None))?,
            _ = context.ct.cancelled() => return Err(ErrorData::internal_error("request cancelled", None)),
        };
        let (reply, receiver) = oneshot::channel();
        self.sender
            .try_send(Request { command, reply })
            .map_err(|e| ErrorData::internal_error(format!("editor queue unavailable: {e}"), None))?;
        let response = tokio::select! {
            result = tokio::time::timeout(Duration::from_secs(300), receiver) => result
                .map_err(|_| ErrorData::internal_error("editor response timed out", None))?
                .map_err(|_| ErrorData::internal_error("editor stopped before replying", None))?
                .map_err(|e| ErrorData::invalid_params(e, None))?,
            _ = context.ct.cancelled() => return Err(ErrorData::internal_error("request cancelled", None)),
        };
        Ok(match response {
            Response::Json(value) => CallToolResult::structured(value),
            Response::Image { png, metadata } => CallToolResult::success(vec![
                ContentBlock::image(base64::engine::general_purpose::STANDARD.encode(png), "image/png"),
                ContentBlock::text(metadata.to_string()),
            ]),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chords_and_steps_become_frames() {
        let k = chord("Ctrl+Shift+Z").unwrap();
        assert_eq!(k.iter().map(|k| k.code).collect::<Vec<_>>(), [KeyCode::ControlLeft, KeyCode::ShiftLeft, KeyCode::KeyZ]);
        assert_eq!(chord("PageUp").unwrap()[0].code, KeyCode::PageUp);
        assert_eq!(chord("F5").unwrap()[0].egui, Some(egui::Key::F5));
        assert_eq!(chord("Space").unwrap()[0].ch, Some(' '));
        assert!(chord("Hyper").is_err());
        let mut at = Vec2::ZERO;
        let drag = Step::Drag { from: [10.0, 10.0], to: [30.0, 10.0], button: Button::Left, frames: 4, with: vec!["Ctrl".into()] };
        let f = frames(&drag, &mut at).unwrap();
        // Ctrl and move there, press, four moves, release, Ctrl up.
        assert_eq!(f.len(), 8);
        assert!(matches!(f[1][0], Ev::Button(MouseButton::Left, _, true)));
        assert!(matches!(f[6][0], Ev::Button(MouseButton::Left, _, false)));
        assert!(matches!(f[7][0], Ev::Key(_, false)));
        assert_eq!(at, Vec2::new(30.0, 10.0));
        let step: Step = serde_json::from_value(json!({"do": "click", "at": [5, 6]})).unwrap();
        assert_eq!(frames(&step, &mut at).unwrap().len(), 3);
    }
}
