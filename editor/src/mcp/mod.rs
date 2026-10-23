//! The editor's MCP server: tools for agents that build levels or help someone
//! build theirs, and (in developer builds) for testing the editor.
//!
//! Two transports serve the same tools:
//! - `--mcp`: JSON-RPC on stdio, for an agent that starts the editor itself (stdout
//!   then carries nothing else). With `--headless` there is no OS window: the
//!   window is a plain Bevy entity and everything draws into an image (see
//!   `headless_display`).
//! - streamable HTTP at `/mcp`, so an agent can join an editor someone already has
//!   open: switched on in Tools > Agents (or with `--mcp-listen 127.0.0.1:7801`).
//!   Only local addresses are accepted, and every request needs the bearer token
//!   the Agents window shows (or `--mcp-token`). See `agents`.
//!
//! The tools come in sets:
//! - reading and viewing: `editor_status`, `editor_level`, `editor_locate`,
//!   `editor_screenshot`, `editor_camera`;
//! - editing (`edit`): the level changed directly, each call one undo step, while
//!   the person at the editor keeps their mouse and keyboard;
//! - testing (`input`, the `dev-mcp` feature, stdio only): mouse and keyboard
//!   played in through the real input path, and widgets found by label.
//!
//! Positions on screen are window pixels (egui points), origin top-left; positions
//! in the level are world units (a tile is 256, y down) or cells (x, layer, z).
//! The server threads hand requests to the main thread over a channel and wait for
//! replies, as psx-bevy's game MCP does.

mod agents;
mod edit;
#[cfg(feature = "dev-mcp")]
mod input;

use crate::tools::{entity_kinds, marker, Sel};
use crate::ui::Ui;
use crate::view3d::{to_view, Built, EditorCam};
use crate::Document;
use base64::Engine;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::{CachedPipelineState, PipelineCache, TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::window::PrimaryWindow;
use bevy_egui::egui;
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
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use tokio::sync::{oneshot, Semaphore};

const TF: f32 = T as f32;

pub(crate) use agents::{window as agents_window, Agents};

pub struct EditorMcpPlugin {
    /// Serve on stdio; the editor quits when that client goes.
    pub stdio: bool,
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
    /// From an agent over HTTP (shown in the Agents window).
    remote: bool,
}

enum Command {
    Status,
    Locate(LocateArgs),
    Level(LevelArgs),
    Camera(CameraArgs),
    Screenshot(ShotArgs),
    Edit(edit::Edit),
    #[cfg(feature = "dev-mcp")]
    Testing(input::Command),
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
    Summary,
    Entities,
    Lights,
    Areas,
    Cells,
    Paint,
    Settings,
    Ron,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CameraArgs {
    /// Radians: yaw about the vertical (0 looks toward world +z), pitch up.
    yaw: Option<f32>,
    pitch: Option<f32>,
    /// World point to look at, from `distance` tiles away.
    look_at: Option<[f32; 3]>,
    /// Look at this entity instead.
    entity: Option<u32>,
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
    /// Set when the stdio client goes away.
    closed: Arc<AtomicBool>,
    ready: RenderReady,
    /// Screenshots asked for before the first frames rendered.
    waiting: Vec<(ShotArgs, Reply)>,
}

#[derive(Resource, Clone)]
struct RenderReady(Arc<AtomicBool>);

/// `--headless`: the image the window's camera draws into.
#[derive(Resource, Default)]
struct HeadlessTarget(Option<Handle<Image>>);

/// Frames since the server started, reported in replies.
#[derive(Resource, Default)]
struct Frame(u64);

/// The server threads and what they hand requests on with.
#[derive(Resource)]
struct Servers {
    /// Made when first needed: most editors never serve anything.
    runtime: Option<tokio::runtime::Runtime>,
    sender: mpsc::SyncSender<Request>,
    /// One request at a time, from either transport: each waits for the main
    /// thread's answer.
    slots: Arc<Semaphore>,
    http: Option<agents::Http>,
}

impl Servers {
    fn runtime(&mut self) -> &tokio::runtime::Runtime {
        self.runtime.get_or_insert_with(|| {
            tokio::runtime::Builder::new_multi_thread().worker_threads(2).thread_name("editor-mcp").enable_all().build().unwrap()
        })
    }
}

impl Plugin for EditorMcpPlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = mpsc::sync_channel(16);
        let closed = Arc::new(AtomicBool::new(false));
        let mut servers = Servers { runtime: None, sender, slots: Arc::new(Semaphore::new(1)), http: None };
        if self.stdio {
            // Requests are served even while the window is unfocused.
            app.insert_resource(bevy::winit::WinitSettings::continuous());
            let server = EditorServer { sender: servers.sender.clone(), slots: servers.slots.clone(), testing: cfg!(feature = "dev-mcp"), remote: false };
            let closed = closed.clone();
            servers.runtime().spawn(async move {
                match server.serve(rmcp::transport::stdio()).await {
                    Ok(service) => {
                        if let Err(e) = service.waiting().await {
                            eprintln!("editor MCP: {e}");
                        }
                    }
                    Err(e) => eprintln!("editor MCP startup: {e}"),
                }
                closed.store(true, Ordering::Release);
            });
        }
        let ready = RenderReady(Arc::new(AtomicBool::new(false)));
        app.sub_app_mut(RenderApp)
            .insert_resource(ready.clone())
            .add_systems(Render, mark_render_ready.after(RenderSystems::Render).before(RenderSystems::Cleanup));
        app.insert_resource(Bridge { receiver: Mutex::new(receiver), closed, ready, waiting: Vec::new() })
            .insert_resource(servers)
            .init_resource::<Frame>()
            .add_systems(PreUpdate, (agents::switch, serve).chain());
        #[cfg(feature = "dev-mcp")]
        if self.stdio {
            input::build(app);
        }
        if self.headless {
            app.init_resource::<HeadlessTarget>().add_systems(Update, headless_display);
        }
    }
}

/// A token nobody can guess: 128 bits from the standard library's per-process
/// random hasher keys and the time.
fn new_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let part = |salt: u64| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(salt);
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
        h.finish()
    };
    format!("{:016x}{:016x}", part(1), part(2))
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
        if layers.intersects(&RenderLayers::layer(DISPLAY_LAYER)) && !matches!(&*t, RenderTarget::Image(i) if i.handle == handle) {
            *t = RenderTarget::Image(handle.clone().into());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn serve(
    mut commands: Commands,
    mut bridge: ResMut<Bridge>,
    mut frame: ResMut<Frame>,
    mut ui: ResMut<Ui>,
    mut doc: ResMut<Document>,
    mut cam: ResMut<EditorCam>,
    built: Res<Built>,
    headless: Option<Res<HeadlessTarget>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    #[cfg(feature = "dev-mcp")] mut testing: Option<ResMut<input::Pending>>,
    mut exit: MessageWriter<AppExit>,
) {
    frame.0 += 1;
    if bridge.closed.load(Ordering::Acquire) {
        exit.write(AppExit::Success);
        return;
    }
    // Answers describe the editor as drawn, so wait for the first frame of panels.
    if ui.screen.is_none() {
        return;
    }
    let scale = windows.single().map_or(1.0, |w| w.scale_factor());
    let ready = bridge.ready.0.load(Ordering::Acquire);
    if ready {
        for (args, reply) in std::mem::take(&mut bridge.waiting) {
            screenshot(&mut commands, &ui, headless.as_deref(), scale, args, reply);
        }
    }
    let requests: Vec<Request> = bridge.receiver.lock().unwrap().try_iter().collect();
    for Request { command, reply, remote } in requests {
        if reply.is_closed() {
            continue;
        }
        if remote {
            ui.agents.last_request = Some(std::time::Instant::now());
        }
        let result = match command {
            Command::Status => Ok(status(&ui, &doc.0, &built, &cam, windows.single().ok(), headless.is_some(), frame.0)),
            Command::Locate(args) => locate(&ui, &doc, &built, &cam, &args),
            Command::Level(args) => level(&doc, args.part),
            Command::Camera(args) => camera(&doc, &mut cam, args),
            Command::Edit(e) => edit::apply(e, &mut ui, &mut doc.0, &mut cam),
            Command::Screenshot(args) => {
                if ready {
                    screenshot(&mut commands, &ui, headless.as_deref(), scale, args, reply);
                } else {
                    bridge.waiting.push((args, reply));
                }
                continue;
            }
            #[cfg(feature = "dev-mcp")]
            Command::Testing(t) => match testing.as_mut() {
                Some(pending) => {
                    pending.0.push((t, reply));
                    continue;
                }
                None => Err("testing tools are only on stdio".into()),
            },
        };
        let _ = reply.send(result.map(Response::Json));
    }
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
        "hover": t.hover.map(|h| json!({"cell": h.cell, "face": face_name(h.face)})),
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
fn face_point(cell: [i32; 3], face: Face) -> Vec3 {
    let [x, l, z] = cell.map(|c| c as f32);
    let (step, mid) = (STEP as f32, |c: f32| (c + 0.5) * TF);
    to_view(match face {
        Face::Floor => [mid(x), -l * step, mid(z)],
        Face::Ceiling => [mid(x), -(l + 1.0) * step, mid(z)],
        Face::Wall(Dir::E) => [(x + 1.0) * TF, -(l + 0.5) * step, mid(z)],
        Face::Wall(Dir::W) => [x * TF, -(l + 0.5) * step, mid(z)],
        Face::Wall(Dir::S) => [mid(x), -(l + 0.5) * step, (z + 1.0) * TF],
        Face::Wall(Dir::N) => [mid(x), -(l + 0.5) * step, z * TF],
    })
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

/// "floor", "ceiling", or a wall by the side of the cell it's on: "n", "e", "s", "w".
fn face_of(name: &str) -> Result<Face, String> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "floor" => Face::Floor,
        "ceiling" => Face::Ceiling,
        d => Face::Wall(dir_of(d).map_err(|_| format!("face {name}: floor, ceiling, n, e, s or w"))?),
    })
}

fn dir_of(name: &str) -> Result<Dir, String> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "n" | "north" => Dir::N,
        "e" | "east" => Dir::E,
        "s" | "south" => Dir::S,
        "w" | "west" => Dir::W,
        _ => return Err(format!("direction {name}: n, e, s or w")),
    })
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
        (face_point(c, face_of(a.face.as_deref().unwrap_or("floor"))?), [c[0] as f32 + 0.5, c[2] as f32 + 0.5])
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
        Part::Summary => {
            let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
            for (&(x, z), runs) in &l.columns {
                for r in runs {
                    for (i, (a, b)) in [(x, x + 1), (r.from, r.to), (z, z + 1)].into_iter().enumerate() {
                        lo[i] = lo[i].min(a);
                        hi[i] = hi[i].max(b);
                    }
                }
            }
            json!({
                "file": doc.0.path, "name": l.name, "textures": l.textures,
                "columns": l.columns.len(),
                "bounds": (!l.columns.is_empty()).then(|| json!({"min": lo, "max": hi})),
                "entities": l.entities.iter().map(|e| json!({"id": e.id, "kind": crate::ui::kind_name(&e.kind), "pos": e.pos})).collect::<Vec<_>>(),
                "lights": l.lights.len(), "areas": l.areas.iter().map(|a| &a.name).collect::<Vec<_>>(),
                "painted_faces": l.paint.len(),
            })
        }
        Part::Entities => json!({"entities": j(serde_json::to_value(&l.entities))?}),
        Part::Lights => json!({"lights": j(serde_json::to_value(&l.lights))?}),
        Part::Areas => json!({"areas": j(serde_json::to_value(&l.areas))?}),
        // JSON keys must be strings, so the maps become lists.
        Part::Cells => json!({"columns": l.columns.iter().map(|(&(x, z), runs)| json!({"x": x, "z": z, "runs": runs})).collect::<Vec<_>>()}),
        Part::Paint => json!({"paint": l.paint.iter().map(|(&((x, y, z), f), &slot)| json!({"cell": [x, y, z], "face": face_name(f), "slot": slot})).collect::<Vec<_>>()}),
        Part::Settings => j(serde_json::to_value(edit::Settings::of(l)))?,
        Part::Ron => json!({"ron": l.to_ron()}),
    })
}

fn camera(doc: &Document, cam: &mut EditorCam, c: CameraArgs) -> Result<Value, String> {
    if let Some(y) = c.yaw {
        cam.yaw = y;
    }
    if let Some(p) = c.pitch {
        cam.pitch = p.clamp(-1.55, 1.55);
    }
    let at = match (c.entity, c.look_at) {
        (Some(id), _) => Some(doc.0.level.entities.iter().find(|e| e.id == id).ok_or(format!("no entity {id}"))?.pos),
        (None, at) => at,
    };
    if let Some(at) = at {
        cam.frame(to_view(at), c.distance.unwrap_or(cam.focus).clamp(0.5, 200.0));
    }
    Ok(json!({"pos": crate::view3d::to_world(cam.pos), "yaw": cam.yaw, "pitch": cam.pitch}))
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

#[derive(Clone)]
struct EditorServer {
    sender: mpsc::SyncSender<Request>,
    slots: Arc<Semaphore>,
    /// Whether this client gets the testing tools (developer builds, stdio).
    #[cfg_attr(not(feature = "dev-mcp"), allow(dead_code))]
    testing: bool,
    /// Over HTTP.
    remote: bool,
}

fn schema(v: Value) -> Arc<serde_json::Map<String, Value>> {
    Arc::new(v.as_object().unwrap().clone())
}

/// The reading and viewing tools.
fn view_tools() -> Vec<(&'static str, &'static str, Value)> {
    let triple = |t: &str| json!({"type": "array", "items": {"type": t}, "minItems": 3, "maxItems": 3});
    vec![
        ("editor_status", "The editor's state: file, unsaved changes, undo depth, the tool and any drag under way, selection, the face under the pointer, the status message, camera, view toggles, counts, where each panel and the plan map are, and the window.", json!({"type": "object", "properties": {}, "additionalProperties": false})),
        ("editor_level", "Read the level being edited. part: summary (default: bounds, entity list, area names), entities, lights, areas, cells (each column's open runs of layers), paint (painted faces), settings (name, textures, ambient, fog, sun, surface, strangeness), or ron (the whole file).", json!({"type": "object", "properties": {"part": {"type": "string", "enum": ["summary", "entities", "lights", "areas", "cells", "paint", "settings", "ron"], "default": "summary"}}, "additionalProperties": false})),
        ("editor_locate", "Where something is on screen: in the 3D view (with what a click there would hit, and the panel covering it, if any) and in the plan. Give one of entity (id), light (index), point (world units, y down), cell [x, layer, z] with face (floor, ceiling, n, e, s, w), or tile [x, z] (at the plan's slice).", json!({"type": "object", "properties": {"entity": {"type": "integer"}, "light": {"type": "integer"}, "point": triple("number"), "cell": triple("integer"), "face": {"type": "string", "enum": ["floor", "ceiling", "n", "e", "s", "w"]}, "tile": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}, "additionalProperties": false})),
        ("editor_camera", "Point the 3D view: look at a world point or an entity from distance tiles away, and/or set yaw and pitch (radians; yaw 0 looks toward world +z, pitch up). Moves the view of whoever is using the editor.", json!({"type": "object", "properties": {"look_at": triple("number"), "entity": {"type": "integer"}, "distance": {"type": "number"}, "yaw": {"type": "number"}, "pitch": {"type": "number"}}, "additionalProperties": false})),
        ("editor_screenshot", "PNG of the editor as rendered: the whole window, one panel (its tab title, e.g. Viewport), or a rectangle [x, y, w, h].", json!({"type": "object", "properties": {"panel": {"type": "string"}, "rect": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}, "additionalProperties": false})),
    ]
}

impl EditorServer {
    fn tools(&self) -> Vec<Tool> {
        #[allow(unused_mut)]
        let mut all = view_tools();
        all.extend(edit::tools());
        #[cfg(feature = "dev-mcp")]
        if self.testing {
            all.extend(input::tools());
        }
        all.into_iter().map(|(name, description, s)| Tool::new(name, description, schema(s))).collect()
    }

    fn command(&self, name: &str, value: Value) -> Result<Command, ErrorData> {
        fn de<T: serde::de::DeserializeOwned>(v: Value) -> Result<T, ErrorData> {
            serde_json::from_value(v).map_err(|e| ErrorData::invalid_params(e.to_string(), None))
        }
        let bad = |e: String| ErrorData::invalid_params(e, None);
        Ok(match name {
            "editor_status" => Command::Status,
            "editor_level" => Command::Level(de(value)?),
            "editor_locate" => Command::Locate(de(value)?),
            "editor_camera" => Command::Camera(de(value)?),
            "editor_screenshot" => Command::Screenshot(de(value)?),
            _ => {
                if let Some(e) = edit::parse(name, value.clone()) {
                    return e.map(Command::Edit).map_err(bad);
                }
                #[cfg(feature = "dev-mcp")]
                if self.testing {
                    if let Some(t) = input::parse(name, value) {
                        return t.map(Command::Testing).map_err(bad);
                    }
                }
                return Err(ErrorData::invalid_params(format!("unknown editor tool {name}"), None));
            }
        })
    }
}

impl ServerHandler for EditorServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "neurofield level editor. Levels are columns of open cells on a grid: x and z in tiles, height in layers \
                 (a tile is 256 world units, a layer 64; world y points down, so layer L's floor is at y = -64 L). \
                 Read with editor_level and editor_status, look with editor_camera and editor_screenshot, and change \
                 the level with the level_* tools: each call is one undo step the person at the editor can undo. \
                 Open, new and save with editor_open and editor_save.",
            )
    }
    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.tools().into_iter().find(|t| t.name == name)
    }
    // ListToolsResult is non-exhaustive, so it cannot be built with a literal.
    #[allow(clippy::field_reassign_with_default)]
    async fn list_tools(&self, _: Option<PaginatedRequestParams>, _: RequestContext<RoleServer>) -> Result<ListToolsResult, ErrorData> {
        let mut result = ListToolsResult::default();
        result.tools = self.tools();
        Ok(result)
    }
    async fn call_tool(&self, request: CallToolRequestParams, context: RequestContext<RoleServer>) -> Result<CallToolResponse, ErrorData> {
        let command = self.command(request.name.as_ref(), Value::Object(request.arguments.unwrap_or_default()))?;
        let _permit = tokio::select! {
            permit = self.slots.acquire() => permit.map_err(|_| ErrorData::internal_error("editor queue closed", None))?,
            _ = context.ct.cancelled() => return Err(ErrorData::internal_error("request cancelled", None)),
        };
        let (reply, receiver) = oneshot::channel();
        self.sender
            .try_send(Request { command, reply, remote: self.remote })
            .map_err(|e| ErrorData::internal_error(format!("editor queue unavailable: {e}"), None))?;
        let response = tokio::select! {
            result = tokio::time::timeout(Duration::from_secs(300), receiver) => result
                .map_err(|_| ErrorData::internal_error("editor response timed out", None))?
                .map_err(|_| ErrorData::internal_error("editor stopped before replying", None))?,
            _ = context.ct.cancelled() => return Err(ErrorData::internal_error("request cancelled", None)),
        };
        // A refused edit is the tool's answer, not a protocol error: the agent reads
        // why and can try again.
        Ok(match response {
            Ok(Response::Json(value)) => CallToolResult::structured(value),
            Ok(Response::Image { png, metadata }) => CallToolResult::success(vec![
                ContentBlock::image(base64::engine::general_purpose::STANDARD.encode(png), "image/png"),
                ContentBlock::text(metadata.to_string()),
            ]),
            Err(e) => CallToolResult::error(vec![ContentBlock::text(e)]),
        }
        .into())
    }
}

/// Paths agents give are tried as they are, then in the levels folder.
fn level_path(ui: &Ui, p: &str) -> PathBuf {
    let p = PathBuf::from(p);
    if p.is_absolute() || p.exists() || p.parent().is_some_and(|d| !d.as_os_str().is_empty() && d.exists()) {
        p
    } else {
        ui.levels_dir().join(p)
    }
}
