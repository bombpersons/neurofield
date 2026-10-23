//! Everything drawn with egui over the 3D view: the menu bar, the status bar and
//! the dockable panels. The dock's centre (the Viewport tab) is see-through, and
//! the pointer over it drives the 3D camera.

use crate::doc::Doc;
use crate::settings::Settings;
use crate::tools::{links, Link, Sel, Tool, Tools};
use crate::view3d::{entity_colour, to_view, to_world, Built, EditorCam, ViewInput, ViewMode};
use crate::Document;
use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future::poll_once, AsyncComputeTaskPool, Task};
use bevy_egui::{egui, EguiContexts};
use engine::world::config::Config;
use engine::world::creature::Kind;
use engine::world::level::{Entity as LevelEntity, EntityKind, Hinge, Level, PropShape, SlopeKind, Sun};
use engine::world::scene::{STEP, T};
use egui::{Color32, Key, KeyboardShortcut, Modifiers, RichText};
use egui_dock::{DockArea, DockState, NodeIndex, TabViewer};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const TF: f32 = T as f32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tab {
    Viewport,
    Plan,
    Tools,
    Properties,
    Entities,
    Textures,
    Areas,
    Issues,
}

impl Tab {
    /// Every panel but the viewport, which never closes.
    const PANELS: [Tab; 7] = [Tab::Plan, Tab::Tools, Tab::Properties, Tab::Entities, Tab::Textures, Tab::Areas, Tab::Issues];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Viewport => "Viewport",
            Tab::Plan => "Plan",
            Tab::Tools => "Tools",
            Tab::Properties => "Properties",
            Tab::Entities => "Entities",
            Tab::Textures => "Textures",
            Tab::Areas => "Areas",
            Tab::Issues => "Issues",
        }
    }
}

pub fn default_dock() -> DockState<Tab> {
    let mut d = DockState::new(vec![Tab::Viewport]);
    let tree = d.main_surface_mut();
    // Each split's fraction is what the old node keeps.
    let [main, _tools] = tree.split_left(NodeIndex::root(), 0.12, vec![Tab::Tools]);
    let [main, right] = tree.split_right(main, 0.74, vec![Tab::Properties]);
    tree.split_below(right, 0.5, vec![Tab::Entities, Tab::Areas, Tab::Issues]);
    tree.split_below(main, 0.7, vec![Tab::Plan, Tab::Textures]);
    d
}

/// Something that waits on the user saying what to do with unsaved changes.
#[derive(Clone, Debug)]
enum After {
    New,
    Open(Option<PathBuf>),
    Quit,
}

#[derive(Resource)]
pub struct Ui {
    pub settings: Settings,
    pub config: Config,
    pub quit_requested: bool,
    pub(crate) tools: Tools,
    pub(crate) sel: Sel,
    pub(crate) message: String,
    /// The unsaved-changes question, and what to do once it is answered.
    confirm: Option<After>,
    /// An answered (or unasked) New / Open / Quit, to carry out this frame.
    confirm_done: Option<After>,
    quit_now: bool,
    open_dialog: Option<Task<Option<PathBuf>>>,
    save_dialog: Option<Task<Option<PathBuf>>>,
    /// What to do once a Save As finishes.
    after_save: Option<After>,
    styled: bool,
    /// The file the plan view was last centred for, and whether the level was
    /// empty then (so it centres again once something is built).
    plan_for: Option<(Option<PathBuf>, bool)>,
    fps: f32,
    pub(crate) plan: PlanView,
    /// The Entities outliner's filter.
    filter: String,
    /// Where each shown panel was drawn this frame (for the developer MCP).
    pub(crate) panels: Vec<(Tab, egui::Rect)>,
    /// The whole window in egui points, which the 3D view fills.
    pub(crate) screen: Option<egui::Rect>,
    /// The agent server and its window (Tools > Agents).
    pub(crate) agents: crate::mcp::Agents,
}

pub(crate) struct PlanView {
    /// Tile coordinates at the panel's centre, and pixels per tile.
    pub centre: egui::Vec2,
    pub zoom: f32,
    /// The map's own rectangle, when the Plan panel was drawn this frame.
    pub rect: Option<egui::Rect>,
}

impl Ui {
    pub fn new(mut settings: Settings, config: Config) -> Ui {
        let agents = crate::mcp::Agents::new(&mut settings);
        Ui {
            settings,
            config,
            quit_requested: false,
            tools: Tools::default(),
            sel: Sel::None,
            message: String::new(),
            confirm: None,
            confirm_done: None,
            quit_now: false,
            open_dialog: None,
            save_dialog: None,
            after_save: None,
            styled: false,
            plan_for: None,
            fps: 60.0,
            plan: PlanView { centre: egui::vec2(8.0, 8.0), zoom: 18.0, rect: None },
            filter: String::new(),
            panels: Vec::new(),
            screen: None,
            agents,
        }
    }

    /// Forget the tool, selection and plan position, as for a newly opened level.
    #[cfg_attr(not(feature = "dev-mcp"), allow(dead_code))]
    pub(crate) fn reset(&mut self) {
        self.tools = Tools::default();
        self.sel = Sel::None;
        self.message.clear();
        self.plan_for = None;
    }

    pub(crate) fn levels_dir(&self) -> PathBuf {
        self.config.levels.clone().unwrap_or_else(|| PathBuf::from("levels"))
    }
}

fn compact(ctx: &egui::Context) {
    ctx.set_visuals(egui::Visuals::dark());
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = egui::vec2(6.0, 3.0);
        s.spacing.button_padding = egui::vec2(5.0, 1.0);
        s.spacing.interact_size = egui::vec2(36.0, 18.0);
        s.spacing.window_margin = egui::Margin::same(6);
        s.spacing.menu_margin = egui::Margin::same(4);
        s.spacing.indent = 12.0;
        for (style, size) in [
            (egui::TextStyle::Body, 12.0),
            (egui::TextStyle::Button, 12.0),
            (egui::TextStyle::Small, 10.0),
            (egui::TextStyle::Monospace, 12.0),
            (egui::TextStyle::Heading, 14.0),
        ] {
            if let Some(f) = s.text_styles.get_mut(&style) {
                f.size = size;
            }
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub fn draw(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<Ui>,
    mut doc: ResMut<Document>,
    mut built: ResMut<Built>,
    mut input: ResMut<ViewInput>,
    mut cam: ResMut<EditorCam>,
    time: Res<Time>,
    mut window: Query<&mut Window>,
    atlas: Res<crate::view3d::Atlas>,
    sculpts: Res<crate::flesh::Sculpts>,
    mut exit: MessageWriter<AppExit>,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    let atlas_id = contexts.image_id(&atlas.0);
    let s = &mut *ui_state;
    if !s.styled {
        compact(&ctx);
        built.mode = s.settings.mode;
        built.full_colour = s.settings.full_colour;
        s.styled = true;
    }
    let level_centre_tiles = |l: &Level| {
        let w = to_world(level_centre(l));
        egui::vec2(w[0] / TF, w[2] / TF)
    };
    let empty = doc.0.level.columns.is_empty();
    match &s.plan_for {
        Some((path, _)) if *path != doc.0.path => s.plan_for = None,
        // Built on from empty (by an agent, say): bring it into view, unless it is
        // in view already, as a room just drawn in the plan is.
        Some((_, true)) if !empty && !s.tools.busy() => {
            let c = level_centre_tiles(&doc.0.level);
            let shown = s.plan.rect.is_some_and(|r| r.shrink(8.0).contains(r.center() + (c - s.plan.centre) * s.plan.zoom));
            if !shown {
                s.plan.centre = c;
            }
            s.plan_for = Some((doc.0.path.clone(), false));
        }
        _ => {}
    }
    if s.plan_for.is_none() {
        s.plan.centre = level_centre_tiles(&doc.0.level);
        s.plan_for = Some((doc.0.path.clone(), empty));
    }
    s.fps = s.fps * 0.95 + 0.05 / time.delta_secs().max(1e-4);
    let doc = &mut doc.0;
    s.panels.clear();
    s.plan.rect = None;
    s.screen = Some(ctx.viewport_rect());
    built.brushing = s.tools.tool == Tool::Strange;

    shortcuts(&ctx, s, doc, &mut built, &mut input, &cam);
    poll_dialogs(s, doc);

    let mut root = egui::Ui::new(
        ctx.clone(),
        "root".into(),
        egui::UiBuilder::new().layer_id(egui::LayerId::background()).max_rect(ctx.viewport_rect()),
    );
    egui::Panel::top("menu").show(&mut root, |ui| menu_bar(ui, s, doc, &mut built, &mut input, &cam));
    egui::Panel::bottom("status").show(&mut root, |ui| status_bar(ui, s, doc, &built, &cam, sculpts.pending()));

    let mut dock = std::mem::replace(&mut s.settings.dock, DockState::new(Vec::new()));
    let mut style = egui_dock::Style::from_egui(root.style());
    style.dock_area_padding = None;
    style.main_surface_border_stroke = egui::Stroke::NONE;
    let mut viewer = Viewer { s, doc, built: &mut built, input: &mut input, cam: &mut cam, atlas: atlas_id, viewport_hovered: false };
    DockArea::new(&mut dock)
        .style(style)
        .show_leaf_collapse_buttons(false)
        .show_inside(&mut root, &mut viewer);
    let hovered = viewer.viewport_hovered;
    s.settings.dock = dock;
    // Floating panels sit over the viewport tab: the pointer is only the 3D view's
    // when no egui layer above the background wants it.
    input.hovered = hovered && !ctx.layer_id_at(ctx.pointer_hover_pos().unwrap_or_default()).is_some_and(|l| l.order != egui::Order::Background);
    input.typing = ctx.egui_wants_keyboard_input();

    crate::mcp::agents_window(&ctx, s);
    dialogs(&ctx, s, doc, &mut exit);
    doc.settle(!ctx.egui_is_using_pointer() && !ctx.egui_wants_keyboard_input() && !s.tools.busy());

    if let Ok(mut w) = window.single_mut() {
        let title = format!("{}{} — neurofield editor", doc.name(), if doc.dirty() { " •" } else { "" });
        if w.title != title {
            w.title = title;
        }
    }
    Ok(())
}

fn shortcuts(ctx: &egui::Context, s: &mut Ui, doc: &mut Doc, built: &mut Built, input: &mut ViewInput, cam: &EditorCam) {
    let cmd = |m: Modifiers, k: Key| ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(m, k)));
    let shift_cmd = Modifiers::COMMAND | Modifiers::SHIFT;
    if cmd(shift_cmd, Key::S) {
        save_as(s, None);
    } else if cmd(Modifiers::COMMAND, Key::S) {
        save(s, doc, None);
    }
    if cmd(Modifiers::COMMAND, Key::O) {
        request(s, doc, After::Open(None));
    }
    if cmd(Modifiers::COMMAND, Key::N) {
        request(s, doc, After::New);
    }
    if cmd(Modifiers::COMMAND, Key::Q) {
        s.quit_requested = true;
    }
    if cmd(shift_cmd, Key::Z) || cmd(Modifiers::COMMAND, Key::Y) {
        s.tools.cancel(doc);
        doc.redo();
    } else if cmd(Modifiers::COMMAND, Key::Z) {
        s.tools.cancel(doc);
        doc.undo();
    }
    if cmd(Modifiers::SHIFT, Key::F5) {
        play(s, doc, Some(cam));
    } else if cmd(Modifiers::NONE, Key::F5) {
        play(s, doc, None);
    }
    if ctx.egui_wants_keyboard_input() {
        return;
    }
    // Plain keys; while flying (right mouse held) WASD and friends belong to the camera.
    if ctx.input(|i| i.pointer.secondary_down()) {
        return;
    }
    let pressed = |k: Key| ctx.input(|i| i.key_pressed(k) && i.modifiers.is_none());
    for t in Tool::ALL {
        if pressed(t.key()) && t.coming().is_none() {
            s.tools.set_tool(t, doc);
        }
    }
    if pressed(Key::PageUp) {
        built.slice += 1;
    }
    if pressed(Key::PageDown) {
        built.slice -= 1;
    }
    if pressed(Key::Delete) || pressed(Key::Backspace) {
        delete_selected(s, doc);
    }
    if pressed(Key::Escape) && !s.tools.cancel(doc) {
        s.sel = Sel::None;
        s.tools.faces.clear();
    }
    if pressed(Key::F) {
        input.frame_request = Some(frame_target(s, doc));
    }
    if pressed(Key::Home) {
        input.frame_request = Some(level_centre(&doc.level));
    }
    // Tab is the editor's own key here, not egui's focus to the next widget (which
    // would take the next Tab into the Entity tool's facing, say).
    if pressed(Key::Tab) {
        ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
    }
    if pressed(Key::Tab) && s.tools.tool == Tool::Place {
        let quarter = std::f32::consts::FRAC_PI_2;
        s.tools.yaw = ((s.tools.yaw / quarter).round() + 1.0).rem_euclid(4.0) * quarter;
    } else if pressed(Key::Tab) && s.tools.tool == Tool::Slope {
        s.tools.kind = match s.tools.kind {
            SlopeKind::Stairs => SlopeKind::Ramp,
            SlopeKind::Ramp => SlopeKind::Stairs,
        };
    } else if pressed(Key::Tab) {
        built.mode = if built.mode == ViewMode::Plain { ViewMode::Psx } else { ViewMode::Plain };
        s.settings.mode = built.mode;
    }
}

pub fn level_centre(l: &Level) -> Vec3 {
    if l.columns.is_empty() {
        return Vec3::ZERO;
    }
    let n = l.columns.len() as f32;
    let (sx, sz) = l.columns.keys().fold((0.0, 0.0), |(a, b), k| (a + k.0 as f32 + 0.5, b + k.1 as f32 + 0.5));
    let y = l.columns.values().flatten().map(|r| r.from).min().unwrap_or(0) as f32 * STEP as f32 / TF;
    to_view([sx / n * TF, -(y + 0.5) * TF, sz / n * TF])
}

fn frame_target(s: &Ui, doc: &Doc) -> Vec3 {
    let l = &doc.level;
    let at = match s.sel {
        Sel::Entity(id) => l.entities.iter().find(|e| e.id == id).map(|e| to_view(e.pos) + Vec3::Y * 0.5),
        Sel::Light(i) => l.lights.get(i).map(|light| to_view(light.pos)),
        Sel::Area(i) => l.areas.get(i).and_then(|a| a.boxes.first()).map(|(lo, hi)| {
            let mid = |i: usize| (lo[i] + hi[i]) as f32 / 2.0;
            Vec3::new(mid(0), lo[1] as f32 * STEP as f32 / TF + 0.5, -mid(2))
        }),
        Sel::None => None,
    };
    at.unwrap_or_else(|| level_centre(l))
}

fn delete_selected(s: &mut Ui, doc: &mut Doc) {
    let l = &mut doc.level;
    match std::mem::take(&mut s.sel) {
        Sel::Entity(id) => l.entities.retain(|e| e.id != id),
        Sel::Light(i) if i < l.lights.len() => _ = l.lights.remove(i),
        Sel::Area(i) if i < l.areas.len() => _ = l.areas.remove(i),
        _ => {}
    }
}

fn play(s: &mut Ui, doc: &Doc, from: Option<&EditorCam>) {
    let start = from.map(|c| c.player_start());
    s.message = match crate::play(&doc.level, start) {
        Ok(()) => if start.is_some() { "playing from the camera".into() } else { "playing".into() },
        Err(e) => e,
    };
}

/// New or Open, asking about unsaved changes first.
fn request(s: &mut Ui, doc: &Doc, what: After) {
    if doc.dirty() {
        s.confirm = Some(what);
    } else {
        proceed(s, what);
    }
}

fn proceed(s: &mut Ui, what: After) {
    match what {
        After::Open(None) => {
            let dir = s.levels_dir();
            s.open_dialog = Some(AsyncComputeTaskPool::get().spawn(async move {
                rfd::AsyncFileDialog::new().add_filter("level", &["ron"]).set_directory(dir).pick_file().await.map(|f| f.path().to_path_buf())
            }));
        }
        other => s.confirm_done = Some(other),
    }
}

fn save(s: &mut Ui, doc: &mut Doc, then: Option<After>) {
    match doc.path.clone() {
        Some(p) => match doc.save(&p) {
            Ok(()) => {
                s.message = format!("saved {}", p.display());
                s.settings.opened(&p);
                if let Some(t) = then {
                    proceed(s, t);
                }
            }
            Err(e) => s.message = e,
        },
        None => save_as(s, then),
    }
}

fn save_as(s: &mut Ui, then: Option<After>) {
    let dir = s.levels_dir();
    s.after_save = then;
    s.save_dialog = Some(AsyncComputeTaskPool::get().spawn(async move {
        rfd::AsyncFileDialog::new()
            .add_filter("level", &["ron"])
            .set_directory(dir)
            .set_file_name("level.ron")
            .save_file()
            .await
            .map(|f| f.path().to_path_buf())
    }));
}

fn poll_dialogs(s: &mut Ui, doc: &mut Doc) {
    if let Some(task) = s.open_dialog.as_mut() {
        if let Some(picked) = block_on(poll_once(task)) {
            s.open_dialog = None;
            if let Some(p) = picked {
                s.confirm_done = Some(After::Open(Some(p)));
            }
        }
    }
    if let Some(task) = s.save_dialog.as_mut() {
        if let Some(picked) = block_on(poll_once(task)) {
            s.save_dialog = None;
            let then = s.after_save.take();
            if let Some(p) = picked {
                let p = if p.extension().is_none() { p.with_extension("ron") } else { p };
                doc.path = Some(p);
                save(s, doc, then);
            }
        }
    }
    match s.confirm_done.take() {
        Some(After::New) => {
            *doc = Doc::new(Level::new("untitled"), None);
            s.sel = Sel::None;
            s.tools.reset();
            s.message = "new level".into();
        }
        Some(After::Open(Some(p))) => match Doc::open(&p) {
            Ok(d) => {
                *doc = d;
                s.sel = Sel::None;
                s.tools.reset();
                s.settings.opened(&p);
                s.message = format!("opened {}", p.display());
            }
            Err(e) => s.message = e,
        },
        Some(After::Open(None)) => proceed(s, After::Open(None)),
        Some(After::Quit) => s.quit_now = true,
        None => {}
    }
}

/// The unsaved-changes question, and quitting.
fn dialogs(ctx: &egui::Context, s: &mut Ui, doc: &mut Doc, exit: &mut MessageWriter<AppExit>) {
    if s.quit_requested {
        s.quit_requested = false;
        if doc.dirty() {
            s.confirm = Some(After::Quit);
        } else {
            s.quit_now = true;
        }
    }
    if let Some(what) = s.confirm.clone() {
        let mut answer = None;
        egui::Modal::new("unsaved".into()).show(ctx, |ui| {
            ui.heading("Unsaved changes");
            ui.label(format!("{} has changes that aren't saved.", doc.name()));
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    answer = Some(0);
                }
                if ui.button("Discard").clicked() {
                    answer = Some(1);
                }
                if ui.button("Cancel").clicked() {
                    answer = Some(2);
                }
            });
        });
        match answer {
            Some(0) => {
                s.confirm = None;
                save(s, doc, Some(what));
            }
            Some(1) => {
                s.confirm = None;
                proceed(s, what);
            }
            Some(_) => s.confirm = None,
            None => {}
        }
    }
    if s.quit_now {
        s.settings.save();
        exit.write(AppExit::Success);
    }
}

fn menu_bar(ui: &mut egui::Ui, s: &mut Ui, doc: &mut Doc, built: &mut Built, input: &mut ViewInput, cam: &EditorCam) {
    egui::MenuBar::new().ui(ui, |ui| {
        ui.menu_button("File", |ui| {
            if ui.add(egui::Button::new("New").shortcut_text("Ctrl+N")).clicked() {
                request(s, doc, After::New);
            }
            if ui.add(egui::Button::new("Open…").shortcut_text("Ctrl+O")).clicked() {
                request(s, doc, After::Open(None));
            }
            ui.menu_button("Open recent", |ui| {
                if s.settings.recent.is_empty() {
                    ui.label("nothing yet");
                }
                for p in s.settings.recent.clone() {
                    if ui.button(p.display().to_string()).clicked() {
                        request(s, doc, After::Open(Some(p)));
                    }
                }
            });
            ui.separator();
            if ui.add(egui::Button::new("Save").shortcut_text("Ctrl+S")).clicked() {
                save(s, doc, None);
            }
            if ui.add(egui::Button::new("Save as…").shortcut_text("Ctrl+Shift+S")).clicked() {
                save_as(s, None);
            }
            if ui.add_enabled(doc.dirty() && doc.path.is_some(), egui::Button::new("Revert")).clicked() {
                if let Some(p) = doc.path.clone() {
                    s.confirm_done = Some(After::Open(Some(p)));
                }
            }
            ui.separator();
            if ui.add(egui::Button::new("Quit").shortcut_text("Ctrl+Q")).clicked() {
                s.quit_requested = true;
            }
        });
        ui.menu_button("Edit", |ui| {
            if ui.add_enabled(doc.can_undo(), egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                s.tools.cancel(doc);
                doc.undo();
            }
            if ui.add_enabled(doc.can_redo(), egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z")).clicked() {
                s.tools.cancel(doc);
                doc.redo();
            }
            ui.separator();
            if ui.add_enabled(s.sel != Sel::None, egui::Button::new("Delete").shortcut_text("Del")).clicked() {
                delete_selected(s, doc);
            }
            if ui.add(egui::Button::new("Select none").shortcut_text("Esc")).clicked() {
                s.sel = Sel::None;
                s.tools.faces.clear();
            }
        });
        ui.menu_button("View", |ui| {
            for t in Tab::PANELS {
                let open = s.settings.dock.find_tab(&t).is_some();
                if ui.selectable_label(open, t.title()).clicked() {
                    match s.settings.dock.find_tab(&t) {
                        Some(path) => {
                            s.settings.dock.remove_tab(path);
                        }
                        None => s.settings.dock.push_to_focused_leaf(t),
                    }
                }
            }
            if ui.button("Reset layout").clicked() {
                s.settings.dock = default_dock();
            }
            ui.separator();
            ui.label("3D view (Tab)");
            ui.radio_value(&mut built.mode, ViewMode::Plain, "Plain");
            ui.radio_value(&mut built.mode, ViewMode::Psx, "PS1");
            s.settings.mode = built.mode;
            ui.checkbox(&mut built.show_grid, "Grid");
            ui.checkbox(&mut built.cut, "Cut above the slice");
            ui.checkbox(&mut built.show_entities, "Entity markers");
            ui.checkbox(&mut built.overlay, "Strangeness overlay").on_hover_text("Tint the 3D view and the plan by strangeness (always on with the Strangeness tool)");
            ui.checkbox(&mut built.full_colour, "Full-colour creatures");
            s.settings.full_colour = built.full_colour;
            ui.separator();
            if ui.add(egui::Button::new("Frame selection").shortcut_text("F")).clicked() {
                input.frame_request = Some(frame_target(s, doc));
            }
            if ui.add(egui::Button::new("Frame level").shortcut_text("Home")).clicked() {
                input.frame_request = Some(level_centre(&doc.level));
            }
            if ui.add(egui::Button::new("Slice up").shortcut_text("PgUp")).clicked() {
                built.slice += 1;
            }
            if ui.add(egui::Button::new("Slice down").shortcut_text("PgDn")).clicked() {
                built.slice -= 1;
            }
        });
        ui.menu_button("Tools", |ui| {
            for t in Tool::ALL {
                let label = format!("{}  ({:?})", t.name(), t.key());
                if ui.add_enabled(t.coming().is_none(), egui::RadioButton::new(s.tools.tool == t, label)).clicked() {
                    s.tools.set_tool(t, doc);
                }
            }
            ui.separator();
            if ui.button("Agents…").on_hover_text("Let AI agents connect and help build the level").clicked() {
                s.agents.window = true;
            }
        });
        ui.menu_button("Play", |ui| {
            if ui.add(egui::Button::new("Play").shortcut_text("F5")).clicked() {
                play(s, doc, None);
            }
            if ui.add(egui::Button::new("Play from here").shortcut_text("Shift+F5")).clicked() {
                play(s, doc, Some(cam));
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("▶ From here").on_hover_text("Play from the camera (Shift+F5)").clicked() {
                play(s, doc, Some(cam));
            }
            if ui.button("▶ Play").on_hover_text("Play from the player start (F5)").clicked() {
                play(s, doc, None);
            }
            if s.agents.listening.is_some() {
                let (text, hint) = match s.agents.active() {
                    true => (RichText::new("• Agent at work").color(Color32::from_rgb(120, 220, 120)), "An agent is working on the level"),
                    false => (RichText::new("• Agents on"), "Agents can connect (Tools > Agents)"),
                };
                if ui.add(egui::Button::new(text).frame(false)).on_hover_text(hint).clicked() {
                    s.agents.window = true;
                }
            }
        });
    });
}

fn status_bar(ui: &mut egui::Ui, s: &Ui, doc: &Doc, built: &Built, cam: &EditorCam, sculpting: usize) {
    ui.horizontal(|ui| {
        let t = s.tools.tool;
        let hint = match (s.tools.link, t.coming()) {
            (Some(Link::Key(_)), _) => "Link: click the key it opens with, in the 3D view or the plan (Esc cancels)".into(),
            (Some(Link::Target(_)), _) => "Link: click a door for the lever to work (Esc cancels)".into(),
            (None, None) => format!("{}: {}", t.name(), t.hint()),
            (None, Some(m)) => format!("{}: arrives in milestone {m}", t.name()),
        };
        ui.label(hint);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(format!("{:.0} fps", s.fps));
            ui.separator();
            ui.label(format!("{} tris", built.scene.tris.len()));
            ui.separator();
            if !s.tools.faces.is_empty() {
                ui.label(format!("{} face{}", s.tools.faces.len(), if s.tools.faces.len() == 1 { "" } else { "s" }));
                ui.separator();
            }
            if let Some(h) = s.tools.hover {
                let [x, l, z] = h.cell;
                let face = match h.face {
                    engine::world::level::Face::Wall(d) => format!("{d:?} wall"),
                    f => format!("{f:?}").to_lowercase(),
                };
                let tex = crate::tools::face_tex(&doc.level, h.cell, h.face);
                let strange = built.whole().strangeness(to_world(h.point));
                ui.label(format!("cell ({x}, {l}, {z}) {face}, texture {tex}, strangeness {strange:.2}"));
                ui.separator();
            }
            if sculpting > 0 {
                ui.label(format!("sculpting {sculpting} creature{}", if sculpting == 1 { "" } else { "s" }));
                ui.separator();
            }
            let w = to_world(cam.pos);
            ui.label(format!(
                "camera ({}, {}, {})  slice {}",
                (w[0] / TF).floor(),
                (-w[1] / STEP as f32).floor(),
                (w[2] / TF).floor(),
                built.slice
            ));
            ui.separator();
            let name = format!("{}{}", doc.name(), if doc.dirty() { " •" } else { "" });
            ui.label(RichText::new(name).strong());
            if !s.message.is_empty() {
                ui.separator();
                ui.label(RichText::new(&s.message).weak());
            }
        });
    });
}

struct Viewer<'a> {
    s: &'a mut Ui,
    doc: &'a mut Doc,
    built: &'a mut Built,
    input: &'a mut ViewInput,
    cam: &'a mut EditorCam,
    atlas: Option<egui::TextureId>,
    viewport_hovered: bool,
}

impl TabViewer for Viewer<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(tab.title())
    }

    fn is_closeable(&self, tab: &Tab) -> bool {
        *tab != Tab::Viewport
    }

    fn clear_background(&self, tab: &Tab) -> bool {
        *tab != Tab::Viewport
    }

    fn scroll_bars(&self, tab: &Tab) -> [bool; 2] {
        match tab {
            Tab::Viewport | Tab::Plan => [false, false],
            _ => [true, true],
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        self.s.panels.push((*tab, ui.max_rect()));
        match tab {
            Tab::Viewport => self.viewport(ui),
            Tab::Plan => self.plan(ui),
            Tab::Tools => self.tools(ui),
            Tab::Properties => self.properties(ui),
            Tab::Entities => self.entities(ui),
            Tab::Textures => self.textures(ui),
            Tab::Areas => self.areas(ui),
            Tab::Issues => self.issues(ui),
        }
    }
}

impl Viewer<'_> {
    fn viewport(&mut self, ui: &mut egui::Ui) {
        let r = ui.allocate_rect(ui.max_rect(), egui::Sense::hover());
        self.viewport_hovered = r.hovered() || r.contains_pointer();
        // A strip of view toggles in the corner.
        let mut corner = ui.new_child(egui::UiBuilder::new().max_rect(r.rect.shrink(6.0)));
        let toggles = corner.horizontal(|ui| {
            let b = &mut *self.built;
            ui.selectable_value(&mut b.mode, ViewMode::Plain, "Plain");
            ui.selectable_value(&mut b.mode, ViewMode::Psx, "PS1");
            self.s.settings.mode = b.mode;
            ui.separator();
            ui.toggle_value(&mut b.show_grid, "Grid");
            ui.toggle_value(&mut b.cut, "Cut").on_hover_text("Hide everything more than a tile above the slice (PgUp/PgDn)");
            ui.toggle_value(&mut b.show_entities, "Markers");
            ui.add_enabled_ui(!b.brushing, |ui| ui.toggle_value(&mut b.overlay, "Strange"))
                .response
                .on_hover_text("Tint by strangeness: blue grounded, through teal and yellow, to magenta (always on with the Strangeness tool)");
            ui.toggle_value(&mut b.full_colour, "Flesh")
                .on_hover_text("Creatures in full colour, as the game draws them (sculpted in the background)");
            self.s.settings.full_colour = b.full_colour;
        });
        // The tools get the pointer where no panel (or the toggles) has it.
        let ctx = ui.ctx().clone();
        let pos = ctx.pointer_latest_pos();
        let over = self.viewport_hovered
            && pos.is_some_and(|p| {
                !toggles.response.rect.contains(p) && !ctx.layer_id_at(p).is_some_and(|l| l.order != egui::Order::Background)
            });
        let s = &mut *self.s;
        s.tools.viewport(&ctx, ctx.viewport_rect(), over, self.doc, self.built, self.cam, &mut s.sel, &mut s.message, self.input);
    }

    fn tools(&mut self, ui: &mut egui::Ui) {
        for t in Tool::ALL {
            let text = format!("{}  {:?}", t.name(), t.key());
            let r = ui.add_enabled(t.coming().is_none(), egui::Button::selectable(self.s.tools.tool == t, text));
            let r = match t.coming() {
                Some(m) => r.on_disabled_hover_text(format!("{}; arrives in milestone {m}", t.hint())),
                None => r.on_hover_text(t.hint()),
            };
            if r.clicked() {
                self.s.tools.set_tool(t, self.doc);
            }
        }
        ui.separator();
        self.s.tools.options(ui);
    }

    fn properties(&mut self, ui: &mut egui::Ui) {
        let l = &mut self.doc.level;
        let mut delete = false;
        match self.s.sel {
            Sel::Entity(id) if l.entities.iter().any(|e| e.id == id) => {
                let named = |want: fn(&EntityKind) -> bool| -> Vec<(u32, String)> {
                    l.entities.iter().filter(|e| want(&e.kind)).map(|e| (e.id, entity_label(e))).collect()
                };
                let keys = named(|k| matches!(k, EntityKind::Key { .. }));
                let doors = named(|k| matches!(k, EntityKind::Door { .. }));
                let e = l.entities.iter_mut().find(|e| e.id == id).unwrap();
                entity_properties(ui, e, &keys, &doors, &mut self.s.tools.link, &mut delete);
            }
            Sel::Light(i) if i < l.lights.len() => light_properties(ui, i, &mut l.lights[i], &mut delete),
            Sel::Area(i) if i < l.areas.len() => {
                let default = l.surface;
                area_properties(ui, &mut l.areas[i], default, &mut delete);
            }
            _ => level_properties(ui, l),
        }
        if delete {
            delete_selected(self.s, self.doc);
        }
    }

    fn entities(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.menu_button("Add...", |ui| {
                for (name, kind) in crate::tools::entity_kinds() {
                    if ui.button(name).clicked() {
                        let at = self.focus_on_floor();
                        let id = crate::tools::add_entity(&mut self.doc.level, kind, at, 0.0);
                        self.s.sel = Sel::Entity(id);
                        ui.close();
                    }
                }
            })
            .response
            .on_hover_text("at the point the camera is looking at (or place them with the Entity tool, E)");
            ui.add(egui::TextEdit::singleline(&mut self.s.filter).hint_text("filter").desired_width(f32::INFINITY));
        });
        let filter = self.s.filter.to_lowercase();
        let shown = |label: &str| filter.is_empty() || label.to_lowercase().contains(&filter);
        ui.separator();
        let kinds = ["PlayerStart", "Creature", "Door", "Key", "Chest", "Prop", "Lever"];
        for k in kinds {
            let items: Vec<(u32, String, Color32)> = self
                .doc
                .level
                .entities
                .iter()
                .filter(|e| kind_name(&e.kind) == k)
                .map(|e| (e.id, entity_label(e), egui_colour(entity_colour(&e.kind))))
                .filter(|(_, label, _)| shown(label) || shown(k))
                .collect();
            if items.is_empty() {
                continue;
            }
            egui::CollapsingHeader::new(format!("{k} ({})", items.len())).default_open(true).show(ui, |ui| {
                for (id, label, colour) in items {
                    let r = ui.selectable_label(self.s.sel == Sel::Entity(id), RichText::new(label).color(colour));
                    if r.clicked() {
                        self.s.sel = Sel::Entity(id);
                    }
                    if r.double_clicked() {
                        self.s.sel = Sel::Entity(id);
                        self.input.frame_request = Some(frame_target(self.s, self.doc));
                    }
                }
            });
        }
        let lights: Vec<String> = self
            .doc
            .level
            .lights
            .iter()
            .enumerate()
            .map(|(i, l)| format!("light {i} ({:.1}, {:.1}, {:.1})", l.pos[0] / TF, up(l.pos[1]), l.pos[2] / TF))
            .filter(|label| shown(label))
            .collect();
        if !lights.is_empty() {
            egui::CollapsingHeader::new(format!("Light ({})", lights.len())).default_open(true).show(ui, |ui| {
                for (i, label) in lights.into_iter().enumerate() {
                    let r = ui.selectable_label(self.s.sel == Sel::Light(i), label);
                    if r.clicked() || r.double_clicked() {
                        self.s.sel = Sel::Light(i);
                    }
                    if r.double_clicked() {
                        self.input.frame_request = Some(frame_target(self.s, self.doc));
                    }
                }
            });
        }
    }

    /// The floor point under what the camera looks at.
    fn focus_on_floor(&self) -> [f32; 3] {
        let p = to_world(self.cam.pos + self.cam.forward() * self.cam.focus);
        let floor = self.built.scene.floor_y(p[0], p[1], p[2]).unwrap_or(p[1]);
        [p[0], floor, p[2]]
    }

    fn textures(&mut self, ui: &mut egui::Ui) {
        let Some(atlas) = self.atlas else { return };
        ui.label(format!("level set: {}", self.doc.level.textures));
        ui.horizontal_wrapped(|ui| {
            for slot in 0..8u8 {
                let (u0, v0) = engine::world::tex::uv_origin(slot);
                let (w, h) = (256.0, 64.0 * 4.0);
                let uv = egui::Rect::from_min_size(egui::pos2(u0 as f32 / w, v0 as f32 / h), egui::vec2(64.0 / w, 64.0 / h));
                ui.vertical(|ui| {
                    let on = self.s.tools.slot == slot;
                    let image = egui::Image::new(egui::load::SizedTexture::new(atlas, egui::vec2(56.0, 56.0))).uv(uv);
                    let r = ui.add(egui::Button::image(image).selected(on)).on_hover_text("Paint with this (B)");
                    if r.clicked() {
                        self.s.tools.slot = slot;
                        self.s.tools.set_tool(Tool::Paint, self.doc);
                    }
                    ui.label(RichText::new(format!("{slot}")).strong().color(if on { Color32::WHITE } else { Color32::GRAY }));
                });
            }
        });
    }

    fn areas(&mut self, ui: &mut egui::Ui) {
        let faces = &self.s.tools.faces;
        let r = ui
            .add_enabled(!faces.is_empty(), egui::Button::new("New area from the selected faces"))
            .on_hover_text("The box around the selected faces' cells, floor to ceiling")
            .on_disabled_hover_text("Select faces first (Select tool), or ctrl-click to finish a Room");
        if r.clicked() {
            if let Some(b) = crate::tools::cells_box(&self.doc.level, faces) {
                let l = &mut self.doc.level;
                let name = format!("area {}", l.areas.len() + 1);
                l.areas.push(engine::world::level::Area { name, label: "room".into(), boxes: vec![b], surface: None, notes: String::new(), strangeness: None });
                self.s.sel = Sel::Area(l.areas.len() - 1);
            }
        }
        ui.separator();
        if self.doc.level.areas.is_empty() {
            ui.label(RichText::new("No areas yet. Ctrl-click to finish a Room, or select faces and make one.").weak());
        }
        for (i, a) in self.doc.level.areas.iter().enumerate() {
            let text = format!("{}  ·  {}", a.name, a.label);
            let r = ui.selectable_label(self.s.sel == Sel::Area(i), text);
            if r.clicked() || r.double_clicked() {
                self.s.sel = Sel::Area(i);
            }
            if r.double_clicked() {
                self.input.frame_request = Some(frame_target(self.s, self.doc));
            }
        }
    }

    fn issues(&mut self, ui: &mut egui::Ui) {
        let issues = validate(&self.doc.level, self.built.whole());
        if issues.is_empty() {
            ui.label(RichText::new("No problems found.").color(Color32::LIGHT_GREEN));
        }
        for (text, at) in issues {
            let r = ui.selectable_label(false, format!("! {text}"));
            if let (true, Some(p)) = (r.clicked(), at) {
                self.input.frame_request = Some(p);
            }
        }
    }

    fn plan(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("slice");
            ui.add(egui::DragValue::new(&mut self.built.slice).speed(0.1));
            ui.label(RichText::new("layer (PgUp/PgDn); drag to pan, wheel to zoom, double-click to fly there").weak());
        });
        let (rect, r) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let pv = &mut self.s.plan;
        pv.rect = Some(rect);
        let edits = matches!(self.s.tools.tool, Tool::Room | Tool::Carve | Tool::Slope | Tool::Place | Tool::Strange);
        if r.dragged_by(egui::PointerButton::Secondary)
            || r.dragged_by(egui::PointerButton::Middle)
            || (!edits && r.dragged_by(egui::PointerButton::Primary))
        {
            pv.centre -= r.drag_delta() / pv.zoom;
        }
        if r.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            pv.zoom = (pv.zoom * (1.0 + scroll * 0.002)).clamp(4.0, 80.0);
        }
        let to_screen = |x: f32, z: f32| rect.center() + (egui::vec2(x, z) - pv.centre) * pv.zoom;
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_rgb(18, 18, 22));
        let slice = self.built.slice;
        for (&(x, z), runs) in &self.doc.level.columns {
            let cell = egui::Rect::from_min_max(to_screen(x as f32, z as f32), to_screen(x as f32 + 1.0, z as f32 + 1.0));
            if !rect.intersects(cell) {
                continue;
            }
            let here = runs.iter().find(|r| slice >= r.from && slice < r.to);
            let colour = match here {
                // Brighter the nearer its floor is to the slice; slopes tinted.
                Some(r) => {
                    let depth = (slice - r.from).clamp(0, 12) as u8;
                    let base = 190 - depth * 9;
                    if r.slope.is_some() {
                        Color32::from_rgb(base, base - 30, 90)
                    } else if r.sky {
                        Color32::from_rgb(110, base - 40, base)
                    } else {
                        Color32::from_rgb(base, base, base - 15)
                    }
                }
                None if runs.iter().any(|r| r.to <= slice) => Color32::from_rgb(45, 45, 55),
                None => Color32::from_rgb(30, 30, 36),
            };
            // The overlay colours open cells by strangeness a little above their floor.
            let colour = match here {
                Some(r) if self.built.overlay_on() => {
                    let y = -((r.from.max(slice - 3) * STEP) as f32) - TF * 0.5;
                    let h = crate::view3d::heat(self.built.whole().strangeness([(x as f32 + 0.5) * TF, y, (z as f32 + 0.5) * TF]));
                    let k = colour.r() as f32 / 190.0;
                    Color32::from_rgb((h[0] * 255.0 * k) as u8, (h[1] * 255.0 * k) as u8, (h[2] * 255.0 * k) as u8)
                }
                _ => colour,
            };
            painter.rect_filled(cell.shrink(0.5), 0.0, colour);
        }
        for (a, b, c) in links(&self.doc.level) {
            let (a, b) = (to_screen(a[0] / TF, a[2] / TF), to_screen(b[0] / TF, b[2] / TF));
            painter.add(egui::Shape::dashed_line(&[a, b], egui::Stroke::new(1.0, egui_colour(c)), 4.0, 3.0));
        }
        for e in &self.doc.level.entities {
            let p = to_screen(e.pos[0] / TF, e.pos[2] / TF);
            let c = egui_colour(entity_colour(&e.kind));
            if let EntityKind::Door { width, .. } = e.kind {
                let along = egui::vec2(e.yaw.cos(), e.yaw.sin()) * width as f32 * 0.5 * pv.zoom;
                painter.line_segment([p - along, p + along], egui::Stroke::new(3.0, c));
            }
            let rad = if self.s.sel == Sel::Entity(e.id) { 5.0 } else { 3.5 };
            painter.circle_filled(p, rad, c);
            if self.s.sel == Sel::Entity(e.id) {
                painter.circle_stroke(p, rad + 2.0, egui::Stroke::new(1.0, Color32::WHITE));
            }
        }
        for (i, l) in self.doc.level.lights.iter().enumerate() {
            let p = to_screen(l.pos[0] / TF, l.pos[2] / TF);
            let c = Color32::from_rgb((l.colour[0].min(1.0) * 255.0) as u8, (l.colour[1].min(1.0) * 255.0) as u8, (l.colour[2].min(1.0) * 255.0) as u8);
            painter.circle_stroke(p, l.radius / TF * pv.zoom, egui::Stroke::new(1.0, c.gamma_multiply(0.35)));
            painter.circle_filled(p, 3.0, c);
            if self.s.sel == Sel::Light(i) {
                painter.circle_stroke(p, 5.0, egui::Stroke::new(1.0, Color32::WHITE));
            }
        }
        if let Sel::Area(i) = self.s.sel {
            for (lo, hi) in self.doc.level.areas.get(i).map_or(&[][..], |a| &a.boxes) {
                let r = egui::Rect::from_min_max(to_screen(lo[0] as f32, lo[2] as f32), to_screen(hi[0] as f32, hi[2] as f32));
                painter.rect_stroke(r, 0.0, egui::Stroke::new(1.5, Color32::from_rgb(80, 215, 255)), egui::StrokeKind::Middle);
            }
        }
        // The tool, at the tile under the pointer.
        let (centre, zoom) = (pv.centre, pv.zoom);
        let to_screen = move |x: f32, z: f32| rect.center() + (egui::vec2(x, z) - centre) * zoom;
        let tile = r.hover_pos().or(r.interact_pointer_pos()).map(|p| (p - rect.center()) / zoom + centre);
        let mods = ui.input(|i| i.modifiers);
        let s = &mut *self.s;
        if let Some((lo, hi)) = s.tools.plan(&r, mods, tile, slice, self.doc, &self.built.scene, &mut s.sel, &mut s.message) {
            let to_screen = |x: f32, z: f32| rect.center() + (egui::vec2(x, z) - centre) * zoom;
            let area = egui::Rect::from_min_max(to_screen(lo.0 as f32, lo.1 as f32), to_screen(hi.0 as f32 + 1.0, hi.1 as f32 + 1.0));
            painter.rect_stroke(area, 0.0, egui::Stroke::new(1.5, Color32::from_rgb(255, 140, 50)), egui::StrokeKind::Middle);
        }
        if self.s.tools.tool == Tool::Strange {
            if let Some(t) = tile.filter(|_| r.hovered() || self.s.tools.busy()) {
                let b = self.s.tools.brush;
                let c = Color32::from_rgb(255, 240, 130);
                painter.circle_stroke(to_screen(t.x, t.y), b.radius * zoom, egui::Stroke::new(1.5, c));
                let inner = b.radius * (1.0 - b.softness);
                if inner > 0.05 {
                    painter.circle_stroke(to_screen(t.x, t.y), inner * zoom, egui::Stroke::new(1.0, c.gamma_multiply(0.5)));
                }
            }
        }
        // The camera, and which way it looks.
        let c = to_world(self.cam.pos);
        let f = crate::view3d::dir_to_world(self.cam.forward());
        let p = to_screen(c[0] / TF, c[2] / TF);
        let d = egui::vec2(f[0], f[2]).normalized() * 12.0;
        painter.line_segment([p, p + d], egui::Stroke::new(2.0, Color32::from_rgb(120, 200, 255)));
        painter.circle_filled(p, 3.0, Color32::from_rgb(120, 200, 255));

        if let Some(pos) = r.interact_pointer_pos() {
            let tile = (pos - rect.center()) / zoom + centre;
            let linking = self.s.tools.link.is_some();
            if r.clicked() && (!edits || linking) {
                // The nearest entity or light within a few pixels.
                let l = &self.doc.level;
                let entities = l.entities.iter().map(|e| (Sel::Entity(e.id), e.pos));
                let lights = l.lights.iter().enumerate().map(|(i, light)| (Sel::Light(i), light.pos));
                let near = entities
                    .chain(lights)
                    .map(|(s, p)| (s, (to_screen(p[0] / TF, p[2] / TF) - pos).length()))
                    .filter(|&(_, d)| d < 8.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or(Sel::None, |(s, _)| s);
                match (self.s.tools.link.take(), near) {
                    (Some(link), Sel::Entity(id)) => self.s.message = crate::tools::make_link(&mut self.doc.level, link, id),
                    (Some(_), _) => self.s.message = "no entity there: link cancelled".into(),
                    (None, near) => self.s.sel = near,
                }
            }
            if r.double_clicked() {
                let y = slice as f32 * STEP as f32 / TF + 0.6;
                self.input.frame_request = Some(to_view([tile.x * TF, -y * TF, tile.y * TF]));
            }
        }
    }
}

fn egui_colour(c: Color) -> Color32 {
    let s = c.to_srgba();
    Color32::from_rgb((s.red * 255.0) as u8, (s.green * 255.0) as u8, (s.blue * 255.0) as u8)
}

pub(crate) fn kind_name(k: &EntityKind) -> &'static str {
    match k {
        EntityKind::PlayerStart => "PlayerStart",
        EntityKind::Creature { .. } => "Creature",
        EntityKind::Door { .. } => "Door",
        EntityKind::Key { .. } => "Key",
        EntityKind::Chest { .. } => "Chest",
        EntityKind::Prop { .. } => "Prop",
        EntityKind::Lever { .. } => "Lever",
    }
}

/// Height in tiles, up from world y (which points down), without a "-0".
fn up(y: f32) -> f32 {
    -y / TF + 0.0
}

fn entity_label(e: &LevelEntity) -> String {
    let at = format!("({:.1}, {:.1}, {:.1})", e.pos[0] / TF, up(e.pos[1]), e.pos[2] / TF);
    match &e.kind {
        EntityKind::Creature { kind, .. } => format!("#{} {kind:?} {at}", e.id),
        EntityKind::Key { name } => format!("#{} {name} {at}", e.id),
        EntityKind::Prop { shape, .. } => format!("#{} {shape:?} {at}", e.id),
        _ => format!("#{} {at}", e.id),
    }
}

fn colour_edit(ui: &mut egui::Ui, c: &mut [f32; 3]) {
    // Colours can go above 1 (bright lights); the picker shows them clamped. It
    // also rounds what it shows, so only take its colour when it is changed.
    let mut shown = c.map(|v| v.min(1.0));
    if ui.color_edit_button_rgb(&mut shown).changed() {
        *c = shown;
    }
    for v in c.iter_mut() {
        ui.add(egui::DragValue::new(v).speed(0.01).range(0.0..=4.0).max_decimals(2));
    }
}

/// The button that picks a link by clicking an entity in the view or the plan.
fn pick_button(ui: &mut egui::Ui, pending: &mut Option<Link>, link: Link) {
    let on = *pending == Some(link);
    let r = ui.selectable_label(on, "⌖").on_hover_text("Pick it: click it in the 3D view or the plan (Esc cancels)");
    if r.clicked() {
        *pending = if on { None } else { Some(link) };
    }
}

/// A lever's target doors: each with a button to drop it, and a way to add more.
fn targets_edit(ui: &mut egui::Ui, me: u32, targets: &mut Vec<u32>, doors: &[(u32, String)], pending: &mut Option<Link>) {
    ui.vertical(|ui| {
        let mut drop = None;
        for (i, t) in targets.iter().enumerate() {
            ui.horizontal(|ui| {
                let name = doors.iter().find(|d| d.0 == *t).map_or(format!("#{t} (not a door)"), |d| d.1.clone());
                ui.label(name);
                if ui.small_button("×").on_hover_text("Drop this target").clicked() {
                    drop = Some(i);
                }
            });
        }
        if let Some(i) = drop {
            targets.remove(i);
        }
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("add target").selected_text("add a door").show_ui(ui, |ui| {
                let mut add = None;
                for (id, name) in doors.iter().filter(|d| !targets.contains(&d.0)) {
                    if ui.selectable_label(false, name).clicked() {
                        add = Some(*id);
                    }
                }
                targets.extend(add);
            });
            pick_button(ui, pending, Link::Target(me));
        });
    });
}

/// The key a door or chest opens with: one of the level's keys, or none.
fn key_edit(ui: &mut egui::Ui, me: u32, key: &mut Option<u32>, keys: &[(u32, String)], pending: &mut Option<Link>) {
    let shown = match key {
        None => "no key".to_string(),
        Some(k) => keys.iter().find(|x| x.0 == *k).map_or(format!("#{k} (missing)"), |x| x.1.clone()),
    };
    // The pick button first, from the right, so a narrow panel shortens the combo
    // rather than pushing the button out of sight.
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        pick_button(ui, pending, Link::Key(me));
        // The width is the text's; the arrow and padding come on top.
        let width = (ui.available_width() - 30.0).max(40.0);
        egui::ComboBox::from_id_salt("key").selected_text(shown).width(width).truncate().show_ui(ui, |ui| {
            ui.selectable_value(key, None, "no key");
            for (id, name) in keys {
                ui.selectable_value(key, Some(*id), name);
            }
        });
    });
}

fn entity_properties(
    ui: &mut egui::Ui,
    e: &mut LevelEntity,
    keys: &[(u32, String)],
    doors: &[(u32, String)],
    link: &mut Option<Link>,
    delete: &mut bool,
) {
    let me = e.id;
    ui.horizontal(|ui| {
        ui.heading(format!("{} #{}", kind_name(&e.kind), e.id));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Delete").clicked() {
                *delete = true;
            }
        });
    });
    egui::Grid::new("entity").num_columns(2).show(ui, |ui| {
        ui.label("position (tiles)");
        ui.horizontal(|ui| {
            let mut t = [e.pos[0] / TF, up(e.pos[1]), e.pos[2] / TF];
            let mut changed = false;
            for (v, n) in t.iter_mut().zip(["x", "up", "z"]) {
                changed |= ui.add(egui::DragValue::new(v).speed(0.02).prefix(format!("{n} ")).max_decimals(2)).changed();
            }
            if changed {
                e.pos = [t[0] * TF, -t[1] * TF, t[2] * TF];
            }
        });
        ui.end_row();
        ui.label("facing (°)");
        let mut deg = e.yaw.to_degrees();
        if ui.add(egui::DragValue::new(&mut deg).speed(1.0).max_decimals(0)).changed() {
            e.yaw = deg.to_radians();
        }
        ui.end_row();
        match &mut e.kind {
            EntityKind::PlayerStart => {}
            EntityKind::Creature { kind, size, seed } => {
                ui.label("kind");
                egui::ComboBox::from_id_salt("kind").selected_text(format!("{kind:?}")).show_ui(ui, |ui| {
                    for k in [Kind::Humanoid, Kind::Beast, Kind::Blob, Kind::Floater, Kind::Mass] {
                        ui.selectable_value(kind, k, format!("{k:?}"));
                    }
                });
                ui.end_row();
                ui.label("size");
                ui.add(egui::Slider::new(size, 0.3..=2.5));
                ui.end_row();
                ui.label("seed");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(seed));
                    if ui.button("re-roll").clicked() {
                        *seed = engine::world::rng::hash(*seed ^ 0x9E37_79B9) as u64 % 100_000;
                    }
                });
                ui.end_row();
            }
            EntityKind::Door { width, height, hinge, open, locked, key } => {
                ui.label("width (tiles)");
                ui.add(egui::DragValue::new(width).range(1..=4));
                ui.end_row();
                ui.label("height (layers)");
                ui.add(egui::DragValue::new(height).range(4..=16));
                ui.end_row();
                ui.label("hinge");
                ui.horizontal(|ui| {
                    for h in [Hinge::Left, Hinge::Right, Hinge::Up] {
                        ui.selectable_value(hinge, h, format!("{h:?}"));
                    }
                });
                ui.end_row();
                ui.label("starts");
                ui.checkbox(open, "open");
                ui.end_row();
                ui.label("lock");
                ui.horizontal(|ui| {
                    ui.checkbox(locked, "locked");
                    key_edit(ui, me, key, keys, link);
                });
                ui.end_row();
            }
            EntityKind::Key { name } => {
                ui.label("name");
                ui.text_edit_singleline(name);
                ui.end_row();
            }
            EntityKind::Chest { locked, key, contents } => {
                ui.label("lock");
                ui.horizontal(|ui| {
                    ui.checkbox(locked, "locked");
                    key_edit(ui, me, key, keys, link);
                });
                ui.end_row();
                ui.label("contents");
                let mut text = contents.join(", ");
                if ui.text_edit_singleline(&mut text).changed() {
                    *contents = text.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
                }
                ui.end_row();
            }
            EntityKind::Prop { shape, size, tex, tint } => {
                ui.label("shape");
                egui::ComboBox::from_id_salt("shape").selected_text(format!("{shape:?}")).show_ui(ui, |ui| {
                    for p in [PropShape::Box, PropShape::Pillar, PropShape::Pyramid, PropShape::Floating] {
                        ui.selectable_value(shape, p, format!("{p:?}"));
                    }
                });
                ui.end_row();
                ui.label("size (units)");
                ui.horizontal(|ui| {
                    for v in size.iter_mut() {
                        ui.add(egui::DragValue::new(v).speed(1.0).range(4.0..=1024.0));
                    }
                });
                ui.end_row();
                ui.label("texture");
                ui.add(egui::DragValue::new(tex).range(0..=7));
                ui.end_row();
                ui.label("tint");
                ui.horizontal(|ui| colour_edit(ui, tint));
                ui.end_row();
            }
            EntityKind::Lever { targets } => {
                ui.label("targets");
                targets_edit(ui, me, targets, doors, link);
                ui.end_row();
            }
        }
    });
}

fn light_properties(ui: &mut egui::Ui, i: usize, light: &mut engine::world::level::PointLight, delete: &mut bool) {
    ui.horizontal(|ui| {
        ui.heading(format!("Light {i}"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Delete").clicked() {
                *delete = true;
            }
        });
    });
    egui::Grid::new("light").num_columns(2).show(ui, |ui| {
        ui.label("position (tiles)");
        ui.horizontal(|ui| {
            let mut t = [light.pos[0] / TF, up(light.pos[1]), light.pos[2] / TF];
            let mut changed = false;
            for (v, n) in t.iter_mut().zip(["x", "up", "z"]) {
                changed |= ui.add(egui::DragValue::new(v).speed(0.02).prefix(format!("{n} ")).max_decimals(2)).changed();
            }
            if changed {
                light.pos = [t[0] * TF, -t[1] * TF, t[2] * TF];
            }
        });
        ui.end_row();
        ui.label("colour");
        ui.horizontal(|ui| colour_edit(ui, &mut light.colour));
        ui.end_row();
        ui.label("radius (tiles)");
        let mut r = light.radius / TF;
        if ui.add(egui::DragValue::new(&mut r).speed(0.02).range(0.25..=16.0).max_decimals(2)).changed() {
            light.radius = r * TF;
        }
        ui.end_row();
        ui.label("when strange");
        ui.horizontal(|ui| {
            let mut own = light.strange.is_some();
            ui.checkbox(&mut own, "own colour").on_hover_text("Otherwise the level's seed picks one");
            if own != light.strange.is_some() {
                light.strange = own.then_some([0.6, 0.2, 0.9]);
            }
            if let Some(c) = &mut light.strange {
                colour_edit(ui, c);
            }
        });
        ui.end_row();
    });
}

fn area_properties(ui: &mut egui::Ui, a: &mut engine::world::level::Area, default: engine::world::level::Surface, delete: &mut bool) {
    ui.horizontal(|ui| {
        ui.heading("Area");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Delete").clicked() {
                *delete = true;
            }
        });
    });
    egui::Grid::new("area").num_columns(2).show(ui, |ui| {
        ui.label("name");
        ui.text_edit_singleline(&mut a.name);
        ui.end_row();
        ui.label("label");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut a.label).desired_width(90.0));
            egui::ComboBox::from_id_salt("labels").selected_text("").width(18.0).show_ui(ui, |ui| {
                for l in ["room", "corridor", "shrine", "stairwell", "outdoors"] {
                    if ui.selectable_label(a.label == l, l).clicked() {
                        a.label = l.into();
                    }
                }
            });
        });
        ui.end_row();
        ui.label("boxes");
        ui.vertical(|ui| {
            let mut remove = None;
            for (i, (lo, hi)) in a.boxes.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("({}, {}, {}) to ({}, {}, {})", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]));
                    if a.boxes.len() > 1 && ui.small_button("x").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                a.boxes.remove(i);
            }
        });
        ui.end_row();
    });
    ui.separator();
    let mut own = a.surface.is_some();
    ui.checkbox(&mut own, "Its own surfaces").on_hover_text("Floor, wall and ceiling textures and tint for unpainted faces inside it");
    if own != a.surface.is_some() {
        a.surface = own.then_some(default);
    }
    if let Some(s) = &mut a.surface {
        egui::Grid::new("area surface").num_columns(2).show(ui, |ui| {
            for (name, v) in [("floor", &mut s.floor), ("wall", &mut s.wall), ("ceiling", &mut s.ceiling)] {
                ui.label(name);
                ui.add(egui::DragValue::new(v).range(0..=7));
                ui.end_row();
            }
            ui.label("tint");
            ui.horizontal(|ui| colour_edit(ui, &mut s.tint));
            ui.end_row();
        });
    }
    ui.separator();
    ui.label("notes");
    ui.add(egui::TextEdit::multiline(&mut a.notes).desired_rows(3).desired_width(f32::INFINITY));
}

fn level_properties(ui: &mut egui::Ui, l: &mut Level) {
    ui.heading("Level");
    egui::Grid::new("level").num_columns(2).show(ui, |ui| {
        ui.label("name");
        ui.text_edit_singleline(&mut l.name);
        ui.end_row();
        ui.label("texture set");
        egui::ComboBox::from_id_salt("set").selected_text(&l.textures).show_ui(ui, |ui| {
            for t in ["grounded", "mid", "deep", "strange"] {
                ui.selectable_value(&mut l.textures, t.to_string(), t);
            }
        });
        ui.end_row();
        ui.label("ambient");
        ui.horizontal(|ui| colour_edit(ui, &mut l.ambient));
        ui.end_row();
        ui.label("fog (tiles)");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut l.fog.0).speed(0.05).prefix("near ").range(0.0..=40.0));
            ui.add(egui::DragValue::new(&mut l.fog.1).speed(0.05).prefix("far ").range(0.5..=60.0));
        });
        ui.end_row();
        ui.label("sun");
        let mut has = l.sun.is_some();
        ui.checkbox(&mut has, "outdoor light");
        if has != l.sun.is_some() {
            l.sun = has.then(|| Sun { dir: [0.4, 1.0, 0.3], colour: [0.8, 0.75, 0.65] });
        }
        ui.end_row();
        if let Some(sun) = &mut l.sun {
            ui.label("  direction");
            ui.horizontal(|ui| {
                for v in sun.dir.iter_mut() {
                    ui.add(egui::DragValue::new(v).speed(0.01).range(-1.0..=1.0));
                }
            });
            ui.end_row();
            ui.label("  colour");
            ui.horizontal(|ui| colour_edit(ui, &mut sun.colour));
            ui.end_row();
        }
    });
    ui.separator();
    ui.label(RichText::new("Default surfaces").strong());
    egui::Grid::new("surface").num_columns(2).show(ui, |ui| {
        let s = &mut l.surface;
        for (name, v) in [("floor", &mut s.floor), ("wall", &mut s.wall), ("ceiling", &mut s.ceiling)] {
            ui.label(name);
            ui.add(egui::DragValue::new(v).range(0..=7));
            ui.end_row();
        }
        ui.label("tint");
        ui.horizontal(|ui| colour_edit(ui, &mut s.tint));
        ui.end_row();
    });
    ui.separator();
    ui.label(RichText::new("Strangeness").strong());
    let st = &mut l.strangeness;
    egui::Grid::new("strange").num_columns(2).show(ui, |ui| {
        ui.label("noise");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut st.noise.0).speed(0.005).prefix("strength ").range(0.0..=1.0));
            ui.add(egui::DragValue::new(&mut st.noise.1).speed(0.01).prefix("scale ").range(0.05..=4.0));
        });
        ui.end_row();
        ui.label("fog");
        ui.horizontal(|ui| colour_edit(ui, &mut st.fog));
        ui.end_row();
        ui.label("ambient");
        ui.horizontal(|ui| colour_edit(ui, &mut st.ambient));
        ui.end_row();
        ui.label("seed");
        ui.add(egui::DragValue::new(&mut st.seed));
        ui.end_row();
    });
    ui.label("depth curve: strangeness by layer");
    let mut remove = None;
    for (i, p) in st.depth.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.0).speed(0.2).prefix("layer "));
            ui.add(egui::Slider::new(&mut p.1, 0.0..=1.0));
            if ui.small_button("x").clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        st.depth.remove(i);
    }
    if ui.small_button("+ point").clicked() {
        let last = st.depth.last().copied().unwrap_or((0.0, 0.0));
        st.depth.push((last.0 - 16.0, last.1));
    }
    st.depth.sort_by(|a, b| a.0.total_cmp(&b.0));
}

/// Problems worth fixing before playing, each with where to look.
fn validate(l: &Level, scene: &engine::world::scene::Scene) -> Vec<(String, Option<Vec3>)> {
    let mut out = Vec::new();
    let starts: Vec<_> = l.entities.iter().filter(|e| e.kind == EntityKind::PlayerStart).collect();
    match starts.len() {
        0 => out.push(("no player start".to_string(), None)),
        1 => {}
        n => out.push((format!("{n} player starts; Play uses the first"), Some(to_view(starts[1].pos)))),
    }
    if let Some(s) = starts.first() {
        match scene.node_at(s.pos[0], s.pos[1], s.pos[2]) {
            None => out.push(("the player start is inside rock".into(), Some(to_view(s.pos)))),
            Some(n) => {
                let reach: std::collections::HashSet<_> = scene.reachable(n).into_iter().map(|e| e.0).collect();
                let all = scene.floor_cells();
                let lost: Vec<_> = all.iter().filter(|c| !reach.contains(c)).collect();
                if !lost.is_empty() {
                    let c = lost[0];
                    let f = scene.node_floor(*c, [(c.0 as f32 + 0.5) * TF, (c.1 as f32 + 0.5) * TF]);
                    out.push((
                        format!("{} of {} floor cells can't be reached from the start", lost.len(), all.len()),
                        Some(to_view([(c.0 as f32 + 0.5) * TF, f, (c.1 as f32 + 0.5) * TF]) + Vec3::Y * 0.5),
                    ));
                }
            }
        }
    }
    for e in &l.entities {
        if e.kind != EntityKind::PlayerStart && scene.node_at(e.pos[0], e.pos[1], e.pos[2]).is_none() {
            out.push((format!("{} #{} is inside rock", kind_name(&e.kind), e.id), Some(to_view(e.pos))));
        }
        let key = match &e.kind {
            EntityKind::Door { key, .. } | EntityKind::Chest { key, .. } => *key,
            _ => None,
        };
        if let Some(k) = key {
            if !l.entities.iter().any(|o| o.id == k && matches!(o.kind, EntityKind::Key { .. })) {
                out.push((format!("{} #{} needs key #{k}, which isn't a key", kind_name(&e.kind), e.id), Some(to_view(e.pos))));
            }
        }
    }
    out
}
