//! The MCP's testing tools (the `dev-mcp` feature, stdio only): mouse and keyboard
//! played into the editor through its real input path, and the widgets egui drew,
//! found by label. While they are served only injected input reaches egui, so
//! someone moving the mouse over the window changes nothing.
//!
//! Input goes to egui as events sent straight to its context, and to the 3D camera
//! as Bevy's keyboard and mouse messages, with a window egui maps to no context so
//! it never sees them twice.

use super::{Reply, Response};
use crate::ui::Ui;
use crate::view3d::{Built, EditorCam};
use crate::Document;
use bevy::input::keyboard::{Key as BevyKey, KeyboardInput};
use bevy::input::mouse::{MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::input::{ButtonState, InputSystems};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::egui;
use bevy_egui::input::{EguiInputEvent, FocusedNonWindowEguiContext};
use bevy_egui::{EguiContext, EguiFullOutput, EguiGlobalSettings, EguiInputSet, EguiPostUpdateSet, PrimaryEguiContext};
use egui::accesskit;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeSet, VecDeque};

/// What the testing tools ask of the main thread.
pub(super) enum Command {
    Input(Vec<Step>),
    Widgets(WidgetArgs),
    /// Resize the window (logical pixels).
    Window([u32; 2]),
}

#[derive(Resource, Default)]
pub(super) struct Pending(pub Vec<(Command, Reply)>);

pub(super) fn build(app: &mut App) {
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
    app.init_resource::<Script>()
        .init_resource::<Widgets>()
        .init_resource::<Pending>()
        .add_systems(
            PreUpdate,
            (focus, take, inject).chain().after(super::serve).before(InputSystems).before(EguiInputSet::InitReading),
        )
        .add_systems(PostUpdate, read_widgets.after(EguiPostUpdateSet::EndPass).before(EguiPostUpdateSet::ProcessOutput));
}

/// egui gets the keyboard as if its window had focus, and fills in its widget tree.
fn focus(mut commands: Commands, mut contexts: Query<(Entity, &mut EguiContext), With<PrimaryEguiContext>>, mut enabled: Local<bool>) {
    if let Ok((entity, mut ctx)) = contexts.single_mut() {
        commands.insert_resource(FocusedNonWindowEguiContext(entity));
        if !*enabled {
            ctx.get_mut().enable_accesskit();
            *enabled = true;
        }
    }
}

/// Starts what `serve` passed on.
fn take(
    mut pending: ResMut<Pending>,
    mut script: ResMut<Script>,
    widgets: Res<Widgets>,
    ui: Res<Ui>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    for (command, reply) in std::mem::take(&mut pending.0) {
        let result = match command {
            Command::Input(steps) => {
                // One request at a time (see `EditorServer::slots`), so nothing is queued ahead.
                let mut at = script.pointer;
                let mut all = VecDeque::new();
                match steps.iter().try_for_each(|s| frames(s, &mut at).map(|f| all.extend(f))) {
                    Ok(()) => {
                        let n = all.len();
                        script.queue.push_back((all, reply, n));
                        continue;
                    }
                    Err(e) => Err(e),
                }
            }
            Command::Widgets(args) => widget_list(&widgets.0, &ui, &args),
            Command::Window([w, h]) => match windows.single_mut() {
                Ok(mut window) => {
                    window.resolution.set(w as f32, h as f32);
                    // Answer once the editor has drawn at the new size.
                    script.queue.push_back((VecDeque::from([Vec::new(), Vec::new()]), reply, 0));
                    continue;
                }
                Err(_) => Err("no window".into()),
            },
        };
        let _ = reply.send(result.map(Response::Json));
    }
}

/// `editor_input`'s arguments.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InputArgs {
    pub steps: Vec<Step>,
}

/// One step of `editor_input`. Each becomes one or more frames of events.
#[derive(Deserialize, Debug)]
#[serde(tag = "do", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Step {
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
pub(super) enum Button {
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
pub(super) struct WidgetArgs {
    /// Labels, values or hints containing this (any case).
    text: Option<String>,
    /// Roles containing this (any case): button, checkbox, combobox, label...
    role: Option<String>,
    /// Only widgets inside this panel (its tab title).
    panel: Option<String>,
    limit: Option<usize>,
}

/// Input being played into the editor, one frame of events at a time.
#[derive(Resource, Default)]
pub(super) struct Script {
    /// Each request's frames of events, its reply, and how many frames it had (0
    /// for a plain wait before answering).
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
    headless: Option<Res<super::HeadlessTarget>>,
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
        let mut state = super::status(&ui, &doc.0, &built, &cam, windows.single().ok(), headless.is_some(), script.frame);
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

fn widget_list(widgets: &[Widget], ui: &Ui, args: &WidgetArgs) -> Result<Value, String> {
    let within = args.panel.as_deref().map(|p| super::panel_rect(ui, p)).transpose()?;
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
            let mut o = json!({"role": w.role, "rect": super::rect_json(w.rect), "centre": [c.x, c.y]});
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


/// The testing tools' names, descriptions and argument schemas.
pub(super) fn tools() -> Vec<(&'static str, &'static str, Value)> {
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
    vec![
        ("editor_input", "Play mouse and keyboard input into the editor, a frame at a time, through the same path real input takes (egui and the 3D camera). Positions are window pixels, origin top-left. Returns what changed once the editor has seen the last step. Only injected input reaches egui while this server runs.", json!({"type": "object", "properties": {"steps": {"type": "array", "items": step, "minItems": 1, "maxItems": 256}}, "required": ["steps"], "additionalProperties": false})),
        ("editor_widgets", "Find widgets egui drew last frame (buttons, labels, check boxes, combo boxes, text fields and their hints, menu items) by text, role or panel, with their rectangles [x, y, w, h] and centres to click. Widgets scrolled out of sight are left out.", json!({"type": "object", "properties": {"text": {"type": "string", "description": "Label, value or hint contains this (any case)"}, "role": {"type": "string", "description": "Role contains this: button, checkbox, combobox, label, textinput..."}, "panel": {"type": "string", "description": "Tab title: Viewport, Plan, Tools, Properties, Entities, Textures, Areas, Issues"}, "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 60}}, "additionalProperties": false})),
        ("editor_window", "Resize the window, in pixels. Answers once the editor has drawn at the new size.", json!({"type": "object", "properties": {"size": {"type": "array", "items": {"type": "integer", "minimum": 200, "maximum": 4096}, "minItems": 2, "maxItems": 2}}, "required": ["size"], "additionalProperties": false})),
    ]
}

/// The testing tool `name` asks for, if it is one.
pub(super) fn parse(name: &str, value: Value) -> Option<Result<Command, String>> {
    fn de<T: serde::de::DeserializeOwned>(v: Value) -> Result<T, String> {
        serde_json::from_value(v).map_err(|e| e.to_string())
    }
    Some(match name {
        "editor_input" => de::<InputArgs>(value).and_then(|a| {
            if a.steps.is_empty() || a.steps.len() > 256 {
                Err("give 1..256 steps".into())
            } else {
                Ok(Command::Input(a.steps))
            }
        }),
        "editor_widgets" => de(value).map(Command::Widgets),
        "editor_window" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Size {
                size: [u32; 2],
            }
            de::<Size>(value).map(|s| Command::Window(s.size))
        }
        _ => return None,
    })
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
