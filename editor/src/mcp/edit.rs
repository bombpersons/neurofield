//! The MCP's editing tools: the level changed directly rather than through the
//! mouse, so an agent can build alongside someone using the editor. Each call is
//! one undo step (with the status bar saying what it did), and none is taken while
//! the person is part way through a drag, so the two never tangle.

use super::{dir_of, face_of, level_path};
use crate::doc::Doc;
use crate::tools::{add_entity, entity_kinds, fill, has_face, lattice_near, make_link, paint, place_at, set_point, stairs, Brush, Link, Sel};
use crate::ui::Ui;
use crate::view3d::EditorCam;
use engine::world::level::{Area, AreaStrange, CeilingKind, Dir, EntityKind, Level, PointLight, SlopeDef, SlopeKind, Strangeness, Sun, Surface};
use engine::world::scene::{STEP, T};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::f32::consts::{FRAC_PI_2, PI};

const TF: f32 = T as f32;

/// The level's settings `level_settings` may change (and `editor_level` shows).
const SETTINGS: [&str; 7] = ["name", "textures", "ambient", "fog", "sun", "surface", "strangeness"];

/// The level's settings on their own (the whole level won't go to JSON: its maps
/// have tuple keys).
#[derive(Serialize, Deserialize)]
pub struct Settings {
    name: String,
    textures: String,
    ambient: [f32; 3],
    fog: (f32, f32),
    sun: Option<Sun>,
    surface: Surface,
    strangeness: Strangeness,
}

impl Settings {
    pub fn of(l: &Level) -> Settings {
        Settings {
            name: l.name.clone(),
            textures: l.textures.clone(),
            ambient: l.ambient,
            fog: l.fog,
            sun: l.sun.clone(),
            surface: l.surface,
            strangeness: l.strangeness.clone(),
        }
    }

    fn apply(self, l: &mut Level) {
        (l.name, l.textures, l.ambient, l.fog, l.sun, l.surface, l.strangeness) =
            (self.name, self.textures, self.ambient, self.fog, self.sun, self.surface, self.strangeness);
    }
}

/// Cells one `level_box` may change, to catch a slip of a digit.
const MAX_CELLS: i64 = 64 * 64 * 64;

pub enum Edit {
    Box(BoxArgs),
    Stairs(StairsArgs),
    Sky(SkyArgs),
    Paint(PaintArgs),
    AddEntity(AddEntityArgs),
    EditEntity(EditEntityArgs),
    DeleteEntity(IdArgs),
    Link(LinkArgs),
    AddLight(LightArgs),
    EditLight(IndexEdit),
    DeleteLight(IndexArgs),
    AddArea(AreaArgs),
    EditArea(IndexEdit),
    DeleteArea(IndexArgs),
    Settings(SetArgs),
    Strange(StrangeArgs),
    Undo(UndoArgs),
    Open(OpenArgs),
    Save(SaveArgs),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxArgs {
    /// Cells min..max (exclusive), each [x, layer, z].
    min: [i32; 3],
    max: [i32; 3],
    #[serde(default)]
    fill: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StairsArgs {
    /// The open cell whose floor is the first step.
    cell: [i32; 3],
    dir: String,
    steps: i32,
    #[serde(default = "one")]
    rise: i32,
    #[serde(default)]
    ramp: bool,
    #[serde(default)]
    flat_ceiling: bool,
}

fn one() -> i32 {
    1
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkyArgs {
    cell: [i32; 3],
    sky: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaceArg {
    cell: [i32; 3],
    face: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PaintArgs {
    faces: Vec<FaceArg>,
    slot: u8,
    /// Spread over the joined faces of each one's plane that share its texture.
    #[serde(default)]
    fill: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddEntityArgs {
    /// A kind's name ("door"), or {name: {fields}} to change its defaults.
    kind: Value,
    /// On the floor in the middle of this open cell...
    cell: Option<[i32; 3]>,
    /// ...or, for a door, on this side of it.
    edge: Option<String>,
    /// Or exactly here (world units): on the floor under it.
    pos: Option<[f32; 3]>,
    facing: Option<String>,
    yaw: Option<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditEntityArgs {
    id: u32,
    /// Fields to change, merged into the entity as `editor_level` shows it.
    #[serde(default)]
    set: Map<String, Value>,
    facing: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdArgs {
    id: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkArgs {
    /// A door or chest (gets `to` as its key, and locks), or a lever (gets door
    /// `to` as a target).
    from: u32,
    to: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightArgs {
    pos: [f32; 3],
    #[serde(default = "warm")]
    colour: [f32; 3],
    #[serde(default = "three_tiles")]
    radius: f32,
}

fn warm() -> [f32; 3] {
    [1.0, 0.8, 0.6]
}
fn three_tiles() -> f32 {
    3.0 * TF
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexEdit {
    index: usize,
    set: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexArgs {
    index: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AreaArgs {
    name: String,
    #[serde(default)]
    label: String,
    boxes: Vec<([i32; 3], [i32; 3])>,
    surface: Option<Surface>,
    #[serde(default)]
    notes: String,
    strangeness: Option<AreaStrange>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrangeArgs {
    /// Lattice coordinates [x, up, z] in tiles.
    centre: Option<[f32; 3]>,
    #[serde(default = "two")]
    radius: f32,
    /// Lowest and highest up reached above a disc, instead of a ball.
    column: Option<[f32; 2]>,
    /// Or a box of lattice points, inclusive.
    min: Option<[i32; 3]>,
    max: Option<[i32; 3]>,
    #[serde(default = "add")]
    mode: String,
    #[serde(default = "tenths3")]
    amount: f32,
    #[serde(default = "half")]
    softness: f32,
}

fn two() -> f32 {
    2.0
}

fn add() -> String {
    "add".into()
}

fn tenths3() -> f32 {
    0.3
}

fn half() -> f32 {
    0.5
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetArgs {
    set: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UndoArgs {
    #[serde(default)]
    redo: bool,
    #[serde(default = "one")]
    steps: i32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenArgs {
    path: Option<String>,
    #[serde(default)]
    new: bool,
    #[serde(default)]
    demo: bool,
    /// Drop unsaved changes.
    #[serde(default)]
    discard: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaveArgs {
    path: Option<String>,
}

/// The editing tool `name` asks for, if it is one.
pub fn parse(name: &str, value: Value) -> Option<Result<Edit, String>> {
    fn de<T: serde::de::DeserializeOwned>(v: Value) -> Result<T, String> {
        serde_json::from_value(v).map_err(|e| e.to_string())
    }
    Some(match name {
        "level_box" => de(value).map(Edit::Box),
        "level_stairs" => de(value).map(Edit::Stairs),
        "level_sky" => de(value).map(Edit::Sky),
        "level_paint" => de(value).map(Edit::Paint),
        "level_add_entity" => de(value).map(Edit::AddEntity),
        "level_edit_entity" => de(value).map(Edit::EditEntity),
        "level_delete_entity" => de(value).map(Edit::DeleteEntity),
        "level_link" => de(value).map(Edit::Link),
        "level_add_light" => de(value).map(Edit::AddLight),
        "level_edit_light" => de(value).map(Edit::EditLight),
        "level_delete_light" => de(value).map(Edit::DeleteLight),
        "level_add_area" => de(value).map(Edit::AddArea),
        "level_edit_area" => de(value).map(Edit::EditArea),
        "level_delete_area" => de(value).map(Edit::DeleteArea),
        "level_settings" => de(value).map(Edit::Settings),
        "level_strange" => de(value).map(Edit::Strange),
        "editor_undo" => de(value).map(Edit::Undo),
        "editor_open" => de(value).map(Edit::Open),
        "editor_save" => de(value).map(Edit::Save),
        _ => return None,
    })
}

/// Carries out an edit. Level changes become one undo step of their own, and are
/// refused while the person at the editor is part way through one.
pub fn apply(edit: Edit, ui: &mut Ui, doc: &mut Doc, cam: &mut EditorCam) -> Result<Value, String> {
    let busy = || "the editor is in the middle of an edit (a drag or a pick); try again in a moment".to_string();
    match edit {
        Edit::Undo(a) => {
            if ui.tools.busy() {
                return Err(busy());
            }
            for _ in 0..a.steps.clamp(1, 200) {
                if a.redo {
                    doc.redo();
                } else {
                    doc.undo();
                }
            }
            ui.message = format!("agent: {}", if a.redo { "redo" } else { "undo" });
            let (undo, redo) = doc.history();
            Ok(json!({"undo": undo, "redo": redo}))
        }
        Edit::Open(a) => {
            if doc.dirty() && !a.discard {
                return Err("the level has unsaved changes: save it first, or pass discard: true".into());
            }
            *doc = match (&a.path, a.new) {
                (Some(p), false) => Doc::open(&level_path(ui, p))?,
                (None, true) => Doc::new(if a.demo { Level::demo() } else { Level::new("") }, None),
                _ => return Err("give path, or new: true".into()),
            };
            ui.reset();
            cam.frame(crate::ui::level_centre(&doc.level), 9.0);
            ui.message = format!("agent: opened {}", doc.name());
            Ok(json!({"file": doc.path, "name": doc.level.name}))
        }
        Edit::Save(a) => {
            let path = match (&a.path, &doc.path) {
                (Some(p), _) => level_path(ui, p),
                (None, Some(p)) => p.clone(),
                (None, None) => return Err("the level has no file yet: give a path".into()),
            };
            if path.extension().is_none_or(|e| e != "ron") {
                return Err("level files end in .ron".into());
            }
            doc.save(&path)?;
            ui.message = format!("agent: saved {}", path.display());
            Ok(json!({"file": path}))
        }
        edit => {
            if ui.tools.busy() {
                return Err(busy());
            }
            // Anything the person changed becomes its own step first.
            doc.settle(true);
            let before = doc.level.clone();
            match change(edit, ui, &mut doc.level) {
                Ok((said, out)) => {
                    doc.settle(true);
                    ui.message = format!("agent: {said}");
                    let mut out = out;
                    out["done"] = json!(said);
                    Ok(out)
                }
                Err(e) => {
                    doc.level = before;
                    Err(e)
                }
            }
        }
    }
}

/// A level change: what to say it did, and anything to report back.
fn change(edit: Edit, ui: &mut Ui, l: &mut Level) -> Result<(String, Value), String> {
    let none = json!({});
    Ok(match edit {
        Edit::Box(a) => {
            let size = [0, 1, 2].map(|i| a.max[i] - a.min[i]);
            if size.iter().any(|&s| s <= 0) {
                return Err("max must be beyond min on every axis (it is exclusive)".into());
            }
            if a.min[0] < 0 || a.min[2] < 0 {
                return Err("x and z start at 0".into());
            }
            if size.iter().map(|&s| s as i64).product::<i64>() > MAX_CELLS {
                return Err(format!("more than {MAX_CELLS} cells at once"));
            }
            l.set_box(a.min, a.max, !a.fill);
            let [w, h, d] = size;
            (format!("{} {w} x {d} tiles, {h} layers", if a.fill { "filled" } else { "opened" }), none)
        }
        Edit::Stairs(a) => {
            let dir = dir_of(&a.dir)?;
            if !(1..=4).contains(&a.rise) || !(1..=64).contains(&a.steps) {
                return Err("rise is 1..4 layers a tile, steps 1..64".into());
            }
            let kind = if a.ramp { SlopeKind::Ramp } else { SlopeKind::Stairs };
            let ceiling = if a.flat_ceiling { CeilingKind::Flat } else { CeilingKind::Follow };
            stairs(l, a.cell, dir, a.steps, SlopeDef { dir, rise: a.rise, kind, ceiling })?;
            (format!("{} of {} climbing {:?}", if a.ramp { "a ramp" } else { "stairs" }, a.steps, dir), none)
        }
        Edit::Sky(a) => {
            let [x, y, z] = a.cell;
            let from = l.run_at(x, y, z).ok_or("that cell isn't open")?.from;
            l.run_mut(x, z, from).unwrap().sky = a.sky;
            (format!("{} over ({x}, {z})", if a.sky { "open sky" } else { "a ceiling" }), none)
        }
        Edit::Paint(a) => {
            if a.slot > 7 {
                return Err("slots are 0..7".into());
            }
            let mut faces = Vec::new();
            for f in &a.faces {
                let face = face_of(&f.face)?;
                if !has_face(l, f.cell, face) {
                    return Err(format!("cell {:?} has no {} face", f.cell, f.face));
                }
                faces.push((f.cell, face));
            }
            let n = if a.fill {
                faces.iter().map(|&(c, face)| fill(l, c, face, a.slot)).sum()
            } else {
                paint(l, &faces, a.slot);
                faces.len()
            };
            (format!("painted {n} faces with slot {}", a.slot), none)
        }
        Edit::AddEntity(a) => {
            let kind = entity_kind(&a.kind)?;
            let yaw = match (&a.facing, a.yaw) {
                (Some(f), _) => facing_yaw(dir_of(f)?),
                (None, y) => y.unwrap_or(0.0),
            };
            let (p, free) = match (a.cell, a.pos) {
                (Some(c), None) => {
                    let run = l.run_at(c[0], c[1], c[2]).ok_or(format!("cell {c:?} isn't open"))?;
                    let (mut x, mut z) = ((c[0] as f32 + 0.5) * TF, (c[2] as f32 + 0.5) * TF);
                    if let Some(e) = &a.edge {
                        let (dx, dz) = dir_of(e)?.delta();
                        x += dx as f32 * 0.4 * TF;
                        z += dz as f32 * 0.4 * TF;
                    }
                    ([x, -(run.from as f32) * STEP as f32, z], false)
                }
                (None, Some(p)) => (p, true),
                _ => return Err("give cell (with edge for a door) or pos".into()),
            };
            if matches!(kind, EntityKind::Door { .. }) && a.cell.is_some() && a.edge.is_none() {
                return Err("a door goes on an edge of its cell: give edge (n, e, s or w)".into());
            }
            let (pos, yaw) = place_at(&l.to_scene(), &kind, p, yaw, free).ok_or("no floor there")?;
            let name = crate::ui::kind_name(&kind);
            let id = add_entity(l, kind, pos, yaw);
            ui.sel = Sel::Entity(id);
            (format!("added {name} #{id}"), json!({"id": id, "pos": pos, "yaw": yaw}))
        }
        Edit::EditEntity(a) => {
            let i = l.entities.iter().position(|e| e.id == a.id).ok_or(format!("no entity {}", a.id))?;
            if a.set.contains_key("id") {
                return Err("ids don't change".into());
            }
            let mut set = a.set;
            if let Some(f) = &a.facing {
                set.insert("yaw".into(), json!(facing_yaw(dir_of(f)?)));
            }
            let e = &mut l.entities[i];
            *e = merged(&*e, set)?;
            (format!("changed #{}", a.id), json!({"entity": e}))
        }
        Edit::DeleteEntity(a) => {
            let before = l.entities.len();
            l.entities.retain(|e| e.id != a.id);
            if l.entities.len() == before {
                return Err(format!("no entity {}", a.id));
            }
            // Nothing keeps a link to it.
            for e in &mut l.entities {
                match &mut e.kind {
                    EntityKind::Door { key, .. } | EntityKind::Chest { key, .. } if *key == Some(a.id) => *key = None,
                    EntityKind::Lever { targets } => targets.retain(|&t| t != a.id),
                    _ => {}
                }
            }
            if ui.sel == Sel::Entity(a.id) {
                ui.sel = Sel::None;
            }
            (format!("deleted #{}", a.id), none)
        }
        Edit::Link(a) => {
            let from = l.entities.iter().find(|e| e.id == a.from).ok_or(format!("no entity {}", a.from))?;
            let link = match from.kind {
                EntityKind::Door { .. } | EntityKind::Chest { .. } => Link::Key(a.from),
                EntityKind::Lever { .. } => Link::Target(a.from),
                _ => return Err("link from a door or chest (to its key) or a lever (to a door)".into()),
            };
            let said = make_link(l, link, a.to);
            if said.contains("cancelled") {
                return Err(said.replace(": link cancelled", ""));
            }
            (said, none)
        }
        Edit::AddLight(a) => {
            l.lights.push(PointLight { pos: a.pos, colour: a.colour, radius: a.radius, strange: None });
            let index = l.lights.len() - 1;
            ui.sel = Sel::Light(index);
            (format!("added light {index}"), json!({"index": index}))
        }
        Edit::EditLight(a) => {
            let light = l.lights.get_mut(a.index).ok_or(format!("no light {}", a.index))?;
            *light = merged(&*light, a.set)?;
            (format!("changed light {}", a.index), json!({"light": light}))
        }
        Edit::DeleteLight(a) => {
            if a.index >= l.lights.len() {
                return Err(format!("no light {}", a.index));
            }
            l.lights.remove(a.index);
            if matches!(ui.sel, Sel::Light(_)) {
                ui.sel = Sel::None;
            }
            (format!("deleted light {} (later ones move down one)", a.index), none)
        }
        Edit::AddArea(a) => {
            if a.boxes.iter().any(|(lo, hi)| (0..3).any(|i| hi[i] <= lo[i])) {
                return Err("each box is [min, max] with max beyond min (exclusive)".into());
            }
            l.areas.push(Area { name: a.name.clone(), label: a.label, boxes: a.boxes, surface: a.surface, notes: a.notes, strangeness: a.strangeness });
            let index = l.areas.len() - 1;
            ui.sel = Sel::Area(index);
            (format!("added area {}", a.name), json!({"index": index}))
        }
        Edit::EditArea(a) => {
            let area = l.areas.get_mut(a.index).ok_or(format!("no area {}", a.index))?;
            *area = merged(&*area, a.set)?;
            (format!("changed area {}", area.name), json!({"area": area}))
        }
        Edit::DeleteArea(a) => {
            if a.index >= l.areas.len() {
                return Err(format!("no area {}", a.index));
            }
            let gone = l.areas.remove(a.index);
            if matches!(ui.sel, Sel::Area(_)) {
                ui.sel = Sel::None;
            }
            (format!("deleted area {} (later ones move down one)", gone.name), none)
        }
        Edit::Settings(a) => {
            if let Some(k) = a.set.keys().find(|k| !SETTINGS.contains(&k.as_str())) {
                return Err(format!("{k} isn't a setting ({})", SETTINGS.join(", ")));
            }
            let keys: Vec<String> = a.set.keys().cloned().collect();
            merged(&Settings::of(l), a.set)?.apply(l);
            (format!("changed {}", keys.join(", ")), none)
        }
        Edit::Strange(a) => {
            let near: Vec<((i32, i32, i32), f32)> = match (a.centre, a.min, a.max) {
                (Some(c), None, None) => {
                    if !(0.25..=32.0).contains(&a.radius) {
                        return Err("radius is 0.25..32 tiles".into());
                    }
                    let brush = Brush { radius: a.radius, strength: 1.0, softness: a.softness };
                    lattice_near(c, a.radius, a.column.map(|[lo, hi]| (lo, hi))).into_iter().map(|(p, d)| (p, brush.falloff(d))).collect()
                }
                (None, Some(lo), Some(hi)) => {
                    if (0..3).any(|i| hi[i] < lo[i]) {
                        return Err("max must not be below min (both are inclusive)".into());
                    }
                    if (0..3).map(|i| (hi[i] - lo[i] + 1) as i64).product::<i64>() > MAX_CELLS {
                        return Err(format!("more than {MAX_CELLS} points at once"));
                    }
                    let mut v = Vec::new();
                    for x in lo[0]..=hi[0] {
                        for y in lo[1]..=hi[1] {
                            for z in lo[2]..=hi[2] {
                                v.push(((x, y, z), 1.0));
                            }
                        }
                    }
                    v
                }
                _ => return Err("give centre (with radius, and column for a column) or min and max".into()),
            };
            let at = |l: &Level, p: (i32, i32, i32)| l.strange_points.get(&p).copied().unwrap_or(0.0);
            let new: Vec<_> = near
                .iter()
                .map(|&(p, f)| {
                    let v = at(l, p);
                    let to = match a.mode.as_str() {
                        "add" => Ok(v + a.amount * f),
                        "set" => Ok(v + (a.amount - v) * f),
                        "clear" => Ok(v * (1.0 - f)),
                        "smooth" => {
                            let (x, y, z) = p;
                            let n = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];
                            let mean = n.iter().map(|o| at(l, (x + o.0, y + o.1, z + o.2))).sum::<f32>() / 6.0;
                            Ok(v + (mean - v) * (a.amount * f).clamp(0.0, 1.0))
                        }
                        m => Err(format!("mode {m}: add, set, smooth or clear")),
                    }?;
                    Ok((p, to))
                })
                .collect::<Result<_, String>>()?;
            let mut changed = 0;
            for (p, v) in new {
                let before = at(l, p);
                set_point(l, p, v);
                changed += (at(l, p) != before) as usize;
            }
            (format!("{} strangeness at {changed} points", a.mode), json!({"changed": changed, "points": l.strange_points.len()}))
        }
        Edit::Undo(_) | Edit::Open(_) | Edit::Save(_) => unreachable!("handled in apply"),
    })
}

/// `value` with `set` merged in: objects field by field, anything else replaced.
fn merged<T: serde::Serialize + serde::de::DeserializeOwned>(value: &T, set: Map<String, Value>) -> Result<T, String> {
    fn merge(into: &mut Value, from: Value) {
        match (into, from) {
            (Value::Object(a), Value::Object(b)) => {
                for (k, v) in b {
                    match a.get_mut(&k) {
                        Some(old) => merge(old, v),
                        None => _ = a.insert(k, v),
                    }
                }
            }
            (into, from) => *into = from,
        }
    }
    let mut v = serde_json::to_value(value).map_err(|e| e.to_string())?;
    merge(&mut v, Value::Object(set));
    serde_json::from_value(v).map_err(|e| e.to_string())
}

/// Walker yaw facing `d`: forward is (-sin, cos) in x, z, and north is -z.
fn facing_yaw(d: Dir) -> f32 {
    match d {
        Dir::S => 0.0,
        Dir::W => FRAC_PI_2,
        Dir::N => PI,
        Dir::E => -FRAC_PI_2,
    }
}

/// An entity kind from its name ("door", "player start", "Lever"), or from
/// {name: {fields}} changing that kind's defaults.
fn entity_kind(v: &Value) -> Result<EntityKind, String> {
    let squash = |s: &str| s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
    let (name, fields) = match v {
        Value::String(s) => (s.as_str(), None),
        Value::Object(o) if o.len() == 1 => {
            let (k, f) = o.iter().next().unwrap();
            (k.as_str(), Some(f.clone()))
        }
        _ => return Err("kind is a name, or {name: {fields}}".into()),
    };
    let kinds = entity_kinds();
    let base = kinds.iter().find(|(n, _)| squash(n) == squash(name)).map(|(_, k)| k.clone()).ok_or_else(|| {
        format!("no kind {name}: {}", kinds.iter().map(|k| k.0).collect::<Vec<_>>().join(", "))
    })?;
    match fields {
        None => Ok(base),
        Some(Value::Object(f)) => {
            // Its serialised form is {"Door": {...}}: merge into the inner fields.
            let variant = crate::ui::kind_name(&base);
            merged(&base, Map::from_iter([(variant.to_string(), Value::Object(f))]))
        }
        Some(_) => Err("a kind's fields are an object".into()),
    }
}

/// The editing tools' names, descriptions and argument schemas.
pub fn tools() -> Vec<(&'static str, &'static str, Value)> {
    let cell = json!({"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3, "description": "[x, layer, z]"});
    let point = json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "world units [x, y, z], y down"});
    let dir = json!({"type": "string", "enum": ["n", "e", "s", "w"]});
    let face = json!({"type": "string", "enum": ["floor", "ceiling", "n", "e", "s", "w"]});
    let set = json!({"type": "object", "description": "Fields to change, merged in (objects field by field)"});
    let index = json!({"type": "object", "properties": {"index": {"type": "integer", "minimum": 0}}, "required": ["index"], "additionalProperties": false});
    let index_set = json!({"type": "object", "properties": {"index": {"type": "integer", "minimum": 0}, "set": set}, "required": ["index", "set"], "additionalProperties": false});
    let kinds: Vec<&str> = entity_kinds().iter().map(|k| k.0).collect();
    vec![
        ("level_box", "Open (carve out) every cell from min to max, exclusive, each [x, layer, z]; or with fill: true, fill them with rock. A room 4 tiles square and 8 layers high on floor layer 0 is min [0, 0, 0], max [4, 8, 4]. Columns joined by open cells become one space.", json!({"type": "object", "properties": {"min": cell, "max": cell, "fill": {"type": "boolean", "default": false}}, "required": ["min", "max"], "additionalProperties": false})),
        ("level_stairs", "Stairs (or a ramp) from the floor of an open cell, climbing rise layers a tile toward dir for steps tiles, opening headroom as they go. The cells beyond must be reachable; the cell needs more headroom than the rise.", json!({"type": "object", "properties": {"cell": cell, "dir": dir, "steps": {"type": "integer", "minimum": 1, "maximum": 64}, "rise": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1}, "ramp": {"type": "boolean", "default": false}, "flat_ceiling": {"type": "boolean", "default": false}}, "required": ["cell", "dir", "steps"], "additionalProperties": false})),
        ("level_sky", "Open the column at an open cell to the sky (no ceiling), or roof it again.", json!({"type": "object", "properties": {"cell": cell, "sky": {"type": "boolean"}}, "required": ["cell", "sky"], "additionalProperties": false})),
        ("level_paint", "Paint faces with a texture slot (0..7 of the level's texture set). Each face is a side of an open cell: floor, ceiling, or the wall on its n, e, s or w side. fill spreads over the joined faces on each one's plane that share its texture.", json!({"type": "object", "properties": {"faces": {"type": "array", "items": {"type": "object", "properties": {"cell": cell, "face": face}, "required": ["cell", "face"], "additionalProperties": false}, "minItems": 1}, "slot": {"type": "integer", "minimum": 0, "maximum": 7}, "fill": {"type": "boolean", "default": false}}, "required": ["faces", "slot"], "additionalProperties": false})),
        ("level_add_entity", "Add an entity on the floor: in the middle of an open cell, or exactly at pos. Doors go on an edge of their cell (edge n, e, s or w) facing into it, and open into it. kind is a name or {name: {fields}} to change its defaults, e.g. {\"door\": {\"hinge\": \"Up\", \"width\": 2}}. Facing n, e, s or w (or yaw, radians; 0 faces +z, south). Only one player start: adding one moves it. Returns the id.", json!({"type": "object", "properties": {"kind": {"description": format!("One of: {}", kinds.join(", "))}, "cell": cell, "edge": dir, "pos": point, "facing": dir, "yaw": {"type": "number"}}, "required": ["kind"], "additionalProperties": false})),
        ("level_edit_entity", "Change an entity: set merges fields into it as editor_level shows it, e.g. {\"pos\": [512, 0, 640]} or {\"kind\": {\"Door\": {\"open\": true}}} or {\"kind\": {\"Creature\": {\"seed\": 9}}}. facing n, e, s or w sets its yaw.", json!({"type": "object", "properties": {"id": {"type": "integer"}, "set": set, "facing": dir}, "required": ["id"], "additionalProperties": false})),
        ("level_delete_entity", "Delete an entity, and any links to it.", json!({"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"], "additionalProperties": false})),
        ("level_link", "Link entities: from a door or chest to the key that opens it (which also locks it), or from a lever to a door it works.", json!({"type": "object", "properties": {"from": {"type": "integer"}, "to": {"type": "integer"}}, "required": ["from", "to"], "additionalProperties": false})),
        ("level_add_light", "Add a point light at pos (world units), with a colour (0..1 each) and radius (world units, default 768: three tiles). Returns its index.", json!({"type": "object", "properties": {"pos": point, "colour": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "radius": {"type": "number"}}, "required": ["pos"], "additionalProperties": false})),
        ("level_edit_light", "Change a light by index: set merges into it (pos, colour, radius, strange).", index_set.clone()),
        ("level_delete_light", "Delete a light by index; later lights move down one.", index.clone()),
        ("level_add_area", "Add a named area (a room, a corridor...) made of boxes [[min], [max]] of cells, max exclusive, with a label (e.g. \"crypt\"), optional surface {floor, wall, ceiling: slots, tint: [r, g, b]} for its default paint, notes, and strangeness {\"Add\": v} (added inside it) or {\"Pin\": v} (held at v inside it), fading out over a tile outside.", json!({"type": "object", "properties": {"name": {"type": "string"}, "label": {"type": "string"}, "boxes": {"type": "array", "items": {"type": "array", "items": cell, "minItems": 2, "maxItems": 2}, "minItems": 1}, "surface": {"type": "object"}, "notes": {"type": "string"}, "strangeness": {"type": "object"}}, "required": ["name", "boxes"], "additionalProperties": false})),
        ("level_edit_area", "Change an area by index: set merges into it (name, label, boxes, surface, notes, strangeness: {\"Add\": v}, {\"Pin\": v} or null).", index_set),
        ("level_delete_area", "Delete an area by index; later areas move down one.", index),
        ("level_settings", "Change the level's settings: set merges into name, textures (grounded, mid, deep or strange), ambient, fog [near, far] in tiles, sun {dir, colour} or null, surface (default paint and tint), strangeness. editor_level part settings shows them.", json!({"type": "object", "properties": {"set": set}, "required": ["set"], "additionalProperties": false})),
        ("level_strange", "Paint strangeness: offsets on a lattice of points one tile apart, [x, up, z] in tiles (up is layers / 4; a floor at layer 8 is up 2), added to the depth curve and noise (level_settings strangeness) and smooth between points. Either a ball (centre, radius in tiles, softness 0 hard edge .. 1 fading from the centre), a column over a disc (centre x and z, radius, column [lowest up, highest up]) or a box of points (min, max, inclusive). mode add adds amount (negative takes off), set moves points toward amount, smooth eases them toward their neighbours, clear takes them back to nothing. Offsets stay within -1..1; the strangeness itself is clamped to 0..1. editor_level part strangeness shows them.", json!({"type": "object", "properties": {"centre": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "radius": {"type": "number", "default": 2.0}, "column": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "min": cell, "max": cell, "mode": {"type": "string", "enum": ["add", "set", "smooth", "clear"], "default": "add"}, "amount": {"type": "number", "default": 0.3}, "softness": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5}}, "additionalProperties": false})),
        ("editor_undo", "Undo (or with redo: true, redo) steps of the editor's history, the person's edits and agents' alike.", json!({"type": "object", "properties": {"redo": {"type": "boolean", "default": false}, "steps": {"type": "integer", "minimum": 1, "maximum": 200, "default": 1}}, "additionalProperties": false})),
        ("editor_open", "Open a level file (a path, or a name in the levels folder), or start a new one (new: true; demo: true for the demo level). Refused while there are unsaved changes unless discard: true.", json!({"type": "object", "properties": {"path": {"type": "string"}, "new": {"type": "boolean"}, "demo": {"type": "boolean"}, "discard": {"type": "boolean"}}, "additionalProperties": false})),
        ("editor_save", "Save the level: to its file, or to path (a .ron path, or a name in the levels folder).", json!({"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": false})),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::face_name;

    #[test]
    fn kinds_and_merges() {
        assert_eq!(entity_kind(&json!("player start")).unwrap(), EntityKind::PlayerStart);
        let door = entity_kind(&json!({"Door": {"hinge": "Up", "width": 2}})).unwrap();
        assert!(matches!(door, EntityKind::Door { width: 2, hinge: engine::world::level::Hinge::Up, height: 8, .. }));
        assert!(entity_kind(&json!("dragon")).is_err());
        let light = PointLight { pos: [0.0; 3], colour: [1.0; 3], radius: 10.0, strange: None };
        let moved: PointLight = merged(&light, Map::from_iter([("radius".to_string(), json!(20.0))])).unwrap();
        assert_eq!((moved.radius, moved.colour), (20.0, [1.0; 3]));
        assert!(merged::<PointLight>(&light, Map::from_iter([("radius".to_string(), json!("big"))])).is_err());
        let mut l = Level::demo();
        merged(&Settings::of(&l), Map::from_iter([("fog".to_string(), json!([3.0, 9.0]))])).unwrap().apply(&mut l);
        assert_eq!((l.fog, l.name.as_str()), ((3.0, 9.0), "demo"));
        // The face names round-trip.
        for f in ["floor", "ceiling", "n", "e", "s", "w"] {
            assert_eq!(face_name(face_of(f).unwrap()), f);
        }
    }
}
