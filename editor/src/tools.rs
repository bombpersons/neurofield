//! The editing tools (docs/editor.md): what the pointer does in the 3D view and
//! the plan, and the changes to the level they make. Drags change the level as
//! they go, from a copy taken when they started, so the view shows the result
//! live and the whole drag undoes as one step.

use crate::doc::Doc;
use crate::pick::{self, Hit, Ray, LAYER};
use crate::view3d::{Built, EditorCam, ViewInput};
use bevy::prelude::*;
use bevy_egui::egui;
use engine::world::creature::Kind;
use engine::world::level::{
    Area, CeilingKind, Dir, Entity, EntityKind, Face, Hinge, Level, PointLight, PropShape, SlopeDef, SlopeKind, Surface,
};
use engine::world::scene::{Scene, STEP};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Select,
    Room,
    PushPull,
    Carve,
    Slope,
    Paint,
    Strange,
    Place,
    Light,
}

impl Tool {
    pub const ALL: [Tool; 9] =
        [Tool::Select, Tool::Room, Tool::PushPull, Tool::Carve, Tool::Slope, Tool::Paint, Tool::Strange, Tool::Place, Tool::Light];

    pub fn name(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::Room => "Room",
            Tool::PushPull => "Push/Pull",
            Tool::Carve => "Carve / Fill",
            Tool::Slope => "Slope",
            Tool::Paint => "Paint",
            Tool::Strange => "Strangeness",
            Tool::Place => "Entity",
            Tool::Light => "Light",
        }
    }

    pub fn key(self) -> egui::Key {
        use egui::Key;
        match self {
            Tool::Select => Key::V,
            Tool::Room => Key::R,
            Tool::PushPull => Key::P,
            Tool::Carve => Key::C,
            Tool::Slope => Key::S,
            Tool::Paint => Key::B,
            Tool::Strange => Key::X,
            Tool::Place => Key::E,
            Tool::Light => Key::L,
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Tool::Select => "click a face or entity, shift adds, drag a box of faces, drag an entity to move it; Delete removes it",
            Tool::Room => "drag a rectangle on a floor, move up or down for the height, click; ctrl makes it an area",
            Tool::PushPull => "drag a face (or the selected faces) in or out",
            Tool::Carve => "click or drag to carve a cell, shift to fill one",
            Tool::Slope => "click near a floor's edge to climb away from it, drag for a flight; wheel: rise, Tab: ramp or stairs",
            Tool::Paint => "click or drag to paint faces (the selection, if you click in it), shift-click fills the plane, ctrl-click picks up a face's texture",
            Tool::Strange => "brush strangeness on, shift to take it off, ctrl to smooth",
            Tool::Place => "click a floor to place the chosen entity (doors go on the nearest edge), drag to move it; ctrl: don't snap, Tab: turn",
            Tool::Light => "click a face to put a light by it, drag to lift it off",
        }
    }

    /// The milestone that brings it (docs/editor.md), for tools not built yet.
    pub fn coming(self) -> Option<u32> {
        match self {
            Tool::Strange => Some(6),
            _ => None,
        }
    }
}

pub type FaceRef = ([i32; 3], Face);

/// The thing the Properties window shows (faces are selected separately).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sel {
    #[default]
    None,
    Entity(u32),
    /// Lights and areas by their index in the level.
    Light(usize),
    Area(usize),
}

/// A link being picked in the view or the plan: the next entity clicked becomes
/// the key of door (or chest) `Key`, or a target of lever `Target`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    Key(u32),
    Target(u32),
}

/// Where an entity at `pos` (world) is clicked in the 3D view (view space).
pub fn marker(pos: [f32; 3]) -> Vec3 {
    crate::view3d::to_view(pos) + Vec3::Y * 0.15
}

/// The kinds the Entity tool places, with their names.
pub fn entity_kinds() -> Vec<(&'static str, EntityKind)> {
    vec![
        ("Player start", EntityKind::PlayerStart),
        ("Creature", EntityKind::Creature { kind: Kind::Humanoid, size: 1.0, seed: 1 }),
        ("Door", EntityKind::Door { width: 1, height: 8, hinge: Hinge::Left, open: false, locked: false, key: None }),
        ("Key", EntityKind::Key { name: "key".into() }),
        ("Chest", EntityKind::Chest { locked: false, key: None, contents: Vec::new() }),
        ("Prop", EntityKind::Prop { shape: PropShape::Box, size: [80.0, 120.0, 80.0], tex: 2, tint: [0.8, 0.75, 0.7] }),
        ("Lever", EntityKind::Lever { targets: Vec::new() }),
    ]
}

/// A drag or click under way.
pub enum Op {
    Idle,
    /// Room: the rectangle on the plane at layer `base`, corner to corner (x, z).
    RoomRect { base: i32, start: (i32, i32), end: (i32, i32) },
    /// Room: moving the pointer up or down for the height; a click opens it.
    RoomHeight { base: i32, start: (i32, i32), end: (i32, i32), h: i32, before: Level },
    Push { faces: Vec<FaceRef>, origin: Vec3, normal: Vec3, unit: f32, k: i32, before: Level },
    Carve { cell: [i32; 3], face: Face, fill: bool },
    Slope { cell: [i32; 3], dir: Dir, n: i32, before: Level },
    /// Select: a box of faces on the plane of `face` at `cell`.
    Boxes { cell: [i32; 3], face: Face, end: [i32; 3] },
    /// Paint: a stroke along the plane of `face` at `cell`.
    Paint { cell: [i32; 3], face: Face },
    /// Light: lifting light `index` off the face at `origin` along `normal`.
    Light { index: usize, origin: Vec3, normal: Vec3 },
    /// Moving entity `id` over the floors, once the pointer has left `from` (screen).
    Move { id: u32, from: egui::Pos2, moving: bool },
}

impl Op {
    #[cfg_attr(not(feature = "dev-mcp"), allow(dead_code))]
    pub fn name(&self) -> &'static str {
        match self {
            Op::Idle => "idle",
            Op::RoomRect { .. } => "room rectangle",
            Op::RoomHeight { .. } => "room height",
            Op::Push { .. } => "push",
            Op::Carve { .. } => "carve",
            Op::Slope { .. } => "slope",
            Op::Boxes { .. } => "box select",
            Op::Paint { .. } => "paint",
            Op::Light { .. } => "light",
            Op::Move { .. } => "move",
        }
    }
}

pub struct Tools {
    pub tool: Tool,
    pub op: Op,
    /// The face under the pointer in the 3D view.
    pub hover: Option<Hit>,
    pub faces: BTreeSet<FaceRef>,
    /// Layers the plan's Room and Carve open above the slice.
    pub height: i32,
    pub rise: i32,
    pub kind: SlopeKind,
    pub ceiling: CeilingKind,
    /// The texture slot Paint puts on.
    pub slot: u8,
    /// The Entity tool's kind (an index into `entity_kinds`) and facing, radians.
    pub place: usize,
    pub yaw: f32,
    pub link: Option<Link>,
    /// Something finished on a press: wait for the button to come up.
    swallow: bool,
    /// The plan's Room rectangle, from its first corner.
    plan_start: Option<(i32, i32)>,
}

impl Default for Tools {
    fn default() -> Self {
        Tools {
            tool: Tool::Select,
            op: Op::Idle,
            hover: None,
            faces: BTreeSet::new(),
            height: 12,
            rise: 4,
            kind: SlopeKind::Stairs,
            ceiling: CeilingKind::Follow,
            slot: 0,
            place: 1,
            yaw: 0.0,
            link: None,
            swallow: false,
            plan_start: None,
        }
    }
}

const HOVER: Color = Color::srgb(1.0, 0.95, 0.5);
const SELECTED: Color = Color::srgb(0.3, 0.85, 1.0);
const DRAWING: Color = Color::srgb(1.0, 0.55, 0.2);

impl Tools {
    pub fn busy(&self) -> bool {
        !matches!(self.op, Op::Idle) || self.plan_start.is_some()
    }

    /// Stop whatever is under way, putting the level back. False if nothing was.
    pub fn cancel(&mut self, doc: &mut Doc) -> bool {
        self.plan_start = None;
        let linking = self.link.take().is_some();
        match std::mem::replace(&mut self.op, Op::Idle) {
            Op::Idle => return linking,
            Op::RoomHeight { before, .. } | Op::Push { before, .. } | Op::Slope { before, .. } => doc.level = before,
            // A light being placed goes again.
            Op::Light { index, .. } if index < doc.level.lights.len() => _ = doc.level.lights.remove(index),
            _ => {}
        }
        true
    }

    /// Forget everything about the level before (a new one was opened).
    pub fn reset(&mut self) {
        *self = Tools {
            tool: self.tool,
            height: self.height,
            rise: self.rise,
            kind: self.kind,
            ceiling: self.ceiling,
            slot: self.slot,
            place: self.place,
            yaw: self.yaw,
            ..Tools::default()
        };
    }

    pub fn set_tool(&mut self, t: Tool, doc: &mut Doc) {
        if t != self.tool {
            self.cancel(doc);
            self.tool = t;
        }
    }

    /// The pointer in the 3D view, which fills `screen`: `over` when it is on the
    /// view and not on a panel. Draws its outlines into `built.lines`.
    #[allow(clippy::too_many_arguments)]
    pub fn viewport(
        &mut self,
        ctx: &egui::Context,
        screen: egui::Rect,
        over: bool,
        doc: &mut Doc,
        built: &mut Built,
        cam: &EditorCam,
        sel: &mut Sel,
        message: &mut String,
        input: &mut ViewInput,
    ) {
        built.lines.clear();
        let (pressed, released, down, mods, wheel, cancel) = ctx.input(|i| {
            (
                i.pointer.primary_pressed(),
                i.pointer.primary_released(),
                i.pointer.primary_down(),
                i.modifiers,
                i.events.iter().map(|e| if let egui::Event::MouseWheel { delta, .. } = e { delta.y } else { 0.0 }).sum::<f32>(),
                i.pointer.secondary_pressed(),
            )
        });
        let ray = ctx.pointer_latest_pos().map(|p| pick::ray(cam, screen, p));
        self.hover = if over { ray.and_then(|r| pick::pick(&built.scene, &doc.level, r)) } else { None };
        if cancel && self.cancel(doc) {
            *message = "cancelled".into();
        }
        if self.swallow {
            self.swallow = down;
        }
        // A quick click can press and release within one frame.
        let up = released || !down;
        // Alt + left orbits the camera.
        let start = pressed && over && !mods.alt && !self.swallow;

        let floor_hover = self.hover.filter(|h| h.face == Face::Floor);
        input.wheel_taken = self.tool == Tool::Slope && floor_hover.is_some();
        if input.wheel_taken && wheel != 0.0 {
            self.rise = (self.rise + wheel.signum() as i32).clamp(1, 4);
        }

        if matches!(self.op, Op::Idle) {
            // Faces that undo, or another edit, took away.
            self.faces.retain(|&(c, f)| has_face(&doc.level, c, f));
        }
        let op = std::mem::replace(&mut self.op, Op::Idle);
        self.op = match (op, ray) {
            (Op::Idle, _) if start => self.begin(ray, mods, doc, built, cam, screen, ctx, sel, message),
            (Op::RoomRect { base, start, end }, Some(r)) => {
                let end = floor_cell(r, base).unwrap_or(end);
                if up {
                    Op::RoomHeight { base, start, end, h: 0, before: doc.level.clone() }
                } else {
                    Op::RoomRect { base, start, end }
                }
            }
            (Op::RoomHeight { base, start, end, mut h, before }, Some(r)) => {
                let corner = Vec3::new(end.0 as f32 + 0.5, base as f32 * LAYER, -(end.1 as f32 + 0.5));
                if let Some(s) = pick::along_line(r, corner, Vec3::Y) {
                    let want = ((s / LAYER).round() as i32).clamp(-64, 64);
                    if want != h {
                        h = want;
                        doc.level = before.clone();
                        open_room(&mut doc.level, base, start, end, h);
                    }
                }
                let (w, d) = ((start.0 - end.0).abs() + 1, (start.1 - end.1).abs() + 1);
                *message = format!("room {w} x {d} tiles, {h} layers{}", if mods.ctrl { ", as an area" } else { "" });
                if pressed {
                    if h != 0 && mods.ctrl {
                        let (lo, hi) = room_box(base, start, end, h);
                        let name = format!("room {}", doc.level.areas.len() + 1);
                        doc.level.areas.push(Area {
                            name,
                            label: "room".into(),
                            boxes: vec![(lo, hi)],
                            surface: None,
                            notes: String::new(),
                            strangeness: None,
                        });
                    }
                    self.swallow = true;
                    Op::Idle
                } else {
                    Op::RoomHeight { base, start, end, h, before }
                }
            }
            (Op::Push { faces, origin, normal, unit, mut k, before }, Some(r)) => {
                if let Some(s) = pick::along_line(r, origin, normal) {
                    let want = (s / unit).round() as i32;
                    if want != k {
                        k = want;
                        doc.level = before.clone();
                        push(&mut doc.level, &faces, k);
                    }
                }
                let what = if unit < 1.0 { "layer" } else { "tile" };
                *message = format!("{} by {} {what}{}", if k > 0 { "pulled" } else { "pushed" }, k.abs(), if k.abs() == 1 { "" } else { "s" });
                if up {
                    // A pushed selection stays selected where it went.
                    if faces.iter().all(|f| self.faces.contains(f)) && k != 0 {
                        self.faces = faces.iter().map(|&(c, f)| (moved(c, f, k), f)).collect();
                    }
                    Op::Idle
                } else {
                    Op::Push { faces, origin, normal, unit, k, before }
                }
            }
            (Op::Carve { cell, face, fill }, Some(r)) => {
                if let Some(c) = pick::cell_on_plane(r, cell, face) {
                    if has_face(&doc.level, c, face) {
                        push(&mut doc.level, &[(c, face)], if fill { 1 } else { -1 });
                    }
                }
                if up { Op::Idle } else { Op::Carve { cell, face, fill } }
            }
            (Op::Slope { cell, dir, mut n, before }, Some(r)) => {
                let (p0, normal) = pick::plane(cell, Face::Floor);
                if let Some(p) = pick::on_plane(r, p0, normal) {
                    let (dx, dz) = dir.delta();
                    let centre = Vec3::new(cell[0] as f32 + 0.5, p0.y, -(cell[2] as f32 + 0.5));
                    let t = (p - centre).dot(Vec3::new(dx as f32, 0.0, -dz as f32));
                    let want = 1 + (t.round() as i32).clamp(0, 31);
                    if want != n {
                        n = want;
                        doc.level = before.clone();
                        let _ = stairs(&mut doc.level, cell, dir, n, self.slope());
                    }
                }
                *message = format!("{n} column{} of {:?}, rise {}", if n == 1 { "" } else { "s" }, self.kind, self.rise);
                if up { Op::Idle } else { Op::Slope { cell, dir, n, before } }
            }
            (Op::Boxes { cell, face, end }, Some(r)) => {
                let end = pick::cell_on_plane(r, cell, face).unwrap_or(end);
                if up {
                    if end == cell {
                        let unit = unit(&doc.level, cell, face);
                        if mods.shift && unit.iter().all(|f| self.faces.contains(f)) {
                            unit.iter().for_each(|f| _ = self.faces.remove(f));
                        } else {
                            self.faces.extend(unit);
                        }
                    } else {
                        self.faces.extend(faces_in(&doc.level, cell, end, face));
                    }
                    Op::Idle
                } else {
                    Op::Boxes { cell, face, end }
                }
            }
            (Op::Paint { cell, face }, Some(r)) => {
                if let Some(c) = pick::cell_on_plane(r, cell, face) {
                    if has_face(&doc.level, c, face) && face_tex(&doc.level, c, face) != self.slot {
                        let faces = unit(&doc.level, c, face);
                        paint(&mut doc.level, &faces, self.slot);
                    }
                }
                if up { Op::Idle } else { Op::Paint { cell, face } }
            }
            (Op::Light { index, origin, normal }, Some(r)) => {
                if let (Some(s), Some(l)) = (pick::along_line(r, origin, normal), doc.level.lights.get_mut(index)) {
                    let pos = crate::view3d::to_world(origin + normal * s.clamp(LIFT, 12.0));
                    if pos != l.pos {
                        l.pos = pos;
                    }
                }
                if up { Op::Idle } else { Op::Light { index, origin, normal } }
            }
            (Op::Move { id, from, mut moving }, _) => {
                let p = ctx.pointer_latest_pos().unwrap_or(from);
                moving |= (p - from).length() > 4.0;
                let e = doc.level.entities.iter().find(|e| e.id == id).cloned();
                if let (true, Some(h), Some(e)) = (moving, self.hover, e) {
                    if let Some((pos, yaw)) = place_on(&built.scene, &h, &e.kind, e.yaw, mods.ctrl) {
                        let e = doc.level.entities.iter_mut().find(|e| e.id == id).unwrap();
                        if e.pos != pos || e.yaw != yaw {
                            e.pos = pos;
                            e.yaw = yaw;
                        }
                    }
                }
                if up { Op::Idle } else { Op::Move { id, from, moving } }
            }
            (op, _) => op,
        };
        self.outlines(doc, built, ray, *sel);
    }

    /// A press on the view starting the current tool's operation.
    #[allow(clippy::too_many_arguments)]
    fn begin(
        &mut self,
        ray: Option<Ray>,
        mods: egui::Modifiers,
        doc: &mut Doc,
        built: &Built,
        cam: &EditorCam,
        screen: egui::Rect,
        ctx: &egui::Context,
        sel: &mut Sel,
        message: &mut String,
    ) -> Op {
        let Some(r) = ray else { return Op::Idle };
        let p = ctx.pointer_latest_pos().unwrap_or_default();
        // The entity or light marker near the pointer.
        let near = {
            let entities = doc.level.entities.iter().map(|e| (Sel::Entity(e.id), marker(e.pos)));
            let lights = doc.level.lights.iter().enumerate().map(|(i, l)| (Sel::Light(i), crate::view3d::to_view(l.pos)));
            entities
                .chain(lights)
                .filter_map(|(s, v)| Some((s, (pick::project(cam, screen, v)? - p).length())))
                .filter(|&(_, d)| d < 10.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(s, _)| s)
        };
        if let Some(link) = self.link.take() {
            match near {
                Some(Sel::Entity(id)) => *message = make_link(&mut doc.level, link, id),
                _ => *message = "no entity there: link cancelled".into(),
            }
            self.swallow = true;
            return Op::Idle;
        }
        match self.tool {
            Tool::Select => {
                if let Some(s) = near {
                    *sel = s;
                    self.faces.clear();
                    return match s {
                        Sel::Entity(id) => Op::Move { id, from: p, moving: false },
                        _ => Op::Idle,
                    };
                }
                *sel = Sel::None;
                if !mods.shift {
                    self.faces.clear();
                }
                match self.hover {
                    Some(h) => Op::Boxes { cell: h.cell, face: h.face, end: h.cell },
                    None => Op::Idle,
                }
            }
            Tool::Room => {
                let base = match self.hover {
                    Some(Hit { cell, face: Face::Floor, .. }) => cell[1],
                    Some(Hit { cell, face: Face::Ceiling, .. }) => cell[1] + 1,
                    _ => built.slice,
                };
                match floor_cell(r, base) {
                    Some(c) => Op::RoomRect { base, start: c, end: c },
                    None => Op::Idle,
                }
            }
            Tool::PushPull => {
                let Some(h) = self.hover else { return Op::Idle };
                let faces = if self.faces.contains(&(h.cell, h.face)) {
                    self.faces.iter().copied().collect()
                } else {
                    unit(&doc.level, h.cell, h.face)
                };
                let (_, normal) = pick::plane(h.cell, h.face);
                let unit = if matches!(h.face, Face::Wall(_)) { 1.0 } else { LAYER };
                Op::Push { faces, origin: h.point, normal, unit, k: 0, before: doc.level.clone() }
            }
            Tool::Carve => {
                let Some(h) = self.hover else { return Op::Idle };
                push(&mut doc.level, &[(h.cell, h.face)], if mods.shift { 1 } else { -1 });
                Op::Carve { cell: h.cell, face: h.face, fill: mods.shift }
            }
            Tool::Slope => {
                let Some(h) = self.hover.filter(|h| h.face == Face::Floor) else { return Op::Idle };
                let [x, l, z] = h.cell;
                if mods.shift {
                    if let Some(run) = doc.level.run_mut(x, z, l) {
                        run.slope = None;
                    }
                    self.swallow = true;
                    return Op::Idle;
                }
                let dir = climb_dir(h);
                let before = doc.level.clone();
                match stairs(&mut doc.level, h.cell, dir, 1, self.slope()) {
                    Ok(()) => Op::Slope { cell: h.cell, dir, n: 1, before },
                    Err(e) => {
                        *message = e;
                        Op::Idle
                    }
                }
            }
            Tool::Paint => {
                let Some(h) = self.hover else { return Op::Idle };
                if mods.ctrl {
                    self.slot = face_tex(&doc.level, h.cell, h.face);
                    *message = format!("picked texture {}", self.slot);
                    self.swallow = true;
                    return Op::Idle;
                }
                if mods.shift {
                    let n = fill(&mut doc.level, h.cell, h.face, self.slot);
                    *message = format!("filled {n} face{}", if n == 1 { "" } else { "s" });
                    self.swallow = true;
                    return Op::Idle;
                }
                let faces = if self.faces.contains(&(h.cell, h.face)) {
                    self.faces.iter().copied().collect()
                } else {
                    unit(&doc.level, h.cell, h.face)
                };
                paint(&mut doc.level, &faces, self.slot);
                Op::Paint { cell: h.cell, face: h.face }
            }
            Tool::Light => {
                let Some(h) = self.hover else { return Op::Idle };
                let (_, normal) = pick::plane(h.cell, h.face);
                let tf = crate::view3d::TF;
                doc.level.lights.push(PointLight {
                    pos: crate::view3d::to_world(h.point + normal * LIFT),
                    colour: [1.0, 0.8, 0.6],
                    radius: 3.0 * tf,
                    strange: None,
                });
                let index = doc.level.lights.len() - 1;
                *sel = Sel::Light(index);
                Op::Light { index, origin: h.point, normal }
            }
            Tool::Place => {
                let kind = entity_kinds()[self.place].1.clone();
                let placed = self.hover.and_then(|h| place_on(&built.scene, &h, &kind, self.yaw, mods.ctrl));
                let Some((pos, yaw)) = placed else {
                    *message = "entities go on floors".into();
                    self.swallow = true;
                    return Op::Idle;
                };
                let id = add_entity(&mut doc.level, kind, pos, yaw);
                *sel = Sel::Entity(id);
                Op::Move { id, from: p, moving: true }
            }
            _ => Op::Idle,
        }
    }

    fn slope(&self) -> SlopeDef {
        SlopeDef { dir: Dir::E, rise: self.rise, kind: self.kind, ceiling: self.ceiling }
    }

    /// Outlines for the selection, what the pointer is over and what is being drawn.
    fn outlines(&self, doc: &Doc, built: &mut Built, ray: Option<Ray>, sel: Sel) {
        let l = &doc.level;
        let lines = &mut built.lines;
        for &(c, f) in &self.faces {
            outline(lines, pick::quad(l, c, f), SELECTED);
        }
        match sel {
            Sel::Entity(id) => {
                if let Some(e) = l.entities.iter().find(|e| e.id == id) {
                    ghost(lines, &e.kind, e.pos, e.yaw, SELECTED);
                }
            }
            Sel::Area(i) => {
                for &(lo, hi) in l.areas.get(i).map_or(&[][..], |a| &a.boxes) {
                    cuboid(lines, lo, hi, SELECTED);
                }
            }
            Sel::Light(i) => {
                if let Some(light) = l.lights.get(i) {
                    let p = crate::view3d::to_view(light.pos);
                    let lo = p - Vec3::splat(0.2);
                    let hi = p + Vec3::splat(0.2);
                    let c = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
                    for (a, b) in [
                        (c(lo.x, lo.y, lo.z), c(hi.x, lo.y, lo.z)),
                        (c(lo.x, hi.y, lo.z), c(hi.x, hi.y, lo.z)),
                        (c(lo.x, lo.y, hi.z), c(hi.x, lo.y, hi.z)),
                        (c(lo.x, hi.y, hi.z), c(hi.x, hi.y, hi.z)),
                        (c(lo.x, lo.y, lo.z), c(lo.x, hi.y, lo.z)),
                        (c(hi.x, lo.y, lo.z), c(hi.x, hi.y, lo.z)),
                        (c(lo.x, lo.y, hi.z), c(lo.x, hi.y, hi.z)),
                        (c(hi.x, lo.y, hi.z), c(hi.x, hi.y, hi.z)),
                        (c(lo.x, lo.y, lo.z), c(lo.x, lo.y, hi.z)),
                        (c(hi.x, lo.y, lo.z), c(hi.x, lo.y, hi.z)),
                        (c(lo.x, hi.y, lo.z), c(lo.x, hi.y, hi.z)),
                        (c(hi.x, hi.y, lo.z), c(hi.x, hi.y, hi.z)),
                    ] {
                        lines.push((a, b, SELECTED));
                    }
                    // A drop line to the floor below, to see where it hangs.
                    let floor = built.scene.floor_y(light.pos[0], light.pos[1], light.pos[2]);
                    if let Some(f) = floor {
                        let below = crate::view3d::to_view([light.pos[0], f, light.pos[2]]);
                        lines.push((p, below, SELECTED));
                    }
                }
            }
            _ => {}
        }
        match &self.op {
            Op::RoomRect { base, start, end } => {
                let (lo, hi) = room_box(*base, *start, *end, 0);
                cuboid(lines, lo, hi, DRAWING);
            }
            Op::RoomHeight { base, start, end, h, .. } => {
                let (lo, hi) = room_box(*base, *start, *end, *h);
                cuboid(lines, lo, hi, DRAWING);
            }
            Op::Boxes { cell, face, end } => {
                for (c, f) in faces_in(l, *cell, *end, *face) {
                    outline(lines, pick::quad(l, c, f), DRAWING);
                }
            }
            Op::Push { faces, .. } => {
                for &(c, f) in faces {
                    outline(lines, pick::quad(l, c, f), DRAWING);
                }
            }
            Op::Idle => {
                let Some(h) = self.hover else {
                    // The Room tool can start on the slice's grid.
                    if let (Tool::Room, Some(r)) = (self.tool, ray) {
                        if let Some(c) = floor_cell(r, built.slice) {
                            let (lo, hi) = room_box(built.slice, c, c, 0);
                            cuboid(lines, lo, hi, HOVER);
                        }
                    }
                    return;
                };
                match self.tool {
                    Tool::Select | Tool::PushPull | Tool::Paint => {
                        for (c, f) in unit(l, h.cell, h.face) {
                            outline(lines, pick::quad(l, c, f), HOVER);
                        }
                    }
                    Tool::Carve => outline(lines, pick::quad(l, h.cell, h.face), HOVER),
                    Tool::Light => {
                        let (_, n) = pick::plane(h.cell, h.face);
                        let p = h.point + n * LIFT;
                        lines.push((h.point, p, HOVER));
                        for a in [Vec3::X, Vec3::Y, Vec3::Z] {
                            lines.push((p - a * 0.1, p + a * 0.1, HOVER));
                        }
                    }
                    Tool::Room => {
                        let base = match h.face {
                            Face::Floor => h.cell[1],
                            Face::Ceiling => h.cell[1] + 1,
                            Face::Wall(_) => built.slice,
                        };
                        if let Some(c) = ray.and_then(|r| floor_cell(r, base)) {
                            let (lo, hi) = room_box(base, c, c, 0);
                            cuboid(lines, lo, hi, HOVER);
                        }
                    }
                    Tool::Place => {
                        let kind = &entity_kinds()[self.place].1;
                        if let Some((pos, yaw)) = place_on(&built.scene, &h, kind, self.yaw, false) {
                            ghost(lines, kind, pos, yaw, HOVER);
                        }
                    }
                    Tool::Slope if h.face == Face::Floor => {
                        let q = pick::quad(l, h.cell, h.face);
                        outline(lines, q, HOVER);
                        // An arrow up the way it would climb.
                        let (dx, dz) = climb_dir(h).delta();
                        let d = Vec3::new(dx as f32, 0.0, -dz as f32);
                        let c = (q[0] + q[2]) / 2.0;
                        let (a, b) = (c - d * 0.35, c + d * 0.35 + Vec3::Y * self.rise as f32 * LAYER * 0.7);
                        let side = d.cross(Vec3::Y) * 0.12;
                        lines.extend([(a, b, HOVER), (b, b - d * 0.15 + side, HOVER), (b, b - d * 0.15 - side, HOVER)]);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// The pointer in the plan, at `tile` (x, z in tiles) when over it. Returns a
    /// rectangle (min, max tile, inclusive) to draw for a Room under way.
    pub fn plan(
        &mut self,
        r: &egui::Response,
        mods: egui::Modifiers,
        tile: Option<egui::Vec2>,
        slice: i32,
        doc: &mut Doc,
        scene: &Scene,
        sel: &mut Sel,
        message: &mut String,
    ) -> Option<((i32, i32), (i32, i32))> {
        const TF: f32 = crate::view3d::TF;
        let cell = tile.map(|t| (t.x.floor() as i32, t.y.floor() as i32));
        let (primary_down, primary_pressed) = r.ctx.input(|i| (i.pointer.primary_down(), i.pointer.primary_pressed()));
        match self.tool {
            Tool::Room => {
                if r.drag_started_by(egui::PointerButton::Primary) || (primary_pressed && r.hovered() && self.plan_start.is_none()) {
                    self.plan_start = cell;
                }
                let start = self.plan_start?;
                let end = cell.unwrap_or(start);
                if !primary_down {
                    self.plan_start = None;
                    let (lo, hi) = room_box(slice, start, end, self.height);
                    doc.level.set_box(lo, hi, true);
                    if mods.ctrl {
                        let name = format!("room {}", doc.level.areas.len() + 1);
                        doc.level.areas.push(Area { name, label: "room".into(), boxes: vec![(lo, hi)], surface: None, notes: String::new(), strangeness: None });
                    }
                    *message = format!("opened {} x {} tiles", (end.0 - start.0).abs() + 1, (end.1 - start.1).abs() + 1);
                    return None;
                }
                Some(((start.0.min(end.0), start.1.min(end.1)), (start.0.max(end.0), start.1.max(end.1))))
            }
            Tool::Carve => {
                let (x, z) = cell?;
                if primary_down && r.is_pointer_button_down_on() && x >= 0 && z >= 0 {
                    if mods.shift {
                        doc.level.close(x, z, slice, slice + self.height);
                    } else {
                        doc.level.open(x, z, slice, slice + self.height);
                    }
                }
                None
            }
            Tool::Place => {
                let t = tile?;
                if r.clicked() {
                    let kind = entity_kinds()[self.place].1.clone();
                    let y = -(slice * STEP) as f32 - 8.0;
                    match place_at(scene, &kind, [t.x * TF, y, t.y * TF], self.yaw, mods.ctrl) {
                        Some((pos, yaw)) => {
                            let id = add_entity(&mut doc.level, kind, pos, yaw);
                            *sel = Sel::Entity(id);
                        }
                        None => *message = "nothing open at the slice there".into(),
                    }
                }
                None
            }
            Tool::Slope => {
                let (x, z) = cell?;
                if r.clicked() {
                    let t = tile?;
                    let Some(from) = doc.level.run_at(x, slice, z).map(|r| r.from) else {
                        *message = "nothing open at the slice there".into();
                        return None;
                    };
                    let dir = if mods.shift { None } else { Some(edge_dir(t.x - x as f32, t.y - z as f32)) };
                    let s = self.slope();
                    if let Some(run) = doc.level.run_mut(x, z, from) {
                        run.slope = dir.map(|dir| SlopeDef { dir, ..s });
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// The current tool's settings, under the palette.
    pub fn options(&mut self, ui: &mut egui::Ui) {
        match self.tool {
            Tool::Room | Tool::Carve => {
                ui.horizontal(|ui| {
                    ui.label("plan height");
                    ui.add(egui::DragValue::new(&mut self.height).range(1..=64).suffix(" layers"));
                })
                .response
                .on_hover_text("How many layers the plan view opens above the slice");
            }
            Tool::Slope => {
                ui.horizontal(|ui| {
                    ui.label("rise");
                    ui.add(egui::Slider::new(&mut self.rise, 1..=4).suffix(" layers"));
                });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.kind, SlopeKind::Stairs, "Stairs");
                    ui.selectable_value(&mut self.kind, SlopeKind::Ramp, "Ramp");
                });
                ui.horizontal(|ui| {
                    ui.label("ceiling");
                    ui.selectable_value(&mut self.ceiling, CeilingKind::Follow, "Follows");
                    ui.selectable_value(&mut self.ceiling, CeilingKind::Flat, "Flat");
                });
            }
            Tool::Paint => {
                ui.horizontal(|ui| {
                    ui.label("texture");
                    ui.add(egui::DragValue::new(&mut self.slot).range(0..=7));
                })
                .response
                .on_hover_text("Pick one in the Textures window, or ctrl-click a face");
            }
            Tool::Place => {
                ui.horizontal_wrapped(|ui| {
                    for (i, (name, kind)) in entity_kinds().iter().enumerate() {
                        let c = crate::view3d::entity_colour(kind).to_srgba();
                        let c = egui::Color32::from_rgb((c.red * 255.0) as u8, (c.green * 255.0) as u8, (c.blue * 255.0) as u8);
                        ui.selectable_value(&mut self.place, i, egui::RichText::new(*name).color(c));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("facing");
                    let mut deg = self.yaw.to_degrees().round();
                    if ui.add(egui::DragValue::new(&mut deg).speed(1.0).suffix("°")).changed() {
                        self.yaw = deg.to_radians();
                    }
                })
                .response
                .on_hover_text("Tab turns it a quarter; doors face into the cell you click in, and open into it");
            }
            Tool::Select | Tool::PushPull if !self.faces.is_empty() => {
                ui.horizontal(|ui| {
                    ui.label(format!("{} face{} selected", self.faces.len(), if self.faces.len() == 1 { "" } else { "s" }));
                    if ui.small_button("clear").clicked() {
                        self.faces.clear();
                    }
                });
            }
            _ => {}
        }
    }
}

/// The cell (x, z) under the ray on the horizontal plane at layer `base`.
fn floor_cell(r: Ray, base: i32) -> Option<(i32, i32)> {
    let p = pick::on_plane(r, Vec3::new(0.0, base as f32 * LAYER, 0.0), Vec3::Y)?;
    Some((p.x.floor() as i32, (-p.z).floor() as i32))
}

/// The box (min, max exclusive, each (x, layer, z)) a Room covers: the rectangle
/// `a`..`b` (inclusive), `h` layers up from `base` (or down if negative).
fn room_box(base: i32, a: (i32, i32), b: (i32, i32), h: i32) -> ([i32; 3], [i32; 3]) {
    let (lo, hi) = if h >= 0 { (base, base + h) } else { (base + h, base) };
    ([a.0.min(b.0), lo, a.1.min(b.1)], [a.0.max(b.0) + 1, hi, a.1.max(b.1) + 1])
}

fn open_room(l: &mut Level, base: i32, a: (i32, i32), b: (i32, i32), h: i32) {
    if h == 0 {
        return;
    }
    let (lo, hi) = room_box(base, a, b, h);
    l.set_box([lo[0].max(0), lo[1], lo[2].max(0)], hi, true);
}

/// The slope direction for a floor hit: climbing away from the nearest edge.
fn climb_dir(h: Hit) -> Dir {
    let w = crate::view3d::to_world(h.point);
    let tf = crate::view3d::TF;
    edge_dir(w[0] / tf - h.cell[0] as f32, w[2] / tf - h.cell[2] as f32)
}

/// Away from the edge nearest (u, v) across a tile (u east, v south).
fn edge_dir(u: f32, v: f32) -> Dir {
    let edges = [(u, Dir::E), (1.0 - u, Dir::W), (v, Dir::S), (1.0 - v, Dir::N)];
    edges.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
}

fn outline(lines: &mut Vec<(Vec3, Vec3, Color)>, q: [Vec3; 4], c: Color) {
    for i in 0..4 {
        lines.push((q[i], q[(i + 1) % 4], c));
    }
}

/// The edges of a box of cells, (x, layer, z) from `lo` to `hi` exclusive.
fn cuboid(lines: &mut Vec<(Vec3, Vec3, Color)>, lo: [i32; 3], hi: [i32; 3], c: Color) {
    let p = |x: i32, l: i32, z: i32| Vec3::new(x as f32, l as f32 * LAYER, -z as f32);
    let (a, b) = (lo, hi);
    let corners = |l: i32| [p(a[0], l, a[2]), p(b[0], l, a[2]), p(b[0], l, b[2]), p(a[0], l, b[2])];
    let (bottom, top) = (corners(a[1]), corners(b[1]));
    outline(lines, bottom.map(|v| v + Vec3::Y * 0.015), c);
    if b[1] != a[1] {
        outline(lines, top, c);
        for i in 0..4 {
            lines.push((bottom[i], top[i], c));
        }
    }
}

/// Where an entity of `kind` goes for the pointer on `h`, and its facing: see `place_at`.
pub fn place_on(scene: &Scene, h: &Hit, kind: &EntityKind, yaw: f32, free: bool) -> Option<([f32; 3], f32)> {
    if h.face != Face::Floor {
        return None;
    }
    place_at(scene, kind, crate::view3d::to_world(h.point), yaw, free)
}

/// Where an entity of `kind` goes for world point `p` and its facing: on the floor
/// at the middle of the cell (or right at `p` if `free`), facing `yaw`. Doors go
/// on the middle of the cell's nearest edge instead, facing into the cell (so
/// they open into it).
pub fn place_at(scene: &Scene, kind: &EntityKind, p: [f32; 3], yaw: f32, free: bool) -> Option<([f32; 3], f32)> {
    const TF: f32 = crate::view3d::TF;
    let (x, z) = ((p[0] / TF).floor(), (p[2] / TF).floor());
    let (cx, cz) = ((x + 0.5) * TF, (z + 0.5) * TF);
    // The floor of the cell, near the pointer's height.
    let floor = |px: f32, pz: f32| scene.floor_y(px, p[1] - 8.0, pz);
    if let EntityKind::Door { width, .. } = kind {
        let (fx, fz) = (p[0] / TF - x - 0.5, p[2] / TF - z - 0.5);
        let (dx, dz) = if fx.abs() > fz.abs() { (fx.signum(), 0.0) } else { (0.0, fz.signum()) };
        // Forward is (-sin, cos): back into the cell, the way it swings open.
        let yaw = dx.atan2(-dz);
        // A wider door spreads to the right from this edge.
        let shift = (*width - 1).max(0) as f32 * 0.5 * TF;
        let (ex, ez) = (cx + dx * TF / 2.0 + yaw.cos() * shift, cz + dz * TF / 2.0 + yaw.sin() * shift);
        let fy = floor(cx, cz)?;
        return Some(([ex, fy, ez], yaw));
    }
    let (px, pz) = if free { (p[0], p[2]) } else { (cx, cz) };
    Some(([px, floor(px, pz)?, pz], yaw))
}

/// Add an entity, or move the player start if there is one already (there's
/// only ever one). Returns its id.
pub fn add_entity(l: &mut Level, kind: EntityKind, pos: [f32; 3], yaw: f32) -> u32 {
    if kind == EntityKind::PlayerStart {
        if let Some(e) = l.entities.iter_mut().find(|e| e.kind == EntityKind::PlayerStart) {
            e.pos = pos;
            e.yaw = yaw;
            return e.id;
        }
    }
    let id = l.next_id();
    l.entities.push(Entity { id, kind, pos, yaw });
    id
}

/// Make `link` to entity `to`, if it's the right kind. Says what happened.
pub fn make_link(l: &mut Level, link: Link, to: u32) -> String {
    let kind = |id: u32| l.entities.iter().find(|e| e.id == id).map(|e| e.kind.clone());
    match (link, kind(to)) {
        (Link::Key(from), Some(EntityKind::Key { name })) => {
            if let Some(e) = l.entities.iter_mut().find(|e| e.id == from) {
                if let EntityKind::Door { key, locked, .. } | EntityKind::Chest { key, locked, .. } = &mut e.kind {
                    *key = Some(to);
                    *locked = true;
                }
            }
            format!("#{from} now opens with {name} (#{to})")
        }
        (Link::Target(from), Some(EntityKind::Door { .. })) => {
            if let Some(EntityKind::Lever { targets }) = l.entities.iter_mut().find(|e| e.id == from).map(|e| &mut e.kind) {
                if !targets.contains(&to) {
                    targets.push(to);
                }
            }
            format!("lever #{from} now works door #{to}")
        }
        (Link::Key(_), _) => "that isn't a key: link cancelled".into(),
        (Link::Target(_), _) => "that isn't a door: link cancelled".into(),
    }
}

/// The links between entities as lines from one position to another: keys to
/// what they open, levers to their doors.
pub fn links(l: &Level) -> Vec<([f32; 3], [f32; 3], Color)> {
    let at = |id: u32| l.entities.iter().find(|e| e.id == id).map(|e| e.pos);
    let mut out = Vec::new();
    for e in &l.entities {
        match &e.kind {
            EntityKind::Door { key: Some(k), .. } | EntityKind::Chest { key: Some(k), .. } => {
                out.extend(at(*k).map(|p| (p, e.pos, Color::srgb(1.0, 0.9, 0.2))));
            }
            EntityKind::Lever { targets } => {
                out.extend(targets.iter().filter_map(|&t| at(t)).map(|p| (e.pos, p, Color::srgb(0.4, 0.8, 1.0))));
            }
            _ => {}
        }
    }
    out
}

/// An entity's outline at `pos` facing `yaw`: a door's leaf in its doorway,
/// otherwise a box on the floor and an arrow the way it faces.
fn ghost(lines: &mut Vec<(Vec3, Vec3, Color)>, kind: &EntityKind, pos: [f32; 3], yaw: f32, c: Color) {
    let base = crate::view3d::to_view(pos);
    // World x/z directions in the view's axes (its z is the world's -z).
    let (s, co) = yaw.sin_cos();
    let fwd = Vec3::new(-s, 0.0, -co);
    let right = Vec3::new(co, 0.0, -s);
    if let EntityKind::Door { width, height, .. } = kind {
        let (w, h) = (*width as f32 * 0.5, *height as f32 * LAYER);
        let q = [base - right * w, base + right * w, base + right * w + Vec3::Y * h, base - right * w + Vec3::Y * h];
        outline(lines, q, c);
        lines.push((base, base + fwd * 0.3, c));
        return;
    }
    let (r, h) = (0.2, 0.4);
    let q = [base - right * r - fwd * r, base + right * r - fwd * r, base + right * r + fwd * r, base - right * r + fwd * r];
    outline(lines, q.map(|v| v + Vec3::Y * 0.01), c);
    outline(lines, q.map(|v| v + Vec3::Y * h), c);
    for v in q {
        lines.push((v, v + Vec3::Y * h, c));
    }
    lines.push((base + Vec3::Y * 0.01, base + Vec3::Y * 0.01 + fwd * 0.45, c));
}

/// How far off a face a new light starts, in tiles.
const LIFT: f32 = 0.25;

/// The surface paint a cell takes when its faces aren't painted: its area's, or the level's.
pub fn surface_at(l: &Level, c: [i32; 3]) -> &Surface {
    // As `Level::to_scene` does: the first area holding the floor of the cell's run.
    let from = l.run_at(c[0], c[1], c[2]).map_or(c[1], |r| r.from);
    l.areas.iter().find(|a| a.contains([c[0], from, c[2]])).and_then(|a| a.surface.as_ref()).unwrap_or(&l.surface)
}

/// The texture slot a face is drawn with.
pub fn face_tex(l: &Level, c: [i32; 3], face: Face) -> u8 {
    l.paint.get(&((c[0], c[1], c[2]), face)).copied().unwrap_or_else(|| {
        let s = surface_at(l, c);
        match face {
            Face::Floor => s.floor,
            Face::Ceiling => s.ceiling,
            Face::Wall(_) => s.wall,
        }
    })
}

/// Paint faces with `slot`, leaving those where it is the default unpainted.
pub fn paint(l: &mut Level, faces: &[FaceRef], slot: u8) {
    for &(c, face) in faces {
        let key = ((c[0], c[1], c[2]), face);
        l.paint.remove(&key);
        if face_tex(l, c, face) != slot {
            l.paint.insert(key, slot);
        }
    }
}

/// Paint the faces joined to this one on its plane that share its texture.
/// Returns how many.
pub fn fill(l: &mut Level, c: [i32; 3], face: Face, slot: u8) -> usize {
    let was = face_tex(l, c, face);
    let steps: [[i32; 3]; 4] = match face {
        Face::Floor | Face::Ceiling => [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]],
        Face::Wall(Dir::E | Dir::W) => [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        Face::Wall(Dir::N | Dir::S) => [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
    };
    let mut seen = std::collections::HashSet::from([c]);
    let mut todo = vec![c];
    let mut found = Vec::new();
    // Big enough for any floor or wall of a sane room.
    while let Some(at) = todo.pop() {
        if found.len() >= 4096 {
            break;
        }
        if !has_face(l, at, face) || face_tex(l, at, face) != was {
            continue;
        }
        found.push((at, face));
        for s in steps {
            let next = [at[0] + s[0], at[1] + s[1], at[2] + s[2]];
            if seen.insert(next) {
                todo.push(next);
            }
        }
    }
    paint(l, &found, slot);
    found.len()
}

/// The box (min, max exclusive) around the cells of `faces`, each from its run's
/// floor to its ceiling.
pub fn cells_box(l: &Level, faces: &BTreeSet<FaceRef>) -> Option<([i32; 3], [i32; 3])> {
    let mut b: Option<([i32; 3], [i32; 3])> = None;
    for &(c, _) in faces {
        let (from, to) = l.run_at(c[0], c[1], c[2]).map_or((c[1], c[1] + 1), |r| (r.from, r.to));
        let (lo, hi) = ([c[0], from, c[2]], [c[0] + 1, to, c[2] + 1]);
        b = Some(match b {
            None => (lo, hi),
            Some((a, z)) => ([0, 1, 2].map(|i| a[i].min(lo[i])), [0, 1, 2].map(|i| z[i].max(hi[i]))),
        });
    }
    b
}

/// The rock cell on the other side of a face.
pub fn behind(c: [i32; 3], face: Face) -> [i32; 3] {
    let [x, l, z] = c;
    match face {
        Face::Floor => [x, l - 1, z],
        Face::Ceiling => [x, l + 1, z],
        Face::Wall(d) => {
            let (dx, dz) = d.delta();
            [x + dx, l, z + dz]
        }
    }
}

/// Whether open cell `c` has this face: rock on its other side (and for a
/// ceiling, no sky).
pub fn has_face(l: &Level, c: [i32; 3], face: Face) -> bool {
    let b = behind(c, face);
    let Some(run) = l.run_at(c[0], c[1], c[2]) else { return false };
    !l.is_open(b[0], b[1], b[2]) && !(face == Face::Ceiling && run.sky)
}

/// What one click takes: a floor or ceiling tile, or a wall's whole height in
/// its column.
pub fn unit(l: &Level, c: [i32; 3], face: Face) -> Vec<FaceRef> {
    if !matches!(face, Face::Wall(_)) || !has_face(l, c, face) {
        return vec![(c, face)];
    }
    let at = |layer: i32| [c[0], layer, c[2]];
    let (mut lo, mut hi) = (c[1], c[1]);
    while has_face(l, at(lo - 1), face) {
        lo -= 1;
    }
    while has_face(l, at(hi + 1), face) {
        hi += 1;
    }
    (lo..=hi).map(|layer| (at(layer), face)).collect()
}

/// The faces like `face` on its plane between cells `a` and `b` (inclusive).
pub fn faces_in(l: &Level, a: [i32; 3], b: [i32; 3], face: Face) -> Vec<FaceRef> {
    let r = |i: usize| a[i].min(b[i])..=a[i].max(b[i]);
    let (xs, ls, zs) = match face {
        Face::Floor | Face::Ceiling => (r(0), a[1]..=a[1], r(2)),
        Face::Wall(Dir::E | Dir::W) => (a[0]..=a[0], r(1), r(2)),
        Face::Wall(Dir::N | Dir::S) => (r(0), r(1), a[2]..=a[2]),
    };
    let mut out = Vec::new();
    for x in xs {
        for layer in ls.clone() {
            for z in zs.clone() {
                if has_face(l, [x, layer, z], face) {
                    out.push(([x, layer, z], face));
                }
            }
        }
    }
    out
}

/// Where the face of cell `c` is after `push` by `k`.
fn moved(c: [i32; 3], face: Face, k: i32) -> [i32; 3] {
    let [x, l, z] = c;
    match face {
        Face::Floor => [x, l + k, z],
        Face::Ceiling => [x, l - k, z],
        Face::Wall(d) => {
            let (dx, dz) = d.delta();
            [x - k * dx, l, z - k * dz]
        }
    }
}

/// Move faces `k` steps along their normals: out into the open cell (closing
/// it) for positive `k`, back into the rock (opening it) for negative. Floors
/// and ceilings move by layers, walls by tiles; a floor keeps its slope.
pub fn push(l: &mut Level, faces: &[FaceRef], k: i32) {
    if k == 0 {
        return;
    }
    for &([x, y, z], face) in faces {
        match face {
            Face::Floor => {
                let slope = l.run_at(x, y, z).and_then(|r| r.slope);
                if k > 0 {
                    l.close(x, z, y, y + k);
                } else {
                    l.open(x, z, y + k, y);
                }
                if let Some(r) = l.run_mut(x, z, y + k) {
                    r.slope = slope;
                }
            }
            Face::Ceiling => {
                if k > 0 {
                    l.close(x, z, y + 1 - k, y + 1);
                } else {
                    l.open(x, z, y + 1, y + 1 - k);
                }
            }
            Face::Wall(d) => {
                let (dx, dz) = d.delta();
                if k > 0 {
                    for i in 0..k {
                        l.close(x - i * dx, z - i * dz, y, y + 1);
                    }
                } else {
                    for i in 1..=-k {
                        l.open(x + i * dx, z + i * dz, y, y + 1);
                    }
                }
            }
        }
    }
}

/// A flight of `n` sloped columns from the floor at `c` toward `dir`, each
/// `slope.rise` layers above the last, keeping the first column's headroom (or
/// its ceiling, if flat). Rock in the way is carved; floors below are raised.
pub fn stairs(l: &mut Level, c: [i32; 3], dir: Dir, n: i32, slope: SlopeDef) -> Result<(), String> {
    let [x, b, z] = c;
    let run = *l.run_at(x, b, z).ok_or("no floor there")?;
    let height = run.to - run.from;
    if height <= slope.rise {
        return Err(format!("only {height} layers of headroom; a rise of {} needs more", slope.rise));
    }
    let (dx, dz) = dir.delta();
    for i in 0..n {
        let (cx, cz) = (x + i * dx, z + i * dz);
        if cx < 0 || cz < 0 {
            break;
        }
        let f = b + slope.rise * i;
        let top = match slope.ceiling {
            CeilingKind::Follow => f + height,
            CeilingKind::Flat => run.to.max(f + slope.rise + 1),
        };
        if i > 0 {
            l.close(cx, cz, b, f);
        }
        l.open(cx, cz, f, top);
        if let Some(r) = l.run_mut(cx, cz, f) {
            r.slope = Some(SlopeDef { dir, ..slope });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::world::level::Run;

    fn room() -> Level {
        let mut l = Level::new("t");
        l.set_box([0, 0, 0], [4, 12, 4], true);
        l
    }

    #[test]
    fn floors_and_ceilings_push_by_layers() {
        let mut l = room();
        // Raise one tile of floor by two layers: a step.
        push(&mut l, &[([1, 0, 1], Face::Floor)], 2);
        assert_eq!(l.runs(1, 1), &[Run { from: 2, to: 12, slope: None, sky: false }]);
        // Dig another down three: a pit.
        push(&mut l, &[([2, 0, 2], Face::Floor)], -3);
        assert_eq!(l.runs(2, 2)[0].from, -3);
        // Lift a ceiling by four.
        push(&mut l, &[([0, 11, 0], Face::Ceiling)], -4);
        assert_eq!(l.runs(0, 0)[0].to, 16);
        // And lower one by one.
        push(&mut l, &[([3, 11, 3], Face::Ceiling)], 1);
        assert_eq!(l.runs(3, 3)[0].to, 11);
    }

    #[test]
    fn walls_push_by_tiles_one_layer_each() {
        let mut l = room();
        let wall = unit(&l, [3, 5, 1], Face::Wall(Dir::E));
        assert_eq!(wall.len(), 12);
        // Push the east wall of one column two tiles into the rock: an alcove.
        push(&mut l, &wall, -2);
        assert!(l.is_open(4, 0, 1) && l.is_open(5, 11, 1) && !l.is_open(6, 5, 1));
        assert!(!l.is_open(4, 0, 2));
        // Pull it back past where it was.
        let wall = unit(&l, [5, 3, 1], Face::Wall(Dir::E));
        push(&mut l, &wall, 3);
        assert!(!l.is_open(3, 6, 1) && l.is_open(2, 6, 1));
        assert!(has_face(&l, moved([5, 6, 1], Face::Wall(Dir::E), 3), Face::Wall(Dir::E)));
    }

    #[test]
    fn floor_pushes_keep_slopes() {
        let mut l = room();
        let s = SlopeDef { dir: Dir::E, rise: 2, kind: SlopeKind::Ramp, ceiling: CeilingKind::Follow };
        l.run_mut(1, 1, 0).unwrap().slope = Some(s);
        push(&mut l, &[([1, 0, 1], Face::Floor)], 3);
        assert_eq!(l.runs(1, 1)[0].slope, Some(s));
        push(&mut l, &[([1, 3, 1], Face::Floor)], -5);
        assert_eq!(l.runs(1, 1)[0], Run { from: -2, to: 12, slope: Some(s), sky: false });
    }

    #[test]
    fn faces_on_a_plane() {
        let mut l = room();
        l.close(2, 2, 0, 3);
        // The floor at layer 0, all but the raised tile.
        assert_eq!(faces_in(&l, [0, 0, 0], [3, 0, 3], Face::Floor).len(), 15);
        // A 2 x 3 patch of the west wall.
        assert_eq!(faces_in(&l, [0, 2, 1], [0, 4, 2], Face::Wall(Dir::W)).len(), 6);
        assert!(!has_face(&l, [1, 0, 1], Face::Ceiling));
    }

    #[test]
    fn flights_climb_and_carve() {
        let mut l = room();
        let s = SlopeDef { dir: Dir::E, rise: 4, kind: SlopeKind::Stairs, ceiling: CeilingKind::Follow };
        // Three columns from the east side of the room into the rock.
        stairs(&mut l, [3, 0, 1], Dir::E, 3, s).unwrap();
        assert_eq!(l.runs(3, 1)[0], Run { from: 0, to: 12, slope: Some(s), sky: false });
        assert_eq!(l.runs(4, 1), &[Run { from: 4, to: 16, slope: Some(s), sky: false }]);
        assert_eq!(l.runs(5, 1), &[Run { from: 8, to: 20, slope: Some(s), sky: false }]);
        // Too little headroom for the rise.
        let mut low = Level::new("t");
        low.open(0, 0, 0, 3);
        assert!(stairs(&mut low, [0, 0, 0], Dir::E, 1, s).is_err());
    }

    #[test]
    fn paint_fills_planes_and_stays_sparse() {
        let mut l = room();
        let floor = Surface::default().floor;
        l.close(2, 2, 0, 3);
        // The floor at layer 0 is 15 tiles; the raised one is on its own plane.
        assert_eq!(fill(&mut l, [0, 0, 0], Face::Floor, 6), 15);
        assert_eq!(face_tex(&l, [3, 0, 3], Face::Floor), 6);
        assert_eq!(face_tex(&l, [2, 3, 2], Face::Floor), floor);
        // Painting back to the default leaves nothing stored.
        let all: Vec<_> = l.paint.keys().map(|&((x, y, z), f)| ([x, y, z], f)).collect();
        paint(&mut l, &all, floor);
        assert!(l.paint.is_empty());
        // An area's surface is the default inside it.
        let s = Surface { floor: 5, ..Surface::default() };
        l.areas.push(Area { name: "a".into(), label: "room".into(), boxes: vec![([0, 0, 0], [2, 12, 2])], surface: Some(s), notes: String::new(), strangeness: None });
        assert_eq!(face_tex(&l, [1, 0, 1], Face::Floor), 5);
        assert_eq!(face_tex(&l, [3, 0, 1], Face::Floor), floor);
        // A wall fill covers the wall's plane, all layers.
        assert_eq!(fill(&mut l, [0, 0, 0], Face::Wall(Dir::W), 2), 4 * 12);
    }

    #[test]
    fn rooms_open_up_or_down() {
        let mut l = Level::new("t");
        open_room(&mut l, 0, (3, 1), (1, 2), 8);
        assert_eq!(l.columns.len(), 6);
        assert_eq!(l.runs(1, 1), &[Run { from: 0, to: 8, slope: None, sky: false }]);
        open_room(&mut l, 0, (1, 1), (1, 1), -4);
        assert_eq!(l.runs(1, 1)[0].from, -4);
    }

    #[test]
    fn slopes_climb_away_from_the_nearest_edge() {
        assert_eq!(edge_dir(0.1, 0.5), Dir::E);
        assert_eq!(edge_dir(0.9, 0.4), Dir::W);
        assert_eq!(edge_dir(0.5, 0.05), Dir::S);
        assert_eq!(edge_dir(0.5, 0.97), Dir::N);
    }

    #[test]
    fn entities_snap_to_cells_and_doors_to_edges() {
        let mut l = room();
        push(&mut l, &[([2, 0, 1], Face::Floor)], 2);
        let scene = l.to_scene();
        let tf = crate::view3d::TF;
        let creature = &entity_kinds()[1].1;
        // Near a corner of cell (2, 1), whose floor is two layers up.
        let (pos, yaw) = place_at(&scene, creature, [2.1 * tf, -200.0, 1.9 * tf], 0.5, false).unwrap();
        assert_eq!(pos, [2.5 * tf, -128.0, 1.5 * tf]);
        assert_eq!(yaw, 0.5);
        let (pos, _) = place_at(&scene, creature, [2.1 * tf, -200.0, 1.9 * tf], 0.5, true).unwrap();
        assert_eq!([pos[0], pos[2]], [2.1 * tf, 1.9 * tf]);
        // A door near the east edge of (0, 0) sits on it facing back west, into the cell.
        let door = &entity_kinds()[2].1;
        let (pos, yaw) = place_at(&scene, door, [0.9 * tf, -10.0, 0.4 * tf], 0.0, false).unwrap();
        assert_eq!(pos, [tf, 0.0, 0.5 * tf]);
        assert!((-yaw.sin() + 1.0).abs() < 1e-6 && yaw.cos().abs() < 1e-6, "{yaw}");
        // Near the north edge: facing south (+z).
        let (pos, yaw) = place_at(&scene, door, [0.5 * tf, -10.0, 0.1 * tf], 0.0, false).unwrap();
        assert_eq!(pos, [0.5 * tf, 0.0, 0.0]);
        assert!((yaw.cos() - 1.0).abs() < 1e-6, "{yaw}");
        // Nothing to stand on in the rock.
        assert!(place_at(&scene, creature, [6.5 * tf, -10.0, 6.5 * tf], 0.0, false).is_none());
    }

    #[test]
    fn one_player_start_and_links_of_the_right_kind() {
        let mut l = room();
        let a = add_entity(&mut l, EntityKind::PlayerStart, [1.0; 3], 0.0);
        let b = add_entity(&mut l, EntityKind::PlayerStart, [2.0; 3], 1.0);
        assert_eq!(a, b);
        assert_eq!(l.entities.iter().filter(|e| e.kind == EntityKind::PlayerStart).count(), 1);
        let door = add_entity(&mut l, entity_kinds()[2].1.clone(), [0.0; 3], 0.0);
        let key = add_entity(&mut l, entity_kinds()[3].1.clone(), [0.0; 3], 0.0);
        let lever = add_entity(&mut l, entity_kinds()[6].1.clone(), [0.0; 3], 0.0);
        assert!(make_link(&mut l, Link::Key(door), lever).contains("isn't a key"));
        make_link(&mut l, Link::Key(door), key);
        let d = l.entities.iter().find(|e| e.id == door).unwrap();
        assert!(matches!(d.kind, EntityKind::Door { key: Some(k), locked: true, .. } if k == key));
        assert!(make_link(&mut l, Link::Target(lever), key).contains("isn't a door"));
        make_link(&mut l, Link::Target(lever), door);
        make_link(&mut l, Link::Target(lever), door);
        let v = l.entities.iter().find(|e| e.id == lever).unwrap();
        assert_eq!(v.kind, EntityKind::Lever { targets: vec![door] });
        assert_eq!(links(&l).len(), 2);
    }
}
