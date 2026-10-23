//! Hand-built levels: the file the editor saves (RON, `levels/*.ron`) and its
//! conversion into a `Scene`, so the renderers, network tokens, walker and game
//! take them exactly like generated ones. See docs/editor.md.
//!
//! A level stores open space; everything else is rock. Across the ground the grid
//! is one tile (`T`); upward it is one layer (`STEP`, a quarter tile). Layer `l`
//! spans y from -(l+1)·STEP to -l·STEP (y points down). Each column keeps its open
//! layers as runs, bottom first; a run's floor can slope and its top can be sky.

use crate::world::creature::Kind;
use crate::world::rng::Rng;
use crate::world::scene::{Light, Room, Rock, Scene, Slope, Span, Strange, StrangeArea, STEP, T};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub const VERSION: u32 = 1;
const TF: f32 = T as f32;

/// Compass directions on the grid: east is +x, south is +z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Dir {
    N,
    E,
    S,
    W,
}

impl Dir {
    pub const ALL: [Dir; 4] = [Dir::N, Dir::E, Dir::S, Dir::W];

    pub fn delta(self) -> (i32, i32) {
        match self {
            Dir::N => (0, -1),
            Dir::E => (1, 0),
            Dir::S => (0, 1),
            Dir::W => (-1, 0),
        }
    }

    pub fn of_delta(d: (i32, i32)) -> Dir {
        match d {
            (0, -1) => Dir::N,
            (1, 0) => Dir::E,
            (0, 1) => Dir::S,
            _ => Dir::W,
        }
    }
}

/// A side of a cell, for paint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Face {
    Floor,
    Ceiling,
    Wall(Dir),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlopeKind {
    Ramp,
    Stairs,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CeilingKind {
    /// Slopes with the floor, keeping the headroom.
    #[default]
    Follow,
    Flat,
}

/// A run's floor climbing `rise` layers (1..=4) across its tile toward `dir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlopeDef {
    pub dir: Dir,
    pub rise: i32,
    pub kind: SlopeKind,
    #[serde(default)]
    pub ceiling: CeilingKind,
}

/// Open layers `from..to` of one column.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub from: i32,
    pub to: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slope: Option<SlopeDef>,
    /// No ceiling: open to the sky.
    #[serde(default, skip_serializing_if = "is_false")]
    pub sky: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Texture slots (0..8) and tint for a floor, its walls and its ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Surface {
    pub floor: u8,
    pub wall: u8,
    pub ceiling: u8,
    pub tint: [f32; 3],
}

impl Default for Surface {
    fn default() -> Self {
        Surface { floor: 3, wall: 0, ceiling: 1, tint: [0.85, 0.8, 0.75] }
    }
}

/// A named part of the level (a room, a corridor, ...): boxes of cells, from
/// `min` to `max` exclusive, each (x, layer, z).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Area {
    pub name: String,
    pub label: String,
    pub boxes: Vec<([i32; 3], [i32; 3])>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<Surface>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
    /// Adds to or pins strangeness inside the area.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strangeness: Option<AreaStrange>,
}

/// What an area does to strangeness inside it (fading out over a tile outside).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AreaStrange {
    /// Adds this much to it.
    Add(f32),
    /// Holds it at this value.
    Pin(f32),
}

impl Area {
    pub fn contains(&self, c: [i32; 3]) -> bool {
        self.boxes.iter().any(|(lo, hi)| (0..3).all(|i| c[i] >= lo[i] && c[i] < hi[i]))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointLight {
    /// World units (y down).
    pub pos: [f32; 3],
    pub colour: [f32; 3],
    pub radius: f32,
    /// The colour it turns where the dungeon is strange; picked from the level's seed if unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strange: Option<[f32; 3]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sun {
    /// Direction the light travels (y down, so a high sun has positive y).
    pub dir: [f32; 3],
    pub colour: [f32; 3],
}

/// How strange each place is: a curve over depth, noise, and painted points
/// (`Level::strange_points`), and what strangeness looks like.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Strangeness {
    /// (layer, strangeness) points, smooth between them and flat beyond.
    pub depth: Vec<(f32, f32)>,
    /// Noise strength, and features per tile.
    pub noise: (f32, f32),
    /// Fog, ambient and sun colours where it is fully strange.
    pub fog: [f32; 3],
    pub ambient: [f32; 3],
    pub sun: [f32; 3],
    pub seed: u64,
    /// Strangeness where the textures start and finish morphing.
    #[serde(default = "default_morph")]
    pub morph: (f32, f32),
    /// Where lights, ambient and sun start and finish turning to their strange colours.
    #[serde(default = "default_lurid")]
    pub lurid: (f32, f32),
    /// Where the fog starts and finishes taking on its strange colour.
    #[serde(default = "default_fog_ramp")]
    pub fog_ramp: (f32, f32),
    /// The hues (in turns, 0..1 round the colour wheel, `to` may pass 1) that lights
    /// without a strange colour of their own drift toward.
    #[serde(default = "default_light_hues")]
    pub light_hues: (f32, f32),
}

fn default_morph() -> (f32, f32) {
    crate::world::scene::MORPH
}

fn default_lurid() -> (f32, f32) {
    crate::world::scene::LURID
}

fn default_fog_ramp() -> (f32, f32) {
    crate::world::scene::FOG_RAMP
}

fn default_light_hues() -> (f32, f32) {
    (0.0, 1.0)
}

impl Default for Strangeness {
    fn default() -> Self {
        Strangeness {
            depth: vec![(0.0, 0.0)],
            noise: (0.12, 0.8),
            fog: [0.45, 0.05, 0.6],
            ambient: [0.5, 0.15, 0.45],
            sun: [0.2, 0.9, 0.6],
            seed: 1,
            morph: default_morph(),
            lurid: default_lurid(),
            fog_ramp: default_fog_ramp(),
            light_hues: default_light_hues(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Hinge {
    Left,
    Right,
    /// Slides up into the frame.
    Up,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropShape {
    Box,
    /// Floor to ceiling.
    Pillar,
    Pyramid,
    /// A box hanging in the air.
    Floating,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EntityKind {
    PlayerStart,
    /// Sits on the cell edge it faces across; `width` in tiles, `height` in layers.
    Door { width: i32, height: i32, hinge: Hinge, open: bool, locked: bool, key: Option<u32> },
    Key { name: String },
    Chest { locked: bool, key: Option<u32>, contents: Vec<String> },
    Creature { kind: Kind, size: f32, seed: u64 },
    /// `size` is (width, height, depth) in world units.
    Prop { shape: PropShape, size: [f32; 3], tex: u8, tint: [f32; 3] },
    Lever { targets: Vec<u32> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,
    /// World units (y down); the floor point it stands on.
    pub pos: [f32; 3],
    /// Radians about the vertical axis.
    #[serde(default)]
    pub yaw: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Level {
    pub version: u32,
    pub name: String,
    /// Which texture set the level was built for: grounded, mid, deep or strange.
    pub textures: String,
    pub ambient: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sun: Option<Sun>,
    /// Fog near and far, in tiles.
    pub fog: (f32, f32),
    /// Paint and tint wherever no area or face paint says otherwise.
    pub surface: Surface,
    pub strangeness: Strangeness,
    /// Open runs of each column (x, z), bottom first.
    pub columns: BTreeMap<(i32, i32), Vec<Run>>,
    /// Texture slot of single faces, keyed by cell (x, layer, z).
    #[serde(default)]
    pub paint: BTreeMap<((i32, i32, i32), Face), u8>,
    /// Painted strangeness offsets on the lattice (x, up, z), one tile apart.
    #[serde(default)]
    pub strange_points: BTreeMap<(i32, i32, i32), f32>,
    #[serde(default)]
    pub areas: Vec<Area>,
    #[serde(default)]
    pub lights: Vec<PointLight>,
    #[serde(default)]
    pub entities: Vec<Entity>,
}

impl Level {
    pub fn new(name: &str) -> Level {
        Level {
            version: VERSION,
            name: name.into(),
            textures: "grounded".into(),
            ambient: [0.22, 0.2, 0.2],
            sun: None,
            fog: (2.0, 7.0),
            surface: Surface::default(),
            strangeness: Strangeness::default(),
            columns: BTreeMap::new(),
            paint: BTreeMap::new(),
            strange_points: BTreeMap::new(),
            areas: Vec::new(),
            lights: Vec::new(),
            entities: Vec::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Level, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Level::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Level, String> {
        let level: Level = ron::from_str(text).map_err(|e| e.to_string())?;
        if level.version > VERSION {
            return Err(format!("made by a newer editor (version {} > {VERSION})", level.version));
        }
        Ok(level)
    }

    pub fn to_ron(&self) -> String {
        let pretty = ron::ser::PrettyConfig::new().depth_limit(2).struct_names(true);
        ron::ser::to_string_pretty(self, pretty).expect("levels always serialise")
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        // Write then rename, so a crash never leaves half a level.
        let tmp = path.with_extension("ron.tmp");
        std::fs::write(&tmp, self.to_ron())?;
        std::fs::rename(tmp, path)
    }

    pub fn runs(&self, x: i32, z: i32) -> &[Run] {
        self.columns.get(&(x, z)).map_or(&[], |r| r.as_slice())
    }

    /// The run holding layer `l` of column (x, z).
    pub fn run_at(&self, x: i32, l: i32, z: i32) -> Option<&Run> {
        self.runs(x, z).iter().find(|r| l >= r.from && l < r.to)
    }

    pub fn is_open(&self, x: i32, l: i32, z: i32) -> bool {
        self.run_at(x, l, z).is_some()
    }

    /// Open layers `from..to` of column (x, z), joining any runs they touch. The
    /// joined run keeps the slope of its lowest floor and the sky of its top.
    pub fn open(&mut self, x: i32, z: i32, from: i32, to: i32) {
        if from >= to || x < 0 || z < 0 {
            return;
        }
        let runs = self.columns.entry((x, z)).or_default();
        let mut new = Run { from, to, slope: None, sky: false };
        runs.retain(|r| {
            let touches = r.from <= new.to && r.to >= new.from;
            if touches {
                if r.from <= new.from {
                    new.slope = r.slope;
                }
                if r.to >= new.to {
                    new.sky = r.sky;
                }
                new.from = new.from.min(r.from);
                new.to = new.to.max(r.to);
            }
            !touches
        });
        runs.push(new);
        runs.sort_by_key(|r| r.from);
    }

    /// Fill layers `from..to` of column (x, z) with rock.
    pub fn close(&mut self, x: i32, z: i32, from: i32, to: i32) {
        let Some(runs) = self.columns.get_mut(&(x, z)) else { return };
        let mut out = Vec::new();
        for r in runs.drain(..) {
            if r.to <= from || r.from >= to {
                out.push(r);
                continue;
            }
            if r.from < from {
                out.push(Run { from: r.from, to: from, slope: r.slope, sky: false });
            }
            if r.to > to {
                out.push(Run { from: to, to: r.to, slope: None, sky: r.sky });
            }
        }
        if out.is_empty() {
            self.columns.remove(&(x, z));
        } else {
            *runs = out;
        }
    }

    /// Open (or with `open` false, fill) every cell of the box `min..max` (x, layer, z).
    pub fn set_box(&mut self, min: [i32; 3], max: [i32; 3], open: bool) {
        for x in min[0]..max[0] {
            for z in min[2]..max[2] {
                if open {
                    self.open(x, z, min[1], max[1]);
                } else {
                    self.close(x, z, min[1], max[1]);
                }
            }
        }
    }

    /// The run whose floor is at layer `from`, to change its slope or sky.
    pub fn run_mut(&mut self, x: i32, z: i32, from: i32) -> Option<&mut Run> {
        self.columns.get_mut(&(x, z))?.iter_mut().find(|r| r.from == from)
    }

    pub fn player_start(&self) -> Option<([f32; 3], f32)> {
        self.entities.iter().find(|e| e.kind == EntityKind::PlayerStart).map(|e| (e.pos, e.yaw))
    }

    pub fn next_id(&self) -> u32 {
        self.entities.iter().map(|e| e.id + 1).max().unwrap_or(1)
    }

    /// Two storeys joined by stairs, with a bridge over a pit and a courtyard under the sky.
    pub fn demo() -> Level {
        let mut l = Level::new("demo");
        // Lower hall, 6 x 6 tiles, 3 tiles high.
        l.set_box([0, 0, 0], [6, 12, 6], true);
        // A pit in the middle, two tiles deep, crossed by a bridge one tile wide.
        l.set_box([2, -8, 1], [4, 0, 5], true);
        l.set_box([2, -8, 3], [4, 0, 4], false);
        l.open(2, 3, 0, 12);
        l.open(3, 3, 0, 12);
        l.close(2, 3, -8, 0);
        l.close(3, 3, -8, 0);
        // Stairs up the east side: four columns climbing a tile each (4 layers).
        for i in 0..4 {
            let x = 6 + i;
            l.open(x, 2, 4 * i, 4 * i + 12);
            l.run_mut(x, 2, 4 * i).unwrap().slope =
                Some(SlopeDef { dir: Dir::E, rise: 4, kind: SlopeKind::Stairs, ceiling: CeilingKind::Follow });
        }
        // Upper room over the far end, floor at layer 16, partly over a lower room.
        l.set_box([10, 16, 0], [14, 28, 5], true);
        l.set_box([10, 0, 0], [14, 8, 5], true);
        // Courtyard under the sky beside the upper room.
        l.set_box([10, 16, 5], [14, 24, 8], true);
        for x in 10..14 {
            for z in 5..8 {
                l.run_mut(x, z, 16).unwrap().sky = true;
            }
        }
        l.lights.push(PointLight { pos: [3.0 * TF, -2.5 * TF, 3.0 * TF], colour: [1.0, 0.8, 0.6], radius: 3.0 * TF, strange: None });
        l.entities.push(Entity { id: 1, kind: EntityKind::PlayerStart, pos: [1.5 * TF, 0.0, 1.5 * TF], yaw: 0.0 });
        l.entities.push(Entity {
            id: 2,
            kind: EntityKind::Creature { kind: Kind::Humanoid, size: 1.0, seed: 5 },
            pos: [4.5 * TF, 0.0, 0.5 * TF],
            yaw: 2.5,
        });
        l.entities.push(Entity {
            id: 3,
            kind: EntityKind::Prop { shape: PropShape::Pillar, size: [80.0, 0.0, 80.0], tex: 7, tint: [0.8, 0.75, 0.7] },
            pos: [0.5 * TF, 0.0, 4.5 * TF],
            yaw: 0.3,
        });
        // A locked door at the foot of the stairs with its key in the hall, and a
        // portcullis at the top that a lever by the stairs raises.
        let west = std::f32::consts::FRAC_PI_2;
        l.entities.push(Entity {
            id: 4,
            kind: EntityKind::Door { width: 1, height: 8, hinge: Hinge::Left, open: false, locked: true, key: Some(5) },
            pos: [6.0 * TF, 0.0, 2.5 * TF],
            yaw: west,
        });
        l.entities.push(Entity { id: 5, kind: EntityKind::Key { name: "iron key".into() }, pos: [1.5 * TF, 0.0, 3.5 * TF], yaw: 0.0 });
        l.entities.push(Entity {
            id: 6,
            kind: EntityKind::Door { width: 1, height: 8, hinge: Hinge::Up, open: false, locked: false, key: None },
            pos: [10.0 * TF, -16.0 * STEP as f32, 2.5 * TF],
            yaw: west,
        });
        l.entities.push(Entity { id: 7, kind: EntityKind::Lever { targets: vec![6] }, pos: [5.5 * TF, 0.0, 1.2 * TF], yaw: west });
        // A band of different wall along the hall's west side, two layers high.
        for z in 0..6 {
            for layer in 4..6 {
                l.paint.insert(((0, layer, z), Face::Wall(Dir::W)), 6);
            }
        }
        // The pit grows strange toward its bottom, and the room under the upper room glows.
        l.strangeness.depth = vec![(-8.0, 0.9), (0.0, 0.05), (16.0, 0.0)];
        for x in 10..15 {
            for z in 0..6 {
                l.strange_points.insert((x, 1, z), 0.8);
            }
        }
        l
    }

    /// Strangeness as the scene sees it, before any painted points.
    fn strange(&self) -> Strange {
        let s = &self.strangeness;
        Strange {
            base: 0.0,
            grad: [0.0; 2],
            centre: [0.0; 2],
            bumps: Vec::new(),
            depth: s.depth.clone(),
            points: self.strange_points.iter().map(|(&(x, y, z), &v)| ([x, y, z], v)).collect(),
            noise: s.noise,
            seed: s.seed,
            fog: s.fog,
            ambient: s.ambient,
            sun: s.sun,
            morph: s.morph,
            lurid: s.lurid,
            fog_ramp: s.fog_ramp,
            areas: self
                .areas
                .iter()
                .filter_map(|a| Some((a, a.strangeness?)))
                .flat_map(|(a, mode)| {
                    a.boxes.iter().map(move |&(lo, hi)| StrangeArea {
                        lo: [lo[0] as f32 * TF, -hi[1] as f32 * STEP as f32, lo[2] as f32 * TF],
                        hi: [hi[0] as f32 * TF, -lo[1] as f32 * STEP as f32, hi[2] as f32 * TF],
                        mode,
                    })
                })
                .collect(),
        }
    }

    /// The level as a scene: geometry, lights, strangeness, creatures, props, and
    /// doors, keys and levers (`Scene::things`).
    pub fn to_scene(&self) -> Scene {
        let w = self.columns.keys().map(|k| k.0 + 1).max().unwrap_or(1);
        let d = self.columns.keys().map(|k| k.1 + 1).max().unwrap_or(1);
        let top = self.columns.values().flatten().map(|r| r.to).max().unwrap_or(4);
        let room = |s: &Surface, b: Option<&([i32; 3], [i32; 3])>| {
            let (lo, hi) = b.copied().unwrap_or(([0; 3], [w, top, d]));
            Room {
                x: lo[0],
                z: lo[2],
                w: hi[0] - lo[0],
                d: hi[2] - lo[2],
                height: ((hi[1] - lo[1]) * STEP + T - 1) / T,
                floor: -lo[1] * STEP,
                floor_tex: s.floor,
                wall_tex: s.wall,
                ceil_tex: s.ceiling,
                tint: s.tint,
            }
        };
        let mut rooms = vec![room(&self.surface, None)];
        for a in &self.areas {
            rooms.push(room(a.surface.as_ref().unwrap_or(&self.surface), a.boxes.first()));
        }

        let mut cols = vec![Vec::new(); (w * d) as usize];
        for (&(x, z), runs) in &self.columns {
            if x < 0 || z < 0 {
                continue;
            }
            let col = &mut cols[(z * w + x) as usize];
            // Scene spans run top first.
            for r in runs.iter().rev() {
                let area = self.areas.iter().position(|a| a.contains([x, r.from, z]));
                col.push(span_of(r, area.map_or(0, |i| i + 1)));
            }
        }

        let mut rng = Rng::new(self.strangeness.seed);
        let lights = self
            .lights
            .iter()
            .map(|l| Light {
                pos: l.pos,
                col: l.colour,
                weird: l.strange.unwrap_or_else(|| {
                    let (from, to) = self.strangeness.light_hues;
                    crate::world::scene::hue(from + (to - from) * rng.f(), rng.fr(0.75, 1.0), rng.fr(0.8, 1.5))
                }),
                radius: l.radius,
            })
            .collect();
        let (sun_dir, sun_col) = match &self.sun {
            Some(s) => (crate::world::scene::norm(s.dir), s.colour),
            None => ([0.0, 1.0, 0.0], [0.0; 3]),
        };
        let mut scene = Scene::empty(w, d, rooms, cols, Rock { height: top / 4 + 1, tex: self.surface.wall });
        scene.lights = lights;
        scene.sun_dir = sun_dir;
        scene.sun_col = sun_col;
        scene.ambient = self.ambient;
        scene.strange = self.strange();
        scene.paint = self.paint.iter().map(|(&((x, l, z), f), &t)| (([x, l, z], f), t)).collect();
        scene.build_level();

        for e in &self.entities {
            let [x, y, z] = e.pos;
            let Some(n) = scene.node_at(x, y, z) else { continue };
            let floor = scene.node_floor(n, [x, z]);
            let ceiling = if scene.span(n).sky { floor - 6.0 * TF } else { scene.node_ceiling(n, [x, z]) };
            match &e.kind {
                EntityKind::Creature { kind, size, seed } => {
                    let mut rng = Rng::new(*seed);
                    scene.grow_creature(&mut rng, [x, floor, z], floor - ceiling, Some(*kind), *size, Some(e.yaw));
                    scene.span_mut(n).blocked = true;
                }
                EntityKind::Prop { shape, size, tex, tint } => {
                    let half = [size[0] / 2.0, size[2] / 2.0];
                    match shape {
                        PropShape::Box => scene.add_box([x, floor, z], half, size[1], e.yaw, *tex, *tint),
                        PropShape::Pillar => scene.add_box([x, floor, z], half, floor - ceiling, e.yaw, *tex, *tint),
                        PropShape::Pyramid => scene.add_pyramid([x, floor, z], half[0], size[1], *tex, *tint),
                        PropShape::Floating => scene.add_box([x, y, z], half, size[1], e.yaw, *tex, *tint),
                    }
                    scene.span_mut(n).blocked = true;
                }
                // A stand-in until chests get their own mesh.
                EntityKind::Chest { .. } => {
                    scene.add_box([x, floor, z], [0.3 * TF, 0.2 * TF], 0.35 * TF, e.yaw, 2, [0.7, 0.55, 0.4]);
                    scene.span_mut(n).blocked = true;
                }
                EntityKind::PlayerStart | EntityKind::Door { .. } | EntityKind::Key { .. } | EntityKind::Lever { .. } => {}
            }
        }
        scene.build_things(&self.entities);
        scene
    }
}

/// A run as a scene span.
fn span_of(r: &Run, room: usize) -> Span {
    let (fy, cy) = (-r.from * STEP, -r.to * STEP);
    let mut s = Span {
        room,
        y0: fy,
        y1: fy,
        c0: cy,
        c1: cy,
        axis: 0,
        slope: Slope::Flat,
        sky: r.sky,
        blocked: false,
        reserved: false,
    };
    if let Some(sl) = r.slope {
        let rise = sl.rise.clamp(1, 4) * STEP;
        s.axis = matches!(sl.dir, Dir::N | Dir::S) as u8;
        s.slope = match sl.kind {
            SlopeKind::Ramp => Slope::Ramp,
            SlopeKind::Stairs => Slope::Stairs,
        };
        // The floor climbs toward `dir`; spans measure from the low (-x / -z) edge.
        let up_high = matches!(sl.dir, Dir::E | Dir::S);
        let (lo_f, hi_f) = if up_high { (fy, fy - rise) } else { (fy - rise, fy) };
        s.y0 = lo_f;
        s.y1 = hi_f;
        if sl.ceiling == CeilingKind::Follow {
            // Keep the headroom: lower the ceiling over the low end of the slope.
            let (lo_c, hi_c) = if up_high { (cy + rise, cy) } else { (cy, cy + rise) };
            s.c0 = lo_c;
            s.c1 = hi_c;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walking by hand (as in the game) up the steep stairs and back down, once
    /// the doors on the way are open.
    #[test]
    fn player_climbs_stairs() {
        crate::world::flesh::set_psx(true);
        let mut scene = Level::demo().to_scene();
        let mut rng = Rng::new(1);
        // In the hall by the foot of the stairs, facing east up them.
        let east = -std::f32::consts::FRAC_PI_2;
        let at = [5.5 * TF, 0.0, 2.5 * TF];
        let mut w = crate::world::walker::Walker::at(&scene, at, east, &mut rng).unwrap();
        for _ in 0..50 {
            w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        }
        assert!(w.x < 6.0 * TF, "the locked door is in the way: x {}", w.x);
        assert_eq!(scene.use_thing(at, east, &[]).as_deref(), Some("locked"));
        let keys = scene.take_pickups([1.5 * TF, 0.0, 3.5 * TF]);
        assert_eq!(scene.use_thing(at, east, &keys).as_deref(), Some("unlocked with the iron key"));
        // The lever, to the north, raises the portcullis at the top.
        assert!(scene.use_thing(at, std::f32::consts::PI, &keys).unwrap().contains("lever"));
        for _ in 0..30 {
            scene.tick_things(0.1);
        }
        for _ in 0..200 {
            w.drive(&scene, 1.0, 0.0, 0.0, 0.0);
        }
        let feet = w.y + crate::world::walker::EYE;
        assert!(w.x > 10.0 * TF && (feet + 16.0 * STEP as f32).abs() < 20.0, "reached the upper room: x {} feet {feet}", w.x);
        for _ in 0..200 {
            w.drive(&scene, -1.0, 0.0, 0.0, 0.0);
        }
        let feet = w.y + crate::world::walker::EYE;
        assert!(w.x < 6.0 * TF && feet.abs() < 20.0, "back down in the hall: x {} feet {feet}", w.x);
    }

    #[test]
    fn runs_join_and_split() {
        let mut l = Level::new("t");
        l.open(0, 0, 0, 4);
        l.open(0, 0, 8, 12);
        l.open(0, 0, 4, 8);
        assert_eq!(l.runs(0, 0).len(), 1);
        assert_eq!((l.runs(0, 0)[0].from, l.runs(0, 0)[0].to), (0, 12));
        l.close(0, 0, 4, 6);
        let r: Vec<_> = l.runs(0, 0).iter().map(|r| (r.from, r.to)).collect();
        assert_eq!(r, vec![(0, 4), (6, 12)]);
        l.close(0, 0, -10, 20);
        assert!(l.columns.is_empty());
    }

    #[test]
    fn saves_and_loads() {
        let l = Level::demo();
        let back = Level::parse(&l.to_ron()).unwrap();
        assert_eq!(l, back);
    }

    /// Levels saved before the look settings get the ramps generated levels use.
    #[test]
    fn old_strangeness_loads() {
        let old = "(depth: [(0.0, 0.2)], noise: (0.12, 0.8), fog: (0.4, 0.0, 0.6), ambient: (0.5, 0.1, 0.4), sun: (0.2, 0.9, 0.6), seed: 3)";
        let s: Strangeness = ron::from_str(old).unwrap();
        let d = Strangeness::default();
        assert_eq!((s.morph, s.lurid, s.fog_ramp, s.light_hues), (d.morph, d.lurid, d.fog_ramp, d.light_hues));
    }

    /// An area pins strangeness inside it, fading out over a tile outside, and
    /// another adds to the depth curve.
    #[test]
    fn areas_pin_and_add() {
        let mut l = Level::new("t");
        for x in 0..12 {
            l.open(x, 0, 0, 12);
        }
        l.strangeness.noise.0 = 0.0;
        l.strangeness.depth = vec![(0.0, 0.1)];
        l.areas.push(Area { name: "a".into(), label: "room".into(), boxes: vec![([0, 0, 0], [3, 12, 1])], surface: None, notes: String::new(), strangeness: Some(AreaStrange::Pin(0.8)) });
        l.areas.push(Area { name: "b".into(), label: "room".into(), boxes: vec![([8, 0, 0], [10, 12, 1])], surface: None, notes: String::new(), strangeness: Some(AreaStrange::Add(0.3)) });
        let scene = l.to_scene();
        let at = |x: f32| scene.strangeness([x * TF, -STEP as f32, 0.5 * TF]);
        assert!((at(1.5) - 0.8).abs() < 1e-4, "pinned inside: {}", at(1.5));
        assert!((at(3.0) - 0.8).abs() < 1e-4, "pinned on its wall: {}", at(3.0));
        assert!(at(3.5) < 0.8 && at(3.5) > 0.1, "fading outside: {}", at(3.5));
        assert!((at(5.5) - 0.1).abs() < 1e-4, "the depth curve beyond: {}", at(5.5));
        assert!((at(9.0) - 0.4).abs() < 1e-4, "added inside: {}", at(9.0));
    }

    /// The textures morph only across the level's morph range.
    #[test]
    fn morph_range() {
        let mut l = Level::new("t");
        l.open(0, 0, 0, 12);
        l.strangeness.noise.0 = 0.0;
        l.strangeness.depth = vec![(0.0, 0.5)];
        let p = [0.5 * TF, -STEP as f32, 0.5 * TF];
        assert!((l.to_scene().morph(p) - 0.5).abs() < 1e-4);
        l.strangeness.morph = (0.4, 0.6);
        assert!((l.to_scene().morph(p) - 0.5).abs() < 1e-4);
        l.strangeness.morph = (0.6, 0.9);
        assert_eq!(l.to_scene().morph(p), 0.0);
        l.strangeness.morph = (0.1, 0.3);
        assert_eq!(l.to_scene().morph(p), 1.0);
    }

    /// The walker can get from the lower hall to the upper room by the stairs,
    /// and across the bridge without falling into the pit.
    #[test]
    fn storeys_connect() {
        crate::world::flesh::set_psx(true);
        let scene = Level::demo().to_scene();
        let start = scene.node_at(1.5 * TF, 0.0, 1.5 * TF).unwrap();
        let reach: Vec<_> = scene.reachable(start).into_iter().map(|e| e.0).collect();
        let upper = scene.node_at(12.5 * TF, -16.0 * STEP as f32, 2.5 * TF).unwrap();
        let below = scene.node_at(12.5 * TF, 0.0, 2.5 * TF).unwrap();
        assert_ne!(upper, below, "the upper room and the room under it are separate spans");
        assert!(reach.contains(&upper), "upper room reachable by the stairs");
        let bridge = scene.node_at(2.5 * TF, 0.0, 3.5 * TF).unwrap();
        assert!(reach.contains(&bridge), "bridge reachable");
        assert!(scene.tris.len() > 100);
    }
}
