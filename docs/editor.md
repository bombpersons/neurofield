# Level editor

A Bevy + egui editor for building levels by hand. The levels are for playing,
and for studying later: we'll pick out what makes them good and write a
generator that makes more like them for training data.

## Goals

- Build King's Field-style dungeons quickly, on a grid. That covers rooms,
  corridors, stairs, pits, platforms and bridges.
- Hand-built levels drop into everything we already have: the PS1 renderer,
  the full-colour creature pass, network tokens, the game and datagen. None
  of those should need a special case.
- The file format is easy to read in code and easy to diff, because the
  levels are also data to analyse.
- Playing is one click away, and you can start from wherever the camera is.

Not goals for now: game logic beyond what levels need to store (combat,
inventory), multiplayer editing, or arbitrary meshes.

## Units and axes

The same as datagen: one **tile** is `T = 256` units, and **y points down**,
as on the PS1.

- Across the ground (x and z) the grid is one tile.
- Vertically the grid is one **layer**, which is 64 units (`STEP`). Four layers
  make one tile.
- A **cell** is one tile × one layer × one tile. It is addressed as
  `(x, layer, z)`, with layer 0 at y = 0 and layers counting upward.
  Layer `l` covers `y ∈ [-(l+1)·64, -l·64]`.

**The dungeon is one map.** All its floors, and the stairs and shafts between
them, live in a single file. A dungeon can be up to 256 × 256 tiles across and
1024 layers (256 tiles) deep. It's stored sparsely, so only what's built takes
space, and the game and editor work on it in chunks of 16 × 16 tiles (see
Shared code).

## The level model

### Space, not blocks

A level stores the **open** cells: the air you can be in. Everything else is
solid rock. Geometry is drawn wherever an open cell meets a solid one, facing
into the open cell. This suits dungeons, where rooms are carved out of rock,
and it makes every editing operation simple:

| You want | You do |
|---|---|
| A room | Open a box of cells |
| A raised floor or platform | Close the bottom layers of some columns |
| A pit | Open cells below the floor |
| A bridge you can walk under | Leave solid cells with open cells above and below |
| A low ceiling or beam | Close the top layers |
| A window between rooms | Open cells through a wall |

Outdoor areas have a **sky** flag on the top run of a column. Open cells at the top of a
sky column draw no ceiling, and the solid around them is drawn as raised
blocks, the way the generator does outdoors now.

### Slopes

On its own the grid can't make a slope, so a column can have a **slope** where
its floor (the top of its solid) meets open space:

```
Slope { dir: N|E|S|W, rise: 1..=4 layers, kind: Ramp | Stairs, ceiling: Follow | Flat }
```

The floor rises by `rise` layers across the tile in direction `dir`. A rise of
4 is a full tile per tile: steep stairs. Stairs use the generator's look of
four treads per tile, each `rise × 16` units high.
The cells the slope passes through must be open.

**The ceiling follows the slope.** A sloped column's ceiling (the bottom of the
solid above its open run) slopes by the same rise in the same direction. A
staircase corridor then keeps the same headroom all the way up, as in the
generator now. A slope can set `ceiling: Flat` instead, for stairs climbing
up the side of a tall room.

### Paint

Every face can be painted with a texture slot (0–7), a quarter-turn rotation,
and a tile-aligned offset. Paint is keyed by `(cell, face direction)`.
Unpainted faces take their **area's** floor, wall and ceiling defaults
(see Areas), or the level's defaults if they aren't in an area. In practice
you paint an area once and touch up single faces.

Texture slots mean the same as in datagen. The level's texture set (grounded,
mid, deep or strange) is chosen per level, and strangeness morphs it as it
does now.

### Strangeness

Strangeness at a point is built in layers, each tunable in the editor:

1. **A depth curve.** Strangeness against depth (layer), edited as a curve with
   points you can drag. A dungeon can stay grounded for its first few floors,
   then slide. This is the main dial for the whole dungeon.
2. **Noise.** Its strength and scale, so the depth curve isn't perfectly flat
   on each floor.
3. **Painted aura.** Offsets stored on a **lattice of grid points**: one point
   every tile across the ground and every tile up, at cell corners. It's sparse
   and unset points are 0. Between points it's interpolated smoothly
   (trilinear, then smoothstep), so painting with a soft brush gives soft
   auras. Paint pushes up (an infected room) or down (a spared one).
4. **Area overrides.** An area can add to, or pin, strangeness inside it.

The sum is clamped to 0–1. Everything that uses strangeness now uses this field
instead of the generated one: triangle column 34, texture steps, fog colour,
and light colour.

**How strangeness looks** is tunable too, per dungeon:
- the strange fog colour range;
- the hue range the lights drift toward;
- the strangeness at which the texture morph starts and ends.

Defaults come from what datagen does now.


### Areas

An area is a named, labelled box of cells (or several boxes). It holds:

- a label: `room`, `corridor`, `shrine`, `stairwell`, `outdoors`, or anything
  else you type;
- default floor, wall and ceiling paint;
- a colour tint, ambient light and fog (near, far);
- notes.

Areas are mostly for the analysis afterwards. They tell the future generator
what a room is and what a corridor is, and how they connect. They're also the
quickest way to paint and light whole rooms.

### Lights

- **Level-wide:** ambient colour, sun direction and colour (it only matters
  outdoors), and fog defaults.
- **Point lights:** position, colour, radius, and an optional *strange*
  colour that the light shifts toward as strangeness rises. If no strange
  colour is set, one is picked from a seed, the way the generator does it.

Lights are baked into vertex colours exactly as `Scene::shade` does now.

### Entities

Each entity has a stable `id`, a `kind`, a position (snapped to the grid by
default but stored as floats), a yaw, and properties that depend on its kind:

| Kind | Properties |
|---|---|
| `PlayerStart` | (exactly one) |
| `Door` | Sits on a cell face between two open cells. Width in tiles, height in layers, `hinge` (left, right, or slides up), `open` (starting state), `locked`, `key` (an entity id) |
| `Key` | `name` (shown in the UI) |
| `Chest` | `contents` (entity ids or item names), `locked`, `key` |
| `Creature` | `kind` (humanoid, beast, blob, floater, mass), `size`, `seed`, an optional skin motif, `extra_parts` |
| `Prop` | `shape` (box, pillar, pyramid, floating cube, …), size, texture, tint |
| `Light` | as in Lights above |
| `Lever` | `targets` (entity ids) |

Links (key → door, lever → targets) are entity ids. The
viewport draws them as lines, and the Properties window edits them with a
picker. Creatures are grown from `kind`, `size` and `seed` by the same code as
`build_creatures`. The editor shows the result and has a *re-roll* button that
picks a new seed.

### Doors

Doors open and close in the game. A door is its own small mesh, a slab with a
frame, not part of the wall:
- **Opening:** it swings on its hinge or slides up over about half a second.
  Its triangles are rebuilt each frame while it moves.
- **Collision:** a closed or moving door blocks the doorway, and an open door
  doesn't.
- **Use key:** with the doorway in reach, it toggles the door.
- **Locked doors:** these need their key. Walking over a key picks it up, the
  status line names the keys you hold, and a locked door you have no key for
  rattles and stays shut. A lever toggles its target doors.

Door states aren't saved in the level file: every play starts from each door's
`open` setting.

## File format

RON, at `levels/<name>.ron`, committed to the repo. It's readable and diffs
well, and Python can parse it with a small converter, or we can also write
JSON when analysing.

```ron
Level(
    version: 1,
    name: "demo",
    textures: "grounded",
    ambient: (0.22, 0.2, 0.2),
    fog: (2.0, 7.0),                              // near, far in tiles
    surface: Surface(floor: 3, wall: 0, ceiling: 1, tint: (0.85, 0.8, 0.75)),
    strangeness: Strangeness(
        depth: [(-8.0, 0.9), (0.0, 0.05), (16.0, 0.0)],   // (layer, value)
        noise: (0.12, 0.8),                                // strength, features per tile
        fog: (0.45, 0.05, 0.6), ambient: (0.5, 0.15, 0.45), sun: (0.2, 0.9, 0.6),
        seed: 1,
    ),
    // Open layers of each column (x, z), bottom first.
    columns: {
        (3, 4): [Run(from: 0, to: 12)],
        (6, 2): [Run(from: 0, to: 12, slope: Some(SlopeDef(dir: E, rise: 4, kind: Stairs, ceiling: Follow)))],
        (10, 2): [Run(from: 0, to: 8), Run(from: 16, to: 28)],
        (11, 6): [Run(from: 16, to: 24, sky: true)],
    },
    paint: { ((0, 4, 2), Wall(W)): 6 },          // (cell (x, layer, z), face): texture slot
    strange_points: { (12, 1, 2): 0.8 },          // lattice (x, up, z) in tiles: offset
    areas: [ Area(name: "entry hall", label: "room", boxes: [((0, 0, 0), (6, 12, 6))]) ],
    lights: [ PointLight(pos: (768.0, -640.0, 768.0), colour: (1.0, 0.8, 0.6), radius: 768.0) ],
    entities: [
        Entity(id: 1, kind: PlayerStart, pos: (384.0, 0.0, 384.0), yaw: 0.0),
        Entity(id: 2, kind: Creature(kind: Humanoid, size: 1.0, seed: 5), pos: (1152.0, 0.0, 128.0), yaw: 2.5),
    ],
)
```

Open space is stored as runs per column, so the file is small and a column
reads like a slice through the level. A run's slope and sky belong to the run,
so one column can hold a flat cellar under a staircase under the sky.
`datagen level-demo --out levels/demo.ron` writes the example level that the
tests use. `version` gets bumped with a migration
function whenever the format changes, so old levels keep loading.

## Shared code: `engine::world::level`

The file format and its conversion live in the engine (`engine::world`), so datagen, the game and
the editor all share them.

- **`Level`** is the serde types above, with load and save.
- **`Grid`** is open-cell occupancy with the queries the walker and the game
  need. These replace the single-floor-per-cell queries:
  - `floor_below(x, y, z)` and `ceiling_above(x, y, z)`
  - `can_step(a, b)` (up to `MAX_STEP`, ledges no)
  - `reachable(start)` for the validation overlay and walker planning
  - `line_clear(a, b)`
- **`Level::to_scene(region)`** builds the `Scene` everything downstream uses,
  for one region of chunks:
  - triangles, the same per-tile quads and per-tread stairs the generator
    makes, so the network sees the same kind of geometry;
  - baked lights;
  - the strangeness field;
  - creatures grown from their entities;
  - props;
  - doors, each with its own range of triangles so it can move.

The generator then moves onto the same `Grid` (its cells map straight across),
so there's one set of collision code. That's the riskiest refactor here, so it
comes first and keeps the existing walker test passing.

**Chunks.** A whole dungeon is too big to turn into one `Scene` and draw
every frame. That's especially true on the laptop, since the PS1 renderer
draws every triangle it's given. So the game and editor keep a `Scene` for the
3 × 3 chunks around the camera, across all floors, and rebuild the edge as
you cross into a new chunk. Baking runs on a background thread. Triangles
across a chunk boundary must come out the same whichever region they're
built in, so lighting looks at lights in neighbouring chunks too. Re-baking
strangeness while dragging a slider (as in the game now) only touches the
live region.

**Triangle budget.** The network sees the 512 nearest triangles. Rooms the
size of the generator's (up to 6 × 6 × 4 tiles) are fine. A huge hall would
spend its budget on near floor tiles. The status bar shows the count for the
current view, so you'll notice.

## The editor app

There will be a new crate, `editor/`: Bevy 0.20, bevy_egui, and `egui_dock` for
dockable windows. (We need to check that an `egui_dock` release matches
bevy_egui 0.43's egui: egui_dock 0.21 does.) It runs with `cargo run -p editor [-- levels/x.ron]`.

### Theme

Compact and dark, with 12 px body text, tight spacing and small buttons, so the
viewport gets most of the window on the laptop.

### Layout

The 3D view fills the whole window. Everything else is egui drawn over it: the
menu bar, the status bar, and panels docked at the edges or floating. The centre
of the dock layout is a see-through Viewport tab: where the pointer is over it,
the mouse drives the 3D camera. The PS1 view mode fills the window the same way.

```
┌ File  Edit  View  Tools  Play ─────────────────────────── ▶ Play  ▶ From here ┐
│ ┌ Tools ┐ ┌ Viewport ─────────────────────────┐ ┌ Properties ──────────────┐ │
│ │ ⬚ sel │ │                                   │ │ Door #12                 │ │
│ │ ▭ room│ │            3D view                │ │ width 1  height 8        │ │
│ │ ⇕ push│ │                                   │ │ locked ☑  key: #7 ⌖      │ │
│ │ ◢ slop│ │                                   │ ├ Entities ────────────────┤ │
│ │ 🖌 pnt │ ├ Plan ─────────────────────────────┤ │ ▾ creatures (4)          │ │
│ │ ✺ str │ │       top-down, height slice      │ │ ▾ doors (3)  ▾ keys (1)  │ │
│ │ ⊕ ent │ │                                   │ ├ Textures ┬ Areas ┬ Issues┤ │
│ └───────┘ └───────────────────────────────────┘ └──────────────────────────┘ │
│ room tool: drag a rectangle, then drag up · cell (4, 3, 7) · 312/512 tris · ● │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Menus**

| Menu | Items |
|---|---|
| File | New, Open, Open recent, Save, Save as, Revert, Screenshot, Quit |
| Edit | Undo, Redo, Cut, Copy, Paste, Duplicate, Delete, Select all, Select none, Preferences |
| View | Each window on or off, Reset layout, viewport mode (plain / PS1 / neural), Grid, Strangeness overlay, Links, Lights, Height slice up/down |
| Tools | One entry per tool, with its shortcut |
| Play | Play, Play from here |

**Status bar**, left to right:
- the current tool's hint;
- the hovered cell and face;
- the selection size;
- the number of triangles in view out of 512;
- FPS;
- an unsaved-changes dot.

**Dockable windows**

| Window | What it does |
|---|---|
| Viewport | The see-through hole onto the 3D view, with view toggles in its corner |
| Plan | Top-down 2D, with pan and zoom. It shows the cells at the current height slice, plus entities and links. You can edit in it with the same tools. |
| Tools | The tool palette |
| Properties | Whatever is selected: an entity, area, light, faces or cells. Multi-select edits shared fields. |
| Entities | An outliner grouped by kind, with a filter box. Click to select, double-click to frame it. |
| Textures | The slot palette with previews for the level's texture set. Click to pick a texture for painting. |
| Areas | A list of areas. Select one to highlight it. |
| Issues | Validation results. Click one to jump to it. |

The layout is saved between sessions.

### Viewport modes

- **Plain:** a Bevy mesh built from the scene's triangles, with textures and
  baked vertex colours, plus gizmos. It's fast, and it's the default for
  editing.
- **PS1:** datagen's software renderer, as the game draws it now.
- **Neural:** the trained network. It's slow on the laptop, so it's for
  checking things, not for editing.

### Camera

- **Fly:** hold the right mouse button and use WASD, Q/E for down and up, and
  Shift to go faster.
- **Orbit:** Alt + drag or the middle mouse button. The scroll wheel zooms.
- **F** frames the selection.
- **Height slice:** PgUp and PgDn move a cut plane. Everything above it is
  hidden in both views, so you can work on the lower floors.

### Tools

| Tool | Key | What it does |
|---|---|---|
| Select | V | Click a cell face, entity or light. Shift adds to the selection, drag selects a box. Delete removes. |
| Room | R | SketchUp-style. Drag a rectangle on the floor (or any horizontal face), then move the mouse up to extrude it in layers, and click. That opens the box. Ctrl while extruding makes it an area too. |
| Push/Pull | P | Drag selected faces along their normal. Floors move by layers; walls and ceilings move by tiles for walls and layers for ceilings. Pushing into rock opens it, pulling closes it. This is how you raise the ground or dig a pit. |
| Carve / Fill | C | A single-cell brush. Click to open, Shift-click to close. |
| Slope | S | Click a floor edge to make a ramp or stairs climbing away from it. The scroll wheel sets the rise, Tab switches between ramp and stairs. |
| Paint | B | Click faces to paint them. Drag paints along a surface. Shift-click fills connected faces on the same plane. Alt-click picks up a face's paint. `[` and `]` rotate the texture. |
| Strangeness | X | A brush over the lattice, with a radius (in tiles), a strength and a falloff. Click adds, Shift-click removes, Ctrl-click smooths. The overlay tints the view by strangeness while this tool is active. |
| Entity | E | Pick a kind in the Properties window, then click to place it. Placement snaps to cell centres, or to cell edges for doors. R rotates. |
| Light | L | Click to place a point light on a face. Drag up to lift it off the face. |

### Undo

The editor snapshots the whole `Level` after each committed operation (a
finished drag, a property edit when it loses focus), and keeps up to 200
snapshots. Levels are small, so this is simple, and diffs can't drift from
real state. If memory ever matters, we switch to storing diffs.

### Play

**Play** saves the dungeon to a temporary file and starts the game in a separate
process, with `game --level-file tmp.ron`. The player spawns at
`PlayerStart`.

**Play from here** does the same with `--start x,y,z,yaw`, placing the player
on the floor under the viewport camera and facing the same way.

A separate process keeps the editor and the game independent, and a crash in
one doesn't lose work in the other. The game gains `--level-file` and
`--start`, chunk streaming, and doors, keys and levers. It also gets a use
key, and in the egui panel a strangeness slider that offsets the field, as
now.

### Validation (Issues window)

The editor reports:
- steps taller than `MAX_STEP` between neighbouring walkable cells, which are
  only a problem where they cut a path;
- open cells that can't be reached from `PlayerStart`;
- a missing `PlayerStart`, or more than one;
- doors not sitting between two open cells;
- locked doors and chests whose key doesn't exist, or can't be reached
  without passing through that door;
- creatures that don't fit in their space;
- views where the triangle budget overflows badly (sampled along walkable
  cells).

## Training on hand-built levels

`datagen gen --levels levels/ --level-share 0.3` mixes hand-built levels in with
generated ones. A share of the scenes then comes from a random region of a
hand-built dungeon, with lighting and strangeness jittered so each region gives
many variations. Doors are rendered open, closed and part-open, so the
network learns them moving. The
walker and the creature cameras work unchanged on `Grid`.

Later, a script reads every `levels/*.ron` and records statistics: area sizes
and labels, how areas connect, height changes, slope use, paint choices per
label, entity placement relative to areas, and strangeness shapes. That
becomes the spec for the new generator.

## Milestones

1. **Level core.** `engine::world::level`: the format, `Grid`, chunked `to_scene`,
   and the strangeness field. The generator moves onto `Grid`.
   `game --level-file` with chunk streaming. Tests: round-trip a level
   through save and load; walker connectivity on generated and hand-made
   levels.
2. **Editor shell.** Dock layout, menus, status bar, theme, plain viewport,
   fly and orbit camera, open and save, undo.
3. **Shaping.** The Room, Push/Pull, Carve and Slope tools, the Plan view, and
   the height slice.
4. **Paint and atmosphere.** The Paint tool, the Textures window, areas,
   lights.
5. **Entities.** Placement, Properties, the outliner, links, creature
   previews. In the game: doors that open, keys, levers.
6. **Strangeness.** The depth curve, noise, look settings, the brush and
   overlay, plus the PS1 and neural viewport modes.
7. **Play and checks.** Play, Play from here, the Issues window.
8. **Training.** `datagen --levels`. After you've built a set of levels, the
   analysis and the new generator.

## Decided

- Ceilings follow slopes by default. A slope can be set flat.
- Doors open and close in the game, and can be locked with keys.
- A dungeon is one map: all its floors are in one file.
- Strangeness is tunable in the editor: depth curve, noise, painted aura,
  area overrides, and its look (fog, lights, morph).
- The strangeness lattice has one point per tile in every axis.
