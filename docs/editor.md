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

1. **A depth curve.** Strangeness against depth (layer), drawn as a graph in
   Properties over the level's layers with the slice marked: drag a point,
   double-click to add one, right-click to remove one. A dungeon can stay
   grounded for its first few floors, then slide. This is the main dial for the
   whole dungeon.
2. **Noise.** Its strength and scale, so the depth curve isn't perfectly flat
   on each floor.
3. **Painted aura.** Offsets stored on a **lattice of grid points**: one point
   every tile across the ground and every tile up, at cell corners. It's sparse
   and unset points are 0. Between points it's interpolated smoothly
   (trilinear, then smoothstep), so painting with a soft brush gives soft
   auras. Paint pushes up (an infected room) or down (a spared one). Offsets
   stay within −1..1.
4. **Area overrides.** An area can add to, or pin (hold at a value),
   strangeness inside it, fading out over a tile outside so a pinned room
   doesn't jump at its walls. Adds come before pins.

The sum is clamped to 0–1. Everything that uses strangeness now uses this field
instead of the generated one: triangle column 34, texture steps, fog colour,
and light colour.

**How strangeness looks** is tunable too, per dungeon:
- the fog, ambient and sun colours where it is fully strange;
- the hue range lights drift toward (a light can also have its own strange colour);
- the strangeness where the textures start and finish morphing (`morph`), where
  lights, ambient and sun start and finish turning (`lurid`), and the same for
  the fog (`fog_ramp`).

Defaults come from what datagen does, so generated levels look as they always
have. Properties shows the level's wall texture at each morph step, under the
strangeness it appears at. The network itself is told the strangeness as it is
(triangle column 34); the look settings change only how the renderer draws it.


### Areas

An area is a named, labelled box of cells (or several boxes). It holds:

- a label: `room`, `corridor`, `shrine`, `stairwell`, `outdoors`, or anything
  else you type;
- default floor, wall and ceiling paint;
- a colour tint (with the paint, as its **surface**);
- ambient light and fog (near, far): not in the file format yet;
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
viewport and the plan draw them as lines (yellow from a key, blue from a lever).
The Properties window edits them: a list of the level's keys for a door or chest,
and the target doors of a lever with a button to drop each and a list to add more.
The ⌖ button beside each picks the other end by clicking it in the 3D view or the
plan. Linking a key to a door also locks it. Creatures are grown from `kind`, `size` and `seed` by the same code as
`build_creatures`. The editor shows the result and has a *re-roll* button that
picks a new seed.

### Doors

Doors open and close in the game. A door is its own small mesh, a slab with a
frame, not part of the wall. The frame, and the wall above it up to the ceiling,
are part of the level:
- **Opening:** it swings a quarter turn on its hinge toward the way it faces, or
  slides up, over about half a second. A swinging leaf reaches down to the floor it
  swings over, and a sliding one to the lower floor.
  Its triangles (a `Part` in `engine::world::things`) are rebuilt each frame while it
  moves, apart from the level's, so only its own meshes and features are redone.
- **Collision:** a closed or moving door blocks the doorway, and an open door
  doesn't.
  The frame's posts aren't solid.
- **Use key:** E (or cross) with the doorway in reach (1.6 tiles) and in front
  toggles the door, or pulls a lever in front, whichever is nearer.
- **Locked doors:** these need their key. Walking over a key picks it up, the
  status line names the keys you hold, and a locked door you have no key for
  says so and stays shut; the first use with the key unlocks and opens it. A lever
  toggles its target doors, locked or not, and its handle flips over.
- Keys lie on a small plinth until picked up. Levers stand on the floor and
  don't block the cell.

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
        morph: (0.0, 1.0), lurid: (0.15, 0.85), fog_ramp: (0.2, 0.9),  // (start, end)
        light_hues: (0.0, 1.0),                                      // turns round the colour wheel
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
    areas: [ Area(name: "entry hall", label: "room", boxes: [((0, 0, 0), (6, 12, 6))], strangeness: Some(Pin(0.1))) ],
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
| View | Each window on or off, Reset layout, viewport mode (plain / PS1 / neural), Grid, Strangeness overlay (blue where grounded, through teal and yellow, to magenta; also the viewport's *Strange* toggle), Links, Lights, Height slice up/down |
| Tools | One entry per tool, with its shortcut |
| Play | Play, Play from here |

**Status bar**, left to right:
- the current tool's hint;
- the hovered cell and face, its texture and the strangeness there;
- the selection size;
- the number of triangles in view out of 512;
- FPS;
- an unsaved-changes dot.

**Dockable windows**

| Window | What it does |
|---|---|
| Viewport | The see-through hole onto the 3D view, with view toggles in its corner |
| Plan | Top-down 2D, with pan and zoom. It shows the cells at the current height slice, plus entities and links. Room, Carve, Slope and Strangeness work in it too: Room and Carve open (or with Shift fill) the **plan height** layers above the slice, set in the Tools window, and the Strangeness brush reaches that high. With the overlay on, open cells are coloured by strangeness. Pan with the right or middle button. |
| Tools | The tool palette |
| Properties | Whatever is selected: an entity, an area (name, label, strangeness, its own surfaces, notes) or a light (position, colour, radius, strange colour); otherwise the level, with its strangeness (see Strangeness). |
| Entities | An outliner grouped by kind, with a filter box (matching labels or kind names). Click to select, double-click to frame it. Add... puts one where the camera looks. |
| Textures | The slot palette with previews for the level's texture set. Click one to paint with it (this switches to the Paint tool). |
| Areas | A list of areas. Select one to highlight it in both views, double-click to frame it. *New area from the selected faces* makes one boxing the selected faces' cells, floor to ceiling. |
| Issues | Validation results, grouped, errors in red and warnings in amber; the tab shows how many. Click one to fly there, select its entity and centre the plan on it. |

The layout is saved between sessions.

### Viewport modes

- **Plain:** a Bevy mesh built from the scene's triangles, with textures and
  baked vertex colours, plus gizmos. It's fast, and it's the default for
  editing.
- **PS1:** datagen's software renderer, as the game draws it now.
- **Neural:** the trained network, found through `neurofield.ron` and loaded
  the first time the mode is chosen. It runs on a worker thread that always
  draws the newest view, so editing stays smooth however slow a frame is (about
  10 ms on an RX 580, seconds on a CPU); the status bar says how long frames
  take and on what. The frame fills the middle of the view as the PS1 screen
  would, with the tools' outlines and entity markers drawn over it. Without a
  model the editor says so and stays in Plain.

In every mode creatures are drawn in full colour as the game draws them (the
**Flesh** toggle, also in the View menu). Sculpting one takes 30–400 ms, so each
is sculpted in the background once the level has been still for a moment, and
kept by its shape: until then it is drawn through the PS1 pipeline like the
level. Moving a light only bakes a kept sculpt's light again; moving, resizing
or re-rolling a creature sculpts it afresh. The status bar counts sculpts in
progress.

### Camera

- **Fly:** hold the right mouse button and use WASD, Q/E for down and up, and
  Shift to go faster.
- **Orbit:** Alt + drag or the middle mouse button. The scroll wheel zooms.
- **F** frames the selection.
- **Height slice:** PgUp and PgDn move the slice, the layer the grid and the
  Plan view show. With **Cut** on (a viewport toggle, also in the View menu),
  everything more than a tile above the slice is hidden, so you can work on the
  lower floors. The cut is made by filling the rock in above it, so the tops of
  the walls show open from above, and picking sees the level as it is drawn.

### Tools

| Tool | Key | What it does |
|---|---|---|
| Select | V | Click a cell face or entity. Shift adds to the selection, drag selects a box of faces on the plane where the drag started. Drag an entity to move it over the floors (snapping as the Entity tool does). Delete removes an entity. |
| Room | R | SketchUp-style. Drag a rectangle on the floor (or any horizontal face, or the grid at the slice where there is none), then move the mouse up to extrude it in layers, and click. That opens the box. Moving down instead digs it below the floor. Ctrl while clicking makes it an area too. |
| Push/Pull | P | Drag a face, or the selected faces, along its normal. Floors and ceilings move by layers, walls by tiles. Pushing into rock opens it, pulling closes it. This is how you raise the ground or dig a pit. A floor keeps its slope as it moves. |
| Carve / Fill | C | A single-cell brush. Click to open the cell behind a face, Shift-click to fill the cell in front of it. A drag stays on the plane it started on, so it can't dig away under itself. |
| Slope | S | Click near a floor's edge to make a ramp or stairs climbing away from it. Dragging on across the floor makes a flight: each column rises by the rise from the last, keeping the first one's headroom, carving through rock and raising floors in the way. Shift-click takes a slope off. The scroll wheel sets the rise, Tab switches between ramp and stairs. |
| Paint | B | Click faces to paint them (a wall's whole height in that column, or the whole selection if you click in it). Drag paints along the surface. Shift-click fills the joined faces on the same plane that share a texture. Ctrl-click picks up a face's texture (Alt is taken by orbiting). Painting a face its default leaves it unpainted in the file. Rotating the texture (`[` and `]`) waits on paint gaining a rotation in the format. |
| Strangeness | X | A brush over the lattice, with a radius (in tiles), a strength and a falloff. Click or drag adds, Shift takes off, Ctrl smooths. In one stroke a point moves at most the strength (times the falloff) however often the brush passes, so going over a patch evens it out; stroke again to go further. In the 3D view it is a ball round the point under the pointer; in the plan, a column from the slice up the plan height. The overlay tints the view and the plan by strangeness while this tool is active. |
| Entity | E | Pick a kind in the Tools window, then click a floor to place it, and drag on to move it. Placement snaps to cell centres (Ctrl: exactly where you point), and doors to the middle of the nearest cell edge, facing into the cell you clicked so they open into it. Tab turns the next one a quarter (R is the Room tool). The pointer shows where it will go. There's only one player start: placing another moves it. Works in the plan too, at the slice. |
| Light | L | Click to place a point light just off a face. Drag to lift it further off. Select picks lights by their markers, in the 3D view or the plan, where their radius shows as a circle. |

Picking uses the triangles as drawn: the nearest one facing the camera under the
pointer, turned into the open cell it bounds and which face of that cell it is. A
click on a wall takes its whole height in that column; a floor or ceiling, one tile.
Drags change the level as they go, so the view shows the result live; Esc or the
right button puts it back, and the whole drag is one undo step.

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

The game runs its walk on psx-bevy's shell tick, so a developer build
(`--features dev-mcp`) can be driven by its MCP: `game --mcp --headless --software
--level-file … --view raster`, then `game_step` with pad buttons (up/down walk,
left/right turn, l2/r2 strafe, cross uses), `game_screenshot` of the rasteriser's
screen, and `game_query {"resources":["Report"]}` for the walker's position, keys
held, door states and the last message.

### Validation (Issues window)

The checks (`editor/src/checks.rs`) run again whenever the level changes and
nothing is being dragged. The walk from `PlayerStart` is solved with the doors:
a locked door opens once its key, or a lever that moves it, has been reached,
and the walk goes again until nothing more opens. The editor reports:
- a missing `PlayerStart`, or more than one, or one inside rock;
- places that can't be reached, each with what cuts it off: a locked door, or
  the lowest step taller than `MAX_STEP` between it and somewhere reached. An
  empty place is a warning (a pit's floor may be meant to be out of reach); one
  holding a key, lever or chest is an error;
- doors not sitting between two open cells, or over a step too tall to climb;
- locked doors whose key is behind the door itself or can't be reached, locked
  doors with neither key nor lever, chests whose key can't be reached, keys
  that point at something that isn't a key, and levers that move something
  that isn't a door;
- entities inside rock;
- creatures pressed against the ceiling (so shrunk to fit) or reaching into the
  walls;
- views that see far more triangles (half as many again) than the network's
  512: sampled from up to 160 walkable cells, four ways round at eye height,
  counted in the background since it takes a moment.

Errors stop the level playing as meant; warnings are worth a look. Agents get
the same list from `editor_level` with `part: issues`.

## MCP

The editor serves MCP tools, so agents can build levels on their own or help
someone build theirs. Two transports serve the same tools:

```bash
# An agent starts the editor itself and talks over stdio (stdout carries nothing else).
cargo run -p editor -- --mcp --headless --software --size 1280x800
# An agent joins an editor someone has open, over HTTP (or Tools > Agents).
cargo run -p editor -- --mcp-listen 127.0.0.1:7801 levels/demo.ron
```

`--headless` runs without an OS window: the window is only a size and the editor
draws into an image.

To let an agent into an editor you have open, use **Tools > Agents** and tick
*Let agents connect*. The window shows the address, the token and the command
that adds the editor to Claude Code, each with a Copy button:

```bash
claude mcp add --transport http neurofield-editor http://127.0.0.1:7801/mcp --header "Authorization: Bearer TOKEN"
```

Only this computer can connect, and every request needs that bearer token. The
port, the token and whether the server is on are kept in the editor's settings,
so the command keeps working from one session to the next; *New* makes a fresh
token and disconnects agents holding the old one. While the server is on, the
menu bar shows *• Agents on*, and *• Agent at work* in green when an agent has
asked something in the last few seconds; click it to open the window.
`--mcp-listen` and `--mcp-token` switch it on with that address and token for
one run without changing the settings.

Runs started with `--mcp` begin from the default layout, open the demo level
unless given a file, and never save the person's settings.

**Reading and viewing.**

| Tool | |
|---|---|
| `editor_status` | Tool, drag under way, selection, the face under the pointer, the status message, unsaved changes, undo depth, camera, view toggles, and where each panel is. |
| `editor_level` | A summary (bounds, entities, areas), or the entities, lights, areas, cells (each column's open runs), painted faces, settings or strangeness (settings, painted lattice points, area overrides) as JSON, the Issues window's checks (`issues`), or all of it as RON. |
| `editor_locate` | Where an entity, light, world point, cell face or plan tile is on screen, with what a click there would hit. |
| `editor_camera` | Point the 3D view at a world point or an entity. |
| `editor_screenshot` | A PNG of the window, one panel, or a rectangle. |

**Editing.** Each call is one undo step, and the status bar says what it did
("agent: ..."). Edits are refused while the person is part way through a drag
or a pick, and their mouse and keyboard keep working throughout.

| Tool | |
|---|---|
| `level_box` | Open (or fill) a box of cells: rooms, corridors, pits. |
| `level_stairs` | Stairs or a ramp from a floor, climbing a direction. |
| `level_sky` | Open a column to the sky, or roof it. |
| `level_paint` | Paint faces with a texture slot, or fill across a plane. |
| `level_add_entity`, `level_edit_entity`, `level_delete_entity` | Entities on a cell's floor (doors on one of its edges) or at a point. Kinds by name, with fields to change their defaults. Edits merge fields in; deleting drops links to it. |
| `level_link` | A door or chest to its key (which locks it), or a lever to a door. |
| `level_add_light`, `level_edit_light`, `level_delete_light` | Point lights. |
| `level_add_area`, `level_edit_area`, `level_delete_area` | Named areas made of boxes of cells, which can add to or pin strangeness. |
| `level_strange` | Paint strangeness on the lattice: a ball, a column over a disc, or a box of points; add, set, smooth or clear. |
| `level_settings` | Name, texture set, ambient, fog, sun, default surface, strangeness. |
| `editor_undo` | Undo or redo, the person's steps and agents' alike. |
| `editor_open`, `editor_save` | Open a file or a new level (refused over unsaved changes unless told to discard them), save. |

**Testing** (developer builds, `--features dev-mcp`, on stdio only). While these
are served, only injected input reaches the panels, so moving the real mouse
over the window changes nothing.

| Tool | |
|---|---|
| `editor_input` | Mouse and keyboard, a frame at a time, through egui and the 3D camera's input: `move`, `click`, `double_click`, `press`, `release`, `drag`, `wheel`, `key` (`"Ctrl+Z"`, `"Tab"`, `"F5"`), `key_down`, `key_up`, `hold`, `text`, `wait`. Clicks and drags can hold modifiers (`with: ["Ctrl"]`). Positions are window pixels. Answers once the editor has seen the last step. |
| `editor_widgets` | The widgets drawn last frame (from egui's accessibility tree), by text, hint, role or panel: rectangle and centre to click, toggled and disabled states. Widgets scrolled out of sight are left out. |
| `editor_window` | Resize the window. |

Native file dialogs don't work headless: use `editor_open` and `editor_save`.
The game has its own MCP, psx-bevy's, for gameplay (see Play above).

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
