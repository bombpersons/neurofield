# Living walls: NCA textures with hallucinations painted onto them

Plan, agreed in outline 2026-11-15 (branch `nca-walls`, from `dream-atlas`). Builds on
`dream_atlas.md` (the per-face wall pages) and `shell_relief.md` (the marched relief). It
**retires** the whole-screen DeepDream of `dream_atlas.md` on this branch (decided 2026-11-15). It changes the role of
`distilled_dream_net.md`: from running the whole dream to making hallucinations better.

## The idea

- **Walls are grown, not dreamed.** Each wall material (bricks, tiles, intestines, teeth,
  psychedelic fractals, ...) is a small **neural cellular automaton (NCA)**. Each NCA is trained
  on one exemplar and grows an endless, animated, repeating texture: colour **and a learned
  height** (plus normals, and maybe roughness).
- **Painted in the editor:** materials per room, or painted onto faces, and a **motion
  direction** painted onto the walls, so flesh can creep up a shaft, ooze down a wall, or swirl
  round a pillar.
- **The shell relief** (`shell_relief.md`) takes the NCA's height everywhere, so the whole
  dungeon swells, bulges and moves. The NCA's colour is the wall's colour: it's made to look
  like what the wall should be.
- **Hallucinations are painted onto the wall pages,** steered by a network (robust ResNet,
  CLIP, later our own student; `distilled_dream_net.md`). They're placed in the editor or by
  the game, grow gradually (the ramped, masked reveal tested in `distilled_dream_net.md`), and
  **stop once fully formed**, becoming part of the wall.
- **The walls notice you:** they undulate faster where you look, and hallucinations can be
  spawned where the player stares.
- Creatures and wall tentacles go on top, as now.

### Scope and targets (decided 2026-11-15)

- **All of the level's surfaces:** walls, floors, ceilings, stairs. Not doors, levers, items and
  other things in the level, which keep the PS1 look.
- **Controls, how the game uses them decided later:** each room (or the whole level) has
  - a **speed** (how fast its NCAs step);
  - a **depth** (how far the shell pushes by their height);
  - an **off switch**, back to the level's own PS1 textures and flat walls.
- **Growing in:** a room's surfaces can visibly grow in from a seed when first seen, or be
  pre-grown when the level loads.
- **Saves:** the grown surfaces and the finished hallucinations are saved, including opened
  secret doors, not regrown on load.
- **Hallucination budget:** a few forming at once, capped at about 5 ms of GPU a frame. Heavier
  networks just form more slowly.
- **What retiring the whole-screen dream removes:** only its per-frame loop over the screen.
  Kept: the networks' forward and backward passes (hallucinations run them on pages), the
  pages, the shell, and the GPU drawing of the level.
- **Creatures and tentacles:** they exist already. Tentacles are drawn as creatures are, spawning
  from the walls and attacking the player when affliction is high. Both must be depth-tested
  against the shell's depth, so they hide behind bulges; check that they are in step 2.
- **Frame rate:** **60 fps** on the RX 580, with either the PS1 look or the modern look.

### Why

- **Cheaper.** Today a full network runs over the whole screen every frame (about 23–27 ms of
  GPU on the RX 580 in the stair corridor). An NCA is a 3×3 neighbourhood and a tiny MLP per
  texel, run only on the pages in sight: likely a millisecond or two (to be measured).
  Hallucinations cost only while forming, and only on a few pages.
- **Hallucinations can use expensive networks.** As a hallucination forms over seconds rather
  than per frame, it no longer needs a real-time network. We can use networks we've seen look
  better: the Wide ResNet, CLIP with many cutouts, even a one-step diffusion pass at about a
  second a step. Prompt-able, directed hallucinations become affordable without waiting for a
  distilled student.
- **Painted onto the page, not the screen.** A wall's page is flat and faces the network
  square-on, so a hallucination is painted *onto* the wall, not cut into it like a window (the
  screen-space test's weakness). The shell then gives it its bulge.
- **Real height.** The NCA learns height from a height map, not from brightness.
- **Designable.** Materials and motion are art-directed per room, and the dream's look is
  chosen, not left to a network's taste.

## Prior work

- **[MeshNCA](https://meshnca.github.io)** ([paper](https://arxiv.org/abs/2311.02820), TOG 2024,
  [code](https://github.com/IVRL/MeshNCA)) is close to what we want:
  - It grows textures on meshes directly, without UVs, and generalises to any mesh after
    training on an icosphere.
  - Its 16-channel state has 3 channels of albedo, 3 of normal, 1 of height, 1 of roughness and
    1 of ambient occlusion; the rest are hidden.
  - It was trained on 72 PBR texture sets (plus DyNCA's RGB textures).
  - It runs in real time in WebGL on low-end devices.
  - Its demo has a "bumpiness" slider that pushes the mesh out by the height, as our shell does.
  - Test-time controls: orientation, grafting (blending textures), self-regeneration, motion.
- **[DyNCA](https://dynca.github.io)** ([paper](https://arxiv.org/abs/2211.11417), CVPR 2023)
  trains an NCA on an appearance target and a **motion target**:
  - The motion is a hand-made optical-flow field or a dynamic-texture video.
  - At run time: motion speed, motion direction, and an editing brush.
  - It makes arbitrary-size, endless texture videos in real time on a low-end GPU.
- **[Self-Organising Textures](https://distill.pub/selforg/2021/textures/)** (Distill 2021) is
  the base texture NCA. It is stable over long runs, tiles naturally, and regenerates.

## The NCAs

- **One NCA per material,** trained offline, blended where materials meet by MeshNCA's
  grafting: two trained NCAs work together on the cells near the boundary, so one texture
  turns smoothly into the other (decided 2026-11-15). Its exact mixing is to be read from the
  paper before step 2.
  - The other option, for later if wanted: one NCA for several materials, chosen by a few
    "genome" channels in each cell ([Signal Responsive NCA](https://arxiv.org/abs/2407.05991),
    up to 8 textures in one network). It morphs smoothly between materials, and could let a
    material spread across the walls during the game.
- **State per texel:** about 12–16 channels. Albedo (3), height (1), maybe normal (3, or derived
  from the height) and roughness (1); the rest hidden.
- **Update:** fixed perception filters (identity, Sobel x and y, Laplacian) over a 3×3
  neighbourhood, then a 2-layer MLP (about 100 hidden units), then a stochastic update (each
  texel updates with probability about 0.5). A few thousand parameters.
- **Motion (DyNCA):** the NCA is trained to move its pattern along a target flow. At run time
  the direction comes from the **flow painted in the editor** (a 2D vector per texel of the
  page, in the wall's own plane), and the speed from a global rate times the gaze factor below.
  The direction control is DyNCA's: the perception filters are rotated to the local direction.
- **Training targets:**
  - **Bricks, tiles, stone, wood:** CC0 PBR sets (ambientCG, Poly Haven) with real height maps.
    CC0 is fine for a sold game.
  - **Flesh, intestines, teeth, fractals:** no PBR sets, so we make one. Take an exemplar
    image (photo, painting, our own render), make a height map with Depth Anything V2 Small
    (Apache-2.0) once, offline, and derive the normals from it. The exemplars' licences must
    suit a sold game.
  - **Motion:** a hand-made flow (straight, swirl, pulse) or a reference video (CC0 or our own).
- **Losses:** DyNCA's and MeshNCA's. A style loss (Gram matrices of VGG features, or their
  optimal-transport loss) on the albedo; plain losses on the height and normals; a motion loss
  for the flow.
  - **Licence:** VGG's ImageNet weights are only used as a loss during training, nothing of
    them ships. Still worth a check.
- **Cost:** minutes to an hour or so per material on one GPU (a guess). Local CPU for tiny
  tests, Kaggle or Modal for the library.

## In the game

### Running the NCA on the pages

- The NCA's state lives on the atlas pages, which already hold the dream: a page per tile-sized
  face (`dream_atlas.md`).
  - A page's state stores the NCA's channels, at a texel density to be chosen. Memory: pages ×
    texels × channels; 16 channels in half floats at 64×64 a page is 128 KB a page.
  - The pages' layout may need changing from colour and height to the NCA's state. To be decided
    against `dream-renderer`'s current page format.
- **Step only the pages in sight, plus a margin,** a few NCA steps a frame. Pages out of sight
  freeze (their texture is kept), as the dream's pages do now.
- **Cells are the page texels; neighbours cross the surface (decided 2026-11-15).** We keep
  MeshNCA's idea (each cell's neighbours found across the surface, so texture and height flow
  round corners and stairs without seams) without its dense mesh. The level's triangles stay
  coarse, and the shell gives the shape.
  - **Cells:** one per page texel; the NCA's state lives on the pages, which the shell and the
    drawing already read.
  - **Within a page:** an ordinary 3×3 neighbourhood, as a flat-image NCA.
  - **Across a page's edge:** a precomputed neighbour map for each page's border texels gives
    the matching texels on the neighbouring page, including round creases. These are the same
    face-to-face links the shell uses to make heights meet at creases.
  - **Directions:** each page has its own axes, so across an edge the neighbour's vector
    quantities (the perception's gradients, the painted flow) are rotated into this page's axes,
    by a rotation stored per edge (like a cube map's edges).
  - **Where three or more pages meet:** a corner texel's missing or doubled neighbours are
    averaged, as cube maps do.
  - Try it on a corner and a stair first.
- **Painted motion** comes from the editor's flow, stored per page, in the page's own axes.
- **Shown:** the NCA's albedo is the wall's colour, its height is the shell's height (the march
  as now), and its normals (or the height's) light it.

### The walls notice you

- **Gaze speed-up:** each page's step rate rises with how near the centre of view it is, and
  maybe with how long it's been looked at. Walls you watch undulate faster; walls at the edge
  of sight slow.
- **Spawning by gaze:** where the player stares long enough (a page's look-time passes a
  threshold), the game may start a hallucination there, as well as at the editor's placements.

### Hallucinations

- **Where:** placed in the editor (a mask on the faces, a class or prompt, how long to grow) or
  by the game (random, or by gaze), on the wall pages.
- **How:** the ramped, masked reveal (`distilled_dream_net.md`) in page space:
  - Dream the network's goal (a class, later a prompt) inside the mask, on the page's colour
    over the NCA's albedo.
  - The steering strength ramps up over the formation time.
  - The page is flat and faces the network square-on. Mask edges are feathered so it grows out
    of the wall.
- **Stop when formed:** once the network is sure enough (a confidence threshold, the
  confidence step's `p`) or the time is up, the hallucination freezes and becomes part of the
  page. It then costs nothing.
- **The NCA around it: blended (decided 2026-11-15).** Inside the mask the hallucination holds
  (the NCA doesn't "heal" over it); towards the mask's feathered edge it blends into the NCA, so
  the living wall flows up to it and around it.
- **Its height (decided 2026-11-15):** the hallucination's own height plus a fraction of the
  NCA's height under it, so it still swells and moves a little with the living wall. Towards
  the mask's feathered edge the fraction rises to all of the NCA's height, as the colour blends.
  The hallucination's own height comes from a depth network (Depth Anything V2 Small) run once on
  the finished page, or from the dream-lamp height as it forms (the shell's current method).
- **Which network:** not tied to real time.
  - Robust ResNet-50 ε3 now.
  - Wide ResNet-50-2 ε3: the most photo-like in our tests.
  - CLIP with cutouts, for composed prompts ("a door made of bone").
  - The distilled student later.
  - Possibly a one-step diffusion pass at low strength for a final sharpening, licence
    permitting.
- **Secret doors:** a door hallucination placed by the level is one of these. The door opens
  once it's formed (the network's confidence passes a threshold), or once the player has
  looked at it long enough to grow it.

## In the editor

- **Material:** per room as a default, and painted per face.
- **Motion:** a flow vector per face (or brushed per texel), shown as arrows on the walls.
- **Hallucination placements:** a mask on faces, a class or prompt, a growth time, and whether
  it's a secret door.
- **Saved in the level file** with the rest (the editor's format is meant to be easy to read
  and diff, `editor.md`).

## Steps

1. **Offline NCA look test:**
   - Train a texture NCA (with height) on one CC0 brick set and one flesh exemplar (height from
     Depth Anything), on a flat plane.
   - Then one with a DyNCA motion target.
   - See the look, the height and the stability over long runs.
   - Compare with MeshNCA's released models if useful.
2. **NCA on the pages in the game:**
   - An NCA step kernel in `dream-renderer`: each texel's 3×3 perception (across page edges by
     the neighbour map, rotated into its page's axes), then a tiny MLP.
   - The neighbour map and the per-edge rotations, built from the level's faces.
   - The NCA's state on the pages, its colour and height read by the drawing and the shell.
   - Retire the whole-screen dream's passes from the frame.
   - The shell reading the NCA's height.
   - Measure its cost against the current dream on the RX 580.
3. **Hallucination on a page:** the masked, ramped reveal in page space with the robust ResNet,
   stopping when formed. Then its height.
4. **Motion and gaze:** the painted flow, the gaze speed-up.
5. **Editor:** materials, flow painting and hallucination placements.
6. **Material library:** bricks, tiles, flesh, intestines, teeth, fractals, ...
7. **Better hallucinations:** the Wide ResNet, CLIP with cutouts, the distilled student
   (`distilled_dream_net.md`).

## Progress

### Step 2, first pass (2026-11-15)

Running, uncommitted, with MeshNCA's demo models as stand-ins (their repo states no licence:
for testing only, not to ship). Try it with:

```sh
target/release/game --dream-atlas runs/dream/googlenet.safetensors --dream-nca runs/nca/meshnca_demo.safetensors --dream-psx
```

- **Models:** `runs/nca/meshnca_demo.safetensors`, all 72 of MeshNCA's demo models as plain
  matrices (80 -> 128 -> 16; colour in channels 0-2, height in 6). Their perception, learned on
  an icosphere's 6 neighbours, works on our grid's 8 with its sums rescaled (4 neighbours work
  too but less evenly). Brick models' bricks come out 2-3 cells across: small.
- **On the pages** (`dream-renderer/src/nca.rs`, `nca.wgsl`): 64×64 cells a page, stepped
  where pages are in sight (others kept), painted onto the atlas's colour and the pages'
  heights. Pages start empty, so walls grow in when first seen. Neighbours carry across tile
  edges along a wall and **round corners**: inner corners (wall to floor), flat, and outer
  corners, folded about the crease (unit-tested). Walls get one model, floors and ceilings
  another (`--dream-nca-walls`, `--dream-nca-floors`, or the panel).
- **In the PS1 pass** (`--dream-psx`): the shell is drawn inside the PS1 camera's own frame: a
  system after `main_opaque_pass_3d` in its Core3d schedule runs the NCA, the shell's passes and
  the march into the **camera's own depth buffer**, then lays the shaded pixels into its
  colour target (`psx_blit.wgsl`). The level's own flat meshes are hidden (the shell is the
  walls); doors, levers, things and creatures stay ordinary PS1 meshes and depth-test against
  the bulges. At 320×240, chunky nearest cells, the PS1's 15-bit colour and 4×4 dither.
- **Cost on the RX 580:** NCA about 1.25 ms a step for ~18 pages in sight; the whole dream about
  6.5 ms a frame in the PS1 pass (march ~1.1 ms) against 13.8 ms at 3× (march ~7.5 ms). The
  game as a whole runs at 40-50 fps either way: something else holds it under 60, still to find.
- **Not yet:** creatures hiding behind bulges is wired but unchecked (none came in sight in the
  test walks); the old whole-screen dream's loop is still there (idle at step size 0); the
  shell's lighting and fog are its own, close to but not exactly the PS1 material's; the premade
  models don't move once grown (motion is step 4); psx-bevy is ours now (forked from the
  template) but the shell's code lives in `dream-renderer` and `game/src/dream_atlas.rs`, to
  be sorted out in the big review.

## Open questions

- **Texel density per page.** Pages are 256 texels a side now (`Config::page_res`). The NCA's
  cost grows with the texels it steps: about 100 pages in sight at 256² is about 6.5M cells a
  step, likely several ms a step on the RX 580; at 64² it's 16 times fewer. 64 a tile is about
  what PS1 games used. Options: step the NCA at 64 or 128 and show it larger; step far pages
  less often. To be measured in step 2.
- **PS1 look or modern look for the walls?** The shell's march works at any resolution, so it
  can be drawn PS1-style:
  - the march at the PS1 screen's own 320×240 (about 9 times fewer pixels than at 3×);
  - low-res, nearest-sampled NCA texels;
  - banded lighting, 15-bit colour with dithering, the fog.
  It can't be true to the PS1 hardware (which had no per-pixel marching), but it can match its
  look, and it's much cheaper. Bulges' silhouettes would be chunky, which may suit it. Try both
  in step 2: the march at 1× with PS1 shading, and at 3× with the modern lighting.
- Does the NCA's height read well through the shell at our relief heights? To be seen by
  trying it. If its features are too small, scale them up: step the NCA on fewer cells per
  page (bigger features, and cheaper) or train on a downscaled exemplar; the shell's
  smoothing keeps a coarser height smooth.

Decided 2026-11-15: the whole-screen dream is retired on this branch; the NCA blends around
hallucinations, and a hallucination's height is its own plus a fraction of the NCA's; the NCA's cells are the page texels, with neighbours across page edges (MeshNCA's
idea without its dense mesh; the shell gives the shape); one NCA per material, grafted where
materials meet.
