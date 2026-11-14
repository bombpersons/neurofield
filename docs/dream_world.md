# The dream world: a modern renderer for the dreamed levels

Plan, agreed in outline 2026-11-14 (branch `dream-atlas`). Follows `dream_atlas.md`.

## Where we are

The dream atlas runs today as a post-process on the PS1 picture:

- psx-bevy draws the level (lit, fogged, 240 lines). The picture is **read back** to the CPU.
- A **CPU G-buffer** (5–6 ms) works out, for each pixel, the wall page and the point on it.
- `dream-renderer`, on **its own wgpu device**, dreams on the screen layer and carries it along by motion vectors (made on the CPU from the last view). It remembers the dream on the wall pages a band a frame and fills newly seen pixels from them. It shows the result at 3×, which is uploaded back to Bevy as an image.
- Creatures from the creature net are drawn into the PS1 picture and masked out of the dream by their depth.
- The relief is parallax from the dream's blurred brightness, and only in the walls mode.

What it can't do: real depth, hiding things behind bulges, silhouettes, real lighting. It also pays for the readback, the CPU G-buffer, a second device and 1–2 frames of lag.

## Where we're going

The dreamed level is drawn by a **modern renderer**: perspective-correct, displaced, PBR-lit, inside `dream-renderer` (game-agnostic), on **Bevy's own GPU device**. The PS1 renderer keeps the sword, HUD, people and things, which keeps the low-poly PS1 look against the dream.

A frame:

1. **Geometry pass** (raster, ours): the level's triangles, subdivided and **pushed out along their normals by the dreamed height**. Drawn into a G-buffer:
   - depth;
   - world position;
   - normal;
   - face and page, and the point on the page;
   - **unlit** albedo (the level texture);
   - baked vertex light.
2. **Dream** (compute, as now, but reading the G-buffer on the GPU):
   - motion vectors from the G-buffer and the last view;
   - warp, prepare, GoogLeNet, step;
   - the dream now steps a **colour and a height**, through a fixed "dream lamp" (below);
   - remember on the pages, fill from them.
3. **Lighting** (compute or full-screen):
   - albedo: the texture plus the dream colour;
   - normal: the geometry's, bent by the height;
   - lights: GGX PBR from the torch, with the baked vertex light as the ambient term;
   - roughness from ownership, so the dreamed wall is wet and glossy;
   - screen-space contact shadows;
   - fog last.
4. **Compositing:**
   - the creature net's colour and depth, depth-tested against the dreamed depth;
   - the PS1 layer (sword, HUD, people, things) over the top, depth-tested where it has depth.

### The dreamed height (option 1 in the conversation)

A height channel next to the colour, in the screen layer and on the pages. The dream doesn't see it directly. Before GoogLeNet, the albedo plus the dream colour is shaded by the height's normals under a **dream lamp**:

- Backpropagation then carries the gradient through the shading into the height, so the dream sculpts as well as paints.
- The lamp **moves every step**, a random direction in the hemisphere facing the wall, so a painted shadow can't stand in for a real shape (DreamFusion's trick).
- **Every few steps the network sees the shading alone**, with no colour, so features have to come from the shape by themselves.
- The `colour` knob stays low, to push the dream toward height.
- The dream lamp is only for the dream's own input. The picture shown is lit by the real lights.

Related work: Kato et al., *Neural 3D Mesh Renderer* (CVPR 2018), did 3D DeepDream on vertices. Text2Mesh (CVPR 2022) did normal displacement plus colour under CLIP. DreamFusion (2022) did shading with random lights and shading-only renders. Ours is the real-time, per-frame, height-field version.

### Displacement

- **Subdivision:** each wall face (a tile side) is drawn as an N×N grid, pre-built, with 3–4 levels by distance (for example 32, 16, 8, 2 a side), chosen per face on the CPU each frame. With hardware tessellation absent in wgpu, this is the plan; a compute pass building adaptive meshes is the upgrade if close-ups look faceted.
- **Bulges across edges and corners, with no cracks.** A bulge may cross a tile's edge onto the next wall tile, and round a corner onto the wall or floor it meets:
  - **Neighbours:** each face knows its neighbours across its four edges (built with the level mesh, by matching edges in world space). The page table carries each face's page and its level of detail.
  - **Shared edge heights:** a vertex on an edge (or corner) takes the **average** of the height of every page that meets there, each read at that point. Both sides compute the same number, so they meet exactly.
  - **Direction:** an edge vertex is pushed along the **average of the meeting faces' normals**. On a flat wall that's the wall's normal; at an outside corner it's diagonal, so the corner itself bulges round. At an inside corner the two faces' bulges meet in the crease.
  - **Levels of detail:** where a neighbour is coarser, this face's extra vertices along that edge take the position interpolated along the coarse edge (the usual geomorphing trick). So edges match without stitching index buffers. A face's level is chosen per frame on the CPU and written to the page table.
  - **The dream's side:** it grows on the screen, so its colour and height already run continuously across edges. The pages each remember their own side, and the averaging above joins them.
  - A face whose neighbour has no page yet (not seen) averages with 0 there, which flattens that edge until the neighbour gets one.
- **Height on the pages:**
  - The vertices read the height from the pages, so it lives there in world space. It's written every frame while carried along, which is cheap with one channel.
  - The screen layer keeps a sharper height for normals.
  - The pages keep the height at half the colour resolution (128² a page, 32 MB for 512 pages).
- **Hiding:** bulges hide whatever is behind them, through the depth buffer, and silhouettes come out right.

### Faces and pages on the GPU

The CPU G-buffer assigns pages per pixel. On the GPU:

- **Face ids:** each face of the level gets a fixed id when the mesh is built, carried as a vertex attribute.
- **The page table:** a buffer maps face id to page (−1 for none). The CPU keeps the LRU allocation as now.
- **Faces seen:** the geometry pass marks the faces it drew in a bit buffer, which is read back (a few hundred bytes). The CPU allocates pages for newly seen faces for the next frame.
- **Lag:** a face seen for the first time has no page for one frame, so it doesn't dream or remember for that frame and fills from its page the frame after.

### What the game gives the crate

- the level's triangles (position, normal, texture coordinate, texture, baked light, face id), once per level;
- the textures, as an array;
- the camera each frame;
- the lights: the torch, with its flicker;
- the creature net's colour and depth;
- the PS1 layer (sword, HUD, people, things) and its depth.

What it gets back: the finished picture, as a texture Bevy shows, and its depth.

## Steps

Each step ends in the game, compared against the one before (frame rate, recordings).

1. **Bevy's device.**
   - `dream-renderer` is made on Bevy's `RenderDevice` (wgpu 30.0.1 both), and runs as a node in Bevy's render world.
   - The PS1 picture goes in as a texture and the shown frame comes out as one. There's no readback or upload, but the CPU G-buffer stays for now.
   - *Checks:* the same picture as today; the frame rate; the lag.
   - **Done 2026-11-14.**
     - **Library:** `DreamRenderer::encode` records a frame into a caller's encoder, picture texture in and frame texture out. It's checked identical to the CPU path (`examples/check.rs`).
     - **The game:** a camera of its own (`DreamGraph`, order 1001, with the display moved to 1002) runs it in the render world, among Bevy's commands. Two gotchas:
       - the camera needs a `Projection`, or Bevy never sizes its target and never runs it;
       - a frame's work must be one command list, because Bevy submits everything together. Hence the double-buffered surfaces and the separate slot-table sets.
     - **Results:**
       - with no creatures, a median of 45 fps on the reprojected walk, against roughly 26–37 before;
       - with a creature in view, 19–20 fps against 17–18. The creature net (its own device, waited on) now waits about 42 ms instead of 24, because the dream has the GPU. Moving it onto Bevy's device too belongs to step 6.
2. **The geometry pass and the GPU G-buffer.**
   - The crate rasterises the level itself: flat for now, with face ids, the page table and faces-seen feedback.
   - Motion vectors are made on the GPU, and the CPU G-buffer and its 5–6 ms go.
   - The picture is lit as today: albedo times the baked light plus the torch, Lambert, PS1 fog. So it can be compared with the PS1 picture.
   - *Checks:* pages and motion match the CPU G-buffer's on the same frames, compared with `GAME_DREAM_PAGES`-style dumps.
   - **Done 2026-11-14**, but drawn from the PS1 picture still: the crate's own lit picture moves to step 4, where the lighting is replaced anyway.
     - **Library** (`dream-renderer/src/level.rs`, dream.wgsl `gbuffer`):
       - `set_level` takes the triangles, lists the faces (each cell a triangle's plane passes through) into a hash table on the GPU, and is called again when they change (doors).
       - `encode_view` takes a `View` and the creatures' depth. A render pass draws each pixel's triangle id, and `gbuffer` works out the surface, 1/z, motion from the last view, and the face's page.
       - The faces seen are marked per band of rows (16 bands) and read back without waiting, 2–3 frames late. The crate gives pages out from those reports (LRU, a page unseen for 8 frames may be taken), and they also give the pages for dreaming and the bands for remembering.
     - **Checks:**
       - In `examples/check.rs`, a room with a pillar, a step and a slope, against a CPU reference: all 76 800 pixels paged, all but one on the same face, uv within 3e-6, motion within 1e-4 px.
       - In the game (`GAME_DREAM_CHECK=1`, which also runs the old CPU G-buffer): every pixel on the same face as the CPU's, uv within 1.4e-6, motion within 2e-4 px. The only differences are about 600 pixels past the fog's far end, which the GPU leaves out (their dream has no strength).
     - **Results:** the same walk as step 1's, frame for frame:
       - the main world's part goes from 5 ms to nothing;
       - with a creature in view, 22–24 fps against 18–20;
       - facing a wall, 48–50 against 44–47.
3. **The dreamed height.**
   - A height channel goes into the screen layer, the pages, the warp and remembering.
   - The dream-lamp shading and its backward pass go in, with the moving lamp and the shading-only steps.
   - It's shown flat, with the height as normals only, to judge whether shapes get sculpted.
   - *Checks:* the height gradient against a finite-difference check in the `check` example; a recording of the height and the shading alone.
4. **Displacement and PBR.**
   - Subdivided faces with levels by distance, and vertex displacement from the page height, shared across edges and corners with the neighbours' (geomorphed where their levels differ).
   - GGX lighting from the torch, roughness from ownership, contact shadows and fog.
   - The relief parallax goes, since the geometry does it now.
   - *Checks:* silhouettes at corners, bulges crossing tile edges and corners with no cracks (also at level-of-detail changes), bulges hiding walls behind, and frame time against step 2.
5. **Compositing.**
   - Creatures depth-tested against the dreamed depth.
   - The PS1 layer (sword, HUD, people, things) over the top with its depth.
   - Doorways and linked levels: the joined level is drawn through the doorway as it's drawn now, by the PS1 renderer, until the crate handles more than one level.
6. **Speed**, once the look is settled:
   - overlapping the creature net with the dream;
   - LOD tuning;
   - dreaming every other frame with creatures in view.

## Risks

- **Running in Bevy's render world.** Bevy 0.20's render graph and the `RenderDevice` → `wgpu::Device` access need checking first; step 1 is mostly that. The fallback is the current two-device setup, with textures copied between them on the GPU.
- **The height may stay flat** if the dream prefers paint. The moving lamp, the shading-only steps and a low colour limit are the levers. Step 3 shows this before the displacement work.
- **Things on the walls:** doors, levers and chests are part of the level meshes or the scene's things. They have to be drawn by one side or the other, not both.
- **Memory:** the page atlas is 268 MB now, and height adds about 32 MB. That's fine on an 8 GB RX 580, tighter on 4 GB, where `max_pages` can come down.
- **The training-data and editor paths** use the PS1 renderer and stay as they are.
