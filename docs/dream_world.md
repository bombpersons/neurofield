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
   - **Done 2026-11-14**, while carried along only (`reproject`; the still and walls modes don't grow height).
     - **Where it lives:** a fifth plane of each screen layer, carried by the warp. A page height at half the page's resolution is stored a band at a time and filled from.
     - **The lamp:** `prepare` shades the input by the height's slope along the page, worked out across the net's own pixels through the surface's inverse Jacobian. The lamp is in each wall's own frame, at a random direction 20–70° up, with ambient light 0.3. Every 3rd step the net sees the shading alone on grey, and the colour doesn't step.
     - **Back to the height:** `shade_back` takes the gradient through the central differences into each pixel's height. `blur_height` blurs it with a Gaussian of `height_blur` pixels (default 3), and `screen_step` adds it, normalised by its own mean.
       - Without the blur, the height grew pixel speckle: at 0.003 tiles a pixel, small differences in height are steep slopes.
       - A sparse binomial (like the colour's) let through patterns at its tap spacing.
     - **Shown** shaded by the height as the eye's light lights it (`show_shade`), or the height alone (`--dream-show-height`).
     - **Checks:** against nudging 24 pixels' heights both ways by 1e-4, cosine 0.9987 with colour, 0.9960 shading alone, worst point within 2.2%. It converges as the nudge shrinks, so the remaining gap is the net's own nonlinearity.
     - **The look:** the height sculpts folded, brain-like ridges, carried along and remembered. Faint seams show at some tile edges where it's filled from the pages, which step 4's shared edge heights should fix. The cost is within the frame-rate noise at rest (46–47 fps).
4. **Displacement and PBR.**
   - Subdivided faces with levels by distance, and vertex displacement from the page height, shared across edges and corners with the neighbours' (geomorphed where their levels differ).
   - GGX lighting from the torch, roughness from ownership, contact shadows and fog.
   - The relief parallax goes, since the geometry does it now.
   - *Checks:* silhouettes at corners, bulges crossing tile edges and corners with no cracks (also at level-of-detail changes), bulges hiding walls behind, and frame time against step 2.
   - **4a done 2026-11-14:** the walls bulge and glint, over the PS1 picture still.
     - **Mesh** (`level.rs` `subdivide`): each level triangle is cut so no edge is longer than `Config::spacing` (1/8 tile). Each of its edges takes points from that edge's own division, so both triangles on an edge share them, and points in the same place are welded.
       - Each vertex knows the faces it's on (its cell a hair behind it, and the cells either side where it's on a cell edge) and the mean of its triangles' different normals.
       - There's one level of detail for now; distance-based levels are left for step 6, if needed.
     - **`displace`** (each frame, before the raster): a vertex moves along its normal by the mean of the page heights of every face it's on, so edges and corners meet, and a corner bulges round.
     - **The raster** draws the moved mesh, writing the triangle id and the flat point it came from with its view depth. `gbuffer` takes the page and uv from the flat point, the depth and motion from the moved one, and duv from the next pixel's flat point on the same plane.
     - **`Surface`** gains `pic` (where the flat point shows in the PS1 picture, whose colour it takes) and `dist`.
     - **The torch's glint** in `show`: GGX with the torch at the eye, roughness from dry to wet with ownership, falling off with the torch's reach. It's on top of the step-3 diffuse relighting.
     - **Checks** (`examples/check.rs`):
       - flat, it matches the CPU G-buffer to uv 2e-4 (interpolated points, against 3e-6 before);
       - with a random height on every page from −0.1 to 0.2, there are no holes in the closed room from four views, including close to the pillar's corner and a room corner;
       - 75 638 of 76 800 pixels' distances moved, and the corner bulges round.
     - **In the game:** 44–45 fps at rest, against 46–47 before.
     - **Left for 4b:** the crate's own textured, lit picture in place of the PS1 picture. Where the PS1 picture hides the flat point (behind a bulge's silhouette), the colour is wrong.
   - **4b done 2026-11-14:** the level under the dream is the crate's own picture (`Step::own_picture`, `--dream-picture own|ps1`).
     - **What the game gives** (`set_level` with `LevelTri`, `set_textures`), all from the level's triangle features and `LevelTextures`:
       - each corner's uv;
       - the baked light (`col`, 128 as is);
       - the torch albedo (`alb`);
       - the texture (its strange step at the triangle's morph).
       - The 64×64 palettised textures go in as an array.
     - **The raster** writes the triangle id, the texture times the light and times the albedo (both RGBA8, halved), packed in one `Rgba32Uint` target, because the bytes per sample limit is 32.
     - **`own_picture`** puts the torch on each pixel at its bulged point (the PS1's formula: falloff squared, Lambert from the flat normal), clamps, fogs with the fog colour, and writes over the picture where the level shows and no creature is nearer. On still frames it's redrawn from the last view.
       - Its first version skipped still frames, so the PS1 picture showed through whenever the camera rested; comparing against `--dream-picture ps1` found that.
     - **Against the PS1 picture**, not dreaming: a mean of 4–6 of 255 apart, about 8–10% of pixels over 24. The remaining differences are by design: the torch per pixel, perspective-correct texturing, and no 5-bit colour or dither.
     - **Frame rate** unchanged: 46 fps at rest, 20–23 with a creature in view.
     - The diffuse relighting by the dreamed height (in `show`) is still the ratio against the flat wall's, applied to the whole colour. Splitting it to the torch term alone would need the light parts kept to `show`.
5. **Compositing.**
   - Creatures depth-tested against the dreamed depth.
   - The PS1 layer (sword, HUD, people, things) over the top with its depth.
   - Doorways and linked levels: the joined level is drawn through the doorway as it's drawn now, by the PS1 renderer, until the crate handles more than one level.
6. **Speed**, once the look is settled:
   - overlapping the creature net with the dream;
   - LOD tuning;
   - dreaming every other frame with creatures in view.

## The size of the dream (2026-11-14)

- Each octave can see the picture at any scale (`Step::scale`, 0.25–1.5, coarse; `Step::fine_scale`, 1–3, fine; `--dream-scale`, `--dream-fine-scale`, panel sliders). Smaller dreams bigger things.
  - Below 1×, `prepare` averages up to 4×4 looks a pixel, so the net doesn't see an aliased picture. `screen_step` maps the step back at the same scale.
- At rest on one view: 0.35/1.5 gives big but streaky noise; 0.6/2 gives big shapes, with a dog's face; 1/3 is the default; 1.5/3 gives dense small folds.
  - Frame rate: 60 (capped), 60, 43 and 28 fps.
- **Next, if wanted:** a per-pixel size, by blending the steps of octaves at a few sizes with per-pixel weights. Weighted by distance, things stay one size in the world; times strangeness or a painted value, rooms can dream big or fine.

## The robust ResNet-50 (2026-11-14)

- The renderer runs either network, chosen per frame (`Step::net`; `Config::nets`). Pass several weights to `--dream-atlas` with commas, then switch in the panel ("dreaming with") or with `--dream-net 1`.
  - `train/export_resnet.py` writes the robust ResNet-50 (L2 eps 3) with batch norm folded into the convs. It also writes its ImageNet input normalisation as `input.scale` and `input.offset`.
  - The net's input is now plain rgb, and each net's normalisation is a step of its own plan. GoogLeNet's is 2 rgb − 1.
  - Bottlenecks are built from the existing convolutions, plus an `elementwise` kernel for the shortcut copy, its gradient, and the ReLU after the sum.
- **Against PyTorch (`check`):** activations match to 1e-6, gradients to cosine 1.000000. GoogLeNet is unchanged.
- **Speed:** the frame time is about 2× GoogLeNet's with `layer3` at full size. At `--dream-scale 0.6 --dream-fine-scale 2` it runs at 39 fps; `layer2` alone runs at 58 fps.
- **The look:** much bigger, coherent shapes, with whole eyes and sweeping contours.
  - At GoogLeNet's step it saturates into black-and-white grating, because its gradients agree from step to step.
  - The panel keeps each net's own steps: ResNet's are 0.004 and 0.003, and `Step::grad_blur` (a Gaussian on the gradient, in net pixels) is 2. That turns the grating into broad zebra stripes.
  - `layer2` makes bold coloured blobs, `layer3` stripes and eyes, `layer4` broader painterly bands. `layer3,layer4` together runs at 17 fps.
  - A fine dash grain is still there, and the dream fills up to its colour limit at rest. The next levers are a lower `limit`, more decay, or a transformation-robustness trick (a random scale or rotation each step, not just a shift).

## Whole features in the height (2026-11-14)

- **Why the height stayed fine:** the net only sees the height through shading, and shading through slope. So its push on the height is a difference of neighbours: strong on bumps a few pixels across, nearly blind to a broad dome.
- **Fixes to the height's growth:**
  - **Patch seams:** the fine octave's patches left their outlines. At a patch's edge the net sees past what its step can reach, so each patch left a colour seam that the next steps took for an edge and grew height along. Both steps now fade out over a sixth of the patch towards its edges (`screen_step`, `g1`).
  - **Wrap seams:** `shade_back` and `blur_height` now take neighbours in the picture's own places, not across the jitter's wrap seam.
  - **Unread pixels:** below 1×, `net_height` averages every screen pixel a net pixel covers. Before, some rows and columns were never read and drifted.
  - **Flattening:** the height used to pile up flat against its limit. Now outward pushes shrink as it nears the limit, and `height_decay` (`--dream-height-decay`, panel) lets a share go each step, so it settles where the net holds it.
- **New knobs:**
  - `height_spread` and `height_power`: the coarse octave's height step blurred at several sizes, each twice the last, with bigger ones weighing more, to integrate the difference back. Bigger shapes do rise, but long straight creases grow across the screen once it weighs the big sizes much (power 1 and up). It seems to be feedback: the net reinforces any straight crease as an edge. Power 0–0.5 at 2–3 sizes is about the limit.
  - `height_from_colour` (`--dream-height-colour`, −1..1): part of the height's step is the brightness of the net's colour step, so the height rises in the shapes the dream paints. It stays smooth and follows the features.
- **Best so far:** the robust ResNet on `layer2` with `--dream-height-colour 0.5 --dream-height-spread 3 --dream-height-power 0.5 --dream-height-decay 0.01` grows bold, smooth, rounded relief. On GoogLeNet the same settings keep its dog faces and grass, with no seams.

- **Screen-edge seams:** the coarse octave's steps now fade out over a twelfth of the frame at the screen's edges. The net sees the jitter's wrap seam there, and what grew lopsided at the edge used to be carried in as a dark seam when the view turned.
- **Matte light on the height (`Step::show_wrap`, `--dream-wrap`, default 0.5):** `show` lit the height by Lambert against the flat wall's Lambert. Where the wall is seen glancing that ratio ran from black up to five times as bright. Light that wraps round (half-Lambert at 1) keeps it to about 0.7–2×. The game's glint is lower and rougher too (0.2; dry 0.8, wet 0.4).
- **Deeper relief:** the relief and height limit sliders go to 10 tiles, and the parallax relief takes more steps the deeper it is (12 up to 64).

## Lights from where they are (2026-11-14)

- **The torch is held by the sword:** `View::torch_at`, along the view's axes from the eye, in pages. The game uses right 0.22, down 0.2, forward 0.3, with sliders in the panel. `own_picture` lights the level from there, and the glint uses the half vector between the torch and the eye.
- **The level's lights reach the renderer:** `DreamRenderer::set_lights(ambient, &[Light])` takes each point light's position, colour and radius, or with radius 0 a light from far off (the sun). Each frame the nearest 16 go into `mem` after the views, the torch first.
  - The game gets them from `Scene::lights_now`: the same ambient, sun and point lights `shade` bakes into the level, turned lurid by the strangeness.
- **How the height is lit:** `gbuffer` keeps each pixel's point and the side of its face that's seen (`LIT` = 4 floats). `show` then lights the height by every light from where it is: the torch fading over its reach, the point lights as they're baked (1 / (1 + (d/r)²)), the sun, and the ambient. The sum is divided by the same lights on the flat wall, wrapped round by `show_wrap`.
  - That divisor matters because the level's own light (baked, and the torch on the own picture) is already in the picture. The height adds light only where it turns towards a light and takes it away where it turns from one.
  - Off (`Step::lights` false, `--dream-eye-light`): the old light at the eye.
- **The look:** lit from the hand, the height has a lit side and a shaded side, and reads as relief. From the eye it only had dark rims.
- **Next, if wanted:** a level with dark rooms and placed lights, where the torch matters. The height doesn't cast shadows yet.
- **The dream is lit:** it changes the wall's own colour, so it takes the wall's light. The rasteriser now keeps the level's light, the torch's albedo and the texture apart, and `own_picture` keeps each pixel's light (`LIT` is 8 floats: point, side, light).
  - `seen` multiplies the dream, both the pages' and the screen layer's, by that light (`light_on`, through `mem`'s first floats, `GLOBALS`), `Step::dream_lit` of the way. At 0 it glows as before.
  - In the darkened cellars (`levels/dungeon1.ron`: ambient 0.08, two pools of light, an orange corridor torch, a green light in the east room's corner) the dream is dim in the dark and shows where the light falls. Before, it painted bright shapes over the dark.

## Risks

- **Running in Bevy's render world.** Bevy 0.20's render graph and the `RenderDevice` → `wgpu::Device` access need checking first; step 1 is mostly that. The fallback is the current two-device setup, with textures copied between them on the GPU.
- **The height may stay flat** if the dream prefers paint. The moving lamp, the shading-only steps and a low colour limit are the levers. Step 3 shows this before the displacement work.
- **Things on the walls:** doors, levers and chests are part of the level meshes or the scene's things. They have to be drawn by one side or the other, not both.
- **Memory:** the page atlas is 268 MB now, and height adds about 32 MB. That's fine on an 8 GB RX 580, tighter on 4 GB, where `max_pages` can come down.
- **The training-data and editor paths** use the PS1 renderer and stay as they are.
