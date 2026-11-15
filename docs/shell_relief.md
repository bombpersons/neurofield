# Shell-mapped relief

Plan, 2026-11-14. The dream's height made into real relief at texture resolution, going into the walls as far as it comes out of them: bulges that stand out of the walls and hollows sunk into them, hide what's behind them and are lit from where the lights are. It's drawn by marching each pixel's line of sight through a thin shell over each triangle of the level, not by subdividing the mesh. The dream itself keeps growing on flat walls, so the relief never moves where its steps land.

Follows [dream_world.md](dream_world.md) (steps 1–4b, the size knobs, the lights).

## Where we are

- The raster draws the level **subdivided and pushed out** by the height: triangles 1/8 of a tile across, moved by `displace`, a vertex at a time. Its G-buffer (240 lines) feeds both the dream's steps and the shown frame.
  - So the relief already shifts what the net sees and where its steps land.
  - Anything finer than about a quarter tile shows only as shading. Bulges can't reach past 1/8-tile facets.
- The shown frame (`show`, 3× the picture) looks each pixel up in the base pixel's surface. The `relief` setting (`on_relief`) marches *into* the wall for parallax, but has no occlusion and no outward bulges.
- The height lives in two places: the screen layer (5th plane, shown resolution, flat screen space) while it's carried along, and the pages (128 × 128 a tile), laid down every `remember_every` frames.

## The idea

Two G-buffers a frame:

1. **The dream's: flat.** The level rastered with no displacement, at 240 lines as now. Everything the dream does uses it: the net's picture, its steps, the screen layer and carrying it along, laying it on the pages, the dream lamp. The height still grows, through the lamp's shading of the net's picture, but it never moves anything.
2. **The shown frame's: shell-mapped.** At the shown resolution (3×, up to 1344 × 720), each pixel finds where its line of sight first meets the relief. That gives the hit's triangle, the flat point under it (its *foot*), the height there, the bumped normal and the depth. `show` then shades the hit:
   - the dream from the pages at the foot;
   - the screen layer at the foot's projection on the flat screen;
   - the texture, light and albedo from the triangle at the foot;
   - the lights on the bumped normal;
   - creatures and fog by the hit's depth.

### The shells

- **One prism a level triangle,** its corners pushed out along the triangle's own normal by `+H` and in by `−H` (H = `height_max`). Its sides are flat, because every corner moves along the same normal.
- This handles every triangle alike: whole tiles, parts of tiles at doorways and edges, stairs and slopes. Neighbouring triangles on one plane give prisms that meet without overlapping, so the space over a wall is cut into pieces that don't overlap.
- **Built once at `set_level`,** about 8 triangles a prism. Each vertex carries a −1/+1 that the vertex shader multiplies by `H` from a uniform, so changing the height limit needs no rebuild.
- **Drawn back faces only.** Each pixel gets one fragment per prism its line of sight passes through, even with the eye inside a prism (close to a wall).
  - The fragment works out where the line of sight enters and leaves the prism analytically: the two caps and three side planes, entering no nearer than the near plane.
  - It marches only that stretch. Depth testing keeps the nearest hit across prisms, which is right because the prisms don't overlap.

### The march

- **Along the stretch, height against the line of sight's height above the plane.**
  - A fixed number of linear steps, more the longer the stretch is in height cells (16–64), then a few steps of bisection.
  - A sample's height is the height at its foot: on the plane, through the face table (as `gbuffer` finds a face) to the page.
- **The height read:**
  - from the screen layer at the foot's projection while it's carried along, being the freshest;
  - from the pages where the foot is off the screen, or with no screen layer.
  - A switch compares the two, since the pages lag by up to `remember_every` frames.
- **On a hit:**
  - the fragment writes `frag_depth` (the hit's view depth, so bulges hide each other, creatures and other walls);
  - the triangle;
  - the foot (three floats);
  - the height;
  - the height's slope along the page at the hit (two floats, by central differences, for the normal).
- **A miss** (out through the inner cap, at `−H`) means the line of sight went into the wall's rock: a hit at the cap. Hollows go as deep as bulges stand out, so the inner cap is the deepest any hollow can be.
- **No hit in the stretch:** discard.

### Shading the hit (`show`)

- **The dream:**
  - the foot gives the page and the place on it (`dream_at`);
  - projecting the foot through the view gives the screen layer's pixel (`screen_px`);
  - how much of the wall it owns comes as now.
- **The wall:** the texture, light and albedo by barycentrics of the foot in the level triangle (its corners from a storage buffer, as `LevelTri` gives them), lit as `own_picture` lights it, from the torch where it's held. The dream on it is lit by the same light (`light_on`).
- **The lights:** `lit_by_lights` with the hit's own point and the bumped normal, instead of the flat wall's ratio. The wall's light is the real light on the bump, so no gain over the flat wall is needed.
- **Creatures:** the creatures' depth (a base pixel's) against the hit's depth. Where a creature is nearer, the picture shows through, as now.
- **Fog** by the hit's distance.
- `on_relief` and the `relief` setting go, or stay only for the PS1 picture.

### The frame's order

Today: `displace` → the raster pass → compute (G-buffer, dream, show). With shells:

1. the flat raster pass (dream G-buffer);
2. compute: G-buffer, the dream's steps, the height;
3. **the shell pass** (after the height has grown this frame);
4. compute: `show`.

`record_split` takes a second split, for the shell pass between the dream and `show`.

## Steps

Each step ends in the game, compared with the one before, and with `examples/check.rs` extended.

1. **The dream on flat walls.**
   - The raster draws the level unsubdivided and not pushed out, for the dream's G-buffer. The subdivided, displaced mesh stays behind a switch, for comparing.
   - *Checks:* the G-buffer check as now; the dream's steps the same with any height (moving the height changes nothing but the picture shown).
   - *Done:* `Step::displace` (`--dream-displace`, panel) at 0, the default, draws the level's own triangles, unsubdivided, with no `displace` dispatch; above 0 it draws the subdivided mesh pushed out, as before. `check.rs`: the flat G-buffer matches the CPU's, and with a random height on every page none of its pixels change. Until step 2 the shown frame has no bulging silhouettes, only the shading and `relief`'s parallax.
2. **The shells and the march.**
   - The prisms are built at `set_level`, with targets at the shown resolution (an `Rgba32Uint` for the triangle and packed slope, an `Rgba32Float` for the foot and height, and depth).
   - The march samples page heights first, the screen layer second.
   - *Checks:* a CPU march of the same height (`check.rs`, a few hundred pixels: same triangle, foot within 1/256 of a tile, depth within 0.5%); no holes at a room's inner and outer corners with random page heights; the height 0 gives the flat raster's picture.
   - *Done:* `shell.rs` and `shell.wgsl`; `Step::shell` (`--dream-shell 1`, panel) turns the march on, the page heights times it, the prisms `height_max` times it out and in; `show_shell` (`--dream-show-shell`, panel) shows the hits alone, grey by height and lit by slope, dark red where nothing's met. Its targets are copied into `img` for `show`. Two rules came out of the corners:
     - **Ties at outer corners.** Coming into a prism through a side already under its surface happens where the side plane lies on the next wall, at the same depth as that wall's own hit. Such hits are put a hair (0.05%) farther off, so the real surface wins.
     - **Hollows at inner corners.** A hollow running into the other wall leaves its prism through a side, under the surface, into rock no prism covers, and met nothing (thin red seams along the cellar's wall tops). `rock_beyond` works out at `set_level` what lies beyond each triangle edge: a triangle going on in the same plane, one falling away behind (an outer corner, the other prism's), or one rising in front or none (rock). Leaving into rock under the surface is a hit there.
   - *Checked:* `check.rs` marches 400 pixels on the CPU (2000 steps and halvings, the same tie and rock rules) in three views, flat and with a smooth bump of its own size on each page: every pixel on the same triangle, feet within 4.5e-3 tiles, depth within 0.11%; no pixel meets nothing.
   - *Cost* (RX 580, the cellar, 1344-wide): a few ms looking at walls, about 20 ms looking along the bumped floor, where the stretches are longest (step 5). Grazing stretches show terraces on the bumps' sides at 96 steps.
3. **Shading from the hit.**
   - `show` takes the shell G-buffer: the dream, screen layer, texture, lights, creatures and fog as above.
   - *Checks:* at height 0 the shown frame matches today's flat one; creatures behind a bulge are hidden; the frame rate.
   - *Done:* with `shell` above 0, `show` takes each shown pixel from the shell's hit (dream.wgsl `shell_seen`) wherever it met the relief nearer than a creature. Elsewhere (nothing met, a creature nearer) it falls back to the base pixel, as before.
     - The shell pass now samples the level's texture at the foot and interpolates its light and the torch's albedo (barycentrics of the foot), as the raster does. The wall is lit there as `own_picture` lights it (the torch where it's held, from the hit), fogged by the hit's distance; the dream comes from the foot's page, the screen layer from where the foot shows on the flat screen, and both are lit as the wall is. The lights' shading (`lights_at`, split from `lit_by_lights`) is worked out at the hit with the slope there, and the glint likewise.
     - Two targets of 16 bytes are all the RX 580 takes (32 bytes a pixel), so the foot isn't stored: it's the hit (from its view depth) less the hit's height along the normal. The targets are (triangle, slope, depth, page) and (hit's height, light, albedo, texel).
     - Under the PS1 picture (`own_picture` off), the wall is the picture where the foot shows in it.
   - *Checked:* at height 0, the shown frame with the shell against without: mean difference 0.2–1.2 of 255, 97–99.6% of pixels within 4, most of the rest within a base pixel's reach (the wall's texture is now sampled at each shown pixel, not a base pixel's). In the cellar the floor's dream stands up in lobes that hide what's behind them, and creatures in front still show.
4. **Deep relief, in as far as out.**
   - The height stays symmetric: held within ±`height_max`, bright shapes rising and dark ones sinking around their own mean (the brightness-shaped height taken about its mean, so a dim wall isn't all hollows).
   - The soft limit eases into both ends alike.
   - The game's defaults: `height_max` about 0.5, a bigger step, less decay.
   - *Done:*
     - The limit was already symmetric (±`height_max`), easing in at both ends (a push towards the nearer end shrinks as it nears it). What was missing: each step is now taken about its own signed mean over the octave (`Step::height_centred`, on by default; `--dream-height-uncentred`, panel), so pushes leaning one way (the brightness term especially) don't swell or sink the wall as a whole. In the cellar the lean turned out small: the height's spread barely moved.
     - The game's defaults: the shell on (`shell` 1), `height_max` 0.5, the height's step 0.001 (twice), `relief` (the old parallax) off.
     - **The prisms reach only as far as the height does.** Prisms half a tile thick made the march so long, and so many prisms cover each pixel, that the cellar fell to 4 fps (net 450 ms). Each frame the largest |height| on any page (a max reduction over the 512 pages, ~2M floats) sets how far the prisms reach, a hair past it, within `height_max`; vertex and fragment shaders read it from mem. One value for all of them, so coplanar prisms still meet exactly. Back to 15–16 fps in the cellar with the floor's lobes standing tall.
   - *Checked:* `check.rs`'s CPU march takes the same reach (0.122 with its bumps of up to 0.12, 0.001 flat); all 400 pixels agree in every view, no holes.
5. **Speed and the corners** (outer and inner alike, since hollows meet at inner corners as bulges do at outer ones).
   - Fewer steps far off (by the stretch's length in the shown pixels it spans), and a coarser height far off (a mip of each page's height).
   - The wedge where two prisms part at an outer corner: the height tapered over the last few cells of a page at an outer edge, or the march carried on into the neighbouring face's page.
   - *Done:*
     - **Steps by what a pixel covers.** (The stretch's length on screen can't count: a line of sight's whole stretch lands on its one pixel.) Each step is a height cell along the plane or, far off, where a cell is smaller than a pixel, a pixel's width at that distance, at most 96; then 8 halvings (was 6), which shrinks the terraces on grazing bumps.
     - **Edges by what lies beyond** (`shell::edges`, `Edge`): the wall going on in the same plane, an outer corner (falling away behind), an inner corner (rising in front), or nothing. Rock lies past inner and open edges (step 2's rule).
     - **The height tapers to nothing towards outer and open edges** (`Step::shell_taper`, 0.15 tiles, smoothstep), so no bulge is cut off where no prism carries it on, and no wedge opens. Inner corners aren't tapered: their bulges overlap and depth sorts them, their hollows meet rock.
     - The slope for the lighting is taken from the tapered height across the plane (through the face table, so across a page's edge from the next page's), not within one page clamped.
     - The coarser height far off (a mip of each page) isn't done: the pixel-sized steps already take the far cost down.
   - *Checked:* `check.rs`'s CPU march takes the same edges and taper; all 400 pixels agree in every view, no holes. In the cellar: 18 fps looking along the floor (16 before), 22–24 elsewhere; the stair corridor's walls stand out in relief on both sides, and the steps' edges show no wedges.
   - *After a look with GoogLeNet* (easier to see the stretch on): every corner showed a flat area with the texture and dream stretched over it.
     - The taper itself made them: a height of up to half a tile falling to nothing in 0.15 made a steep, smooth ramp, with what's laid flat on the plane stretched up it. The taper's width is now at least twice the height (`TAPER_RUN`), so the slope stays under about 0.75.
     - Hollows were still cut off at inner corners (a flat rock face with the edge's colours stretched over it). The height now tapers at every edge the wall doesn't go on past, bulges and hollows alike.
     - Two tiles of one wall met with a step: each face's page was read alone, clamped at its edge, so the height jumped where they met, shown as a cliff with the edge stretched down it. `height_at` now reads the height cells across the plane's grid, each from the page of the tile it's in (bilinear, those of tiles with no page left out), so it's continuous across pages; `check.rs`'s CPU march does the same (the worst foot difference fell from 4.5e-3 tiles to 3.5e-5).
     - `show_shell` tints how each hit was met (blue: coming in by a side under the surface; green: going out into rock; kept in the page's top four bits). With all that, the stairs and two strips of the corridor walls still show blue: lines of sight still meet prisms' flat sides there. Tapering hides corners rather than mending them; step 6 does that.
6. **Shells round corners.** The prisms pushed along averaged normals, so the shells of two walls meet at a corner and a bulge or hollow wraps round it, rounded, as the subdivided mesh's did, at texture resolution. The standard shell map (Porumbescu et al., *Shell Maps*, SIGGRAPH 2005): each prism split into three tetrahedra, inside each of which the mapping from space back to the wall (foot and height) is linear.
   - **Why the flat prisms can't.** Each pushes out along its own wall's normal. At a corner the two walls' shells point different ways: apart at an outer corner (a wedge nothing covers), across each other at an inner one. Matching the two walls' heights at the corner isn't enough: the same height `h` at the shared edge is the edge plus `h` along one wall's normal and plus `h` along the other's, two points. The directions have to match too.
   - **The corners' directions.** Each corner of the level (welded by place, as `edges` keys them) is pushed along the mean of the normals of the faces meeting there, each distinct normal once (as `displace` did).
     - A mitre: scaled by 1 / (direction . normal) of its faces (the largest), so the shell stands `h` off each face, not less at a corner; held to 2 at most (a room's corner, three walls: √3).
     - Coplanar triangles share the normal itself, so a flat wall's shells are as now.
   - **The reach at each corner.** Where the geometry is thinner than the reach, the shells' inner sides (and at inner corners their outer ones) fold through each other, the tetrahedra turn inside out and the march breaks: the stairs (treads and risers about a quarter of a tile), a pillar's edge, a doorway's jamb.
     - Each corner's reach is the frame's (the largest |height|, as now, within `height_max`) held to half its shortest *crease* edge (an edge the wall doesn't go on past), or unheld at a corner with none (a flat wall's middle, so splitting a wall into small triangles doesn't make it shallow).
     - Inside a prism the reach is interpolated, and so are the pushed corners; the height on the pages is held to the reach there (as `lim` holds it now), so a bulge on a stair is shallower than one on a wall, never folded.
     - The vertex shader reads the frame's reach from mem as now and takes the least of it and the corner's own.
   - **The tetrahedra.** Each prism (its corners at -reach, then +reach) split into three, the diagonal of each side quad chosen by the corners' welded ids (from the lower id), so two prisms sharing a side split it the same way and nothing cracks.
     - Each tetrahedron's corners carry where they are, their place on the triangle (barycentrics) and their height coordinate (-reach to +reach at that corner).
     - Drawn as now, back faces only: four triangles a tetrahedron, twelve a prism (eight now). Each fragment works out where its line of sight enters and leaves its tetrahedron (four planes, not a prism's five), entering no nearer than the near plane.
   - **The march in a tetrahedron.** Along the stretch everything is linear: the point's barycentrics on the triangle (its foot) and its height coordinate, both from the tetrahedron's four corners by the point's own barycentrics in it.
     - Met where the height coordinate is at or under the height at the foot (as now, `above`), by even steps (a height cell, or a pixel far off) then halvings.
     - Out through any face without meeting it: the next tetrahedron has it (in this prism or the next, round a corner as along a wall). Out through the inner shell: met there, the deepest a hollow goes (as now).
     - So the rock rule, the side-entry tie and the taper all go. `edges` stays, for the creases and the corner heights.
   - **The height at a corner.** A bulge stays whole round the corner only if both walls give the same height at their shared edge.
     - Near a crease (within the taper's width, at least twice the height there, as now), each wall's height is mixed towards the mean of its own and the other wall's at the nearest point of the edge, all of it at the edge. Both walls reach the same height there, so the shell is continuous across.
     - The other wall's height at the edge: its face at that point by its own orientation (each edge's neighbour's orientation stored with the triangle), its page, the cells across the plane as `height_at` reads them now.
     - At an open edge (nothing past it) the height still tapers to nothing, as now.
   - **What the hit gives `show`.** The foot is no longer the hit less its height along the triangle's normal, since the height runs along the interpolated direction.
     - The hit's place on its page (two halves) replaces its height in the second target. The page and the dream come from that exactly.
     - The foot for the screen layer, the picture's smooth blur and the lights is the hit dropped onto the triangle's plane. It differs from the true foot only near corners, by a fraction of the height, where nothing shows the difference.
     - `show_shell` takes the height from the page at that place.
   - **Steps.**
     1. The corners' directions and reaches, and the tetrahedra; the march in them with a height of 0. *Check:* the shown frame matches step 5's flat one; no pixel meets nothing.
     2. The march against the pages' height. *Check:* `check.rs`'s CPU march through the same tetrahedra (fine steps): same triangle, foot and depth as now, at 400 pixels a view. The pillar's corners with a bump crossing them: no pixel meets nothing, and the hits either side of the edge differ in height by under a height cell.
     3. The height at corners, and the old rules gone. *Check:* in the game with GoogLeNet, no flat stretched areas at the cellar's corners or on the stairs; `show_shell` has nothing tinted blue or green to show; the frame rate against step 5's (18 fps looking along the cellar's floor).
   - **Cost.** Three times the volumes, each about a third as long: about as many steps a pixel, the same height lookups a step, plus a second lookup in the band near a crease. Twelve triangles a prism to draw instead of eight. Expected to be about as fast as step 5, to be measured.
   - **Risks.**
     - *The reach's rule misjudges a feature.* Thin geometry with no short crease edge (a long, thin ledge whose short sides are T-split elsewhere) can still fold. A check at `set_level` (each tetrahedron's volume positive at the largest reach) can find them and hold those corners' reach down.
     - *Corners where more than two walls meet,* and the things after the level's own triangles (doors, levers, keys): their normals are averaged the same way, and a door is rebuilt with the level when it moves (`set_level`).
     - *Stretching on steep sides* is unchanged: the dream and texture are still laid on the flat page, so a steep bulge stretches them. Rounded corners are no steeper than walls' bulges, so they don't add to it.
   - *Done:*
     - **Directions, reaches, tetrahedra** (`shell::pushed`, `Pushed`, `TETS`): as planned. Each corner's direction is the mean of the distinct normals at its welded place (those within 120° of the triangle's own, so a thin wall's two sides keep theirs), mitred, at most 2. Its reach is held to half its shortest crease edge, then halved again wherever a tetrahedron would turn inside out at a reach of 1. In the check room that holds the low block's corners to 0.0625–0.25 and leaves the room's own corners unheld. The tetrahedra are drawn as planned, 12 triangles a prism, each listed so its volume is positive. The plan's middle tetrahedron at first had negative volume, so the inside-out check halved every reach to nothing.
     - **The march isn't linear in the tetrahedra.** The tetrahedra's own linear mapping bends the height 0 off the wall's plane wherever the corners' directions differ, by about the reach times how much they differ: 0.005 tiles at a reach of 0.01, a quarter of a tile at the game's 0.5. A flat wall's dream visibly slid about at corners. So the tetrahedra only make the fragments and bound each one's stretch of the line of sight. Along it, each point's place is worked out from the prism's exact mapping: x = Σ b_k (corner_k + s · dir_k · reach_k), with s from -1 on the inner shell to +1 on the outer, and height s · Σ b_k reach_k.
       - Solving for s is a cubic, two Newton steps from the last sample's.
       - Height 0 is then exactly the wall's plane. The inner and outer shells are exactly the tetrahedra's flat faces, so the inner shell is met there with no special case.
       - It's precise in f32 however thin the shell: the foot within 1e-5 tiles at a reach of 0.001. The linear mapping, inverting thin tetrahedra, was off by 4e-3.
     - **Met on coming in under the surface** stays, but only as the hand-over between tetrahedra (one stops a hair short of their shared face, the next starts under the surface). The side-entry tie, the rock rule and the taper at creases are gone. Open edges still taper.
     - **Heights at creases:** each triangle's height near a crease is eased by half the difference between the wall beyond's height and its own at the nearest point of the edge (over max(`shell_taper`, 2 × that), smoothstep). Both walls reach their mean there. Edges meeting at a room's corner each ease separately, so right at a three-wall corner the walls only roughly agree.
       - Read in the march, the wall beyond's height cost half the frame (7–9 fps).
       - So a small compute pass (`crease.wgsl`, before the march) samples each crease of each triangle a height cell apart. The march reads one value per nearby crease, interpolated along the edge.
     - **What `show` gets:** the second target's first word is now the foot's place on its tile (two unorm halves). `show` and `read_shell` rebuild the foot from it, on the tile the hit lies over. `show_shell` greys by the hit's height above its plane, and its blue and green tints are gone.
   - *Checked:* `check.rs`'s CPU march goes through the same tetrahedra with the same exact mapping (in f64, Newton to convergence, 700 steps a tetrahedron), and reads the crease heights directly rather than from samples.
     - All 400 pixels agree in every view: flat, bumped, and with a bulge and a hollow across the pillar's corner. No pixel meets nothing.
     - Flat feet are within 2.5e-5 tiles. Bumped feet are within 4e-4 tiles, with depth within 0.02%.
     - Across the pillar's corner, the 736 neighbouring pixel pairs either side of the edge hit at most 0.0026 tiles apart (a height cell is 0.0078).
     - At height 0, the shown frame against the frame without the shell differs by a mean of 0.2–0.4 of 255.
     - In the cellar with GoogLeNet, the stair corridor's streaks (the texture stretched down the side walls and over the steps) are gone, and `show_shell` has no blue or green.
     - About 17–19 fps in the stair corridor, against step 5's 21 there. The march itself is no costlier; the difference is the tetrahedra's three fragments a prism and the crease pass. Looking along the floor wasn't measured again.

7. **Smoother relief** (after a look at step 6: the depth too noisy, and the relief as detailed far off and glancing as close up, where the dream itself is laid thinner).
   - *Done:* each frame before the creases, the pages' heights are smoothed (`smooth.wgsl`, in `Shell::record`), and the march and the crease pass read the smoothed heights.
     - `mip_pages`: each page averaged down two by two to two cells a side, one workgroup a page.
     - `smooth_faces`: each face's page rebuilt at full size from a level of those mips by a cubic B-spline. Past its tile's edges it reads the neighbouring tiles' pages in its plane (found through the face table), so there's no seam where tiles meet.
     - `Step::shell_smooth` (`--dream-shell-smooth`, panel) sets the level: 0 as dreamed, each level twice as broad, with fractions mixing two. The game's default is 3.
     - `Step::shell_far` (`--dream-shell-far`, panel) mixes in a version two levels smoother by the view depth, reaching all of it 4 / `shell_far` tiles off. The game's default is 0.5. It goes by depth, not the angle, so every prism agrees on a point's height.
     - Smoothing also flattens: the relief's amplitude falls as its shapes broaden, which `shell` (the height's gain) can make up.
   - *Checked:* `check.rs` smooths the same heights on the CPU (`cpu_smooth`): the GPU's smoothed pages are within 2e-8. With smoothing at 2.5 and `shell_far` 0.6 on the bumped pages, the march agrees with the CPU's at all 400 pixels in each of the three views, with no holes. (The check now gives every face in sight its page before listing them; a page given out later is clear on the GPU, where the CPU had left it out.)
   - In the cellar the relief at 3 rises in broad, soft shapes, and at 5 it's nearly flat. The cost is about 1.3 fps (15.7 to 14.4 in the stair corridor), from smoothing every face's page every frame. Only the pages in sight need it, if the cost matters.

8. **Speed, the look unchanged** (the cellar's stair corridor, RX 580: 13 fps before, 24 after the shell's work, 28.5 after the net's).
   - *How it's measured:* `GAME_DREAM_PROFILE=1` (profile.rs) puts GPU timestamps between every dispatch and pass on every 15th frame, and prints the mean milliseconds of each kernel every 20 timed frames.
   - *Before:* 67 ms of GPU a frame. The march 28 ms, the smoothing 14, the dream's net ~20 (Matmul 9.4, Store 4, Col2im 4), the rest ~3.
   - *Done, each checked by `check.rs` (all pixels agree with the CPU, no holes; the pages' store unchanged to the last digit):*
     - **Smoothing** (14 to 1.6–2 ms): each level's patch of mips, with the two cells round it, is read into workgroup memory once, and its offsets worked out once. Faces with no page leave at once.
     - **Inner caps first** (28 to 20 ms): a depth-only pass of the prisms' inner faces, seen from above. A line of sight reaching an inner shell has met the relief there or before, so a tetrahedron it comes into past the nearest inner cap is let go before marching. (The exit test at the inner shell now uses the last marched sample, whose height is converged; the early guess made 18 holes.)
     - **Steps as long as the smoothing allows:** half the smoothing's cells, not one cell. The height can't change much within that. It saved little: the steps weren't the cost.
     - **The reach from the smoothed heights' largest** (worked out in the smoothing pass): thinner shells and a narrower crease band. Little in the cellar, where the smoothed peaks stay near the limit.
     - **Height reads** (17.7 to ~14 ms). Tried first, the height read as a formula took the march to 5 ms, so the reads were the cost:
       - The four cells of a sample are read from one page lookup when they're on one tile, as they mostly are.
       - The far-off smoothing is mixed in once per cell in the smoothing pass, by the cell's view depth, rather than at every sample by the sample's. This is the one change that isn't bit-identical; the difference is far smaller than a pixel. `check.rs`'s CPU mixes the same way.
       - Each triangle's home tile and its page are found once a frame (`tri_homes`) and kept to hand in the march.
     - **Store** (4 to 1 ms): each 16 × 16 block reads its accumulators (and the two texels round it) into workgroup memory once for its 25-tap spreads, rather than 100 atomic loads a texel. Texels the height doesn't keep are skipped.
   - *Tried, and not the cost:* steps capped at 8 (17.7 to 17.3 ms), 2 halvings not 8 (16.5), creases and tapers off (17.0). Page lookups made free took it from 13.8 to 10.7.
   - *Then:* ~38 ms of GPU. The march 14–15 (varying with the dream's heights), the net ~17 (Matmul 9.4, Col2im 4), smoothing ~2, the rest ~4.
   - *The net, done* (`PROFILE_SHAPES=1` with the profile labels each dispatch by its sizes; `check.rs` takes `GAME_DREAM_PROFILE` too). Each checked by `check.rs`: GoogLeNet's and ResNet's activations and input gradients against PyTorch's as before (the gradient's 1.9% largest difference was there before).
     - **Col2im** (the frame's input gradient through the first layer, 3 channels, was 7.8 ms of the check's frame, now 0.9): a thread a (pixel, channel) rather than a workgroup of 64 over the channels, which left 61 of 64 idle; and only the taps the stride lands on visited. Bit-identical.
     - **`matmul_big`** for products with at least 200 tiles of 64 × 64 (the early layers'): 8 × 8 a thread, k in steps of 8 prefetched, reads coalesced along k, the workgroup's pieces padded against bank conflicts. The biggest goes from 1.35 to 0.89 ms. On fewer tiles the 32 × 32 `matmul` wins, so it keeps those. (A k step of 16 or no prefetch was slower: at 256 registers only one wave a SIMD runs.) The same sums in the same order.
     - **Split k** for products with fewer than 160 32 × 32 tiles (inception 4's, 15 × 20 pixels): each part of k a workgroup, its sums to a scratch room, `matmul_sum` adding the parts in order. The only change that isn't bit-identical: the order of the float sums, differences ~1e-5 of activations of ~7.
     - *Tried, and slower:* `matmul`'s 16 k steps as a loop (64 registers, not 184, but the small products got slower overall).
   - *The march again, done* (the march's sums the same to the last digit: `check.rs`'s comparisons with the CPU unchanged in every figure):
     - **The smoothed heights in a texture** too (`Shell::heights`, 16 pages a row, written by `smooth_faces` as it writes `near`): a sample's four cells come in one `textureGather` when they're on one page, rather than four loads from the buffer. In the game the march went from 11.6 / 13.1 / 14.7 ms at the profile's three printouts to 8.4 / 9.1 / 11.3, the same in two runs.
     - **Fewer registers** (204 to 128 a thread, so two waves a SIMD rather than one, to hide the reads' wait): the triangle's edges, its side planes and its creases' places aren't kept through the march but worked out (or read) where they're used. On `check.rs`'s fixed heights (`SHELL_TIME=1` times the whole frame from each view) frames go from 19–27 ms to 16–22.
     - *Tried, no help:* the corners and directions read from the buffer where used: the compiler keeps them in registers anyway.
     - **Finishing once a pixel** (`resolve.wgsl`): counting showed each pixel marching 3–4.4 tetrahedra, of which about 1 meets the relief by stepping and halving, 1.3–1.7 are entered already under it (behind the nearest hit, mostly) and about 1 passes over it; every one that met it worked out the slope (4 more height reads), page, uv, light and texel, to lose the depth test. Now the march writes only the triangle, the hit's barycentrics and its depth, and a pass over the screen finishes the nearest (`begin` and `finish`, shared). The march's 114 registers before scheduling come down to 111; the resolve pass takes 0.8 ms. Frames from `check.rs`'s views 1.9–2.6 ms faster (12–15%); in the game the march and resolve together 4.7 / 5.5 / 8.3 ms against 6.1 / 7.2 / 10.1 in the faster runs before, about 31 fps in the stair corridor. The march's figures against the CPU are unchanged; 9 of 691200 pixels of the height-0 comparison cross its 4-of-255 line (the finishing's floats rounded in another shader), its mean unchanged.
     - *Tried, no help:* not working out the far end of a tetrahedron entered under the surface (no gain).
     - *Tried, and less accurate:* false position in place of the 8 halvings (3–5 rounds; 10–20% faster, but heights out by up to 2e-2 tiles where the relief bends, against 2.7e-4); 4 halvings (1–2.5 ms faster, heights out by 4.3e-3 tiles, half a height cell). Not kept: they'd change the look a little.
   - *Measuring in the game:* the march's cost depends on the heights the dream grows, which differ from run to run, so runs of the same build fall into faster and slower groups (march / the rest of the GPU's time in the same printout, which cancels the clock: 0.35 / 0.40 / 0.58 in two runs, 0.49–0.51 / 0.56–0.62 / 0.80–0.87 in two others, against 0.66 / 0.74 / 0.86 before). `check.rs`'s fixed heights are the steadier comparison.
   - *Then:* ~31–34 ms of GPU, 26.5–28.5 fps (varying run to run as above). The march ~11.5, smoothing 2, the net and the pages' work ~17 (matmuls ~7.5, col2im 1.2, store 1, bake 0.9, clear_acc 0.8, blur_height 0.8).

## Cost

- **The march:** about 1M shown pixels, up to 2–3 prisms a pixel at grazing angles, 16–64 samples a fragment, 4 reads a sample (bilinear). That's roughly 100–400M reads, a few ms on the RX 580.
- **Steps 1 and 3 save work:** no subdivided mesh and no `displace` for the dream, and `show` no longer marches `on_relief`.
- **Memory:** the shell targets at 1344 × 720 are about 31 MB, and the prisms about 8 triangles a level triangle.

## Risks

- **Corners.** Going both ways, each kind of corner has a gap.
  - At an outer corner the two prisms part outside the walls, where a bulge crossing the corner is cut off. They overlap inside, where a hollow near the corner meets the other wall's prism.
  - At an inner corner it's the other way round: bulges overlap, and hollows crossing the corner are cut off.
  - Overlaps are settled by depth; gaps show as a wedge. Tapering the height near a page's edges hides both (step 5), at the cost of flat, stretched ramps where it's steep and lines of sight still meeting prisms' flat sides; shells pushed along averaged normals mend them (step 6, done). What's left: three walls meeting at a room's corner agree only roughly right at the corner, and thin features the reach rule misjudges are held only by the inside-out check.
- **The height's lag.** Pages lag the screen layer. Marching the screen layer's height by projection needs the foot on the screen, and the screen layer's resolution falls at grazing angles.
- **Grazing views** need long marches. Capping the samples bands the relief at a glance; bisection hides most of it.
- **Creatures** are composited by a base pixel's depth, so their outlines against a bulge are as blocky as the picture's pixels (as now).
- **The net sees flat walls.** It shapes the height only through the dream lamp's shading, never by how a bulge looks from the side. That's the point, but bulges it never sees may grow in ways it wouldn't choose.
- **Steep bulges stretch the dream.** The dream's texels are laid on the flat page, so a steep bulge's side shows them stretched. Small height slopes keep this mild; outward relief with soft shapes suits it.
