# The dream atlas: DeepDream that grows on the walls (experiment)

Status: proposed 2026-11-13, branch `dream-atlas`, nothing built. It replaces, for the dungeon,
the level net and its per-pixel stage (`docs/deferred_stage.md`): that net drew the plain
walls almost perfectly but every fold of flesh as a smear, because the folds it was graded on
were never in its inputs. Here nothing is graded against a target.

## The idea in one paragraph

The PS1 renderer draws the dungeon as it is. Every wall, floor and ceiling face has a patch
of its own in one big texture, the **dream atlas**, which starts empty and is laid over the
face's PS1 texture. Each frame an image network (GoogLeNet, as the original DeepDream) looks
at the picture and says how each pixel should change to excite it more; a little of that is
written back into the atlas texels those pixels show. So the dream deepens a step at a time,
stays on the walls when the player moves, and grows deepest where they stare. The atlas is
drawn at the window's resolution, sharp, while the sword, shield and people stay 240-line PS1.

```
PS1 frame (240 high) ─┐
                      ├─► composite ─► GoogLeNet forward + backward ─► d(picture)
atlas lookup ─────────┘        (dream octaves: 240p each frame, full-res a tile a frame)
                                                                │
atlas texels on screen ◄── gather by depth test ◄───────────────┘
      │
      └─► drawn at window resolution under the PS1 sword, shield, people
```

## The switch

`dream_res`: **sharp** (atlas always drawn at window resolution) or **settle** (drawn at PS1
resolution while the camera moves, easing up to window resolution over about a second when it
stops, so the world pixelates as you walk and the dream sharpens and crawls when you look).
Both read the same atlas; only the display pass changes.

## The atlas

- One page per visible face quad (a wall side of a cell, a floor, a ceiling), with its own
  0..1 coordinates across the face. Pages 256² RGB + a strength channel (how far that texel has
  dreamed), fp16: 512 KB a page, ~150 MB for 300 faces. Faces only get a page once seen.
- 512² pages later, only for faces the player has stood close to (a virtual-texture style page
  table), if 256² looks soft up close.
- Mips: faces seen far off are written at a coarser mip, so a few screen pixels don't scribble
  over a whole page.
- Picture = PS1 texel + atlas residual (RGB in -1..1, clamped). Empty atlas = plain PS1.

## A dream step

1. Composite the frame (PS1 + atlas) at the octave's resolution.
2. GoogLeNet up to `inception4c` (the layer known for eyes and dog faces; `4d`/`4e` for whole
   creatures, `3b` for patterns), objective = mean of the activations squared, gradient with
   respect to the picture, normalised by its mean absolute value (as DeepDream does), with a
   random sub-pixel jitter of the picture each step so the network's grid doesn't print.
3. Gather: every atlas texel of a face on screen projects to a pixel; if that pixel's depth
   matches the face, the texel takes `step * gradient` there. Gather, not scatter, so no
   atomics and every texel is written once.
4. Keep it from running away: residual clamped, a slow pull back toward zero (faster when
   calm or under a clarity potion, which fades the atlas out), step scaled by affliction.

Octaves: a 240p step every frame for large shapes; a full-resolution step on a quarter (or a
sixteenth) of the screen each frame, in rotation, for fine detail. A fixed network dreams
features of a fixed pixel size, so without the coarse octave full resolution gives only tiny
eyes.

Not in the atlas: doors, creatures, people, items, the sword and shield. They're drawn by the
PS1 renderer over the result, depth-tested against the PS1 depth upscaled, so their outlines
stay chunky against the sharp walls.

## Steering it toward eyes, mouths, teeth

Phase 1 uses ImageNet GoogLeNet as it is: whatever it dreams (eyes, dog faces, pagodas).
Phase 2 fine-tunes its last inception blocks as a classifier on our photo sets (`data/eye`:
flesh, eyes, teeth, mouths, suckers, veins, scales, chitin, hair, wrinkles, bone, ~150 each)
and dreams toward chosen classes, so the dungeon grows what we pick, per level or by
affliction. A small fine-tune, minutes on a T4.

### Steered by strangeness (Cat, 2026-11-13)

The fine-tuned classes include ordinary ones (brick, stone, planks, moss: from the level
textures and photos) beside the creepy ones (eyes, mouths, teeth, flesh, veins). The
classifier is a 1x1 conv on the `inception4c` map, so it scores every patch; the dream
objective is each patch's class scores weighted by the strangeness under it (per triangle,
`tris.npy` column 34, looked up like the face pages). Grounded walls dream more mortar,
cracks and grain; strange ones dream eyes and mouths, and a wall turns as its strangeness
rises.

### Faces in the dream become creatures (Cat, 2026-11-13)

On the screen, not the atlas: the creatures are what the player's dreaming vision makes of
the walls, illusions of the view included (a face half moss, half doorway shadow). The
detection head is the classifier's 1x1 conv on the `inception4c` features the dream step
already computes, so it costs next to nothing. Each screen patch's eye / face / mouth score
builds a sighting while it stays high, tracked by the world point under the patch (from the
depth) so small moves don't reset it. Over a threshold, a creature of that kind spawns at that
point and the texels under the patch fade out as it tears free. A switch for when: at once,
while watched, or the next time the spot is out of view (the face on the wall is gone when
you look back). Staring makes the dream deepen, and a deep enough dream makes monsters;
strangeness and affliction set how readily.

## Cost (RX 580, estimates to check)

GoogLeNet to `inception4c` is ~1 GFLOP at 224²; forward + backward ~3x that.

| Pass | Work | Estimate |
|---|---|---|
| 240p octave, every frame | ~4.5 GFLOP | 3-5 ms |
| 960x720 octave, a quarter a frame | ~10 GFLOP | 5-8 ms |
| Gather into the atlas | a texture read a texel on screen | < 1 ms |
| Display at window res | a texture read a pixel | < 1 ms |

If it's too slow: fewer full-res tiles a frame, or distil `inception4c`'s features into a
smaller CNN on our own frames (feature regression, well posed, unlike the CLIP generators).

## Plan

1. **Prototype in PyTorch, on the CPU** (`train/dream_atlas.py`). Inputs from a new
   `datagen atlas-walk`: a walk of a few hundred frames through a dungeon drawn by the engine's
   own PS1 rasteriser, each frame's RGB, depth, face id and face coordinates at 240p and at
   960x720. The script keeps the atlas, runs the steps above, writes frames out both ways
   (sharp / settle) with a PS1 sword pasted over, and a video. Judge: does it look right,
   does it stay put when the camera moves, how fast does it deepen.
2. Pick the layer, the step size, the pull back, the page size from the prototype.
3. **The engine**: atlas pages and the face ids / face coordinates in the PS1 material
   (`engine/src/render`), GoogLeNet forward and backward in WGSL (the conv kernels already
   exist in `gpu_net.wgsl`; backward is the same convs transposed), the gather pass, the
   display pass with the switch, a `--dream-atlas` flag and sliders in the debug panel.
4. Phase 2: fine-tune on the photo sets, classes chosen per level.

## Risks

- DeepDream on a moving camera: each view adds features for its own angle; a feature dreamt
  side-on looks stretched head-on. Probably fine, dreamlike; watch it in the prototype.
- Seams between pages: the gradient is taken on the screen, so it flows across face edges;
  only filtering at page borders needs padding (a texel or two of gutter).
- Runaway to noise if the pull back is too weak; colour drift toward ImageNet's palette
  (fine-tuning may fix it, or a weak pull toward the PS1 texture's colour).
- GoogLeNet weights: torchvision's `googlenet-1378be20.pth` (ImageNet, BSD-3), downloaded to
  `~/.cache/torch/hub/checkpoints`. Its Inception v1 is the original DeepDream's architecture
  but trained differently, so the look may differ from the 2015 pictures.
