# How the dream atlas prototype works

Written 2026-11-13 for whoever takes the idea to video. The code is
`/home/maisy/projects/neural_render/train/dream_atlas.py` (branch `dream-atlas`), the game
plan `docs/dream_atlas.md`. It runs on the CPU in PyTorch over a recorded walk through a
level and writes frames and videos.

## The idea

Real DeepDream (Mordvintsev, 2015), not a network trained to look dreamy. A classifier looks
at the picture; we ask how each pixel should change to excite one of its layers more, and
change the picture a little that way. Repeated on the same picture, the network's ideas grow
out of it: eyes, fur, snouts, dog faces out of moss.

What makes it work on a moving view: **the change is kept in a layer of its own (the dream),
and only a small step is added to it each frame.** The dream deepens over seconds and is still
there on the next frame. In the game the dream lives on the level's surfaces, so it stays put
as the camera moves; anything that keeps the dream attached to what it was drawn on (for
video, carrying it along with the picture's motion) plays the same part.

## Where the dream is stored

The atlas: every face of the level (one tile of a wall, floor or ceiling, found from the
surface normal's main axis and the tile the point lies in) gets a page of 256x256 RGB
residuals, made the first time the face is seen. Each pixel on screen knows its face and
where on it (0..1, 0..1) from the level's G-buffer (which triangle, its plane, the pixel's
ray). Faces are looked up bilinearly, clamped inside the page.

What's shown: `picture = clamp(ps1_frame + D * (1 - fog), 0, 1)`, where D is the atlas
looked up at each pixel (`compose`). Before anything has dreamed, D is 0 and the picture is
the plain frame.

## A dream step (`dream_step`)

1. Compose the picture from the frame and the atlas, with the atlas as a tensor that wants
   gradients.
2. Jitter: roll the picture by a random few pixels (4 at 240 lines, 6 at 720), so the
   network's 8- and 16-pixel grids don't print into the dream.
3. Run the network to the chosen layer; loss = mean square of that layer's activations. For
   a mix of layers, each layer's mean square over its own detached value, summed, so each
   counts alike whatever its scale (`Dreamer.loss`).
4. Backpropagate to the atlas. Autograd carries the image gradient back through the bilinear
   lookups into the texels each pixel showed.
5. Normalise per texel by how much it was seen (the same lookups' weights, summed per texel),
   after blurring both with a 5x5 binomial kernel within each page, so pages seen from afar
   (few pixels per texel) get a smooth update and not every few texels. In screen space this
   step isn't needed.
6. DeepDream's step rule: divide by the mean absolute update over the texels that were seen,
   and add `lr` times that.

After both octaves, each frame: `D <- clamp(D * (1 - decay), -limit, limit)`.

## Octaves and settings

Two steps a frame, because a fixed network dreams features of a fixed size in pixels: the
coarse one grows big shapes, the fine one sharp detail. With only the fine one you get only
tiny things.

| | Coarse | Fine |
|---|---|---|
| Picture fed to the network | whole frame, 320x240 | a quarter of the frame at 960x720, a different quarter each frame, jittered by up to an eighth |
| `lr` (colour 0..1, per step) | 0.012 | 0.008 |
| jitter | 4 px | 6 px |
| `decay` per frame | 0.002 | |
| `limit` | 0.6 | |

Displayed both ways: always at 720 lines ("sharp"), or at 240 while the camera moves,
easing to 720 over a second once it stops ("settle"). The PS1 parts (a stand-in sword) are
drawn over at 240 lines, scaled up blocky.

## Networks and layers

- **GoogLeNet** (torchvision `googlenet` IMAGENET1K_V1, BSD-3, 52 MB). Input normalised with
  ImageNet's mean and std, then its own `_transform_input`. Layers:
  - `inception3b`: strokes, swirls, scales, paisley; cheapest.
  - `inception4c`: eyes, fur, snouts, animal parts. The first run used it: moss walls grew
    dog and bear faces, with eyes.
  - `inception4d`/`4e`: whole faces, dogs, birds, arches.
- **Robust ResNet-50** (Salman et al. 2020, L2 eps 3; `madrylab/robust-imagenet-models` on
  Hugging Face, `resnet50_l2_eps3.ckpt`, 205 MB, the classifier half of it pulled out to
  safetensors by `dream_atlas.py robust-weights`). Same normalisation, no transform.
  Layers `layer2` (texture) to `layer4` (objects). Adversarially trained, so its dreams are
  known to be cleaner and more recognisable, at about three times GoogLeNet's cost.

Comparison runs (being made as this is written, in `runs/` of neurofield): `atlas_g3b`,
`atlas_g4d`, `atlas_gmix` (3b+4d), `atlas_gsplit` (4d coarse, 3b fine), `atlas_r3`,
`atlas_r23` (layer2+3), `atlas_rsplit` (layer3 coarse, layer2 fine). Each has `sharp.mp4`
and `settle.mp4`. The first run, 4c, is `atlas_proto1`.

## What we've seen

- It works and stays on the surfaces as the view moves; pauses visibly deepen what's in
  view.
- Where the picture is dark and regular (brick at night) the dream is mostly rainbow speckle
  and thin branching lines. Bright, textured areas (moss) grow faces. Damping the colour
  (letting D change brightness more than hue) is the next thing to try.
- About 2 s a frame on 8 CPU threads (one 240-line step plus one 480x360 step, forward and
  backward through GoogLeNet to 4c). GoogLeNet to 4c is about 1 GFLOP a 224x224 image
  forward; forward and backward about three times that.

## Planned, not built

- Steering: fine-tune GoogLeNet's later blocks on chosen classes (eyes, mouths, teeth,
  flesh, or brick and stone) and put the classifier on as a 1x1 conv over the layer's map;
  each patch's objective is its class scores weighted by how strange that part of the
  level is.
- The same per-patch scores, on the screen, as a detector: faces that form in the dream
  spawn creatures where they were seen.
