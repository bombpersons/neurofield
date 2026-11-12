# A per-pixel stage for the level net (design sketch)

Status: proposed 2026-11-12, nothing built. Background: `reports/Neural flesh rendering
approaches.md` (the research) and the sculpt pilot (`runs/modal_sculpt1`), whose net drew the
plain level well but every kind of flesh as one blurred ring pattern and ignored its organ
tokens.

## The idea in one paragraph

The transformer keeps the jobs it does well: what is where, what hides what, lighting, fog and
the dream effects, at one token per 16x16 block. A new stage after it works at every pixel. The
plain level is rasterised into a G-buffer that says, per pixel, which texture it shows and
where on it, where it falls on the flesh texture, how far that spot has turned and how high
its fold stands, and where it falls on any eye, sphincter or vein. The stage looks up
**learned latent textures** at those coordinates and decodes them, with the transformer's
features for that pixel, through a small per-pixel MLP. Detail is anchored to the surface and
comes from somewhere that can hold it, instead of being squeezed through a 256-number token.

```
level tris ──► transformer (as now) ──► head (as now, to 16 ch at full res) ──┐
     │                                                                       ├─► per-pixel MLP ──► + coarse RGB ──► frame
     └──► G-buffer raster ──► latent texture lookups, organ coords ──────────┘
```

## Per-pixel inputs (about 48 channels)

| Group | Channels | Where from |
|---|---|---|
| Transformer features | 16 | the current head's last 16-channel layer, before its RGB conv |
| Level texture latent | 8 | latent texture of the pixel's texture id (12 of them, 128² each: the level textures are 64², so two latent texels a texel), sampled at its UV |
| Flesh latent | 12 | one flesh latent per painted slot (8 of them, 256², as the painted flesh), sampled at the flesh coordinates `turning.npy` gives per corner |
| Turning | 3 | the dial's reach (interpolated), the view's amount, the fold height (interpolated over the 6 samples: corners and edge middles, as a quadratic) |
| Geometry | 4 | inverse depth, facing (normal in view space) |
| Organs | 8 | for the eye or sphincter quad over the pixel: kind, local u and v, distance from its middle, and its state (eye showing, blink, gaze x, gaze y; sphincter showing, clench); for a vein ribbon: distance across it, distance along it with its beat |

Eyes get their own latent: an atlas indexed by local (u, v) with gaze applied as a shift of
the iris and blink as a third coordinate (8 blink steps), so a filmed eye has somewhere to
live. The research found every gaze-controllable renderer models the eye explicitly; the net
painting over a token-described eye is exactly what the pilot did.

## The decoder

Per-pixel MLP 48 -> 64 -> 64 -> 3 (GELU), its output added to the coarse RGB the current head
makes. Its last layer starts at zero, so a model loaded from run 7 renders exactly what it did
before training begins. An optional 3x3 conv at 16 channels before it lets neighbours share
detail (0.9 ms more on the RX 580).

Measured on the RX 580 with the game network's own kernels (`engine/examples/pixel_stage_bench.rs`),
240 x 320: the network now 10.3 ms; this MLP 1.9 ms; the lookups 1.5 ms with a naive buffer
kernel. Real textures (four channels a fetch, hardware filtering) and a fused MLP kernel
(weights in workgroup memory, layers in registers) should cut both; about 2-3 ms more a frame is
the target. A faster card buys resolution (cost grows with pixels) or a wider decoder.

## Training data: nothing new to draw

Every input above can be computed from what the shards already hold: `tris.npy` (positions,
UVs, texture ids), `turning.npy`, `organs.npy` (each organ's place, axes, size and state; veins
in lengths), `views.npy`. So the G-buffer is rasterised **in the trainer, on the GPU**, per
batch, and the shards the Kaggle kernels are drawing now (with the T-junction and fold fixes)
can be used as they are.

The rasteriser is written in PyTorch (about 512 triangles a view: tile the screen, test each
tile's triangles, keep the nearest), so it runs on the T4s and, slowly, on the CPU here for
tests. It must agree with the game's: the same projection (`format.json`) and clipping at
view.z = 0.02, and the PS1's affine texturing. The renderer interpolates UVs affinely across
triangles it has first split until no edge is longer than a quarter tile
(`engine/src/render/mod.rs`, `MAX_EDGE`), so the rasteriser splits the same way and
interpolates in screen space. A check draws the G-buffer's UVs through the real textures and
compares against a plain frame.

## Training

1. Load run 7 (transformer and head); the new parts start so the output is unchanged.
2. First ~10k steps: the transformer frozen, the new parts at a higher learning rate. Then all
   of it, the transformer at a tenth of the rate.
3. Loss: L1 + LPIPS at about 1.0 (the research: shallow perceptual features, no Gram loss), and
   L1 weighted 5-20x inside organ quads. No GAN to start with: once the detail can be told
   from the inputs, plain losses can learn it, and the earlier GAN damaged the shared trunk.
4. Watch, beyond val L1/LPIPS: the flesh and plain parts of the frame apart, the organs'
   effect (the frame with and without them), and a 2-D spectrum of the output for the 16-px
   grid.

The stage adds little training compute (about 0.1 TFLOP a step at batch 32). A pilot is 2.5 h
on a T4, from Kaggle's quota after it resets (Saturdays) or Modal (about $2.30); a full run
about 9.5 h.

Later, only if flesh is still soft: an adversarial loss on the per-pixel stage alone, with a
discriminator on frozen pretrained features and R1/R2 penalties, the transformer's gradients
cut off. And a blend with the previous frame along the camera's motion, if it flickers.

## In the game

* A G-buffer pass in the engine: the plain level drawn to a few rgba16f targets (texture id
  and UV; flesh slot and coordinates; reach, fold height; depth and facing), then the organ
  quads and vein ribbons over them into the organ targets. Their places and states are what
  `dreamed::organs` already works out each frame.
* Latent textures uploaded as texture arrays (rgba16f, four channels a layer).
* One new WGSL kernel after the head: read the G-buffer and the head's 16 channels, sample the
  latents, run the MLP with its weights in workgroup memory, add to the coarse RGB.
* The dream effects carry over: `tex_mix` mixes the level latents as it mixes texture
  embeddings now.

## Steps, cheapest first

1. PyTorch G-buffer rasteriser and its check against plain frames (CPU, here). Done:
   `train/gbuffer.py`, checked by `train/check_gbuffer.py`. On 20 plain views from the
   validation shard and a local editor-level shard, frames rebuilt from it match the game's
   to a mean error of 0.010 (the dither and the textures' 16-colour palettes, which the
   rebuild leaves out), 0.4% of pixels off by more than 0.15, and a pixel's shift any way
   doubles the error. About a second a view on the CPU.
2. The model change, export, and a CPU test that the run-7 start renders as before.
3. Pilot, 2.5 h. Decide on the full run from the flesh/organ numbers and the samples.
4. Engine: G-buffer pass, latent textures, the WGSL kernel; compare against PyTorch on the
   same view (as `datagen check` does for the network now) and time it.

## Open questions

* The split into quarter-tile triangles raises the rasteriser's triangle count well past 512 a
  view; it may need to cull by tile first to stay quick on the T4.
* 8 blink steps and a gaze shift: enough for the filmed eye (155 frames, 7 x 5 gazes)?
* Should the transformer shrink once detail lives elsewhere? It is 9 of the 10 ms.
