# neurofield

Train a network to render PS1-style first-person dungeons, in the spirit of
King's Field, from geometry + camera, and see what dream-like strangeness
comes out when it gets things wrong.

![PS1 rasteriser (left) and the network (right) on four frames of a walk](docs/comparison.png)

*Work in progress: the PS1 rasteriser (left) and the network (right) after
68,500 training steps, on frames of a walk through a level it never saw. It has
the layout, lighting and textures (bricks, tiles, wood grain, the wire grid),
softened and a little wavy; fine detail such as the door's iron bands and the
far end of a corridor still dissolves.*

## Data generator (`datagen/`)

Rust, using the software GTE and GPU from
[bombpersons/psx](https://github.com/bombpersons/psx) (`psx-gte`, `psx-gpu`, in `external/psx`):
RTPT projection, NCLIP culling, ordering-table sorting (no depth buffer),
affine textures, Gouraud vertex colours, dithering, fog to black. Fully
headless and fast (~600 frames/s on 12 cores).

Levels are grids of rooms at different floor levels, joined by corridors that
climb between them by ramps or stairs, with lintels where ceilings change.
Indoors, corridors usually meet rooms through a door frame with the door
swung open against the wall. Rooms hold props and low-poly creatures in random
poses: skeletons, knights, ghouls and lizardmen (some giant, some armed),
four-legged beasts, slimes and floating eyes trailing tentacles. Every room is
reachable on foot (`cargo test` checks this over 300 levels).

Lighting is baked into vertex colours (ambient, sun, room lamps), plus a torch
carried with the camera that is added at render time, so it is part of each
view's parameters rather than the scene's.

```bash
cd datagen && cargo build --release
```

```bash
datagen/target/release/datagen gen --out data/train --scenes 5000 --views 8 --shard-scenes 250
```

```bash
datagen/target/release/datagen walk --out data/walk0 --frames 1500 --seed 42
```

```bash
datagen/target/release/datagen textures --out data/textures.png
```

`gen` samples views from first-person walks by default (`--camera fp`): fixed
eye height and lens, no roll, gaze mostly level, turning to face the way it
walks and sometimes stopping to look around. `--camera free` gives random
cameras with varied height, tilt, roll and lens instead. `walk` writes one
continuous walk, as PNG frames plus the same arrays.

Twelve fixed 64x64 textures (eight for the level, three for creatures, one for
doors). Each shard holds `tris.npy`, `tri_offsets.npy`, `views.npy`, `images.npy` and a
`preview.png` contact sheet; `format.json` documents the layouts and the
shading. Scenes average about 650 triangles. One frame is 230 KB uncompressed,
so 1000 scenes × 8 views ≈ 1.9 GB, generated in about 13 s on 12 cores.

Rendering rules worth knowing when reading the data:
- Geometry nearer than H/2 is not drawn (the GTE divide saturates there),
  and big triangles near the camera are subdivided before that test.
- Triangles are sorted by mean depth, not z-buffered, so overlaps can be wrong,
  exactly as on the hardware.
- Vertex colour 128/255 is neutral texture modulation; lighting is baked into
  the colours, fog is applied at render time from camera depth.

## Realtime viewer (`viewer/`)

Walks a generated level live, drawn by the PS1 rasteriser or by the trained
network (on the CPU with ONNX Runtime, about 30 fps on 12 cores, or on the GPU). The
quick way, which exports the newest `runs/*/ckpt.pt` if needed (setting up
`.venv` the first time), builds the viewer and opens it:

```bash
./view.sh --seed 7
```

By hand: export a checkpoint, using a venv with `onnx`, `onnxscript` and `onnxruntime`:

```bash
.venv/bin/python train/export_onnx.py --ckpt runs/colab2/ckpt.pt --out runs/colab2/model.onnx
```

```bash
cd viewer && cargo run --release -- --model ../runs/colab2/model.onnx --seed 7
```

The network can also run on the GPU, as hand-written WGSL compute shaders
(`viewer/src/gpu_net.wgsl`) through [wgpu](https://wgpu.rs): Vulkan, Metal or DX12,
in pure Rust. It reads a `.safetensors` export (`train/export_weights.py`), which
`view.sh` makes when given `--gpu`:

```bash
./view.sh --gpu --seed 7
```

The whole frame is one submission of about 140 small kernels over one buffer, with
no per-shape compiling. On an RX 580 it takes about 10 ms a frame against 28 ms for
ONNX Runtime on a Ryzen 5 3600, and matches PyTorch to within 4e-6. On Linux the
GPU needs access to `/dev/dri/renderD*`, which a remote-desktop session may not have.

Space switches between rasteriser and network, B shows both side by side, Tab
switches between the auto-walk and driving yourself (arrows or WASD, Q/E to
strafe, PgUp/PgDn to look), N makes a new level, P pauses. F and G step
through dream effects that disturb the network and drift slowly with time:
textures morphing into each other, triangles forgotten and remembered,
the level rippling in slow waves, the attention smeared or made stark, fog, torch and lens
wandering, and all at once. With the same
seed the level and auto-walk match `datagen walk --seed S`.
`--check N --out DIR` skips the window and writes the first N frames' tokens
and network images, to compare against `train/data.py` and PyTorch.

The psx crates come from a git submodule pinned to a known commit, so clone with:

```bash
git clone --recurse-submodules https://github.com/bombpersons/neurofield
```
