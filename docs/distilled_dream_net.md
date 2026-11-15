# A dream net of our own: a small, robust CLIP student with depth

Plan, agreed in outline 2026-11-15 (branch `dream-atlas`). Follows `dream_atlas.md` and
`shell_relief.md`. Not started: it waits for the Kaggle creature distil run
(`neurofield-distil`) to finish.

## Why

The dream atlas dreams with a fixed ImageNet network. What we want from it:

- **Recognisable things, steered by a room:** a dog room, a tree room, lore images, a door that
  appears where a secret door is.
- **The gradual look:** the dream grows a little each frame, over many frames.
- **A height to stand the relief off the walls** (`shell_relief.md`).

What we've found:

| Network | Steering | Look |
|---|---|---|
| GoogLeNet (layers) | none | busy strokes, the classic DeepDream look |
| GoogLeNet (classes) | 1000 ImageNet classes | psychedelic texture with hints of the class; it's sure of the class from invisible noise, so the confidence step stops at once |
| Robust ResNet-50, L2 ε3 | 1000 classes | real objects (gorillas, doors, jellyfish), photo-like with the confidence step |
| Robust Wide ResNet-50-2 ε3 | 1000 classes | the most photo-like, but heavier |
| Robust ResNet-50 ε1, ε0.5 | 1000 classes | noisier, murkier |

So robustness is what turns "pushing pixels at a network" into pictures of things. CLIP would
replace the 1000 classes with **any text**: "a door made of bone", "a cathedral of teeth". But
plain CLIP isn't robust, and the robust CLIPs released so far (FARE, TeCoA) are large ViTs.

The plan is to distil our own: a **small, robust student** that copies CLIP's image embedding,
with a **depth head** beside it, built from blocks `dream-renderer` already runs.

## The student

- **Body:** ResNet-style bottleneck blocks, so no new shader kernels. Somewhere between ResNet-18
  and ResNet-50 in cost, picked by the pilot. It must stay within a few ms forward and back at
  the dream's size on an RX 580, like the robust ResNet-50 now.
- **Embedding head:** a mean pool, then a linear layer to CLIP's embedding size. This replaces
  CLIP's attention pooling. Steering works like class steering does now (`Net::classes`, the
  `confidence` kernel): the goal is the cosine between the pooled embedding and a **text
  embedding baked offline** per room or prompt. Its gradient at the last block is still one
  vector over the pixels, plus a normalisation term.
- **Dense embeddings, to steer *where* (adjustment 1, after the research):** a pooled embedding
  steers the whole picture ("this room is doggy"); a secret door needs "a door, *here*". So the
  student also learns the teacher's **per-patch** embeddings (its patch tokens through the
  projection, as dense-CLIP methods use them), from the same linear layer before the pool. In the
  game, a masked region of the picture (or of a wall's pages) is steered towards its own prompt:
  the goal is the cosine at each patch inside the mask, so its gradient is that patch's
  embedding direction at the pixels it covers, as class steering seeds a row per pixel now.
- **Depth head:** a small decoder off two or three of the body's levels, giving relative depth at
  a quarter of the input's size. In the game it gives the shell its height directly, in place of
  (or mixed with) the brightness and dream-lamp heights. It's taught by Depth Anything V2 Small,
  or Video Depth Anything Small (steadier from frame to frame, which suits a wall that dreams
  over time).
- The text encoder never runs in the game. Prompts are embedded once by the teacher's text
  encoder and stored next to the weights, like `imagenet_classes.txt`.

## Teachers

| Teacher | Job | Licence |
|---|---|---|
| OpenAI CLIP ViT-B/16 (or ViT-L/14) | image embedding to copy; text embeddings for prompts | MIT (weights); trained on undisclosed web data |
| Depth Anything V2 **Small** | relative depth targets | Apache-2.0 (Base, Large and Giant are CC-BY-NC: not usable) |
| Video Depth Anything **Small** (alternative) | depth steady across frames | Apache-2.0 (Large is CC-BY-NC); confirm against the upstream LICENSE before use |

Teachers run offline once per training picture (or per crop), with their answers cached.
ViT-B/16 and Depth Anything V2 Small are already in the local Hugging Face cache.

## Data

The game may be sold, so everything must be usable commercially.

- **General photos: Open Images V7** (images CC BY 2.0, annotations CC BY 4.0). The student only
  knows what it has seen, so prompts like "gorilla" need gorillas in the data. Plan on about a
  million images, starting smaller for the pilot. We keep each image's source and licence from
  Open Images' metadata, for **attribution**. Google doesn't guarantee each image's licence, so
  get a legal look at the training data before release. **Not ImageNet** (non-commercial).
- **Dungeon frames:** walks through the levels from datagen (`runs/atlas_walk*`, random levels).
- **Half-dreamed frames:** dream offline with the teacher on dungeon frames, so the student has
  seen pictures part way to a dream, as the game shows it.

Crops are taken at the sizes the game dreams at (the coarse octave's frame and the fine octave's
pieces), with jitter.

## Training

- **Robustness, FARE's recipe:** attack the input with a few steps of L2 PGD, then make the
  student's embedding of the attacked picture match the teacher's embedding of the clean one (a
  cosine or L2 loss). The attack size is chosen like the robust ResNets' ε, about 1–3 at 224 px,
  and scaled for our sizes.
- **Dense embeddings:** a cosine loss between the student's per-patch embeddings and the
  teacher's (pooled to the student's grid), alongside the pooled loss, attacked the same way.
- **Depth:** a scale- and shift-invariant loss against Depth Anything's map. Attacks optional.
- **Start from a robust body:** initialise from the robust ResNet-50 ε3 (or its first layers),
  which should cut the training needed. Check madrylab's licence for commercial use first; it
  was trained on ImageNet, so starting from it may carry ImageNet's terms. If in doubt,
  start from scratch.
- **Where:** Kaggle T4s (30 GPU-h a week) using the creature distil run's setup (shards,
  checkpoints, logs), with Modal for any extra.
- **Cost, roughly:** robust training runs several times slower than plain training (each step
  also does the attack). Rough guess: the pilot takes a few GPU-hours; the full run, one to two
  weeks of Kaggle's free hours. To be firmed up by the pilot.

## Steps

1. **Offline look test (running 2026-11-15, below):** dream with plain CLIP in the game's way, to
   see whether CLIP-steered dreaming gives the look we want.
2. **Ramped regional steering, with the robust ResNet-50 (adjustment 2; can start now):** the
   steering strength is a mask that rises over time in a region, so a thing surfaces there
   gradually; a shaped mask (a door's outline) makes that shape surface out of the wall, as the
   research's ramped "illusion" ControlNet does. Offline first, then in the game: masks on the
   wall pages, set per room or per secret door.
3. **Pilot:** a ResNet-18-sized student, a slice of the data (on the order of 100k images), a
   few hours. Then offline dream sheets compared with plain CLIP and the robust ResNet-50 on
   the same frames:
   - whole-frame prompts;
   - **a masked door test** (a door must surface in a given region, and only there);
   - the depth head's maps.
   Go on only if it dreams things rather than noise.
4. **Full run.**
5. **Game:**
   - Export the student (an `export_student.py` like `export_resnet.py`).
   - Check it against PyTorch with `check.rs`.
   - Add a prompt goal next to the class goal.
   - Add the depth head's height to the shell.
   - Prompts in the panel, then per room or area, and per region with the ramped masks.
6. **Optional, only if the student isn't sharp enough: slow diffusion cleanup on the pages.**
   Our dream does the gradual appearing and the depth in real time; a one-step diffusion pass
   occasionally sharpens a wall page in the background at low strength (about a page a second
   on the RX 580, a guess). This is the research's atlas idea crossed with its "robust seed,
   diffusion cleanup". It needs a model whose licence suits a sold game (not SD-Turbo's
   under-$1M terms unless that's accepted).

Flow-warped noise (from the research) needs no change: the dream already rides the walls by
motion vectors and the pages.

## Open questions

- How small can the student be and still dream recognisable things?
- Does the attack need to be at the dream's own scale (the coarse octave), or is 224 px enough?
- Depth head: does its height read better than the dream-lamp height, or should they be mixed?
- Does the student need to see the PS1 picture under the dream (low-res, dithered, fogged)?
- Are dense embeddings from a small student sharp enough to place a door within a mask, or does
  steering bleed outside it?
- How fast should a reveal ramp so it reads as "slowly appearing", not "fading in"?

## Offline look test

2026-11-15. Plain OpenAI CLIP (ViT-B/32 and B/16) on a dungeon frame
(`runs/atlas_walk1/frames/00060.png`), dreamed as the game steps (fixed-size step, small decay,
limit), 400 steps, prompts "a gorilla", "a door made of bone", "a jellyfish", "a stone wall
covered in eyes". Two ways of showing it the picture:
- **"game":** the whole jittered frame once a step, as the coarse octave sees it;
- **"cutouts":** 4 random crops a step, as CLIP-guided art does.

Findings:
- **The game's way:** each prompt is readable, but in the noisy plain-CLIP look: rainbow
  speckle, small faces, and the prompt's **word written on the walls** ("GORILLA", "BONE"). The
  jitter between frames is *not* enough on its own.
- **Cutouts:** much more structured: a gorilla with a face, a doorway framed in bones with a
  skull beside it, a tentacled jellyfish, rows of eyes in stone. B/16 with cutouts is the most
  coherent. Still dim, speckled and busy, though, and nowhere near the robust ResNet's
  cleanliness.
- **What CLIP adds:** composed prompts work ("a door made of bone", "a stone wall covered in
  eyes"), which ImageNet's classes can't express.
- **Robust ResNet-50 on the same frame:** a clean gorilla face, a glassy jellyfish.

So: CLIP's steering is what we want, but not plain CLIP's look. That supports the robust student.
For the pilot, show the student several crops a step (or the fine octave's pieces) as well as
the whole frame.

## Ramped regional steering test (2026-11-15)

Offline, robust ResNet-50 ε3, on a dungeon frame (`runs/atlas_walk1/frames/00060.png`). The class
is dreamed only inside a mask, its strength ramping from 0 to full over 400 steps, with the
network judging a box around the mask (a third of its size as margin) and the confidence step
on:
- **Gorilla in an arch:** a clear gorilla in a jungle by step 200, sharpening after.
- **Sliding door in an arch:** a glazed door, recognisable from about step 200.
- **Jellyfish in a disc:** a purple, tentacled blob; weak.

It surfaces gradually and stays inside the mask. But it reads as a **window cut into the wall**:
a hard edge, its own scene inside it, ignoring the wall's perspective and stone. To try next:
- a feathered mask and strength edge;
- keep the wall's own structure;
- steer on the wall's page (flat, facing the camera) rather than the screen.

## Outside research (2026-11-15)

A web-only research pass, deliberately given no knowledge of this project, only the goals:
recognisable objects (not noisy, not blurry) appearing gradually over many frames, with depth,
in real time, cheap to run and build, directable in what appears and where. Its findings, with
speeds as the authors report them (high-end NVIDIA GPUs):

- **No single published system meets all the goals.** The cheapest combination that does is
  **one-step distilled diffusion run as image-to-image every frame**:
  - SD-Turbo ([ADD](https://arxiv.org/abs/2311.17042)), [SDXS](https://arxiv.org/abs/2403.16627),
    or [StreamDiffusion](https://arxiv.org/abs/2312.12491), about 90 FPS on an RTX 4090.
  - Gradual appearance by **ramping the strength per region**, frame by frame. This is the
    agent's own inference; no paper does exactly this.
  - Direction by regional prompts ([StreamMultiDiffusion](https://arxiv.org/abs/2403.09055)),
    example images ([IP-Adapter](https://arxiv.org/abs/2308.06721)) or ControlNet.
  - Depth from [Depth Anything V2 Small](https://arxiv.org/abs/2406.09414) or
    [Video Depth Anything](https://arxiv.org/abs/2501.12375) run on the output.
- **Its ranked shortlist:**
  1. The above, with no training.
  2. The same, **painted into a persistent texture atlas over the real geometry**
     ([TEXTure](https://arxiv.org/abs/2302.01721)-style), for true depth and parallax and
     gradual build-up. This is the dream atlas's own architecture, with diffusion in place of
     the dreaming network.
  3. A small one-step model fine-tuned to our look ([pix2pix-turbo](https://arxiv.org/abs/2403.12036)),
     optionally with a depth head.
  4. Joint colour-and-depth diffusion ([LDM3D](https://arxiv.org/abs/2305.10853),
     [Orchid](https://arxiv.org/abs/2501.13087)), distilled to a few steps.
  5. **Robust-classifier gradient ascent** ([Santurkar et al. 2019](https://arxiv.org/abs/1906.09453)),
     which is what we do now. Cheapest and naturally gradual, but less recognisable than
     diffusion and limited to ImageNet's classes. It suggests pairing it with an occasional
     low-strength one-step diffusion pass to clean it up.
  6. Real-time video and world models ([Self Forcing](https://arxiv.org/abs/2506.08009),
     StreamDiffusionV2) and 3D scene generators: not viable on modest hardware yet.
- **Ideas it flagged as overlooked:**
  - **Ramped "illusion" ControlNet**
    ([QR Code Monster](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster)):
    a chosen silhouette surfaces out of the existing picture as its weight is raised.
  - **Flow-warped noise** ([Go-with-the-Flow](https://arxiv.org/abs/2501.08331)): noise kept
    attached to the scene as the camera moves, so it doesn't boil.
  - **Depth read from the generator's own features** with a tiny probe
    ([Beyond Surface Statistics](https://arxiv.org/abs/2306.05720)).
  - **Neural cellular automata** ([MeshNCA](https://arxiv.org/abs/2311.02820)): grow gradually
    on meshes in real time, but give textures, not recognisable objects.
- **Gaps it noted:** no published speeds on mid-range or AMD GPUs, and no training costs for
  pix2pix-turbo or DMD2.

Our caveats:
- **Speed:** every real-time diffusion figure is on RTX 4090 or H100 class hardware with
  NVIDIA-only tooling (TensorRT). On the RX 580 through our own shaders, a one-step SD-Turbo
  pass is more likely about a second at 512², so it can't run every frame.
- **Licence:** SD-Turbo's licence is free commercially only under $1M annual revenue (Stability
  Community Licence). Check each candidate's licence before relying on it.
- **Sources:** the agent worked from search results without opening the papers.
