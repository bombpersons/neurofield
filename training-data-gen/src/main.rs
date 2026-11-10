//! PS1-style training data for a neural renderer.
//!
//! `datagen gen --out DIR [--scenes N] [--views K] [--shard-scenes S] [--seed X] [--camera fp|free] [--creature-views P]
//!              [--levels DIR] [--level-share Q] [--width W]`
//!     Random scenes, K views each, in shards of S scenes. `fp` (default) samples
//!     views from King's Field-style first-person walks; `free` uses random cameras.
//!     With `fp`, a share P (default 0.5) of views instead stand near a creature
//!     and look at it, since walks alone rarely get close to one. With `--levels`,
//!     a share Q (default 0.3) of the scenes are instead walks through the editor
//!     levels (*.ron) in DIR, each from a random place. Every view gets its own
//!     screen width, a multiple of 16 from 320 (4:3) to 576 (2.4:1) at random, or W.
//! `datagen walk --out DIR [--frames N] [--seed X]`
//!     One scene and a continuous first-person walk through it.
//! `datagen textures --out FILE.png`
//!     The fixed texture atlas, as the network would have to memorise it.
//! `datagen skins --out FILE.png`
//!     Sample creature skins: per motif, three densities, two seeds each.
//! `datagen level --file LEVEL.ron --out FILE.png [--seed X]`
//!     16 views of a hand-built level (see `level`), walking from its player start.
//! `datagen level-demo --out LEVEL.ron`
//!     Write the built-in demo level: two storeys, stairs, a bridge over a pit, a courtyard.
//! `datagen creatures --out FILE.png [--seed X] [--zoom K]`
//!     16 close views of creatures with their masks, for eyeballing the generator;
//!     `--zoom` lengthens the lens K times for a closer look.
//! `datagen check --frames N --out DIR [--model M] [--seed 7] [--effect E] [--width W]`
//!     The neural renderer on the first N frames of the walk `datagen walk --seed S`
//!     makes, as tokens and network images in DIR (with dream effect E), to compare
//!     against `train/data.py` and PyTorch. The model and textures are
//!     `neurofield.ron`'s unless given.
//!
//! With `--sculpt EYE_DIR`, `gen` instead makes the level net's sculpted targets
//! (`world::dreamed`): no creatures, every level turning to flesh and each view at
//! its own place on the dial and moment. The plain triangles stay the input;
//! `turning.npy`, `organs.npy` and `amounts.npy` say what the net can't guess. It
//! needs `--strange` to be the painted textures with height maps. A share
//! `--organ-views` (default 0.4) of the views stand in front of an eye or sphincter.
//!
//! `gen` keeps the creatures apart from the frame (see `format.json`); the rest
//! draw them over it, as the game does.
//!
//! Frames are drawn by the game's renderer (`engine::render` on psx-bevy) without a
//! window, so they need a GPU adapter: Vulkan, or Mesa's software one (lavapipe)
//! on machines without. Scenes are built on every core meanwhile.
//!
//! Every command takes `--photos DIR` to cut creature skins from the photos in
//! DIR/<motif>/ (see `photo/fetch_commons.py`) instead of drawing them. `--level DIR`
//! takes level textures 0-7 from DIR/<slot>.png (see `photo/fetch_ambientcg.py`);
//! `--strange DIR` takes them at every strangeness step from DIR/<step>/<slot>.png
//! (see `photo/bake_strange.py`); `--strangeness X` makes every level that strange
//! at its base rather than at random.
//!
//! See `format.json` in each output directory for the array layouts.

mod check;
mod npy;

use engine::neural::features::{part_features, tri_features, view_features, PART_FEATURES, TRI_FEATURES, VIEW_FEATURES};
use engine::render::headless::{Frame, Renderer};
use engine::world::level::Level;
use engine::world::creature::Creature;
use engine::world::dreamed::{self, Dreamed, ORGAN_FEATURES, TURNING_FEATURES};
use engine::world::rng::Rng;
use engine::world::scene::{Camera, Scene};
use engine::world::view::{H, W};
use engine::world::{tex, walkable_scene};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Training views are drawn this wide and cropped to their own width: the
/// widest, a multiple of 64 so GPU readback rows need no unpadding.
const DRAW_W: u32 = 576;
/// Narrowest view width (4:3).
const MIN_W: u32 = 320;


const FORMAT: &str = r#"{
  "units": "positions and distances in tiles (256 world units); y points down, the lowest floor is y = 0 and floors rise to y = -1",
  "tris.npy": "float32 [T, 34] per level triangle (creatures are not in here; see parts.npy): v0 xyz, v1 xyz, v2 xyz (0-8); v0 uv, v1 uv, v2 uv in texture tiles, 0..1 (9-14); texture id 0-11 (15); v0 rgb, v1 rgb, v2 rgb baked vertex colour / 255 (16-24); v0 rgb, v1 rgb, v2 rgb unlit surface tint / 255 (25-33); strangeness at the centre, 0 grounded to 1 deep (34): level textures 0-7 morph with it (drawn at the nearest of 8 baked steps), and the baked colours already include its lurid lights. On both colour scales 128/255 is neutral texture modulation. Triangles are wound so (v1 - v0) x (v2 - v0) is the outward normal",
  "textures": "0-7 level (brick, stone, planks, tiles, moss, grate, wallpaper, rings), 8 scales, 9 bone, 10 fur, 11 door; see `datagen textures`",
  "tri_offsets.npy": "int64 [S + 1]: scene s owns tris[tri_offsets[s] : tri_offsets[s + 1]]",
  "parts.npy": "float32 [P, 49] per creature part, world space: joint xyz (0-2); centre xyz (3-5); local axes as the columns of a row-major 3x3 (6-14; y runs from the joint end to the far end); radius x, half-length y, radius z (15-17); bounding box in the local frame, lo xyz then hi xyz (18-23); shape 0 ellipsoid, 1 capsule, 2 cone, 3 tentacle (24); taper, bend, bulge, lumps (25-28); skin motif 0-10 (29; flesh, eyes, teeth, mouths, suckers, veins, scales, chitin, hair, wrinkles, bone); skin density 0..1 (30); tiles per skin repeat (31); tint rgb, 1 = neutral (32-34); baked light at the centre as linear spherical harmonics, 1 = neutral: mean rgb (35-37), then the gradient along world x, y, z, rgb each (38-46); parent part index within the scene, -1 for a root (47); creature index within the scene (48)",
  "part_offsets.npy": "int64 [S + 1]: scene s owns parts[part_offsets[s] : part_offsets[s + 1]]",
  "views.npy": "float32 [V, 27] per view: scene index within the shard (0); camera position xyz (1-3); world-to-view rotation, row-major (4-12; row 2 is the view direction); projection distance H in pixels, screen centre (width / 2, 120) (13); fog near, fog far (14-15); yaw, pitch, roll in turns (16-18); torch rgb strength, 0 = no torch (19-21); torch reach (22); fog colour rgb 0..1, black when grounded (23-25); screen width in pixels, a multiple of 16, 240 high (26)",
  "images.npy": "uint8 [sum of 240 * width * 3]: the views' frames one after another, each [240, width, 3], without creatures (gen) or with them drawn over (walk); view v starts at image_offsets[v]",
  "image_offsets.npy": "int64 [V + 1]: byte offsets of each view's frame in images.npy",
  "creature_boxes.npy": "int32 [V, 4] per view: x0, y0, x1, y1, the pixels creatures touch (x0 <= x < x1, y0 <= y < y1); empty (x0 = x1) when none do",
  "creatures.npy": "uint8 [sum of box area * 4]: per view its box's pixels, rows top to bottom, as rgba: the colour of the creature samples in the pixel (2x2 per pixel), averaged, and alpha = samples covered / 4 (0, 64, 128, 191 or 255); rgb is 0 where alpha is. In gen every creature is whole, however much of the level stands in front of it; in walk the level hides them as on screen. The game lays them over the frame as mix(frame, rgb, alpha) in linear light (the screen is sRGB); view v starts at creature_offsets[v]",
  "creature_offsets.npy": "int64 [V + 1]: byte offsets of each view's box in creatures.npy",
  "turning.npy": "sculpted data only. float32 [T, 15] per triangle of tris.npy: how much of the dial reaches each corner (0-2); the corners on the flesh textures, in repeats (3-8); height of the folds in tiles at the corners and at the middles of edges ab, ac, bc (9-14). Zeros on what stays stone",
  "amounts.npy": "sculpted data only. float32 [V]: how far the level has turned in this view, 0 plain stone to 1 all of it",
  "organs.npy": "sculpted data only. float32 [O, 19] per eye, sphincter or length of vein in a view: kind 0 eye, 1 sphincter, 2 vein (0); where it is now, tiles (1-3); what it faces (4-6); across and up (7-12; an eye's are the image's right and up, a vein's along it and to the side); half its size along those, tiles (13-14; a vein's half length and radius now); state (15-18): eye showing, blink frame as a share (0 open), gaze across and down 0..1; sphincter showing and clench; vein beat",
  "organ_offsets.npy": "sculpted data only. int64 [V + 1]: view v owns organs[organ_offsets[v] : organ_offsets[v + 1]]",
  "projection": "view = R (p - pos); screen = (width / 2 + H * view.x / view.z, 120 + H * view.y / view.z); triangles are clipped at view.z = 0.02 tiles",
  "shading": "level, per vertex: colour = baked + tint * torch * max(0, 1 - d / reach)^2 * max(0, n . l), d and l toward the camera; per pixel: texel * colour / 128 (affine), toward the fog colour by clamp((distance - fog near) / (fog far - fog near), 0, 1), dithered to 15 bits. Creatures are drawn over that in full colour, anti-aliased (see engine/src/render/flesh.wesl)"
}
"#;

fn write_png(path: &Path, w: u32, h: u32, rgb: &[u8]) {
    let file = std::io::BufWriter::new(std::fs::File::create(path).expect("create png"));
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().and_then(|mut wr| wr.write_image_data(rgb)).expect("write png");
}

/// Up to 16 frames in a 4x4 grid.
fn contact_sheet(path: &Path, frames: &[Vec<u8>]) {
    let (w, h) = (W as usize, H as usize);
    let mut sheet = vec![0u8; 16 * w * h * 3];
    for (i, f) in frames.iter().take(16).enumerate() {
        let (ox, oy) = (i % 4 * w, i / 4 * h);
        for y in 0..h {
            let dst = ((oy + y) * 4 * w + ox) * 3;
            sheet[dst..dst + w * 3].copy_from_slice(&f[y * w * 3..(y + 1) * w * 3]);
        }
    }
    write_png(path, 4 * w as u32, 4 * h as u32, &sheet);
}

struct Batch {
    tris: Vec<f32>,
    offsets: Vec<i64>,
    parts: Vec<f32>,
    part_offsets: Vec<i64>,
    views: Vec<f32>,
    images: Vec<u8>,
    image_offsets: Vec<i64>,
    boxes: Vec<i32>,
    creatures: Vec<u8>,
    creature_offsets: Vec<i64>,
    /// The first 4:3 frames, for the preview sheet.
    frames: Vec<Vec<u8>>,
    /// Sculpted data only: what the net is told beyond the plain triangles.
    turning: Vec<f32>,
    amounts: Vec<f32>,
    organs: Vec<f32>,
    organ_offsets: Vec<i64>,
}

/// The box round the pixels of a `width`-wide RGBA layer with any alpha, as
/// x0, y0, x1, y1 (all 0 when none have).
fn alpha_box(rgba: &[u8], width: usize) -> [i32; 4] {
    let (mut lo, mut hi) = ([usize::MAX; 2], [0usize; 2]);
    for (i, p) in rgba.chunks_exact(4).enumerate() {
        if p[3] > 0 {
            let (x, y) = (i % width, i / width);
            lo = [lo[0].min(x), lo[1].min(y)];
            hi = [hi[0].max(x + 1), hi[1].max(y + 1)];
        }
    }
    if lo[0] == usize::MAX {
        return [0; 4];
    }
    [lo[0], lo[1], hi[0], hi[1]].map(|c| c as i32)
}

impl Batch {
    fn new() -> Self {
        Batch {
            tris: Vec::new(),
            offsets: vec![0],
            parts: Vec::new(),
            part_offsets: vec![0],
            views: Vec::new(),
            images: Vec::new(),
            image_offsets: vec![0],
            boxes: Vec::new(),
            creatures: Vec::new(),
            creature_offsets: vec![0],
            frames: Vec::new(),
            turning: Vec::new(),
            amounts: Vec::new(),
            organs: Vec::new(),
            organ_offsets: vec![0],
        }
    }

    fn add_scene(&mut self, scene: &Scene) -> usize {
        tri_features(scene, &mut self.tris);
        self.offsets.push((self.tris.len() / TRI_FEATURES) as i64);
        part_features(scene, &mut self.parts);
        self.part_offsets.push((self.parts.len() / PART_FEATURES) as i64);
        self.offsets.len() - 2
    }

    /// Keep `cam`'s view of `scene`, as drawn `width` pixels wide, and its creatures' box.
    fn add_view(&mut self, scene_index: usize, scene: &Scene, cam: &Camera, frame: &Frame, width: u32) {
        view_features(scene, scene_index, cam, width, &mut self.views);
        self.images.extend_from_slice(&frame.rgb);
        self.image_offsets.push(self.images.len() as i64);
        let b = alpha_box(&frame.creatures, width as usize);
        let (x0, x1) = (b[0] as usize * 4, b[2] as usize * 4);
        for y in b[1] as usize..b[3] as usize {
            let row = y * width as usize * 4;
            self.creatures.extend_from_slice(&frame.creatures[row + x0..row + x1]);
        }
        self.boxes.extend(b);
        self.creature_offsets.push(self.creatures.len() as i64);
        if self.frames.len() < 16 && width == W {
            self.frames.push(frame.rgb.clone());
        }
    }

    fn write(&self, dir: &Path) {
        std::fs::create_dir_all(dir).expect("create output dir");
        let nv = self.views.len() / VIEW_FEATURES;
        npy::write(&dir.join("tris.npy"), &[self.tris.len() / TRI_FEATURES, TRI_FEATURES], &self.tris).unwrap();
        npy::write(&dir.join("tri_offsets.npy"), &[self.offsets.len()], &self.offsets).unwrap();
        npy::write(&dir.join("parts.npy"), &[self.parts.len() / PART_FEATURES, PART_FEATURES], &self.parts).unwrap();
        npy::write(&dir.join("part_offsets.npy"), &[self.part_offsets.len()], &self.part_offsets).unwrap();
        npy::write(&dir.join("views.npy"), &[nv, VIEW_FEATURES], &self.views).unwrap();
        npy::write(&dir.join("images.npy"), &[self.images.len()], &self.images).unwrap();
        npy::write(&dir.join("image_offsets.npy"), &[nv + 1], &self.image_offsets).unwrap();
        npy::write(&dir.join("creature_boxes.npy"), &[nv, 4], &self.boxes).unwrap();
        npy::write(&dir.join("creatures.npy"), &[self.creatures.len()], &self.creatures).unwrap();
        npy::write(&dir.join("creature_offsets.npy"), &[nv + 1], &self.creature_offsets).unwrap();
        if !self.amounts.is_empty() {
            assert_eq!(self.turning.len() / TURNING_FEATURES, self.tris.len() / TRI_FEATURES);
            npy::write(&dir.join("turning.npy"), &[self.turning.len() / TURNING_FEATURES, TURNING_FEATURES], &self.turning).unwrap();
            npy::write(&dir.join("amounts.npy"), &[nv], &self.amounts).unwrap();
            npy::write(&dir.join("organs.npy"), &[self.organs.len() / ORGAN_FEATURES, ORGAN_FEATURES], &self.organs).unwrap();
            npy::write(&dir.join("organ_offsets.npy"), &[nv + 1], &self.organ_offsets).unwrap();
        }
        contact_sheet(&dir.join("preview.png"), &self.frames);
    }
}

/// How training views are chosen.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    /// Frames sampled from first-person walks (eye height, lens and level gaze fixed).
    FirstPerson,
    /// Independent random cameras with varied height, tilt, roll and lens.
    Free,
}

/// One of the scene's creatures, at random.
fn pick_creature<'a>(scene: &'a Scene, rng: &mut Rng) -> Option<&'a Creature> {
    scene.creatures.get(rng.next() as usize % scene.creatures.len().max(1))
}

/// Where a run's scenes come from and how its views are framed.
struct Sources {
    mode: CameraMode,
    /// Share of first-person views that look at a creature.
    creature_views: f32,
    /// Editor levels, and the share of scenes walked through them.
    levels: Arc<Vec<Level>>,
    level_share: f32,
    /// Every view this wide, or (None) each its own width at random.
    width: Option<u32>,
    /// Where the filmed eye's frames are, to draw sculpted targets, and the share of views
    /// to stand in front of its eyes and sphincters.
    sculpt: Option<PathBuf>,
    organ_views: f32,
}

/// One view to draw: where from, how wide, and for sculpted data the moment and the dial.
struct Shot {
    cam: Camera,
    width: u32,
    time: f32,
    amount: f32,
}

/// Where on the dial a view is: mostly anywhere, with both ends common, since the game
/// spends a lot of its time at them.
fn pick_amount(rng: &mut Rng) -> f32 {
    if rng.chance(0.1) {
        0.0
    } else if rng.chance(0.35) {
        1.0
    } else {
        rng.f()
    }
}

/// A view's screen width: a multiple of 16 from 320 to `DRAW_W`.
fn pick_width(src: &Sources, rng: &mut Rng) -> u32 {
    src.width.unwrap_or_else(|| MIN_W + 16 * rng.range(0, ((DRAW_W - MIN_W) / 16 + 1) as i32) as u32)
}

/// A walker somewhere in editor level `level`'s scene with room to walk.
fn level_scene(level: &Level, rng: &mut Rng) -> Option<(Scene, engine::world::walker::Walker)> {
    let scene = level.to_scene();
    let walker = (0..50).find_map(|_| engine::world::walker::Walker::new(&scene, rng))?;
    Some((scene, walker))
}

/// Scene `i` of a run and the cameras (with their widths) to draw it from.
fn scene_and_views(i: usize, seed: u64, views: usize, src: &Sources) -> (Scene, Option<Dreamed>, Vec<Shot>) {
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003) ^ i as u64);
    let mut dreamed = None;
    let (scene, cams) = match src.mode {
        CameraMode::Free => {
            let scene = Scene::generate(&mut rng);
            let cams = (0..views).map(|_| scene.random_camera(&mut rng)).collect();
            (scene, cams)
        }
        CameraMode::FirstPerson => {
            let from_level = (!src.levels.is_empty() && rng.chance(src.level_share))
                .then(|| {
                    let level = &src.levels[rng.next() as usize % src.levels.len()];
                    level_scene(level, &mut rng)
                })
                .flatten();
            let (mut scene, mut walker) = from_level.unwrap_or_else(|| walkable_scene(&mut rng, 6));
            if let Some(eyes) = &src.sculpt {
                // The level net draws the level alone.
                scene.creatures.clear();
                scene.flesh.tris.clear();
                scene.flesh.skins.clear();
                scene.flesh.lights.clear();
                scene.flesh.light_at.clear();
                dreamed = Some(dreamed::sculpt(&mut scene, rng.next() as u64, Some(eyes)));
            }
            let sights = dreamed.as_ref().map(dreamed::sights).unwrap_or_default();
            for _ in 0..rng.range(0, 60) {
                walker.step(&scene, &mut rng);
            }
            let mut out = Vec::with_capacity(views);
            while out.len() < views {
                for _ in 0..rng.range(6, 24) {
                    walker.step(&scene, &mut rng);
                }
                // Creature views come in orbits of 1-3 round the same one, so
                // the network sees it from several sides.
                let orbit: Vec<Camera> = if !sights.is_empty() {
                    if rng.chance(src.organ_views) {
                        let (at, n) = sights[rng.next() as usize % sights.len()];
                        walker.view_of_wall(&scene, at, n, &mut rng).into_iter().collect()
                    } else {
                        Vec::new()
                    }
                } else if rng.chance(src.creature_views) {
                    let n = rng.range(1, 4).min((views - out.len()) as i32) as usize;
                    pick_creature(&scene, &mut rng)
                        .map(|c| (0..n).filter_map(|_| walker.view_of(&scene, c, &mut rng)).collect())
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                if orbit.is_empty() {
                    out.push(walker.camera());
                } else {
                    out.extend(orbit);
                }
            }
            (scene, out)
        }
    };
    let shots = cams
        .into_iter()
        .map(|cam| Shot { cam, width: pick_width(src, &mut rng), time: rng.fr(0.0, 60.0), amount: pick_amount(&mut rng) })
        .filter(|s| !dreamed.as_ref().is_some_and(|d| dreamed::crowds(d, &s.cam, s.amount)))
        .collect();
    (scene, dreamed, shots)
}

/// Scenes are built on every core, a batch at a time in order, while this
/// thread draws the last batch's views on the GPU; shards are written behind.
fn gen(out: &Path, scenes: usize, views: usize, shard_scenes: usize, seed: u64, src: Sources) {
    let batch_len = rayon::current_num_threads() * 2;
    let sculpted = src.sculpt.is_some();
    let (send, recv) = std::sync::mpsc::sync_channel::<Vec<(usize, Scene, Option<Dreamed>, Vec<Shot>)>>(1);
    let builder = std::thread::spawn(move || {
        for start in (0..scenes).step_by(batch_len) {
            let built = (start..(start + batch_len).min(scenes))
                .into_par_iter()
                .map(|i| {
                    let (scene, dreamed, shots) = scene_and_views(i, seed, views, &src);
                    (i, scene, dreamed, shots)
                })
                .collect();
            if send.send(built).is_err() {
                return;
            }
        }
    });
    // The flesh of a sculpted level is part of its frame.
    let mut renderer = if sculpted { Renderer::with_width(DRAW_W) } else { Renderer::layers(DRAW_W) };
    let mut writers = Vec::new();
    let mut batch = Batch::new();
    let finish = |batch: Batch, s: usize, writers: &mut Vec<std::thread::JoinHandle<()>>| {
        let dir = out.join(format!("shard_{s:04}"));
        writers.push(std::thread::spawn(move || {
            batch.write(&dir);
            eprintln!("shard {s:04} done");
        }));
    };
    let (t0, mut waiting, mut drawing, mut drawn) = (std::time::Instant::now(), 0.0, 0.0, 0);
    loop {
        let t = std::time::Instant::now();
        let Ok(built) = recv.recv() else { break };
        waiting += t.elapsed().as_secs_f64();
        for (i, mut scene, dreamed, shots) in built {
            let t = std::time::Instant::now();
            let si = batch.add_scene(&scene);
            if let Some(mut d) = dreamed {
                dreamed::turning_features(&scene, &d, &mut batch.turning);
                for s in &shots {
                    d.amount = s.amount;
                    dreamed::pose(&mut scene, &d, &s.cam, s.time);
                    scene.flesh.tris.extend(dreamed::walls(&d));
                    renderer.set_scene(&scene);
                    let mut frame = renderer.render(&scene, &s.cam).crop(s.width);
                    // All of it is the target: there is no layer to keep apart.
                    frame.creatures.fill(0);
                    batch.add_view(si, &scene, &s.cam, &frame, s.width);
                    batch.amounts.push(s.amount);
                    dreamed::organs(&d, &s.cam, s.width, s.time, &mut batch.organs);
                    batch.organ_offsets.push((batch.organs.len() / ORGAN_FEATURES) as i64);
                }
            } else {
                renderer.set_scene(&scene);
                let only: Vec<Camera> = shots.iter().map(|s| s.cam).collect();
                for (s, frame) in shots.iter().zip(renderer.render_all(&scene, &only)) {
                    batch.add_view(si, &scene, &s.cam, &frame.crop(s.width), s.width);
                }
            }
            drawing += t.elapsed().as_secs_f64();
            drawn += shots.len();
            if (i + 1) % shard_scenes == 0 || i + 1 == scenes {
                finish(std::mem::replace(&mut batch, Batch::new()), i / shard_scenes, &mut writers);
            }
        }
    }
    builder.join().unwrap();
    for w in writers {
        w.join().unwrap();
    }
    eprintln!(
        "{drawn} views in {:.0} s: drawing {:.1} ms a view, {:.0} s waiting for scenes",
        t0.elapsed().as_secs_f64(),
        drawing * 1000.0 / drawn.max(1) as f64,
        waiting
    );
    std::fs::write(out.join("format.json"), FORMAT).unwrap();
}

/// Every editor level (*.ron) in `dir`, sorted by name.
fn load_levels(dir: &Path) -> Vec<Level> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "ron"))
        .collect();
    files.sort();
    let levels: Vec<Level> = files.iter().map(|f| Level::load(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()))).collect();
    eprintln!("{} levels from {}", levels.len(), dir.display());
    levels
}

/// One level and a continuous first-person walk through it.
fn walk(out: &Path, frames: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    // Walks want room to move.
    let (scene, mut walker) = walkable_scene(&mut rng, 30);
    let mut renderer = Renderer::new();
    renderer.set_scene(&scene);
    let mut batch = Batch::new();
    let si = batch.add_scene(&scene);
    std::fs::create_dir_all(out.join("frames")).unwrap();
    for f in 0..frames {
        let cam = walker.camera();
        let frame = renderer.render(&scene, &cam);
        batch.add_view(si, &scene, &cam, &frame, W);
        write_png(&out.join(format!("frames/{f:05}.png")), W, H, &frame.rgb);
        walker.step(&scene, &mut rng);
    }
    batch.frames.clear();
    batch.write(out);
    std::fs::write(out.join("format.json"), FORMAT).unwrap();
}

/// A first-person walk through a level as the level net could be taught to dream it
/// (`world::dreamed`), as `out/frame_NNNNN.png`.
fn sculpt_walk(out: &Path, frames: usize, seed: u64, eyes: Option<&Path>, stop_at: Option<usize>, sweep: bool) {
    let mut rng = Rng::new(seed);
    let (mut scene, mut walker) = walkable_scene(&mut rng, 30);
    let t0 = std::time::Instant::now();
    let mut dreamed = engine::world::dreamed::sculpt(&mut scene, seed, eyes);
    eprintln!("sculpted to {} triangles in {:.1} s", scene.tris.len(), t0.elapsed().as_secs_f64());
    let mut renderer = Renderer::new();
    std::fs::create_dir_all(out).unwrap();
    let t0 = std::time::Instant::now();
    for f in 0..frames {
        // With a stopping frame, nothing before it is drawn and the camera rests from there.
        if stop_at.is_some_and(|s| f < s) {
            walker.step(&scene, &mut rng);
            continue;
        }
        let cam = walker.camera();
        // A sweep goes from plain stone to fully turned and back while the frames last.
        if sweep {
            let start = stop_at.unwrap_or(0);
            let k = (f - start) as f32 / (frames - start).max(1) as f32;
            dreamed.amount = 1.0 - (2.0 * k - 1.0).abs();
        }
        engine::world::dreamed::pose(&mut scene, &dreamed, &cam, f as f32 / 30.0);
        scene.flesh.tris.extend(engine::world::dreamed::walls(&dreamed));
        renderer.set_scene(&scene);
        let frame = renderer.render(&scene, &cam);
        write_png(&out.join(format!("frame_{f:05}.png")), W, H, &frame.rgb);
        if stop_at.is_none() {
            walker.step(&scene, &mut rng);
        }
    }
    eprintln!("{:.0} ms a frame", t0.elapsed().as_secs_f64() * 1000.0 / frames.max(1) as f64);
}

/// 16 close views of creatures from random scenes, each with its mask tinted red
/// beside it, 8 rows of two pairs.
fn creatures(out: &Path, seed: u64, zoom: f32) {
    let mut renderer = Renderer::new();
    let mut rng = Rng::new(seed);
    let mut frames = Vec::new();
    while frames.len() < 32 {
        let (scene, walker) = walkable_scene(&mut rng, 6);
        let Some(mut cam) = pick_creature(&scene, &mut rng).and_then(|c| walker.view_of(&scene, c, &mut rng)) else {
            continue;
        };
        cam.h = (cam.h as f32 * zoom) as u16;
        renderer.set_scene(&scene);
        let drawn = renderer.render(&scene, &cam);
        let (frame, mask) = (drawn.rgb.clone(), drawn.mask());
        let tinted = frame.chunks(3).zip(&mask).flat_map(|(p, &m)| if m == 1 { [p[0] / 2 + 128, p[1] / 2, p[2] / 2] } else { [p[0] / 3, p[1] / 3, p[2] / 3] }).collect();
        eprintln!("{} creature tris, {} skins", scene.flesh.tris.len(), scene.flesh.skins.len());
        frames.push(frame);
        frames.push(tinted);
    }
    let (w, h) = (W as usize, H as usize);
    let mut sheet = vec![0u8; 32 * w * h * 3];
    for (i, f) in frames.iter().enumerate() {
        let (ox, oy) = (i % 4 * w, i / 4 * h);
        for y in 0..h {
            let dst = ((oy + y) * 4 * w + ox) * 3;
            sheet[dst..dst + w * 3].copy_from_slice(&f[y * w * 3..(y + 1) * w * 3]);
        }
    }
    write_png(out, 4 * w as u32, 8 * h as u32, &sheet);
}
/// A first-person walk through a hand-built level from its player start, 16 frames apart.
fn level_views(out: &Path, level: &engine::world::level::Level, seed: u64) {
    let mut rng = Rng::new(seed);
    let t = std::time::Instant::now();
    let scene = level.to_scene();
    eprintln!("{} tris, {} creatures, built in {:?}", scene.tris.len(), scene.creatures.len(), t.elapsed());
    let (pos, yaw) = level.player_start().expect("the level needs a PlayerStart");
    let mut walker = engine::world::walker::Walker::at(&scene, pos, yaw, &mut rng).expect("player start is in rock");
    let mut renderer = Renderer::new();
    renderer.set_scene(&scene);
    let mut frames = Vec::new();
    for i in 0..16 {
        let cam = walker.camera();
        if std::env::var("DEBUG_WALK").is_ok() {
            let n = scene.node_at(cam.pos[0] as f32, cam.pos[1] as f32 + engine::world::walker::EYE, cam.pos[2] as f32);
            eprintln!("{i:2} pos {:?} yaw {} pitch {} node {n:?}", cam.pos, cam.yaw, cam.pitch);
        }
        frames.push(renderer.render(&scene, &cam).rgb);
        for _ in 0..40 {
            walker.step(&scene, &mut rng);
        }
    }
    contact_sheet(out, &frames);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let mut opts = HashMap::new();
    for pair in args[2.min(args.len())..].chunks(2) {
        match pair {
            [k, v] if k.starts_with("--") => {
                opts.insert(k[2..].to_string(), v.clone());
            }
            _ => panic!("expected --key value pairs, got {pair:?}"),
        }
    }
    let num = |k: &str, d: u64| opts.get(k).map_or(d, |v| v.parse().expect("numeric option"));
    let out = PathBuf::from(opts.get("out").expect("--out is required"));
    if cmd == "check" {
        // The model's own textures, as the game has them.
        engine::world::config::Config::load().load_textures(&opts);
        let frames = opts.get("frames").expect("--frames N is required").parse().expect("--frames is a count");
        check::run(&opts, frames, &out);
        return;
    }
    if cmd == "sculpt-walk" {
        engine::world::config::Config::load().load_textures(&opts);
        sculpt_walk(&out, num("frames", 360) as usize, opts.get("seed").map_or(1, |s| s.parse().expect("--seed is a number")), opts.get("eye").map(Path::new), opts.get("stop-at").map(|s| s.parse().expect("--stop-at is a frame")), opts.contains_key("sweep"));
        return;
    }
    if cmd == "dream-check" {
        check::dream_check(&opts);
        return;
    }
    if cmd == "dream-walk" {
        engine::world::config::Config::load().load_textures(&opts);
        check::dream_walk(&opts, num("frames", 180) as usize, &out);
        return;
    }
    if let Some(dir) = opts.get("photos") {
        engine::world::photo::load(Path::new(dir));
    }
    if let Some(dir) = opts.get("strange") {
        tex::load_strange(Path::new(dir));
    } else if let Some(dir) = opts.get("level") {
        tex::load_level(Path::new(dir));
    }
    if let Some(s) = opts.get("strangeness") {
        engine::world::scene::force_strangeness(Some(s.parse().expect("--strangeness 0..1")));
    }
    match cmd {
        "gen" => {
            let mode = match opts.get("camera").map(String::as_str) {
                None | Some("fp") => CameraMode::FirstPerson,
                Some("free") => CameraMode::Free,
                Some(m) => panic!("--camera must be fp or free, got {m}"),
            };
            let frac = |k: &str, d: f32| opts.get(k).map_or(d, |v| v.parse().expect("numeric option"));
            let width = opts.get("width").map(|v| v.parse::<u32>().expect("--width is a number of pixels"));
            if let Some(w) = width {
                assert!(w % 16 == 0 && (MIN_W..=DRAW_W).contains(&w), "--width must be a multiple of 16 from {MIN_W} to {DRAW_W}");
            }
            let src = Sources {
                mode,
                creature_views: frac("creature-views", 0.5),
                levels: Arc::new(opts.get("levels").map(|d| load_levels(Path::new(d))).unwrap_or_default()),
                level_share: frac("level-share", 0.3),
                width,
                sculpt: opts.get("sculpt").map(PathBuf::from),
                organ_views: frac("organ-views", 0.4),
            };
            std::fs::create_dir_all(&out).unwrap();
            gen(&out, num("scenes", 1000) as usize, num("views", 8) as usize, num("shard-scenes", 100) as usize, num("seed", 1), src);
        }
        "walk" => walk(&out, num("frames", 300) as usize, num("seed", 7)),
        "textures" => {
            let (w, h, rgb) = tex::atlas_rgb(&tex::make_all());
            write_png(&out, w, h, &rgb);
        }
        "creatures" => creatures(&out, num("seed", 3), opts.get("zoom").map_or(1.0, |v| v.parse().expect("numeric option"))),
        "level" => {
            let file = opts.get("file").expect("--file LEVEL.ron is required");
            let level = engine::world::level::Level::load(Path::new(file)).unwrap_or_else(|e| panic!("{e}"));
            level_views(&out, &level, num("seed", 1));
        }
        "level-demo" => engine::world::level::Level::demo().save(&out).expect("write level"),
        "skins" => {
            let (w, h, rgb) = engine::world::skin::sheet();
            write_png(&out, w, h, &rgb);
        }
        _ => {
            eprintln!("usage: datagen gen|walk|textures|skins|creatures|level|level-demo|check --out PATH [...]");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod debug {
    use super::*;
    use engine::world::scene;

    /// Print one generated level: per cell, floor level (in steps above 0) / clearance, or #.
    #[test]
    #[ignore]
    fn dump_level() {
        let i: u64 = std::env::var("SCENE").unwrap().parse().unwrap();
        let mut rng = Rng::new(1u64.wrapping_mul(1_000_003) ^ i);
        let (scene, _) = walkable_scene(&mut rng, 6);
        for z in 0..scene.d {
            let row: Vec<String> = (0..scene.w)
                .map(|x| match scene.spans(x, z).first() {
                    None => "  ## ".into(),
                    Some(s) => {
                        let height = (s.y0 - s.c0) / 256;
                        format!("{}{}/{}{} ", -s.y0 / 64, -s.y1 / 64, height, if s.slope == scene::Slope::Flat { ' ' } else { '~' })
                    }
                })
                .collect();
            println!("z{z:2} {}", row.join(""));
        }
        for r in &scene.rooms {
            println!("room x{} z{} w{} d{} floor {} h {}", r.x, r.z, r.w, r.d, r.floor / 64, r.height);
        }
    }
}
