//! The dream atlas (docs/dream_atlas.md, the `dream-renderer` crate): DeepDream growing on the
//! dungeon's walls over the PS1 picture, with `--dream-atlas runs/dream/googlenet.safetensors`.
//!
//! The level's triangles are given to the dream renderer (again whenever they change), and each
//! frame the camera the PS1 screen is drawn from goes to the render world with the frame's
//! settings (`Job`), where the dream renderer runs on Bevy's own GPU device as a camera of its
//! own (`DreamGraph`, after the PS1 screen is drawn and before the window shows it): it draws
//! the level itself to say which face each pixel shows and where on it (a face is a tile of one
//! of the world's planes, by the surface's main axis: one page of the atlas), reads the PS1
//! screen's texture, dreams on it, and writes the frame to show, three times the size, the dream
//! sharp over the PS1's pixels, into the image that stands in for the network's on its sprite.
//! Nothing is read back but which faces were seen.
//!
//! GAME_DREAM_CHECK=1 also makes each frame's G-buffer on the CPU (`DreamAtlas::surfaces`, as it
//! was made before the renderer drew the level) and, every 60th frame, compares the renderer's
//! with it.
//!
//! Known gaps: what the level's triangles don't know (people, the sword) is dreamed over as if
//! it were the wall behind it; creatures are left out of the dream by their depth.

use super::{Mode, NetworkSprite, Sim};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::camera::CameraRenderGraph;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::renderer::{PendingCommandBuffers, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::RenderApp;
use bevy_egui::{egui, EguiContexts};
use dream_renderer::{Config, DreamRenderer, Light, LevelTri, NetConfig, Motion, Step, Surface, View};
use engine::neural::features::TRI_FEATURES;
use engine::neural::gpu_net::MAX_W;
use engine::world::scene::Camera;
use engine::world::view::H;
use psx_bevy::psx::{PsxScreen, DISPLAY_LAYER};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const UP: u32 = 3;
/// dream_renderer::Config's default `max_touched`.
const MAX_TOUCHED: u32 = 128;
/// A face: main axis of its normal, which way it faces, and the tile its points lie in.
type Face = [i32; 5];
/// Tiles: the level camera's near plane.
const NEAR: f32 = 0.02;
/// Seconds the "settle" display takes to sharpen once the camera stops.
const SETTLE_SECS: f32 = 1.0;

/// The PS1 pictures read back: (frame number, RGBA8 rows padded to 256 bytes, width).
/// The dream camera's order: after the PS1 screen pass (999) and the flesh overlay (1000), before
/// the window's display (moved after it).
const DREAM_ORDER: isize = 1001;

/// A frame's dreaming, from the main world to the render world.
struct Job {
    width: u32,
    screen: AssetId<Image>,
    out: AssetId<Image>,
    /// The level's triangles and their textures (64 a side), when they're new.
    level: Option<(Vec<LevelTri>, Vec<Vec<u8>>)>,
    /// None: the view hasn't moved.
    view: Option<View>,
    /// The creatures' depth a pixel (f32::MAX where there are none), if any are in sight.
    creatures: Option<Vec<f32>>,
    /// Carried along on the screen.
    carry: bool,
    step: Step,
    /// Wake from the dream.
    wake: bool,
    /// The level's light from everywhere and its lights, as they are now.
    lights: ([f32; 3], Vec<Light>),
    /// GAME_DREAM_CHECK: the CPU's G-buffer of this view.
    check: Option<Check>,
}

/// The CPU's G-buffer of a view: each pixel's face (as the renderer's `Face`), surface and motion.
struct Check {
    faces: Vec<Option<dream_renderer::Face>>,
    surfaces: Vec<Surface>,
    motion: Vec<Motion>,
}

/// What the main and render worlds share: the next job, and how the last went.
#[derive(Default)]
struct Shared {
    job: Option<Job>,
    /// The renderer's setup, taken when the render world makes it.
    config: Option<Config>,
    ms: f32,
    pages_seen: usize,
    pages_in_use: usize,
    error: Option<String>,
}

/// The dream renderer's own camera schedule in the render world.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct DreamGraph;

/// The render world's side.
#[derive(Resource)]
struct DreamGpu {
    shared: Arc<Mutex<Shared>>,
    renderer: Option<DreamRenderer>,
    /// The CPU's G-buffer of the last frame drawn, to compare once its work is done.
    check: Option<Check>,
}

/// Where the dream grows, and how it stays put as the view moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DreamMode {
    /// Always on the walls' pages.
    Walls,
    /// On the screen while the view stays still (sharp, no G-buffer), laid on the walls as it
    /// moves.
    Still,
    /// Always on the screen, carried along as the view moves to where each pixel's point was:
    /// sharp, but forgetting what goes out of sight.
    Reproject,
}

impl DreamMode {
    pub fn parse(s: &str) -> Option<DreamMode> {
        match s {
            "walls" => Some(DreamMode::Walls),
            "still" => Some(DreamMode::Still),
            "reproject" => Some(DreamMode::Reproject),
            _ => None,
        }
    }
}

/// A view's camera as the G-buffer uses it, and the depth it saw.
struct CpuView {
    pos: [f32; 3],
    r: [[f32; 3]; 3],
    hp: f32,
    width: u32,
    nearest: Vec<f32>,
}

pub struct DreamAtlas {
    shared: Arc<Mutex<Shared>>,
    pub step: Step,
    /// Where the torch is held, along the view's axes (right, down, forward) from the eye, in
    /// tiles: by the sword.
    pub torch_at: [f32; 3],
    /// The networks' names (their files'), for `step.net`, and each one's steps (coarse, fine)
    /// and gradient blur, kept as it was left when switching away.
    net_names: Vec<String>,
    net_steps: Vec<[f32; 3]>,
    /// The dream sharp only when the camera rests (else always).
    pub settle: bool,
    /// Off: the PS1 picture through the renderer with no dreaming.
    pub dreaming: bool,
    pub mode: DreamMode,
    /// The camera and width of the last picture dreamed on.
    drawn: Option<(Camera, u32)>,
    pub stills: u64,
    image: Handle<Image>,
    pages: HashMap<Face, u32>,
    owners: Vec<Option<Face>>,
    seen: Vec<u64>,
    frame: u64,
    /// Each picture asked for: its camera, width, and the creatures' depth in it (tiles a
    /// pixel, f32::MAX where there are none; empty for none at all).
    sharp: f32,
    last_cam: Option<Camera>,
    surfaces: Vec<Surface>,
    motion: Vec<Motion>,
    /// The last view the surfaces were made for, and its depth (1/z a pixel), for the motion.
    prev: Option<CpuView>,
    nearest: Vec<f32>,
    row_of: Vec<i32>,
    pub ms: f32,
    pub gbuffer_ms: f32,
    pages_seen: usize,
    pages_in_use: usize,
    pub clear: bool,
    /// The level's triangle features as last given to the renderer.
    level: Vec<f32>,
    check: bool,
}

impl DreamAtlas {
    /// Dream with network `i` of those given, with its own steps.
    pub fn switch_net(&mut self, i: usize) {
        let i = i.min(self.net_names.len() - 1);
        let old = self.step.net.min(self.net_steps.len() - 1);
        self.net_steps[old] = [self.step.lr, self.step.lr_fine, self.step.grad_blur];
        [self.step.lr, self.step.lr_fine, self.step.grad_blur] = self.net_steps[i];
        self.step.net = i;
    }

    /// Each network's weights, and the layers each octave dreams with (none: the network's own;
    /// the fine octave's none: as the coarse).
    pub fn new(weights: &[PathBuf], layers: &[Vec<String>], fine_layers: &[Vec<String>], images: &mut Assets<Image>) -> Result<DreamAtlas, String> {
        let mut cfg = Config::new("");
        cfg.nets = weights
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let layers = layers.get(i).cloned().unwrap_or_default();
                let fine = fine_layers.get(i).filter(|f| !f.is_empty()).cloned().unwrap_or(layers.clone());
                eprintln!("dream atlas: {} dreaming with {layers:?} and {fine:?} (none: its own)", w.display());
                NetConfig { weights: w.clone(), layers, fine_layers: fine }
            })
            .collect();
        let net_names: Vec<String> = weights.iter().map(|w| w.file_stem().map_or("?".into(), |s| s.to_string_lossy().into_owned())).collect();
        cfg.max_width = MAX_W;
        cfg.height = H;
        cfg.up = UP;
        cfg.max_touched = MAX_TOUCHED;
        let max_pages = cfg.max_pages as usize;
        let size = Extent3d { width: MAX_W * UP, height: H * UP, depth_or_array_layers: 1 };
        let mut image = Image::new_fill(size, TextureDimension::D2, &[0, 0, 0, 255], TextureFormat::Rgba8UnormSrgb, RenderAssetUsages::default());
        image.sampler = ImageSampler::nearest();
        Ok(DreamAtlas {
            shared: Arc::new(Mutex::new(Shared { config: Some(cfg), ..default() })),
            net_steps: net_names
                .iter()
                .map(|n| {
                    let d = Step::default();
                    // The robust ResNet's gradients agree from step to step, so it wants smaller
                    // ones, blurred against its grating.
                    if n.contains("resnet") { [0.004, 0.003, 2.0] } else { [d.lr, d.lr_fine, d.grad_blur] }
                })
                .collect(),
            net_names,
            torch_at: [0.22, 0.2, 0.3],
            // The relief shell-mapped (docs/shell_relief.md), deep and in as far as out.
            step: Step { fine_split: 4, fine_every: 2, relief: 0.0, height: 0.001, height_max: 0.5, shell: 1.0, shell_smooth: 3.0, shell_far: 0.5, shell_pixel: 0.25, glint: 0.2, rough_dry: 0.8, rough_wet: 0.4, ..Step::default() },
            settle: false,
            mode: DreamMode::Still,
            drawn: None,
            stills: 0,
            dreaming: true,
            image: images.add(image),
            pages: HashMap::new(),
            owners: vec![None; max_pages],
            seen: vec![0; max_pages],
            frame: 0,
            sharp: 1.0,
            last_cam: None,
            surfaces: Vec::new(),
            motion: Vec::new(),
            prev: None,
            nearest: Vec::new(),
            row_of: Vec::new(),
            ms: 0.0,
            gbuffer_ms: 0.0,
            pages_seen: 0,
            pages_in_use: 0,
            clear: false,
            level: Vec::new(),
            check: std::env::var_os("GAME_DREAM_CHECK").is_some(),
        })
    }

    /// The page for a face, taking the longest unseen one when all are in use.
    fn page(&mut self, face: Face, evicted: &mut Vec<u32>) -> u32 {
        if let Some(&p) = self.pages.get(&face) {
            self.seen[p as usize] = self.frame;
            return p;
        }
        let p = match self.owners.iter().position(Option::is_none) {
            Some(p) => p,
            None => {
                let p = (0..self.seen.len()).filter(|&p| self.seen[p] < self.frame).min_by_key(|&p| self.seen[p]).unwrap_or(0);
                if let Some(old) = self.owners[p].take() {
                    self.pages.remove(&old);
                }
                evicted.push(p as u32);
                p
            }
        };
        self.owners[p] = Some(face);
        self.pages.insert(face, p as u32);
        self.seen[p] = self.frame;
        p as u32
    }

    /// (GAME_DREAM_CHECK: the renderer makes its own on the GPU.) Each pixel's face and where on it, from the level's triangles as `cam` sees them on a
    /// screen `width` wide: the nearest triangle at each pixel's middle (cut at the near plane,
    /// not split as the renderer splits them: the dream doesn't need the PS1's affine warp),
    /// then where the pixel's ray meets its plane, and a pixel right and down.
    fn surfaces(&mut self, sim: &Sim, cam: &Camera, width: u32, creatures: &[f32], evicted: &mut Vec<u32>) {
        let v = sim.level.view(cam, width);
        let tris = &sim.level.tris;
        let (w, h) = (width as usize, H as usize);
        let pos = [v[1], v[2], v[3]];
        let r = [[v[4], v[5], v[6]], [v[7], v[8], v[9]], [v[10], v[11], v[12]]];
        let (hp, fog_near, fog_far) = (v[13], v[14], v[15]);
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        self.nearest.clear();
        self.nearest.resize(w * h, 0.0);
        self.row_of.clear();
        self.row_of.resize(w * h, -1);
        for (ti, t) in tris.chunks_exact(TRI_FEATURES).enumerate() {
            let tv = [0, 1, 2].map(|j| {
                let d = sub([t[3 * j], t[3 * j + 1], t[3 * j + 2]], pos);
                [dot(r[0], d), dot(r[1], d), dot(r[2], d)]
            });
            let n = cross(sub(tv[1], tv[0]), sub(tv[2], tv[0]));
            if dot(n, tv[0]) >= 0.0 || tv.iter().all(|q| q[2] < NEAR) || tv.iter().all(|q| q[2] > fog_far) {
                continue;
            }
            // Cut at the near plane: up to four corners.
            let mut poly: Vec<[f32; 3]> = Vec::with_capacity(4);
            for j in 0..3 {
                let (a, b) = (tv[j], tv[(j + 1) % 3]);
                if a[2] >= NEAR {
                    poly.push(a);
                }
                if (a[2] >= NEAR) != (b[2] >= NEAR) {
                    let k = (NEAR - a[2]) / (b[2] - a[2]);
                    poly.push([a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, NEAR]);
                }
            }
            let sc: Vec<[f32; 3]> = poly.iter().map(|q| [w as f32 / 2.0 + hp * q[0] / q[2], 120.0 + hp * q[1] / q[2], 1.0 / q[2]]).collect();
            for k in 1..sc.len().saturating_sub(1) {
                let (a, b, c) = (sc[0], sc[k], sc[k + 1]);
                let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                if area.abs() <= 1e-6 {
                    continue;
                }
                let x0 = (a[0].min(b[0]).min(c[0]) - 0.5).ceil().max(0.0);
                let x1 = (a[0].max(b[0]).max(c[0]) - 0.5).floor().min(w as f32 - 1.0);
                let y0 = (a[1].min(b[1]).min(c[1]) - 0.5).ceil().max(0.0);
                let y1 = (a[1].max(b[1]).max(c[1]) - 0.5).floor().min(h as f32 - 1.0);
                if x1 < x0 || y1 < y0 {
                    continue;
                }
                for y in y0 as usize..=y1 as usize {
                    let cy = y as f32 + 0.5;
                    for x in x0 as usize..=x1 as usize {
                        let cx = x as f32 + 0.5;
                        let l0 = ((b[0] - cx) * (c[1] - cy) - (b[1] - cy) * (c[0] - cx)) / area;
                        let l1 = ((c[0] - cx) * (a[1] - cy) - (c[1] - cy) * (a[0] - cx)) / area;
                        let l2 = 1.0 - l0 - l1;
                        if l0 < 0.0 || l1 < 0.0 || l2 < 0.0 {
                            continue;
                        }
                        let iz = l0 * a[2] + l1 * b[2] + l2 * c[2];
                        let i = y * w + x;
                        if iz > self.nearest[i] {
                            self.nearest[i] = iz;
                            self.row_of[i] = ti as i32;
                        }
                    }
                }
            }
        }
        // The creatures drawn in front of the level: no surface there (the dream leaves them be),
        // and their depth for the next frame's motion, so the wall behind where one was is new.
        if creatures.len() == w * h {
            for (i, &z) in creatures.iter().enumerate() {
                if z < f32::MAX && 1.0 / z.max(1e-6) > self.nearest[i] {
                    self.nearest[i] = 1.0 / z.max(1e-6);
                    self.row_of[i] = -1;
                }
            }
        }
        let ray = |x: f32, y: f32| {
            let d = [(x - width as f32 / 2.0) / hp, (y - 120.0) / hp, 1.0];
            [0, 1, 2].map(|j| r[0][j] * d[0] + r[1][j] * d[1] + r[2][j] * d[2])
        };
        self.surfaces.clear();
        self.surfaces.resize(w * h, Surface::NONE);
        // Where each pixel's point was in the last view, if it was in sight there: in front of
        // it, on its screen, and as near as the nearest thing it saw there.
        let prev = self.prev.take().filter(|p| p.width == width && self.mode == DreamMode::Reproject);
        self.motion.clear();
        self.motion.resize(w * h, Motion::NONE);
        let was = |p: &CpuView, q: [f32; 3]| {
            let d = sub(q, p.pos);
            let z = dot(p.r[2], d).max(1e-6);
            [w as f32 / 2.0 + p.hp * dot(p.r[0], d) / z, 120.0 + p.hp * dot(p.r[1], d) / z, z]
        };
        // Neighbouring pixels nearly always share a triangle and a face.
        let mut plane: (i32, [f32; 3], f32, usize) = (-1, [0.0; 3], 0.0, 0);
        let mut last: Option<(Face, u32)> = None;
        for i in 0..w * h {
            let row = self.row_of[i];
            if row < 0 {
                continue;
            }
            if plane.0 != row {
                let t = &tris[row as usize * TRI_FEATURES..][..9];
                let c0 = [t[0], t[1], t[2]];
                let mut n = cross(sub([t[3], t[4], t[5]], c0), sub([t[6], t[7], t[8]], c0));
                let len = dot(n, n).sqrt().max(1e-12);
                n = n.map(|k| k / len);
                let axis = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs())).unwrap();
                plane = (row, n, dot(sub(c0, pos), n), axis);
            }
            let (_, n, to, axis) = plane;
            let hit = |d: [f32; 3]| {
                let dn = dot(d, n);
                let s = to / if dn.abs() < 1e-6 { 1e-6f32.copysign(dn) } else { dn };
                [0, 1, 2].map(|k| pos[k] + d[k] * s)
            };
            let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            let (p0, px, py) = (hit(ray(x, y)), hit(ray(x + 1.0, y)), hit(ray(x, y + 1.0)));
            let others = match axis {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            let cell = [0, 1, 2].map(|k| (p0[k] - n[k] * 0.01).floor() as i32);
            let face = [axis as i32, (n[axis] > 0.0) as i32, cell[0], cell[1], cell[2]];
            let page = match last {
                Some((f, p)) if f == face => p,
                _ => {
                    let p = self.page(face, evicted);
                    last = Some((face, p));
                    p
                }
            };
            let uv = others.map(|k| p0[k] - p0[k].floor());
            let dist = dot(sub(p0, pos), sub(p0, pos)).sqrt();
            let fog = ((dist - fog_near) / (fog_far - fog_near).max(1e-6)).clamp(0.0, 1.0);
            // The line of sight off the surface, along the page, per unit risen towards the eye.
            let eye = sub(pos, p0);
            let rise = dot(eye, n).abs().max(1e-4);
            self.surfaces[i] = Surface {
                page: page as f32,
                uv: uv.map(|u| u.clamp(0.0, 0.9999)),
                duv_dx: others.map(|k| px[k] - p0[k]),
                duv_dy: others.map(|k| py[k] - p0[k]),
                strength: 1.0 - fog,
                slope: others.map(|k| eye[k] / rise),
                pic: [x, y],
                dist,
            };
            if let Some(p) = &prev {
                let [x0, y0, z] = was(p, p0);
                let (xi, yi) = (x0.floor(), y0.floor());
                let seen = z > NEAR && xi >= 0.0 && yi >= 0.0 && (xi as usize) < w && (yi as usize) < h && {
                    let near = p.nearest[yi as usize * w + xi as usize];
                    (near - 1.0 / z).abs() <= 0.02 / z
                };
                if seen {
                    let (a, b) = (was(p, px), was(p, py));
                    self.motion[i] = Motion { from: [x0, y0], dfrom_dx: [a[0] - x0, a[1] - y0], dfrom_dy: [b[0] - x0, b[1] - y0] };
                }
            }
        }
        let mut nearest = prev.map(|p| p.nearest).unwrap_or_default();
        std::mem::swap(&mut nearest, &mut self.nearest);
        self.prev = Some(CpuView { pos, r, hp, width, nearest });
    }
}

/// The dream renderer in the render world, on Bevy's device, behind a camera of its own; and
/// the window's display moved after it.
pub fn setup(app: &mut App, atlas: &DreamAtlas) {
    let shared = atlas.shared.clone();
    let image = atlas.image.clone();
    app.add_systems(Startup, move |mut commands: Commands| {
        // (A projection, though it draws nothing, or Bevy never works out its target's size
        // and so never runs it.)
        commands.spawn((
            bevy::camera::Camera { order: DREAM_ORDER, ..default() },
            CameraRenderGraph::new(DreamGraph),
            RenderTarget::Image(image.clone().into()),
            Projection::default(),
        ));
    });
    app.add_systems(Update, display_after);
    app.sub_app_mut(RenderApp)
        .insert_resource(DreamGpu { shared, renderer: None, check: None })
        .init_schedule(DreamGraph)
        .add_systems(DreamGraph, dream);
}

fn display_after(mut cams: Query<(&mut bevy::camera::Camera, Option<&RenderLayers>)>) {
    for (mut cam, layers) in &mut cams {
        if layers.is_some_and(|l| l.intersects(&RenderLayers::layer(DISPLAY_LAYER))) && cam.order <= DREAM_ORDER {
            cam.order = DREAM_ORDER + 1;
        }
    }
}

/// The dream camera's schedule: this frame's job, recorded among Bevy's own commands (after
/// the PS1 screen's, before the window's).
fn dream(mut gpu: ResMut<DreamGpu>, device: Res<RenderDevice>, queue: Res<RenderQueue>, images: Res<RenderAssets<GpuImage>>, mut pending: ResMut<PendingCommandBuffers>) {
    let gpu = &mut *gpu;
    let mut shared = gpu.shared.lock().unwrap();
    if gpu.renderer.is_none() {
        let Some(cfg) = shared.config.take() else { return };
        match DreamRenderer::new(device.wgpu_device().clone(), (**queue).clone(), cfg) {
            Ok(r) => gpu.renderer = Some(r),
            Err(e) => {
                eprintln!("dream atlas: {e}");
                shared.error = Some(e);
                return;
            }
        }
    }
    let Some(r) = gpu.renderer.as_mut() else { return };
    if let Some(check) = gpu.check.take() {
        compare(r, &check);
    }
    let (Some(job), Some(_), Some(_)) = (shared.job.as_ref(), shared.job.as_ref().and_then(|j| images.get(j.screen)), shared.job.as_ref().and_then(|j| images.get(j.out))) else { return };
    let (screen, out) = (images.get(job.screen).unwrap(), images.get(job.out).unwrap());
    let job = shared.job.take().unwrap();
    if let Some((tris, textures)) = &job.level {
        if let Err(e) = r.set_textures(64, textures).and_then(|_| r.set_level(tris)) {
            eprintln!("dream atlas: {e}");
        }
    }
    if job.wake {
        let _ = r.wake();
    }
    r.set_lights(job.lights.0, &job.lights.1);
    let mut enc = device.wgpu_device().create_command_encoder(&Default::default());
    if let Err(e) = r.encode_view(&mut enc, &screen.texture, job.width, job.view.as_ref(), job.creatures.as_deref(), job.carry, &job.step, &out.texture) {
        eprintln!("dream atlas: {e}");
        return;
    }
    pending.push_encoder(enc, "dream atlas");
    gpu.check = job.check.filter(|_| job.view.is_some());
    shared.ms = shared.ms * 0.9 + 0.1 * r.last_ms;
    shared.pages_seen = r.pages_seen;
    shared.pages_in_use = r.pages_in_use();
}

/// GAME_DREAM_CHECK: the renderer's G-buffer of the last frame against the CPU's.
fn compare(r: &DreamRenderer, check: &Check) {
    let (gs, gm) = match r.read_gbuffer() {
        Ok(g) => g,
        Err(e) => return eprintln!("dream atlas check: {e}"),
    };
    if gs.len() != check.surfaces.len() {
        return eprintln!("dream atlas check: {} pixels against the CPU's {}", gs.len(), check.surfaces.len());
    }
    let (mut shown, mut paged, mut same, mut other, mut moved, mut both) = (0, 0, 0, 0, 0, 0);
    let mut only = [0, 0];
    let mut worst = [0.0f32; 3];
    for (i, (g, c)) in gs.iter().zip(&check.surfaces).enumerate() {
        let gface = (g.page >= 0.0).then(|| r.face_of_page(g.page as u32)).flatten();
        let cface = check.faces[i];
        shown += cface.is_some() as u32;
        paged += (cface.is_some() && gface.is_some()) as u32;
        if let (Some(cf), Some(gf)) = (cface, gface) {
            if cf == gf {
                same += 1;
                let d = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs().max((a[1] - b[1]).abs());
                worst[0] = worst[0].max(d(g.uv, c.uv));
                worst[1] = worst[1].max((g.strength - c.strength).abs());
            } else {
                other += 1;
            }
        }
        let (gv, cv) = (gm[i].from[0] > -1e8, check.motion[i].from[0] > -1e8);
        moved += cv as u32;
        if gv && cv {
            both += 1;
            worst[2] = worst[2].max((gm[i].from[0] - check.motion[i].from[0]).abs().max((gm[i].from[1] - check.motion[i].from[1]).abs()));
        } else if gv != cv {
            only[gv as usize] += 1;
        }
    }
    eprintln!(
        "dream atlas check: {shown} pixels on a face by the CPU, {paged} with a page by the GPU, {same} on the same face, {other} on another; largest uv difference {:.1e}, strength {:.1e}; motion {moved} by the CPU, {both} by both (only the CPU {}, only the GPU {}), largest difference {:.1e} pixels",
        worst[0], worst[1], only[0], only[1], worst[2]
    );
}

/// The level's triangles as the dream renderer draws them (from their features: corners,
/// place on the texture, the level's light and the torch's albedo, each 128 for as it is), each
/// on the texture the PS1 renderer gives it (its strange step where the level's strange), and
/// those textures, in the order the triangles count them.
fn level_tris(level: &engine::neural::Level, textures: Option<&engine::render::LevelTextures>) -> (Vec<LevelTri>, Vec<Vec<u8>>) {
    use engine::world::tex;
    let scene = &level.scene;
    let plain = scene.sky.is_some();
    let mut layers: Vec<Vec<u8>> = Vec::new();
    let mut layer_of: HashMap<u8, u32> = HashMap::new();
    let tris = level
        .tris
        .chunks_exact(TRI_FEATURES)
        .enumerate()
        .map(|(i, t)| {
            let base = t[15] as u8;
            // Things (doors, keys, levers) after the level's own triangles keep their texture.
            let id = if i < scene.tris.len() {
                let centre = [0, 1, 2].map(|c| (t[c] + t[3 + c] + t[6 + c]) / 3.0 * engine::world::scene::T as f32);
                tex::strange_variant(base, scene.morph(centre)).0
            } else {
                base
            };
            let texture = *layer_of.entry(id).or_insert_with(|| {
                layers.push(textures.and_then(|tx| tx.rgba(id, plain)).unwrap_or_else(|| vec![128; 64 * 64 * 4]));
                layers.len() as u32 - 1
            });
            let neutral = 255.0 / 128.0;
            LevelTri {
                pos: std::array::from_fn(|k| t[k]),
                uv: std::array::from_fn(|k| t[9 + k] * 64.0 / 63.0),
                light: std::array::from_fn(|k| t[16 + k] * neutral),
                albedo: std::array::from_fn(|k| t[25 + k] * neutral),
                texture,
            }
        })
        .collect();
    (tris, layers)
}

/// Each frame: the G-buffer of the PS1 screen's view (unless it hasn't moved), handed to the
/// render world to dream on, and the sprite shown the frame it makes.
pub fn draw(
    time: Res<Time>,
    sim: Res<Sim>,
    screen: Res<PsxScreen>,
    creatures: Res<crate::Creatures>,
    shown: Res<engine::render::DreamCreatures>,
    textures: Option<Res<engine::render::LevelTextures>>,
    atlas: Option<NonSendMut<DreamAtlas>>,
    mut sprite: Query<&mut Sprite, With<NetworkSprite>>,
) {
    let Some(mut atlas) = atlas else { return };
    if sim.shown() == Mode::Raster {
        return;
    }
    let atlas = &mut *atlas;
    atlas.frame += 1;
    let n = atlas.frame;
    let width = screen.size().x;
    if width > MAX_W {
        return;
    }
    let cam = sim.level.walker.camera();
    let moving = atlas.last_cam.is_some_and(|c| c.pos != cam.pos || c.yaw != cam.yaw || c.pitch != cam.pitch);
    atlas.last_cam = Some(cam);
    atlas.sharp = if moving { 0.0 } else { (atlas.sharp + time.delta_secs() / SETTLE_SECS).min(1.0) };
    // The creatures' depth where they're drawn (f32::MAX elsewhere), to leave them out.
    let creatures: Vec<f32> = if shown.on && creatures.near.len() == (width * H) as usize {
        creatures.near.iter().enumerate().map(|(i, &z)| if creatures.rgba[i * 4 + 3] > 0.5 { z } else { f32::MAX }).collect()
    } else {
        Vec::new()
    };
    let t0 = std::time::Instant::now();
    // The same view as the last frame's, with no creatures in it: its surfaces still hold.
    let still = atlas.mode != DreamMode::Walls && !atlas.clear && creatures.iter().all(|&z| z == f32::MAX) && atlas.drawn.is_some_and(|(c, w)| w == width && c.pos == cam.pos && c.yaw == cam.yaw && c.pitch == cam.pitch);
    atlas.drawn = Some((cam, width));
    // The level's triangles, given again when they change (a door moves, its light changes,
    // another level).
    let level = (sim.level.tris != atlas.level).then(|| {
        atlas.level = sim.level.tris.clone();
        level_tris(&sim.level, textures.as_deref())
    });
    let v = sim.level.view(&cam, width);
    let view = View {
        pos: [v[1], v[2], v[3]],
        axes: [[v[4], v[5], v[6]], [v[7], v[8], v[9]], [v[10], v[11], v[12]]],
        focal: v[13],
        fog: [v[14], v[15]],
        torch: [v[19], v[20], v[21], v[22]],
        fog_colour: [v[23], v[24], v[25]],
        torch_at: atlas.torch_at,
    };
    let mut check = None;
    if atlas.check && !still {
        let mut evicted = Vec::new();
        atlas.surfaces(&sim, &cam, width, &creatures, &mut evicted);
        check = (n % 60 == 0).then(|| Check {
            faces: atlas.surfaces.iter().map(|s| (s.page >= 0.0).then(|| atlas.owners[s.page as usize].map(|f| [f[0] * 2 + f[1], f[2], f[3], f[4]])).flatten()).collect(),
            surfaces: atlas.surfaces.clone(),
            motion: atlas.motion.clone(),
        });
    }
    let wake = std::mem::take(&mut atlas.clear);
    atlas.gbuffer_ms = atlas.gbuffer_ms * 0.9 + 0.1 * t0.elapsed().as_secs_f32() * 1000.0;
    // GAME_DREAM_PAGES=DIR: every 60th frame, each pixel's page (a colour a page, black for
    // none), to check the G-buffer by.
    if let Some(dir) = std::env::var_os("GAME_DREAM_PAGES").filter(|_| n % 60 == 0 && !still && atlas.check) {
        let (w, h) = (width as usize, H as usize);
        let mut ppm = format!("P6 {w} {h} 255\n").into_bytes();
        for i in 0..w * h {
            let p = atlas.surfaces[i].page;
            ppm.extend(if p < 0.0 { [0, 0, 0] } else { [0u32, 1, 2].map(|c| ((p as u32 * 2654435761u32).rotate_left(c * 8) >> 24) as u8 | 64) });
        }
        let _ = std::fs::write(Path::new(&dir).join(format!("pages_{n:05}.ppm")), ppm);
    }
    let mut step = atlas.step;
    if atlas.settle {
        step.sharpness = atlas.sharp;
    }
    if !atlas.dreaming {
        (step.lr, step.lr_fine) = (0.0, 0.0);
    }
    atlas.stills += still as u64;
    let job = Job {
        width,
        screen: screen.image.id(),
        out: atlas.image.id(),
        level,
        view: (!still).then_some(view),
        creatures: (!still && !creatures.is_empty()).then_some(creatures),
        carry: atlas.mode == DreamMode::Reproject,
        step,
        wake,
        check,
        lights: level_lights(&sim.level.scene, &cam),
    };
    {
        let mut shared = atlas.shared.lock().unwrap();
        // A job the render world hasn't taken yet: its level and waking still stand, and a
        // view it gave still holds if this one doesn't give its own.
        let job = match shared.job.take() {
            Some(old) => {
                let (view, creatures) = if job.view.is_some() { (job.view, job.creatures) } else { (old.view, old.creatures) };
                Job { level: job.level.or(old.level), wake: old.wake || job.wake, view, creatures, check: job.check.or(old.check), ..job }
            }
            None => job,
        };
        shared.job = Some(job);
        atlas.ms = shared.ms;
        atlas.pages_seen = shared.pages_seen;
        atlas.pages_in_use = shared.pages_in_use;
    }
    if std::env::var_os("GAME_FPS").is_some() && n % 60 == 0 {
        eprintln!(
            "dream atlas: {:.1} ms dreaming (recording), {:.1} ms the main world's part, {} pages in sight, {} in all, {} frames still",
            atlas.ms,
            atlas.gbuffer_ms,
            atlas.pages_seen,
            atlas.pages_in_use,
            atlas.stills
        );
    }
    if let Ok(mut sprite) = sprite.single_mut() {
        if sprite.image != atlas.image {
            sprite.image = atlas.image.clone();
        }
        sprite.rect = Some(Rect::new(0.0, 0.0, (width * UP) as f32, (H * UP) as f32));
    }
}

pub fn panel(mut contexts: EguiContexts, sim: Res<Sim>, atlas: Option<NonSendMut<DreamAtlas>>) -> Result {
    let Some(mut atlas) = atlas else { return Ok(()) };
    if !sim.panel {
        return Ok(());
    }
    let a = &mut *atlas;
    egui::Window::new("dream atlas").default_pos([340.0, 60.0]).resizable(false).show(contexts.ctx_mut()?, |ui| {
        ui.label(format!("{:.1} ms a frame dreaming, {:.1} ms the main world's part", a.ms, a.gbuffer_ms));
        ui.label(format!("{} pages in sight (up to {MAX_TOUCHED} dreamed a frame), {} in all", a.pages_seen, a.pages_in_use));
        ui.checkbox(&mut a.dreaming, "dreaming");
        if a.net_names.len() > 1 {
            ui.horizontal(|ui| {
                ui.label("dreaming with");
                let mut net = a.step.net;
                for (i, name) in a.net_names.iter().enumerate() {
                    ui.radio_value(&mut net, i, name.as_str());
                }
                if net != a.step.net {
                    a.switch_net(net);
                }
            });
        }
        ui.add(egui::Slider::new(&mut a.step.scale, 0.25..=1.5).text("coarse: the picture seen at (smaller dreams bigger)"));
        ui.add(egui::Slider::new(&mut a.step.fine_scale, 1.0..=3.0).text("fine: the picture seen at (smaller dreams bigger)"));
        ui.add(egui::Slider::new(&mut a.step.lr, 0.0..=0.05).text("step, coarse"));
        ui.add(egui::Slider::new(&mut a.step.lr_fine, 0.0..=0.05).text("step, fine"));
        ui.add(egui::Slider::new(&mut a.step.coarse_every, 1..=8).text("coarse every n frames"));
        ui.add(egui::Slider::new(&mut a.step.fine_every, 1..=8).text("fine every n frames"));
        ui.add(egui::Slider::new(&mut a.step.fine_offset, 0..=7).text("fine offset (frames)"));
        ui.add(egui::Slider::new(&mut a.step.fine_split, 1..=6).text("fine piece: 1/n of the frame each way"));
        ui.add(egui::Slider::new(&mut a.step.colour, 0.0..=1.0).text("colour (0: light and shade only)"));
        ui.add(egui::Slider::new(&mut a.step.colour_blur, 1..=8).text("colour spread wider by"));
        ui.add(egui::Slider::new(&mut a.step.grad_blur, 0.0..=4.0).text("gradient blurred (net pixels; against ResNet's grating)"));
        ui.add(egui::Slider::new(&mut a.step.take_over, 0.0..=0.1).logarithmic(true).text("takes the wall over a step"));
        ui.add(egui::Slider::new(&mut a.step.own_max, 0.0..=1.0).text("most of the wall it owns"));
        ui.add(egui::Slider::new(&mut a.step.relief, 0.0..=10.0).logarithmic(true).text("relief (tiles off the wall)"));
        ui.add(egui::Slider::new(&mut a.step.relief_rate, 0.0..=0.3).logarithmic(true).text("relief rises a frame"));
        ui.add(egui::Slider::new(&mut a.step.decay, 0.0..=0.02).logarithmic(true).text("let go a frame"));
        ui.add(egui::Slider::new(&mut a.step.limit, 0.05..=1.0).text("most it may change"));
        ui.add(egui::Slider::new(&mut a.step.jitter, 0..=12).text("jitter, coarse"));
        ui.add(egui::Slider::new(&mut a.step.jitter_fine, 0..=16).text("jitter, fine"));
        ui.checkbox(&mut a.settle, "settle: blocky while moving, sharp when still");
        ui.horizontal(|ui| {
            ui.label("dream on");
            ui.radio_value(&mut a.mode, DreamMode::Walls, "the walls");
            ui.radio_value(&mut a.mode, DreamMode::Still, "the screen when still");
            ui.radio_value(&mut a.mode, DreamMode::Reproject, "the screen, carried along");
        });
        ui.add(egui::Slider::new(&mut a.step.remember_every, 0..=16).text("carried along: put on the walls over n frames (0: forget)"));
        ui.label("carried along, the dreamed height:");
        ui.add(egui::Slider::new(&mut a.step.height, 0.0..=0.1).logarithmic(true).text("raised a step (tiles; 0: flat)"));
        ui.add(egui::Slider::new(&mut a.step.height_max, 0.0..=10.0).logarithmic(true).text("highest (tiles)"));
        ui.add(egui::Slider::new(&mut a.step.height_blur, 1..=12).text("each step blurred by (pixels)"));
        ui.add(egui::Slider::new(&mut a.step.height_spread, 1..=6).text("blurred at this many sizes, each twice the last (whole shapes rise)"));
        ui.add(egui::Slider::new(&mut a.step.height_power, 0.0..=2.0).text("bigger sizes weigh more by (1: half way, 2: integrated)"));
        ui.add(egui::Slider::new(&mut a.step.height_from_colour, -1.0..=1.0).text("rises where the dream brightens (vs the lamp)"));
        ui.add(egui::Slider::new(&mut a.step.height_decay, 0.0..=0.05).logarithmic(true).text("height let go a step (settles, not filling up)"));
        ui.checkbox(&mut a.step.height_centred, "each step about its own mean (rises and sinks alike)");
        ui.add(egui::Slider::new(&mut a.step.lamp, 0.0..=1.0).text("dream lamp: how much it shades what the dream sees"));
        ui.add(egui::Slider::new(&mut a.step.lamp_ambient, 0.0..=1.0).text("dream lamp: light from everywhere"));
        ui.add(egui::Slider::new(&mut a.step.shade_only_every, 0..=8).text("the dream sees the shading alone every n steps (0: never)"));
        ui.add(egui::Slider::new(&mut a.step.show_shade, 0.0..=1.0).text("shown shaded by the height"));
        ui.add(egui::Slider::new(&mut a.step.show_wrap, 0.0..=2.0).text("its light wraps round (0: hard, 1: soft, matte)"));
        ui.checkbox(&mut a.step.show_height, "show the height alone");
        ui.add(egui::Slider::new(&mut a.step.displace, 0.0..=2.0).text("the walls pushed out by it, subdivided (0: flat, the dream unmoved by it)"));
        ui.add(egui::Slider::new(&mut a.step.shell, 0.0..=2.0).text("the relief marched in shells over the flat walls (0: none)"));
        ui.add(egui::Slider::new(&mut a.step.shell_smooth, 0.0..=6.0).text("the relief smoothed (levels, each twice as broad; 0: as dreamed)"));
        ui.add(egui::Slider::new(&mut a.step.shell_far, 0.0..=2.0).text("and smoother far off (two levels more at 4 / this tiles; 0: none)"));
        ui.add(egui::Slider::new(&mut a.step.shell_pixel, 0.0..=1.0).text("the relief found to this much of a pixel (0: as finely as it goes)"));
        ui.checkbox(&mut a.step.show_shell, "show the shells' relief alone (grey by height, red: nothing met)");
        ui.checkbox(&mut a.step.own_picture, "the level drawn by the dream renderer (not the PS1 picture) under the dream");
        ui.label("the torch, held (tiles from the eye):");
        ui.add(egui::Slider::new(&mut a.torch_at[0], -0.5..=0.5).text("right"));
        ui.add(egui::Slider::new(&mut a.torch_at[1], -0.5..=0.5).text("down"));
        ui.add(egui::Slider::new(&mut a.torch_at[2], -0.5..=0.8).text("forward"));
        ui.checkbox(&mut a.step.lights, "the height lit by the torch and the level's lights from where they are (else from the eye)");
        ui.add(egui::Slider::new(&mut a.step.dream_lit, 0.0..=1.0).text("the dream lit as the wall is (0: it glows)"));
        ui.label("the torch's glint:");
        ui.add(egui::Slider::new(&mut a.step.glint, 0.0..=3.0).text("how bright (0: none)"));
        ui.add(egui::Slider::new(&mut a.step.torch_reach, 1.0..=15.0).text("how far it reaches (tiles)"));
        ui.add(egui::Slider::new(&mut a.step.rough_dry, 0.05..=1.0).text("rough, plain wall"));
        ui.add(egui::Slider::new(&mut a.step.rough_wet, 0.05..=1.0).text("rough, where the dream owns it"));
        if !a.settle {
            ui.add(egui::Slider::new(&mut a.step.sharpness, 0.0..=1.0).text("sharpness"));
        }
        if ui.button("wake up (clear the dream)").clicked() {
            a.clear = true;
        }
    });
    Ok(())
}

/// The level's light from everywhere and its lights as they are now (strangeness and all), in
/// tiles, for lighting the dreamed height from where they are: the sun from far off (towards it).
fn level_lights(scene: &engine::world::scene::Scene, cam: &Camera) -> ([f32; 3], Vec<Light>) {
    let t = engine::render::TF;
    let (ambient, sun_dir, sun, points) = scene.lights_now(cam.pos.map(|c| c as f32));
    let mut lights: Vec<Light> = points.into_iter().map(|(pos, colour, radius)| Light { pos: pos.map(|c| c / t), colour, radius: radius / t }).collect();
    if sun.iter().any(|&c| c > 1e-3) {
        lights.push(Light { pos: sun_dir.map(|c| -c), colour: sun, radius: 0.0 });
    }
    (ambient, lights)
}
