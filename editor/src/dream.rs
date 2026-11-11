//! Creatures as the game draws them: the dream net (`engine::neural::dream_net`), found
//! through `neurofield.ron`'s `dream`. It runs on a worker thread, which loads it the first
//! time it is wanted and always draws the newest view, so editing stays smooth however slow
//! a frame is. In the Plain and PS1 modes its picture is laid into the 3D view through the
//! flesh pass (`engine::render::DreamCreatures`), hidden by the level in front; in the Neural
//! mode the network's worker lays it over the network's frame (`crate::neural`). The
//! creatures stand as they were placed: nothing walks them about in the editor.

use crate::view3d::{Built, EditorCam, ViewMode};
use bevy::prelude::*;
use engine::neural::dream_net::DreamLayer;
use engine::world::creature::Creature;
use engine::world::growths::Growths;
use engine::world::level::Level;
use engine::world::scene::{Camera as PsxCamera, Scene};
use engine::world::view::H;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The dream net's seed, which picks each creature's look along with its shape.
const SEED: u64 = 7;

/// What the creatures are drawn from: the level (cut at a slice, as shown), which creatures
/// it has, and the camera.
#[derive(Clone)]
pub struct Creatures {
    /// `Built::version`: the level changed.
    pub version: u64,
    pub level: Arc<Level>,
    pub cut: Option<i32>,
    /// The creatures' shapes together: when they change, so do their looks.
    pub shapes: u64,
    pub cam: PsxCamera,
}

/// The shapes of a level's creatures, as one number.
pub fn shapes_key(shapes: &[u64]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    shapes.hash(&mut h);
    h.finish()
}

/// The level, shared with a worker: copied once each time it changes.
#[derive(Default)]
pub struct SharedLevel(Option<(u64, Arc<Level>)>);

impl SharedLevel {
    pub fn of(&mut self, version: u64, level: &Level) -> Arc<Level> {
        if !self.0.as_ref().is_some_and(|s| s.0 == version) {
            self.0 = Some((version, Arc::new(level.clone())));
        }
        self.0.as_ref().unwrap().1.clone()
    }
}

/// The dream net with the scene it draws, on a worker thread.
pub struct Dreamer {
    layer: DreamLayer,
    /// The scene, built here from the level, with the extra creature the dream net is
    /// always given (the tentacles' in the game; none grow in the editor).
    scene: Option<(u64, Option<i32>, Scene, Creature)>,
}

impl Dreamer {
    pub fn load(path: &Path) -> Result<Dreamer, String> {
        Ok(Dreamer { layer: DreamLayer::load(path, SEED)?, scene: None })
    }

    pub fn adapter(&self) -> String {
        self.layer.adapter().to_string()
    }

    fn take(&mut self, c: &Creatures) -> (&mut DreamLayer, &mut Scene, Creature) {
        if !self.scene.as_ref().is_some_and(|s| (s.0, s.1) == (c.version, c.cut)) {
            let scene = match c.cut {
                Some(slice) => crate::view3d::cut_level(&c.level, slice).to_scene(),
                None => c.level.to_scene(),
            };
            let extra = Growths::new(&scene).creature();
            self.scene = Some((c.version, c.cut, scene, extra));
        }
        let (_, _, scene, extra) = self.scene.as_mut().unwrap();
        (&mut self.layer, scene, extra.clone())
    }

    /// Lay the creatures over `rgb` ([H][width][3]), hidden by the level triangles `tris`
    /// (`features::tri_features`).
    pub fn over(&mut self, c: &Creatures, tris: &[f32], rgb: &mut [f32], width: u32) -> Result<(), String> {
        let (layer, scene, extra) = self.take(c);
        layer.draw(scene, c.shapes, tris, extra, |p| p, &c.cam, rgb, width)
    }

    /// The creatures on their own: `rgba` ([H][width][4]) and their depth `near` ([H][width]).
    pub fn alone(&mut self, c: &Creatures, rgba: &mut [f32], near: &mut [f32], width: u32) -> Result<(), String> {
        let (layer, scene, extra) = self.take(c);
        layer.draw_alone(scene, c.shapes, &[], extra, |p| p, &c.cam, rgba, near, width)
    }
}

/// What a frame was drawn from.
#[derive(Clone, Copy, PartialEq)]
struct Key {
    version: u64,
    cut: Option<i32>,
    shapes: u64,
    cam: [i32; 5],
    width: u32,
}

struct Job {
    key: Key,
    creatures: Creatures,
}

enum Reply {
    Loaded(String),
    Failed(String),
    Frame { key: Key, rgba: Vec<f32>, near: Vec<f32>, ms: f32 },
}

#[derive(Clone, PartialEq)]
pub enum State {
    Off,
    Loading,
    /// On the named adapter.
    Ready(String),
    Failed(String),
}

/// The dream net's pictures for the Plain and PS1 modes.
#[derive(Resource)]
pub struct DreamView {
    /// Where the dream net is (`neurofield.ron`'s `dream`).
    path: Option<PathBuf>,
    jobs: Option<mpsc::Sender<Job>>,
    /// In a mutex only so the view can live in a resource.
    replies: Option<Mutex<mpsc::Receiver<Reply>>>,
    pub state: State,
    busy: bool,
    asked: Option<Key>,
    level: SharedLevel,
    /// Milliseconds the last frame took.
    pub ms: f32,
}

impl DreamView {
    pub fn new(path: Option<PathBuf>) -> DreamView {
        DreamView { path, jobs: None, replies: None, state: State::Off, busy: false, asked: None, level: SharedLevel::default(), ms: 0.0 }
    }

    /// Whether the dream net is there to draw the creatures.
    pub fn ready(&self) -> bool {
        matches!(self.state, State::Ready(_))
    }

    fn start(&mut self) {
        let path = match &self.path {
            Some(p) if p.is_file() => p.clone(),
            p => {
                let at = p.as_ref().map_or("(none set)".into(), |p| p.display().to_string());
                self.state = State::Failed(format!("no dream net at {at}: set `dream` in neurofield.ron"));
                return;
            }
        };
        let (jobs, rx) = mpsc::channel::<Job>();
        let (tx, replies) = mpsc::channel();
        std::thread::Builder::new()
            .name("editor-dream".into())
            .spawn(move || {
                let mut dreamer = match Dreamer::load(&path) {
                    Ok(d) => d,
                    Err(e) => {
                        _ = tx.send(Reply::Failed(e));
                        return;
                    }
                };
                _ = tx.send(Reply::Loaded(dreamer.adapter()));
                while let Ok(mut job) = rx.recv() {
                    // Only the newest view matters.
                    while let Ok(newer) = rx.try_recv() {
                        job = newer;
                    }
                    let t0 = Instant::now();
                    let n = (job.key.width * H) as usize;
                    let (mut rgba, mut near) = (vec![0.0; n * 4], vec![f32::MAX; n]);
                    let reply = match dreamer.alone(&job.creatures, &mut rgba, &mut near, job.key.width) {
                        Ok(()) => Reply::Frame { key: job.key, rgba, near, ms: t0.elapsed().as_secs_f32() * 1000.0 },
                        Err(e) => Reply::Failed(e),
                    };
                    if tx.send(reply).is_err() {
                        return;
                    }
                }
            })
            .expect("starting the dream view's thread");
        self.jobs = Some(jobs);
        self.replies = Some(Mutex::new(replies));
        self.state = State::Loading;
    }
}

/// An image the size of a dream frame, for `DreamCreatures`.
fn float_image(width: u32, format: bevy::render::render_resource::TextureFormat, data: Vec<f32>) -> Image {
    use bevy::render::render_resource::{Extent3d, TextureDimension};
    let mut image = Image::new(
        Extent3d { width, height: H, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data.iter().flat_map(|f| f.to_ne_bytes()).collect(),
        format,
        bevy::asset::RenderAssetUsages::default(),
    );
    image.sampler = bevy::image::ImageSampler::nearest();
    image
}

/// Ask for the creatures as the camera now sees them, when the level or the camera has
/// moved and the last frame is done; lay finished frames into the view. Only in the Plain
/// and PS1 modes, with the Flesh toggle on and creatures to draw.
#[allow(clippy::too_many_arguments)]
pub fn update(
    mut view: ResMut<DreamView>,
    doc: Res<crate::Document>,
    mut built: ResMut<Built>,
    cam: Res<EditorCam>,
    windows: Query<&Window>,
    mut shown: ResMut<engine::render::DreamCreatures>,
    mut images: ResMut<Assets<Image>>,
) {
    let want = built.full_colour && !built.scene.creatures.is_empty();
    // In the Neural mode the network's worker draws them (`crate::neural`).
    if want && built.mode != ViewMode::Neural && view.state == State::Off {
        view.start();
    }
    let dreamed = want && view.ready();
    if built.dreamed != dreamed {
        built.dreamed = dreamed;
    }
    while let Some(reply) = view.replies.as_ref().and_then(|r| r.lock().ok()?.try_recv().ok()) {
        match reply {
            Reply::Loaded(adapter) => view.state = State::Ready(adapter),
            Reply::Failed(e) => {
                eprintln!("dream view: {e}");
                view.state = State::Failed(e);
                view.jobs = None;
            }
            Reply::Frame { key, rgba, near, ms } => {
                view.busy = false;
                view.ms = ms;
                use bevy::render::render_resource::TextureFormat;
                for (handle, image) in [
                    (&shown.colour, float_image(key.width, TextureFormat::Rgba32Float, rgba)),
                    (&shown.near, float_image(key.width, TextureFormat::R32Float, near)),
                ] {
                    if let Some(mut old) = images.get_mut(handle) {
                        *old = image;
                    }
                }
            }
        }
    }
    let on = dreamed && built.mode != ViewMode::Neural && view.asked.is_some();
    if shown.on != on {
        shown.on = on;
    }
    if !dreamed || built.mode == ViewMode::Neural || view.busy {
        return;
    }
    let Ok(window) = windows.single() else { return };
    // The width of a 240-line screen the window's shape, as the PS1 mode's.
    let width = psx_bevy::psx::width_for_aspect(H, window.size());
    let c = cam.psx();
    let key = Key {
        version: built.version,
        cut: built.cut_at,
        shapes: shapes_key(&built.shapes),
        cam: [c.pos[0], c.pos[1], c.pos[2], c.yaw, c.pitch],
        width,
    };
    if view.asked == Some(key) {
        return;
    }
    let level = view.level.of(built.version, &doc.0.level);
    let creatures = Creatures { version: built.version, level, cut: built.cut_at, shapes: key.shapes, cam: c };
    if view.jobs.as_ref().is_some_and(|j| j.send(Job { key, creatures }).is_ok()) {
        view.busy = true;
        view.asked = Some(key);
    }
}
