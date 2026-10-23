//! The Neural view mode: the trained network drawing the editor camera's view.
//! It runs on a worker thread, which loads the model the first time it is asked
//! for, so editing stays smooth however long a frame takes; the latest frame is
//! drawn over the 3D view, with the tools' outlines on top.

use crate::view3d::{Built, EditorCam};
use bevy_egui::egui;
use engine::neural::{dream::Dream, features, tokens, Network};
use engine::world::view::{H, W};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What a frame was drawn from: the level as built, and where the camera was.
#[derive(Clone, Copy, PartialEq)]
struct Key {
    version: u64,
    cut: Option<i32>,
    cam: [u32; 5],
}

struct Job {
    key: Key,
    tris: Arc<Vec<f32>>,
    view: Vec<f32>,
}

enum Reply {
    Loaded(String),
    Failed(String),
    Frame { key: Key, rgb: Vec<f32>, ms: f32 },
}

#[derive(Clone, PartialEq)]
pub enum State {
    Off,
    Loading,
    /// On the named adapter.
    Ready(String),
    Failed(String),
}

pub struct NeuralView {
    jobs: Option<mpsc::Sender<Job>>,
    /// In a mutex only so the view can live in a resource.
    replies: Option<Mutex<mpsc::Receiver<Reply>>>,
    pub state: State,
    /// A frame is being drawn.
    busy: bool,
    /// What the frame shown (or being drawn) is of.
    asked: Option<Key>,
    /// The triangle features of the level as built, by `Built::version` and cut.
    tris: Option<(u64, Option<i32>, Arc<Vec<f32>>)>,
    /// Milliseconds the last frame took.
    pub ms: f32,
    pub frame: Option<egui::TextureHandle>,
}

impl Default for NeuralView {
    fn default() -> Self {
        NeuralView { jobs: None, replies: None, state: State::Off, busy: false, asked: None, tris: None, ms: 0.0, frame: None }
    }
}

impl NeuralView {
    /// Start the worker, which loads the model at `model`.
    fn start(&mut self, model: PathBuf) {
        let (jobs, rx) = mpsc::channel::<Job>();
        let (tx, replies) = mpsc::channel();
        std::thread::Builder::new()
            .name("editor-neural".into())
            .spawn(move || {
                let mut net = match Network::try_load(&model) {
                    Ok(n) => n,
                    Err(e) => {
                        _ = tx.send(Reply::Failed(e));
                        return;
                    }
                };
                _ = tx.send(Reply::Loaded(net.adapter().to_string()));
                let dream = Dream::calm();
                while let Ok(mut job) = rx.recv() {
                    // Only the newest view matters.
                    while let Ok(newer) = rx.try_recv() {
                        job = newer;
                    }
                    let t0 = Instant::now();
                    let t = tokens::tokens(&job.tris, &job.view);
                    let reply = match net.try_render(&t, &dream) {
                        Ok(rgb) => Reply::Frame { key: job.key, rgb, ms: t0.elapsed().as_secs_f32() * 1000.0 },
                        Err(e) => Reply::Failed(e),
                    };
                    if tx.send(reply).is_err() {
                        return;
                    }
                }
            })
            .expect("starting the neural view's thread");
        self.jobs = Some(jobs);
        self.replies = Some(Mutex::new(replies));
        self.state = State::Loading;
    }

    /// Ask for a new frame when the level or the camera has moved and the last one
    /// is done, and take in finished ones. `model` is where the network is, if set.
    pub fn update(&mut self, ctx: &egui::Context, built: &Built, cam: &EditorCam, model: Option<PathBuf>) {
        if self.state == State::Off {
            match model {
                Some(m) if m.is_file() => self.start(m),
                m => {
                    self.state = State::Failed(format!(
                        "no model at {}: run ./model.sh, or set `model` in neurofield.ron",
                        m.map_or("(none set)".into(), |m| m.display().to_string())
                    ));
                }
            }
        }
        while let Some(reply) = self.replies.as_ref().and_then(|r| r.lock().ok()?.try_recv().ok()) {
            match reply {
                Reply::Loaded(adapter) => self.state = State::Ready(adapter),
                Reply::Failed(e) => {
                    self.state = State::Failed(e);
                    self.jobs = None;
                }
                Reply::Frame { key, rgb, ms } => {
                    self.busy = false;
                    self.ms = ms;
                    let pixels = rgb.chunks_exact(3).map(|c| egui::Color32::from_rgb(byte(c[0]), byte(c[1]), byte(c[2]))).collect();
                    let image = egui::ColorImage::new([W as usize, H as usize], pixels);
                    match &mut self.frame {
                        Some(t) => t.set(image, egui::TextureOptions::NEAREST),
                        None => self.frame = Some(ctx.load_texture("neural frame", image, egui::TextureOptions::NEAREST)),
                    }
                    if self.asked != Some(key) {
                        ctx.request_repaint();
                    }
                }
            }
        }
        if !matches!(self.state, State::Ready(_)) || self.busy {
            return;
        }
        let c = cam.psx();
        let key = Key {
            version: built.version,
            cut: built.cut_at,
            cam: [c.pos[0] as u32, c.pos[1] as u32, c.pos[2] as u32, c.yaw as u32, c.pitch as u32],
        };
        if self.asked == Some(key) {
            return;
        }
        if !self.tris.as_ref().is_some_and(|t| (t.0, t.1) == (built.version, built.cut_at)) {
            let mut v = Vec::new();
            features::tri_features(&built.scene, &mut v);
            self.tris = Some((built.version, built.cut_at, Arc::new(v)));
        }
        let mut view = Vec::new();
        features::view_features(&built.scene, 0, &c, &mut view);
        let tris = self.tris.as_ref().unwrap().2.clone();
        if self.jobs.as_ref().is_some_and(|j| j.send(Job { key, tris, view }).is_ok()) {
            self.busy = true;
            self.asked = Some(key);
        }
    }

    /// A frame is on its way, for one of the camera or level that has since changed.
    pub fn pending(&self) -> bool {
        self.busy
    }
}

fn byte(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0) as u8
}

/// Draw the network's frame over the 3D view (which fills `screen`), as the PS1
/// screen would show it: 4:3, the window's height, in the middle. Then the tools'
/// outlines and the entity markers, which the frame covers.
pub fn draw(painter: &egui::Painter, screen: egui::Rect, view: &NeuralView, built: &Built, cam: &EditorCam, markers: &[(bevy::prelude::Vec3, egui::Color32)]) {
    let h = screen.height();
    let rect = egui::Rect::from_center_size(screen.center(), egui::vec2(h * W as f32 / H as f32, h));
    painter.rect_filled(screen, 0.0, egui::Color32::BLACK);
    if let Some(t) = &view.frame {
        painter.image(t.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
    }
    let colour = |c: bevy::prelude::Color| {
        let c = c.to_srgba();
        egui::Color32::from_rgba_unmultiplied((c.red * 255.0) as u8, (c.green * 255.0) as u8, (c.blue * 255.0) as u8, (c.alpha * 255.0) as u8)
    };
    for &(a, b, c) in &built.lines {
        if let (Some(a), Some(b)) = (crate::pick::project(cam, screen, a), crate::pick::project(cam, screen, b)) {
            painter.line_segment([a, b], egui::Stroke::new(1.5, colour(c)));
        }
    }
    for &(p, c) in markers {
        if let Some(p) = crate::pick::project(cam, screen, p) {
            painter.circle(p, 3.5, c, egui::Stroke::new(1.0, egui::Color32::BLACK));
        }
    }
    let text = match &view.state {
        State::Off | State::Loading => "loading the network…".to_string(),
        State::Ready(_) if view.frame.is_none() => "drawing…".to_string(),
        State::Ready(_) => String::new(),
        State::Failed(e) => e.clone(),
    };
    if !text.is_empty() {
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(14.0), egui::Color32::from_gray(200));
    }
}
