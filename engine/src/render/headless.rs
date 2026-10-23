//! The game's renderer without a window, for the training data: the same
//! psx-bevy screen, level meshes, flesh pass and `View`, with each finished
//! frame and its creature mask read back from the GPU.

use crate::render::{flesh, FleshTarget, Gfx, LevelGfxPlugin, View};
use bevy::app::PluginsState;
use bevy::ecs::system::SystemState;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::RenderPlugin;
use bevy::window::ExitCondition;
use crate::world::scene::{Camera as PsxCamera, Scene};
use crate::world::view::{H, W};
use psx_bevy::psx::{PsxPlugin, PsxScreen};
use std::sync::{Arc, Mutex};

/// Frames to run after the shaders are in, so every library they import has
/// loaded and the first pipelines are made.
const SETTLE_FRAMES: u32 = 8;
/// Give up waiting for a frame after this many updates.
const MAX_WAIT: u32 = 600;

pub struct Renderer {
    app: App,
    parts: Vec<Entity>,
}

/// One rendered view: the 320x240 RGB8 frame as the PS1 screen holds it, and
/// per pixel whether creatures cover at least half of it (1) or not (0).
pub struct Frame {
    pub rgb: Vec<u8>,
    pub mask: Vec<u8>,
}

type Slot = Arc<Mutex<Option<Vec<u8>>>>;

impl Renderer {
    pub fn new() -> Renderer {
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin { primary_window: None, exit_condition: ExitCondition::DontExit, close_when_requested: false, ..default() })
                .set(ImagePlugin::default_nearest())
                // Every pipeline is ready before a frame is drawn, so none drops a mesh.
                .set(RenderPlugin { synchronous_pipeline_compilation: true, ..default() })
                .set(bevy::log::LogPlugin { filter: "warn,wgpu=error,naga=error".into(), ..default() })
                .disable::<bevy::winit::WinitPlugin>()
                .disable::<bevy::gilrs::GilrsPlugin>()
                // Each update draws its own frame, start to finish.
                .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>(),
        )
        .add_plugins(PsxPlugin { screen: PsxScreen { width: W, height: H, ..default() } })
        .add_plugins(LevelGfxPlugin);
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
        let mut frames = 0;
        while !flesh::shaders_ready(app.world()) && frames < MAX_WAIT {
            app.update();
            frames += 1;
        }
        for _ in 0..SETTLE_FRAMES {
            app.update();
        }
        Renderer { app, parts: Vec::new() }
    }

    /// Show `scene` from now on (replacing the last one).
    pub fn set_scene(&mut self, scene: &Scene) {
        let world = self.app.world_mut();
        for e in self.parts.drain(..) {
            world.despawn(e);
        }
        let mut state: SystemState<(Commands, Gfx)> = SystemState::new(world);
        let (mut commands, mut gfx) = state.get_mut(world).expect("level graphics resources");
        self.parts = crate::render::spawn(&mut commands, &mut gfx, scene);
        state.apply(world);
    }

    /// `cam`'s view of the scene last set (`set_scene`, which must be `scene`).
    pub fn render(&mut self, scene: &Scene, cam: &PsxCamera) -> Frame {
        self.render_all(scene, std::slice::from_ref(cam)).pop().unwrap()
    }

    /// Views of the scene last set, in order. Each update draws the next view
    /// while earlier ones are still on their way back from the GPU.
    pub fn render_all(&mut self, scene: &Scene, cams: &[PsxCamera]) -> Vec<Frame> {
        let mut slots = Vec::with_capacity(cams.len());
        for cam in cams {
            let world = self.app.world_mut();
            *world.resource_mut::<View>() = View::of(scene, cam);
            let screen = world.resource::<PsxScreen>().image.clone();
            let flesh = world.resource::<FleshTarget>().image.clone();
            let (a, b) = (read_back(world, screen), read_back(world, flesh));
            self.app.update();
            // Only this update's frame.
            let world = self.app.world_mut();
            for e in [a.1, b.1] {
                if let Ok(mut e) = world.get_entity_mut(e) {
                    e.remove::<Readback>();
                }
            }
            slots.push((a.0, b.0));
        }
        let done = |slots: &[(Slot, Slot)]| slots.iter().all(|(a, b)| a.lock().unwrap().is_some() && b.lock().unwrap().is_some());
        let mut waited = 0;
        while !done(&slots) {
            assert!(waited < MAX_WAIT, "no frame back from the GPU");
            self.app.update();
            waited += 1;
        }
        slots
            .into_iter()
            .map(|(rgb, mask)| {
                let rgba = rgb.lock().unwrap().take().unwrap();
                let flesh = mask.lock().unwrap().take().unwrap();
                Frame { rgb: rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect(), mask: creature_mask(&flesh) }
            })
            .collect()
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Read `image` back as it is at the end of updates while the entity returned
/// keeps its `Readback` (remove it after one); the slot gets the first.
fn read_back(world: &mut World, image: Handle<Image>) -> (Slot, Entity) {
    let slot: Slot = Arc::default();
    let into = slot.clone();
    let e = world
        .spawn(Readback::texture(image))
        .observe(move |ev: On<ReadbackComplete>, mut commands: Commands| {
            into.lock().unwrap().get_or_insert_with(|| ev.data.clone());
            commands.entity(ev.entity).despawn();
        })
        .id();
    (slot, e)
}

/// The flesh image (half floats, twice the screen's size) to one byte per screen
/// pixel: 1 where creatures cover at least two of its four samples.
fn creature_mask(flesh: &[u8]) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let alpha = |x: usize, y: usize| {
        let o = (y * 2 * w + x) * 8 + 6;
        half::f16::from_le_bytes([flesh[o], flesh[o + 1]]).to_f32() > 0.5
    };
    let mut mask = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let n = (0..4).filter(|i| alpha(x * 2 + i % 2, y * 2 + i / 2)).count();
            mask[y * w + x] = (n >= 2) as u8;
        }
    }
    mask
}
