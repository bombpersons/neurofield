//! The 3D view behind the editor's panels: the level drawn by psx-bevy (see
//! `engine::render`) from the same triangles the world bakes, an editor camera, and
//! gizmos for the grid and entities.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use engine::world::level::EntityKind;
use engine::world::scene::{Camera as PsxCamera, Scene, STEP};
use engine::world::tex;
pub use engine::render::{dir_to_world, to_view, to_world, TF};
use psx_bevy::psx::{PsxLook, PsxScreen};
use std::collections::HashSet;
use std::f32::consts::{FRAC_PI_2, TAU};

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ViewMode {
    /// The same renderer at the window's resolution, without wobble, dither or
    /// 15-bit colour: easier to edit in.
    Plain,
    /// The PS1 look as the game shows it: 240 lines, the width following the window.
    Psx,
}

/// The editor's camera: a position and a heading, flown or orbited.
#[derive(Resource)]
pub struct EditorCam {
    pub pos: Vec3,
    /// Radians: yaw about +y (0 looks along -z), pitch up.
    pub yaw: f32,
    pub pitch: f32,
    /// Distance to the point orbiting turns around.
    pub focus: f32,
}

impl EditorCam {
    pub fn rotation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation() * Vec3::NEG_Z
    }

    /// Look at `target` from `dist` away, keeping the heading.
    pub fn frame(&mut self, target: Vec3, dist: f32) {
        self.pos = target - self.forward() * dist;
        self.focus = dist;
    }

    /// The same view as a world camera: GTE angles (4096 a turn; its yaw
    /// faces (-sin, cos) in x/z and its pitch looks down).
    pub fn psx(&self) -> PsxCamera {
        let f = dir_to_world(self.forward());
        let yaw = (-f[0]).atan2(f[2]);
        let p = to_world(self.pos);
        PsxCamera {
            pos: p.map(|c| c.round() as i32),
            yaw: (yaw / TAU * 4096.0).round() as i32,
            pitch: (-self.pitch / TAU * 4096.0).round() as i32,
            roll: 0,
            h: engine::world::walker::H,
            fog_near: (3.0 * TF) as i32,
            fog_far: (12.0 * TF) as i32,
            torch: [0.9, 0.75, 0.55],
            torch_radius: (3.0 * TF) as i32,
        }
    }

    /// Where the player would stand to see this: (world position, walker yaw in radians).
    pub fn player_start(&self) -> ([f32; 3], f32) {
        let f = dir_to_world(self.forward());
        (to_world(self.pos), (-f[0]).atan2(f[2]))
    }
}

/// What the panels tell the view each frame: whether the pointer is over the
/// open 3D area (not a panel), and the slice above which nothing is shown.
#[derive(Resource, Default)]
pub struct ViewInput {
    pub hovered: bool,
    /// Keyboard focus belongs to a text field.
    pub typing: bool,
    pub frame_request: Option<Vec3>,
    /// The wheel belongs to the current tool, not the camera.
    pub wheel_taken: bool,
}

/// The level as built for viewing.
#[derive(Resource)]
pub struct Built {
    /// The level as shown (cut at the slice, if `cut`).
    pub scene: Scene,
    /// The whole level, when `scene` is cut.
    pub full: Option<Scene>,
    pub version: u64,
    pub mode: ViewMode,
    pub slice: i32,
    /// Hide everything more than `CUT_ABOVE` layers above the slice.
    pub cut: bool,
    /// The slice `scene` was cut at.
    pub cut_at: Option<i32>,
    pub show_grid: bool,
    pub show_entities: bool,
    /// Creatures in full colour once sculpted (see `crate::flesh`), else as PS1 meshes.
    pub full_colour: bool,
    /// `Creature::shape_hash` of each of the scene's creatures.
    pub shapes: Vec<u64>,
    /// When the level last changed (`Time::elapsed_secs_f64`).
    pub changed_at: f64,
    /// The level's mesh entities.
    pub parts: Vec<Entity>,
    /// Shapes of the creatures left out of `parts` because they are shown in full colour.
    pub hidden: Vec<u64>,
    /// Outlines from the current tool, drawn over the level.
    pub lines: Vec<(Vec3, Vec3, Color)>,
}

/// With the cut on, how many layers above the slice still show: a tile of wall.
pub const CUT_ABOVE: i32 = 4;

impl Built {
    /// The whole level, cut or not.
    pub fn whole(&self) -> &Scene {
        self.full.as_ref().unwrap_or(&self.scene)
    }
}

/// The level with everything from `CUT_ABOVE` layers above `slice` filled in, and
/// the entities up there left out. From above, the fill's undersides are back faces,
/// so the cut shows as open tops.
fn cut_level(level: &engine::world::level::Level, slice: i32) -> engine::world::level::Level {
    let mut l = level.clone();
    let top = slice + CUT_ABOVE;
    let keys: Vec<_> = l.columns.keys().copied().collect();
    for (x, z) in keys {
        l.close(x, z, top, i32::MAX);
    }
    l.entities.retain(|e| -e.pos[1] < top as f32 * STEP as f32);
    l
}

/// The level texture atlas, for the Textures panel.
#[derive(Resource)]
pub struct Atlas(pub Handle<Image>);

/// The level texture atlas as an image, with nearest sampling.
pub fn atlas_image(textures: &[tex::Texture]) -> Image {
    let (w, h, rgb) = tex::atlas_rgb(textures);
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for c in rgb.chunks(3) {
        rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }
    let mut image = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    image
}

/// Rebuild the scene when the level changes, and its meshes when it or the set of
/// creatures shown in full colour does.
#[allow(clippy::too_many_arguments)]
pub fn rebuild(
    mut commands: Commands,
    doc: Res<crate::Document>,
    time: Res<Time>,
    mut built: ResMut<Built>,
    mut sculpts: ResMut<crate::flesh::Sculpts>,
    mut gfx: engine::render::Gfx,
) {
    let cut_at = built.cut.then_some(built.slice);
    let changed = built.version != doc.0.version || built.version == 0 || built.cut_at != cut_at;
    if changed {
        built.version = doc.0.version.max(1);
        built.cut_at = cut_at;
        match cut_at {
            Some(slice) => {
                built.scene = cut_level(&doc.0.level, slice).to_scene();
                built.full = Some(doc.0.level.to_scene());
            }
            None => {
                built.scene = doc.0.level.to_scene();
                built.full = None;
            }
        }
        built.shapes = built.scene.creatures.iter().map(|c| c.shape_hash()).collect();
        built.changed_at = time.elapsed_secs_f64();
    }
    let ready: Vec<u64> = if built.full_colour {
        built.shapes.iter().copied().filter(|&s| sculpts.is_done(s)).collect()
    } else {
        Vec::new()
    };
    if !changed && ready == built.hidden {
        return;
    }
    let b = &mut *built;
    for e in b.parts.drain(..) {
        commands.entity(e).despawn();
    }
    // The level without the PS1 meshes of creatures drawn in full colour.
    let all = if ready.is_empty() {
        None
    } else {
        let hide: HashSet<u64> = ready.iter().copied().collect();
        let kept = b
            .scene
            .creature_spans
            .iter()
            .zip(&b.shapes)
            .filter(|(_, s)| !hide.contains(s))
            .flat_map(|(span, _)| b.scene.creature_tris[span.clone()].iter().copied())
            .collect();
        Some(std::mem::replace(&mut b.scene.creature_tris, kept))
    };
    b.parts = engine::render::spawn(&mut commands, &mut gfx, &b.scene);
    if let Some(all) = all {
        b.scene.creature_tris = all;
    }
    let shown = if b.full_colour { &b.shapes[..] } else { &[] };
    sculpts.show(&mut commands, &mut gfx, &b.scene, shown);
    b.hidden = ready;
}

/// Fly with the right mouse button held (WASD, Q/E down/up, Shift faster), orbit
/// with the middle button, dolly with the wheel. Only over the open 3D area.
#[allow(clippy::too_many_arguments)]
pub fn camera_control(
    time: Res<Time>,
    key: Res<ButtonInput<KeyCode>>,
    button: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut input: ResMut<ViewInput>,
    mut cam: ResMut<EditorCam>,
    mut flying: Local<bool>,
    mut orbiting: Local<bool>,
) {
    if let Some(target) = input.frame_request.take() {
        cam.frame(target, 4.0);
    }
    // A drag that started over the 3D area keeps going over panels.
    let rmb = button.pressed(MouseButton::Right);
    let mmb = button.pressed(MouseButton::Middle) || (button.pressed(MouseButton::Left) && key.pressed(KeyCode::AltLeft));
    *flying = rmb && (*flying || (input.hovered && button.just_pressed(MouseButton::Right)));
    *orbiting = mmb && (*orbiting || input.hovered);
    let look = motion.delta * 0.004;
    if *flying {
        cam.yaw -= look.x;
        cam.pitch = (cam.pitch - look.y).clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
        if !input.typing {
            let held = |k: KeyCode| key.pressed(k) as i32 as f32;
            let speed = if key.pressed(KeyCode::ShiftLeft) { 12.0 } else { 4.0 } * time.delta_secs();
            let rot = cam.rotation();
            let mv = rot * Vec3::NEG_Z * (held(KeyCode::KeyW) - held(KeyCode::KeyS))
                + rot * Vec3::X * (held(KeyCode::KeyD) - held(KeyCode::KeyA))
                + Vec3::Y * (held(KeyCode::KeyE) - held(KeyCode::KeyQ));
            cam.pos += mv * speed;
        }
    } else if *orbiting {
        let centre = cam.pos + cam.forward() * cam.focus;
        cam.yaw -= look.x;
        cam.pitch = (cam.pitch - look.y).clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
        let f = cam.focus;
        cam.frame(centre, f);
    }
    if input.hovered && !input.wheel_taken && scroll.delta.y != 0.0 {
        let step = scroll.delta.y.signum() * (cam.focus * 0.15).clamp(0.25, 4.0);
        let f = cam.forward();
        cam.pos += f * step;
        cam.focus = (cam.focus - step).max(0.5);
    }
}

/// Put the level camera where the editor camera is, with the lens, fog and
/// torch the game would have there.
pub fn apply_camera(cam: Res<EditorCam>, built: Res<Built>, mut view: ResMut<engine::render::View>) {
    let c = cam.psx();
    view.set_if_neq(engine::render::View {
        transform: Some(Transform::from_translation(cam.pos).with_rotation(cam.rotation())),
        // Editing wants to see the whole level.
        clear: built.mode == ViewMode::Plain,
        ..engine::render::View::of(&built.scene, &c)
    });
}

/// The screen and look for the view mode: PS1 lines, or the window's resolution
/// without the PS1 artefacts. The width always follows the window.
pub fn screen_mode(built: Res<Built>, windows: Query<&Window>, mut screen: ResMut<PsxScreen>, mut look: ResMut<PsxLook>) {
    let Ok(w) = windows.single() else { return };
    let (height, on) = match built.mode {
        ViewMode::Psx => (240, true),
        ViewMode::Plain => ((w.physical_height()).clamp(240, 1440), false),
    };
    if screen.height != height {
        let width = psx_bevy::psx::width_for_aspect(height, w.size());
        screen.set_size(UVec2::new(width, height));
    }
    let want = PsxLook { vertex_snap: on, dither: on, quantize: on };
    if *look != want {
        *look = want;
    }
}

/// The grid at the height slice, the tool's outlines, and markers for entities and lights.
pub fn gizmos(mut g: Gizmos, built: Res<Built>, doc: Res<crate::Document>) {
    let level = &doc.0.level;
    for &(a, b, c) in &built.lines {
        g.line(a, b, c);
    }
    let y = built.slice as f32 * STEP as f32 / TF;
    if built.show_grid {
        let (w, d) = (built.scene.w.max(8) + 2, built.scene.d.max(8) + 2);
        let colour = Color::srgba(0.6, 0.65, 0.7, 0.25);
        // In tiles: the world's +z (south) is the view's -z.
        let at = |x: f32, z: f32| Vec3::new(x, y, -z);
        for x in -2..=w {
            g.line(at(x as f32, -2.0), at(x as f32, d as f32), colour);
        }
        for z in -2..=d {
            g.line(at(-2.0, z as f32), at(w as f32, z as f32), colour);
        }
        g.line(at(0.0, 0.0), at(1.0, 0.0), Color::srgb(0.9, 0.2, 0.2));
        g.line(at(0.0, 0.0), at(0.0, 1.0), Color::srgb(0.2, 0.4, 0.95));
    }
    if !built.show_entities {
        return;
    }
    for e in &level.entities {
        let p = to_view(e.pos);
        let colour = entity_colour(&e.kind);
        let h = 0.3;
        g.line(p, p + Vec3::Y * h, colour);
        g.line(p + Vec3::new(-0.15, h, 0.0), p + Vec3::new(0.15, h, 0.0), colour);
        g.line(p + Vec3::new(0.0, h, -0.15), p + Vec3::new(0.0, h, 0.15), colour);
        // Facing, for those that have one: the walker's (-sin, cos) in the world's x/z.
        let f = Vec3::new(-e.yaw.sin(), 0.0, -e.yaw.cos()) * 0.35;
        g.line(p + Vec3::Y * 0.05, p + Vec3::Y * 0.05 + f, colour);
    }
    // Links, a little off the floor, with a tick at the far end.
    for (a, b, c) in crate::tools::links(level) {
        let (a, b) = (to_view(a) + Vec3::Y * 0.2, to_view(b) + Vec3::Y * 0.2);
        g.line(a, b, c);
        g.line(b, b + Vec3::Y * 0.15, c);
    }
    for l in &level.lights {
        let p = to_view(l.pos);
        let c = Color::srgb(l.colour[0].min(1.0), l.colour[1].min(1.0), l.colour[2].min(1.0));
        for a in [Vec3::X, Vec3::Y, Vec3::Z] {
            g.line(p - a * 0.12, p + a * 0.12, c);
        }
    }
}

pub fn entity_colour(k: &EntityKind) -> Color {
    match k {
        EntityKind::PlayerStart => Color::srgb(0.2, 1.0, 0.3),
        EntityKind::Creature { .. } => Color::srgb(1.0, 0.25, 0.3),
        EntityKind::Door { .. } => Color::srgb(0.85, 0.6, 0.3),
        EntityKind::Key { .. } => Color::srgb(1.0, 0.9, 0.2),
        EntityKind::Chest { .. } => Color::srgb(0.7, 0.45, 0.2),
        EntityKind::Prop { .. } => Color::srgb(0.6, 0.6, 0.7),
        EntityKind::Lever { .. } => Color::srgb(0.4, 0.8, 1.0),
    }
}
