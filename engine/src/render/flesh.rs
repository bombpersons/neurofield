//! The creatures in full colour (see `world::flesh` for what they are made of),
//! drawn over the finished PS1 screen so that they look like they do not belong
//! in it (perspective-correct, smooth-shaded, wet, no dithering or 15-bit colour).
//!
//! A camera that follows the `PsxView` one draws them at twice the screen's
//! resolution into an image of their own, with the level drawn depth-only so it
//! hides them. After psx-bevy's screen pass, an overlay averages each 2x2 block
//! of samples and lays it over the screen by how many of them were covered.
//!
//! Shading works in the world's own axes (y down) and in encoded 0..1 colour;
//! the skins' mip chains are `world::flesh`'s.

use crate::render::{to_view, View, TF};
use bevy::asset::{uuid_handle, RenderAssetUsages};
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Hdr, RenderTarget};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::{MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology, VertexFormat};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, Face, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError, TextureDataOrder,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};
use bevy::shader::{Shader, ShaderImport, ShaderRef};
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};
use crate::world::flesh::Flesh;
use crate::world::scene::Scene;
use psx_bevy::psx::{PsxScreen, PsxView, DISPLAY_LAYER};

/// What the flesh camera sees: the creatures and the level's depth.
pub const FLESH_LAYER: usize = 27;
/// The overlay that lays the creatures over the screen.
const OVERLAY_LAYER: usize = 28;
/// After psx-bevy's screen pass (999), before its window display (moved to 1001).
const OVERLAY_ORDER: isize = 1000;
/// Samples per pixel along each axis.
const SS: u32 = 2;

const FLESH_SHADER: Handle<Shader> = uuid_handle!("3b7f0c52-1d4e-4a8b-9f61-2c5e8d0a7b11");
const OCCLUDER_SHADER: Handle<Shader> = uuid_handle!("3b7f0c52-1d4e-4a8b-9f61-2c5e8d0a7b12");
const OVERLAY_SHADER: Handle<Shader> = uuid_handle!("3b7f0c52-1d4e-4a8b-9f61-2c5e8d0a7b13");
const DREAM_SHADER: Handle<Shader> = uuid_handle!("3b7f0c52-1d4e-4a8b-9f61-2c5e8d0a7b14");
/// The parts' numbers for flesh with nothing that moves (see `FleshMaterial::parts`).
const NO_PARTS: Handle<ShaderBuffer> = uuid_handle!("3b7f0c52-1d4e-4a8b-9f61-2c5e8d0a7b15");

// A flesh vertex. Each skin's values ride with each corner (the triangle's own
// ones repeated): the normal and blend weight, per skin the wrapped and cap
// coordinates, the wrap's tangent and wrap weight, its bitangent and texture
// layer, the four cap axes, and the baked light (as linear SH, already blended
// by weight, which is linear so interpolates exactly).
const ATTR_NW: MeshVertexAttribute = MeshVertexAttribute::new("Flesh_NW", 0x6e66_0001, VertexFormat::Float32x4);
const ATTR_UVCAP: [MeshVertexAttribute; 2] = [
    MeshVertexAttribute::new("Flesh_UvCap0", 0x6e66_0002, VertexFormat::Float32x4),
    MeshVertexAttribute::new("Flesh_UvCap1", 0x6e66_0003, VertexFormat::Float32x4),
];
const ATTR_TW: [MeshVertexAttribute; 2] = [
    MeshVertexAttribute::new("Flesh_TW0", 0x6e66_0004, VertexFormat::Float32x4),
    MeshVertexAttribute::new("Flesh_TW1", 0x6e66_0005, VertexFormat::Float32x4),
];
const ATTR_BS: [MeshVertexAttribute; 2] = [
    MeshVertexAttribute::new("Flesh_BS0", 0x6e66_0006, VertexFormat::Float32x4),
    MeshVertexAttribute::new("Flesh_BS1", 0x6e66_0007, VertexFormat::Float32x4),
];
const ATTR_AXES: [MeshVertexAttribute; 3] = [
    MeshVertexAttribute::new("Flesh_Axes0", 0x6e66_0008, VertexFormat::Float32x4),
    MeshVertexAttribute::new("Flesh_Axes1", 0x6e66_0009, VertexFormat::Float32x4),
    MeshVertexAttribute::new("Flesh_Axes2", 0x6e66_000a, VertexFormat::Float32x4),
];
const ATTR_SH: [MeshVertexAttribute; 3] = [
    MeshVertexAttribute::new("Flesh_Sh0", 0x6e66_000b, VertexFormat::Float32x4),
    MeshVertexAttribute::new("Flesh_Sh1", 0x6e66_000c, VertexFormat::Float32x4),
    MeshVertexAttribute::new("Flesh_Sh2", 0x6e66_000d, VertexFormat::Float32x4),
];

const ATTR_GROW: MeshVertexAttribute = MeshVertexAttribute::new("Flesh_Grow", 0x6e66_000e, VertexFormat::Float32);
/// How a corner of flesh that moves (`world::dreamed::Moving`) is moved: only those have it.
const ATTR_MOVES: MeshVertexAttribute = MeshVertexAttribute::new("Flesh_Moves", 0x6e66_000f, VertexFormat::Float32x4);

#[derive(Clone, Copy, Default, ShaderType, Debug)]
struct FleshUniform {
    /// The eye in world units, and one tile in them.
    eye: Vec4,
    /// Torch colour and radius.
    torch: Vec4,
    /// Fog near and far.
    fog: Vec4,
    fog_colour: Vec4,
}

/// The creatures of one scene: their skins as texture arrays, one layer per skin.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct FleshMaterial {
    #[uniform(0)]
    uniform: FleshUniform,
    /// Albedo (tinted) and gloss.
    #[texture(1, dimension = "2d_array")]
    #[sampler(3)]
    colour: Handle<Image>,
    /// Tangent-space normals, encoded n / 2 + 0.5.
    #[texture(2, dimension = "2d_array")]
    normal: Handle<Image>,
    /// What each part of the flesh that moves is doing now (`world::dreamed::Moving::state`).
    #[storage(4, read_only)]
    parts: Handle<ShaderBuffer>,
}

impl Material for FleshMaterial {
    fn vertex_shader() -> ShaderRef {
        FLESH_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        FLESH_SHADER.into()
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let mut attrs = vec![Mesh::ATTRIBUTE_POSITION.at_shader_location(0), ATTR_NW.at_shader_location(1)];
        let rest = [ATTR_UVCAP[0], ATTR_TW[0], ATTR_BS[0], ATTR_UVCAP[1], ATTR_TW[1], ATTR_BS[1]]
            .into_iter()
            .chain(ATTR_AXES)
            .chain(ATTR_SH);
        attrs.extend(rest.enumerate().map(|(i, a)| a.at_shader_location(2 + i as u32)));
        attrs.push(ATTR_GROW.at_shader_location(14));
        if layout.0.contains(ATTR_MOVES) {
            attrs.push(ATTR_MOVES.at_shader_location(15));
            descriptor.vertex.shader_defs.push("MOVING".into());
        }
        descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        // Triangles face out of the creature (see `mesh`); the rest are culled.
        descriptor.primitive.cull_mode = Some(Face::Back);
        Ok(())
    }
}

/// The level in the flesh camera: depth only, so walls hide creatures. Writes
/// nothing a clear would not.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
pub struct OccluderMaterial {}

impl Material for OccluderMaterial {
    fn fragment_shader() -> ShaderRef {
        OCCLUDER_SHADER.into()
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The level's back faces are culled here too.
        descriptor.primitive.cull_mode = Some(Face::Back);
        Ok(())
    }
}

/// Creatures drawn some other way, laid into the flesh image (`DreamCreatures`).
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
struct DreamMaterial {
    #[texture(0, sample_type = "float", filterable = false)]
    colour: Handle<Image>,
    #[texture(1, sample_type = "float", filterable = false)]
    near: Handle<Image>,
}

impl Material for DreamMaterial {
    fn vertex_shader() -> ShaderRef {
        DREAM_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        DREAM_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.vertex.buffers = vec![layout.0.get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?];
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Creatures drawn some other way than as flesh (the dream net draws them as a picture),
/// to be hidden by the level and the flesh in front of them as the flesh is, and laid over
/// the screen with it. `colour` is [H][W] RGBA32 floats, encoded colour and straight
/// alpha; `near` is [H][W] R32 floats, their depth in tiles. Either may be replaced with
/// another size of the screen's shape (the editor's are 240 lines at any resolution); they
/// are stretched over it. Shown while `on`.
#[derive(Resource)]
pub struct DreamCreatures {
    pub colour: Handle<Image>,
    pub near: Handle<Image>,
    pub on: bool,
}

#[derive(Component)]
struct DreamQuad;

fn float_image(format: TextureFormat, texel: usize) -> Image {
    use crate::world::view::{H, W};
    let mut image = Image::new_fill(
        Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &vec![0; texel],
        format,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    image
}

fn dream_on(
    dream: Res<DreamCreatures>,
    images: Res<Assets<Image>>,
    mut materials: ResMut<Assets<DreamMaterial>>,
    mut size: Local<UVec2>,
    mut quads: Query<&mut Visibility, With<DreamQuad>>,
) {
    let want = if dream.on { Visibility::Visible } else { Visibility::Hidden };
    for mut v in &mut quads {
        if *v != want {
            *v = want;
        }
    }
    // Pictures of a new size are new textures, which the material must be bound to afresh.
    let now = images.get(&dream.colour).map_or(UVec2::ZERO, |i| i.size());
    if *size != now {
        *size = now;
        for _ in materials.iter_mut() {}
    }
}

/// Averages 2x2 blocks of the flesh image over the screen.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
struct OverlayMaterial {
    #[texture(0, sample_type = "float", filterable = false)]
    flesh: Handle<Image>,
}

impl Material2d for OverlayMaterial {
    fn fragment_shader() -> ShaderRef {
        OVERLAY_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

/// Keep the creatures apart from the screen (for the training data): no
/// overlay, so the screen holds the level alone, and no level depth in the flesh
/// camera, so the flesh image holds every creature whole. Set before spawning a
/// scene.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct FleshLayers(pub bool);

#[derive(Component)]
struct FleshCamera;

#[derive(Component)]
struct Overlay;

#[derive(Component)]
struct OverlayCamera;

/// The flesh image: twice the screen's size, the creatures' colour where its
/// alpha is 1 (their mask).
#[derive(Resource)]
pub struct FleshTarget {
    pub image: Handle<Image>,
    size: UVec2,
}

/// Shaders not yet added: Bevy loads the libraries they import at startup, and
/// one compiled before them warns of an unresolved import (as psx-bevy's do).
#[derive(Resource)]
struct Pending(Vec<(Handle<Shader>, Shader)>);

fn add_shaders(mut commands: Commands, mut pending: ResMut<Pending>, mut shaders: ResMut<Assets<Shader>>, mut frames: Local<u32>) {
    *frames += 1;
    let give_up = *frames > 600;
    let (now, later): (Vec<_>, Vec<_>) =
        std::mem::take(&mut pending.0).into_iter().partition(|(_, s)| give_up || imports_loaded(s, &shaders));
    for (handle, shader) in now {
        let _ = shaders.insert(handle.id(), shader);
    }
    pending.0 = later;
    if pending.0.is_empty() {
        commands.remove_resource::<Pending>();
    }
}

/// Whether every module `shader` imports is loaded, and everything those import.
/// WESL lists an import both as its module and as the item in it, and its
/// parents may be bare namespaces: a namespace counts as loaded once some
/// module under it is.
fn imports_loaded(shader: &Shader, shaders: &Assets<Shader>) -> bool {
    let module = |path: &str| {
        shaders.iter().map(|(_, s)| s).find(|s| matches!(&s.import_path, ShaderImport::Custom(p) if p == path))
    };
    let namespace = |path: &str| {
        shaders.iter().any(|(_, s)| {
            matches!(&s.import_path, ShaderImport::Custom(p) if p.strip_prefix(path).is_some_and(|r| r.starts_with("::")))
        })
    };
    let mut stack = vec![shader];
    let mut seen = std::collections::HashSet::new();
    while let Some(shader) = stack.pop() {
        for import in &shader.imports {
            let ShaderImport::Custom(path) = import else { continue };
            let parent = path.rsplit_once("::").map(|(p, _)| p);
            let found = module(path).or_else(|| parent.and_then(|p| module(p)));
            match found {
                Some(lib) => {
                    if seen.insert(lib.path.clone()) {
                        stack.push(lib);
                    }
                }
                None if namespace(path) => {}
                None => return false,
            }
        }
    }
    true
}

/// Whether the flesh pass's shaders are in (once Bevy's libraries they import have loaded).
pub(crate) fn shaders_ready(world: &World) -> bool {
    !world.contains_resource::<Pending>()
}

pub(crate) fn plugin(app: &mut App) {
    let wesl = |source: &'static str, name: &str| Shader::from_wesl(source, format!("embedded://engine/render/{name}"));
    app.insert_resource(Pending(vec![
        (FLESH_SHADER, wesl(include_str!("flesh.wesl"), "flesh.wesl")),
        (OCCLUDER_SHADER, wesl(include_str!("occluder.wesl"), "occluder.wesl")),
        (OVERLAY_SHADER, wesl(include_str!("flesh_overlay.wesl"), "flesh_overlay.wesl")),
        (DREAM_SHADER, wesl(include_str!("dream_creatures.wesl"), "dream_creatures.wesl")),
    ]))
    .add_systems(PreUpdate, add_shaders.run_if(resource_exists::<Pending>));
    app.add_plugins((
        MaterialPlugin::<FleshMaterial>::default(),
        MaterialPlugin::<OccluderMaterial>::default(),
        MaterialPlugin::<DreamMaterial>::default(),
        Material2dPlugin::<OverlayMaterial>::default(),
    ))
    .add_systems(Startup, (setup, no_parts))
    .init_resource::<FleshLayers>()
    .add_systems(Update, (display_after_overlay, overlay_on, dream_on))
    .add_systems(
        PostUpdate,
        (follow_screen, follow_camera.after(crate::render::apply_view)).before(bevy::camera::CameraUpdateSystems),
    )
    .add_systems(PostUpdate, update_uniforms);
}

fn no_parts(mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let none = ShaderBuffer::new(vec![[0.0f32; 4]], RenderAssetUsages::default());
    buffers.insert(&NO_PARTS, none).expect("a handle of our own");
}

fn flesh_image(size: UVec2) -> Image {
    // Half floats: the shading's values pass through unrounded to the overlay.
    let mut image = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image
}

fn setup(
    mut commands: Commands,
    screen: Res<PsxScreen>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut overlays: ResMut<Assets<OverlayMaterial>>,
    mut dreams: ResMut<Assets<DreamMaterial>>,
) {
    let size = screen.size() * SS;
    let image = images.add(flesh_image(size));
    commands.spawn((
        Camera3d::default(),
        Camera { order: -10, clear_color: ClearColorConfig::Custom(Color::NONE), ..default() },
        RenderTarget::Image(image.clone().into()),
        Hdr,
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        Projection::default(),
        RenderLayers::layer(FLESH_LAYER),
        FleshCamera,
    ));
    // Stacked on the screen pass's camera (same target and settings), so it
    // draws over the finished screen.
    commands.spawn((
        Camera2d,
        Camera { order: OVERLAY_ORDER, clear_color: ClearColorConfig::None, ..default() },
        RenderTarget::Image(screen.image.clone().into()),
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        RenderLayers::layer(OVERLAY_LAYER),
        OverlayCamera,
    ));
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(screen.width as f32, screen.height as f32))),
        MeshMaterial2d(overlays.add(OverlayMaterial { flesh: image.clone() })),
        RenderLayers::layer(OVERLAY_LAYER),
        Overlay,
    ));
    commands.insert_resource(FleshTarget { image, size });
    let colour = images.add(float_image(TextureFormat::Rgba32Float, 16));
    let near = images.add(float_image(TextureFormat::R32Float, 4));
    let quad = Mesh::from(Rectangle::new(2.0, 2.0));
    commands.spawn((
        Mesh3d(meshes.add(quad)),
        MeshMaterial3d(dreams.add(DreamMaterial { colour: colour.clone(), near: near.clone() })),
        RenderLayers::layer(FLESH_LAYER),
        bevy::camera::visibility::NoFrustumCulling,
        Visibility::Hidden,
        DreamQuad,
    ));
    commands.insert_resource(DreamCreatures { colour, near, on: false });
}

/// psx-bevy shows the screen in the window at order 1000; the overlay must be
/// drawn into the screen first.
fn display_after_overlay(mut cams: Query<(&mut Camera, &RenderLayers), With<Camera2d>>) {
    for (mut cam, layers) in &mut cams {
        if layers.intersects(&RenderLayers::layer(DISPLAY_LAYER)) && cam.order <= OVERLAY_ORDER {
            cam.order = OVERLAY_ORDER + 1;
        }
    }
}

fn overlay_on(layers: Res<FleshLayers>, mut cams: Query<&mut Camera, With<OverlayCamera>>) {
    for mut cam in &mut cams {
        if cam.is_active == layers.0 {
            cam.is_active = !layers.0;
        }
    }
}

/// New flesh image and overlay when the screen's size or image changes.
#[allow(clippy::too_many_arguments)]
fn follow_screen(
    screen: Res<PsxScreen>,
    mut target: ResMut<FleshTarget>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut overlays: ResMut<Assets<OverlayMaterial>>,
    mut flesh_cam: Query<&mut RenderTarget, With<FleshCamera>>,
    mut overlay_cam: Query<&mut RenderTarget, (With<OverlayCamera>, Without<FleshCamera>)>,
    quads: Query<(&Mesh2d, &MeshMaterial2d<OverlayMaterial>), With<Overlay>>,
) {
    if !screen.is_changed() {
        return;
    }
    // The overlay's camera draws into the screen image, which psx-bevy replaces on resize.
    for mut t in &mut overlay_cam {
        let stale = match &*t {
            RenderTarget::Image(img) => img.handle != screen.image,
            _ => true,
        };
        if stale {
            *t = RenderTarget::Image(screen.image.clone().into());
        }
    }
    let size = screen.size() * SS;
    if size == target.size {
        return;
    }
    let image = images.add(flesh_image(size));
    for mut t in &mut flesh_cam {
        *t = RenderTarget::Image(image.clone().into());
    }
    for (quad, material) in &quads {
        if let Some(mut mesh) = meshes.get_mut(&quad.0) {
            *mesh = Rectangle::new(screen.width as f32, screen.height as f32).into();
        }
        if let Some(mut m) = overlays.get_mut(&material.0) {
            m.flesh = image.clone();
        }
    }
    *target = FleshTarget { image, size };
}

/// The flesh camera sees what the `PsxView` 3D camera sees.
fn follow_camera(
    view: Query<(&Transform, &Projection, &Camera), (With<PsxView>, With<Camera3d>, Without<FleshCamera>)>,
    mut flesh: Query<(&mut Transform, &mut Projection, &mut Camera), With<FleshCamera>>,
) {
    let Ok((t, p, c)) = view.single() else { return };
    for (mut ft, mut fp, mut fc) in &mut flesh {
        if *ft != *t {
            *ft = *t;
        }
        let same = match (p, &*fp) {
            (Projection::Perspective(a), Projection::Perspective(b)) => a.fov == b.fov && a.near == b.near && a.far == b.far,
            _ => false,
        };
        if !same {
            *fp = p.clone();
        }
        if fc.is_active != c.is_active {
            fc.is_active = c.is_active;
        }
    }
}

fn uniform(view: &View) -> FleshUniform {
    let Some(cam) = view.cam else { return FleshUniform::default() };
    // No fog: as far off as the level's.
    let (near, far) = if view.clear { (1e4 * TF, 2e4 * TF) } else { (cam.fog_near as f32, cam.fog_far as f32) };
    FleshUniform {
        eye: Vec4::new(cam.pos[0] as f32, cam.pos[1] as f32, cam.pos[2] as f32, TF),
        torch: Vec4::new(cam.torch[0], cam.torch[1], cam.torch[2], cam.torch_radius.max(1) as f32),
        fog: Vec4::new(near, far, 0.0, 0.0),
        fog_colour: Vec4::new(view.fog_colour[0], view.fog_colour[1], view.fog_colour[2], 0.0),
    }
}

fn update_uniforms(view: Res<View>, mut materials: ResMut<Assets<FleshMaterial>>) {
    if !view.is_changed() {
        return;
    }
    let u = uniform(&view);
    let ids: Vec<_> = materials.ids().collect();
    for id in ids {
        if let Some(mut m) = materials.get_mut(id) {
            m.uniform = u;
        }
    }
}

/// The level's parts again as depth for the flesh camera, and the scene's
/// creatures as one mesh with its skins' material (none without creatures). Grown
/// creatures that move (`Scene::creatures_move`) are left out, to be drawn as they stand
/// some other way (`spawn_creatures`); the material is still made, for other flesh to use.
pub(crate) fn spawn(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<FleshMaterial>,
    occluder: Handle<OccluderMaterial>,
    level_meshes: &[Handle<Mesh>],
    scene: &Scene,
    view: &View,
) -> Vec<Entity> {
    let mut out: Vec<Entity> = level_meshes
        .iter()
        .map(|m| commands.spawn((Mesh3d(m.clone()), MeshMaterial3d(occluder.clone()), RenderLayers::layer(FLESH_LAYER))).id())
        .collect();
    if !scene.creatures_move {
        out.extend(spawn_flesh(commands, meshes, images, materials, &scene.flesh, view));
        return out;
    }
    let moving = |i: usize| scene.grown.iter().any(|g| g.flesh.contains(&i));
    let mut still: Vec<_> = scene.flesh.tris.iter().enumerate().filter(|&(i, _)| !moving(i)).map(|(_, t)| *t).collect();
    // A triangle with no area keeps the skins' material for the creatures to borrow.
    if still.is_empty() {
        if let Some(t) = scene.flesh.tris.first() {
            still.push(crate::world::flesh::Tri { v: [t.v[0]; 3], ..*t });
        }
    }
    out.extend(spawn_flesh_tris(commands, meshes, images, materials, &scene.flesh, &still, view));
    out
}

/// `fl` as one mesh with its skins' material, or nothing when it is empty.
pub(crate) fn spawn_flesh(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<FleshMaterial>,
    fl: &Flesh,
    view: &View,
) -> Option<Entity> {
    spawn_flesh_tris(commands, meshes, images, materials, fl, &fl.tris, view)
}

/// `tris` with `fl`'s skins and lights, as `spawn_flesh`.
fn spawn_flesh_tris(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<FleshMaterial>,
    fl: &Flesh,
    tris: &[crate::world::flesh::Tri],
    view: &View,
) -> Option<Entity> {
    if tris.is_empty() || fl.skins.is_empty() {
        return None;
    }
    let (colour, normal) = skin_arrays(&fl.skins, fl.matte);
    let material = FleshMaterial { uniform: uniform(view), colour: images.add(colour), normal: images.add(normal), parts: NO_PARTS };
    let mesh = meshes.add(mesh(fl, tris, None));
    Some(commands.spawn((Mesh3d(mesh), MeshMaterial3d(materials.add(material)), RenderLayers::layer(FLESH_LAYER))).id())
}

/// Flesh remade as it changes, which borrows its skins from the scene's own (`spawn_flesh`).
#[derive(Component)]
pub struct MovingFlesh;

/// Triangles that change every frame, with `fl`'s lights and the skins of `material`.
pub(crate) fn spawn_moving(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &MeshMaterial3d<FleshMaterial>,
    fl: &Flesh,
    tris: &[crate::world::flesh::Tri],
) -> Entity {
    commands.spawn((Mesh3d(meshes.add(mesh(fl, tris, None))), material.clone(), RenderLayers::layer(FLESH_LAYER), MovingFlesh)).id()
}

/// The flesh that moves, built once (see `world::dreamed::Moving`), with the skins of
/// `material` and `fl`'s lights; and the numbers that move it, for `set_parts`.
pub(crate) fn spawn_parts(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<FleshMaterial>,
    buffers: &mut Assets<ShaderBuffer>,
    material: &MeshMaterial3d<FleshMaterial>,
    fl: &Flesh,
    moving: &crate::world::dreamed::Moving,
) -> Option<(Entity, Handle<ShaderBuffer>)> {
    let mut own = materials.get(&material.0)?.clone();
    own.parts = buffers.add(ShaderBuffer::new(moving.state.clone(), RenderAssetUsages::default()));
    let parts = own.parts.clone();
    let mesh = meshes.add(mesh(fl, &moving.tris, Some(&moving.how)));
    let e = commands.spawn((Mesh3d(mesh), MeshMaterial3d(materials.add(own)), RenderLayers::layer(FLESH_LAYER), MovingFlesh)).id();
    Some((e, parts))
}

/// What the parts of `spawn_parts`'s flesh are doing now.
pub(crate) fn set_parts(buffers: &mut Assets<ShaderBuffer>, parts: &Handle<ShaderBuffer>, state: &[[f32; 4]]) {
    if let Some(slice) = buffers.get_mut(parts).and_then(|b| b.into_inner().cast_slice_mut::<[f32; 4]>()) {
        slice.copy_from_slice(state);
    }
}

/// `tris` as a mesh, with how each corner moves (`spawn_parts`) when they do.
fn mesh(fl: &Flesh, tris: &[crate::world::flesh::Tri], how: Option<&[[[f32; 4]; 3]]>) -> Mesh {
    let n = tris.len() * 3;
    let mut pos = Vec::with_capacity(n);
    let mut nw = Vec::with_capacity(n);
    let mut uvcap = [Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut tw = [Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut bs = [Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut axes = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut sh = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut grow = Vec::with_capacity(n);
    let mut moves = Vec::with_capacity(if how.is_some() { n } else { 0 });
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    for (i, t) in tris.iter().enumerate() {
        // Wound to face out of the creature, by its vertex normals.
        let p = t.v.map(|v| v.p);
        let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
        let face = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let out = t.v.iter().fold([0.0; 3], |a, v| [a[0] + v.n[0], a[1] + v.n[1], a[2] + v.n[2]]);
        let order = if face[0] * out[0] + face[1] * out[1] + face[2] * out[2] < 0.0 { [0, 2, 1] } else { [0, 1, 2] };
        let lights = t.light.map(|l| &fl.lights[l as usize]);
        let ax = t.cap_axes;
        let a = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]].concat();
        for k in order {
            let v = &t.v[k];
            pos.push(to_view(v.p).to_array());
            nw.push([v.n[0], v.n[1], v.n[2], v.w]);
            grow.push(v.grow);
            if let Some(how) = how {
                moves.push(how[i][k]);
            }
            for s in 0..2 {
                uvcap[s].push([v.uv[s][0], v.uv[s][1], v.cap[s][0], v.cap[s][1]]);
                tw[s].push([v.t[s][0], v.t[s][1], v.t[s][2], v.wrap[s]]);
                bs[s].push([v.b[s][0], v.b[s][1], v.b[s][2], t.skin[s] as f32]);
            }
            for (i, list) in axes.iter_mut().enumerate() {
                list.push([a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]);
            }
            let w = v.w.clamp(0.0, 1.0);
            let mut blended = [0.0f32; 12];
            for r in 0..4 {
                for c in 0..3 {
                    blended[r * 3 + c] = match &t.corners {
                        Some(light) => light[k][r][c],
                        None => w * lights[0][r][c] + (1.0 - w) * lights[1][r][c],
                    };
                }
            }
            for (i, list) in sh.iter_mut().enumerate() {
                list.push([blended[i * 4], blended[i * 4 + 1], blended[i * 4 + 2], blended[i * 4 + 3]]);
            }
        }
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(ATTR_NW, nw);
    mesh.insert_attribute(ATTR_GROW, grow);
    if how.is_some() {
        mesh.insert_attribute(ATTR_MOVES, moves);
    }
    for s in 0..2 {
        mesh.insert_attribute(ATTR_UVCAP[s], std::mem::take(&mut uvcap[s]));
        mesh.insert_attribute(ATTR_TW[s], std::mem::take(&mut tw[s]));
        mesh.insert_attribute(ATTR_BS[s], std::mem::take(&mut bs[s]));
    }
    for i in 0..3 {
        mesh.insert_attribute(ATTR_AXES[i], std::mem::take(&mut axes[i]));
        mesh.insert_attribute(ATTR_SH[i], std::mem::take(&mut sh[i]));
    }
    mesh
}

/// The skins as two texture arrays with their mip chains: albedo (tinted) with
/// gloss in alpha, and normals. Smaller skins are scaled up to the largest by
/// their own bilinear sampling, below which their mips carry on. `matte` takes
/// away the gloss and most of the fine relief that catches it.
fn skin_arrays(skins: &[crate::world::flesh::Skin], matte: f32) -> (Image, Image) {
    let size = skins.iter().map(|s| s.mips()[0].0).max().unwrap_or(1);
    let levels = size.trailing_zeros() + 1;
    let (mut colour, mut normal) = (Vec::new(), Vec::new());
    for skin in skins {
        let tint = skin.tint();
        for level in 0..levels as usize {
            let s = size >> level;
            let texels: Vec<crate::world::flesh::Texel> = match skin.mips().iter().find(|(m, _)| *m == s) {
                Some((_, t)) => t.clone(),
                None => (0..s * s)
                    .map(|i| skin.bilinear(0, ((i % s) as f32 + 0.5) / s as f32, ((i / s) as f32 + 0.5) / s as f32))
                    .collect(),
            };
            for t in texels {
                for c in [t.c[0] * tint[0], t.c[1] * tint[1], t.c[2] * tint[2], t.wet * (1.0 - matte)] {
                    colour.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
                }
                let flat = 1.0 - 0.7 * matte;
                let n = Vec3::new(t.n[0] * flat, t.n[1] * flat, t.n[2]).normalize();
                normal.extend(n.to_array().map(|c| ((c * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8));
                normal.push(255);
            }
        }
    }
    let array = |data: Vec<u8>, format: TextureFormat| {
        let mut image = Image::new_uninit(
            Extent3d { width: size as u32, height: size as u32, depth_or_array_layers: skins.len() as u32 },
            TextureDimension::D2,
            format,
            RenderAssetUsages::default(),
        );
        image.data = Some(data);
        image.texture_descriptor.mip_level_count = levels;
        // An array even with one skin.
        image.texture_view_descriptor =
            Some(TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..default() });
        image.data_order = TextureDataOrder::LayerMajor;
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            ..default()
        });
        image
    };
    (array(colour, TextureFormat::Rgba16Float), array(normal, TextureFormat::Rgba8Unorm))
}
