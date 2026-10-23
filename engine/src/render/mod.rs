//! The world's `Scene` drawn by psx-bevy with the PS1 look: for the game, the
//! editor and (headless) the training data.
//!
//! The level's triangles keep their baked vertex light (128 = neutral, up to
//! 2x brighter), its 64x64 16-colour textures and its strangeness steps; the
//! torch carried with the camera is added to the vertex colours each frame on the
//! CPU, as PS1 games did, and fog is the camera's `DistanceFog`. Creatures sculpted
//! for the full-colour pass are drawn over the PS1 screen
//! (see `flesh`); those meshed for the PS1 pipeline are drawn with the level.
//!
//! Bevy's world is the engine's divided by a tile and turned half over about x (y
//! and z negated), so one unit is one tile, y points up, and nothing is mirrored.

pub mod flesh;
pub mod headless;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::system::SystemParam;
use bevy::image::ImageSampler;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use crate::world::scene::{Camera as PsxCamera, Scene, Tri, T};
use crate::world::tex;
use psx_bevy::psx::{PsxMaterial, PsxView};
use std::collections::HashMap;

pub use flesh::FleshTarget;

pub const TF: f32 = T as f32;
/// Level triangles are split until no edge is longer than this (world units):
/// textures are affine, as on the PS1, and big triangles near the eye warp. The
/// PS1 rasteriser split near ones (down to an eighth of a tile); this is static.
const MAX_EDGE: f32 = TF / 4.0;

/// Datagen world point to the view's.
pub fn to_view(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0] / TF, -p[1] / TF, -p[2] / TF)
}

/// The view's point to the world's.
pub fn to_world(v: Vec3) -> [f32; 3] {
    [v.x * TF, -v.y * TF, -v.z * TF]
}

/// The view's direction in the world's axes (unscaled).
pub fn dir_to_world(v: Vec3) -> [f32; 3] {
    [v.x, -v.y, -v.z]
}

/// A Bevy camera transform for a world camera: its position and the GTE's
/// world-to-view rotation (`view::view_matrix`, with roll), as `views.npy` records it.
pub fn camera_transform(cam: &PsxCamera) -> Transform {
    let m = crate::world::view::view_matrix(cam).m.map(|r| r.map(|e| e as f32 / 4096.0));
    // Rows 1 and 2 of the view rotation are the screen's down and forward in
    // the world; Bevy's camera looks down -z with y up, and its world is the
    // engine's with y and z negated.
    let world = |r: [f32; 3]| Vec3::new(r[0], -r[1], -r[2]);
    let (down, forward) = (world(m[1]), world(m[2]));
    Transform::from_translation(to_view(cam.pos.map(|c| c as f32))).looking_to(forward, -down)
}

/// The GTE lens as a vertical field of view: focal length `h` over a
/// 240-pixel-high screen.
pub fn fov(h: u16) -> f32 {
    2.0 * (120.0 / h as f32).atan()
}

/// The camera's fog: linear in distance from `fog_near` to `fog_far`, toward
/// `colour` (the scene's where the camera is; black where the level is grounded).
pub fn fog(cam: &PsxCamera, colour: [f32; 3]) -> DistanceFog {
    DistanceFog {
        color: Color::srgb(colour[0], colour[1], colour[2]),
        falloff: FogFalloff::Linear { start: cam.fog_near as f32 / TF, end: cam.fog_far as f32 / TF },
        ..default()
    }
}

/// What the level camera shows, for the game, the editor and the training data
/// alike: set it whenever the camera moves. Places the camera, its lens and fog,
/// the torch and the creatures' light.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct View {
    pub cam: Option<PsxCamera>,
    /// The fog's colour where the camera is.
    pub fog_colour: [f32; 3],
    /// Where to put the camera instead of `cam`'s place, for a free camera (the
    /// editor's); `cam` still sets the lens, light and fog.
    pub transform: Option<Transform>,
    /// No fog, to see a whole level.
    pub clear: bool,
    /// Don't draw (when nothing shows the level).
    pub hidden: bool,
}

impl View {
    pub fn of(scene: &Scene, cam: &PsxCamera) -> View {
        View { cam: Some(*cam), fog_colour: scene.fog_colour(cam.pos), ..default() }
    }
}

/// The camera that draws the level into psx-bevy's screen (spawned by `LevelGfxPlugin`).
#[derive(Component)]
pub struct LevelCamera;

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        PsxView,
        Projection::Perspective(PerspectiveProjection { fov: fov(crate::world::walker::H), near: 0.02, ..default() }),
        DistanceFog::default(),
        LevelCamera,
    ));
}

/// Follow `View`: camera, lens, fog and torch.
fn apply_view(
    view: Res<View>,
    mut torch: ResMut<Torch>,
    mut cams: Query<(&mut Camera, &mut Transform, &mut Projection, &mut DistanceFog), With<LevelCamera>>,
) {
    if !view.is_changed() {
        return;
    }
    let Some(cam) = view.cam else { return };
    let t = Torch::of(&cam);
    if *torch != t {
        *torch = t;
    }
    for (mut camera, mut transform, mut projection, mut fog) in &mut cams {
        camera.is_active = !view.hidden;
        *transform = view.transform.unwrap_or_else(|| camera_transform(&cam));
        if let Projection::Perspective(p) = &mut *projection {
            p.fov = fov(cam.h);
        }
        *fog = if view.clear {
            DistanceFog { falloff: FogFalloff::Linear { start: 1e4, end: 2e4 }, ..default() }
        } else {
            self::fog(&cam, view.fog_colour)
        };
    }
}

/// The light carried with the camera (world units), applied to every
/// `LevelPart`'s vertex colours when it changes.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Default)]
pub struct Torch {
    pub pos: [f32; 3],
    pub colour: [f32; 3],
    pub radius: f32,
}

impl Torch {
    pub fn of(cam: &PsxCamera) -> Torch {
        Torch { pos: cam.pos.map(|c| c as f32), colour: cam.torch, radius: cam.torch_radius as f32 }
    }

    /// Datagen's torch: albedo x colour x falloff x Lambert, in vertex-colour units.
    fn light(&self, p: [f32; 3], n: [f32; 3], alb: [f32; 3]) -> [f32; 3] {
        if self.colour == [0.0; 3] {
            return [0.0; 3];
        }
        let to = [0, 1, 2].map(|c| self.pos[c] - p[c]);
        let dist = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]).sqrt().max(1.0);
        let fall = (1.0 - dist / self.radius.max(1.0)).max(0.0).powi(2);
        let lambert = ((n[0] * to[0] + n[1] * to[1] + n[2] * to[2]) / dist).max(0.0);
        [0, 1, 2].map(|c| alb[c] * self.colour[c] * fall * lambert)
    }
}

/// One mesh of a shown scene: its triangles' world vertices, kept to relight.
#[derive(Component)]
pub struct LevelPart {
    verts: Vec<PartVert>,
}

#[derive(Clone)]
struct PartVert {
    p: [f32; 3],
    n: [f32; 3],
    col: [f32; 3],
    alb: [f32; 3],
}

/// Texture images and materials, made once per texture id.
#[derive(Resource)]
pub struct LevelTextures {
    base: Vec<tex::Texture>,
    materials: HashMap<u8, Handle<PsxMaterial>>,
}

impl LevelTextures {
    /// For the level textures as loaded now (call after `tex::load_level` / `load_strange`).
    pub fn new() -> LevelTextures {
        LevelTextures { base: tex::make_all(), materials: HashMap::new() }
    }

    fn material(&mut self, id: u8, images: &mut Assets<Image>, materials: &mut Assets<PsxMaterial>) -> Option<Handle<PsxMaterial>> {
        if let Some(m) = self.materials.get(&id) {
            return Some(m.clone());
        }
        let t = tex::level_texture(&self.base, id)?;
        let m = materials.add(material(images.add(image(t))));
        self.materials.insert(id, m.clone());
        Some(m)
    }
}

impl Default for LevelTextures {
    fn default() -> Self {
        Self::new()
    }
}

fn image(t: &tex::Texture) -> Image {
    let size = tex::SIZE as u32;
    let mut image = Image::new(
        Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        TextureDimension::D2,
        t.rgba(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    image
}

/// Baked light is in the vertex colours, so the material is unlit.
fn material(image: Handle<Image>) -> PsxMaterial {
    PsxMaterial { texture: Some(image), lit: false, fog: true, ..default() }
}

/// What `spawn` makes a scene's meshes with.
#[derive(SystemParam)]
pub struct Gfx<'w> {
    meshes: ResMut<'w, Assets<Mesh>>,
    images: ResMut<'w, Assets<Image>>,
    materials: ResMut<'w, Assets<PsxMaterial>>,
    flesh: ResMut<'w, Assets<flesh::FleshMaterial>>,
    occluders: ResMut<'w, Assets<flesh::OccluderMaterial>>,
    textures: ResMut<'w, LevelTextures>,
    torch: Res<'w, Torch>,
    view: Res<'w, View>,
}

/// Spawn `scene` as meshes: the level and any PS1-pipeline creatures one mesh
/// per texture, each texture chosen at the strangeness where the triangle is,
/// and full-colour creatures for the flesh pass. Returns the entities, to
/// despawn when the scene goes.
pub fn spawn(commands: &mut Commands, gfx: &mut Gfx, scene: &Scene) -> Vec<Entity> {
    let Gfx { meshes, images, materials, flesh, occluders, textures, torch, view } = gfx;
    let (mut out, level) = spawn_psx(commands, meshes, images, materials, textures, scene, torch);
    let occluder = occluders.add(flesh::OccluderMaterial::default());
    out.extend(flesh::spawn(commands, meshes, images, flesh, occluder, &level, scene, view));
    out
}

/// The PS1 pipeline's meshes, and the level's own (for the flesh pass's depth).
fn spawn_psx(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<PsxMaterial>,
    textures: &mut LevelTextures,
    scene: &Scene,
    torch: &Torch,
) -> (Vec<Entity>, Vec<Handle<Mesh>>) {
    let mut groups: HashMap<u8, Vec<&Tri>> = HashMap::new();
    for t in &scene.tris {
        let centre = [0, 1, 2].map(|c| t.v.iter().map(|v| v.p[c] as f32).sum::<f32>() / 3.0);
        let (id, _) = tex::strange_variant(t.tex, scene.strangeness(centre));
        groups.entry(id).or_default().push(t);
    }
    let (mut out, mut level) = (Vec::new(), Vec::new());
    let mut ids: Vec<_> = groups.keys().copied().collect();
    ids.sort();
    for id in ids {
        let Some(m) = textures.material(id, images, materials) else { continue };
        let (e, mesh) = spawn_part(commands, meshes, &groups[&id], m, torch);
        out.push(e);
        level.push(mesh);
    }
    // Creatures drawn through the PS1 pipeline use the scene's own skins.
    let mut skins: HashMap<u8, Vec<&Tri>> = HashMap::new();
    for t in &scene.creature_tris {
        skins.entry(t.tex).or_default().push(t);
    }
    for (id, tris) in skins {
        let Some(skin) = scene.skins.get(id.wrapping_sub(tex::SKIN_BASE) as usize) else { continue };
        let m = materials.add(material(images.add(image(skin))));
        out.push(spawn_part(commands, meshes, &tris, m, torch).0);
    }
    (out, level)
}

fn spawn_part(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    tris: &[&Tri],
    m: Handle<PsxMaterial>,
    torch: &Torch,
) -> (Entity, Handle<Mesh>) {
    let (mut pos, mut uv, mut verts) = (Vec::new(), Vec::new(), Vec::new());
    for t in tris {
        let p = t.v.map(|v| v.p.map(|c| c as f32));
        let e1 = [0, 1, 2].map(|c| p[1][c] - p[0][c]);
        let e2 = [0, 1, 2].map(|c| p[2][c] - p[0][c]);
        let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let len = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt().max(1e-6);
        let normal = cr.map(|c| c / len);
        let (u0, v0) = tex::uv_origin(t.tex);
        let corner = |k: usize| {
            let v = &t.v[k];
            // Texel 63 is the texture's last, so 0..63 spans it all.
            let uv = [(v.uv[0] - u0) as f32 / 63.0, (v.uv[1] - v0) as f32 / 63.0];
            (PartVert { p: p[k], n: normal, col: v.col.map(|c| c as f32), alb: v.alb.map(|c| c as f32) }, uv)
        };
        split([corner(0), corner(1), corner(2)], &mut |c| {
            for (v, tc) in c {
                pos.push(to_view(v.p).to_array());
                uv.push(tc);
                verts.push(v);
            }
        });
    }
    let n = verts.len();
    let part = LevelPart { verts };
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, part.colours(torch));
    mesh.insert_indices(Indices::U32((0..n as u32).collect()));
    let mesh = meshes.add(mesh);
    (commands.spawn((Mesh3d(mesh.clone()), MeshMaterial3d(m), part)).id(), mesh)
}

type Corner = (PartVert, [f32; 2]);

/// Split a triangle into four at its edges' midpoints until no edge is longer
/// than `MAX_EDGE`, handing on each piece.
fn split(c: [Corner; 3], out: &mut impl FnMut([Corner; 3])) {
    let d = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let longest = (0..3).map(|i| d(c[i].0.p, c[(i + 1) % 3].0.p)).fold(0.0, f32::max);
    if longest <= MAX_EDGE {
        out(c);
        return;
    }
    let mid = |a: &Corner, b: &Corner| -> Corner {
        let h = |x: [f32; 3], y: [f32; 3]| [0, 1, 2].map(|k| (x[k] + y[k]) / 2.0);
        (
            PartVert { p: h(a.0.p, b.0.p), n: a.0.n, col: h(a.0.col, b.0.col), alb: h(a.0.alb, b.0.alb) },
            [(a.1[0] + b.1[0]) / 2.0, (a.1[1] + b.1[1]) / 2.0],
        )
    };
    let [a, b, cc] = c;
    let (ab, bc, ca) = (mid(&a, &b), mid(&b, &cc), mid(&cc, &a));
    split([a, ab.clone(), ca.clone()], out);
    split([ab.clone(), b, bc.clone()], out);
    split([ca.clone(), bc.clone(), cc], out);
    split([ab, bc, ca], out);
}

impl LevelPart {
    /// Vertex colours with the torch added. The material reads them as linear
    /// and encodes them, so pass the linear value whose encoding is col / 128.
    fn colours(&self, torch: &Torch) -> Vec<[f32; 4]> {
        self.verts
            .iter()
            .map(|v| {
                let t = torch.light(v.p, v.n, v.alb);
                let c = [0, 1, 2].map(|k| srgb_to_linear(((v.col[k] + t[k]) / 128.0).min(255.0 / 128.0)));
                [c[0], c[1], c[2], 1.0]
            })
            .collect()
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Relights every part when the torch moves or changes.
pub fn apply_torch(torch: Res<Torch>, parts: Query<(Ref<LevelPart>, &Mesh3d)>, mut meshes: ResMut<Assets<Mesh>>) {
    for (part, mesh) in &parts {
        if !torch.is_changed() && !part.is_added() {
            continue;
        }
        if let Some(mut m) = meshes.get_mut(&mesh.0) {
            m.insert_attribute(Mesh::ATTRIBUTE_COLOR, part.colours(&torch));
        }
    }
}

/// The level camera and its `View`, the `Torch` and `LevelTextures` resources,
/// the relighting system and the flesh pass. Needs psx-bevy's `PsxPlugin`.
pub struct LevelGfxPlugin;

impl Plugin for LevelGfxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Torch>()
            .init_resource::<View>()
            .init_resource::<LevelTextures>()
            .add_systems(Startup, spawn_camera)
            .add_systems(PostUpdate, (apply_view, apply_torch).chain().before(bevy::camera::CameraUpdateSystems));
        flesh::plugin(app);
    }
}
