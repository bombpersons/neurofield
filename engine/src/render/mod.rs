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
pub mod sky;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::system::SystemParam;
use bevy::image::ImageSampler;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use crate::world::scene::{Camera as PsxCamera, Scene, Tri, T};
use crate::world::tex;
use psx_bevy::psx::{PsxMaterial, PsxView};
use std::collections::HashMap;

pub use flesh::{DreamCreatures, FleshLayers, FleshTarget};

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
    // Hidden (only the network's picture showing), the level isn't relit as the torch moves;
    // it is when it shows again.
    let t = Torch::of(&cam);
    if *torch != t && !view.hidden {
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
    /// The camera's torch. An unlit one (by day) is the same wherever it is, so walking
    /// about doesn't relight the level for nothing.
    pub fn of(cam: &PsxCamera) -> Torch {
        if cam.torch == [0.0; 3] || cam.torch_radius <= 0 {
            return Torch::default();
        }
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
    /// The vertices the torch lit last, which go back to their own colours when it moves on.
    lit: Vec<u32>,
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
    /// The drawn textures, never photos, for the waking world (`Scene::sky`).
    drawn: Vec<tex::Texture>,
    /// By texture id, and whether drawn.
    materials: HashMap<(u8, bool), Handle<PsxMaterial>>,
}

impl LevelTextures {
    /// For the level textures as loaded now (call after `tex::load_level` / `load_strange`).
    pub fn new() -> LevelTextures {
        LevelTextures { base: tex::make_all(), drawn: tex::make_drawn(), materials: HashMap::new() }
    }

    /// Texture `id`'s material, from the drawn textures for the waking world (`plain`).
    fn material(&mut self, id: u8, plain: bool, images: &mut Assets<Image>, materials: &mut Assets<PsxMaterial>) -> Option<Handle<PsxMaterial>> {
        if let Some(m) = self.materials.get(&(id, plain)) {
            return Some(m.clone());
        }
        let t = tex::level_texture(if plain { &self.drawn } else { &self.base }, id)?;
        let m = materials.add(material(images.add(image(t))));
        self.materials.insert((id, plain), m.clone());
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
    layers: Res<'w, FleshLayers>,
    buffers: ResMut<'w, Assets<bevy::render::storage::ShaderBuffer>>,
}

impl Gfx<'_> {
    /// Creatures for the flesh pass apart from a scene's own (the editor sculpts
    /// them in the background): `flesh` as one mesh, or nothing when it is empty.
    pub fn spawn_flesh(&mut self, commands: &mut Commands, flesh: &crate::world::flesh::Flesh) -> Option<Entity> {
        flesh::spawn_flesh(commands, &mut self.meshes, &mut self.images, &mut self.flesh, flesh, &self.view)
    }

    /// Creatures drawn through the PS1 pipeline as they are this frame (say
    /// `Scene::low_poly_creatures`), one mesh per skin; `skins` keeps each skin's material
    /// from frame to frame, and must be emptied for a new scene.
    pub fn spawn_creatures(
        &mut self,
        commands: &mut Commands,
        scene: &Scene,
        tris: &[Tri],
        skins: &mut HashMap<u8, Handle<PsxMaterial>>,
    ) -> Vec<Entity> {
        let mut by_skin: HashMap<u8, Vec<&Tri>> = HashMap::new();
        for t in tris {
            by_skin.entry(t.tex).or_default().push(t);
        }
        let mut out = Vec::new();
        for (id, tris) in by_skin {
            let m = match skins.get(&id) {
                Some(m) => m.clone(),
                None => {
                    let Some(skin) = scene.skins.get(id.wrapping_sub(tex::SKIN_BASE) as usize) else { continue };
                    let m = self.materials.add(material(self.images.add(image(skin))));
                    skins.insert(id, m.clone());
                    m
                }
            };
            out.push(spawn_part(commands, &mut self.meshes, &tris, m, &self.torch).0);
        }
        out
    }

    /// People as they are this frame (`Scene::people_tris`), one mesh per texture; `looks`
    /// keeps their materials from frame to frame, redrawing a face's image when it changes,
    /// and must be emptied for a new scene.
    pub fn spawn_people(&mut self, commands: &mut Commands, scene: &Scene, tris: &[Tri], looks: &mut PeopleLooks) -> Vec<Entity> {
        spawn_people(commands, &mut self.meshes, &mut self.images, &mut self.materials, &self.torch, scene, tris, looks)
    }

    /// Triangles of `fl`'s that change every frame, drawn with the skins of `material`.
    pub fn spawn_moving_flesh(
        &mut self,
        commands: &mut Commands,
        material: &MeshMaterial3d<flesh::FleshMaterial>,
        fl: &crate::world::flesh::Flesh,
        tris: &[crate::world::flesh::Tri],
    ) -> Entity {
        flesh::spawn_moving(commands, &mut self.meshes, material, fl, tris)
    }

    /// `fl`'s flesh that moves, built once at its dial's turn, with the skins of `material`:
    /// it and what moves it (`set_moving_parts`), or nothing when `material` has gone.
    pub fn spawn_moving_parts(
        &mut self,
        commands: &mut Commands,
        material: &MeshMaterial3d<flesh::FleshMaterial>,
        fl: &crate::world::flesh::Flesh,
        moving: &crate::world::dreamed::Moving,
    ) -> Option<(Entity, Handle<bevy::render::storage::ShaderBuffer>)> {
        flesh::spawn_parts(commands, &mut self.meshes, &mut self.flesh, &mut self.buffers, material, fl, moving)
    }

    /// Move the flesh `spawn_moving_parts` made as `moving` has it posed now.
    pub fn set_moving_parts(&mut self, parts: &Handle<bevy::render::storage::ShaderBuffer>, moving: &crate::world::dreamed::Moving) {
        flesh::set_parts(&mut self.buffers, parts, &moving.state);
    }
}

/// Spawn `scene` as meshes: the level and any PS1-pipeline creatures one mesh
/// per texture, each texture chosen at the strangeness where the triangle is,
/// the moving things (`spawn_thing`), and full-colour creatures for the flesh
/// pass. Returns the entities, to despawn when the scene goes.
pub fn spawn(commands: &mut Commands, gfx: &mut Gfx, scene: &Scene) -> Vec<Entity> {
    // People stand still unless something moves them (and draws them as they move).
    let m = spawn_level(commands, gfx, scene, None, !scene.creatures_move);
    let mut out = spawn_own(commands, gfx, scene, &m.level);
    out.extend(m.parts);
    out.extend(m.people);
    out
}

/// A level's meshes that stay while it's loaded, whether it's the one being walked or one
/// beside it (`Placed`): its walls, floors and ceilings (`level` the meshes of them), its
/// doors and the like, and its people as they stand.
pub struct LevelMeshes {
    pub parts: Vec<Entity>,
    pub people: Vec<Entity>,
    pub level: Vec<Handle<Mesh>>,
}

/// A scene's level meshes, made ready away from the main thread (`prepare`), for
/// `spawn_level` to spawn.
pub struct Prepared(Vec<(u8, Mesh, LevelPart)>);

/// Make `scene`'s level meshes, one per texture, each texture chosen at the strangeness
/// where the triangle is, lit by the scene alone (the torch comes after).
pub fn prepare(scene: &Scene) -> Prepared {
    let mut groups: HashMap<u8, Vec<&Tri>> = HashMap::new();
    for t in &scene.tris {
        let centre = [0, 1, 2].map(|c| t.v.iter().map(|v| v.p[c] as f32).sum::<f32>() / 3.0);
        let (id, _) = tex::strange_variant(t.tex, scene.morph(centre));
        groups.entry(id).or_default().push(t);
    }
    let mut ids: Vec<_> = groups.keys().copied().collect();
    ids.sort();
    Prepared(ids.into_iter().map(|id| {
        let (mesh, part) = make_part(&groups[&id], &Torch::default());
        (id, mesh, part)
    }).collect())
}

/// Spawn a level's meshes (`LevelMeshes`), from `prepared` if they were made ready (from
/// this scene as it is), its people too if `people`.
pub fn spawn_level(commands: &mut Commands, gfx: &mut Gfx, scene: &Scene, prepared: Option<Prepared>, people: bool) -> LevelMeshes {
    let prepared = prepared.unwrap_or_else(|| prepare(scene));
    let mut out = LevelMeshes { parts: Vec::new(), people: Vec::new(), level: Vec::new() };
    {
        let Gfx { meshes, images, materials, textures, .. } = gfx;
        for (id, mesh, part) in prepared.0 {
            let Some(m) = textures.material(id, scene.sky.is_some(), images, materials) else { continue };
            let fogs = Fogs::of(materials, &m);
            let mesh = meshes.add(mesh);
            out.parts.push(commands.spawn((Mesh3d(mesh.clone()), MeshMaterial3d(m), part, fogs)).id());
            out.level.push(mesh);
        }
    }
    for i in 0..scene.things.parts.len() {
        out.parts.extend(spawn_thing(commands, gfx, scene, i));
    }
    if people {
        let Gfx { meshes, images, materials, torch, .. } = gfx;
        let tris = scene.people_tris(|_| true);
        out.people = spawn_people(commands, meshes, images, materials, torch, scene, &tris, &mut PeopleLooks::default());
    }
    out
}

/// What only the level being walked has (`level` its level meshes): its sky, its creatures
/// as PS1 models while nothing moves them, and the full-colour creatures of the flesh pass.
pub fn spawn_own(commands: &mut Commands, gfx: &mut Gfx, scene: &Scene, level: &[Handle<Mesh>]) -> Vec<Entity> {
    let Gfx { meshes, images, materials, flesh, occluders, torch, view, layers, .. } = gfx;
    let mut out = Vec::new();
    if let Some(s) = &scene.sky {
        out.push(sky::spawn(commands, meshes, images, materials, s, scene.strange.seed));
    }
    // Creatures drawn through the PS1 pipeline use the scene's own skins. Moving ones are
    // drawn as they stand instead (`Gfx::spawn_creatures`).
    let mut skins: HashMap<u8, Vec<&Tri>> = HashMap::new();
    let still = if scene.creatures_move { &[][..] } else { &scene.creature_tris[..] };
    for t in still {
        skins.entry(t.tex).or_default().push(t);
    }
    for (id, tris) in skins {
        let Some(skin) = scene.skins.get(id.wrapping_sub(tex::SKIN_BASE) as usize) else { continue };
        let m = materials.add(material(images.add(image(skin))));
        out.push(spawn_part(commands, meshes, &tris, m, torch).0);
    }
    // Kept apart, the creatures are drawn whole: nothing of the level hides them.
    let level = if layers.0 { Vec::new() } else { level.to_vec() };
    let occluder = occluders.add(flesh::OccluderMaterial::default());
    out.extend(flesh::spawn(commands, meshes, images, flesh, occluder, &level, scene, view));
    out
}

/// A mesh's material in the walked level's fog, and out of it for drawing a level beside it.
#[derive(Component, Clone)]
pub struct Fogs {
    fogged: Handle<PsxMaterial>,
    clear: Handle<PsxMaterial>,
}

impl Fogs {
    fn of(materials: &mut Assets<PsxMaterial>, m: &Handle<PsxMaterial>) -> Fogs {
        let clear = materials.get(m).map(|m| PsxMaterial { fog: false, ..m.clone() }).map(|c| materials.add(c));
        Fogs { fogged: m.clone(), clear: clear.unwrap_or_else(|| m.clone()) }
    }
}

/// One of a level's meshes drawn beside the level being walked, where it stands against it
/// (its transform), as through a doorway into it: by its own light, with no torch, out of
/// the walked level's fog, and only where `clip`'s planes (`PsxMaterial::clip`; those of the
/// doorway it's seen through, `view_plane`s of `link::Doorway::portal`) let it be seen.
/// Taking it away has the mesh drawn as the walked level's again.
#[derive(Component, Clone, Default, PartialEq)]
pub struct Placed {
    pub clip: Vec<Vec4>,
}

/// A plane in world units (`link::Doorway::portal`'s) as the view's world has it.
pub fn view_plane(n: [f32; 4]) -> Vec4 {
    // A view point v is the world's (v.x, -v.y, -v.z) * TF.
    Vec4::new(n[0], -n[1], -n[2], n[3] / TF)
}

/// Where a level placed by `p` (`world::link`) stands in the view's world.
pub fn placed_transform(p: &crate::world::link::Placement) -> Transform {
    let zero = p.point([0.0; 3]);
    Transform { translation: to_view(zero), rotation: Quat::from_rotation_y(p.yaw()), ..default() }
}

/// A level beside the one being walked, drawn where it stands against it (`at`): its
/// `LevelMeshes`, all `Placed`, and none of its creatures. Returns the entities.
pub fn spawn_placed(commands: &mut Commands, gfx: &mut Gfx, scene: &Scene, at: Transform) -> Vec<Entity> {
    let m = spawn_level(commands, gfx, scene, None, true);
    let out: Vec<Entity> = m.parts.into_iter().chain(m.people).collect();
    for &e in &out {
        commands.entity(e).insert((at, Placed::default()));
    }
    out
}

/// Meshes newly `Placed` take their clear materials and lose the torch; those no longer
/// placed take the walked level's fog and torch again. The clear materials follow `clip`.
fn place_parts(
    torch: Res<Torch>,
    changed: Query<(), Changed<Placed>>,
    mut gone: RemovedComponents<Placed>,
    mut parts: Query<(&Fogs, &mut MeshMaterial3d<PsxMaterial>, Option<(&mut LevelPart, &Mesh3d)>, Option<Ref<Placed>>)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PsxMaterial>>,
) {
    let gone: Vec<Entity> = gone.read().collect();
    if changed.is_empty() && gone.is_empty() {
        return;
    }
    let none = Torch::default();
    let mut relight = |part: Option<(Mut<LevelPart>, &Mesh3d)>, t: &Torch| {
        let Some((mut part, mesh)) = part else { return };
        if let Some(mut mesh) = meshes.get_mut(&mesh.0) {
            if let Some(VertexAttributeValues::Float32x4(colours)) = mesh.attribute_mut(Mesh::ATTRIBUTE_COLOR) {
                part.relight(t, colours);
            }
        }
    };
    for e in gone {
        if let Ok((fogs, mut m, part, None)) = parts.get_mut(e) {
            m.0 = fogs.fogged.clone();
            relight(part, &torch);
        }
    }
    for (fogs, mut m, part, placed) in &mut parts {
        let Some(placed) = placed.filter(|p| p.is_changed()) else { continue };
        if materials.get(&fogs.clear).is_some_and(|c| c.clip != placed.clip) {
            if let Some(mut c) = materials.get_mut(&fogs.clear) {
                c.clip = placed.clip.clone();
            }
        }
        if m.0 != fogs.clear {
            m.0 = fogs.clear.clone();
            relight(part, &none);
        }
    }
}

/// People's materials (`Gfx::spawn_people`): each texture's, and each face's image with the
/// `Person::face_version` it shows.
#[derive(Default)]
pub struct PeopleLooks {
    materials: HashMap<u8, Handle<PsxMaterial>>,
    fogs: HashMap<u8, Fogs>,
    faces: HashMap<u8, (Handle<Image>, u32)>,
}

#[allow(clippy::too_many_arguments)]
fn spawn_people(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<PsxMaterial>,
    torch: &Torch,
    scene: &Scene,
    tris: &[Tri],
    looks: &mut PeopleLooks,
) -> Vec<Entity> {
    let mut by_tex: HashMap<u8, Vec<&Tri>> = HashMap::new();
    for t in tris {
        by_tex.entry(t.tex).or_default().push(t);
    }
    let mut out = Vec::new();
    for (id, tris) in by_tex {
        let person = id.checked_sub(tex::FACE_BASE).map(usize::from).and_then(|i| scene.people.get(i));
        if let (Some(p), Some((image, shown))) = (person, looks.faces.get_mut(&id)) {
            if *shown != p.face_version {
                if let Some(mut im) = images.get_mut(&*image) {
                    im.data = Some(p.face_tex.rgba());
                }
                *shown = p.face_version;
            }
        }
        let m = match looks.materials.get(&id) {
            Some(m) => m.clone(),
            None => {
                let (texture, double_sided) = match (id, person) {
                    (tex::PERSON_SKIN, _) => (crate::world::person::skin_texture(), false),
                    // Hair is seen from inside where it falls.
                    (tex::PERSON_HAIR, _) => (crate::world::hair::texture(), true),
                    (c, _) if (tex::CLOTH_BASE..tex::FACE_BASE).contains(&c) => {
                        let Some(&cloth) = crate::world::clothes::CLOTHS.get((c - tex::CLOTH_BASE) as usize) else { continue };
                        (crate::world::clothes::texture(cloth), false)
                    }
                    (_, Some(p)) => (tex::Texture { idx: p.face_tex.idx.clone(), pal: p.face_tex.pal }, false),
                    _ => continue,
                };
                let image = images.add(image(&texture));
                if let Some(p) = person {
                    looks.faces.insert(id, (image.clone(), p.face_version));
                }
                let m = materials.add(PsxMaterial { double_sided, ..material(image) });
                looks.materials.insert(id, m.clone());
                m
            }
        };
        let fogs = looks.fogs.entry(id).or_insert_with(|| Fogs::of(materials, &m)).clone();
        let e = spawn_part(commands, meshes, &tris, m, torch).0;
        commands.entity(e).insert(fogs);
        out.push(e);
    }
    out
}

/// A mesh of one of the scene's moving parts (`Scene::things`), and which.
#[derive(Component)]
pub struct ThingPart(pub usize);

/// Part `i` of the scene's moving things as meshes, one per texture, each
/// also hiding creatures behind it in the flesh pass. All are tagged
/// `ThingPart(i)`, so they can be despawned and spawned again when it moves.
pub fn spawn_thing(commands: &mut Commands, gfx: &mut Gfx, scene: &Scene, i: usize) -> Vec<Entity> {
    let Gfx { meshes, images, materials, textures, torch, occluders, layers, .. } = gfx;
    let mut groups: HashMap<u8, Vec<&Tri>> = HashMap::new();
    for t in &scene.things.parts[i].tris {
        let centre = [0, 1, 2].map(|c| t.v.iter().map(|v| v.p[c] as f32).sum::<f32>() / 3.0);
        groups.entry(tex::strange_variant(t.tex, scene.morph(centre)).0).or_default().push(t);
    }
    let mut out = Vec::new();
    for (id, tris) in groups {
        let Some(m) = textures.material(id, scene.sky.is_some(), images, materials) else { continue };
        let fogs = Fogs::of(materials, &m);
        let (e, mesh) = spawn_part(commands, meshes, &tris, m, torch);
        commands.entity(e).insert((ThingPart(i), fogs));
        out.push(e);
        if layers.0 {
            continue;
        }
        let occluder = occluders.add(flesh::OccluderMaterial::default());
        let layer = bevy::camera::visibility::RenderLayers::layer(flesh::FLESH_LAYER);
        out.push(commands.spawn((Mesh3d(mesh), MeshMaterial3d(occluder), layer, ThingPart(i))).id());
    }
    out
}

fn spawn_part(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    tris: &[&Tri],
    m: Handle<PsxMaterial>,
    torch: &Torch,
) -> (Entity, Handle<Mesh>) {
    let (mesh, part) = make_part(tris, torch);
    let mesh = meshes.add(mesh);
    (commands.spawn((Mesh3d(mesh.clone()), MeshMaterial3d(m), part)).id(), mesh)
}

/// A mesh of `tris`, lit by `torch`, and what relights it.
fn make_part(tris: &[&Tri], torch: &Torch) -> (Mesh, LevelPart) {
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
    let mut part = LevelPart { verts, lit: Vec::new() };
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    let colours = part.colours(torch);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    mesh.insert_indices(Indices::U32((0..n as u32).collect()));
    (mesh, part)
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
    fn colours(&mut self, torch: &Torch) -> Vec<[f32; 4]> {
        self.lit = self.in_reach(torch).collect();
        self.verts.iter().map(|v| Self::colour(v, torch)).collect()
    }

    fn colour(v: &PartVert, torch: &Torch) -> [f32; 4] {
        let t = torch.light(v.p, v.n, v.alb);
        let c = [0, 1, 2].map(|k| srgb_to_linear(((v.col[k] + t[k]) / 128.0).min(255.0 / 128.0)));
        [c[0], c[1], c[2], 1.0]
    }

    /// The vertices `torch` can reach.
    fn in_reach(&self, torch: &Torch) -> impl Iterator<Item = u32> + '_ {
        let torch = *torch;
        let lit = torch.colour != [0.0; 3];
        let r2 = torch.radius * torch.radius;
        self.verts.iter().enumerate().filter_map(move |(i, v)| {
            let d = [0, 1, 2].map(|c| v.p[c] - torch.pos[c]);
            (lit && d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < r2).then_some(i as u32)
        })
    }

    /// Relight `colours` (as `colours` made them for the torch before) for `torch`: only the
    /// vertices it lit before or reaches now change. False when none do.
    fn relight(&mut self, torch: &Torch, colours: &mut [[f32; 4]]) -> bool {
        let now: Vec<u32> = self.in_reach(torch).collect();
        if now.is_empty() && self.lit.is_empty() {
            return false;
        }
        let none = Torch::default();
        for &i in &self.lit {
            colours[i as usize] = Self::colour(&self.verts[i as usize], &none);
        }
        for &i in &now {
            colours[i as usize] = Self::colour(&self.verts[i as usize], torch);
        }
        self.lit = now;
        true
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Relights the parts when the torch moves or changes: only those it reached or reaches,
/// so the rest aren't sent to the GPU again.
pub fn apply_torch(torch: Res<Torch>, mut parts: Query<(&mut LevelPart, &Mesh3d), Without<Placed>>, mut meshes: ResMut<Assets<Mesh>>) {
    for (mut part, mesh) in &mut parts {
        if !torch.is_changed() && !part.is_added() {
            continue;
        }
        // Only a part the torch reached or reaches is touched: getting it to change sends it
        // to the GPU again.
        if part.lit.is_empty() && part.in_reach(&torch).next().is_none() {
            continue;
        }
        let Some(mut m) = meshes.get_mut(&mesh.0) else { continue };
        match m.attribute_mut(Mesh::ATTRIBUTE_COLOR) {
            Some(VertexAttributeValues::Float32x4(colours)) if colours.len() == part.verts.len() => {
                part.relight(&torch, colours);
            }
            _ => {
                let colours = part.colours(&torch);
                m.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
            }
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
            .add_systems(PostUpdate, (apply_view, place_parts, apply_torch, sky::follow).chain().before(bevy::camera::CameraUpdateSystems));
        flesh::plugin(app);
    }
}
