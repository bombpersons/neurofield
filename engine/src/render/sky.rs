//! The waking world's sky (`Scene::sky`): a dome far off round the level camera, wearing
//! `world::sky::texture`, drawn without fog (its horizon is the fog's colour) and moved
//! with the camera so it never comes nearer.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use crate::world::level::Sky;
use psx_bevy::psx::PsxMaterial;
use std::f32::consts::TAU;

/// The dome's radius in tiles: past anything a level holds, short of the camera's far plane.
const RADIUS: f32 = 300.0;
/// Columns round the dome, and how many of them one width of the texture spans.
const AROUND: usize = 16;
const PER_REPEAT: usize = 4;
/// Rings from overhead down, as (elevation in degrees, texture row 0..1); below the horizon
/// it stays the horizon's colour.
const RINGS: [(f32, f32); 8] = [(90.0, 0.0), (70.0, 0.22), (50.0, 0.44), (32.0, 0.64), (18.0, 0.8), (8.0, 0.91), (0.0, 1.0), (-30.0, 1.0)];

#[derive(Component)]
pub struct SkyDome;

pub(crate) fn spawn(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<PsxMaterial>,
    sky: &Sky,
    seed: u64,
) -> Entity {
    let image = images.add(super::image(&crate::world::sky::texture(sky, seed)));
    let material = PsxMaterial { texture: Some(image), lit: false, fog: false, double_sided: true, ..default() };
    commands.spawn((Mesh3d(meshes.add(mesh())), MeshMaterial3d(materials.add(material)), SkyDome, Transform::default())).id()
}

/// The dome, each quad its own vertices so a column's texture coordinates never wrap.
fn mesh() -> Mesh {
    let (mut pos, mut uv) = (Vec::new(), Vec::new());
    let at = |ring: usize, col: usize| {
        let (e, a) = (RINGS[ring].0.to_radians(), col as f32 / AROUND as f32 * TAU);
        Vec3::new(e.cos() * a.cos(), e.sin(), e.cos() * a.sin()) * RADIUS
    };
    for ring in 0..RINGS.len() - 1 {
        for col in 0..AROUND {
            let u = |c: usize| (c - col / PER_REPEAT * PER_REPEAT) as f32 / PER_REPEAT as f32;
            let corner = |r: usize, c: usize| (at(r, c), [u(c), RINGS[r].1]);
            let q = [corner(ring, col), corner(ring, col + 1), corner(ring + 1, col), corner(ring + 1, col + 1)];
            for i in [0, 2, 1, 1, 2, 3] {
                pos.push(q[i].0.to_array());
                uv.push(q[i].1);
            }
        }
    }
    let n = pos.len() as u32;
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0f32; 4]; n as usize]);
    mesh.insert_indices(Indices::U32((0..n).collect()));
    mesh
}

/// Keep the dome centred on the level camera.
pub(crate) fn follow(cams: Query<&Transform, (With<super::LevelCamera>, Without<SkyDome>)>, mut domes: Query<&mut Transform, With<SkyDome>>) {
    let Ok(cam) = cams.single() else { return };
    for mut t in &mut domes {
        if t.translation != cam.translation {
            t.translation = cam.translation;
        }
    }
}
