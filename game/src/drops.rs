//! Things the player has dropped: where they lie, picking them up again, and how they look.
//! Until the world can have plain PS1 models composited into the network's picture, a dropped
//! thing is a little flat-lit cube drawn by `hud`'s software rasteriser, as the hands are,
//! hidden when the level stands between it and the eye and fading into the fog. No Bevy here.

use crate::hands::ScreenTri;
use crate::player::{Item, Piece};
use engine::world::scene::{Camera, Scene, T};

const TF: f32 = T as f32;
/// Half the cube's side, world units.
const HALF: f32 = 0.05 * TF;
/// How far ahead of the feet a dropped thing lands: far enough to be in view.
const AHEAD: f32 = 1.0 * TF;
/// How near the feet must come to pick one up, and how far away they must have been first,
/// so a thing just dropped isn't picked straight up again.
const REACH: f32 = 0.45 * TF;
const NEAR: f32 = 8.0;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Dropped {
    /// What it is, as worn as it was.
    pub item: Piece,
    /// The middle of its bottom face, on the floor.
    pub at: [f32; 3],
    /// Whether the feet have been out of reach since it was dropped.
    armed: bool,
}

/// Put `item` down ahead of feet at `feet` facing `yaw` (radians, as `Sim::feet`), on the
/// floor there, or at the feet if a wall or a drop is in the way.
pub fn drop(scene: &Scene, item: Piece, feet: [f32; 3], yaw: f32) -> Dropped {
    let (s, c) = yaw.sin_cos();
    let ahead = [feet[0] - s * AHEAD, feet[2] + c * AHEAD];
    let knee = |x: f32, z: f32| [x, feet[1] - 0.3 * TF, z];
    let at = match scene.floor_y(ahead[0], feet[1], ahead[1]) {
        Some(y) if (y - feet[1]).abs() < TF && clear(scene, knee(feet[0], feet[2]), knee(ahead[0], ahead[1])) => {
            [ahead[0], y, ahead[1]]
        }
        _ => [feet[0], scene.floor_y(feet[0], feet[1], feet[2]).unwrap_or(feet[1]), feet[2]],
    };
    Dropped { item, at, armed: false }
}

/// A thing the level put on the floor at `at` (an `Item` entity, or out of a chest): on
/// the floor there, to be picked up as soon as the feet reach it.
pub fn lying(scene: &Scene, item: impl Into<Piece>, at: [f32; 3]) -> Dropped {
    let item = item.into();
    let y = scene.floor_y(at[0], at[1], at[2]).unwrap_or(at[1]);
    Dropped { item, at: [at[0], y, at[2]], armed: true }
}

/// The items a hand-built level has lying about, `count` of each a thing; names the game
/// doesn't know are left out and said on stderr.
pub fn of_level(scene: &Scene, level: &engine::world::level::Level) -> Vec<Dropped> {
    let mut out = Vec::new();
    for e in &level.entities {
        let engine::world::level::EntityKind::Item { name, count } = &e.kind else { continue };
        match Item::named(name) {
            Some(item) => out.extend((0..*count).map(|_| lying(scene, item, e.pos))),
            None => eprintln!("item #{}: no item called {name:?}", e.id),
        }
    }
    out
}

/// The dropped things the feet at `feet` can pick up now, by index, and arm those they've
/// left.
pub fn in_reach(drops: &mut [Dropped], feet: [f32; 3]) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, d) in drops.iter_mut().enumerate() {
        let near = (d.at[0] - feet[0]).hypot(d.at[2] - feet[2]) < REACH && (d.at[1] - feet[1]).abs() < TF;
        if !near {
            d.armed = true;
        } else if d.armed {
            out.push(i);
        }
    }
    out
}

/// Not to be picked up until the feet have left it: one there was no room for.
pub fn disarm(d: &mut Dropped) {
    d.armed = false;
}

fn colour(item: Item) -> [f32; 3] {
    match item {
        Item::HealingPotion => [205.0, 35.0, 30.0],
        Item::ManaPotion => [50.0, 85.0, 225.0],
        Item::ClearSeeingPotion => [205.0, 235.0, 230.0],
        Item::CalmingPotion => [165.0, 115.0, 225.0],
        Item::Sword | Item::Buckler => [200.0, 205.0, 215.0],
        Item::LeatherCap | Item::LeatherVest | Item::WorkGloves => [120.0, 80.0, 45.0],
        Item::CopperRing => [215.0, 130.0, 70.0],
    }
}

/// Whether nothing of the level stands between `from` and `to`.
fn clear(scene: &Scene, from: [f32; 3], to: [f32; 3]) -> bool {
    let d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    let n = ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() / 32.0).ceil().max(1.0) as i32;
    (1..n).all(|i| {
        let t = i as f32 / n as f32;
        let p = [from[0] + d[0] * t, from[1] + d[1] * t, from[2] + d[2] * t];
        scene.node_at(p[0], p[1], p[2]).is_some_and(|node| {
            let s = scene.span(node);
            s.floor(node.0, node.1, [p[0], p[2]]) > p[1] && (s.sky || s.ceiling(node.0, node.1, [p[0], p[2]]) < p[1])
        })
    })
}

/// The dropped things as `cam` sees them on a frame `width` by 240, in drawing order.
pub fn triangles(drops: &[Dropped], scene: &Scene, cam: &Camera, width: u32) -> Vec<ScreenTri> {
    let eye = cam.pos.map(|c| c as f32);
    let rows = engine::world::view::view_matrix(cam).m.map(|r| r.map(|e| e as f32 / 4096.0));
    let view = |p: [f32; 3]| {
        let d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
        rows.map(|r| r[0] * d[0] + r[1] * d[1] + r[2] * d[2])
    };
    let (cx, f) = (width as f32 / 2.0, cam.h as f32);
    // Light from above and a little behind the eye; y is down.
    let light = { let l = [0.3f32, -0.85, -0.4]; let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt(); l.map(|x| x / n) };
    let mut out: Vec<(f32, ScreenTri)> = Vec::new();
    for d in drops {
        let centre = [d.at[0], d.at[1] - HALF, d.at[2]];
        let v = view(centre);
        let far = cam.fog_far as f32;
        if v[2] < NEAR || v[2] > far || !clear(scene, eye, [centre[0], centre[1] - HALF, centre[2]]) {
            continue;
        }
        let fog = ((v[2] - cam.fog_near as f32) / (far - cam.fog_near as f32).max(1.0)).clamp(0.0, 1.0);
        let base = colour(d.item.item);
        // Each face: its outward normal (world) and the two axes across it.
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                let mut n = [0.0; 3];
                n[axis] = sign;
                let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
                let corner = |u: f32, w: f32| {
                    let mut p = centre;
                    p[axis] += sign * HALF;
                    p[a] += u * HALF;
                    p[b] += w * HALF;
                    view(p)
                };
                let quad = [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)];
                // Facing away from the eye.
                let to_face = [centre[0] + n[0] * HALF - eye[0], centre[1] + n[1] * HALF - eye[1], centre[2] + n[2] * HALF - eye[2]];
                if n[0] * to_face[0] + n[1] * to_face[1] + n[2] * to_face[2] >= 0.0 || quad.iter().any(|p| p[2] < NEAR) {
                    continue;
                }
                let shade = (0.45 + 0.7 * (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0)) * (1.0 - fog);
                let c = base.map(|x| (x * shade).clamp(0.0, 255.0) as u8);
                let p = quad.map(|q| ((cx + f * q[0] / q[2]).round() as i32, (120.0 + f * q[1] / q[2]).round() as i32));
                let depth = quad.iter().map(|q| q[2]).sum::<f32>() / 4.0;
                out.push((depth, ScreenTri { p: [p[0], p[1], p[2]], colour: [c; 3] }));
                out.push((depth, ScreenTri { p: [p[0], p[2], p[3]], colour: [c; 3] }));
            }
        }
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.into_iter().map(|t| t.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dropped(at: [f32; 3]) -> Dropped {
        Dropped { item: Item::HealingPotion.into(), at, armed: false }
    }

    #[test]
    fn a_thing_dropped_ahead_is_seen_ahead_and_not_from_behind() {
        let mut rng = engine::world::rng::Rng::new(1);
        let level = engine::neural::Level::new(&mut rng);
        let w = &level.walker;
        let yaw = w.yaw / 4096.0 * std::f32::consts::TAU;
        let feet = [w.x, w.y + engine::world::walker::EYE, w.z];
        let d = drop(&level.scene, Item::HealingPotion.into(), feet, yaw);
        let mut cam = w.camera();
        cam.pitch = 400;
        let tris = triangles(&[d.clone()], &level.scene, &cam, 320);
        assert!(!tris.is_empty());
        let xs: Vec<i32> = tris.iter().flat_map(|t| t.p.map(|p| p.0)).collect();
        let mid = (xs.iter().min().unwrap() + xs.iter().max().unwrap()) / 2;
        assert!((mid - 160).abs() < 20, "{mid}");
        cam.yaw += 2048;
        assert!(triangles(&[d], &level.scene, &cam, 320).is_empty());
    }

    #[test]
    fn a_thing_just_dropped_waits_for_the_feet_to_leave_it() {
        let mut drops = vec![dropped([0.0, 0.0, 0.0])];
        assert!(in_reach(&mut drops, [10.0, 0.0, 0.0]).is_empty());
        assert!(in_reach(&mut drops, [TF, 0.0, 0.0]).is_empty());
        assert_eq!(in_reach(&mut drops, [10.0, 0.0, 0.0]), [0]);
        // Not from the floor above or below.
        let mut drops = vec![dropped([0.0, 0.0, 0.0])];
        in_reach(&mut drops, [TF, 0.0, 0.0]);
        assert!(in_reach(&mut drops, [0.0, 2.0 * TF, 0.0]).is_empty());
    }
}
