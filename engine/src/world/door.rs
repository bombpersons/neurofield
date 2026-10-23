//! Doorways where corridors meet rooms: two posts and a beam, wall infill up
//! to the ceiling, and usually a door leaf swung open almost flat against the
//! room wall. Doors are never closed, so they never block the walker.

use crate::world::rng::Rng;
use crate::world::scene::{rot_y, Scene, T};
use crate::world::tex;

const TF: f32 = T as f32;
/// Half the clear opening between the posts.
const OPEN: f32 = 88.0;
const POST: f32 = 20.0;
/// Door height above the higher of the two floors.
const DOOR_H: f32 = 1.55 * TF;

impl Scene {
    pub(crate) fn build_doors(&mut self, rng: &mut Rng) {
        let rooms = self.rooms.clone();
        let in_any_room = |x: i32, z: i32| rooms.iter().any(|r| x >= r.x && x < r.x + r.w && z >= r.z && z < r.z + r.d);
        for (ri, r) in rooms.iter().enumerate() {
            for cz in r.z..r.z + r.d {
                for cx in r.x..r.x + r.w {
                    for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let (nx, nz) = (cx + dx, cz + dz);
                        let corridor = !self.spans(nx, nz).is_empty() && !in_any_room(nx, nz);
                        if corridor && rng.chance(0.65) {
                            self.doorway(ri, (cx, cz), (dx, dz), rng);
                        }
                    }
                }
            }
        }
    }

    /// Door between room cell `c` of room `ri` and the corridor cell in direction `d`.
    fn doorway(&mut self, ri: usize, c: (i32, i32), d: (i32, i32), rng: &mut Rng) {
        let r = self.rooms[ri].clone();
        let (dx, dz) = (d.0 as f32, d.1 as f32);
        // Edge midpoint, the direction along the edge, and the direction into the room.
        let e = [(c.0 as f32 + 0.5 + dx * 0.5) * TF, (c.1 as f32 + 0.5 + dz * 0.5) * TF];
        let lat = [-dz, dx];
        let into = [-dx, -dz];
        let (n, room_side, far_side) = ((c.0 + d.0, c.1 + d.1), [e[0] - dx, e[1] - dz], [e[0] + dx, e[1] + dz]);
        // Generated levels have one span per column.
        let (cn, nn) = ((c.0, c.1, 0), (n.0, n.1, 0));
        if self.spans(c.0, c.1).is_empty() || self.spans(n.0, n.1).is_empty() {
            return;
        }
        let (fr, fc) = (self.node_floor(cn, room_side), self.node_floor(nn, far_side));
        let (cr, cc) = (self.node_ceiling(cn, room_side), self.node_ceiling(nn, far_side));
        // y grows downward: the lower floor has the larger y, the lower ceiling too.
        let (bottom, top_floor, ceiling) = (fr.max(fc), fr.min(fc), cr.max(cc));
        let door_top = (top_floor - DOOR_H).max(ceiling + 24.0);
        if bottom - door_top < 0.9 * TF {
            return;
        }
        let m = rot_y(lat[1].atan2(lat[0]));
        let frame_tex = rng.pick(&[0, 1, 2]);
        let tint = r.tint;
        let point = |l: f32, y: f32, i: f32| [e[0] + lat[0] * l + into[0] * i, y, e[1] + lat[1] * l + into[1] * i];

        for side in [1.0, -1.0] {
            let c = point(side * (OPEN + POST), (bottom + door_top) / 2.0, 0.0);
            self.add_obox(c, m, [POST, (bottom - door_top) / 2.0, 22.0], frame_tex, tint);
        }
        self.add_obox(point(0.0, door_top - 12.0, 0.0), m, [TF / 2.0, 12.0, 24.0], frame_tex, tint);
        let fill_bottom = door_top - 24.0;
        if fill_bottom - ceiling > 1.0 {
            let c = point(0.0, (ceiling + fill_bottom) / 2.0, 0.0);
            self.add_obox(c, m, [TF / 2.0, (fill_bottom - ceiling) / 2.0, 14.0], r.wall_tex, tint);
        }
        self.span_mut(cn).reserved = true;

        // The leaf rests along the room wall beside the opening, so that wall must be there.
        let li = (-d.1, d.0);
        let sides = if rng.chance(0.5) { [1, -1] } else { [-1, 1] };
        for s in sides {
            let beside = (c.0 + li.0 * s, c.1 + li.1 * s);
            let in_room = beside.0 >= r.x && beside.0 < r.x + r.w && beside.1 >= r.z && beside.1 < r.z + r.d;
            if !in_room || !self.spans(beside.0 + d.0, beside.1 + d.1).is_empty() {
                continue;
            }
            let sf = s as f32;
            let alpha = rng.fr(0.03, 0.2);
            let dir =
                [lat[0] * sf * alpha.cos() + into[0] * alpha.sin(), lat[1] * sf * alpha.cos() + into[1] * alpha.sin()];
            let hinge = point(sf * OPEN, 0.0, 26.0);
            let half_w = OPEN - 2.0;
            let c = [hinge[0] + dir[0] * half_w, (fr + door_top) / 2.0, hinge[2] + dir[1] * half_w];
            self.add_obox(
                c,
                rot_y(dir[1].atan2(dir[0])),
                [half_w, (fr - door_top) / 2.0 - 4.0, 4.0],
                tex::DOOR,
                [0.9; 3],
            );
            self.span_mut((beside.0, beside.1, 0)).reserved = true;
            break;
        }
    }
}
