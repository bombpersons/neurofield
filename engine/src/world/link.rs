//! Where one level meets another (`level::EntityKind::Link`), and how the other is placed
//! against this one. Two levels join on a cell edge, as a door sits: each has a link there,
//! at the middle of the edge facing out of its own level into the other, named alike. The
//! edge is a hole in each (the wall that would close it isn't there), each level draws only
//! its own side of it, and the other level is drawn through it (`Doorway::portal`). So a body
//! can step across, each level is open one cell past its edge (`strip`), for walking only;
//! nothing there is drawn (`cut`). The player is handed from one level to the other as their
//! feet cross the edge (`past`).

use crate::world::scene::{Scene, T};

const TF: f32 = T as f32;

/// A link as a level has it: where (the floor point at the middle of its edge) and which way
/// it faces (radians, out of the level), the level file it joins (beside this one's), the
/// name the other's link shares, and the tiles over which the dream fades in past it.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkAt {
    pub id: u32,
    pub to: String,
    pub name: String,
    pub fade: f32,
    pub pos: [f32; 3],
    pub yaw: f32,
}

impl LinkAt {
    /// The link at `pos` facing `yaw`, moved to the middle of the cell edge it faces across.
    pub fn new(id: u32, to: &str, name: &str, fade: f32, pos: [f32; 3], yaw: f32) -> LinkAt {
        let yaw = quarter(yaw);
        let f = [-yaw.sin(), yaw.cos()].map(|c| c.round());
        let mid = |c: f32| ((c / TF).floor() + 0.5) * TF;
        // Across the way it faces, the middle of a cell; along it, the nearest edge.
        let edge = |c: f32| (c / TF).round() * TF;
        let x = if f[0] != 0.0 { edge(pos[0]) } else { mid(pos[0]) };
        let z = if f[1] != 0.0 { edge(pos[2]) } else { mid(pos[2]) };
        LinkAt { id, to: to.into(), name: name.into(), fade, pos: [x, pos[1], z], yaw }
    }

    /// The way it faces across the floor (x, z): out of its level.
    pub fn facing(&self) -> [f32; 2] {
        [-self.yaw.sin(), self.yaw.cos()]
    }

    /// How far `p` is past the link, out of its level (negative on the level's own side).
    pub fn past(&self, p: [f32; 3]) -> f32 {
        let f = self.facing();
        (p[0] - self.pos[0]) * f[0] + (p[2] - self.pos[2]) * f[1]
    }

    /// Whether `p` is in the link's doorway, or would be on either side of it: across from
    /// its middle by less than half a tile, and on its floor give or take a tile.
    pub fn in_doorway(&self, p: [f32; 3]) -> bool {
        let f = self.facing();
        let across = -(p[0] - self.pos[0]) * f[1] + (p[2] - self.pos[2]) * f[0];
        across.abs() < TF / 2.0 && self.past(p).abs() < TF && (p[1] - self.pos[1]).abs() < TF
    }

    /// The cell (x, z) a quarter tile `by` (+1 past the edge, -1 before it) from the link.
    fn cell(&self, by: f32) -> (i32, i32) {
        let f = self.facing();
        let at = |c: f32, d: f32| ((c + d * by * TF / 2.0) / TF).floor() as i32;
        (at(self.pos[0], f[0]), at(self.pos[2], f[1]))
    }

    /// The doorway cell, the level's last before the edge.
    pub fn doorway(&self) -> (i32, i32) {
        self.cell(-1.0)
    }

    /// The cell just past the edge, open for walking across it only.
    pub fn strip(&self) -> (i32, i32) {
        self.cell(1.0)
    }
}

/// A link's doorway in its level, from its floor to its ceiling (world units, y down).
#[derive(Clone, Debug, PartialEq)]
pub struct Doorway {
    pub link: LinkAt,
    pub floor: f32,
    pub ceiling: f32,
}

impl Doorway {
    /// The hole's corners in the edge: left floor, right floor, right ceiling, left ceiling
    /// (world units).
    pub fn corners(&self) -> [[f32; 3]; 4] {
        let (p, f) = (self.link.pos, self.link.facing());
        let along = [-f[1] * TF / 2.0, f[0] * TF / 2.0];
        let at = |s: f32, y: f32| [p[0] + along[0] * s, y, p[2] + along[1] * s];
        [at(-1.0, self.floor), at(1.0, self.floor), at(1.0, self.ceiling), at(-1.0, self.ceiling)]
    }

    /// Planes (x, y, z, d: a point p is kept where n · p + d >= 0, world units) that keep
    /// only what an eye at `eye` sees through the hole: past the edge, and inside the lines
    /// of sight past its sides. With the eye in the hole itself, past the edge alone.
    pub fn portal(&self, eye: [f32; 3]) -> Vec<[f32; 4]> {
        let f = self.link.facing();
        let n = [f[0], 0.0, f[1]];
        // A hair before the edge, so the two levels meet without a crack.
        let mut planes = vec![[n[0], n[1], n[2], -dot(n, self.link.pos) + 1.0]];
        if self.link.past(eye) > -8.0 {
            return planes;
        }
        let c = self.corners();
        let mid = [0, 1, 2].map(|k| c.iter().map(|p| p[k]).sum::<f32>() / 4.0);
        for i in 0..4 {
            let (a, b) = (sub(c[i], eye), sub(c[(i + 1) % 4], eye));
            let mut n = cross(a, b);
            let len = dot(n, n).sqrt();
            if len < 1e-6 {
                continue;
            }
            n = n.map(|v| v / len);
            if dot(n, sub(mid, eye)) < 0.0 {
                n = n.map(|v| -v);
            }
            planes.push([n[0], n[1], n[2], -dot(n, eye)]);
        }
        planes
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A yaw (radians) to the nearest quarter turn.
fn quarter(yaw: f32) -> f32 {
    (yaw / std::f32::consts::FRAC_PI_2).round() * std::f32::consts::FRAC_PI_2
}

/// Note each link's doorway in the scene (`Scene::doorways`), and leave out what of the
/// scene's level triangles lies in the strip past it, which is there for walking only.
pub fn cut(scene: &mut Scene, links: &[LinkAt]) {
    for l in links {
        let (dx, dz) = l.doorway();
        let at = [(dx as f32 + 0.5) * TF, (dz as f32 + 0.5) * TF];
        let Some(n) = scene.node_at(at[0], l.pos[1] - TF / 4.0, at[1]) else { continue };
        let door = Doorway { link: l.clone(), floor: scene.node_floor(n, at), ceiling: scene.node_ceiling(n, at) };
        let (sx, sz) = l.strip();
        const E: f32 = 2.0;
        let (lo, hi) = ([sx as f32 * TF - E, sz as f32 * TF - E], [(sx + 1) as f32 * TF + E, (sz + 1) as f32 * TF + E]);
        scene.tris.retain(|t| {
            let c = [0, 1, 2].map(|k| t.v.iter().map(|v| v.p[k] as f32).sum::<f32>() / 3.0);
            !(c[0] > lo[0] && c[0] < hi[0] && c[2] > lo[1] && c[2] < hi[1] && l.past(c) > 1.0)
        });
        scene.doorways.push(door);
    }
}

/// Where another level stands against this one: a point of the other's is `point` of it in
/// this one's (world units), turned by a number of quarter turns about the link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// Quarter turns, 0..4, the way yaw turns.
    pub turn: i32,
    /// The other level's link, and this one's: the same place.
    pub from: [f32; 3],
    pub to: [f32; 3],
}

impl Placement {
    /// The level whose link is `there` placed against this one's `here`: the links in the
    /// same place, facing opposite ways.
    pub fn between(here: &LinkAt, there: &LinkAt) -> Placement {
        let turns = (here.yaw - there.yaw) / std::f32::consts::FRAC_PI_2 + 2.0;
        Placement { turn: (turns.round() as i32).rem_euclid(4), from: there.pos, to: here.pos }
    }

    /// The turn in radians.
    pub fn yaw(&self) -> f32 {
        self.turn as f32 * std::f32::consts::FRAC_PI_2
    }

    fn rotate(&self, x: f32, z: f32) -> (f32, f32) {
        match self.turn {
            0 => (x, z),
            1 => (-z, x),
            2 => (-x, -z),
            _ => (z, -x),
        }
    }

    /// The other level's point `p` in this one.
    pub fn point(&self, p: [f32; 3]) -> [f32; 3] {
        let (x, z) = self.rotate(p[0] - self.from[0], p[2] - self.from[2]);
        [x + self.to[0], p[1] - self.from[1] + self.to[1], z + self.to[2]]
    }

    /// This level placed against the other.
    pub fn inverse(&self) -> Placement {
        Placement { turn: (4 - self.turn) % 4, from: self.to, to: self.from }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placed_levels_meet_at_their_links_facing_each_other() {
        let tau = std::f32::consts::TAU;
        for (ya, yb) in [(0.0, std::f32::consts::PI), (0.0, 0.0), (1.5708, 0.0), (-1.5708, 3.1416)] {
            let here = LinkAt::new(1, "b.ron", "stairs", 0.0, [1000.0, 0.0, 300.0], ya);
            let there = LinkAt::new(2, "a.ron", "stairs", 0.0, [5000.0, 640.0, 2100.0], yb);
            let p = Placement::between(&here, &there);
            let at = p.point(there.pos);
            assert!((0..3).all(|k| (at[k] - here.pos[k]).abs() < 1e-3), "{at:?}");
            // One step out of the other level comes into this one.
            let f = there.facing();
            let out = p.point([there.pos[0] + f[0] * 100.0, there.pos[1], there.pos[2] + f[1] * 100.0]);
            assert!(here.past(out) < -99.0, "turned {} for {ya} {yb}: {}", p.turn, here.past(out));
            // Its yaw turns its facing to face the other way.
            let d = (there.yaw + p.yaw() - here.yaw - std::f32::consts::PI).rem_euclid(tau);
            assert!(d < 1e-3 || tau - d < 1e-3);
            let q = p.inverse().point(p.point([7.0, -8.0, 9.0]));
            assert!((q[0] - 7.0).abs() < 1e-3 && (q[1] + 8.0).abs() < 1e-3 && (q[2] - 9.0).abs() < 1e-3);
        }
    }

    #[test]
    fn a_link_sits_on_a_cell_edge_and_knows_its_doorway() {
        // Facing +z from inside cell (1, 0): the edge at z = 1 tile.
        let l = LinkAt::new(1, "b.ron", "", 0.0, [300.0, 0.0, 200.0], 0.1);
        assert_eq!((l.pos[0], l.pos[2], l.yaw), (384.0, 256.0, 0.0));
        assert_eq!((l.doorway(), l.strip()), ((1, 0), (1, 1)));
        assert!(l.past([384.0, 0.0, 300.0]) > 0.0 && l.past([384.0, 0.0, 200.0]) < 0.0);
        assert!(l.in_doorway([400.0, -10.0, 300.0]));
        assert!(!l.in_doorway([600.0, 0.0, 300.0]) && !l.in_doorway([400.0, 0.0, 600.0]));
        // Facing -x from cell (2, 5): the edge at x = 2 tiles.
        let w = LinkAt::new(1, "b.ron", "", 0.0, [2.1 * 256.0, 0.0, 5.2 * 256.0], std::f32::consts::FRAC_PI_2);
        assert_eq!((w.pos[0], w.pos[2]), (512.0, 5.5 * 256.0));
        assert_eq!((w.doorway(), w.strip()), ((2, 5), (1, 5)));
    }

    #[test]
    fn a_portal_keeps_only_what_is_seen_through_the_hole() {
        let l = LinkAt::new(1, "b.ron", "", 0.0, [384.0, 0.0, 256.0], 0.0);
        let d = Doorway { link: l, floor: 0.0, ceiling: -512.0 };
        let eye = [384.0, -150.0, -600.0];
        let kept = |p: [f32; 3]| d.portal(eye).iter().all(|n| n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + n[3] >= 0.0);
        // Straight through, and just past the edge: seen. Before the edge, or off to the side
        // where only wall would be: not.
        assert!(kept([384.0, -150.0, 2000.0]) && kept([384.0, -100.0, 300.0]));
        assert!(!kept([384.0, -150.0, 100.0]) && !kept([3000.0, -150.0, 600.0]) && !kept([384.0, -3000.0, 600.0]));
        // In the hole, only the edge counts.
        assert_eq!(d.portal([384.0, -150.0, 250.0]).len(), 1);
    }
}
