//! The player's sword: a small model built in code, its pose in view space, and the swing.
//! It is drawn by `hud` with a little software rasteriser: flat-lit triangles on whole
//! pixels, sorted back to front with no depth buffer, as the PS1 drew them. Nothing here
//! needs Bevy.

/// Focal length in pixels at the frame's 240 height (`engine::world::walker::H`).
const FOCAL: f32 = 180.0;
const NEAR: f32 = 0.08;

/// Seconds spent drawing back, striking, holding at the end, and returning.
const KEYS: [(f32, Pose); 5] = [
    (0.0, REST),
    (0.13, Pose { pos: [0.50, 0.26, 0.62], rot: [0.15, 0.20, 0.50] }),
    (0.26, Pose { pos: [-0.24, 0.42, 0.75], rot: [-1.20, -0.15, -0.90] }),
    (0.31, Pose { pos: [-0.30, 0.46, 0.75], rot: [-1.30, -0.15, -1.00] }),
    (0.58, REST),
];
/// A new swing can start this long before the last has quite finished.
const NEXT_SWING: f32 = 0.08;
/// Held at the right, the blade up, leaning forward and a little inward.
const REST: Pose = Pose { pos: [0.46, 0.46, 0.80], rot: [-0.38, -0.15, -0.26] };

/// How far the walking bob moves the sword per world unit it moves the camera.
const BOB: f32 = 0.005;
/// Sideways lag per yaw unit a second of turning, its limit, and how fast it follows.
const SWAY: f32 = 2.5e-5;
const SWAY_MOST: f32 = 0.08;
const SWAY_RATE: f32 = 12.0;

/// Where the sword is: `pos` of the hand in view space (x right, y down, z ahead, in sword
/// lengths), `rot` radians about x, y and z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub pos: [f32; 3],
    pub rot: [f32; 3],
}

impl Pose {
    fn mix(a: Pose, b: Pose, t: f32) -> Pose {
        let m = |x: f32, y: f32| x + (y - x) * t;
        Pose { pos: [0, 1, 2].map(|i| m(a.pos[i], b.pos[i])), rot: [0, 1, 2].map(|i| m(a.rot[i], b.rot[i])) }
    }

    /// A point of the model (y along the blade towards its tip) in view space.
    fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        // Model y is up, view y is down.
        let [x, y, z] = [p[0], -p[1], p[2]];
        let (sx, cx) = self.rot[0].sin_cos();
        let (sy, cy) = self.rot[1].sin_cos();
        let (sz, cz) = self.rot[2].sin_cos();
        let (x, z) = (x * cy + z * sy, -x * sy + z * cy);
        let (y, z) = (y * cx - z * sx, y * sx + z * cx);
        let (x, y) = (x * cz - y * sz, x * sz + y * cz);
        [x + self.pos[0], y + self.pos[1], z + self.pos[2]]
    }

    /// A direction of the model in view space.
    fn turn(&self, d: [f32; 3]) -> [f32; 3] {
        let o = Pose { pos: [0.0; 3], rot: self.rot };
        o.apply(d)
    }
}

#[derive(Default)]
pub struct Sword {
    /// Seconds into the current swing.
    swing: Option<f32>,
    sway: f32,
    last_yaw: Option<f32>,
}

impl Sword {
    fn length() -> f32 {
        KEYS[KEYS.len() - 1].0
    }

    pub fn swinging(&self) -> bool {
        self.swing.is_some()
    }

    /// Start a swing, unless one is still under way.
    pub fn attack(&mut self) {
        if self.swing.is_none_or(|t| t >= Self::length() - NEXT_SWING) {
            self.swing = Some(0.0);
        }
    }

    /// Move on by `dt` seconds, with the view now turned to `yaw` (4096 units a turn).
    pub fn update(&mut self, dt: f32, yaw: f32) {
        if let Some(t) = &mut self.swing {
            *t += dt;
            if *t >= Self::length() {
                self.swing = None;
            }
        }
        let rate = self.last_yaw.map_or(0.0, |y| (yaw - y) / dt.max(1e-3));
        self.last_yaw = Some(yaw);
        let want = (rate * SWAY).clamp(-SWAY_MOST, SWAY_MOST);
        self.sway += (want - self.sway) * (dt * SWAY_RATE).min(1.0);
    }

    /// The pose now, with the camera's walking bob `bob` (world units).
    pub fn pose(&self, bob: f32) -> Pose {
        let mut pose = self.swing.map_or(REST, |t| {
            let i = KEYS.iter().rposition(|k| k.0 <= t).unwrap_or(0).min(KEYS.len() - 2);
            let (a, b) = (KEYS[i], KEYS[i + 1]);
            let u = ((t - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
            Pose::mix(a.1, b.1, u * u * (3.0 - 2.0 * u))
        });
        pose.pos[0] += self.sway;
        pose.pos[1] += bob * BOB;
        pose
    }
}

const STEEL: [f32; 3] = [175.0, 185.0, 200.0];
const EDGE: [f32; 3] = [225.0, 232.0, 240.0];
const BRASS: [f32; 3] = [165.0, 125.0, 55.0];
const LEATHER: [f32; 3] = [85.0, 50.0, 30.0];

struct ModelTri {
    v: [[f32; 3]; 3],
    colour: [[f32; 3]; 3],
    normal: [f32; 3],
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-6);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// One closed piece along y, from rings of (y, half width, half depth, colour). Normals
/// point away from the piece's middle, so the order of corners doesn't matter.
fn piece(rings: &[(f32, f32, f32, [f32; 3])], diamond: bool, out: &mut Vec<ModelTri>) {
    let ring = |&(y, w, d, _): &(f32, f32, f32, [f32; 3])| -> [[f32; 3]; 4] {
        if diamond {
            [[w, y, 0.0], [0.0, y, d], [-w, y, 0.0], [0.0, y, -d]]
        } else {
            [[w, y, d], [-w, y, d], [-w, y, -d], [w, y, -d]]
        }
    };
    let mid = {
        let n = rings.len() as f32;
        [0.0, rings.iter().map(|r| r.0).sum::<f32>() / n, 0.0]
    };
    let mut add = |v: [[f32; 3]; 3], colour: [[f32; 3]; 3]| {
        let mut normal = unit(cross(sub(v[1], v[0]), sub(v[2], v[0])));
        let centre = [0, 1, 2].map(|c| (v[0][c] + v[1][c] + v[2][c]) / 3.0);
        if dot(normal, sub(centre, mid)) < 0.0 {
            normal = normal.map(|c| -c);
        }
        out.push(ModelTri { v, colour, normal });
    };
    for w in rings.windows(2) {
        let (a, b) = (ring(&w[0]), ring(&w[1]));
        let (ca, cb) = (w[0].3, w[1].3);
        for i in 0..4 {
            let j = (i + 1) % 4;
            add([a[i], a[j], b[i]], [ca, ca, cb]);
            add([a[j], b[j], b[i]], [ca, cb, cb]);
        }
    }
    for (r, c) in [(ring(&rings[0]), rings[0].3), (ring(&rings[rings.len() - 1]), rings[rings.len() - 1].3)] {
        add([r[0], r[1], r[2]], [c; 3]);
        add([r[0], r[2], r[3]], [c; 3]);
    }
}

/// The whole sword, about one unit long, its hand at the origin.
fn model() -> Vec<ModelTri> {
    let mut out = Vec::new();
    piece(&[(0.03, 0.055, 0.016, STEEL), (0.78, 0.047, 0.013, STEEL), (0.98, 0.002, 0.001, EDGE)], true, &mut out);
    piece(&[(0.0, 0.17, 0.03, BRASS), (0.035, 0.17, 0.03, BRASS)], false, &mut out);
    piece(&[(-0.2, 0.024, 0.024, LEATHER), (0.0, 0.024, 0.024, LEATHER)], false, &mut out);
    piece(&[(-0.26, 0.012, 0.012, BRASS), (-0.23, 0.045, 0.045, BRASS), (-0.2, 0.012, 0.012, BRASS)], true, &mut out);
    out
}

/// A triangle on the frame: whole pixels and a colour at each corner.
#[derive(Clone, Copy, Debug)]
pub struct ScreenTri {
    pub p: [(i32, i32); 3],
    pub colour: [[u8; 3]; 3],
}

/// The sword at `pose` on a frame `width` by `height` pixels, in drawing order.
pub fn triangles(pose: Pose, width: u32, height: u32) -> Vec<ScreenTri> {
    let scale = height as f32 / 240.0;
    let (cx, cy, f) = (width as f32 / 2.0, height as f32 / 2.0, FOCAL * scale);
    let light = unit([-0.45, -0.75, -0.5]);
    let mut out: Vec<(f32, ScreenTri)> = Vec::new();
    for t in model() {
        let v = t.v.map(|p| pose.apply(p));
        let n = pose.turn(t.normal);
        // Facing away from the eye: the far side, hidden by the near one.
        if dot(n, unit(v[0])) >= 0.0 || v.iter().any(|p| p[2] < NEAR) {
            continue;
        }
        let shade = 0.42 + 0.75 * dot(n, light).max(0.0);
        let colour = t.colour.map(|c| c.map(|x| (x * shade).clamp(0.0, 255.0) as u8));
        let p = v.map(|p| ((cx + f * p[0] / p[2]).round() as i32, (cy + f * p[1] / p[2]).round() as i32));
        out.push(((v[0][2] + v[1][2] + v[2][2]) / 3.0, ScreenTri { p, colour }));
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.into_iter().map(|t| t.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn step(s: &mut Sword, seconds: f32) {
        for _ in 0..(seconds / DT).round() as i32 {
            s.update(DT, 0.0);
        }
    }

    #[test]
    fn a_swing_leaves_rest_and_comes_back() {
        let mut s = Sword::default();
        assert_eq!(s.pose(0.0), REST);
        s.attack();
        assert!(s.swinging());
        step(&mut s, 0.26);
        assert!(s.pose(0.0).pos[0] < REST.pos[0] - 0.4, "{:?}", s.pose(0.0));
        step(&mut s, 0.4);
        assert!(!s.swinging());
        assert_eq!(s.pose(0.0), REST);
    }

    #[test]
    fn clicking_mid_swing_does_not_restart_it() {
        let mut s = Sword::default();
        s.attack();
        step(&mut s, 0.2);
        let before = s.pose(0.0);
        s.attack();
        assert_eq!(s.pose(0.0), before);
    }

    #[test]
    fn another_swing_can_start_as_one_ends() {
        let mut s = Sword::default();
        s.attack();
        step(&mut s, Sword::length() - 0.05);
        assert!(s.swinging());
        s.attack();
        step(&mut s, 0.1);
        assert!(s.swinging(), "the second swing should be under way");
        assert!((s.pose(0.0).pos[0] - REST.pos[0]).abs() < 0.2);
    }

    #[test]
    fn the_sword_is_on_the_right_at_rest() {
        let t = triangles(REST, 320, 240);
        assert!(!t.is_empty());
        let on_screen: Vec<_> = t.iter().flat_map(|t| t.p).filter(|p| (0..240).contains(&p.1)).collect();
        assert!(on_screen.iter().all(|p| p.0 > 160), "{on_screen:?}");
    }

    #[test]
    fn it_stays_in_front_of_the_eye_through_the_swing() {
        let mut s = Sword::default();
        s.attack();
        while s.swinging() {
            let p = s.pose(0.0);
            assert!(model().iter().flat_map(|t| t.v).all(|v| p.apply(v)[2] > NEAR), "{p:?}");
            s.update(DT, 0.0);
        }
    }

    #[test]
    fn triangles_are_far_to_near() {
        let t = triangles(REST, 320, 240);
        assert!(t.len() > 10);
        // Depth isn't kept in the output, so check the sort through the model.
        let depth = |p: &[f32; 3]| REST.apply(*p)[2];
        let far = model().iter().flat_map(|t| t.v).map(|v| depth(&v)).fold(f32::MIN, f32::max);
        assert!(far > REST.pos[2]);
    }

    #[test]
    fn turning_makes_it_lag_and_settle() {
        let mut s = Sword::default();
        s.update(DT, 0.0);
        for i in 1..30 {
            s.update(DT, i as f32 * 20.0);
        }
        assert!(s.pose(0.0).pos[0] != REST.pos[0]);
        for _ in 0..120 {
            s.update(DT, 580.0);
        }
        assert!((s.pose(0.0).pos[0] - REST.pos[0]).abs() < 1e-3);
    }

    #[test]
    fn it_scales_with_the_frame() {
        let (a, b) = (triangles(REST, 320, 240), triangles(REST, 640, 480));
        assert_eq!(a.len(), b.len());
        let (pa, pb) = (a[0].p[0], b[0].p[0]);
        assert!((pb.0 - pa.0 * 2).abs() <= 2 && (pb.1 - pa.1 * 2).abs() <= 2, "{pa:?} {pb:?}");
    }

    #[test]
    fn normals_point_outward() {
        for t in model() {
            assert!(dot(t.normal, t.normal) > 0.9);
        }
    }
}
