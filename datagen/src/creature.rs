//! Low-poly creatures, built from boxes and octahedra the way many PS1 games
//! built theirs: humanoids (skeletons, knights, ghouls, lizardmen, sometimes
//! giants), four-legged beasts, slimes and floating eyes. Each gets a random
//! pose: limb swings, a lean, a turned head, maybe a weapon. They stand in
//! room cells that the walker then treats as solid.

use crate::rng::Rng;
use crate::scene::{apply, mul, norm, rot_x, rot_y, rot_z, Scene, M3, T};
use crate::tex;

const TF: f32 = T as f32;

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f32; 3], k: f32) -> [f32; 3] {
    a.map(|c| c * k)
}

/// Tint `base` with a little per-creature variation.
fn vary(rng: &mut Rng, base: [f32; 3]) -> [f32; 3] {
    base.map(|c| (c * rng.fr(0.85, 1.1)).clamp(0.2, 1.0))
}

impl Scene {
    /// A limb hanging from `joint` along local +y (down) of `m`; returns its far end.
    fn limb(&mut self, joint: [f32; 3], m: M3, len: f32, thick: f32, tex: u8, tint: [f32; 3]) -> [f32; 3] {
        self.add_obox(add(joint, apply(m, [0.0, len / 2.0, 0.0])), m, [thick, len / 2.0, thick], tex, tint);
        add(joint, apply(m, [0.0, len, 0.0]))
    }

    /// Octahedron with radii `r` along the columns of `m`.
    pub(crate) fn add_octa(&mut self, c: [f32; 3], m: M3, r: [f32; 3], tex: u8, tint: [f32; 3]) {
        let v = |l: [f32; 3]| {
            let w = apply(m, l);
            [0, 1, 2].map(|k| (c[k] + w[k]).round() as i32)
        };
        for sx in [1.0, -1.0] {
            for sy in [1.0, -1.0] {
                for sz in [1.0, -1.0] {
                    let n = norm(apply(m, [sx / r[0], sy / r[1], sz / r[2]]));
                    let (a, b, t) = (v([sx * r[0], 0.0, 0.0]), v([0.0, 0.0, sz * r[2]]), v([0.0, sy * r[1], 0.0]));
                    self.tri([(a, [0, 63]), (b, [63, 63]), (t, [31, 0])], n, tex, tint);
                }
            }
        }
    }

    fn humanoid(&mut self, base: [f32; 3], yaw: f32, h: f32, rng: &mut Rng) {
        // (body texture, head texture, tint)
        let kinds: [(u8, u8, [f32; 3]); 4] = [
            (tex::BONE, tex::BONE, [1.0, 0.95, 0.85]),
            (5, 5, [0.75, 0.8, 0.9]),
            (tex::FUR, tex::SCALES, [0.7, 0.6, 0.55]),
            (tex::SCALES, tex::SCALES, [0.55, 0.85, 0.5]),
        ];
        let (body, head, tint) = kinds[rng.range(0, 4) as usize];
        let tint = vary(rng, tint);
        let skinny = if body == tex::BONE { 0.6 } else { 1.0 };
        let r0 = rot_y(yaw);
        let at = |l: [f32; 3]| add(base, apply(r0, l));

        let leg_len = 0.46 * h;
        let swing = rng.fr(-0.5, 0.5);
        for (side, s) in [(1.0, swing), (-1.0, -swing)] {
            let m = mul(r0, mul(rot_x(s), rot_z(-side * rng.fr(0.0, 0.12))));
            let hip = at([side * 0.07 * h, -leg_len, 0.0]);
            // Hip to foot runs down, so start the limb at the hip.
            self.limb(hip, m, leg_len, 0.045 * h * skinny, body, tint);
        }

        let lean = rng.fr(-0.15, 0.35);
        let mt = mul(r0, rot_x(lean));
        let up = apply(mt, [0.0, -1.0, 0.0]);
        let torso_len = 0.32 * h;
        let hip = at([0.0, -leg_len, 0.0]);
        self.add_obox(
            add(hip, scale(up, torso_len / 2.0)),
            mt,
            [0.13 * h, torso_len / 2.0, 0.07 * h * skinny],
            body,
            tint,
        );
        let neck = add(hip, scale(up, torso_len));

        let weapon = rng.chance(0.4);
        for side in [1.0, -1.0] {
            let shoulder = add(neck, apply(mt, [side * 0.17 * h, 0.03 * h, 0.0]));
            let reach = rng.fr(-0.4, 1.0);
            let m = mul(mt, mul(rot_x(reach), rot_z(-side * rng.fr(0.05, 0.5))));
            let hand = self.limb(shoulder, m, 0.42 * h, 0.035 * h * skinny, body, tint);
            if weapon && side > 0.0 {
                // A blade carried on along the forearm.
                self.add_obox(
                    add(hand, apply(m, [0.0, 0.22 * h, 0.0])),
                    m,
                    [0.012 * h, 0.25 * h, 0.035 * h],
                    5,
                    [0.8; 3],
                );
            }
        }

        let mh = mul(mt, rot_y(rng.fr(-0.6, 0.6)));
        let head_centre = add(neck, scale(up, 0.1 * h));
        if rng.chance(0.5) {
            self.add_obox(head_centre, mh, [0.075 * h, 0.085 * h, 0.08 * h], head, tint);
        } else {
            self.add_octa(head_centre, mh, [0.08 * h, 0.1 * h, 0.09 * h], head, tint);
        }
    }

    fn beast(&mut self, base: [f32; 3], yaw: f32, len: f32, rng: &mut Rng) {
        let skin = if rng.chance(0.5) { tex::FUR } else { tex::SCALES };
        let tint = vary(rng, if skin == tex::FUR { [0.75, 0.6, 0.5] } else { [0.6, 0.75, 0.55] });
        let r0 = rot_y(yaw);
        let at = |l: [f32; 3]| add(base, apply(r0, l));
        let leg_len = 0.32 * len;
        let body_y = -(leg_len + 0.12 * len);
        self.add_obox(at([0.0, body_y, 0.0]), r0, [0.16 * len, 0.12 * len, 0.42 * len], skin, tint);
        for (lx, lz) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
            let m = mul(r0, rot_x(rng.fr(-0.45, 0.45)));
            self.limb(at([lx * 0.12 * len, -leg_len, lz * 0.32 * len]), m, leg_len, 0.04 * len, skin, tint);
        }
        let mh = mul(r0, mul(rot_y(rng.fr(-0.7, 0.7)), rot_x(rng.fr(-0.3, 0.3))));
        let neck = at([0.0, body_y - 0.05 * len, 0.42 * len]);
        self.add_obox(
            add(neck, apply(mh, [0.0, -0.05 * len, 0.12 * len])),
            mh,
            [0.1 * len, 0.09 * len, 0.14 * len],
            skin,
            tint,
        );
        // Tail swept back and up.
        let tail = mul(r0, rot_x(-rng.fr(1.8, 2.6)));
        self.limb(at([0.0, body_y - 0.05 * len, -0.42 * len]), tail, 0.4 * len, 0.03 * len, skin, tint);
    }

    fn slime(&mut self, base: [f32; 3], yaw: f32, r: f32, rng: &mut Rng) {
        let base_tint = if rng.chance(0.5) { [0.5, 0.9, 0.45] } else { [0.75, 0.45, 0.85] };
        let tint = vary(rng, base_tint);
        let m = rot_y(yaw);
        let squash = rng.fr(0.45, 0.7);
        let c = add(base, [0.0, -squash * r, 0.0]);
        self.add_octa(c, m, [r, squash * r, r * rng.fr(0.8, 1.2)], if rng.chance(0.5) { tex::SCALES } else { 4 }, tint);
        for side in [1.0, -1.0] {
            let eye = add(c, apply(m, [side * 0.3 * r, -0.45 * squash * r, 0.6 * r]));
            self.add_obox(eye, m, [0.06 * r, 0.06 * r, 0.06 * r], tex::BONE, [1.0; 3]);
        }
    }

    fn floater(&mut self, base: [f32; 3], yaw: f32, r: f32, lift: f32, rng: &mut Rng) {
        let tint = vary(rng, [1.0, 0.85, 0.7]);
        let m = mul(rot_y(yaw), rot_x(rng.fr(-0.4, 0.4)));
        let c = add(base, [0.0, -lift, 0.0]);
        self.add_octa(c, m, [r, r, r], 7, tint);
        let n = rng.range(3, 6);
        for i in 0..n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let joint = add(c, apply(m, [0.5 * r * a.cos(), 0.6 * r, 0.5 * r * a.sin()]));
            let tm = mul(m, mul(rot_y(a), rot_z(-rng.fr(0.1, 0.6))));
            self.limb(joint, tm, rng.fr(0.4, 0.7) * TF, 0.025 * TF, tex::SCALES, [0.6, 0.5, 0.65]);
        }
    }

    pub(crate) fn build_creatures(&mut self, rng: &mut Rng) {
        for ri in 0..self.rooms.len() {
            let r = self.rooms[ri].clone();
            for _ in 0..rng.range(0, 3) {
                let (cx, cz) = (r.x + rng.range(0, r.w), r.z + rng.range(0, r.d));
                if !self.claim_cell(ri, cx, cz) {
                    continue;
                }
                let base = [(cx as f32 + rng.fr(0.4, 0.6)) * TF, r.floor as f32, (cz as f32 + rng.fr(0.4, 0.6)) * TF];
                let clearance = if self.outdoor { 6.0 * TF } else { r.height as f32 * TF };
                let yaw = rng.fr(0.0, std::f32::consts::TAU);
                match rng.range(0, 10) {
                    0..=4 => {
                        let h = if rng.chance(0.15) { rng.fr(1.4, 1.9) * TF } else { rng.fr(0.75, 1.1) * TF };
                        self.humanoid(base, yaw, h.min(clearance - 30.0), rng);
                    }
                    5..=6 => self.beast(base, yaw, rng.fr(0.55, 0.8) * TF, rng),
                    7 => self.slime(base, yaw, rng.fr(0.25, 0.42) * TF, rng),
                    _ => {
                        let r = rng.fr(0.18, 0.3) * TF;
                        let lift = rng.fr(0.8, 1.4) * TF;
                        if lift + r < clearance - 10.0 {
                            self.floater(base, yaw, r, lift, rng);
                        }
                    }
                }
            }
        }
    }
}
