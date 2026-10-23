//! Creatures as one joined surface for the full-colour pass: every part as a
//! rough signed distance (from its bent axis, against its profile and lumps),
//! smoothly unioned so limbs flow into bodies, then raised veins and warts
//! sculpted on top and the zero set meshed with surface nets. Normals come from
//! the field, so the sculpting catches the light.
//!
//! Each vertex blends the skins of its two nearest parts, each mapped in its
//! own part's coordinates (around and along the axis) like the parametric
//! meshes, so skins run along limbs and cross-fade at the joins. Parts too thin
//! for the grid (claws, spikes, whiskers of tentacle) are left to the caller.
//!
//! Vein and wart layouts come from a hash of the creature's parts, never shown
//! to the network, like the lumps' layout.

use crate::world::creature::{Creature, Part, Placed};
use crate::world::flesh::{Tri, Vertex};
use crate::world::rng::hash;
use rayon::prelude::*;
use std::f32::consts::{PI, TAU};

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt().max(1e-9);
    v.map(|c| c / l)
}

/// Columns of `m` times `v`: local to world.
fn to_world(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

/// Rows of `m` times `v`: world to local (the axes are orthonormal).
fn to_local(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|c| m[0][c] * v[0] + m[1][c] * v[1] + m[2][c] * v[2])
}

/// Smallest cross-section radius a part needs to join the grid, in voxels.
const THIN: f32 = 2.5;

struct Piece<'a> {
    part: &'a Part,
    at: &'a Placed,
    /// Index of the part in its creature.
    index: usize,
    /// World box around the part, grown by the blend and the sculpting.
    lo: [f32; 3],
    hi: [f32; 3],
    /// Fillet size where it joins the rest.
    k: f32,
    waves: [[f32; 4]; 3],
    /// The part's largest profile radius.
    widest: f32,
}

struct Near {
    d: f32,
    t: f32,
    a: f32,
    /// The point in the part's frame.
    l: [f32; 3],
}

impl Piece<'_> {
    fn contains(&self, q: [f32; 3]) -> bool {
        (0..3).all(|c| q[c] >= self.lo[c] && q[c] <= self.hi[c])
    }

    /// Rough signed distance to the part's surface, and where on it (t along, a around).
    fn near(&self, q: [f32; 3]) -> Near {
        let p = self.part;
        let h = p.half;
        let l = to_local(&self.at.axes, sub(q, self.at.centre));
        let bend = p.bend * h[1];
        let axis = |t: f32| {
            let s = (t + 1.0) / 2.0;
            ([bend * s * s, t * h[1], 0.0], [bend * s, h[1], 0.0])
        };
        // Closest point on the (bent) axis, by a few Newton steps.
        let mut t = (l[1] / h[1]).clamp(-1.0, 1.0);
        for _ in 0..4 {
            let (ax, d1) = axis(t);
            let diff = sub(ax, l);
            let g = dot(diff, d1);
            let gp = dot(d1, d1) + diff[0] * bend / 2.0;
            if gp > 1e-6 {
                t = (t - g / gp).clamp(-1.0, 1.0);
            }
        }
        let (ax, d1) = axis(t);
        let tang = norm(d1);
        let dl = sub(l, ax);
        let along = dot(dl, tang);
        let perp = sub(dl, tang.map(|c| c * along));
        let a = (perp[2] / h[2]).atan2(perp[0] / h[0]);
        let lump = self.waves.iter().map(|w| (w[0] * a + w[2]).sin() * (w[1] * t * PI + w[3]).sin()).sum::<f32>() / 3.0;
        let r = p.profile(t) * (1.0 + p.lumps * lump);
        let rs = r * (h[0] * a.cos()).hypot(h[2] * a.sin());
        let rad = dot(perp, perp).sqrt();
        let d_rad = rad - rs;
        let beyond = t.abs() > 0.999 && along * t.signum() > 0.0;
        let d = if beyond { if d_rad > 0.0 { d_rad.hypot(along) } else { along.abs() } } else { d_rad };
        Near { d, t, a, l }
    }

    /// Skin coordinates wrapped around the axis and the world directions of increasing
    /// u and v, then coordinates projected onto the end caps and how much to wrap.
    fn skin_frame(&self, n: &Near, reps: [f32; 2], scale: f32) -> ([f32; 2], [f32; 3], [f32; 3], [f32; 2], f32) {
        let (h, t, a) = (self.part.half, n.t, n.a);
        let s = (t + 1.0) / 2.0;
        let uv = [a.rem_euclid(TAU) / TAU * reps[0], s * reps[1]];
        let tu = to_world(&self.at.axes, norm([-a.sin() * h[0], 0.0, a.cos() * h[2]]));
        let tv = to_world(&self.at.axes, norm([self.part.bend * h[1] * s, h[1], 0.0]));
        let cap = [n.l[0] / scale, n.l[2] / scale];
        (uv, tu, tv, cap, crate::world::flesh::wrap_weight(self.part.profile(t) / self.widest))
    }
}

/// Skin repeats around and along a part, as the parametric meshes have them.
pub fn reps(p: &Part, scale: f32) -> [f32; 2] {
    let circ = PI * (p.half[0] + p.half[2]) * (1.0 + p.bulge) * p.taper.max(1.0);
    let len = 2.0 * p.half[1] * (1.0 + 0.5 * p.bend.abs());
    [circ, len].map(|l| (l / scale).round().clamp(1.0, 8.0))
}

/// Smooth minimum with a blend of width `k`.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k.max(1e-6);
    a.min(b) - h * h * k * 0.25
}

/// Gradient noise in about -1..1.
fn noise(p: [f32; 3], seed: u64) -> f32 {
    let i = p.map(|c| c.floor());
    let f = [0, 1, 2].map(|k| p[k] - i[k]);
    let fade = f.map(|t| t * t * t * (t * (t * 6.0 - 15.0) + 10.0));
    let grad = |x: i32, y: i32, z: i32, d: [f32; 3]| {
        let h = hash(seed ^ (x as u32 as u64) ^ ((y as u32 as u64) << 21) ^ ((z as u32 as u64) << 42));
        let g = [(h & 0x3FF) as f32 / 511.5 - 1.0, ((h >> 10) & 0x3FF) as f32 / 511.5 - 1.0, ((h >> 20) & 0x3FF) as f32 / 511.5 - 1.0];
        dot(g, d)
    };
    let (x, y, z) = (i[0] as i32, i[1] as i32, i[2] as i32);
    let mut out = 0.0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let w = [(dx, 0), (dy, 1), (dz, 2)].iter().map(|&(o, k)| if o == 1 { fade[k] } else { 1.0 - fade[k] }).product::<f32>();
                out += w * grad(x + dx, y + dy, z + dz, [f[0] - dx as f32, f[1] - dy as f32, f[2] - dz as f32]);
            }
        }
    }
    out * 1.4
}

/// Raised veins and warts, as a creature's own.
struct Detail {
    seed: u64,
    origin: [f32; 3],
    /// Vein height (world units, 0 = none) and spacing.
    veins: f32,
    spacing: f32,
    /// Share of wart cells holding a wart, and the cell size.
    warts: f32,
    cell: f32,
}

impl Detail {
    fn new(c: &Creature) -> Detail {
        let seed = c.parts.iter().fold(0x5C0_1F7u64, |s, p| hash(s ^ p.seed as u64) as u64);
        let r = |k: u64| (hash(seed ^ k << 40) & 0xFFFF) as f32 / 65535.0;
        Detail {
            seed,
            origin: c.base,
            veins: if r(1) < 0.3 { 0.0 } else { 0.8 + 2.4 * r(2) },
            spacing: 18.0 + 22.0 * r(3),
            warts: if r(4) < 0.5 { 0.0 } else { 0.05 + 0.35 * r(5) },
            cell: 10.0 + 10.0 * r(6),
        }
    }

    fn reach(&self) -> f32 {
        self.veins + self.cell * 0.2
    }

    /// Outward displacement at world point `q`.
    fn at(&self, q: [f32; 3]) -> f32 {
        let c = sub(q, self.origin);
        let mut d = 0.0;
        if self.veins > 0.0 {
            let ridge = |n: f32, w: f32| (1.0 - n.abs() / w).max(0.0).powi(2);
            let big = ridge(noise(c.map(|x| x / self.spacing), self.seed), 0.09);
            // Finer branches, only in some places.
            let patch = (noise(c.map(|x| x / 70.0), self.seed + 2) * 2.0).clamp(0.0, 1.0);
            let small = ridge(noise(c.map(|x| x / (self.spacing * 0.45)), self.seed + 1), 0.07) * patch;
            d += self.veins * big + 0.5 * self.veins * small;
        }
        if self.warts > 0.0 {
            let cell = c.map(|x| (x / self.cell).floor() as i32);
            let mut best = 0.0f32;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (x, y, z) = (cell[0] + dx, cell[1] + dy, cell[2] + dz);
                        let h = hash(self.seed ^ 0xA87 ^ (x as u32 as u64) ^ ((y as u32 as u64) << 21) ^ ((z as u32 as u64) << 42));
                        if (h & 0xFF) as f32 / 255.0 >= self.warts {
                            continue;
                        }
                        let j = |s: u32| ((h >> s) & 0xFF) as f32 / 255.0;
                        let centre = [x as f32 + j(8), y as f32 + j(16), z as f32 + j(24)].map(|v| v * self.cell);
                        let rad = self.cell * (0.15 + 0.25 * j(4));
                        let e = sub(c, centre);
                        let k = 1.0 - dot(e, e) / (rad * rad);
                        if k > 0.0 {
                            best = best.max(0.5 * rad * k * k);
                        }
                    }
                }
            }
            d += best;
        }
        d
    }
}

struct Body<'a> {
    pieces: Vec<Piece<'a>>,
    detail: Detail,
    /// Distance from the bare surface within which the sculpting is evaluated.
    band: f32,
}

impl Body<'_> {
    fn bare(&self, q: [f32; 3]) -> f32 {
        let mut d = f32::MAX;
        for p in self.pieces.iter().filter(|p| p.contains(q)) {
            let n = p.near(q).d;
            d = if d == f32::MAX { n } else { smin(d, n, p.k) };
        }
        d
    }

    fn dist(&self, q: [f32; 3]) -> f32 {
        let d = self.bare(q);
        if d.abs() < self.band {
            d - self.detail.at(q)
        } else {
            d
        }
    }

    /// Parts nearest `q`: their piece indices and blend weights, heaviest first.
    fn owners(&self, q: [f32; 3]) -> Vec<(usize, f32)> {
        let near: Vec<(usize, f32, f32)> =
            self.pieces.iter().enumerate().filter(|(_, p)| p.contains(q)).map(|(i, p)| (i, p.near(q).d, p.k)).collect();
        let dmin = near.iter().map(|n| n.1).fold(f32::MAX, f32::min);
        let mut w: Vec<(usize, f32)> = near.iter().map(|&(i, d, k)| (i, (-(d - dmin) / (0.25 * k).max(0.5)).exp())).collect();
        w.sort_by(|a, b| b.1.total_cmp(&a.1));
        w
    }
}

/// Mesh the creature's thick parts as one sculpted surface. `skins[i]` and
/// `lights[i]` are the full-colour skin and light index of part i; returns the
/// triangles and which parts were left out as too thin.
pub fn mesh(c: &Creature, placed: &[Placed], skins: &[u16], lights: &[u16]) -> (Vec<Tri>, Vec<usize>) {
    let detail = Detail::new(c);
    // Voxel size from the creature's size.
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    let boxes: Vec<([f32; 3], [f32; 3])> = c
        .parts
        .iter()
        .zip(placed)
        .map(|(p, at)| {
            let (l, h) = p.bounds();
            let (mut wl, mut wh) = ([f32::MAX; 3], [f32::MIN; 3]);
            for k in 0..8 {
                let corner = [0, 1, 2].map(|i| if k >> i & 1 == 1 { h[i] } else { l[i] });
                let w = add(at.centre, to_world(&at.axes, corner));
                for i in 0..3 {
                    wl[i] = wl[i].min(w[i]);
                    wh[i] = wh[i].max(w[i]);
                }
            }
            (wl, wh)
        })
        .collect();
    for (l, h) in &boxes {
        for i in 0..3 {
            lo[i] = lo[i].min(l[i]);
            hi[i] = hi[i].max(h[i]);
        }
    }
    let extent = (0..3).map(|i| hi[i] - lo[i]).fold(0.0, f32::max);
    let vox = (extent / 120.0).clamp(1.8, 4.0);

    let mut thin = Vec::new();
    let mut pieces = Vec::new();
    for (i, (p, at)) in c.parts.iter().zip(placed).enumerate() {
        let r = p.half[0].min(p.half[2]);
        if r < THIN * vox {
            thin.push(i);
            continue;
        }
        let k = 0.6 * r;
        let grow = k + detail.reach() + 2.0 * vox;
        pieces.push(Piece {
            part: p,
            at,
            index: i,
            lo: boxes[i].0.map(|v| v - grow),
            hi: boxes[i].1.map(|v| v + grow),
            k,
            waves: p.lump_waves(),
            widest: p.widest(),
        });
    }
    if pieces.is_empty() {
        return (Vec::new(), thin);
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &pieces {
        for i in 0..3 {
            lo[i] = lo[i].min(p.lo[i]);
            hi[i] = hi[i].max(p.hi[i]);
        }
    }
    let band = detail.reach() + 2.0 * vox;
    let body = Body { pieces, detail, band };

    // The field at grid corners.
    let n = [0, 1, 2].map(|i| ((hi[i] - lo[i]) / vox).ceil() as usize + 1);
    let corner = |x: usize, y: usize, z: usize| [lo[0] + x as f32 * vox, lo[1] + y as f32 * vox, lo[2] + z as f32 * vox];
    let at = |x: usize, y: usize, z: usize| (z * n[1] + y) * n[0] + x;
    let mut field = vec![0.0f32; n[0] * n[1] * n[2]];
    field.par_chunks_mut(n[0] * n[1]).enumerate().for_each(|(z, slice)| {
        for y in 0..n[1] {
            for x in 0..n[0] {
                slice[y * n[0] + x] = body.dist(corner(x, y, z));
            }
        }
    });

    // Surface nets: a vertex in each cell the surface passes through, at the mean of its edge crossings.
    let cells = [n[0] - 1, n[1] - 1, n[2] - 1];
    let cell_at = |x: usize, y: usize, z: usize| (z * cells[1] + y) * cells[0] + x;
    let mut vert_of = vec![u32::MAX; cells[0] * cells[1] * cells[2]];
    let mut verts: Vec<[f32; 3]> = Vec::new();
    const EDGES: [([usize; 3], [usize; 3]); 12] = [
        ([0, 0, 0], [1, 0, 0]),
        ([0, 1, 0], [1, 1, 0]),
        ([0, 0, 1], [1, 0, 1]),
        ([0, 1, 1], [1, 1, 1]),
        ([0, 0, 0], [0, 1, 0]),
        ([1, 0, 0], [1, 1, 0]),
        ([0, 0, 1], [0, 1, 1]),
        ([1, 0, 1], [1, 1, 1]),
        ([0, 0, 0], [0, 0, 1]),
        ([1, 0, 0], [1, 0, 1]),
        ([0, 1, 0], [0, 1, 1]),
        ([1, 1, 0], [1, 1, 1]),
    ];
    for z in 0..cells[2] {
        for y in 0..cells[1] {
            for x in 0..cells[0] {
                let (mut sum, mut cnt) = ([0.0; 3], 0);
                for (a, b) in EDGES {
                    let (da, db) = (field[at(x + a[0], y + a[1], z + a[2])], field[at(x + b[0], y + b[1], z + b[2])]);
                    if (da < 0.0) != (db < 0.0) {
                        let t = da / (da - db);
                        let (pa, pb) = (corner(x + a[0], y + a[1], z + a[2]), corner(x + b[0], y + b[1], z + b[2]));
                        (0..3).for_each(|i| sum[i] += pa[i] + (pb[i] - pa[i]) * t);
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    vert_of[cell_at(x, y, z)] = verts.len() as u32;
                    verts.push(sum.map(|s| s / cnt as f32));
                }
            }
        }
    }
    // A quad across every grid edge the surface crosses, joining the four cells around it.
    let mut quads: Vec<[u32; 4]> = Vec::new();
    for z in 0..n[2] {
        for y in 0..n[1] {
            for x in 0..n[0] {
                let d0 = field[at(x, y, z)];
                for axis in 0..3 {
                    let mut e = [x, y, z];
                    e[axis] += 1;
                    if e[axis] >= n[axis] {
                        continue;
                    }
                    if (d0 < 0.0) == (field[at(e[0], e[1], e[2])] < 0.0) {
                        continue;
                    }
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    let mut q = [0u32; 4];
                    let mut ok = true;
                    for (k, (du, dv)) in [(0, 0), (1, 0), (1, 1), (0, 1)].into_iter().enumerate() {
                        let mut cpos = [x as isize, y as isize, z as isize];
                        cpos[u] -= du;
                        cpos[v] -= dv;
                        if (0..3).any(|i| cpos[i] < 0 || cpos[i] >= cells[i] as isize) {
                            ok = false;
                            break;
                        }
                        let id = vert_of[cell_at(cpos[0] as usize, cpos[1] as usize, cpos[2] as usize)];
                        if id == u32::MAX {
                            ok = false;
                            break;
                        }
                        q[k] = id;
                    }
                    if ok {
                        quads.push(q);
                    }
                }
            }
        }
    }

    // Normals from the field, owners from the parts.
    let e = 0.5 * vox;
    let normals: Vec<[f32; 3]> = verts
        .par_iter()
        .map(|&p| {
            let g = |i: usize| {
                let (mut a, mut b) = (p, p);
                a[i] += e;
                b[i] -= e;
                body.dist(a) - body.dist(b)
            };
            norm([g(0), g(1), g(2)])
        })
        .collect();
    let owners: Vec<Vec<(usize, f32)>> = verts.par_iter().map(|&p| body.owners(p)).collect();

    let mut tris = Vec::with_capacity(quads.len() * 2);
    for q in quads {
        let (p0, p1, p2, p3) = (verts[q[0] as usize], verts[q[1] as usize], verts[q[2] as usize], verts[q[3] as usize]);
        let split02 = dot(sub(p0, p2), sub(p0, p2)) <= dot(sub(p1, p3), sub(p1, p3));
        let pair = if split02 { [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] } else { [[q[0], q[1], q[3]], [q[1], q[2], q[3]]] };
        for t in pair {
            if let Some(tri) = triangle(&body, c, t, &verts, &normals, &owners, skins, lights) {
                tris.push(tri);
            }
        }
    }
    (tris, thin)
}

#[allow(clippy::too_many_arguments)]
fn triangle(
    body: &Body,
    c: &Creature,
    t: [u32; 3],
    verts: &[[f32; 3]],
    normals: &[[f32; 3]],
    owners: &[Vec<(usize, f32)>],
    skins: &[u16],
    lights: &[u16],
) -> Option<Tri> {
    let p = t.map(|i| verts[i as usize]);
    let cr = {
        let (a, b) = (sub(p[1], p[0]), sub(p[2], p[0]));
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    if dot(cr, cr) < 1e-8 {
        return None;
    }
    // The two parts with the most weight over the triangle.
    let mut total: Vec<(usize, f32)> = Vec::new();
    for &i in &t {
        for &(piece, w) in &owners[i as usize] {
            match total.iter_mut().find(|e| e.0 == piece) {
                Some(e) => e.1 += w,
                None => total.push((piece, w)),
            }
        }
    }
    total.sort_by(|a, b| b.1.total_cmp(&a.1));
    let a = total.first()?.0;
    let b = total.get(1).map_or(a, |e| e.0);
    let weight = |v: u32, piece: usize| owners[v as usize].iter().find(|e| e.0 == piece).map_or(0.0, |e| e.1);
    let pieces = [&body.pieces[a], &body.pieces[b]];
    let reps = pieces.map(|pc| reps(pc.part, c.skins[pc.part.skin].scale));
    // Skin coordinates per part, with u unwrapped so the triangle does not span the seam.
    let frames = [0, 1].map(|s| {
        let scale = c.skins[pieces[s].part.skin].scale;
        let mut f = p.map(|q| pieces[s].skin_frame(&pieces[s].near(q), reps[s], scale));
        let u0 = f[0].0[0];
        for k in 1..3 {
            let d = f[k].0[0] - u0;
            f[k].0[0] -= (d / reps[s][0]).round() * reps[s][0];
        }
        f
    });
    let v = [0, 1, 2].map(|k| {
        let (wa, wb) = (weight(t[k], a), if b == a { 0.0 } else { weight(t[k], b) });
        Vertex {
            p: p[k],
            n: normals[t[k] as usize],
            w: if wa + wb > 0.0 { wa / (wa + wb) } else { 0.5 },
            uv: [frames[0][k].0, frames[1][k].0],
            t: [frames[0][k].1, frames[1][k].1],
            b: [frames[0][k].2, frames[1][k].2],
            cap: [frames[0][k].3, frames[1][k].3],
            wrap: [frames[0][k].4, frames[1][k].4],
        }
    });
    let idx = [pieces[0].index, pieces[1].index];
    let cap_axes = pieces.map(|pc| [0, 2].map(|c| [pc.at.axes[0][c], pc.at.axes[1][c], pc.at.axes[2][c]]));
    Some(Tri { v, skin: idx.map(|i| skins[i]), light: idx.map(|i| lights[i]), cap_axes })
}
