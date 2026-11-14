//! What the per-pixel stage of a sculpt-trained model is told about each pixel, as far as the
//! CPU goes: a port of train/gbuffer.py's `project` (the level's triangles split, cut at the
//! near plane and snapped to whole pixels, as the renderer was when the stage was trained) and
//! of its binning of pieces into 16x16 tiles. The rest of train/pixels.py (which piece each
//! pixel shows, and all it reads from that) runs on the GPU (gpu_net.wgsl `stage_inputs`).
//! Keep the two in step; the pieces come in gbuffer.py's order, which settles ties.

use crate::neural::features::TRI_FEATURES as RAW;
use crate::world::dreamed::TURNING_FEATURES;
use crate::world::view::H;

/// Tiles: render::MAX_EDGE as it was (gbuffer.py SPLIT).
const SPLIT: f64 = 0.25;
/// Tiles: the level camera's near plane.
const NEAR: f64 = 0.02;
pub const TILE: usize = 16;
/// Floats per piece: screen corners (6), 1/z at them (3), its row, its corners' weights on
/// the triangle (9).
pub const PIECE: usize = 19;
/// Floats per row (a triangle some piece is of): corners (9), texture coordinates (6), texture,
/// turning (15).
pub const ROW: usize = 9 + 6 + 1 + TURNING_FEATURES;

type V3 = [f64; 3];

/// One view's pieces, ready for the GPU.
pub struct Pieces {
    /// [pieces, PIECE].
    pub pieces: Vec<f32>,
    /// [rows, ROW].
    pub rows: Vec<f32>,
    /// Per tile (row-major, width / 16 across): where its list starts in `lists`, and how long it is.
    pub tiles: Vec<f32>,
    /// The pieces each tile may show, tile after tile.
    pub lists: Vec<f32>,
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mid(a: V3, b: V3) -> V3 {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0]
}

fn length(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// A piece: corners, the triangle it is of, and its corners' weights on that triangle.
#[derive(Clone, Copy)]
struct Piece {
    p: [V3; 3],
    tri: usize,
    w: [V3; 3],
}

/// gbuffer.py `split`: the triangles halved at their edges' middles until no edge is longer
/// than SPLIT, round after round, those done each round first, then each kind of quarter.
fn split(tris: Vec<Piece>) -> Vec<Piece> {
    let mut todo = tris;
    let mut done = Vec::new();
    while !todo.is_empty() {
        let mut next = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for t in todo {
            let edge = (0..3).map(|k| length(sub(t.p[k], t.p[(k + 1) % 3]))).fold(0.0, f64::max);
            if edge <= SPLIT {
                done.push(t);
                continue;
            }
            let [a, b, c] = t.p;
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            let [wa, wb, wc] = t.w;
            let (wab, wbc, wca) = (mid(wa, wb), mid(wb, wc), mid(wc, wa));
            next[0].push(Piece { p: [a, ab, ca], w: [wa, wab, wca], ..t });
            next[1].push(Piece { p: [ab, b, bc], w: [wab, wb, wbc], ..t });
            next[2].push(Piece { p: [ca, bc, c], w: [wca, wbc, wc], ..t });
            next[3].push(Piece { p: [ab, bc, ca], w: [wab, wbc, wca], ..t });
        }
        todo = next.concat();
    }
    done
}

/// gbuffer.py `clip_near`: pieces (in view space) cut at z = NEAR, those wholly in front
/// first, then the two halves of those with one corner behind, then those with two.
fn clip_near(pieces: Vec<Piece>) -> Vec<Piece> {
    let behind = |q: &Piece| q.p.map(|c| c[2] < NEAR);
    let count = |q: &Piece| behind(q).iter().filter(|&&b| b).count();
    let mut out: Vec<Piece> = pieces.iter().filter(|q| count(q) == 0).copied().collect();
    for k in [1, 2] {
        let sel: Vec<Piece> = pieces.iter().filter(|q| count(q) == k).copied().collect();
        let rolled: Vec<Piece> = sel
            .iter()
            .map(|q| {
                let b = behind(q);
                let odd = (0..3).find(|&i| if k == 1 { b[i] } else { !b[i] }).unwrap();
                let r = |i: usize| (odd + i) % 3;
                Piece { p: [q.p[r(0)], q.p[r(1)], q.p[r(2)]], w: [q.w[r(0)], q.w[r(1)], q.w[r(2)]], tri: q.tri }
            })
            .collect();
        let cut = |q: &Piece, i: usize, j: usize| {
            let t = (NEAR - q.p[i][2]) / (q.p[j][2] - q.p[i][2]);
            let lerp = |a: V3, b: V3| [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * t);
            (lerp(q.p[i], q.p[j]), lerp(q.w[i], q.w[j]))
        };
        if k == 1 {
            // A quad in front: (x1, b, c) and (x1, c, x2), all of the first, then all of the second.
            for half in 0..2 {
                for q in &rolled {
                    let ((x1, w1), (x2, w2)) = (cut(q, 0, 1), cut(q, 0, 2));
                    out.push(if half == 0 {
                        Piece { p: [x1, q.p[1], q.p[2]], w: [w1, q.w[1], q.w[2]], tri: q.tri }
                    } else {
                        Piece { p: [x1, q.p[2], x2], w: [w1, q.w[2], w2], tri: q.tri }
                    });
                }
            }
        } else {
            for q in &rolled {
                let ((x1, w1), (x2, w2)) = (cut(q, 0, 1), cut(q, 0, 2));
                out.push(Piece { p: [q.p[0], x1, x2], w: [q.w[0], w1, w2], tri: q.tri });
            }
        }
    }
    out
}

/// The pieces of the raw triangles `tris` (features::TRI_FEATURES a triangle, positions in
/// tiles; `turning` alongside, TURNING_FEATURES each, or none) seen from view `v`
/// (`views.npy` row), binned into tiles of a frame `v[26]` wide.
pub fn pieces(tris: &[f32], turning: &[f32], v: &[f32]) -> Pieces {
    let width = v[26] as usize;
    let pos = [v[1], v[2], v[3]].map(|c| c as f64);
    let r = [[v[4], v[5], v[6]], [v[7], v[8], v[9]], [v[10], v[11], v[12]]].map(|row| row.map(|c| c as f64));
    let h = v[13] as f64;
    let fog_far = v[15] as f64;
    let view = |p: V3| {
        let d = sub(p, pos);
        [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2])
    };
    let (kx, ky) = (width as f64 / 2.0 / h * 1.05, 120.0 / h * 1.05);
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut seen = Vec::new();
    for (i, t) in tris.chunks_exact(RAW).enumerate() {
        let world = [0, 1, 2].map(|j| [0, 1, 2].map(|c| t[3 * j + c] as f64));
        let tv = world.map(view);
        let (e1, e2) = (sub(tv[1], tv[0]), sub(tv[2], tv[0]));
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let front = -(n[0] * tv[0][0] + n[1] * tv[0][1] + n[2] * tv[0][2]) > 0.0;
        let z = tv.map(|q| q[2]);
        let s = tv.map(|q| {
            let zc = q[2].max(1e-3);
            (q[0] / zc, q[1] / zc, q[2] > 0.0)
        });
        let all = |f: &dyn Fn((f64, f64, bool)) -> bool| s.iter().all(|&e| f(e));
        let outside = all(&|(x, _, a)| x > kx && a) || all(&|(x, _, a)| x < -kx && a) || all(&|(_, y, a)| y > ky && a) || all(&|(_, y, a)| y < -ky && a);
        if front && z.iter().any(|&z| z > NEAR) && z[0].min(z[1]).min(z[2]) < fog_far && !outside {
            seen.push(Piece { p: world, tri: i, w: eye });
        }
    }
    let pieces = clip_near(split(seen).into_iter().map(|q| Piece { p: q.p.map(view), ..q }).collect());

    // The rows the pieces are of, in the order of their triangles (np.unique).
    let mut used: Vec<usize> = pieces.iter().map(|q| q.tri).collect();
    used.sort_unstable();
    used.dedup();
    let mut rows = Vec::with_capacity(used.len() * ROW);
    for &i in &used {
        let t = &tris[i * RAW..(i + 1) * RAW];
        rows.extend_from_slice(&t[..16]);
        match turning.get(i * TURNING_FEATURES..(i + 1) * TURNING_FEATURES) {
            Some(turn) => rows.extend_from_slice(turn),
            None => rows.extend([0.0; TURNING_FEATURES]),
        }
    }

    let (gw, gh) = (width / TILE, H as usize / TILE);
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); gw * gh];
    let mut out = Vec::with_capacity(pieces.len() * PIECE);
    for (k, q) in pieces.iter().enumerate() {
        // Corners snap to whole pixels (np.round: halves to even).
        let screen = q.p.map(|c| [width as f64 / 2.0 + h * c[0] / c[2], 120.0 + h * c[1] / c[2]].map(f64::round_ties_even));
        let s32 = screen.map(|c| c.map(|x| x as f32));
        out.extend(s32.iter().flatten());
        out.extend(q.p.map(|c| (1.0 / c[2]) as f32));
        out.push(used.binary_search(&q.tri).unwrap() as f32);
        out.extend(q.w.iter().flatten().map(|&x| x as f32));
        // Its tiles: those its box of pixel centres touches (in f32, as gbuffer.py).
        let [a, b, c] = s32;
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        let lo = [0, 1].map(|i| (a[i].min(b[i]).min(c[i]) - 0.5).ceil().max(0.0));
        let hi = [0, 1].map(|i| (a[i].max(b[i]).max(c[i]) - 0.5).floor());
        let hi = [hi[0].min(width as f32 - 1.0), hi[1].min(H as f32 - 1.0)];
        if area.abs() <= 1e-6 || hi[0] < lo[0] || hi[1] < lo[1] {
            continue;
        }
        let (t0, t1) = (lo.map(|x| x as usize / TILE), hi.map(|x| x as usize / TILE));
        for ty in t0[1]..=t1[1] {
            for tx in t0[0]..=t1[0] {
                bins[ty * gw + tx].push(k as u32);
            }
        }
    }
    let mut tiles = Vec::with_capacity(gw * gh * 2);
    let mut lists = Vec::new();
    for b in &bins {
        tiles.extend([lists.len() as f32, b.len() as f32]);
        lists.extend(b.iter().map(|&k| k as f32));
    }
    Pieces { pieces: out, rows, tiles, lists }
}
