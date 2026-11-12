//! The level as the level net could be taught to dream it: patches of wall turned to folded
//! flesh, with eyes that follow the viewer and mouths in it. Only ever drawn, as a target for
//! the net to match; what the net is told about the level stays the plain triangles.

use crate::world::creature::{add, cross, scale, sub, Creature, Part, Placed, Shape, Skin};
use crate::world::flesh;
use crate::world::photo::read_rgb;
use crate::world::rng::{hash, Rng};
use crate::world::scene::{dot, norm, Camera, Scene, Tri, Vert, T};
use crate::world::sculpt::noise;
use crate::world::skin::Motif;
use crate::world::tex;
use std::collections::HashMap;
use std::f32::consts::TAU;
use std::path::Path;

const TF: f32 = T as f32;
/// Each flesh triangle is cut into N * N.
const N: i32 = 16;
/// How far the folds stand out of or sink into the wall at most (tiles).
const DEPTH: f32 = 0.3;
/// How far all flesh stands out before its folds (tiles), so that none of it sinks behind
/// the stone it grows over, and how softly the deepest creases meet that.
const LIFT: f32 = 0.16;
const LIFT_SOFT: f32 = 0.05;
/// Where along the dial at a place flesh creeps in, and where the wall has finished sagging.
const CREEP: (f32, f32) = (0.25, 0.8);
const SAG: (f32, f32) = (0.35, 0.85);
/// Where along the dial veins run out, and how far they have got by the end (tiles).
const VEINS_RUN: (f32, f32) = (0.6, 0.85);
const VEIN_LENGTH: f32 = 6.5;
/// How far a flesh texture's own relief stands out of the folds at most (tiles).
const RELIEF: f32 = 0.17;
/// How many tiles one repeat of a flesh texture covers.
const SPAN: f32 = 2.5;
/// How far a vein rides above the surface, so that the relief doesn't swallow it (tiles).
const VEIN_LIFT: f32 = 0.05;
/// Level textures with a flesh end free of neon and painted eyes, for those without one.
const CLEAN: [(u8, u8); 4] = [(3, 0), (4, 1), (5, 1), (7, 6)];
/// Seconds between blinks, and how long one takes.
const BLINK_EVERY: f32 = 5.0;
const BLINK: f32 = 0.3;
/// How far from the wall's facing an eye can turn, in radians.
const GAZE: f32 = 1.0;
/// How many frames a second the blink in the footage was filmed at.
const FOOTAGE_RATE: f32 = 30.0;
/// The real eyes' gazes on each side of a filmed frame (see `photo/cut_eye.py`).
const GAZES_X: usize = 7;
const GAZES_Y: usize = 5;
/// How far the iris moves between filmed gazes, as the sine of the eye's turn: its steps
/// (16 and 13 pixels) over the radius of the eyeball, 75 pixels in the same footage.
const GAZE_STEP: [f32; 2] = [16.0 / 75.0, 13.0 / 75.0];
/// The first of the filmed frames with the lid on its way back up (see `photo/cut_eye.py`).
const FIRST_OPENING: usize = 6;
/// Corners across a real eye's patch of wall.
const PATCH: usize = 24;
/// How far apart filmed eyes are, at least (tiles).
const EYES_APART: f32 = 2.5;
/// How dry the features look: skin rather than something just licked.
const MATTE: f32 = 0.85;
/// Seconds between heartbeats, and how far apart (tiles) they run along a vein.
const BEAT_EVERY: f32 = 1.15;
const BEAT_WAVE: f32 = 3.0;
/// How much a beat fattens a vein.
const BEAT_SWELL: f32 = 0.8;
/// Walls to one free vein, in points of the cut surface.
const VEIN_EVERY: usize = 4000;
/// How often a vein leaving an eye has a branch running into it as well.
const EYE_VEINS: f32 = 0.75;
/// At most this many of those.
const FREE_VEINS: usize = 20;
/// How far from the viewer the veins and sphincters are drawn at all (tiles).
const REACH: f32 = 8.0;
/// Floats per level triangle that `turning_features` gives the net, and per organ in `organs`.
pub const TURNING_FEATURES: usize = 15;
pub const ORGAN_FEATURES: usize = 19;
/// At most this many organs are told to the net for one view.
pub const MAX_ORGANS: usize = 128;
/// How many points of a vein one organ token covers: half a tile.
const VEIN_TOKEN: usize = 4;
/// Each lid's side (1 the top), and how far down the eye it comes open and shut (see `lid`).
const LIDS: [(f32, f32, f32); 2] = [(1.0, 0.9, 1.72), (-1.0, 1.05, 1.42)];
/// How a corner of `Moving` moves, as the flesh shader knows it: with its eyeball, as a lid,
/// a vein, a sphincter, or a filmed eye.
const BALL: f32 = 1.0;
const LID: f32 = 2.0;
const VEIN: f32 = 3.0;
const MOUTH: f32 = 4.0;
const FILMED: f32 = 5.0;

fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// An eye in a wall, redrawn each frame to look at the viewer.
struct Eye {
    /// The middle of the socket on the surface, the wall's facing and across and up the wall.
    at: [f32; 3],
    n: [f32; 3],
    across: [f32; 3],
    up: [f32; 3],
    r: f32,
    /// Its skin and light in `Scene::flesh`.
    skin: u16,
    light: u16,
    phase: f32,
    /// Filmed, rather than a ball with lids.
    patch: Option<Patch>,
    /// Where on the dial it starts to open.
    due: f32,
}

/// The flesh textures as skins with their heights (see `photo/height_flesh.py`), to draw the
/// walls in the smooth pass and raise them to match. Slots are the level's.
struct Painted {
    skins: [u16; tex::STRANGE_SLOTS],
    heights: Vec<Vec<f32>>,
}

impl Painted {
    /// None when `dir` has no height maps.
    fn load(scene: &mut Scene, dir: &Path) -> Option<Painted> {
        let mut heights = Vec::new();
        let mut skins = [0; tex::STRANGE_SLOTS];
        for (slot, skin) in skins.iter_mut().enumerate() {
            let h = read_rgb(&dir.join(format!("{slot}_h.png")))?;
            heights.push(h.iter().map(|c| c[0] / 127.5 - 1.0).collect());
            let rgb = read_rgb(&dir.join(format!("{slot}.png")))?;
            *skin = scene.flesh.skins.len() as u16;
            scene.flesh.skins.push(flesh::Skin::from_rgb(256, rgb, [1.0; 3]));
        }
        Some(Painted { skins, heights })
    }

    /// How raised texture `slot` is at `uv` (in repeats), -1..1.
    fn height(&self, slot: usize, uv: [f32; 2]) -> f32 {
        const S: usize = 256;
        let (x, y) = (uv[0].rem_euclid(1.0) * S as f32, uv[1].rem_euclid(1.0) * S as f32);
        let (ix, iy) = (x as usize % S, y as usize % S);
        let (fx, fy) = (x.fract(), y.fract());
        let at = |i: usize, j: usize| self.heights[slot][(j % S) * S + i % S];
        let top = at(ix, iy) * (1.0 - fx) + at(ix + 1, iy) * fx;
        let bottom = at(ix, iy + 1) * (1.0 - fx) + at(ix + 1, iy + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// Where a point falls on the flesh textures of a surface facing `n`, in repeats: laid flat
/// from whichever side the surface faces most, so that they run on across tiles.
fn laid_on(p: [f32; 3], n: [f32; 3]) -> [f32; 2] {
    let (u, v) = if n[1].abs() > 0.7 { (p[0], p[2]) } else if n[0].abs() > n[2].abs() { (p[2], p[1]) } else { (p[0], p[1]) };
    [u / (SPAN * TF), v / (SPAN * TF)]
}

/// The world directions `laid_on`'s two coordinates increase along.
fn laid_axes(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    if n[1].abs() > 0.7 {
        (x, z)
    } else if n[0].abs() > n[2].abs() {
        (z, y)
    } else {
        (x, y)
    }
}

/// A 256-square image at the size skins are.
fn halved(rgb: Vec<[f32; 3]>) -> Vec<[f32; 3]> {
    let s = flesh::SKIN;
    (0..s * s)
        .map(|i| {
            let (x, y) = (i % s * 2, i / s * 2);
            let q = [rgb[y * 2 * s + x], rgb[y * 2 * s + x + 1], rgb[(y + 1) * 2 * s + x], rgb[(y + 1) * 2 * s + x + 1]];
            [0, 1, 2].map(|c| q.iter().map(|t| t[c]).sum::<f32>() / 4.0)
        })
        .collect()
}

/// A filmed eye's skins: [frame][gaze x * GAZES_Y + gaze y], frame 0 open, then the blink.
struct Footage {
    skins: Vec<Vec<u16>>,
    /// The flesh a patch's rim fades to.
    rim: u16,
}

impl Footage {
    /// Read `dir` (see `photo/cut_eye.py`) into `scene`'s skins.
    fn load(scene: &mut Scene, dir: &Path, rim: u16) -> Footage {
        let mut skins = Vec::new();
        loop {
            let f = skins.len();
            if !dir.join(format!("{f}_0_0.png")).exists() {
                break;
            }
            let mut frame = Vec::new();
            for x in 0..GAZES_X {
                for y in 0..GAZES_Y {
                    // The blink was only cut at every other gaze.
                    if f > 0 && (x % 2 == 1 || y % 2 == 1) {
                        frame.push(frame[(x & !1) * GAZES_Y + (y & !1)]);
                        continue;
                    }
                    let path = dir.join(format!("{f}_{x}_{y}.png"));
                    let rgb = read_rgb(&path).unwrap_or_else(|| panic!("can't read {}", path.display()));
                    frame.push(scene.flesh.skins.len() as u16);
                    scene.flesh.skins.push(flesh::Skin::from_rgb(flesh::SKIN, halved(rgb), [1.0; 3]));
                }
            }
            skins.push(frame);
        }
        assert!(skins.len() > 1, "no eye frames in {}", dir.display());
        Footage { skins, rim }
    }
}

/// A sphincter: its triangles relaxed, and how they clench.
struct Mouth {
    tris: Vec<flesh::Tri>,
    at: [f32; 3],
    n: [f32; 3],
    across: [f32; 3],
    up: [f32; 3],
    /// Half the opening's width and height, relaxed.
    size: [f32; 2],
    phase: f32,
    every: f32,
    /// Where on the dial it finishes coming out.
    due: f32,
}

/// A real eye's patch of wall: its corners with what they face and how much of the eye shows.
struct Patch {
    corners: Vec<PatchCorner>,
    /// Half of them face the other way round in the image.
    mirrored: bool,
    /// What's round the eye: the wall's own texture when it has one, and that way up.
    wall: Option<u16>,
    axes: ([f32; 3], [f32; 3]),
}

#[derive(Clone, Copy)]
struct PatchCorner {
    /// On the stone, and how far out along the wall's facing it goes.
    flat: [f32; 3],
    rise: f32,
    n: [f32; 3],
    /// How much is the eye and not the wall.
    w: f32,
    wall_uv: [f32; 2],
    light: [[f32; 3]; 4],
}

/// A point down the middle of a vein, on the folded surface.
#[derive(Clone, Copy)]
struct VeinPoint {
    p: [f32; 3],
    n: [f32; 3],
    r: f32,
    /// How far along from where the blood comes in (tiles).
    along: f32,
    /// How far the surface under it moved from the stone, and how much of the dial reaches it.
    rise: [f32; 3],
    spread: f32,
}

/// A vein or a branch of one, with the light it was given.
struct Vein {
    points: Vec<VeinPoint>,
    light: u16,
}

/// A point of the cut surface, by where it was before the folds moved it.
type Key = [i32; 3];

/// The folded flesh as something to walk over: where its points went and what meets them.
struct Surface {
    links: HashMap<Key, Vec<Key>>,
    at: HashMap<Key, ([f32; 3], [f32; 3])>,
    /// Round each of these, the veins go elsewhere.
    keep_off: Vec<([f32; 3], f32)>,
}

/// How far between `from` and `to` the dial `x` has got, eased.
fn stage(x: f32, (from, to): (f32, f32)) -> f32 {
    smooth((x - from) / (to - from))
}

/// How much of its bulk flesh on a wall facing `n` keeps, for the space in front of it and
/// the rock behind it: all of it on a long wall of a room, under half of it in a corridor a
/// tile across, and less again on a pillar, which swells from every side at once.
fn room_for(scene: &Scene, p: [f32; 3], n: [f32; 3]) -> f32 {
    const FAR: f32 = 3.0;
    const THICK: f32 = 2.0;
    if n[1].abs() > 0.7 {
        return 1.0;
    }
    let open = |t: f32| {
        let q = add(p, scale(n, t * TF));
        scene.node_at(q[0], q[1], q[2]).is_some()
    };
    // Points at the very top or bottom of a wall have nothing to measure from.
    if !open(0.2) {
        return 1.0;
    }
    let free = (1..).map(|i| i as f32 * 0.25).take_while(|&t| t <= FAR && open(t)).last().unwrap_or(0.2);
    // Open again close behind it means a pillar, or a prop standing in the open.
    let thickness = (0..).map(|i| 0.1 + i as f32 * 0.25).take_while(|&t| t <= THICK).find(|&t| open(-t)).unwrap_or(THICK);
    (0.4 + 0.6 * smooth((free - 0.75) / (FAR - 0.75))) * (0.3 + 0.7 * smooth((thickness - 0.5) / (THICK - 0.5)))
}

/// How much of the dial reaches a place of strangeness `s`, 0..1.
fn spread_of(s: f32) -> f32 {
    ((s - 0.47) / 0.18).clamp(0.0, 1.0)
}

/// A point of the flesh walls, as it is on the stone and how far it goes from there.
struct WallPoint {
    base: [f32; 3],
    rise: [f32; 3],
    facing: [f32; 3],
    bent: [f32; 3],
    spread: f32,
    shadow: f32,
    light: [[f32; 3]; 4],
}

struct WallTri {
    points: [u32; 3],
    skin: u16,
    tint: [f32; 3],
    along: [f32; 3],
    down: [f32; 3],
    uv: [[f32; 2]; 3],
}

/// What `sculpt` added that moves.
pub struct Dreamed {
    eyes: Vec<Eye>,
    mouths: Vec<Mouth>,
    veins: Vec<Vein>,
    points: Vec<WallPoint>,
    walls: Vec<WallTri>,
    neutral: u16,
    /// How far the level has turned: 0 is the plain stone and 1 all of this.
    pub amount: f32,
    /// Skins of a vein at rest and with a beat going through it.
    vein_skin: [u16; 2],
    footage: Option<Footage>,
    /// Per level triangle, in `Scene::tris` order: see `turning_features`.
    turning: Vec<[f32; TURNING_FEATURES]>,
    /// How many of the flesh pass's triangles don't move.
    pub fixed: usize,
    lid: u16,
}

/// How the folds of flesh stand at `p` (tiles), about -1..1. They are broad, a couple of
/// tiles from crease to crease, with the creases rounded: the net is told their height at
/// each level triangle's corners and the middles of its edges (`turning_features`), and
/// anything finer than that it couldn't place, only blur. The fine relief is the painted
/// flesh's own (`RELIEF`), which it can learn from where the texture lies.
fn folds(p: [f32; 3], seed: u64) -> f32 {
    let bend = [noise(p.map(|c| c * 0.3), seed ^ 1), noise(p.map(|c| c * 0.3 + 9.0), seed ^ 2), noise(p.map(|c| c * 0.3 + 17.0), seed ^ 3)];
    let q = [0, 1, 2].map(|c| p[c] * 0.45 + bend[c] * 1.2);
    let n = noise(q, seed ^ 4);
    let ridges = 1.0 - 2.5 * (n * n + 0.0025).sqrt();
    0.75 * ridges + 0.5 * noise(p.map(|c| c * 0.25), seed ^ 5)
}

/// Triangles split wherever a corner of another lies partway along one of their edges, so
/// that neighbours share each edge whole. `cut` then makes the same points along it on both
/// sides, and the folded sheet over them doesn't crack open there.
fn split_at_junctions(tris: Vec<Tri>) -> Vec<Tri> {
    // However the corners lie, one triangle doesn't become more pieces than this.
    const MAX_PIECES: usize = 4096;
    let mut corners: Vec<[i32; 3]> = tris.iter().flat_map(|t| t.v.map(|v| v.p)).collect();
    corners.sort_unstable();
    corners.dedup();
    // A corner strictly between `a` and `b`, on the line through them. (Only exactly: allowing
    // the corners of sloping edges that are rounded a unit off it ran away splitting.)
    let along = |a: [i32; 3], b: [i32; 3]| {
        let (lo, hi) = (a[0].min(b[0]), a[0].max(b[0]));
        let from = corners.partition_point(|c| c[0] < lo);
        let to = corners.partition_point(|c| c[0] <= hi);
        let ab = [0, 1, 2].map(|k| (b[k] - a[k]) as i64);
        let len = ab.iter().map(|x| x * x).sum::<i64>();
        corners[from..to].iter().copied().find(|&c| {
            let ac = [0, 1, 2].map(|k| (c[k] - a[k]) as i64);
            let d = (0..3).map(|k| ac[k] * ab[k]).sum::<i64>();
            let straight = ab[1] * ac[2] == ab[2] * ac[1] && ab[2] * ac[0] == ab[0] * ac[2] && ab[0] * ac[1] == ab[1] * ac[0];
            straight && d > 0 && d < len
        })
    };
    let mut out = Vec::with_capacity(tris.len());
    let mut todo = Vec::new();
    for t in tris {
        todo.push(t);
        let mut pieces = 1;
        while let Some(t) = todo.pop() {
            let found = if pieces < MAX_PIECES { (0..3).find_map(|i| along(t.v[i].p, t.v[(i + 1) % 3].p).map(|p| (i, p))) } else { None };
            let Some((i, p)) = found else {
                out.push(t);
                continue;
            };
            pieces += 1;
            let (a, b) = (t.v[i], t.v[(i + 1) % 3]);
            let f = (0..3).map(|k| (p[k] - a.p[k]) as f32).map(|x| x * x).sum::<f32>().sqrt()
                / (0..3).map(|k| (b.p[k] - a.p[k]) as f32).map(|x| x * x).sum::<f32>().sqrt();
            let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * f).round() as u8;
            let m = Vert { p, uv: [0, 1].map(|k| mix(a.uv[k], b.uv[k])), col: [0, 1, 2].map(|k| mix(a.col[k], b.col[k])), alb: [0, 1, 2].map(|k| mix(a.alb[k], b.alb[k])) };
            // Both halves keep the winding: the middle stands in for one end of the edge each.
            let (mut first, mut second) = (t, t);
            first.v[(i + 1) % 3] = m;
            second.v[i] = m;
            todo.push(second);
            todo.push(first);
        }
    }
    out
}

/// How much of a place at `p` (world units) has turned, 0 at the edge of the patch.
fn turned(scene: &Scene, p: [f32; 3]) -> f32 {
    smooth((scene.strangeness(p) - 0.5) / 0.25)
}

fn cut(t: &Tri) -> Vec<Tri> {
    let [a, b, c] = t.v;
    let vert = |i: i32, j: i32| {
        let k = N - i - j;
        let mix = |x: i32, y: i32, z: i32| (x * k + y * i + z * j) / N;
        let byte = |x: u8, y: u8, z: u8| mix(x as i32, y as i32, z as i32) as u8;
        Vert {
            p: [0, 1, 2].map(|d| mix(a.p[d], b.p[d], c.p[d])),
            uv: [0, 1].map(|d| byte(a.uv[d], b.uv[d], c.uv[d])),
            col: [0, 1, 2].map(|d| byte(a.col[d], b.col[d], c.col[d])),
            alb: [0, 1, 2].map(|d| byte(a.alb[d], b.alb[d], c.alb[d])),
        }
    };
    let mut out = Vec::with_capacity((N * N) as usize);
    for j in 0..N {
        for i in 0..N - j {
            out.push(Tri { v: [vert(i, j), vert(i + 1, j), vert(i, j + 1)], ..*t });
            if i + j < N - 1 {
                out.push(Tri { v: [vert(i + 1, j), vert(i + 1, j + 1), vert(i, j + 1)], ..*t });
            }
        }
    }
    out
}

/// An eye's image with the pupil at the corners: what shows through the pole of a sphere.
fn eye_image(seed: u32) -> Vec<[f32; 3]> {
    let size = flesh::SKIN;
    let (hue, deep) = ([170.0, 130.0, 50.0], [60.0, 30.0, 15.0]);
    (0..size * size)
        .map(|i| {
            let wrap = |v: usize| {
                let v = v as f32 / size as f32;
                if v > 0.5 { v - 1.0 } else { v }
            };
            let (x, y) = (wrap(i % size), wrap(i / size));
            let d = x.hypot(y);
            let angle = y.atan2(x);
            // The pupil is a little narrower than it is tall.
            let pupil = (x * 1.5).hypot(y * 0.8);
            if pupil < 0.075 {
                return [6.0, 4.0, 4.0];
            }
            if d < 0.2 {
                let h = hash(((angle / TAU * 64.0).floor() as i64 as u64) ^ seed as u64);
                let k = (d - 0.07) / 0.13 + (h & 0xFF) as f32 / 255.0 * 0.35;
                let dark = 1.0 - smooth((d - 0.17) / 0.03);
                return [0, 1, 2].map(|c| (deep[c] + (hue[c] - deep[c]) * k.clamp(0.0, 1.0)) * (0.35 + 0.65 * dark));
            }
            // Bloodshot towards the edge.
            let h = hash(((angle / TAU * 90.0).floor() as i64 as u64) ^ (seed as u64 + 7));
            let vein = if h & 7 == 0 { smooth((d - 0.24) / 0.2) } else { 0.0 };
            let white = [150.0, 135.0, 112.0];
            let blood = [120.0, 25.0, 25.0];
            [0, 1, 2].map(|c| white[c] + (blood[c] - white[c]) * vein)
        })
        .collect()
}

fn part(shape: Shape, half: [f32; 3], skin: usize) -> Part {
    Part {
        parent: None,
        anchor: [0.0; 3],
        rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pivot: [0.0; 3],
        half,
        shape,
        taper: 1.0,
        bend: 0.0,
        bulge: 0.0,
        lumps: 0.0,
        skin,
        seed: 0,
    }
}

fn columns(x: [f32; 3], y: [f32; 3], z: [f32; 3]) -> [[f32; 3]; 3] {
    [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
}

fn skin_of(motif: Motif, density: f32, tint: [f32; 3], seed: u32) -> Skin {
    Skin { motif, density, scale: 0.5 * TF, tint, seed }
}

/// A mouth of radius `r` at `at` on a wall facing `n`: lips, a dark throat and teeth.
fn mouth(at: [f32; 3], n: [f32; 3], across: [f32; 3], up: [f32; 3], r: f32, rng: &mut Rng) -> (Creature, f32, f32) {
    let skins = vec![
        skin_of(Motif::Flesh, 0.5, [0.75, 0.42, 0.42], rng.next()),
        skin_of(Motif::Flesh, 0.5, [0.03, 0.01, 0.015], rng.next()),
        skin_of(Motif::Bone, 0.5, [0.95, 0.88, 0.7], rng.next()),
    ];
    let wide = rng.fr(1.1, 1.5);
    let open = rng.fr(0.45, 0.75);
    let rim = |a: f32, k: f32| add(add(at, scale(across, a.cos() * r * wide * k)), scale(up, a.sin() * r * open * k));
    let mut parts = Vec::new();
    let mut throat = part(Shape::Ellipsoid, [r * wide, r * 0.1, r * open], 1);
    throat.anchor = at;
    throat.rot = columns(across, n, up);
    parts.push(throat);
    let lips = 14;
    for i in 0..lips {
        let a = i as f32 / lips as f32 * TAU;
        let mut lip = part(Shape::Ellipsoid, [r * 0.3, r * 0.3, r * 0.3], 0);
        lip.anchor = add(rim(a, 1.0), scale(n, r * 0.08));
        lip.lumps = 0.1;
        parts.push(lip);
    }
    let teeth = 18;
    for i in 0..teeth {
        let a = (i as f32 + 0.5) / teeth as f32 * TAU + rng.fr(-0.08, 0.08);
        let root = add(rim(a, 0.92), scale(n, r * 0.12));
        // Pointing into the mouth, and a little out of it.
        let y = norm(add(sub(at, root), scale(n, r * 0.15)));
        let x = norm(cross(y, n));
        let long = r * rng.fr(0.22, 0.42);
        let mut tooth = part(Shape::Cone, [r * 0.085, long, r * 0.085], 2);
        tooth.anchor = root;
        tooth.rot = columns(x, y, cross(x, y));
        tooth.pivot = [0.0, -1.0, 0.0];
        tooth.taper = 0.1;
        parts.push(tooth);
    }
    (Creature { base: [0.0; 3], yaw: 0.0, slots: vec![0; skins.len()], skins, parts, girth: 0.0 }, wide, open)
}

/// Put a sphincter on the wall at `at`, with veins leading away from it.
#[allow(clippy::too_many_arguments)]
fn place_mouth(scene: &mut Scene, at: [f32; 3], n: [f32; 3], across: [f32; 3], up: [f32; 3], r: f32, rng: &mut Rng, roots: &mut Vec<([f32; 3], [f32; 3], f32)>, mouths: &mut Vec<Mouth>) {
    let (c, wide, open) = mouth(at, n, across, up, r, rng);
    for i in 0..rng.range(1, 3) {
        let a = i as f32 * 2.4 + rng.fr(0.0, 1.0);
        let out = add(scale(across, a.cos()), scale(up, a.sin() * 0.7));
        roots.push((add(at, scale(out, r * 1.6)), norm(out), rng.fr(0.06, 0.085)));
    }
    let start = scene.flesh.tris.len();
    // Only meshed: in the scene's creatures the herd would walk it off the wall.
    scene.mesh_creature(&c);
    let tris = scene.flesh.tris.split_off(start);
    mouths.push(Mouth { tris, at, n, across, up, size: [r * wide, r * open], phase: rng.fr(0.0, TAU), every: rng.fr(4.0, 8.0), due: rng.fr(0.9, 1.0) });
}

/// Walk over `s` from `from` for `len` tiles, wandering, forking into thinner branches.
#[allow(clippy::too_many_arguments)]
fn grow(s: &Surface, scene: &Scene, from: Key, dir: [f32; 3], len: f32, r: f32, along: f32, forks: u32, rng: &mut Rng, out: &mut Vec<Vec<VeinPoint>>) {
    let mut trail = vec![from];
    let mut dir = dir;
    let mut turn = 0.0f32;
    let mut walked = 0.0;
    let mut dirs = vec![dir];
    let mut far = vec![0.0];
    while walked < len && trail.len() < 200 {
        let here = *trail.last().unwrap();
        let (p, n) = s.at[&here];
        turn = (turn + rng.fr(-0.14, 0.14)).clamp(-0.45, 0.45);
        dir = norm(add(dir, scale(cross(n, dir), turn)));
        let step = |k: &Key| {
            let d = sub(s.at[k].0, p);
            (dot(d, dir) / dot(d, d).sqrt().max(1.0), *k)
        };
        let Some((ahead, next)) = s.links[&here].iter().filter(|k| !trail.contains(k)).map(step).max_by(|a, b| a.0.total_cmp(&b.0)) else { break };
        let on_eye = s.keep_off.iter().any(|(c, r)| {
            let d = sub(s.at[&next].0, *c);
            dot(d, d).sqrt() < *r
        });
        if ahead < 0.15 || on_eye || turned(scene, next.map(|c| c as f32)) < 0.3 {
            break;
        }
        let d = sub(s.at[&next].0, p);
        walked += dot(d, d).sqrt() / TF;
        // Round corners and over ridges the walk takes the surface's way.
        dir = norm(add(scale(dir, 0.6), scale(norm(d), 0.4)));
        trail.push(next);
        dirs.push(dir);
        far.push(walked);
    }
    if trail.len() < 4 {
        return;
    }
    let mut points: Vec<VeinPoint> = trail
        .iter()
        .zip(&far)
        .map(|(k, &w)| {
            let (p, n) = s.at[k];
            let base = k.map(|c| c as f32);
            VeinPoint { p, n, r: r * (1.0 - w / walked).max(0.0).powf(0.6).max(0.3), along: along + w, rise: sub(p, base), spread: spread_of(scene.strangeness(base)) }
        })
        .collect();
    points.last_mut().unwrap().r = 0.0;
    // The walk goes along the edges of triangles: take the corners off it.
    for _ in 0..8 {
        let old = points.clone();
        for i in 1..old.len() - 1 {
            points[i].p = [0, 1, 2].map(|c| (old[i - 1].p[c] + 2.0 * old[i].p[c] + old[i + 1].p[c]) / 4.0);
            points[i].n = norm([0, 1, 2].map(|c| old[i - 1].n[c] + 2.0 * old[i].n[c] + old[i + 1].n[c]));
        }
    }
    for q in &mut points {
        q.p = add(q.p, scale(q.n, VEIN_LIFT * TF));
    }
    // The surface is cut finer than a vein needs.
    let finest = points.clone();
    points = finest.iter().step_by(2).copied().collect();
    if points.len() % 2 == 0 {
        points.push(*finest.last().unwrap());
    }
    points.last_mut().unwrap().r = 0.0;
    if forks > 0 {
        let mut i = rng.range(3, 8) as usize;
        while i + 3 < trail.len() {
            let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
            let off = norm(add(dirs[i], scale(cross(s.at[&trail[i]].1, dirs[i]), side * rng.fr(0.6, 1.3))));
            grow(s, scene, trail[i], off, (len - far[i]).max(0.5) * rng.fr(0.5, 0.9), finest[i].r * 0.7, finest[i].along, forks - 1, rng, out);
            i += rng.range(4, 12) as usize;
        }
    }
    out.push(points);
}

/// Whether `p` is in the plane of `t`, whose facing is `n`, and inside it.
fn lies_on(t: &Tri, p: [f32; 3], n: [f32; 3]) -> bool {
    let [a, b, c] = t.v.map(|v| v.p.map(|x| x as f32));
    if dot(sub(p, a), n).abs() > 8.0 {
        return false;
    }
    let side = |u: [f32; 3], v: [f32; 3]| dot(cross(sub(v, u), sub(p, u)), n);
    let (x, y, z) = (side(a, b), side(b, c), side(c, a));
    (x >= 0.0 && y >= 0.0 && z >= 0.0) || (x <= 0.0 && y <= 0.0 && z <= 0.0)
}

/// The nearest point of the surface to `p`.
fn nearest(keys: &[Key], s: &Surface, p: [f32; 3]) -> Key {
    let off = |k: &Key| {
        let d = sub(s.at[k].0, p);
        dot(d, d)
    };
    *keys.iter().min_by(|a, b| off(a).total_cmp(&off(b))).unwrap()
}

/// Turn patches of `scene` to flesh and put eyes, mouths and veins in them.
pub fn sculpt(scene: &mut Scene, seed: u64, eyes_from: Option<&Path>) -> Dreamed {
    let s = &mut scene.strange;
    s.base = 0.5;
    s.grad = [0.0; 2];
    s.bumps.clear();
    s.noise = (1.7, 0.22);
    s.seed = seed;
    // The texture changes straight from stone to flesh, and the lights stay as they were.
    s.morph = (0.5, 0.501);
    s.lurid = (2.0, 3.0);
    s.fog_ramp = (2.0, 3.0);
    scene.set_strange_amount(1.0);
    scene.flesh.matte = MATTE;

    let mut rng = Rng::new(seed);
    // Filmed eyes need more wall than flesh happens to cover, so the flesh grows round them.
    let mut sites: Vec<([f32; 3], [f32; 3], f32)> = Vec::new();
    if eyes_from.is_some() {
        let upright: Vec<Tri> = scene.tris.iter().filter(|t| t.normal()[1] > -0.7).copied().collect();
        for _ in 0..upright.len().min(2000) {
            let t = &upright[rng.next() as usize % upright.len()];
            let c = centre_of(t);
            if scene.strangeness(c) < 0.42 || sites.iter().any(|o| dot(sub(o.0, c), sub(o.0, c)).sqrt() < EYES_APART * TF) {
                continue;
            }
            let n = t.normal();
            let half = rng.fr(0.55, 0.85) * TF;
            let flat = |p: [f32; 3]| upright.iter().any(|o| dot(o.normal(), n) > 0.95 && lies_on(o, p, n));
            let up = if n[1].abs() > 0.8 { [0.0, 0.0, 1.0] } else { [0.0, -1.0, 0.0] };
            let across = norm(cross(up, n));
            let up = cross(n, across);
            if (0..12).all(|i| {
                let a = i as f32 / 12.0 * TAU;
                flat(add(add(c, scale(across, a.cos() * 0.95 * half)), scale(up, a.sin() * 0.95 * half)))
            }) {
                sites.push((c, n, half));
            }
        }
        scene.strange.bumps.extend(sites.iter().map(|&(c, _, half)| ([c[0], c[2]], half * 2.4, 0.4)));
    }
    let is_flesh = |scene: &Scene, t: &Tri| t.v.iter().any(|v| scene.strangeness(v.p.map(|c| c as f32)) > 0.45);
    let (keep, turn): (Vec<Tri>, Vec<Tri>) = std::mem::take(&mut scene.tris).into_iter().partition(|t| !is_flesh(scene, t));
    let turn = split_at_junctions(turn);
    let walls: Vec<Tri> = turn.iter().filter(|t| t.normal()[1] > -0.7).copied().collect();
    let small: Vec<Tri> = turn.iter().flat_map(cut).collect();
    let painted = tex::fully_strange_dir().and_then(|dir| Painted::load(scene, &dir));
    // Flesh textures, for pieces that take one, and each corner's rise from them.
    let slot_of = |t: &Tri| {
        let clean = CLEAN.iter().find(|c| c.0 == t.tex).map_or(t.tex, |c| c.1);
        ((clean as usize) < tex::STRANGE_SLOTS).then_some(clean as usize)
    };
    // The stone stays as it is under the flesh that grows over it.
    if painted.is_some() {
        scene.strange.morph = (2.0, 3.0);
    }
    let mut rises: HashMap<[i32; 3], (f32, f32)> = HashMap::new();
    if let Some(p) = &painted {
        for t in &small {
            if let Some(slot) = slot_of(t) {
                for v in &t.v {
                    let e = rises.entry(v.p).or_default();
                    *e = (e.0 + p.height(slot, laid_on(v.p.map(|c| c as f32), t.normal())), e.1 + 1.0);
                }
            }
        }
    }

    // Every point moves along the average facing of what meets there, so nothing parts at corners.
    let mut facing: HashMap<[i32; 3], [f32; 3]> = HashMap::new();
    for t in &small {
        for v in &t.v {
            let n = facing.entry(v.p).or_default();
            *n = add(*n, t.normal());
        }
    }
    let height = |scene: &Scene, p: [i32; 3], n: [f32; 3]| {
        let w = p.map(|c| c as f32);
        // The floor stays nearly flat, to be walked on.
        let floorish = 1.0 - 0.7 * smooth(-n[1] * 1.4);
        DEPTH * folds(w.map(|c| c / TF), seed) * turned(scene, w) * floorish
    };
    // Flesh stands proud of the stone, its creases squeezed up to meet it.
    let standing = painted.is_some();
    let lifted = |scene: &Scene, h: f32, n: [f32; 3], at: [f32; 3]| {
        if !standing {
            return h;
        }
        let x = h + LIFT * (1.0 - 0.7 * smooth(-n[1] * 1.4));
        0.5 * (x + (x * x + 4.0 * LIFT_SOFT * LIFT_SOFT).sqrt()) * room_for(scene, at, n)
    };
    // Sphincters are chosen now, with the eyes, for the wall round them to lie smooth.
    let mut mouth_sites: Vec<([f32; 3], [f32; 3], f32)> = Vec::new();
    if eyes_from.is_some() {
        let spots: Vec<&Tri> = walls.iter().filter(|t| turned(scene, centre_of(t)) > 0.6).collect();
        for _ in 0..spots.len().min(400) {
            let t = spots[rng.next() as usize % spots.len()];
            let c = centre_of(t);
            let clear = sites.iter().map(|o| o.0).chain(mouth_sites.iter().map(|o| o.0)).all(|o| dot(sub(o, c), sub(o, c)).sqrt() >= 2.0 * TF);
            if clear && rng.chance(0.3) {
                mouth_sites.push((c, t.normal(), rng.fr(0.3, 0.5) * TF));
            }
        }
    }
    let zones: Vec<([f32; 3], [f32; 3], f32, f32)> = sites
        .iter()
        .copied()
        .chain(mouth_sites.iter().map(|&(c, n, r)| (c, n, r * 2.4)))
        .map(|(c, n, half)| (c, n, half, lifted(scene, height(scene, c.map(|x| x as i32), n), n, c)))
        .collect();
    let moved: HashMap<[i32; 3], (f32, [i32; 3])> = facing
        .iter()
        .map(|(&p, &n)| {
            let n = norm(n);
            // The texture's own shape on top of the folds, and all of it smoothed under the features.
            let rise = rises.get(&p).map_or(0.0, |r| r.0 / r.1);
            let w = p.map(|c| c as f32);
            let floorish = 1.0 - 0.7 * smooth(-n[1] * 1.4);
            let raw = height(scene, p, n) + RELIEF * rise * turned(scene, w) * floorish;
            let mut h = lifted(scene, raw, n, w);
            for &(c, facing, half, middle) in &zones {
                if dot(n, facing) > 0.5 {
                    let d = sub(w, c);
                    h = middle + (h - middle) * smooth((dot(d, d).sqrt() / half - 0.9) / 0.5);
                }
            }
            let to = add(p.map(|c| c as f32), scale(n, h * TF));
            (p, (raw, to.map(|c| c.round() as i32)))
        })
        .collect();
    // Lit by the folded surface's own facing, and dark in the folds.
    let mut bent: HashMap<[i32; 3], [f32; 3]> = HashMap::new();
    for t in &small {
        let [a, b, c] = t.v.map(|v| moved[&v.p].1.map(|x| x as f32));
        let n = cross(sub(b, a), sub(c, a));
        for v in &t.v {
            let e = bent.entry(v.p).or_default();
            *e = add(*e, n);
        }
    }
    let mut surface = Surface { links: HashMap::new(), at: HashMap::new(), keep_off: Vec::new() };
    for t in &small {
        for i in 0..3 {
            let e = surface.links.entry(t.v[i].p).or_default();
            for o in [t.v[(i + 1) % 3].p, t.v[(i + 2) % 3].p] {
                if !e.contains(&o) {
                    e.push(o);
                }
            }
        }
    }
    for (&k, &(_, to)) in &moved {
        let (n, f) = (bent[&k], facing[&k]);
        surface.at.insert(k, (to.map(|c| c as f32), norm(if dot(n, f) < 0.0 { scale(n, -1.0) } else { n })));
    }
    let mut keys: Vec<Key> = moved.keys().copied().collect();
    keys.sort_unstable();
    let neutral = scene.flesh.lights.len() as u16;
    scene.flesh.lights.push([[1.0; 3]; 4]);
    scene.flesh.light_at.push([0.0; 3]);
    let mut points: Vec<WallPoint> = Vec::new();
    let mut wall_tris: Vec<WallTri> = Vec::new();
    let mut index: HashMap<Key, u32> = HashMap::new();
    let mut rough = Vec::with_capacity(small.len());
    for mut t in small {
        if let Some(p) = &painted {
            let Some(slot) = slot_of(&t) else { continue };
            let (along, down) = laid_axes(t.normal());
            let corners = t.v.map(|v| {
                *index.entry(v.p).or_insert_with(|| {
                    let (h, to) = moved[&v.p];
                    let (base, to) = (v.p.map(|c| c as f32), to.map(|c| c as f32));
                    let out = norm(facing[&v.p]);
                    let n = norm(bent[&v.p]);
                    points.push(WallPoint {
                        base,
                        rise: sub(to, base),
                        facing: out,
                        bent: if dot(n, out) < 0.0 { scale(n, -1.0) } else { n },
                        spread: spread_of(scene.strangeness(base)),
                        shadow: 0.35 + 0.65 * smooth((h + 0.5 * DEPTH) / DEPTH),
                        light: scene.light_sh(to),
                    });
                    points.len() as u32 - 1
                })
            });
            let uv = t.v.map(|v| laid_on(v.p.map(|c| c as f32), t.normal()));
            wall_tris.push(WallTri { points: corners, skin: p.skins[slot], tint: t.tint, along, down, uv });
            continue;
        }
        let centre = centre_of(&t);
        for v in &mut t.v {
            let (h, to) = moved[&v.p];
            let shaded = scene.shade(v.p, norm(bent[&v.p]), t.tint);
            let shadow = 0.35 + 0.65 * smooth((h + 0.5 * DEPTH) / DEPTH);
            v.col = shaded.map(|c| (c as f32 * shadow) as u8);
            v.p = to;
        }
        if let Some(&(_, clean)) = CLEAN.iter().find(|c| c.0 == t.tex).filter(|_| scene.morph(centre) > 0.5) {
            let (from, to) = (tex::uv_origin(t.tex), tex::uv_origin(clean));
            for v in &mut t.v {
                v.uv = [v.uv[0] - from.0 + to.0, v.uv[1] - from.1 + to.1];
            }
            t.tex = clean;
        }
        rough.push(t);
    }
    let mut small = rough;
    scene.tris = keep;
    scene.tris.append(&mut small);
    let mut turning = Vec::new();
    if painted.is_some() {
        turning = vec![[0.0; TURNING_FEATURES]; scene.tris.len()];
        turning.extend(turn.iter().map(|t| {
            let mut row = [0.0; TURNING_FEATURES];
            if slot_of(t).is_none() {
                return row;
            }
            let [a, b, c] = t.v.map(|v| v.p);
            let n = t.normal();
            let laid = t.v.map(|v| laid_on(v.p.map(|x| x as f32), n));
            let from = [0, 1].map(|k| laid.iter().map(|uv| uv[k]).fold(f32::MAX, f32::min).floor());
            // The edges' middles are points of the cut surface too, as `cut` makes them.
            let half = |x: [i32; 3], y: [i32; 3]| [0, 1, 2].map(|d| (x[d] * (N / 2) + y[d] * (N / 2)) / N);
            for (i, k) in [a, b, c].into_iter().enumerate() {
                row[i] = spread_of(scene.strangeness(k.map(|x| x as f32)));
                row[3 + 2 * i] = laid[i][0] - from[0];
                row[4 + 2 * i] = laid[i][1] - from[1];
            }
            for (i, k) in [a, b, c, half(a, b), half(a, c), half(b, c)].into_iter().enumerate() {
                row[9 + i] = moved[&k].0;
            }
            row
        }));
        scene.tris.extend(turn);
    }

    // Features, where the wall has turned properly.
    let lid = scene.flesh.skins.len() as u16;
    scene.flesh.skins.push(flesh::Skin::make(Motif::Flesh, 0.5, seed as u32, [0.5, 0.36, 0.33]));
    let footage = eyes_from.map(|dir| {
        let rim = scene.flesh.skins.len() as u16;
        scene.flesh.skins.push(flesh::Skin::make(Motif::Flesh, 0.5, seed as u32 + 2, [1.0, 0.75, 0.72]));
        Footage::load(scene, dir, rim)
    });
    let spots: Vec<&Tri> = walls.iter().filter(|t| turned(scene, centre_of(t)) > 0.6).collect();
    let mut eyes = Vec::new();
    let mut taken: Vec<[f32; 3]> = Vec::new();
    let mut mouths: Vec<Mouth> = Vec::new();
    let mut eye_paths: Vec<Vec<VeinPoint>> = Vec::new();
    // Where veins leave the features: the point, the way out, and how thick.
    let mut roots: Vec<([f32; 3], [f32; 3], f32)> = Vec::new();
    for &(c, n, half) in &sites {
        let up = if n[1].abs() > 0.8 { [0.0, 0.0, 1.0] } else { [0.0, -1.0, 0.0] };
        let across = norm(cross(up, n));
        let up = cross(n, across);
        let sticking = lifted(scene, height(scene, c.map(|x| x as i32), n), n, c);
        let at = add(c, scale(n, sticking * TF));
        // Tipped a little, so that they don't all stand to attention.
        let tip = rng.fr(-0.25, 0.25);
        let (across, up) = (add(scale(across, tip.cos()), scale(up, tip.sin())), add(scale(up, tip.cos()), scale(across, -tip.sin())));
        taken.push(c);
        let wall_here = walls.iter().find(|t| dot(t.normal(), n) > 0.95 && lies_on(t, c, n));
        let tint = wall_here.map_or([1.0; 3], |t| t.tint);
        let wall = painted.as_ref().zip(wall_here.and_then(slot_of)).map(|(p, s)| p.skins[s]);
        let middle = sticking * TF;
        let place = |x: f32, y: f32| {
            let d = x.hypot(y);
            let base = add(add(c, scale(across, x * half)), scale(up, y * half));
            let dome = half * (0.16 * (1.0 - d * d).max(0.0) - 0.03);
            add(base, scale(n, middle + dome))
        };
        let side = PATCH + 1;
        let mut corners = Vec::with_capacity(side * side);
        for j in 0..side {
            for i in 0..side {
                let (x, y) = ((i as f32 / PATCH as f32) * 2.0 - 1.0, 1.0 - (j as f32 / PATCH as f32) * 2.0);
                let e = 1.0 / PATCH as f32;
                let facing = norm(cross(sub(place(x, y + e), place(x, y - e)), sub(place(x + e, y), place(x - e, y))));
                let facing = if dot(facing, n) < 0.0 { scale(facing, -1.0) } else { facing };
                    let p = place(x, y);
                    let flat = add(add(c, scale(across, x * half)), scale(up, y * half));
                    let at_tiles = p.map(|c| c / TF);
                    // An almond rather than a circle, with an edge that wanders.
                    let wander = 0.2 * noise(at_tiles.map(|c| c * 1.9), seed ^ 21) + 0.09 * noise(at_tiles.map(|c| c * 5.1), seed ^ 22);
                    let sh = scene.light_sh(p).map(|c| [0, 1, 2].map(|i| c[i] * tint[i]));
                    let w = 1.0 - smooth((x.hypot(y * 1.35) + wander - 0.66) / 0.26);
                    corners.push(PatchCorner { flat, rise: dot(sub(p, flat), n), n: facing, w, wall_uv: laid_on(p, n), light: sh });
            }
        }
        let spot = add(at, scale(n, 0.2 * half));
        let light = scene.flesh.lights.len() as u16;
        let sh = scene.light_sh(spot);
        scene.flesh.lights.push(sh);
        scene.flesh.light_at.push(spot);
        eyes.push(Eye { at: spot, n, across, up, r: half, skin: 0, light, phase: rng.fr(0.0, BLINK_EVERY), patch: Some(Patch { corners, mirrored: rng.chance(0.5), wall, axes: laid_axes(n) }), due: rng.fr(0.8, 0.9) });
        surface.keep_off.push((at, half * 0.8));
        for _ in 0..rng.range(1, 3) {
            let a = rng.fr(0.0, TAU);
            let out = add(scale(across, a.cos()), scale(up, a.sin()));
            roots.push((add(at, scale(out, half * 0.95)), out, rng.fr(0.045, 0.07)));
            // Sometimes the vein also comes the other way, over the lids and onto the white.
            if rng.chance(EYE_VEINS) {
                let from = [a.cos() * 0.95, a.sin() * 0.95];
                let to = [a.cos().signum() * rng.fr(0.34, 0.5), rng.fr(-0.12, 0.12)];
                let (bow, r) = (rng.fr(-0.18, 0.18), rng.fr(0.025, 0.045) * TF);
                let phase = rng.fr(0.0, TAU);
                const STEPS: usize = 18;
                let points: Vec<VeinPoint> = (0..=STEPS)
                    .map(|i| {
                        let f = i as f32 / STEPS as f32;
                        let wiggle = bow * (f * TAU / 2.0).sin() + 0.03 * (f * 13.0 + phase).sin();
                        let (x, y) = (from[0] + (to[0] - from[0]) * f - (to[1] - from[1]) * wiggle, from[1] + (to[1] - from[1]) * f + (to[0] - from[0]) * wiggle);
                        let d = 0.04 * half * f;
                        VeinPoint { p: add(place(x, y), scale(n, 0.01 * TF + d * 0.1)), n, r: if i == STEPS { 0.0 } else { r * (1.0 - f).powf(0.6).max(0.3) }, along: 5.75 + f * 0.7, rise: scale(n, middle), spread: 1.0 }
                    })
                    .collect();
                eye_paths.push(points);
            }
        }
    }
    for &(c, n, r) in &mouth_sites {
        let up = if n[1].abs() > 0.8 { [0.0, 0.0, 1.0] } else { [0.0, -1.0, 0.0] };
        let across = norm(cross(up, n));
        let up = cross(n, across);
        let at = add(c, scale(n, lifted(scene, height(scene, c.map(|x| x as i32), n), n, c) * TF));
        place_mouth(scene, at, n, across, up, r, &mut rng, &mut roots, &mut mouths);
    }
    for _ in 0..if footage.is_some() { 0 } else { spots.len().min(400) } {
        let t = spots[rng.next() as usize % spots.len()];
        let c = centre_of(t);
        let apart = 1.1;
        if taken.iter().any(|o| dot(sub(*o, c), sub(*o, c)).sqrt() < apart * TF) {
            continue;
        }
        let n = t.normal();
        let up = if n[1].abs() > 0.8 { [0.0, 0.0, 1.0] } else { [0.0, -1.0, 0.0] };
        let across = norm(cross(up, n));
        let up = cross(n, across);
        let sticking = height(scene, c.map(|x| x as i32), n);
        let at = add(c, scale(n, sticking * TF));
        if rng.chance(0.3) {
            taken.push(c);
            let r = rng.fr(0.3, 0.5) * TF;
            place_mouth(scene, at, n, across, up, r, &mut rng, &mut roots, &mut mouths);
            continue;
        }
        taken.push(c);
        // Mostly alone, sometimes in a little crowd.
        for k in 0..if rng.chance(0.3) { rng.range(2, 5) } else { 1 } {
            let (x, y) = if k == 0 { (0.0, 0.0) } else { (rng.fr(-0.8, 0.8) * TF, rng.fr(-0.6, 0.6) * TF) };
            let spot = add(add(at, scale(across, x)), scale(up, y));
            let r = rng.fr(0.2, 0.38) * TF;
            let seed = rng.next();
            let skin = scene.flesh.skins.len() as u16;
            scene.flesh.skins.push(flesh::Skin::from_rgb(flesh::SKIN, eye_image(seed), [1.0; 3]));
            let light = scene.flesh.lights.len() as u16;
            let sh = scene.light_sh(spot);
            scene.flesh.lights.push(sh);
            scene.flesh.light_at.push(spot);
            eyes.push(Eye { at: spot, n, across, up, r, skin, light, phase: rng.fr(0.0, BLINK_EVERY), patch: None, due: 0.0 });
            for _ in 0..rng.range(1, 3) {
                let a = rng.fr(0.0, TAU);
                let out = add(scale(across, a.cos()), scale(up, a.sin()));
                roots.push((add(spot, scale(out, r * 1.2)), out, rng.fr(0.045, 0.07)));
            }
        }
    }
    // Veins from the features, and some that start from nowhere.
    let mut paths = eye_paths;
    for (p, out, r) in roots {
        let from = nearest(&keys, &surface, p);
        let n = surface.at[&from].1;
        let along_wall = norm(sub(out, scale(n, dot(out, n))));
        grow(&surface, scene, from, along_wall, rng.fr(2.0, 4.5), r * TF, 0.0, 1, &mut rng, &mut paths);
    }
    let free: Vec<Key> = keys.iter().filter(|k| surface.at[*k].1[1] > -0.7 && turned(scene, k.map(|c| c as f32)) > 0.6).copied().collect();
    for _ in 0..(free.len() / VEIN_EVERY).min(FREE_VEINS) {
        let from = free[rng.next() as usize % free.len()];
        let n = surface.at[&from].1;
        let any = [rng.fr(-1.0, 1.0), rng.fr(-1.0, 1.0), rng.fr(-1.0, 1.0)];
        grow(&surface, scene, from, norm(cross(n, any)), rng.fr(2.5, 5.5), rng.fr(0.05, 0.08) * TF, 0.0, 1, &mut rng, &mut paths);
    }
    let vein_skin = [0.0f32, 1.0].map(|k| {
        let tint = [0.3 + 0.6 * k, 0.3 - 0.12 * k, 0.65 - 0.3 * k];
        let i = scene.flesh.skins.len() as u16;
        scene.flesh.skins.push(flesh::Skin::make(Motif::Flesh, 0.5, seed as u32 + 1, tint));
        i
    });
    let veins: Vec<Vein> = paths
        .into_iter()
        .map(|points| {
            let mid = points[points.len() / 2].p;
            let light = scene.flesh.lights.len() as u16;
            let sh = scene.light_sh(mid);
            scene.flesh.lights.push(sh);
            scene.flesh.light_at.push(mid);
            Vein { points, light }
        })
        .collect();
    eprintln!("{} eyes, {} veins", eyes.len(), veins.len());
    // The flesh pass needs something to draw for the skins to be there; the walls come later.
    if let Some(p) = &painted {
        let v = flesh::Vertex { p: [0.0; 3], n: [0.0, 1.0, 0.0], w: 1.0, uv: [[0.0; 2]; 2], t: [[1.0, 0.0, 0.0]; 2], b: [[0.0, 0.0, 1.0]; 2], cap: [[0.0; 2]; 2], wrap: [1.0; 2], grow: 0.0 };
        scene.flesh.tris.push(flesh::Tri { v: [v; 3], skin: [p.skins[0]; 2], light: [neutral; 2], corners: None, cap_axes: [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]; 2] });
    }
    let fixed = scene.flesh.tris.len();
    Dreamed { eyes, mouths, points, walls: wall_tris, neutral, amount: 1.0, veins, vein_skin, footage, turning, fixed, lid }
}

/// What the net is told about how each level triangle turns, as written to `turning.npy`,
/// appended to `out` in `tri_features`' order (the doors and levers after the level never
/// turn): how much of the dial reaches each corner (3); where the corners fall on the flesh
/// textures, in repeats (6); and the height of the folds, in tiles, at the corners and the
/// middles of edges ab, ac, bc (6). All zeros on what stays stone.
pub fn turning_features(scene: &Scene, d: &Dreamed, out: &mut Vec<f32>) {
    assert_eq!(d.turning.len(), scene.tris.len(), "the net can only be told about painted levels");
    out.extend(d.turning.iter().flatten());
    out.extend(std::iter::repeat_n(0.0, scene.part_tris().count() * TURNING_FEATURES));
}

/// Whether the flesh at dial `amount` has swollen up round a viewer at `cam`: a view from
/// inside it shows nothing the net could learn.
pub fn crowds(d: &Dreamed, cam: &Camera, amount: f32) -> bool {
    const CLOSE: f32 = 0.35 * TF;
    let eye = cam.pos.map(|c| c as f32);
    d.points.iter().any(|q| {
        let here = amount * q.spread;
        if stage(here, CREEP) <= 0.0 {
            return false;
        }
        let to = sub(add(q.base, scale(q.rise, stage(here, SAG))), eye);
        dot(to, to) < CLOSE * CLOSE
    })
}

/// Eyes and sphincters as (centre, facing), to stand in front of for training views.
pub fn sights(d: &Dreamed) -> Vec<([f32; 3], [f32; 3])> {
    d.eyes.iter().map(|e| (e.at, e.n)).chain(d.mouths.iter().map(|m| (m.at, m.n))).collect()
}

/// The features in `cam`'s view at `t` seconds, appended to `out` as `ORGAN_FEATURES` floats
/// each, positions and sizes in tiles: kind, 0 eye, 1 sphincter, 2 length of vein (0); where
/// it is now (1-3); what it faces (4-6); its across and up (7-9, 10-12; for an eye the image's
/// right and up, for a vein along it and to the side); half its size along those (13-14;
/// for a vein half its length and its radius now); and its state (15-18): an eye's showing,
/// frame of the blink as a share (0 open) and gaze across and down, 0..1; a sphincter's
/// showing and clench; a vein's beat. The same ones `pose` draws, on or near the screen of a
/// frame `width` wide, eyes and sphincters first and then the nearest.
pub fn organs(d: &Dreamed, cam: &Camera, width: u32, t: f32, out: &mut Vec<f32>) {
    let eye = cam.pos.map(|c| c as f32);
    let rows = crate::world::view::view_matrix(cam).m.map(|r| r.map(|e| e as f32 / 4096.0));
    let reach = Sight { eye, ahead: rows[2] };
    let (kx, ky) = (width as f32 / 2.0 / cam.h as f32 * 1.05, 120.0 / cam.h as f32 * 1.05);
    let on_screen = |at: [f32; 3], size: f32| {
        let to = sub(at, eye);
        let [x, y, z] = rows.map(|r| dot(r, to));
        let margin = 1.5 * size;
        z > -margin && x.abs() - margin <= z.max(0.0) * kx && y.abs() - margin <= z.max(0.0) * ky
    };
    let tiles = |v: [f32; 3]| v.map(|c| c / TF);
    let token = |kind: f32, at: [f32; 3], n: [f32; 3], x: [f32; 3], y: [f32; 3], size: [f32; 2], state: [f32; 4]| -> [f32; ORGAN_FEATURES] {
        let mut row = [0.0; ORGAN_FEATURES];
        row[0] = kind;
        row[1..4].copy_from_slice(&tiles(at));
        row[4..7].copy_from_slice(&n);
        row[7..10].copy_from_slice(&x);
        row[10..13].copy_from_slice(&y);
        row[13] = size[0] / TF;
        row[14] = size[1] / TF;
        row[15..19].copy_from_slice(&state);
        row
    };
    let mut found: Vec<(bool, f32, [f32; ORGAN_FEATURES])> = Vec::new();
    let mut add_one = |at: [f32; 3], row: [f32; ORGAN_FEATURES]| {
        let d = sub(at, eye);
        found.push((row[0] == 2.0, dot(d, d), row));
    };
    if let Some(footage) = &d.footage {
        for e in &d.eyes {
            let (Some(patch), Some(want)) = (&e.patch, reach.of_eye(e)) else { continue };
            let (showing, frame) = filmed_frame(e, footage.skins.len() - 1, t, d.amount);
            if showing <= 0.0 || !on_screen(e.at, e.r) {
                continue;
            }
            let (gx, gy) = gaze_of(e, patch, want);
            let middle = patch.corners[(PATCH + 1) * PATCH / 2 + PATCH / 2];
            let at = add(middle.flat, scale(e.n, middle.rise * stage(d.amount, SAG)));
            let right = if patch.mirrored { scale(e.across, -1.0) } else { e.across };
            let state = [showing, frame as f32 / (footage.skins.len() - 1) as f32, gx as f32 / (GAZES_X - 1) as f32, gy as f32 / (GAZES_Y - 1) as f32];
            add_one(at, token(0.0, at, e.n, right, e.up, [e.r, e.r], state));
        }
    }
    for m in &d.mouths {
        let showing = stage(d.amount, (m.due - 0.1, m.due));
        if showing <= 0.0 || !reach.near(m.at) || dot(sub(eye, m.at), m.n) <= 0.0 || !on_screen(m.at, m.size[0] * 2.0) {
            continue;
        }
        add_one(m.at, token(1.0, m.at, m.n, m.across, m.up, m.size, [showing, clench_of(m, t), 0.0, 0.0]));
    }
    for v in &d.veins {
        let Some(last) = vein_end(v, d.amount).filter(|_| reach.near(v.points[v.points.len() / 2].p)) else { continue };
        let mut i = 0;
        while i < last {
            let j = (i + VEIN_TOKEN).min(last);
            let k = (i + j) / 2;
            let (a, b) = (vein_now(&v.points[i], t, d.amount).0, vein_now(&v.points[j], t, d.amount).0);
            let (_, r, beat) = vein_now(&v.points[k], t, d.amount);
            let at = scale(add(a, b), 0.5);
            let (run, n) = (sub(b, a), v.points[k].n);
            let half = dot(run, run).sqrt() / 2.0;
            i = j;
            if r > 0.0 && dot(sub(eye, at), n) > 0.0 && on_screen(at, half + r) {
                let along = norm(run);
                add_one(at, token(2.0, at, n, along, norm(cross(along, n)), [half, r], [beat, 0.0, 0.0, 0.0]));
            }
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    found.truncate(MAX_ORGANS);
    out.extend(found.iter().flat_map(|f| f.2));
}

fn centre_of(t: &Tri) -> [f32; 3] {
    [0, 1, 2].map(|c| t.v.iter().map(|v| v.p[c] as f32).sum::<f32>() / 3.0)
}

/// One eyelid: a shell just outside the eyeball at `centre`, from the top (`side` 1) or bottom
/// of the eye down to `reach` radians from there at the middle, curving back at the corners.
/// Returns how each corner moves as part `base` of `Moving` (see `LID`).
fn lid(scene: &mut Scene, e: &Eye, centre: [f32; 3], side: f32, reach: f32, skin: u16, base: usize) -> Vec<[[f32; 4]; 3]> {
    const ACROSS: usize = 16;
    const DOWN: usize = 8;
    let radius = 1.07 * e.r;
    let up = scale(e.up, side);
    let vert = |i: usize, j: usize| {
        let a = (i as f32 / ACROSS as f32 * 2.0 - 1.0) * 1.45;
        let along = j as f32 / DOWN as f32;
        // A narrower opening at the corners: an almond.
        let narrow = (0.55 + 0.45 * a.cos().max(0.0).sqrt()).max(0.3);
        let b = along * (reach * narrow + 0.25 * (1.0 - a.cos().max(0.0)));
        // From the pole above the eye, down the sphere at longitude `a` from the front.
        let out = [0, 1, 2].map(|c| up[c] * b.cos() + (e.n[c] * a.cos() + e.across[c] * a.sin()) * b.sin());
        let uv = [i as f32 / ACROSS as f32 * 2.0, along * 0.8];
        let v = flesh::Vertex {
            p: add(centre, scale(out, radius)),
            n: out,
            w: 1.0,
            uv: [uv; 2],
            t: [e.across; 2],
            b: [scale(up, -1.0); 2],
            cap: [uv; 2],
            wrap: [1.0; 2],
            grow: 1.0,
        };
        // How far down the sphere the corner goes for each radian more of `reach`.
        (v, [LID, base as f32, along * narrow, side])
    };
    let mut how = Vec::new();
    for i in 0..ACROSS {
        for j in 0..DOWN {
            let q = [vert(i, j), vert(i + 1, j), vert(i, j + 1), vert(i + 1, j + 1)];
            for [a, b, c] in [[0, 1, 2], [1, 3, 2]] {
                scene.flesh.tris.push(flesh::Tri { v: [q[a].0, q[b].0, q[c].0], skin: [skin; 2], light: [e.light; 2], corners: None, cap_axes: [[e.across, e.up]; 2] });
                how.push([q[a].1, q[b].1, q[c].1]);
            }
        }
    }
    how
}

/// How much of a heartbeat is passing a point, 0..1, `x` beats after the last one: two
/// quick pushes, the second weaker.
fn beat(x: f32) -> f32 {
    let f = x.rem_euclid(1.0);
    let bump = |c: f32, w: f32| (-((f - c) / w).powi(2)).exp();
    (bump(0.1, 0.06) + 0.6 * bump(0.34, 0.07)).min(1.0)
}

/// How far along a vein the blood has pushed it: its last point, None while it's barely out.
fn vein_end(v: &Vein, amount: f32) -> Option<usize> {
    let front = stage(amount, VEINS_RUN) * VEIN_LENGTH;
    match v.points.iter().position(|q| q.along > front + 0.4) {
        Some(0) | Some(1) => None,
        Some(i) => Some(i - 1),
        None => Some(v.points.len() - 1),
    }
}

/// A point of a vein at `t` seconds: where it lies on the wall, its radius, and the beat there.
fn vein_now(q: &VeinPoint, t: f32, amount: f32) -> ([f32; 3], f32, f32) {
    let b = beat(t / BEAT_EVERY - q.along / BEAT_WAVE);
    let r = q.r * (1.0 + BEAT_SWELL * b) * smooth((stage(amount, VEINS_RUN) * VEIN_LENGTH - q.along) / 0.4);
    (sub(q.p, scale(q.rise, 1.0 - stage(amount * q.spread, SAG))), r, b)
}

/// A vein as it stands at `t` seconds: half a tube lying on the wall, fattened where the
/// beat has got to, and flushed red with it. Without `t`, with no beat in it, for the flesh
/// shader to beat as part `base` of `Moving` (see `VEIN`): returns how each corner moves.
fn mesh_vein(scene: &mut Scene, v: &Vein, skin: [u16; 2], t: Option<f32>, amount: f32, base: usize) -> Vec<[[f32; 4]; 3]> {
    const AROUND: usize = 4;
    const SPREAD: f32 = 1.9;
    let Some(last) = vein_end(v, amount) else { return Vec::new() };
    let ring = |i: usize| -> [(flesh::Vertex, [f32; 4]); AROUND + 1] {
        let q = v.points[i];
        let along = norm(sub(v.points[(i + 1).min(last)].p, v.points[i.saturating_sub(1)].p));
        let side = norm(cross(along, q.n));
        let out = cross(side, along);
        let (on_wall, r, b) = vein_now(&q, t.unwrap_or(0.0), amount);
        // A beat swells the tube out of its middle, which stays where it is.
        let rest = r / (1.0 + BEAT_SWELL * b);
        let b = if t.is_some() { b } else { 0.0 };
        let r = rest * (1.0 + BEAT_SWELL * b);
        let mid = add(on_wall, scale(out, 0.15 * rest));
        std::array::from_fn(|j| {
            let a = (j as f32 / AROUND as f32 * 2.0 - 1.0) * SPREAD;
            let n = add(scale(out, a.cos()), scale(side, a.sin()));
            let uv = [j as f32 / AROUND as f32, q.along * 3.0];
            let vert = flesh::Vertex { p: add(mid, scale(n, r)), n, w: 1.0 - b, uv: [uv; 2], t: [side; 2], b: [along; 2], cap: [uv; 2], wrap: [1.0; 2], grow: 1.0 };
            (vert, [VEIN, base as f32, q.along / BEAT_WAVE, rest * BEAT_SWELL])
        })
    };
    let mut how = Vec::new();
    let mut near = ring(0);
    for i in 1..=last {
        let far = ring(i);
        for j in 0..AROUND {
            for [a, b, c] in [[near[j], near[j + 1], far[j]], [near[j + 1], far[j + 1], far[j]]] {
                // Triangles wind so that this is against what they face (see `lid`).
                let flip = dot(cross(sub(b.0.p, a.0.p), sub(c.0.p, a.0.p)), a.0.n) > 0.0;
                let [a, b, c] = if flip { [a, c, b] } else { [a, b, c] };
                scene.flesh.tris.push(flesh::Tri { v: [a.0, b.0, c.0], skin, light: [v.light; 2], corners: None, cap_axes: [[a.0.t[0], a.0.b[0]]; 2] });
                how.push([a.1, b.1, c.1]);
            }
        }
        near = far;
    }
    how
}

/// How clenched a sphincter is at `t` seconds: 0 relaxed, 1 shut, and a little past either.
fn clench_of(m: &Mouth, t: f32) -> f32 {
    let wave = 0.5 + 0.5 * (t * TAU / m.every + m.phase).sin();
    smooth(wave) * 1.2 - 0.2 + 0.03 * smooth(wave) * (t * 19.0 + m.phase).sin()
}

/// A sphincter at `t` seconds: the opening drawn shut towards its middle and bulging, the
/// outside of the rim staying where it is, and then let go.
fn mesh_mouth(scene: &mut Scene, m: &Mouth, t: f32, amount: f32) {
    // The lips' outside, and how close to the middle the opening comes, in halves of the opening.
    const RIM: f32 = 1.9;
    const SHUT: f32 = 0.06;
    let out_of_wall = stage(amount, (m.due - 0.1, m.due));
    if out_of_wall <= 0.0 {
        return;
    }
    let clench = clench_of(m, t);
    let pull = 1.0 + clench * (RIM / (RIM - 1.0 + SHUT) - 1.0);
    for tri in &m.tris {
        let mut tri = *tri;
        for v in &mut tri.v {
            let d = sub(v.p, m.at);
            let (x, y, z) = (dot(d, m.across) / m.size[0], dot(d, m.up) / m.size[1], dot(d, m.n));
            let out = x.hypot(y).max(1e-4);
            let to = if out < RIM { (RIM - (RIM - out) * pull).max(SHUT * 0.3) } else { out };
            let rise = clench.max(0.0) * 0.35 * m.size[1] * (1.0 - (to / RIM).min(1.0));
            let k = to / out * out_of_wall;
            v.p = add(add(add(m.at, scale(m.across, x * k * m.size[0])), scale(m.up, y * k * m.size[1])), scale(m.n, (z + rise) * out_of_wall));
        }
        scene.flesh.tris.push(tri);
    }
}

/// The filmed gaze that looks towards something this far `sideways` and `up` of straight
/// out of the wall (as sines), `(0, 0)` being the far left and top.
fn gaze_cell(sideways: f32, up: f32) -> (usize, usize) {
    let cell = |x: f32, cells: usize| (x + (cells / 2) as f32).round().clamp(0.0, cells as f32 - 1.0) as usize;
    (cell(sideways / GAZE_STEP[0], GAZES_X), cell(-up / GAZE_STEP[1], GAZES_Y))
}

/// How far a filmed eye has come through the wall, and which of the `frames` of the blink it is on.
fn filmed_frame(e: &Eye, frames: usize, t: f32, amount: f32) -> (f32, usize) {
    // The closed lid comes through the wall first, and then the eye opens as it does in the film.
    let (showing, open) = (stage(amount, (e.due - 0.12, e.due)), stage(amount, (e.due, e.due + 0.1)));
    let beat = (t + e.phase).rem_euclid(BLINK_EVERY) * FOOTAGE_RATE;
    let frame = if open < 1.0 {
        FIRST_OPENING + ((frames - FIRST_OPENING + 1) as f32 * open) as usize
    } else if (beat as usize) < frames {
        beat as usize + 1
    } else {
        0
    };
    (showing, if frame > frames { 0 } else { frame })
}

/// The filmed gaze of an eye looking along `want`.
fn gaze_of(e: &Eye, patch: &Patch, want: [f32; 3]) -> (usize, usize) {
    // The image's right is along `across`, or against it where mirrored.
    let sideways = dot(want, e.across) * if patch.mirrored { -1.0 } else { 1.0 };
    gaze_cell(sideways, dot(want, e.up))
}

/// A filmed eye at `t` seconds: the frame of the blink it has got to, turned to the viewer.
fn mesh_filmed(scene: &mut Scene, e: &Eye, patch: &Patch, footage: &Footage, want: [f32; 3], t: f32, amount: f32) {
    let (showing, frame) = filmed_frame(e, footage.skins.len() - 1, t, amount);
    if showing <= 0.0 {
        return;
    }
    let sag = stage(amount, SAG);
    let (gx, gy) = gaze_of(e, patch, want);
    let skin = [footage.skins[frame][gx * GAZES_Y + gy], patch.wall.unwrap_or(footage.rim)];
    let (right, down) = (if patch.mirrored { scale(e.across, -1.0) } else { e.across }, scale(e.up, -1.0));
    let side = PATCH + 1;
    let vert = |i: usize, j: usize| {
        let c = patch.corners[j * side + i];
        let u = i as f32 / PATCH as f32;
        let uv = [[if patch.mirrored { 1.0 - u } else { u }, j as f32 / PATCH as f32], if patch.wall.is_some() { c.wall_uv } else { [u, j as f32 / PATCH as f32] }];
        flesh::Vertex { p: add(c.flat, scale(e.n, c.rise * sag)), n: c.n, w: c.w * showing, uv, t: [right, patch.axes.0], b: [down, patch.axes.1], cap: uv, wrap: [1.0; 2], grow: 1.0 }
    };
    let round = |i: usize, j: usize| (i as f32 / PATCH as f32 * 2.0 - 1.0).hypot(j as f32 / PATCH as f32 * 2.0 - 1.0) <= 1.0;
    for j in 0..PATCH {
        for i in 0..PATCH {
            if !(round(i, j) && round(i + 1, j) && round(i, j + 1) && round(i + 1, j + 1)) {
                continue;
            }
            let q = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)];
            for [a, b, c] in [[0, 1, 2], [1, 3, 2]] {
                let flip = {
                    let (a, b, c) = (vert(q[a].0, q[a].1), vert(q[b].0, q[b].1), vert(q[c].0, q[c].1));
                    dot(cross(sub(b.p, a.p), sub(c.p, a.p)), a.n) > 0.0
                };
                let order = if flip { [q[a], q[c], q[b]] } else { [q[a], q[b], q[c]] };
                let tv = order.map(|(i, j)| vert(i, j));
                let corners = Some(order.map(|(i, j)| patch.corners[j * side + i].light));
                scene.flesh.tris.push(flesh::Tri { v: tv, skin, light: [e.light; 2], corners, cap_axes: [[e.across, e.up]; 2] });
            }
        }
    }
}

/// The flesh walls as far as `d.amount` has grown them, for the smooth pass: they need
/// remaking only when that changes.
pub fn walls(d: &Dreamed) -> Vec<flesh::Tri> {
    walls_where(d, |_| true)
}

/// `walls` as `cam` can see them: nothing wholly behind it or past its fog, which the net
/// is told nothing about either. Far fewer to draw for one view of a large level.
pub fn walls_seen(d: &Dreamed, cam: &Camera) -> Vec<flesh::Tri> {
    let eye = cam.pos.map(|c| c as f32);
    let ahead = crate::world::view::view_matrix(cam).m[2].map(|e| e as f32 / 4096.0);
    let far = cam.fog_far as f32 + TF;
    walls_where(d, |corners| {
        corners.iter().any(|&p| {
            let to = sub(p, eye);
            dot(to, ahead) > -0.5 * TF && dot(to, to) < far * far
        })
    })
}

/// `walls`, keeping only the triangles `keep` says yes to, by their corners as they are now.
fn walls_where(d: &Dreamed, keep: impl Fn([[f32; 3]; 3]) -> bool) -> Vec<flesh::Tri> {
    let now: Vec<([f32; 3], [f32; 3], f32, f32)> = d
        .points
        .iter()
        .map(|q| {
            let here = d.amount * q.spread;
            let (sag, grow) = (stage(here, SAG), stage(here, CREEP));
            // A hair off the stone even where it has only just come through.
            let p = add(add(q.base, scale(q.facing, 5.0)), scale(q.rise, sag));
            (p, norm(add(scale(q.facing, 1.0 - sag), scale(q.bent, sag))), grow, 1.0 + (q.shadow - 1.0) * sag)
        })
        .collect();
    let mut out = Vec::with_capacity(d.walls.len());
    for t in &d.walls {
        let c = t.points.map(|i| now[i as usize]);
        if c.iter().all(|v| v.2 <= 0.0) || !keep(c.map(|v| v.0)) {
            continue;
        }
        let corner = |k: usize| flesh::Vertex { p: c[k].0, n: c[k].1, w: 1.0, uv: [t.uv[k]; 2], t: [t.along; 2], b: [t.down; 2], cap: [t.uv[k]; 2], wrap: [1.0; 2], grow: c[k].2 };
        let light = |k: usize| d.points[t.points[k] as usize].light.map(|r| [0, 1, 2].map(|i| r[i] * t.tint[i] * c[k].3));
        out.push(flesh::Tri { v: [corner(0), corner(1), corner(2)], skin: [t.skin; 2], light: [d.neutral; 2], corners: Some([light(0), light(1), light(2)]), cap_axes: [[t.along, t.down]; 2] });
    }
    out
}

/// Which features a view has any business drawing.
struct Sight {
    eye: [f32; 3],
    ahead: [f32; 3],
}

impl Sight {
    /// Near enough to draw and not behind the viewer, allowing for the time before the next pose.
    fn near(&self, at: [f32; 3]) -> bool {
        let to = sub(at, self.eye);
        let far = dot(to, to).sqrt();
        far < 2.0 * TF || (far < REACH * TF && dot(to, self.ahead) > -0.2 * far)
    }

    /// The way from eye `e` to the viewer, if it's in range and facing them.
    fn of_eye(&self, e: &Eye) -> Option<[f32; 3]> {
        let to = sub(self.eye, e.at);
        let far = dot(to, to).sqrt();
        (far <= 9.0 * TF && dot(to, e.n) >= 0.0).then(|| scale(to, 1.0 / far))
    }
}

impl Sight {
    fn of(cam: &Camera) -> Sight {
        Sight { eye: cam.pos.map(|c| c as f32), ahead: crate::world::view::view_matrix(cam).m[2].map(|e| e as f32 / 4096.0) }
    }
}

/// Where an eyeball sits in its socket.
fn ball_centre(e: &Eye) -> [f32; 3] {
    sub(e.at, scale(e.n, 0.45 * e.r))
}

/// An eyeball's axes as it looks along `want`, as far round as it can turn; straight out of
/// the wall without.
fn ball_axes(e: &Eye, want: Option<[f32; 3]>) -> [[f32; 3]; 3] {
    let y = match want {
        Some(want) => {
            let off = dot(want, e.n).clamp(-1.0, 1.0).acos();
            let k = if off > GAZE { GAZE / off } else { 1.0 };
            norm(add(scale(e.n, 1.0 - k), scale(want, k)))
        }
        None => e.n,
    };
    let x = norm(cross(y, e.up));
    [x, y, cross(x, y)]
}

/// An eyeball looking along `want` (see `ball_axes`).
fn mesh_ball(scene: &mut Scene, e: &Eye, want: Option<[f32; 3]>) {
    let centre = ball_centre(e);
    let [x, y, z] = ball_axes(e, want);
    let ball = part(Shape::Ellipsoid, [e.r; 3], 0);
    let at = Placed { joint: centre, centre, axes: columns(x, y, z) };
    let look = Skin { motif: Motif::Eyes, density: 0.0, scale: 2.0 * e.r, tint: [1.0; 3], seed: 0 };
    scene.mesh_part(&ball, &at, &look, 0, Some((e.skin, e.light)));
}

/// How far shut an eye's lids are at `t` seconds, 0..1: they close over it now and then.
fn shut_of(e: &Eye, t: f32) -> f32 {
    let beat = (t + e.phase).rem_euclid(BLINK_EVERY);
    if beat < BLINK {
        (beat / BLINK * TAU / 2.0).sin()
    } else {
        0.0
    }
}

/// Point the eyes in view at `cam`, blink them and beat the veins, at `t` seconds. `Moving`
/// does the same on the GPU, and the two must look alike.
pub fn pose(scene: &mut Scene, d: &Dreamed, cam: &Camera, t: f32) {
    scene.flesh.tris.truncate(d.fixed);
    let sight = Sight::of(cam);
    for m in &d.mouths {
        if sight.near(m.at) && dot(sub(sight.eye, m.at), m.n) > 0.0 {
            mesh_mouth(scene, m, t, d.amount);
        }
    }
    for v in &d.veins {
        if sight.near(v.points[v.points.len() / 2].p) {
            mesh_vein(scene, v, d.vein_skin, Some(t), d.amount, 0);
        }
    }
    for e in &d.eyes {
        let Some(want) = sight.of_eye(e) else { continue };
        if let (Some(patch), Some(footage)) = (&e.patch, &d.footage) {
            mesh_filmed(scene, e, patch, footage, want, t, d.amount);
            continue;
        }
        mesh_ball(scene, e, Some(want));
        let shut = shut_of(e, t);
        for (side, open, closed) in LIDS {
            lid(scene, e, ball_centre(e), side, open + (closed - open) * shut, d.lid, 0);
        }
    }
}

/// The flesh that moves (the eyes and their lids, the veins and the sphincters) as one mesh
/// built at the dial's `amount`, for the flesh shader to move by `state` each frame: what
/// `pose` remakes on the CPU, without remaking it. Whatever `pose` leaves out for being out
/// of sight is hidden instead.
pub struct Moving {
    pub tris: Vec<flesh::Tri>,
    /// Per corner of `tris`: how it moves (`BALL` and on), where its part's numbers start in
    /// `state`, and two numbers of its own.
    pub how: Vec<[[f32; 4]; 3]>,
    /// The time and the beat's period, then each part's numbers: those that change are
    /// `pose`'s to write.
    pub state: Vec<[f32; 4]>,
    amount: f32,
    /// Where each eye's, sphincter's and vein's numbers start, if it was built.
    eyes: Vec<Option<usize>>,
    mouths: Vec<Option<usize>>,
    veins: Vec<Option<usize>>,
}

impl Moving {
    /// Everything of `d`'s that moves, as far as its dial has turned. Uses `scene`'s flesh to
    /// build in, and leaves it with only what doesn't move.
    pub fn new(scene: &mut Scene, d: &Dreamed) -> Moving {
        scene.flesh.tris.truncate(d.fixed);
        let mut how: Vec<[[f32; 4]; 3]> = Vec::new();
        let mut state = vec![[0.0, BEAT_EVERY, 0.0, 0.0]];
        // A part whose corners all move alike.
        let whole = |scene: &Scene, how: &mut Vec<[[f32; 4]; 3]>, kind: f32, base: usize| {
            how.resize(scene.flesh.tris.len() - d.fixed, [[kind, base as f32, 0.0, 0.0]; 3]);
        };
        let mut mouths = Vec::new();
        for m in &d.mouths {
            if stage(d.amount, (m.due - 0.1, m.due)) <= 0.0 {
                mouths.push(None);
                continue;
            }
            // Where it is, and its size: how far out of the wall and how clenched, `pose`'s.
            let base = state.len();
            state.extend([[m.at[0], m.at[1], m.at[2], 0.0], [m.across[0], m.across[1], m.across[2], m.size[0]]]);
            state.extend([[m.up[0], m.up[1], m.up[2], m.size[1]], [m.n[0], m.n[1], m.n[2], 0.0]]);
            scene.flesh.tris.extend_from_slice(&m.tris);
            whole(scene, &mut how, MOUTH, base);
            mouths.push(Some(base));
        }
        let mut veins = Vec::new();
        for v in &d.veins {
            if vein_end(v, d.amount).is_none() {
                veins.push(None);
                continue;
            }
            // Whether it's shown.
            let base = state.len();
            state.push([0.0; 4]);
            how.extend(mesh_vein(scene, v, d.vein_skin, None, d.amount, base));
            veins.push(Some(base));
        }
        let mut eyes = Vec::new();
        for e in &d.eyes {
            let base = state.len();
            if let (Some(patch), Some(footage)) = (&e.patch, &d.footage) {
                if filmed_frame(e, footage.skins.len() - 1, 0.0, d.amount).0 <= 0.0 {
                    eyes.push(None);
                    continue;
                }
                // The skin it shows, and whether it's shown.
                state.push([0.0; 4]);
                mesh_filmed(scene, e, patch, footage, e.n, 0.0, d.amount);
                whole(scene, &mut how, FILMED, base);
            } else {
                // How it has turned from straight out (columns), whether it's shown, how far
                // each lid is from open; where it sits, and its up.
                let (c, up) = (ball_centre(e), e.up);
                state.extend([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]);
                state.extend([[c[0], c[1], c[2], 0.0], [up[0], up[1], up[2], 0.0]]);
                mesh_ball(scene, e, None);
                whole(scene, &mut how, BALL, base);
                for (side, open, _) in LIDS {
                    how.extend(lid(scene, e, c, side, open, d.lid, base));
                }
            }
            eyes.push(Some(base));
        }
        let tris = scene.flesh.tris.split_off(d.fixed);
        debug_assert_eq!(tris.len(), how.len());
        Moving { tris, how, state, amount: d.amount, eyes, mouths, veins }
    }

    /// The dial's turn it was built at.
    pub fn amount(&self) -> f32 {
        self.amount
    }

    /// Everything as `pose` would have it for `cam` at `t` seconds, in `state`.
    pub fn pose(&mut self, d: &Dreamed, cam: &Camera, t: f32) {
        let sight = Sight::of(cam);
        let s = &mut self.state;
        s[0][0] = t;
        for (m, at) in d.mouths.iter().zip(&self.mouths) {
            let Some(i) = *at else { continue };
            let shown = sight.near(m.at) && dot(sub(sight.eye, m.at), m.n) > 0.0;
            s[i][3] = if shown { stage(self.amount, (m.due - 0.1, m.due)) } else { 0.0 };
            s[i + 3][3] = clench_of(m, t);
        }
        for (v, at) in d.veins.iter().zip(&self.veins) {
            let Some(i) = *at else { continue };
            s[i][0] = sight.near(v.points[v.points.len() / 2].p) as i32 as f32;
        }
        for (e, at) in d.eyes.iter().zip(&self.eyes) {
            let Some(i) = *at else { continue };
            let want = sight.of_eye(e);
            if let (Some(patch), Some(footage)) = (&e.patch, &d.footage) {
                let Some(want) = want else {
                    s[i][1] = 0.0;
                    continue;
                };
                let (_, frame) = filmed_frame(e, footage.skins.len() - 1, t, self.amount);
                let (gx, gy) = gaze_of(e, patch, want);
                s[i] = [footage.skins[frame][gx * GAZES_Y + gy] as f32, 1.0, 0.0, 0.0];
                continue;
            }
            s[i][3] = want.is_some() as i32 as f32;
            let Some(want) = want else { continue };
            // The turn from straight out to `want`: each column is where an axis goes.
            let (rest, now) = (ball_axes(e, None), ball_axes(e, Some(want)));
            for c in 0..3 {
                let column = [0, 1, 2].map(|r| (0..3).map(|k| now[k][r] * rest[k][c]).sum::<f32>());
                s[i + c][..3].copy_from_slice(&column);
            }
            let shut = shut_of(e, t);
            for (k, (_, open, closed)) in LIDS.iter().enumerate() {
                s[i + 1 + k][3] = (closed - open) * shut;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphincters_stay_out_of_the_creatures() {
        let (mut scene, _) = crate::world::walkable_scene(&mut Rng::new(3), 30);
        let before = scene.creatures.len();
        let d = sculpt(&mut scene, 3, None);
        assert!(!d.mouths.is_empty(), "no sphincters to check");
        assert_eq!(scene.creatures.len(), before);
    }

    /// Corners lying partway along another triangle's edge, and the triangles' total area.
    fn junctions(tris: &[Tri]) -> (usize, f64) {
        let mut corners: Vec<[i64; 3]> = tris.iter().flat_map(|t| t.v.map(|v| v.p.map(|c| c as i64))).collect();
        corners.sort_unstable();
        corners.dedup();
        let mut found = 0;
        let mut area = 0.0;
        for t in tris {
            let p = t.v.map(|v| v.p.map(|c| c as i64));
            let cr = |u: [i64; 3], w: [i64; 3]| [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
            let d = |u: [i64; 3], w: [i64; 3]| [0, 1, 2].map(|k| w[k] - u[k]);
            area += cr(d(p[0], p[1]), d(p[0], p[2])).iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt() / 2.0;
            for i in 0..3 {
                let (a, b) = (p[i], p[(i + 1) % 3]);
                let ab = d(a, b);
                let len: i64 = ab.iter().map(|x| x * x).sum();
                found += corners.iter().filter(|&&c| {
                    let ac = d(a, c);
                    let k: i64 = (0..3).map(|j| ac[j] * ab[j]).sum();
                    cr(ab, ac) == [0; 3] && k > 0 && k < len
                }).count();
            }
        }
        (found, area)
    }

    #[test]
    fn neighbours_share_whole_edges_once_split() {
        let v = |x: i32, z: i32| Vert { p: [x, 0, z], uv: [0, 0], col: [128; 3], alb: [128; 3] };
        let tri = |a, b, c| Tri { v: [a, b, c], tex: 0, n: [0.0, -1.0, 0.0], tint: [1.0; 3] };
        // One long edge along z = 0 with two small triangles' corners on it from the other side.
        let tris = vec![
            tri(v(0, 0), v(512, 0), v(256, 256)),
            tri(v(0, 0), v(128, -128), v(128, 0)),
            tri(v(128, 0), v(128, -128), v(384, 0)),
        ];
        let (before, area) = junctions(&tris);
        assert_eq!(before, 2);
        let split = split_at_junctions(tris);
        let (after, area_after) = junctions(&split);
        assert_eq!(after, 0);
        assert_eq!(split.len(), 5);
        assert!((area - area_after).abs() < 1e-6, "{area} became {area_after}");

        let (scene, _) = crate::world::walkable_scene(&mut Rng::new(5), 30);
        let (_, area) = junctions(&scene.tris);
        let split = split_at_junctions(scene.tris.clone());
        let (after, area_after) = junctions(&split);
        assert_eq!(after, 0);
        assert!((area - area_after).abs() / area < 1e-9);
    }

    #[test]
    fn a_filmed_eye_turns_up_and_down_as_well_as_sideways() {
        let middle = (GAZES_X / 2, GAZES_Y / 2);
        assert_eq!(gaze_cell(0.0, 0.0), middle);
        // A small look up already moves it, and up is towards the top of the image.
        assert_eq!(gaze_cell(0.0, 0.2), (middle.0, middle.1 - 1));
        assert_eq!(gaze_cell(0.0, -0.2), (middle.0, middle.1 + 1));
        assert_eq!(gaze_cell(0.25, 0.0), (middle.0 + 1, middle.1));
        assert_eq!(gaze_cell(-1.0, 1.0), (0, 0));
        assert_eq!(gaze_cell(1.0, -1.0), (GAZES_X - 1, GAZES_Y - 1));
    }
}
