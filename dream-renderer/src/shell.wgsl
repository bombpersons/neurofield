// Shell-mapped relief (shell.rs): each fragment of a tetrahedron's back face marches its pixel's
// line of sight through the tetrahedron, from where it enters to where it leaves, against the
// dreamed height on the pages, and keeps where it first meets it. In a tetrahedron the place on
// the triangle (barycentrics) and the height coordinate (-reach in to +reach out at each
// corner) are linear in the point, so along the line of sight too.

struct U {
    pos: vec4<f32>,
    // The view's axes as rows: right, down, forward.
    r0: vec4<f32>,
    // (r1.w: the halvings' stop, of a pixel; shell.rs record's `pixel`.)
    r1: vec4<f32>,
    r2: vec4<f32>,
    // Focal length in shown pixels, half the shown width and height, the fog's far end.
    k: vec4<f32>,
    // The height's limit, the pages' heights times this, where in mem the largest |height| on
    // any page is (its bits; the prisms reach just past it, `reach`), and how far in from an
    // open edge the height tapers from nothing.
    hp: vec4<f32>,
    // The face table (where, its mask), the faces' pages, the pages' heights in mem.
    t: vec4<u32>,
    // The lowest cell the face table's keys count from, the pages' texels a side.
    o: vec4<i32>,
}

@group(0) @binding(0) var<uniform> u: U;
// Each level triangle (TRI vec4s): its corners, its normal in their w; its corners' order by
// welded place (two bits each), its texture, whether the height tapers to each edge (corner k
// to k + 1: bit k; shell.rs edges) and whether each is a crease; its corners' places on the
// texture (two to a vec4); their light; their torch albedo; the way each corner is pushed, and
// the most it's pushed in w; for each crease, how many points along it are in toward and where
// they start (crease.wgsl); its corners' welded corners (lean).
@group(0) @binding(1) var<storage, read> corners: array<vec4<f32>>;
const TRI: u32 = 19u;
// A prism's tetrahedra (shell.rs TETS): each corner the triangle's corner by order (low two
// bits), out (4) or in.
const TETS = array<vec4<u32>, 3>(vec4<u32>(0u, 4u, 5u, 6u), vec4<u32>(0u, 1u, 6u, 5u), vec4<u32>(0u, 1u, 2u, 6u));
// level.rs Level::data.
@group(0) @binding(2) var<storage, read> level: array<u32>;
// dream.wgsl's mem.
@group(0) @binding(3) var<storage, read> mem: array<f32>;
@group(0) @binding(4) var textures: texture_2d_array<f32>;
@group(0) @binding(5) var nearest: sampler;

const NEAR: f32 = 0.02;
const NONE: u32 = 0xffffffffu;
// dream.wgsl HEIGHT_CELL: a page's height is a value each 2 x 2 texels.
const HEIGHT_CELL: i32 = 2;
// The most even steps along a stretch, then halvings between the last two.
const MOST_STEPS: u32 = 96u;
const HALVINGS: u32 = 8u;
// The page's bits of the triangle target's last u32.
const PAGE_BITS: u32 = 0x0fffffffu;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) tri: u32,
    @location(1) @interpolate(flat) tet: u32,
}

// The largest |height| smoothed this frame (smooth.wgsl), f32 bits.
@group(0) @binding(6) var<storage, read> smooth_bound: array<u32>;
// Each welded corner's way it's been looked at from, and its offset across its pushes, added to
// each (lean.wgsl; 0 where it doesn't lean), the shortest edge from it in w.
@group(0) @binding(7) var<storage, read> lean: array<vec4<f32>>;
// A corner pushed `r` leans across by no more than this of its shortest edge, so neighbouring
// corners leaning by different amounts don't turn a small triangle's prism inside out.
const SHORT_HOLD: f32 = 0.25;

fn leaned(weld: u32, r: f32) -> vec3<f32> {
    let o = lean[2u * weld + 1u];
    let across = length(o.xyz) * r;
    return o.xyz * select(1.0, SHORT_HOLD * o.w / across, across > SHORT_HOLD * o.w);
}

// How far the prisms reach out and in (shell.rs reach, LEAST_REACH): past the largest |height|
// on any page, or smoothed, the largest smoothed one.
fn reach() -> f32 {
    var b = mem[bitcast<u32>(u.hp.z)];
    if (u.pos.w > 0.0 || u.r0.w > 0.0) { b = bitcast<f32>(smooth_bound[0]); }
    return clamp(b * u.hp.y * 1.01 + 1e-3, 1e-3, max(u.hp.x, 1e-3));
}

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) dir: vec3<f32>, @location(2) side: f32, @location(3) cap: f32, @location(4) tri: u32, @location(5) tet: u32, @location(6) weld: u32) -> Out {
    let r = min(reach(), cap);
    let d = pos + (dir + leaned(weld, r)) * side * r - u.pos.xyz;
    let q = vec3<f32>(dot(u.r0.xyz, d), dot(u.r1.xyz, d), dot(u.r2.xyz, d));
    var out: Out;
    // As raster.wgsl: screen x = w/2 + focal x/z, y = h/2 + focal y/z (y down); depth NEAR/z.
    out.clip = vec4<f32>(u.k.x * q.x / u.k.y, -u.k.x * q.y / u.k.z, NEAR, q.z);
    out.tri = tri;
    out.tet = tet;
    return out;
}

// dream.wgsl's face_hash, face_of and pick.
fn face_hash(k0: u32, k1: u32) -> u32 {
    var h = (k0 * 0x9e3779b1u) ^ (k1 * 0x85ebca77u);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    return h ^ (h >> 12u);
}

fn face_of(orient: u32, c: vec3<i32>) -> u32 {
    let r = c - u.o.xyz;
    if (any(r < vec3<i32>(0)) || r.x >= (1 << 29u) || r.y >= 65536 || r.z >= 65536) { return NONE; }
    let k0 = orient | (u32(r.x) << 3u);
    let k1 = (u32(r.y) << 16u) | u32(r.z);
    var h = face_hash(k0, k1) & u.t.y;
    for (var i = 0u; i <= u.t.y; i++) {
        let e = u.t.x + h * 3u;
        if (level[e] == NONE) { return NONE; }
        if (level[e] == k0 && level[e + 1u] == k1) { return level[e + 2u]; }
        h = (h + 1u) & u.t.y;
    }
    return NONE;
}

fn pick(v: vec3<f32>, axis: u32) -> vec2<f32> {
    if (axis == 0u) { return v.yz; }
    if (axis == 1u) { return v.xz; }
    return v.xy;
}


// A plane the height's read on: its normal, distance, main axis and orientation (its faces'),
// and whether it's the triangle's own (0) for the cache of pages (`page_at`).
struct Plane {
    n: vec3<f32>,
    d: f32,
    axis: u32,
    orient: u32,
    slot: u32,
}

fn plane_of(m: vec3<f32>, d: f32, slot: u32) -> Plane {
    let am = abs(m);
    var ax = 0u;
    if (am.y > am.x) { ax = 1u; }
    if (am.z > max(am.x, am.y)) { ax = 2u; }
    return Plane(m, d, ax, ax * 2u + select(0u, 1u, m[ax] > 0.0), slot);
}

// Its triangle: its plane (normal, distance), main axis and orientation.
var<private> n: vec3<f32>;
var<private> d0: f32;
var<private> axis: u32;
var<private> orient: u32;
var<private> own: Plane;
// For the triangle's plane, and for the planes beyond its creases (with its orientation): the
// last cell a foot was in, and its face's page (-1 for none). The page under the foot itself,
// `page`.
var<private> home_cell: vec3<i32>;
var<private> home_page: f32;
var<private> own_cell: vec3<i32>;
var<private> own_page: f32;
var<private> other_cell: vec3<i32>;
var<private> other_orient: u32;
var<private> other_page: f32;
var<private> page: f32;
// Its sides (outward normals, and how far along them); which the height tapers to (open
// edges) and which are creases, and for each crease the points along it in toward (shell.rs
// set_level, crease.wgsl): how many, and where they start.
var<private> tapers: u32;
var<private> creases: u32;
// The triangle's first vec4 in `corners`, where its creases' points are found.
var<private> tri_at: u32;
// The prisms' reach this frame.
var<private> frame_lim: f32;

// How much of the height `h` stands at a foot: nothing at an edge it tapers to, all of it the
// taper's width in: hp.w, or wider for a tall one, so the slope it makes stays gentle (a steep
// one stretches the texture and dream laid flat on the plane over it).
const TAPER_RUN: f32 = 2.0;

fn taper(foot: vec3<f32>, h: f32) -> f32 {
    let edges = tapers;
    let width = max(max(u.hp.w, abs(h) * TAPER_RUN), 1e-6);
    var k = 1.0;
    for (var i = 0u; i < 3u; i++) {
        if ((edges & (1u << i)) != 0u) {
            k *= smoothstep(0.0, width, inside(i, foot));
        }
    }
    return k;
}

// How far a foot on the triangle's plane is inside its edge k (from corner k to the next).
// Worked out where it's needed rather than kept: the march has too many registers as it is.
fn inside(k: u32, foot: vec3<f32>) -> f32 {
    let p = tc[k];
    let m = normalize(cross(tc[(k + 1u) % 3u] - p, n));
    return dot(m, p) - dot(m, foot);
}

// The page of the face of `pl` a point of the plane is on (-1 for none), the last cell's kept.
fn page_at(q: vec3<f32>, pl: Plane) -> f32 {
    let c = vec3<i32>(floor(q - pl.n * 0.01));
    if (pl.slot == 0u) {
        if (all(c == home_cell)) { return home_page; }
        if (any(c != own_cell)) {
            own_cell = c;
            own_page = page_of_face(pl.orient, c);
        }
        return own_page;
    }
    if (any(c != other_cell) || pl.orient != other_orient) {
        other_cell = c;
        other_orient = pl.orient;
        other_page = page_of_face(pl.orient, c);
    }
    return other_page;
}

fn page_of_face(o: u32, c: vec3<i32>) -> f32 {
    let face = face_of(o, c);
    if (face == NONE) { return -1.0; }
    return bitcast<f32>(level[u.t.z + face]);
}

// The world's directions along a page's two axes (pick's).
fn axes_of(ax: u32) -> array<vec3<f32>, 2> {
    if (ax == 0u) { return array<vec3<f32>, 2>(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
    if (ax == 1u) { return array<vec3<f32>, 2>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
    return array<vec3<f32>, 2>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
}

fn page_axes() -> array<vec3<f32>, 2> {
    return axes_of(axis);
}

// A page's height cell at `at` (page * r * r + row * r + column): as dreamed, or smoothed
// (u.pos.w, u.r0.w) smooth.wgsl's, the far-off smoothing mixed in there.
fn height_cell(at: u32) -> f32 {
    if (u.pos.w <= 0.0 && u.r0.w <= 0.0) { return mem[u.t.w + at]; }
    return near[at];
}

// The height cells at `at`, the next along the row and the two below (cell `local` of page
// `pg`), from the buffers. The march's four_cells (shell.rs) reads them from a texture of the
// smoothed heights in one fetch instead; the compute passes appended to this use this.
fn cells_from(at: u32, ru: u32) -> vec4<f32> {
    return vec4<f32>(height_cell(at), height_cell(at + 1u), height_cell(at + ru), height_cell(at + ru + 1u));
}

// Pages a row of the smoothed heights' texture (shell.rs HEIGHTS_ROW).
const HEIGHTS_ROW: u32 = 16u;

// The tile cell i of the plane's grid is in (cell i's middle at (i + 1/4) / r).
fn tile_of(i: vec2<i32>, r: i32) -> vec2<i32> {
    return vec2<i32>(floor((vec2<f32>(i) + 0.25) / f32(r)));
}

// Height cell g of the plane's grid (r a tile along the page's axes, cell i's middle at
// (i + 1/4) / r, as page_height has it), from the page of the tile it's in: (its value, 1),
// or 0s where that tile has no page.
fn cell_value(foot: vec3<f32>, g: vec2<i32>, pl: Plane) -> vec2<f32> {
    let r = u.o.w / HEIGHT_CELL;
    let c = (vec2<f32>(g) + 0.25) / f32(r);
    let pf = pick(foot, pl.axis);
    let ax = axes_of(pl.axis);
    var q = foot + ax[0] * (c.x - pf.x) + ax[1] * (c.y - pf.y);
    q -= pl.n * (dot(pl.n, q) - pl.d);
    let pg = page_at(q, pl);
    if (pg < 0.0) { return vec2<f32>(0.0); }
    let local = clamp(g - vec2<i32>(floor(c)) * r, vec2<i32>(0), vec2<i32>(r - 1));
    let ru = u32(r);
    return vec2<f32>(height_cell(u32(pg) * ru * ru + u32(local.y) * ru + u32(local.x)), 1.0);
}

// The height at a point of a plane, bilinear between the height cells around it, across a
// page's edge from the next tile's page (so the wall has no step where two tiles meet), those
// of tiles with no page left out (0 if none has one). Mostly all four are on one tile's page,
// looked up once.
fn raw_height(foot: vec3<f32>, pl: Plane) -> f32 {
    let ri = u.o.w / HEIGHT_CELL;
    let r = f32(ri);
    let g = pick(foot, pl.axis) * r - 0.25;
    let i0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    let wf = vec4<f32>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let t0 = tile_of(i0, ri);
    if (all(t0 == tile_of(i0 + vec2<i32>(1), ri))) {
        let c = (vec2<f32>(i0) + 0.25) / r;
        let pf = pick(foot, pl.axis);
        let ax = axes_of(pl.axis);
        var q = foot + ax[0] * (c.x - pf.x) + ax[1] * (c.y - pf.y);
        q -= pl.n * (dot(pl.n, q) - pl.d);
        let pg = page_at(q, pl);
        if (pg < 0.0) { return 0.0; }
        let local = i0 - t0 * ri;
        let ru = u32(ri);
        let at = u32(pg) * ru * ru + u32(local.y) * ru + u32(local.x);
        let v = four_cells(at, u32(pg), vec2<u32>(local));
        return dot(wf, v) * u.hp.y;
    }
    let a = cell_value(foot, i0, pl);
    let b = cell_value(foot, i0 + vec2<i32>(1, 0), pl);
    let c = cell_value(foot, i0 + vec2<i32>(0, 1), pl);
    let d = cell_value(foot, i0 + vec2<i32>(1, 1), pl);
    let w = wf * vec4<f32>(a.y, b.y, c.y, d.y);
    let sum = w.x + w.y + w.z + w.w;
    if (sum <= 0.0) { return 0.0; }
    return dot(w, vec4<f32>(a.x, b.x, c.x, d.x)) / sum * u.hp.y;
}

// The height at a foot on the triangle: the plane's, met at each crease by the wall beyond's
// (each drawn to the mean of the two at the nearest point of the edge, eased in over the
// taper's width, at least TAPER_RUN times the difference, so a bulge or hollow carries on round
// the corner; the half difference taken this frame at points along the edge, crease.wgsl),
// tapered to nothing at open edges.
fn height_at(foot: vec3<f32>) -> f32 {
    var h = raw_height(foot, own);
    if (creases != 0u) {
        let band = max(u.hp.w, frame_lim * TAPER_RUN);
        for (var k = 0u; k < 3u; k++) {
            if ((creases & (1u << k)) == 0u) { continue; }
            let dist = inside(k, foot);
            if (dist >= band) { continue; }
            let a = tc[k];
            let e = tc[(k + 1u) % 3u] - a;
            let along = bitcast<vec2<u32>>(corners[tri_at + 15u + k].xy);
            let last = along.x - 1u;
            let s = clamp(dot(foot - a, e) / dot(e, e), 0.0, 1.0) * f32(last);
            let i = min(u32(s), last - 1u);
            let at = along.y + i;
            let half = mix(toward[at], toward[at + 1u], s - f32(i));
            let width = max(max(u.hp.w, abs(half) * TAPER_RUN), 1e-6);
            h += half * (1.0 - smoothstep(0.0, width, max(dist, 0.0)));
        }
    }
    if (tapers != 0u) { h *= taper(foot, h); }
    return h;
}

// The prism's mapping, from a point to its place on the triangle (barycentrics b) and its
// height coordinate s (-1 on the inner shell, +1 on the outer): a point is at s over b where
// it's sum_k b_k (corner_k + s dir_k), each corner's direction times its reach (so the shells
// are the tetrahedra's flat inner and outer faces), at the height s sum_k b_k reach_k. At s the
// corners make a triangle with edges e1 + s f1, e2 + s f2 from corner 0 + s dir0, their cross
// c0 + c1 s + c2 s^2; the point's in its plane where f(s) = c(s) . (x - corner 0 - s dir0) = 0,
// a cubic, solved by Newton from a guess near (the last sample's along the line of sight).
// Exact, so the height 0 is the triangle's plane whatever the directions (the tetrahedra's
// linear mapping isn't: it bends off it by the reach times how much the directions differ),
// and in f32 precise however thin the shell.
var<private> tc: array<vec3<f32>, 3>;
var<private> dirs: array<vec3<f32>, 3>;
var<private> reaches: vec3<f32>;
var<private> c0: vec3<f32>;
var<private> c1: vec3<f32>;
var<private> c2: vec3<f32>;
// The line of sight, and the last height found along it.
var<private> ray_o: vec3<f32>;
var<private> ray_d: vec3<f32>;
var<private> guess: f32;

fn mapping() {
    let e1 = tc[1] - tc[0];
    let e2 = tc[2] - tc[0];
    let f1 = dirs[1] - dirs[0];
    let f2 = dirs[2] - dirs[0];
    c0 = cross(e1, e2);
    c1 = cross(e1, f2) + cross(f1, e2);
    c2 = cross(f1, f2);
}

const NEWTON: u32 = 2u;

// The barycentrics and height coordinate of x (in w), from the guess.
fn place_of(x: vec3<f32>) -> vec4<f32> {
    let q0 = x - tc[0];
    var h = guess;
    for (var i = 0u; i < NEWTON; i++) {
        let c = c0 + (c1 + c2 * h) * h;
        let q = q0 - dirs[0] * h;
        let f = dot(c, q);
        let df = dot(c1 + 2.0 * c2 * h, q) - dot(c, dirs[0]);
        if (abs(df) > 1e-12) { h -= f / df; }
    }
    guess = h;
    let c = c0 + (c1 + c2 * h) * h;
    let q = q0 - dirs[0] * h;
    let cc = max(dot(c, c), 1e-20);
    // The triangle at h's edges from corner 0 (e1 + h f1, e2 + h f2).
    let s1 = (tc[1] - tc[0]) + (dirs[1] - dirs[0]) * h;
    let s2 = (tc[2] - tc[0]) + (dirs[2] - dirs[0]) * h;
    let b1 = dot(cross(q, s2), c) / cc;
    let b2 = dot(cross(s1, q), c) / cc;
    return vec4<f32>(1.0 - b1 - b2, b1, b2, h);
}

fn foot_of(b: vec3<f32>) -> vec3<f32> {
    return tc[0] * b.x + tc[1] * b.y + tc[2] * b.z;
}

// The place of the line of sight at view depth t.
fn place_at(t: f32) -> vec4<f32> {
    return place_of(ray_o + ray_d * t);
}

// How far above the relief a place is: its height less the relief's at its foot (held within
// the reach there). At or below 0, it's met it.
fn above_place(w: vec4<f32>) -> f32 {
    let lim = dot(w.xyz, reaches);
    return w.w * lim - clamp(height_at(foot_of(w.xyz)), -lim, lim);
}

fn above(t: f32) -> f32 {
    return above_place(place_at(t));
}

struct Hit {
    // Triangle + 1; the height's slope along the page (du, dv), two halves; the view depth's
    // bits; the page (PAGE_BITS for none).
    @location(0) tri: vec4<u32>,
    // The foot's place on its tile along the page's axes (two unorm halves), and the level's
    // light, the torch's albedo (each RGBA8, halved) and the texture at the foot, as
    // raster.wgsl's.
    @location(1) look: vec4<u32>,
}

// What the march found at a pixel, for resolve.wgsl to finish once the nearest is known:
// triangle + 1 (0 for none), the hit's barycentrics 1 and 2 and its view depth (bits).
struct Marched {
    @location(0) found: vec4<u32>,
    @builtin(frag_depth) depth: f32,
}

// The triangle's state the march and its finishing both work from.
fn begin(tri: u32) {
    let ta = tri * TRI;
    let a = corners[ta];
    let b = corners[ta + 1u];
    let c = corners[ta + 2u];
    let info = corners[ta + 3u];
    tapers = bitcast<u32>(info.z);
    creases = bitcast<u32>(info.w);
    n = vec3<f32>(a.w, b.w, c.w);
    d0 = dot(a.xyz, n);
    own = plane_of(n, d0, 0u);
    axis = own.axis;
    orient = own.orient;
    own_cell = vec3<i32>(-2147483647);
    other_cell = own_cell;
    let home = homes[tri];
    home_cell = home.xyz;
    home_page = bitcast<f32>(home.w);
    let lim = reach();
    frame_lim = lim;
    tc = array<vec3<f32>, 3>(a.xyz, b.xyz, c.xyz);
    let welded = bitcast<vec4<u32>>(corners[ta + 18u]);
    for (var k = 0u; k < 3u; k++) {
        let pk = corners[ta + 12u + k];
        reaches[k] = min(lim, pk.w);
        dirs[k] = (pk.xyz + leaned(welded[k], reaches[k])) * reaches[k];
    }
    mapping();
    tri_at = ta;
}

@fragment
fn fs(in: Out) -> Marched {
    begin(in.tri);
    let ta = in.tri * TRI;
    let info = corners[ta + 3u];
    // The tetrahedron's corners.
    let order = bitcast<u32>(info.x);
    let tet = TETS[in.tet];
    var p: array<vec3<f32>, 4>;
    for (var i = 0u; i < 4u; i++) {
        let k = (order >> (2u * (tet[i] & 3u))) & 3u;
        p[i] = tc[k] + dirs[k] * select(-1.0, 1.0, tet[i] >= 4u);
    }
    // The pixel's line of sight, t its view depth, and where it's inside the tetrahedron:
    // inside each face's plane (each face's normal towards the corner opposite).
    let o = u.pos.xyz;
    let dv = vec3<f32>((in.clip.x - u.k.y) / u.k.x, (in.clip.y - u.k.z) / u.k.x, 1.0);
    let d = u.r0.xyz * dv.x + u.r1.xyz * dv.y + u.r2.xyz * dv.z;
    ray_o = o;
    ray_d = d;
    var t = vec2<f32>(NEAR, u.k.w);
    for (var i = 0u; i < 4u; i++) {
        let a0 = p[(i + 1u) % 4u];
        var m = cross(p[(i + 2u) % 4u] - a0, p[(i + 3u) % 4u] - a0);
        if (dot(m, p[i] - a0) < 0.0) { m = -m; }
        let dm = dot(m, d);
        let room = dot(m, o - a0);
        if (abs(dm) < 1e-20) {
            if (room < 0.0) { discard; }
        } else if (dm > 0.0) {
            t.x = max(t.x, -room / dm);
        } else {
            t.y = min(t.y, -room / dm);
        }
    }
    if (t.y <= t.x) { discard; }
    // Coming in past the nearest inner shell (shell.rs Shell::inner), it's been met nearer.
    let inner_at = textureLoad(inner, vec2<i32>(in.clip.xy), 0);
    if (inner_at > 0.0 && t.x * inner_at > NEAR * 1.0001) { discard; }
    // The first guess at the height: the entry's off the plane over the mean direction's.
    let xe = o + d * t.x;
    guess = (dot(n, xe) - d0) / max(dot(n, dirs[0] + dirs[1] + dirs[2]) / 3.0, 1e-6);
    let enter = place_at(t.x);
    let leave = place_at(t.y);
    guess = enter.w;
    // Even steps along the foot's way, each a height cell (or more, smoothed) or, far off, where
    // that's smaller than a pixel, a pixel's width there; until it's met, then halvings.
    // Smoothed (u.pos.w levels), the height can't change much within half its smoothing's
    // cells, so the steps are as long as that.
    let along = length(foot_of(leave.xyz) - foot_of(enter.xyz));
    let cells = f32(u.o.w / HEIGHT_CELL);
    let pixel = 0.5 * (t.x + t.y) / u.k.x;
    let cell_step = exp2(max(u.pos.w - 1.0, 0.0)) / cells;
    let steps = clamp(u32(ceil(along / max(cell_step, pixel))), 2u, MOST_STEPS);
    var t0 = t.x;
    var t1 = t.x;
    var w = enter;
    // The last sample's place (at the far end, once all's been stepped).
    var last = leave;
    // Under the surface where it comes in: met there (coming on from the tetrahedron before,
    // which met it a hair short of their shared face).
    var met = above_place(enter) <= 0.0;
    for (var i = 1u; i <= steps && !met; i++) {
        let ti = mix(t.x, t.y, f32(i) / f32(steps));
        let wi = place_at(ti);
        last = wi;
        if (above_place(wi) <= 0.0) {
            t1 = ti;
            w = wi;
            met = true;
        } else {
            t0 = ti;
        }
    }
    // Out through a face without meeting it: the next tetrahedron has it, unless that's the
    // inner shell, as deep as a hollow goes, met there.
    if (!met) {
        if (last.w > -1.0 + 1e-4) { discard; }
        t0 = t.y;
        t1 = t.y;
        w = last;
    }
    // Halved no finer than u.r1.w of a pixel's width (0: all HALVINGS), along the line of
    // sight, where it matters as finely as the eye has it and no finer.
    let fine = u.r1.w / (u.k.x * length(d));
    if (t1 > t0) {
        for (var i = 0u; i < HALVINGS; i++) {
            if (t1 - t0 < fine * t0) { break; }
            let tm = 0.5 * (t0 + t1);
            let wm = place_at(tm);
            if (above_place(wm) <= 0.0) {
                t1 = tm;
                w = wm;
            } else {
                t0 = tm;
            }
        }
    }
    var out: Marched;
    out.found = vec4<u32>(in.tri + 1u, bitcast<u32>(w.y), bitcast<u32>(w.z), bitcast<u32>(t1));
    out.depth = NEAR / t1;
    return out;
}

// The march's hit on triangle `tri` (begun) at barycentrics w and view depth t1, finished: the
// height's slope there, its page, and what the raster would have at the foot.
fn finish(tri: u32, w: vec4<f32>, t1: f32) -> Hit {
    let ta = tri * TRI;
    let info = corners[ta + 3u];
    let cells = f32(u.o.w / HEIGHT_CELL);
    let foot = foot_of(w.xyz);
    let held = dot(w.xyz, reaches);
    // The height's slope along the page (tiles a tile), a height cell either side along the
    // page's two axes, kept on the plane (across a page's edge, the next page's), tapered.
    let ax = page_axes();
    let e = 1.0 / cells;
    var slope = vec2<f32>(0.0);
    for (var k = 0u; k < 2u; k++) {
        let pa = foot + ax[k] * e;
        let pb = foot - ax[k] * e;
        let ha = clamp(height_at(pa - n * (dot(n, pa) - d0)), -held, held);
        let hb = clamp(height_at(pb - n * (dot(n, pb) - d0)), -held, held);
        slope[k] = (ha - hb) / (2.0 * e);
    }
    // The page under the foot itself.
    page = page_at(foot, own);
    // What the raster would have at the foot.
    let uv01 = corners[ta + 4u];
    let uv2 = corners[ta + 5u].xy;
    let uv = uv01.xy * w.x + uv01.zw * w.y + uv2 * w.z;
    let light = corners[ta + 6u].xyz * w.x + corners[ta + 7u].xyz * w.y + corners[ta + 8u].xyz * w.z;
    let albedo = corners[ta + 9u].xyz * w.x + corners[ta + 10u].xyz * w.y + corners[ta + 11u].xyz * w.z;
    let layer = bitcast<u32>(info.y);
    let texel = textureSampleLevel(textures, nearest, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), layer, 0.0).rgb;
    var out: Hit;
    let place = pick(foot - floor(foot), axis);
    out.look = vec4<u32>(pack2x16unorm(place), pack4x8unorm(vec4<f32>(0.5 * light, 1.0)), pack4x8unorm(vec4<f32>(0.5 * albedo, 1.0)), pack4x8unorm(vec4<f32>(texel, 1.0)));
    out.tri = vec4<u32>(tri + 1u, pack2x16float(slope), bitcast<u32>(t1), select(NONE, u32(page), page >= 0.0) & PAGE_BITS);
    return out;
}
