// Each welded corner of the level leaned towards where it's been looked at from (shell.rs
// Shell::lean): the way it was dreamed from, so the relief stands out along the lines of sight
// the dream painted it on rather than straight off the wall. Each frame a corner in sight turns
// its way `follow` of the way towards the eye (out of sight it keeps it, as its pages keep the
// dream), and is pushed along its mean push (shell.rs welds) plus `lean` of that way's slant off
// it: an offset across the mean, so the shell still stands its full height off the walls, the
// same at every triangle meeting there, so their prisms still meet.

struct L {
    // The eye, and how far a corner in sight turns towards it a frame (0: never, 1: at once).
    eye: vec4<f32>,
    // The view's axes as rows: right (and the lean in w), down, forward.
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    // Focal length in shown pixels, half the shown width and height.
    k: vec4<f32>,
    // The welded corners' count.
    n: vec4<u32>,
}

@group(0) @binding(0) var<uniform> l: L;
// Each welded corner (BASE vec4s): its place, the shortest edge from it in w (0: it doesn't
// lean, too many walls meet there);
// the unit mean of the ways it's pushed, that mean's length in w; the normals of the walls
// meeting there it's pushed off (w 1, else none).
@group(0) @binding(1) var<storage, read> base: array<vec4<f32>>;
const WALLS: u32 = 4u;
const BASE: u32 = 6u;
// Each welded corner: the way it's been looked at from (unit, towards the eye); its offset
// across its pushes (shell.wgsl adds it to each, times their reach), the shortest edge from it
// in w (shell.wgsl leaned holds it to SHORT_HOLD of that).
@group(0) @binding(2) var<storage, read_write> state: array<vec4<f32>>;

// The most a corner's pushed across for each unit it's pushed out (about 63 degrees), so walls
// seen at a glancing angle aren't smeared along themselves.
const MOST_SLANT: f32 = 2.0;
// Seen from no nearer the wall's own plane than this (the cosine), else not turned.
const LEAST_FACING: f32 = 0.05;
// A corner this far past the screen's edge still counts as in sight (its triangles may be).
const MARGIN: f32 = 1.25;

@compute @workgroup_size(64)
fn lean_corners(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 64u * 65535u;
    if (i >= l.n.x) { return; }
    let place = base[BASE * i];
    let p = place.xyz;
    let mean = base[BASE * i + 1u];
    let a = mean.xyz;
    var s = state[2u * i].xyz;
    let q = p - l.eye.xyz;
    let z = dot(l.r2.xyz, q);
    let e = -normalize(q);
    let seen = z > 0.02
        && abs(l.k.x * dot(l.r0.xyz, q) / z) < l.k.y * MARGIN
        && abs(l.k.x * dot(l.r1.xyz, q) / z) < l.k.z * MARGIN
        && dot(e, a) > LEAST_FACING;
    if (seen && l.eye.w > 0.0) {
        let m = mix(s, e, l.eye.w);
        if (dot(m, m) > 1e-12) { s = normalize(m); }
    }
    // Its slant off the mean, per unit along it, held to MOST_SLANT; times the mean's length,
    // as the pushes are lengthened.
    let up = max(dot(s, a), 1e-3);
    var slant = (s - a * dot(s, a)) / up;
    let len = length(slant);
    if (len > MOST_SLANT) { slant *= MOST_SLANT / len; }
    var off = slant * mean.w * l.r0.w;
    // Leaning only along every wall meeting there (a corner where walls meet is pushed out
    // between them, so leaning across one pushes it into another, the edge's texels smeared
    // over the wedge past it): along a wall freely, along an edge between two only along the
    // edge, where three or more meet not at all. The walls' normals made orthonormal, each's
    // part taken out.
    var basis: array<vec3<f32>, 4>;
    var found = 0u;
    for (var w = 0u; w < WALLS; w++) {
        let m = base[BASE * i + 2u + w];
        if (m.w == 0.0) { continue; }
        var v = m.xyz;
        for (var j = 0u; j < found; j++) { v -= basis[j] * dot(v, basis[j]); }
        if (dot(v, v) > 1e-4 && found < 3u) {
            basis[found] = normalize(v);
            found++;
        }
    }
    for (var j = 0u; j < found; j++) { off -= basis[j] * dot(off, basis[j]); }
    off *= select(0.0, 1.0, place.w > 0.0 && found < 3u);
    state[2u * i] = vec4<f32>(s, 0.0);
    state[2u * i + 1u] = vec4<f32>(off, place.w);
}
