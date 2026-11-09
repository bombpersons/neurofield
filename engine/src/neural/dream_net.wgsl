// Kernels for dream_net.rs, appended to gpu_net.wgsl (its Params, `p`, `mem` and conv3x3).

// Floats per part in `mem` (dream_net.rs PART_FLOATS): the axis's 5 points as x, y (crop
// pixels), z; radius, half-length, noise seed, light rgb; the phrase vector; the colour scheme.
const PART_FLOATS: u32 = 66u;
const PART_RADIUS: u32 = 15u;
const PART_HALF: u32 = 16u;
const PART_SEED: u32 = 17u;
const PART_LIGHT: u32 = 18u;
const PART_PHRASE: u32 = 22u;
const PART_PALETTE: u32 = 54u;
const PHRASE_DIM: u32 = 32u;
const MAX_PARTS: u32 = 32u;
const FIELDS: u32 = 29u; // dream.py FIELD_CHANNELS
// Planes `fields` writes: the net's input (fields, then the phrase vectors), the nearest
// part's colour scheme, its fog and its depth.
const PLANE_PALETTE: u32 = 61u;
const PLANE_FOG: u32 = 73u;
const PLANE_DEPTH: u32 = 74u;

fn dream_hash(i: vec2<i32>, seed: u32, channel: u32) -> f32 {
    // The third lattice coordinate is always 0 here (creatures.py hash3).
    var h = (u32(i.x) * 73856093u) ^ (u32(i.y) * 19349663u) ^ (seed * 2654435761u) ^ (channel * 40503u);
    h = (h ^ (h >> 16u)) * 0x45D9F3Bu;
    h = (h ^ (h >> 16u)) * 0x45D9F3Bu;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFu) / 65535.0;
}

// Value noise in -1..1 (creatures.py noise3, at z = 0).
fn dream_noise(at: vec2<f32>, seed: u32, channel: u32) -> f32 {
    let lattice = floor(at);
    let i = vec2<i32>(lattice);
    var f = at - lattice;
    f = f * f * (3.0 - 2.0 * f);
    let n00 = dream_hash(i, seed, channel);
    let n10 = dream_hash(i + vec2<i32>(1, 0), seed, channel);
    let n01 = dream_hash(i + vec2<i32>(0, 1), seed, channel);
    let n11 = dream_hash(i + vec2<i32>(1, 1), seed, channel);
    return mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y) * 2.0 - 1.0;
}

// The per-pixel fields of dream.py `fields`, as planes [PLANE_DEPTH + 1][m][n] at c_off, for
// k parts at a_off. f0: projection distance; f1, f2: fog near and far.
@compute @workgroup_size(16, 16)
fn dream_fields(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.n || y >= p.m) { return; }
    let plane = p.m * p.n;
    let pixel = y * p.n + x;
    let at = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
    var ds: array<f32, MAX_PARTS>;
    var dmin = 1e4;
    var nearest = 0u;
    var near_along = 0.0;
    var near_across = 0.0;
    var near_depth = 0.0;
    var near_rpix = 0.0;
    for (var q = 0u; q < p.k; q++) {
        let base = p.a_off + q * PART_FLOATS;
        let radius = mem[base + PART_RADIUS];
        var best = 1e9;
        var along = 0.0;
        var across = 0.0;
        var depth = 0.0;
        var rpix = 0.0;
        for (var s = 0u; s < 4u; s++) {
            let i = base + s * 3u;
            let a = vec2<f32>(mem[i], mem[i + 1u]);
            let b = vec2<f32>(mem[i + 3u], mem[i + 4u]);
            let ab = b - a;
            let ap = at - a;
            let u = clamp(dot(ap, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
            let dist = length(ap - u * ab);
            let z = mem[i + 2u] + u * (mem[i + 5u] - mem[i + 2u]);
            let r = max(radius * p.f0 / z, 0.5);
            let d = dist / r;
            if (d < best) {
                best = d;
                along = -1.0 + 0.5 * (f32(s) + u);
                across = sign(ab.x * ap.y - ab.y * ap.x) * dist / r;
                depth = z;
                rpix = r;
            }
        }
        ds[q] = best;
        if (best < dmin) {
            dmin = best;
            nearest = q;
            near_along = along;
            near_across = across;
            near_depth = depth;
            near_rpix = rpix;
        }
    }
    let base = p.a_off + nearest * PART_FLOATS;
    let near = dmin < 4.0;
    var out: array<f32, FIELDS>;
    let radius = mem[base + PART_RADIUS];
    if (near) {
        out[0] = exp(-0.5 * dmin * dmin);
        out[1] = min(dmin, 8.0) / 4.0;
        out[2] = near_along;
        out[3] = clamp(near_across, -8.0, 8.0) / 4.0;
        out[4] = near_depth / 8.0;
        out[5] = log(near_rpix) / 4.0;
        for (var c = 0u; c < 3u; c++) { out[6u + c] = mem[base + PART_LIGHT + c]; }
        let seed = u32(mem[base + PART_SEED]);
        let noise_at = vec2<f32>(near_along * mem[base + PART_HALF] / max(radius, 1e-3), near_across);
        for (var c = 0u; c < 4u; c++) {
            for (var o = 0u; o < 2u; o++) {
                out[9u + c * 2u + o] = dream_noise(noise_at * f32(1u << o), seed, c * 2u + o);
            }
        }
        for (var c = 0u; c < 12u; c++) { out[17u + c] = mem[base + PART_PALETTE + c]; }
    }
    for (var c = 0u; c < FIELDS; c++) { mem[p.c_off + c * plane + pixel] = out[c]; }
    // The parts' phrase vectors, blended by a softmax over how near each is.
    var emb: array<f32, PHRASE_DIM>;
    if (near) {
        let top = -2.0 * dmin * dmin;
        var total = 0.0;
        for (var q = 0u; q < p.k; q++) {
            let w = exp(-2.0 * ds[q] * ds[q] - top);
            total += w;
            let phrase = p.a_off + q * PART_FLOATS + PART_PHRASE;
            for (var c = 0u; c < PHRASE_DIM; c++) { emb[c] += w * mem[phrase + c]; }
        }
        for (var c = 0u; c < PHRASE_DIM; c++) { emb[c] /= total; }
    }
    for (var c = 0u; c < PHRASE_DIM; c++) { mem[p.c_off + (FIELDS + c) * plane + pixel] = emb[c]; }
    // The scheme is written even far from every part: the net's mix uses it everywhere.
    for (var c = 0u; c < 12u; c++) { mem[p.c_off + (PLANE_PALETTE + c) * plane + pixel] = mem[base + PART_PALETTE + c]; }
    let fog = clamp((near_depth - p.f1) / max(p.f2 - p.f1, 1e-3), 0.0, 1.0);
    mem[p.c_off + PLANE_FOG * plane + pixel] = fog;
    mem[p.c_off + PLANE_DEPTH * plane + pixel] = near_depth;
}

// Doubles k planes at a_off to [k][m][n] at c_off, as F.interpolate(bilinear, align_corners=False).
@compute @workgroup_size(16, 16)
fn upsample2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;
    if (x >= p.n || y >= p.m) { return; }
    let hm = i32(p.m / 2u);
    let hn = i32(p.n / 2u);
    let src = p.a_off + c * u32(hm * hn);
    let ox = 2 * i32(x & 1u) - 1;
    let oy = 2 * i32(y & 1u) - 1;
    let x0 = i32(x / 2u);
    let y0 = i32(y / 2u);
    let x1 = clamp(x0 + ox, 0, hn - 1);
    let y1 = clamp(y0 + oy, 0, hm - 1);
    let a = mem[src + u32(y0 * hn + x0)];
    let b = mem[src + u32(y0 * hn + x1)];
    let cc = mem[src + u32(y1 * hn + x0)];
    let d = mem[src + u32(y1 * hn + x1)];
    mem[p.c_off + c * p.m * p.n + y * p.n + x] = 0.75 * (0.75 * a + 0.25 * b) + 0.25 * (0.75 * cc + 0.25 * d);
}

// The net's last step (DreamNet.forward): out [6][plane] at a_off and the fields' planes at
// b_off to [plane][6] at c_off (rgb, alpha, fog, depth). m: pixels.
@compute @workgroup_size(64)
fn dream_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.m) { return; }
    let plane = p.m;
    let o = p.a_off;
    var w = vec4<f32>(mem[o + i], mem[o + plane + i], mem[o + 2u * plane + i], mem[o + 3u * plane + i]) * 3.0;
    w = exp(w - max(max(w.x, w.y), max(w.z, w.w)));
    w /= w.x + w.y + w.z + w.w;
    var mixed = vec3<f32>(0.0);
    let scheme = p.b_off + PLANE_PALETTE * plane;
    for (var c = 0u; c < 3u; c++) {
        for (var k = 0u; k < 4u; k++) {
            mixed[c] += w[k] * mem[scheme + (k * 3u + c) * plane + i];
        }
    }
    let shade = 0.35 + 1.3 / (1.0 + exp(-(mem[o + 4u * plane + i] + 0.3)));
    let rgb = clamp(mixed * shade, vec3<f32>(0.0), vec3<f32>(1.0));
    let presence = mem[p.b_off + i];
    let logit = 10.0 * tanh((mem[o + 5u * plane + i] + 6.0 * presence - 3.0) / 10.0);
    let at = p.c_off + i * 6u;
    mem[at] = rgb.x;
    mem[at + 1u] = rgb.y;
    mem[at + 2u] = rgb.z;
    mem[at + 3u] = 1.0 / (1.0 + exp(-logit));
    mem[at + 4u] = mem[p.b_off + PLANE_FOG * plane + i];
    mem[at + 5u] = mem[p.b_off + PLANE_DEPTH * plane + i];
}
