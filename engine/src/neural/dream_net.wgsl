// Kernels for dream_net.rs, appended to gpu_net.wgsl (its Params, `p`, `mem` and conv3x3).

// Floats per part in `mem` (dream_net.rs PART_FLOATS): the axis's 5 points as x, y (crop
// pixels), z; radius, half-length, noise seed, light rgb; the phrase vector; the colour
// scheme; the part's local x and z axes in camera space.
const PART_RADIUS: u32 = 15u;
const PART_HALF: u32 = 16u;
const PART_SEED: u32 = 17u;
const PART_LIGHT: u32 = 18u;
const PART_PHRASE: u32 = 22u;
const PART_PALETTE: u32 = 86u;
const PART_AX: u32 = 98u;
const PART_AZ: u32 = 101u;
const PART_FLOATS: u32 = 104u;
const PHRASE_DIM: u32 = 64u;
const MAX_PARTS: u32 = 96u;
const FIELDS: u32 = 39u; // dream.py FIELD_CHANNELS
const NOISE: u32 = 4u;
const OCTAVES: u32 = 3u;
const SCREEN_H: f32 = 240.0;
// Where in the fields the noise, the scheme and the surface go.
const OUT_NOISE: u32 = 9u;
const OUT_PALETTE: u32 = 21u;
const OUT_NORMAL: u32 = 33u;
const OUT_AROUND: u32 = 36u;
const OUT_CREASE: u32 = 38u;
// Planes `fields` writes: the net's input (fields, then the phrase vectors), the nearest
// part's colour scheme, its fog and its surface's depth.
const PLANE_PALETTE: u32 = 103u;
const PLANE_FOG: u32 = 115u;
const PLANE_DEPTH: u32 = 116u;
// Where the drawn edge of a limb is, in its radii (dream.py EDGE).
const EDGE: f32 = 1.25;
// How widely parts melt into each other at their joints, and how much the outline swells there.
const MELT: f32 = 1.0;
const FILLET: f32 = 2.2;

fn dream_hash(i: vec3<i32>, seed: u32, channel: u32) -> f32 {
    var h = (u32(i.x) * 73856093u) ^ (u32(i.y) * 19349663u) ^ (u32(i.z) * 83492791u) ^ (seed * 2654435761u) ^ (channel * 40503u);
    h = (h ^ (h >> 16u)) * 0x45D9F3Bu;
    h = (h ^ (h >> 16u)) * 0x45D9F3Bu;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFu) / 65535.0;
}

// Value noise in -1..1 (creatures.py noise3).
fn dream_noise(at: vec3<f32>, seed: u32, channel: u32) -> f32 {
    let lattice = floor(at);
    let i = vec3<i32>(lattice);
    var f = at - lattice;
    f = f * f * (3.0 - 2.0 * f);
    var out = 0.0;
    for (var c = 0; c < 8; c++) {
        let off = vec3<i32>(c & 1, (c >> 1u) & 1, c >> 2u);
        let w = select(1.0 - f.x, f.x, off.x == 1) * select(1.0 - f.y, f.y, off.y == 1) * select(1.0 - f.z, f.z, off.z == 1);
        out += w * dream_hash(i + off, seed, channel);
    }
    return out * 2.0 - 1.0;
}

fn unit(v: vec3<f32>) -> vec3<f32> {
    return v / max(length(v), 1e-6);
}

fn part_vec3(at: u32) -> vec3<f32> {
    return vec3<f32>(mem[at], mem[at + 1u], mem[at + 2u]);
}

// One part at a pixel: its nearest axis segment and what that says.
struct Limb {
    d: f32,
    along: f32,
    across: f32,
    depth: f32,
    rpix: f32,
    off: vec2<f32>,
    tangent: vec3<f32>,
}

fn limb(base: u32, at: vec2<f32>, corner: vec2<f32>) -> Limb {
    let radius = mem[base + PART_RADIUS];
    var l: Limb;
    l.d = 1e9;
    for (var s = 0u; s < 4u; s++) {
        let i = base + s * 3u;
        let a = vec2<f32>(mem[i], mem[i + 1u]);
        let b = vec2<f32>(mem[i + 3u], mem[i + 4u]);
        let ab = b - a;
        let ap = at - a;
        let u = clamp(dot(ap, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
        let off = ap - u * ab;
        let dist = length(off);
        let z = mem[i + 2u] + u * (mem[i + 5u] - mem[i + 2u]);
        let r = max(radius * p.f0 / z, 0.5);
        let d = dist / r;
        if (d < l.d) {
            l.d = d;
            l.along = -1.0 + 0.5 * (f32(s) + u);
            l.across = sign(ab.x * ap.y - ab.y * ap.x) * dist / r;
            l.depth = z;
            l.rpix = r;
            l.off = off;
            let pa = vec3<f32>((a + corner) / p.f0 * mem[i + 2u], mem[i + 2u]);
            let pb = vec3<f32>((b + corner) / p.f0 * mem[i + 5u], mem[i + 5u]);
            l.tangent = unit(pb - pa);
        }
    }
    return l;
}

// The limb's surface normal: round it from the pixel's side to facing the camera.
fn limb_normal(l: Limb, ray: vec3<f32>) -> vec3<f32> {
    let across_limb = unit(cross(l.tangent, ray));
    let side = sign(dot(across_limb.xy, l.off));
    let facing = -unit(ray - l.tangent * dot(ray, l.tangent));
    let round = min(l.d / EDGE, 1.0);
    return across_limb * side * round + facing * sqrt(max(1.0 - round * round, 0.0));
}

// (cos, sin) round the limb and the noise fixed to that.
struct Skin {
    around: vec2<f32>,
    noise: array<f32, 12>,
}

fn skin(l: Limb, base: u32, normal: vec3<f32>) -> Skin {
    var out: Skin;
    let around = vec2<f32>(dot(normal, part_vec3(base + PART_AX)), dot(normal, part_vec3(base + PART_AZ)));
    out.around = around / max(length(around), 1e-3);
    let seed = u32(mem[base + PART_SEED]);
    let place = vec3<f32>(l.along * mem[base + PART_HALF] / max(mem[base + PART_RADIUS], 1e-3), out.around.x, out.around.y);
    for (var c = 0u; c < NOISE; c++) {
        for (var o = 0u; o < OCTAVES; o++) {
            out.noise[c * OCTAVES + o] = dream_noise(place * f32(1u << o), seed, c * OCTAVES + o);
        }
    }
    return out;
}

// The per-pixel fields of dream.py `fields`, as planes [PLANE_DEPTH + 1][m][n] at c_off, for
// k parts at a_off. f0: projection distance; f1, f2: fog near and far; f3: half the screen's
// width; x0, x1: the crop's corner on the screen.
@compute @workgroup_size(16, 16)
fn dream_fields(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.n || y >= p.m) { return; }
    let plane = p.m * p.n;
    let pixel = y * p.n + x;
    let at = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
    let corner = vec2<f32>(f32(p.x0) - p.f3, f32(p.x1) - SCREEN_H / 2.0);
    var ds: array<f32, MAX_PARTS>;
    var zs: array<f32, MAX_PARTS>; // each part's surface depth here
    var ns: array<vec3<f32>, MAX_PARTS>; // and its normal
    let ray = unit(vec3<f32>((at + corner) / p.f0, 1.0));
    var dmin = 1e4;
    var closest = 1e4; // the smallest distance, whichever part is on top
    var nearest = 0u;
    var near_surface = 1e9;
    for (var q = 0u; q < p.k; q++) {
        let base = p.a_off + q * PART_FLOATS;
        let l = limb(base, at, corner);
        ds[q] = l.d;
        ns[q] = limb_normal(l, ray);
        closest = min(closest, l.d);
        let round = min(l.d / EDGE, 1.0);
        let surface = l.depth - mem[base + PART_RADIUS] * EDGE * sqrt(max(1.0 - round * round, 0.0));
        zs[q] = surface;
        // Among the limbs covering this pixel the nearest surface wins.
        let inside = l.d < EDGE;
        let held = dmin < EDGE;
        if ((inside && !held) || (inside == held && select(l.d < dmin, surface < near_surface, inside))) {
            near_surface = surface;
            dmin = l.d;
            nearest = q;
        }
    }
    let base = p.a_off + nearest * PART_FLOATS;
    let near = dmin < 4.0;
    let radius = mem[base + PART_RADIUS];
    // The outline is a smooth minimum, so joints swell; the part this one melts into is the
    // strongest of those level with its surface (one clearly behind counts for nothing).
    var sum = 0.0;
    var partner = 0u;
    var partner_strength = 0.0;
    var normal_sum = vec3<f32>(0.0);
    for (var q = 0u; q < p.k; q++) {
        sum += exp(-FILLET * (ds[q] - closest));
        let wide = max(mem[p.a_off + q * PART_FLOATS + PART_RADIUS], radius);
        let s = exp(-MELT * ds[q] * ds[q]) * (1.0 - smoothstep(wide, 3.0 * wide, abs(zs[q] - near_surface)));
        normal_sum += s * ns[q];
        if (q != nearest) {
            if (s > partner_strength) {
                partner_strength = s;
                partner = q;
            }
        }
    }
    let melted = closest - log(sum) / FILLET;
    let t = partner_strength / max(exp(-MELT * dmin * dmin) + partner_strength, 1e-12);
    let base_b = p.a_off + partner * PART_FLOATS;
    var out: array<f32, FIELDS>;
    var near_depth = 0.0;
    if (near) {
        let a = limb(base, at, corner);
        let b = limb(base_b, at, corner);
        let sa = skin(a, base, ns[nearest]);
        let sb = skin(b, base_b, ns[partner]);
        near_depth = mix(a.depth, b.depth, t);
        out[0] = exp(-0.5 * melted * melted);
        out[1] = min(mix(a.d, b.d, t), 8.0) / 4.0;
        out[2] = mix(a.along, b.along, t);
        out[3] = clamp(mix(a.across, b.across, t), -8.0, 8.0) / 4.0;
        out[4] = near_depth / 8.0;
        out[5] = log(mix(a.rpix, b.rpix, t)) / 4.0;
        for (var c = 0u; c < 3u; c++) { out[6u + c] = mix(mem[base + PART_LIGHT + c], mem[base_b + PART_LIGHT + c], t); }
        for (var c = 0u; c < NOISE * OCTAVES; c++) { out[OUT_NOISE + c] = mix(sa.noise[c], sb.noise[c], t); }
        for (var c = 0u; c < 12u; c++) { out[OUT_PALETTE + c] = mix(mem[base + PART_PALETTE + c], mem[base_b + PART_PALETTE + c], t); }
        // The shading shows seams most, so the normal blends over every part level with the surface.
        let normal = unit(normal_sum);
        for (var c = 0u; c < 3u; c++) { out[OUT_NORMAL + c] = normal[c]; }
        let around = mix(sa.around, sb.around, t);
        let dir = around / max(length(around), 1e-3);
        out[OUT_AROUND] = dir.x;
        out[OUT_AROUND + 1u] = dir.y;
        var crease = 0.0;
        for (var q = 0u; q < p.k; q++) {
            // A soft ring round limbs clearly in front of this surface; joints get none.
            let ahead = mix(near_surface, zs[partner], t) - zs[q];
            let ring = smoothstep(0.9, EDGE, ds[q]) * (1.0 - smoothstep(EDGE, 2.1, ds[q]));
            crease = max(crease, ring * smoothstep(0.4 * radius, 1.2 * radius, ahead) * (1.0 - smoothstep(2.0 * radius, 6.0 * radius, ahead)));
        }
        out[OUT_CREASE] = crease;
    }
    for (var c = 0u; c < FIELDS; c++) { mem[p.c_off + c * plane + pixel] = out[c]; }
    // The parts' phrase vectors, blended by a softmax over how near each is.
    var emb: array<f32, PHRASE_DIM>;
    if (near) {
        let top = -2.0 * closest * closest;
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
    for (var c = 0u; c < 12u; c++) { mem[p.c_off + (PLANE_PALETTE + c) * plane + pixel] = mix(mem[base + PART_PALETTE + c], mem[base_b + PART_PALETTE + c], t); }
    let fog = clamp((near_depth - p.f1) / max(p.f2 - p.f1, 1e-3), 0.0, 1.0);
    mem[p.c_off + PLANE_FOG * plane + pixel] = fog;
    mem[p.c_off + PLANE_DEPTH * plane + pixel] = mix(near_surface, zs[partner], t);
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

// Halves planes [z][2m][2n] at a_off to [z][m][n] at c_off, as F.avg_pool2d(2); z per dispatch depth.
@compute @workgroup_size(16, 16)
fn dream_pool2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.n || y >= p.m) { return; }
    let wide = p.n * 2u;
    let src = p.a_off + gid.z * p.m * 2u * wide;
    let row = y * 2u * wide + x * 2u;
    mem[p.c_off + gid.z * p.m * p.n + y * p.n + x] = 0.25 * (mem[src + row] + mem[src + row + 1u] + mem[src + row + wide] + mem[src + row + wide + 1u]);
}

// DreamNet.film in place on planes [z][m][n] at a_off: x * (1 + g) + b, with g then b, [2z][m][n], at b_off.
@compute @workgroup_size(16, 16)
fn dream_film(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let plane = p.m * p.n;
    let pixel = gid.y * p.n + gid.x;
    let at = p.a_off + gid.z * plane + pixel;
    let g = mem[p.b_off + gid.z * plane + pixel];
    let b = mem[p.b_off + (p.k + gid.z) * plane + pixel];
    mem[at] = mem[at] * (1.0 + g) + b;
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
    // dream.py lighting: a light above and to the left of the viewer (y is down, z is away).
    let n = vec3<f32>(mem[p.b_off + OUT_NORMAL * plane + i], mem[p.b_off + (OUT_NORMAL + 1u) * plane + i], mem[p.b_off + (OUT_NORMAL + 2u) * plane + i]);
    let wrap = clamp((dot(n, normalize(vec3<f32>(-0.3, -0.6, -0.74))) + 0.35) / 1.35, 0.0, 1.0);
    let edge = 1.0 + n.z;
    let crease = mem[p.b_off + OUT_CREASE * plane + i];
    let light = (0.42 + 0.72 * wrap + 0.2 * edge * edge * edge) * (1.0 - 0.3 * crease);
    let rgb = clamp(mixed * shade * light, vec3<f32>(0.0), vec3<f32>(1.0));
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
