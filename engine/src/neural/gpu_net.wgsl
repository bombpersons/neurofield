// Kernels for gpu_net.rs: the renderer's forward pass as hand-written compute shaders.
//
// Everything lives in one storage buffer, `mem` (weights, inputs and activations, all f32);
// each dispatch gets its sizes and offsets (in floats) from its own slot of a uniform buffer.
// The field names below are for the matrix multiply; other kernels say how they use them.

struct Params {
    m: u32, k: u32, n: u32, batch: u32,
    a_off: u32, a_bs: u32, a_rs: u32, a_cs: u32,
    b_off: u32, b_bs: u32, b_rs: u32, b_cs: u32,
    c_off: u32, c_bs: u32, c_rs: u32, bias_off: u32,
    flags: u32, x0: u32, x1: u32, x2: u32,
    f0: f32, f1: f32, f2: f32, f3: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> mem: array<f32>;

const NONE: u32 = 0xffffffffu;
const GELU: u32 = 1u;
const ACCUMULATE: u32 = 2u;
const SHUFFLE: u32 = 4u;
const INTERLEAVE: u32 = 8u;
const PI: f32 = 3.14159265358979;

// erf by Abramowitz and Stegun 7.1.26 (absolute error under 1.5e-7): WGSL has no erf.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return select(-y, y, x >= 0.0);
}

// The exact (erf) GELU, as nn.GELU().
fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf(x * 0.70710678118654752));
}

// ---- Matrix multiply ----------------------------------------------------------------------
// C[z][i][j] (=, or += with ACCUMULATE) act(sum_t A[z][i][t] * B[z][t][j] + bias[j]) for
// i < m, j < n, t < k, z < batch. A[z][i][t] = mem[a_off + z*a_bs + i*a_rs + t*a_cs], B and C
// likewise (C's columns are contiguous); bias_off = NONE for no bias. A 32x32 tile of C per
// workgroup of 8x8 threads, 4x4 per thread, k in steps of 16 with the next step prefetched.
// Loads are branch-free: addresses are clamped into the matrix, and what lies outside is zeroed.

var<workgroup> ta: array<vec4<f32>, 128>; // [t 16][rows / 4: 8]
var<workgroup> tb: array<vec4<f32>, 128>; // [t 16][cols / 4: 8]

fn load_a(a_off: u32, kl: u32, row: u32, t: u32) -> vec4<f32> {
    let base = a_off + min(t, kl - 1u) * p.a_cs;
    let last = p.m - 1u;
    let v = vec4<f32>(mem[base + min(row, last) * p.a_rs], mem[base + min(row + 1u, last) * p.a_rs],
                      mem[base + min(row + 2u, last) * p.a_rs], mem[base + min(row + 3u, last) * p.a_rs]);
    let inside = vec4<bool>(row < p.m, row + 1u < p.m, row + 2u < p.m, row + 3u < p.m) & vec4<bool>(t < kl);
    return select(vec4<f32>(0.0), v, inside);
}

fn load_b(b_off: u32, kl: u32, t: u32, col: u32) -> vec4<f32> {
    let base = b_off + min(t, kl - 1u) * p.b_rs;
    let last = p.n - 1u;
    let v = vec4<f32>(mem[base + min(col, last) * p.b_cs], mem[base + min(col + 1u, last) * p.b_cs],
                      mem[base + min(col + 2u, last) * p.b_cs], mem[base + min(col + 3u, last) * p.b_cs]);
    let inside = vec4<bool>(col < p.n, col + 1u < p.n, col + 2u < p.n, col + 3u < p.n) & vec4<bool>(t < kl);
    return select(vec4<f32>(0.0), v, inside);
}

fn finish(v: f32, o: u32, col: u32) -> f32 {
    var x = v;
    if (p.bias_off != NONE) { x += mem[p.bias_off + col]; }
    if ((p.flags & GELU) != 0u) { x = gelu(x); }
    if ((p.flags & ACCUMULATE) != 0u) { x += mem[o]; }
    return x;
}

@compute @workgroup_size(8, 8)
fn matmul(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let z = wg.z;
    let kl = p.k;
    let a_off = p.a_off + z * p.a_bs;
    let b_off = p.b_off + z * p.b_bs;
    let row0 = wg.y * 32u;
    let col0 = wg.x * 32u;
    let tid = lid.y * 8u + lid.x;
    // 64 threads load 16 t x 8 row-quads of A and 16 t x 8 column-quads of B, two of each.
    let q = tid % 8u;
    let tk = tid / 8u;
    var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
    var na0 = load_a(a_off, kl, row0 + q * 4u, tk); var na1 = load_a(a_off, kl, row0 + q * 4u, tk + 8u);
    var nb0 = load_b(b_off, kl, tk, col0 + q * 4u); var nb1 = load_b(b_off, kl, tk + 8u, col0 + q * 4u);
    for (var t0 = 0u; t0 < kl; t0 += 16u) {
        ta[tk * 8u + q] = na0; ta[(tk + 8u) * 8u + q] = na1;
        tb[tk * 8u + q] = nb0; tb[(tk + 8u) * 8u + q] = nb1;
        workgroupBarrier();
        na0 = load_a(a_off, kl, row0 + q * 4u, t0 + 16u + tk); na1 = load_a(a_off, kl, row0 + q * 4u, t0 + 24u + tk);
        nb0 = load_b(b_off, kl, t0 + 16u + tk, col0 + q * 4u); nb1 = load_b(b_off, kl, t0 + 24u + tk, col0 + q * 4u);
        // Unrolled by hand (WGSL has no unroll hint); a rolled loop waits on each read in turn.
        let x0 = ta[0u + lid.y]; let y0 = tb[0u + lid.x]; let x1 = ta[8u + lid.y]; let y1 = tb[8u + lid.x];
        let x2 = ta[16u + lid.y]; let y2 = tb[16u + lid.x]; let x3 = ta[24u + lid.y]; let y3 = tb[24u + lid.x];
        let x4 = ta[32u + lid.y]; let y4 = tb[32u + lid.x]; let x5 = ta[40u + lid.y]; let y5 = tb[40u + lid.x];
        let x6 = ta[48u + lid.y]; let y6 = tb[48u + lid.x]; let x7 = ta[56u + lid.y]; let y7 = tb[56u + lid.x];
        let x8 = ta[64u + lid.y]; let y8 = tb[64u + lid.x]; let x9 = ta[72u + lid.y]; let y9 = tb[72u + lid.x];
        let xa = ta[80u + lid.y]; let ya = tb[80u + lid.x]; let xb = ta[88u + lid.y]; let yb = tb[88u + lid.x];
        let xc = ta[96u + lid.y]; let yc = tb[96u + lid.x]; let xd = ta[104u + lid.y]; let yd = tb[104u + lid.x];
        let xe = ta[112u + lid.y]; let ye = tb[112u + lid.x]; let xf = ta[120u + lid.y]; let yf = tb[120u + lid.x];
        c0 += x0.x * y0; c1 += x0.y * y0; c2 += x0.z * y0; c3 += x0.w * y0;
        c0 += x1.x * y1; c1 += x1.y * y1; c2 += x1.z * y1; c3 += x1.w * y1;
        c0 += x2.x * y2; c1 += x2.y * y2; c2 += x2.z * y2; c3 += x2.w * y2;
        c0 += x3.x * y3; c1 += x3.y * y3; c2 += x3.z * y3; c3 += x3.w * y3;
        c0 += x4.x * y4; c1 += x4.y * y4; c2 += x4.z * y4; c3 += x4.w * y4;
        c0 += x5.x * y5; c1 += x5.y * y5; c2 += x5.z * y5; c3 += x5.w * y5;
        c0 += x6.x * y6; c1 += x6.y * y6; c2 += x6.z * y6; c3 += x6.w * y6;
        c0 += x7.x * y7; c1 += x7.y * y7; c2 += x7.z * y7; c3 += x7.w * y7;
        c0 += x8.x * y8; c1 += x8.y * y8; c2 += x8.z * y8; c3 += x8.w * y8;
        c0 += x9.x * y9; c1 += x9.y * y9; c2 += x9.z * y9; c3 += x9.w * y9;
        c0 += xa.x * ya; c1 += xa.y * ya; c2 += xa.z * ya; c3 += xa.w * ya;
        c0 += xb.x * yb; c1 += xb.y * yb; c2 += xb.z * yb; c3 += xb.w * yb;
        c0 += xc.x * yc; c1 += xc.y * yc; c2 += xc.z * yc; c3 += xc.w * yc;
        c0 += xd.x * yd; c1 += xd.y * yd; c2 += xd.z * yd; c3 += xd.w * yd;
        c0 += xe.x * ye; c1 += xe.y * ye; c2 += xe.z * ye; c3 += xe.w * ye;
        c0 += xf.x * yf; c1 += xf.y * yf; c2 += xf.z * yf; c3 += xf.w * yf;
        workgroupBarrier();
    }
    let col = col0 + lid.x * 4u;
    let rows = array<vec4<f32>, 4>(c0, c1, c2, c3);
    for (var i = 0u; i < 4u; i++) {
        let row = row0 + lid.y * 4u + i;
        if (row < p.m) {
            let r = rows[i];
            let o = p.c_off + z * p.c_bs + row * p.c_rs + col;
            if (col < p.n) { mem[o] = finish(r.x, o, col); }
            if (col + 1u < p.n) { mem[o + 1u] = finish(r.y, o + 1u, col + 1u); }
            if (col + 2u < p.n) { mem[o + 2u] = finish(r.z, o + 2u, col + 2u); }
            if (col + 3u < p.n) { mem[o + 3u] = finish(r.w, o + 3u, col + 3u); }
        }
    }
}

// ---- Row reductions -------------------------------------------------------------------------

var<workgroup> red: array<f32, 64>;

// Sum (or max, with `is_max`) of each thread's `v` across the 64-thread workgroup.
fn reduce(v: f32, lid: u32, is_max: bool) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (lid < s) {
            red[lid] = select(red[lid] + red[lid + s], max(red[lid], red[lid + s]), is_max);
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// Layer norm (eps 1e-5) of rows: m rows of n; in at a_off (row stride a_rs), out at c_off
// (row stride c_rs), weight at b_off, bias at bias_off. One workgroup per row.
@compute @workgroup_size(64)
fn layer_norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let row = wg.x;
    let lid = lid3.x;
    let x = p.a_off + row * p.a_rs;
    var s = 0.0;
    for (var j = lid; j < p.n; j += 64u) { s += mem[x + j]; }
    let mean = reduce(s, lid, false) / f32(p.n);
    var v = 0.0;
    for (var j = lid; j < p.n; j += 64u) { let d = mem[x + j] - mean; v += d * d; }
    let inv = 1.0 / sqrt(reduce(v, lid, false) / f32(p.n) + 1e-5);
    let o = p.c_off + row * p.c_rs;
    for (var j = lid; j < p.n; j += 64u) {
        mem[o + j] = (mem[x + j] - mean) * inv * mem[p.b_off + j] + mem[p.bias_off + j];
    }
}

// Attention softmax in place over scores [batch (heads)][m][n] at a_off: each score is scaled
// by f0, then, if b_off != NONE, has strength[head] * dist[row][col] taken off (strength at
// b_off + head, dist at x1 with row stride n: the coverage bias; from column k on, if k > 0,
// the organs' strength at c_off + head instead), and columns from x2 on are masked out
// (padding). One workgroup per row.
@compute @workgroup_size(64)
fn softmax(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let head = wg.y;
    let row = wg.x;
    let lid = lid3.x;
    let x = p.a_off + (head * p.m + row) * p.n;
    let cover = p.b_off != NONE;
    var strength = 0.0;
    var organ_strength = 0.0;
    if (cover) { strength = mem[p.b_off + head]; }
    if (cover && p.k > 0u) { organ_strength = mem[p.c_off + head]; }
    var mx = -3.0e38;
    for (var j = lid; j < p.x2; j += 64u) {
        var s = mem[x + j] * p.f0;
        if (cover) { s -= select(strength, organ_strength, p.k > 0u && j >= p.k) * mem[p.x1 + row * p.n + j]; }
        mem[x + j] = s;
        mx = max(mx, s);
    }
    mx = reduce(mx, lid, true);
    var sum = 0.0;
    for (var j = lid; j < p.x2; j += 64u) { let e = exp(mem[x + j] - mx); mem[x + j] = e; sum += e; }
    let inv = 1.0 / reduce(sum, lid, false);
    for (var j = lid; j < p.n; j += 64u) { mem[x + j] = select(0.0, mem[x + j] * inv, j < p.x2); }
}

// ---- Inputs ---------------------------------------------------------------------------------

// The triangle (or organ) MLP's input: m rows of [features (k of them), Fourier features of
// the coordinates from column x0 on (16 each: sin at 8 octaves from pi / 4, then cos)], from
// features at a_off (row stride a_rs) to c_off (row stride n, k + 16 a coordinate). Angles
// are reduced in turns first, exactly, since GPU sin and cos lose accuracy far from zero.
@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= p.m || col >= p.n) { return; }
    var v: f32;
    if (col < p.k) {
        v = mem[p.a_off + row * p.a_rs + col];
    } else {
        let c = col - p.k;
        let x = mem[p.a_off + row * p.a_rs + p.x0 + c / 16u];
        let r = c % 16u;
        // angle = x * 2^(band - 2) * pi = 2 pi * (x * 2^(band - 3)) turns
        let turns = fract(x * exp2(f32(r % 8u) - 3.0));
        let a = 2.0 * PI * turns;
        v = select(cos(a), sin(a), r < 8u);
    }
    mem[p.c_off + row * p.n + col] = v;
}

// Adds texture embeddings: for row i < m, C[1 + i] += table[tex[i]] (rows of n), with the
// texture ids (as floats) at a_off, the table at b_off and C at c_off.
@compute @workgroup_size(64)
fn add_texture(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= p.m || col >= p.n) { return; }
    let t = u32(mem[p.a_off + row]);
    mem[p.c_off + (1u + row) * p.n + col] += mem[p.b_off + t * p.n + col];
}

// Coverage distances, as model.py cover_distance and box_distance: for patch i < m and token
// j < n, the distance in patch widths (f0) from the patch centre (x, y at b_off + 2i) to the
// screen bounding box of triangle j - 1 (NDC at features a_off + (j - 1) * k + 9), 0 for the
// camera token j = 0; from j = x0 on, of organ j - x0's screen box (x0, y0, x1, y1 at x1 +
// (j - x0) * 23 + 12); written to c_off + i * n + j.
@compute @workgroup_size(64)
fn cover_distance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let i = gid.y;
    if (i >= p.m || j >= p.n) { return; }
    var d = 0.0;
    if (j >= p.x0 && p.x0 > 0u) {
        let f = p.x1 + (j - p.x0) * 23u + 12u;
        let lo = vec2<f32>(mem[f], mem[f + 1u]);
        let hi = vec2<f32>(mem[f + 2u], mem[f + 3u]);
        let c = vec2<f32>(mem[p.b_off + 2u * i], mem[p.b_off + 2u * i + 1u]);
        let gap = max(lo - c, vec2<f32>(0.0)) + max(c - hi, vec2<f32>(0.0));
        d = length(gap) / p.f0;
    } else if (j > 0u) {
        let f = p.a_off + (j - 1u) * p.k + 9u;
        let lo = vec2<f32>(min(min(mem[f], mem[f + 2u]), mem[f + 4u]), min(min(mem[f + 1u], mem[f + 3u]), mem[f + 5u]));
        let hi = vec2<f32>(max(max(mem[f], mem[f + 2u]), mem[f + 4u]), max(max(mem[f + 1u], mem[f + 3u]), mem[f + 5u]));
        let c = vec2<f32>(mem[p.b_off + 2u * i], mem[p.b_off + 2u * i + 1u]);
        let gap = max(lo - c, vec2<f32>(0.0)) + max(c - hi, vec2<f32>(0.0));
        d = length(gap) / p.f0;
    }
    mem[p.c_off + i * p.n + j] = d;
}

// ---- Image ----------------------------------------------------------------------------------

// Transpose: A [m][n] at a_off to C [n][m] at c_off (patch queries to the head's channel grid).
@compute @workgroup_size(64)
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= p.m || col >= p.n) { return; }
    mem[p.c_off + col * p.m + row] = mem[p.a_off + row * p.n + col];
}

// 3x3 convolution, padding 1: in [k channels][H = m][W = n] at a_off, weights [x0][k][3][3] at
// b_off, bias at bias_off, out [x0][H][W] at c_off; GELU after with the GELU flag. With SHUFFLE
// the output is pixel-shuffled: channel 4c + 2i + j of pixel (y, x) goes to channel c of pixel
// (2y + i, 2x + j) of an image [x0 / 4][2H][2W]. With INTERLEAVE it is written as interleaved
// RGB [H][W][3]. Each thread makes 8 output channels (group wg.z) of one pixel from one read
// of its 3x3 neighbourhood, with the group's weights for 32 input channels at a time in
// workgroup memory.

// The pixel of a 16x16-workgroup thread. x2, when not 0, is a list in `mem` of the tiles to
// draw (each its column and row of tiles), and the dispatch's x is the place in that list.
fn tile_pixel(gid: vec3<u32>) -> vec2<u32> {
    if (p.x2 == 0u) { return gid.xy; }
    let tile = p.x2 + gid.x / 16u * 2u;
    return vec2<u32>(u32(mem[tile]) * 16u + gid.x % 16u, u32(mem[tile + 1u]) * 16u + gid.y % 16u);
}

var<workgroup> cw: array<vec4<f32>, 576>; // [ci < 32][tap 9][2]: 8 output channels' weights

@compute @workgroup_size(16, 16)
fn conv3x3(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let co0 = wg.z * 8u;
    let cout = p.x0;
    let pixel = tile_pixel(gid);
    let x = pixel.x;
    let y = pixel.y;
    let live = x < p.n && y < p.m;
    let plane = p.m * p.n;
    // x1 > 1: a strided convolution, reading a plane x1 times the size each way.
    let step = max(p.x1, 1u);
    let im = p.m * step;
    let inn = p.n * step;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    // The nine taps' places in a plane and whether each is inside it, once per pixel: the
    // loops below are over channels, and these are the same for every channel.
    let dy0 = i32(y * step) + -1;
    let dx0 = i32(x * step) + -1;
    let in0 = dy0 >= 0 && dy0 < i32(im) && dx0 >= 0 && dx0 < i32(inn);
    let at0 = u32(clamp(dy0, 0, i32(im) - 1)) * inn + u32(clamp(dx0, 0, i32(inn) - 1));
    let dy1 = i32(y * step) + -1;
    let dx1 = i32(x * step) + 0;
    let in1 = dy1 >= 0 && dy1 < i32(im) && dx1 >= 0 && dx1 < i32(inn);
    let at1 = u32(clamp(dy1, 0, i32(im) - 1)) * inn + u32(clamp(dx1, 0, i32(inn) - 1));
    let dy2 = i32(y * step) + -1;
    let dx2 = i32(x * step) + 1;
    let in2 = dy2 >= 0 && dy2 < i32(im) && dx2 >= 0 && dx2 < i32(inn);
    let at2 = u32(clamp(dy2, 0, i32(im) - 1)) * inn + u32(clamp(dx2, 0, i32(inn) - 1));
    let dy3 = i32(y * step) + 0;
    let dx3 = i32(x * step) + -1;
    let in3 = dy3 >= 0 && dy3 < i32(im) && dx3 >= 0 && dx3 < i32(inn);
    let at3 = u32(clamp(dy3, 0, i32(im) - 1)) * inn + u32(clamp(dx3, 0, i32(inn) - 1));
    let dy4 = i32(y * step) + 0;
    let dx4 = i32(x * step) + 0;
    let in4 = dy4 >= 0 && dy4 < i32(im) && dx4 >= 0 && dx4 < i32(inn);
    let at4 = u32(clamp(dy4, 0, i32(im) - 1)) * inn + u32(clamp(dx4, 0, i32(inn) - 1));
    let dy5 = i32(y * step) + 0;
    let dx5 = i32(x * step) + 1;
    let in5 = dy5 >= 0 && dy5 < i32(im) && dx5 >= 0 && dx5 < i32(inn);
    let at5 = u32(clamp(dy5, 0, i32(im) - 1)) * inn + u32(clamp(dx5, 0, i32(inn) - 1));
    let dy6 = i32(y * step) + 1;
    let dx6 = i32(x * step) + -1;
    let in6 = dy6 >= 0 && dy6 < i32(im) && dx6 >= 0 && dx6 < i32(inn);
    let at6 = u32(clamp(dy6, 0, i32(im) - 1)) * inn + u32(clamp(dx6, 0, i32(inn) - 1));
    let dy7 = i32(y * step) + 1;
    let dx7 = i32(x * step) + 0;
    let in7 = dy7 >= 0 && dy7 < i32(im) && dx7 >= 0 && dx7 < i32(inn);
    let at7 = u32(clamp(dy7, 0, i32(im) - 1)) * inn + u32(clamp(dx7, 0, i32(inn) - 1));
    let dy8 = i32(y * step) + 1;
    let dx8 = i32(x * step) + 1;
    let in8 = dy8 >= 0 && dy8 < i32(im) && dx8 >= 0 && dx8 < i32(inn);
    let at8 = u32(clamp(dy8, 0, i32(im) - 1)) * inn + u32(clamp(dx8, 0, i32(inn) - 1));
    for (var c0 = 0u; c0 < p.k; c0 += 32u) {
        let kc = min(32u, p.k - c0);
        workgroupBarrier();
        // This chunk's weights: w[co0 + o][c0 + ci][tap] -> cw[(ci * 9 + tap) * 2 + o / 4][o % 4].
        for (var e = li; e < kc * 9u * 8u; e += 256u) {
            let o = e % 8u;
            let ct = e / 8u; // ci * 9 + tap
            let co = min(co0 + o, cout - 1u);
            let v = select(0.0, mem[p.b_off + co * p.k * 9u + c0 * 9u + ct], co0 + o < cout);
            cw[ct * 2u + o / 4u][o % 4u] = v;
        }
        workgroupBarrier();
        if (live) {
            for (var ci = 0u; ci < kc; ci++) {
                let src = p.a_off + (c0 + ci) * im * inn;
                let w = ci * 18u;
                let v0 = select(0.0, mem[src + at0], in0);
                let v1 = select(0.0, mem[src + at1], in1);
                let v2 = select(0.0, mem[src + at2], in2);
                let v3 = select(0.0, mem[src + at3], in3);
                let v4 = select(0.0, mem[src + at4], in4);
                let v5 = select(0.0, mem[src + at5], in5);
                let v6 = select(0.0, mem[src + at6], in6);
                let v7 = select(0.0, mem[src + at7], in7);
                let v8 = select(0.0, mem[src + at8], in8);
                acc0 += v0 * cw[w + 0u]; acc1 += v0 * cw[w + 1u];
                acc0 += v1 * cw[w + 2u]; acc1 += v1 * cw[w + 3u];
                acc0 += v2 * cw[w + 4u]; acc1 += v2 * cw[w + 5u];
                acc0 += v3 * cw[w + 6u]; acc1 += v3 * cw[w + 7u];
                acc0 += v4 * cw[w + 8u]; acc1 += v4 * cw[w + 9u];
                acc0 += v5 * cw[w + 10u]; acc1 += v5 * cw[w + 11u];
                acc0 += v6 * cw[w + 12u]; acc1 += v6 * cw[w + 13u];
                acc0 += v7 * cw[w + 14u]; acc1 += v7 * cw[w + 15u];
                acc0 += v8 * cw[w + 16u]; acc1 += v8 * cw[w + 17u];
            }
        }
    }
    if (!live) { return; }
    let outs = array<vec4<f32>, 2>(acc0, acc1);
    for (var o = 0u; o < 8u; o++) {
        let co = co0 + o;
        if (co >= cout) { break; }
        var s = outs[o / 4u][o % 4u] + mem[p.bias_off + co];
        if ((p.flags & GELU) != 0u) { s = gelu(s); }
        if ((p.flags & INTERLEAVE) != 0u) {
            mem[p.c_off + (y * p.n + x) * 3u + co] = s;
        } else if ((p.flags & SHUFFLE) != 0u) {
            let c = co / 4u;
            let yy = 2u * y + (co % 4u) / 2u;
            let xx = 2u * x + co % 2u;
            mem[p.c_off + (c * 2u * p.m + yy) * 2u * p.n + xx] = s;
        } else if ((p.flags & ACCUMULATE) != 0u) {
            mem[p.c_off + co * plane + y * p.n + x] += s;
        } else {
            mem[p.c_off + co * plane + y * p.n + x] = s;
        }
    }
}

// C += A elementwise over m floats (A at a_off, C at c_off).
@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 64u + gid.x;
    if (i >= p.m) { return; }
    mem[p.c_off + i] += mem[p.a_off + i];
}

// c = mix(c, a, f0) over m floats: the glitch panel's echoes (last frame's latents kept).
@compute @workgroup_size(64)
fn lerp_into(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 64u + gid.x;
    if (i >= p.m) { return; }
    mem[p.c_off + i] = mix(mem[p.c_off + i], mem[p.a_off + i], p.f0);
}

// Doubles planes [z][m / 2][n / 2] at a_off to [z][m][n] at c_off, bilinearly, as
// F.interpolate(scale_factor=2, mode="bilinear", align_corners=False); z by dispatch depth.
@compute @workgroup_size(16, 16)
fn resize2(@builtin(global_invocation_id) gid: vec3<u32>) {
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

// ---- The per-pixel stage ----------------------------------------------------------------------
// train/pixels.py `one_view` and model.py `Deferred` up to its MLP: for each pixel, which piece
// it shows (gbuffer.py `rasterise`: the nearest by 1/z, of those as near the later), what the
// G-buffer says of it, and the latent textures sampled there, written as planes [66][m][n] at
// c_off: the head's last 16 channels (planes at bias_off), the 22 plain channels, then the
// level (8, times covered), flesh (8, times on flesh), eye (8) and iris (4, both times on an
// eye) latents. Pieces (gbuffer.rs PIECE floats) at a_off, rows (ROW) at b_off, tiles (start,
// count) at x0, their lists at x1, k organs (19 floats each) at x2, the view (views.npy row)
// at a_bs, the latents at a_rs (level), a_cs (flesh), b_bs (eye) and b_rs (iris); the dial's
// amount in f0. A workgroup per 16x16 tile.

const PIECE: u32 = 19u;
const ROW: u32 = 31u;
const LEVEL_SIZE: u32 = 128u;
const FLESH_SIZE: u32 = 256u;
const EYE_SIZE: u32 = 64u;
const BLINKS: u32 = 8u;

fn piece_weights(q: u32, cx: f32, cy: f32) -> vec3<f32> {
    let a = vec2<f32>(mem[q], mem[q + 1u]);
    let b = vec2<f32>(mem[q + 2u], mem[q + 3u]);
    let c = vec2<f32>(mem[q + 4u], mem[q + 5u]);
    let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    let l0 = ((b.x - cx) * (c.y - cy) - (b.y - cy) * (c.x - cx)) / area;
    let l1 = ((c.x - cx) * (a.y - cy) - (c.y - cy) * (a.x - cx)) / area;
    return vec3<f32>(l0, l1, 1.0 - l0 - l1);
}

// gbuffer.py's far(): a ray that never meets a plane is far off it, by the sign of the infinity.
fn clamp_far(x: f32) -> f32 {
    return clamp(x, -1e3, 1e3);
}

// Bilinear from plane `c` [h][w] at `base`, at normalised (x, y), zeros outside: grid_sample
// with align_corners=False.
fn bilinear(base: u32, h: u32, w: u32, x: f32, y: f32) -> f32 {
    let ix = ((x + 1.0) * f32(w) - 1.0) * 0.5;
    let iy = ((y + 1.0) * f32(h) - 1.0) * 0.5;
    let x0 = floor(ix);
    let y0 = floor(iy);
    let fx = ix - x0;
    let fy = iy - y0;
    var s = 0.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let xi = i32(x0) + i;
            let yj = i32(y0) + j;
            if (xi >= 0 && xi < i32(w) && yj >= 0 && yj < i32(h)) {
                let k = select(1.0 - fx, fx, i == 1) * select(1.0 - fy, fy, j == 1);
                s += k * mem[base + u32(yj) * w + u32(xi)];
            }
        }
    }
    return s;
}

@compute @workgroup_size(16, 16)
fn stage_inputs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let width = p.n;
    if (x >= width || y >= p.m) { return; }
    let plane = p.m * width;
    let px = y * width + x;
    // The glitch panel's lies (GBuffer::floats at c_bs): warp, twist, depth, scramble,
    // cells, flesh, ghost, time. All zero, the pixel is told the truth.
    let gl = p.c_bs;
    let g_warp = mem[gl];
    let g_twist = mem[gl + 1u];
    let g_depth = mem[gl + 2u];
    let g_scramble = mem[gl + 3u];
    let g_cells = mem[gl + 4u];
    let g_flesh = mem[gl + 5u];
    let g_ghost = mem[gl + 6u];
    let g_t = mem[gl + 7u];
    let ox = f32(x) + 0.5;
    let oy = f32(y) + 0.5;
    // The G-buffer warp: what this pixel shows is looked up somewhere near it.
    var cx = ox;
    var cy = oy;
    if (g_warp > 0.0) {
        let wv = glitch_wobble(vec2<f32>(ox / 97.0, oy / 83.0), g_t);
        cx = clamp(ox + g_warp * wv.x, 0.5, f32(width) - 0.5);
        cy = clamp(oy + g_warp * wv.y, 0.5, f32(p.m) - 0.5);
    }
    // Which piece: the nearest of those whose triangle holds the pixel centre.
    let tile = p.x0 + 2u * ((u32(cy) / 16u) * (width / 16u) + u32(cx) / 16u);
    let start = u32(mem[tile]);
    let count = u32(mem[tile + 1u]);
    var best = -1.0;
    var who = -1;
    for (var e = 0u; e < count; e++) {
        let k = u32(mem[p.x1 + start + e]);
        let q = p.a_off + k * PIECE;
        let lam = piece_weights(q, cx, cy);
        if (lam.x >= 0.0 && lam.y >= 0.0 && lam.z >= 0.0) {
            let iz = lam.x * mem[q + 6u] + lam.y * mem[q + 7u] + lam.z * mem[q + 8u];
            if (iz > best || (iz == best && i32(k) > who)) {
                best = iz;
                who = i32(k);
            }
        }
    }
    let covered = who >= 0;
    let cov = select(0.0, 1.0, covered);
    var w = vec3<f32>(0.0);
    var row = p.b_off;
    if (covered) {
        let q = p.a_off + u32(who) * PIECE;
        let lam = piece_weights(q, cx, cy);
        for (var j = 0u; j < 3u; j++) {
            w[j] = lam.x * mem[q + 10u + j] + lam.y * mem[q + 13u + j] + lam.z * mem[q + 16u + j];
        }
        row = p.b_off + u32(mem[q + 9u]) * ROW;
    }
    let iz = max(best, 0.0);
    let face = f32((row - p.b_off) / ROW);
    let v = p.a_bs;
    let h = mem[v + 13u];
    let R = mat3x3<f32>(mem[v + 4u], mem[v + 7u], mem[v + 10u], mem[v + 5u], mem[v + 8u], mem[v + 11u], mem[v + 6u], mem[v + 9u], mem[v + 12u]);
    // (R is the view's rows as columns of the transpose: R * d turns a world direction d.)

    // Texture, and where on it.
    var texid = 0u;
    if (covered) { texid = u32(clamp(mem[row + 15u], 0.0, 11.0)); }
    var uv = clamp(vec2<f32>(w.x * mem[row + 9u] + w.y * mem[row + 11u] + w.z * mem[row + 13u],
                             w.x * mem[row + 10u] + w.y * mem[row + 12u] + w.z * mem[row + 14u]) * (64.0 / 63.0), vec2<f32>(0.0), vec2<f32>(1.0));
    // Facing, in view space.
    var n = vec3<f32>(0.0);
    if (covered) {
        let p0 = vec3<f32>(mem[row], mem[row + 1u], mem[row + 2u]);
        let p1 = vec3<f32>(mem[row + 3u], mem[row + 4u], mem[row + 5u]);
        let p2 = vec3<f32>(mem[row + 6u], mem[row + 7u], mem[row + 8u]);
        let c = R * cross(p1 - p0, p2 - p0);
        n = c / max(length(c), 1e-9);
        if (g_twist > 0.0) {
            let tw = vec3<f32>(sin(face * 12.9898 + g_t * 0.5), sin(face * 78.233 + g_t * 0.37), sin(face * 37.719 + g_t * 0.29));
            n = normalize(n + g_twist * tw);
        }
    }
    let near = h / 2.0 / 256.0;
    var inv = clamp(iz * near, 0.0, 2.0);
    if (g_depth > 0.0) {
        inv = clamp(inv * exp(g_depth * sin(cx * 0.021 + g_t * 0.6 + 2.0 * sin(cy * 0.033 - g_t * 0.4))), 0.0, 2.0);
    }
    // Flesh.
    let t = row + 16u;
    var lay = vec2<f32>(w.x * mem[t + 3u] + w.y * mem[t + 5u] + w.z * mem[t + 7u], w.x * mem[t + 4u] + w.y * mem[t + 6u] + w.z * mem[t + 8u]);
    var slot = i32(texid);
    if (texid == 3u) { slot = 0; } else if (texid == 4u || texid == 5u) { slot = 1; } else if (texid == 7u) { slot = 6; } else if (texid >= 8u) { slot = -1; }
    var lays = 0.0;
    for (var j = 3u; j < 9u; j++) { lays += abs(mem[t + j]); }
    let truly = covered && lays > 0.0 && slot >= 0;
    var reach = p.f0 * (w.x * mem[t] + w.y * mem[t + 1u] + w.z * mem[t + 2u]);
    // Flesh told where there is none: its latent faded in by g_flesh, laid by the texture.
    let flesh = truly || (covered && g_flesh > 0.0);
    let fleshness = select(g_flesh, 1.0, truly);
    if (flesh && !truly) {
        slot = max(slot, 0);
        if (lays == 0.0) { lay = uv * 2.0; }
        reach = max(reach, g_flesh);
    }
    // Each face's texture turned and scaled its own way, and the coordinates snapped to cells.
    if (g_scramble > 0.0 && covered) {
        let a = g_scramble * 6.2832 * fract(sin(face * 91.17) * 43758.5);
        let sc = 1.0 + g_scramble * 2.0 * fract(sin(face * 27.3) * 1373.1);
        let turn = mat2x2<f32>(cos(a), sin(a), -sin(a), cos(a));
        uv = fract(turn * (uv - 0.5) * sc + 0.5);
        lay = turn * lay * sc;
    }
    if (g_cells > 0.0) {
        uv = clamp((floor(uv * g_cells) + 0.5) / g_cells, vec2<f32>(0.0), vec2<f32>(1.0));
        lay = (floor(lay * g_cells) + 0.5) / g_cells;
    }
    let fh = t + 9u;
    let fold = mem[fh] * w.x * (2.0 * w.x - 1.0) + mem[fh + 1u] * w.y * (2.0 * w.y - 1.0) + mem[fh + 2u] * w.z * (2.0 * w.z - 1.0)
        + 4.0 * (mem[fh + 3u] * w.x * w.y + mem[fh + 4u] * w.x * w.z + mem[fh + 5u] * w.y * w.z);

    // Organs, where the pixel's ray meets their planes.
    var feat = array<f32, 13>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (p.k > 0u) {
        let ray = vec3<f32>((cx - f32(width) / 2.0) / h, (cy - 120.0) / h, 1.0);
        let depth = select(1e6, 1.0 / max(iz, 1e-6), covered);
        let pos = vec3<f32>(mem[v + 1u], mem[v + 2u], mem[v + 3u]);
        var spot_best = 3.0e38;
        var spot_i = 0u;
        var spot_on = false;
        var spot_u = 0.0;
        var spot_v = 0.0;
        var spot_r2 = 0.0;
        var vein_key = 1e30;
        var vein_i = 0u;
        var vein_u = 0.0;
        var vein_v = 0.0;
        for (var i = 0u; i < p.k; i++) {
            let o = p.x2 + i * 19u;
            let kind = u32(mem[o]);
            let at = R * (vec3<f32>(mem[o + 1u], mem[o + 2u], mem[o + 3u]) - pos);
            let faces = R * vec3<f32>(mem[o + 4u], mem[o + 5u], mem[o + 6u]);
            let across = R * vec3<f32>(mem[o + 7u], mem[o + 8u], mem[o + 9u]);
            let up = R * vec3<f32>(mem[o + 10u], mem[o + 11u], mem[o + 12u]);
            let num = dot(at, faces);
            var den = dot(ray, faces);
            if (!(abs(den) > 1e-6)) { den = 1e-6; }
            let tt = num / den;
            let meets = tt > 0.0;
            var u = 0.0;
            var vv = 0.0;
            if (meets) {
                let d = ray * tt - at;
                u = clamp_far(dot(d, across) / max(mem[o + 13u], 1e-4));
                vv = clamp_far(dot(d, up) / max(mem[o + 14u], 1e-4));
            } else {
                // ray * inf - at: each part infinite with its ray's sign; summed along a way, a
                // mix of signs (or a zero times infinity) is NaN, which far() makes 1e3.
                u = far_sign(ray, across);
                vv = far_sign(ray, up);
            }
            if (g_ghost > 0.0) {
                u = abs(u) - 1.2 * g_ghost;
                vv = vv * (1.0 + g_ghost);
            }
            let seen = meets && tt < depth + 0.1;
            let r2 = u * u + vv * vv;
            if (seen && kind <= 1u && r2 < 1.44 && mem[o + 15u] > 0.0 && r2 < spot_best) {
                spot_best = r2;
                spot_i = i;
                spot_on = true;
                spot_u = u;
                spot_v = vv;
                spot_r2 = r2;
            }
            // The vein nearest across, by exp(-v^2) as pixels.py has it. That is above zero in
            // PyTorch down to the smallest denormal, v^2 about 104, which the GPU flushes to
            // zero, so compare v^2 instead.
            let v2 = vv * vv;
            let near = seen && kind == 2u && abs(u) <= 1.0 && v2 < 103.9;
            if (i == 0u || (near && v2 < vein_key)) {
                vein_key = select(1e30, v2, near);
                vein_i = i;
                vein_u = u;
                vein_v = vv;
            }
        }
        if (spot_on) {
            let o = p.x2 + spot_i * 19u;
            let kind = u32(mem[o]);
            feat[0] = select(0.0, 1.0, kind == 0u);
            feat[1] = select(0.0, 1.0, kind == 1u);
            feat[2] = spot_u;
            feat[3] = spot_v;
            feat[4] = sqrt(spot_r2);
            feat[5] = mem[o + 15u];
            feat[6] = mem[o + 16u];
            feat[7] = mem[o + 17u];
            feat[8] = mem[o + 18u];
        }
        let vein_best = select(0.0, exp(-vein_key), vein_key < 103.9);
        feat[9] = vein_best;
        feat[10] = clamp(vein_v, -3.0, 3.0) / 3.0;
        feat[11] = clamp(vein_u, -1.0, 1.0);
        feat[12] = select(0.0, mem[p.x2 + vein_i * 19u + 15u], vein_best > 0.01);
    }

    let out = p.c_off;
    // The glitch panel's smear (f2 pixels, at f3 seconds): the head's channels read from
    // beside the pixel, so what the network meant slides off the walls.
    var hp = px;
    if (p.f2 > 0.0) {
        let dx = p.f2 * sin(oy * 0.11 + p.f3 * 1.3 + 2.0 * sin(ox * 0.031 - p.f3 * 0.7));
        let dy = 0.5 * p.f2 * sin(ox * 0.083 - p.f3 * 0.9 + 2.0 * sin(oy * 0.047 + p.f3 * 0.5));
        let sx = clamp(i32(round(ox - 0.5 + dx)), 0, i32(width) - 1);
        let sy = clamp(i32(round(oy - 0.5 + dy)), 0, i32(p.m) - 1);
        hp = u32(sy) * width + u32(sx);
    }
    for (var c = 0u; c < 16u; c++) {
        mem[out + c * plane + px] = mem[p.bias_off + c * plane + hp];
    }
    var plain = array<f32, 22>(cov, inv, n.x * cov, n.y * cov, n.z * cov, select(0.0, fleshness, flesh), reach, p.f0, fold * 4.0,
        feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7], feat[8], feat[9], feat[10], feat[11], feat[12]);
    for (var c = 0u; c < 22u; c++) {
        mem[out + (16u + c) * plane + px] = clamp(plain[c], -1e3, 1e3);
    }

    // The level latent: tile `texid` of the 4 x 3 atlas, half a texel inside it.
    let e = 0.5 / f32(LEVEL_SIZE);
    // The glitch panel's warp (f1 of a texture, at f3 seconds): the latents read off true.
    var wuv = uv;
    if (p.f1 > 0.0) { wuv = fract(uv + p.f1 * glitch_wobble(uv, p.f3)); }
    let luv = wuv * (1.0 - 2.0 * e) + e;
    let ax = (f32(texid % 4u) + luv.x) / 4.0 * 2.0 - 1.0;
    let ay = (f32(texid / 4u) + luv.y) / 3.0 * 2.0 - 1.0;
    let ah = 3u * LEVEL_SIZE;
    let aw = 4u * LEVEL_SIZE;
    for (var c = 0u; c < 8u; c++) {
        mem[out + (38u + c) * plane + px] = bilinear(p.a_rs + c * ah * aw, ah, aw, ax, ay) * cov;
    }
    // The flesh latent of the painted slot, wrapping.
    let fs = i32(FLESH_SIZE);
    let wl = lay + p.f1 * glitch_wobble(lay, p.f3);
    let lx = (wl.x - floor(wl.x)) * f32(FLESH_SIZE) - 0.5;
    let ly = (wl.y - floor(wl.y)) * f32(FLESH_SIZE) - 0.5;
    let fx0 = floor(lx);
    let fy0 = floor(ly);
    let ffx = lx - fx0;
    let ffy = ly - fy0;
    let xa = u32(((i32(fx0) % fs) + fs) % fs);
    let xb = u32(((i32(fx0) + 1) % fs + fs) % fs);
    let ya = u32(((i32(fy0) % fs) + fs) % fs) * FLESH_SIZE;
    let yb = u32(((i32(fy0) + 1) % fs + fs) % fs) * FLESH_SIZE;
    for (var c = 0u; c < 8u; c++) {
        var s = 0.0;
        if (flesh) {
            let base = p.a_cs + (u32(max(slot, 0)) * 8u + c) * FLESH_SIZE * FLESH_SIZE;
            let top = mem[base + ya + xa] * (1.0 - ffx) + mem[base + ya + xb] * ffx;
            let bottom = mem[base + yb + xa] * (1.0 - ffx) + mem[base + yb + xb] * ffx;
            s = (top * (1.0 - ffy) + bottom * ffy) * fleshness;
        }
        mem[out + (46u + c) * plane + px] = s;
    }
    // The eye by blink, trilinearly, and the iris shifted by the gaze; only on an eye.
    let on = feat[0] > 0.0;
    let eu = clamp(feat[2], -1.0, 1.0);
    let ev = clamp(feat[3], -1.0, 1.0);
    let ez = clamp(feat[6], 0.0, 1.0) * 2.0 - 1.0;
    let iz3 = ((ez + 1.0) * f32(BLINKS) - 1.0) * 0.5;
    let z0 = floor(iz3);
    let fz = iz3 - z0;
    let es = EYE_SIZE * EYE_SIZE;
    for (var c = 0u; c < 8u; c++) {
        var s = 0.0;
        if (on) {
            for (var k = 0; k < 2; k++) {
                let zi = i32(z0) + k;
                if (zi >= 0 && zi < i32(BLINKS)) {
                    s += select(1.0 - fz, fz, k == 1) * bilinear(p.b_bs + (c * BLINKS + u32(zi)) * es, EYE_SIZE, EYE_SIZE, eu, ev);
                }
            }
        }
        mem[out + (54u + c) * plane + px] = s;
    }
    let gx = feat[7] - 0.5;
    let gy = feat[8] - 0.5;
    for (var c = 0u; c < 4u; c++) {
        var s = 0.0;
        if (on) { s = bilinear(p.b_rs + c * es, EYE_SIZE, EYE_SIZE, eu - 0.8 * gx, ev - 0.8 * gy); }
        mem[out + (62u + c) * plane + px] = s;
    }
}

// The glitch panel's warp: slow waves, about -1..1 on each axis, at `q` and `t` seconds.
fn glitch_wobble(q: vec2<f32>, t: f32) -> vec2<f32> {
    let a = 6.2832 * (1.7 * q.y + 0.13 * t) + 2.0 * sin(6.2832 * (0.9 * q.x - 0.1 * t));
    let b = 6.2832 * (1.3 * q.x - 0.11 * t) + 2.0 * sin(6.2832 * (1.1 * q.y + 0.07 * t));
    return vec2<f32>(sin(a), sin(b));
}

// What dot(ray * inf - at, way) is after far(): +-1e3 by the sign of the infinite sum, 1e3 if
// it is NaN (infinities of both signs, or an infinity times zero).
fn far_sign(ray: vec3<f32>, way: vec3<f32>) -> f32 {
    var pos = false;
    var neg = false;
    var nan = false;
    for (var i = 0; i < 3; i++) {
        let r = ray[i];
        let a = way[i];
        if (r == 0.0) {
            // 0 * inf is NaN: the point itself is NaN along this axis.
            nan = true;
        } else if (a == 0.0) {
            nan = true;
        } else if ((r > 0.0) == (a > 0.0)) {
            pos = true;
        } else {
            neg = true;
        }
    }
    if (nan || (pos && neg)) { return 1e3; }
    return select(-1e3, 1e3, pos);
}
