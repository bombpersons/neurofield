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
// b_off + head, dist at x1 with row stride n: the coverage bias), and columns from x2 on are
// masked out (padding). One workgroup per row.
@compute @workgroup_size(64)
fn softmax(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let head = wg.y;
    let row = wg.x;
    let lid = lid3.x;
    let x = p.a_off + (head * p.m + row) * p.n;
    let cover = p.b_off != NONE;
    var strength = 0.0;
    if (cover) { strength = mem[p.b_off + head]; }
    var mx = -3.0e38;
    for (var j = lid; j < p.x2; j += 64u) {
        var s = mem[x + j] * p.f0;
        if (cover) { s -= strength * mem[p.x1 + row * p.n + j]; }
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

// The triangle MLP's input: m rows of [features (k = 46), Fourier features of the 6 NDC
// coordinates (16 each: sin at 8 octaves from pi / 4, then cos)], from features at a_off to
// c_off (row stride n = k + 96). Angles are reduced in turns first, exactly, since GPU sin and
// cos lose accuracy far from zero.
@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= p.m || col >= p.n) { return; }
    var v: f32;
    if (col < p.k) {
        v = mem[p.a_off + row * p.k + col];
    } else {
        let c = col - p.k;
        let x = mem[p.a_off + row * p.k + 9u + c / 16u];
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

// Coverage distances, as model.py cover_distance: for patch i < m and token j < n, the
// distance in patch widths (f0) from the patch centre (x, y at b_off + 2i) to the screen
// bounding box of triangle j - 1 (NDC at features a_off + (j - 1) * k + 9), 0 for the
// camera token j = 0; written to c_off + i * n + j.
@compute @workgroup_size(64)
fn cover_distance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let i = gid.y;
    if (i >= p.m || j >= p.n) { return; }
    var d = 0.0;
    if (j > 0u) {
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

var<workgroup> cw: array<vec4<f32>, 576>; // [ci < 32][tap 9][2]: 8 output channels' weights

@compute @workgroup_size(16, 16)
fn conv3x3(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let co0 = wg.z * 8u;
    let cout = p.x0;
    let x = gid.x;
    let y = gid.y;
    let live = x < p.n && y < p.m;
    let plane = p.m * p.n;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
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
                let src = p.a_off + (c0 + ci) * plane;
                for (var tap = 0u; tap < 9u; tap++) {
                    let yy = i32(y) + i32(tap / 3u) - 1;
                    let xx = i32(x) + i32(tap % 3u) - 1;
                    let inside = yy >= 0 && yy < i32(p.m) && xx >= 0 && xx < i32(p.n);
                    let at = u32(clamp(yy, 0, i32(p.m) - 1)) * p.n + u32(clamp(xx, 0, i32(p.n) - 1));
                    let v = select(0.0, mem[src + at], inside);
                    let w = (ci * 9u + tap) * 2u;
                    acc0 += v * cw[w];
                    acc1 += v * cw[w + 1u];
                }
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
