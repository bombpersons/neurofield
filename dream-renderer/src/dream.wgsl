// Kernels for the dream renderer (lib.rs): GoogLeNet or ResNet forward and backward to its input, and the
// dream layer kept on surface pages (the atlas).
//
// `mem` holds the weights and every float the network works with; activations are HWC (a
// pixel's channels together). Each dispatch reads its sizes and offsets (in floats, or u32s for
// the other buffers) from its own slot of the uniform buffer; each kernel says how it uses them.

struct Params {
    m: u32, k: u32, n: u32, flags: u32,
    a_off: u32, a_rs: u32, a_cs: u32, b_off: u32,
    b_rs: u32, b_cs: u32, c_off: u32, c_rs: u32,
    bias_off: u32, x0: u32, x1: u32, x2: u32,
    x3: u32, x4: u32, x5: u32, x6: u32,
    f0: f32, f1: f32, f2: f32, f3: f32,
    // g0: an octave's pixels to a pixel of the frame given (prepare, net_height, screen_step).
    g0: f32, g1: f32, g2: f32, g3: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> mem: array<f32>;
// The dream layer: page-major, RES x RES texels a page, each two u32s of packed f16: the colour
// it adds (r, g), (b, and how much of the wall it owns, 0..1).
@group(0) @binding(2) var<storage, read_write> atlas: array<u32>;
// Each touched page's sums of the step (r, g, b) and how much each texel was seen, fixed point.
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<i32>>;
// RGBA8 pictures: the frame given (rows b_cs apart), then the frame shown (rows b_rs apart);
// then the triangle each pixel shows, from the level drawn (level.rs Raster).
@group(0) @binding(4) var<storage, read_write> img: array<u32>;
// The level (level.rs Level): each triangle's normal and plane, the face table, each face's page.
@group(0) @binding(5) var<storage, read_write> level: array<u32>;
// The faces each band of rows shows, a bit a face.
@group(0) @binding(6) var<storage, read_write> faces_seen: array<atomic<u32>>;

const NONE: u32 = 0xffffffffu;
const RELU: u32 = 1u;
const ACCUMULATE: u32 = 2u;
// prepare and show: the screen layer (below) is laid over the dream.
const SCREEN: u32 = 4u;
// prepare and show: the screen layer is the whole dream (while it's carried along), the pages
// not shown.
const SCREEN_ONLY: u32 = 8u;
// warp: what wasn't seen last frame starts from the pages' dream, not from nothing.
const FILL: u32 = 16u;
// show: the dreamed height alone, shaded as the eye lights it, in grey.
const SHOW_HEIGHT: u32 = 32u;
// prepare: the network's input shaded by the dream lamp from the dreamed height (below).
const LAMP: u32 = 64u;
// Texels of a page a side to each of its height's.
const HEIGHT_CELL: u32 = 2u;
// The dream lamp's slots (one an octave: its direction along the page and off it, the ambient
// light, how much it shades, whether the network sees the shading alone, the height's step,
// its most, and how far apart the taps its step is spread over are), then its work.
const LAMP_SLOT: u32 = 12u;
const LAMP_AUX: u32 = 24u;
// Fixed point of `acc`.
const FIXED: f32 = 4096.0;
// Floats a pixel of the surfaces given (lib.rs Surface) and of a net pixel's lookup.
const SURFACE: u32 = 13u;
const LOOKUP: u32 = 6u;

// The flat index of a 2D dispatch of rows (y, spilling into z).
fn row_of(gid: vec3<u32>) -> u32 {
    return gid.y + gid.z * 65535u;
}

// ---- Matrix multiply ----------------------------------------------------------------------
// C[i][j] (=, or += with ACCUMULATE) act(sum_t A[i][t] * B[t][j] + bias[j]) for i < m, j < n,
// t < k. A[i][t] = mem[a_off + i*a_rs + t*a_cs], B likewise, C[i][j] = mem[c_off + i*c_rs + j];
// bias_off = NONE for none. A 32x32 tile of C per workgroup of 8x8 threads, 4x4 a thread, k in
// steps of 16 with the next prefetched (the engine's gpu_net.wgsl `matmul`, with ReLU). With x6
// > 1, k is split into x6 parts (a part a workgroup, the parts along x after the column tiles)
// for products with too few tiles to keep the GPU busy: each part's sums go to x5 (part-major,
// m x n each) and `matmul_sum` adds them up and finishes them.

var<workgroup> ta: array<vec4<f32>, 128>;
var<workgroup> tb: array<vec4<f32>, 128>;

fn load_a(kl: u32, row: u32, t: u32) -> vec4<f32> {
    let base = p.a_off + min(t, kl - 1u) * p.a_cs;
    let last = p.m - 1u;
    let v = vec4<f32>(mem[base + min(row, last) * p.a_rs], mem[base + min(row + 1u, last) * p.a_rs],
                      mem[base + min(row + 2u, last) * p.a_rs], mem[base + min(row + 3u, last) * p.a_rs]);
    let inside = vec4<bool>(row < p.m, row + 1u < p.m, row + 2u < p.m, row + 3u < p.m) & vec4<bool>(t < kl);
    return select(vec4<f32>(0.0), v, inside);
}

fn load_b(kl: u32, t: u32, col: u32) -> vec4<f32> {
    let base = p.b_off + min(t, kl - 1u) * p.b_rs;
    let last = p.n - 1u;
    let v = vec4<f32>(mem[base + min(col, last) * p.b_cs], mem[base + min(col + 1u, last) * p.b_cs],
                      mem[base + min(col + 2u, last) * p.b_cs], mem[base + min(col + 3u, last) * p.b_cs]);
    let inside = vec4<bool>(col < p.n, col + 1u < p.n, col + 2u < p.n, col + 3u < p.n) & vec4<bool>(t < kl);
    return select(vec4<f32>(0.0), v, inside);
}

fn finish(v: f32, o: u32, col: u32) -> f32 {
    var x = v;
    if (p.bias_off != NONE) { x += mem[p.bias_off + col]; }
    if ((p.flags & RELU) != 0u) { x = max(x, 0.0); }
    if ((p.flags & ACCUMULATE) != 0u) { x += mem[o]; }
    return x;
}

@compute @workgroup_size(8, 8)
fn matmul(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let splits = max(p.x6, 1u);
    let tiles = (p.n + 31u) / 32u;
    let part = wg.x / tiles;
    let chunk = (p.k + splits * 16u - 1u) / (splits * 16u) * 16u;
    let kb = part * chunk;
    let kl = min(kb + chunk, p.k);
    let row0 = (wg.y + wg.z * 65535u) * 32u;
    let col0 = (wg.x % tiles) * 32u;
    let tid = lid.y * 8u + lid.x;
    let q = tid % 8u;
    let tk = tid / 8u;
    var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
    var na0 = load_a(kl, row0 + q * 4u, kb + tk); var na1 = load_a(kl, row0 + q * 4u, kb + tk + 8u);
    var nb0 = load_b(kl, kb + tk, col0 + q * 4u); var nb1 = load_b(kl, kb + tk + 8u, col0 + q * 4u);
    for (var t0 = kb; t0 < kl; t0 += 16u) {
        ta[tk * 8u + q] = na0; ta[(tk + 8u) * 8u + q] = na1;
        tb[tk * 8u + q] = nb0; tb[(tk + 8u) * 8u + q] = nb1;
        workgroupBarrier();
        na0 = load_a(kl, row0 + q * 4u, t0 + 16u + tk); na1 = load_a(kl, row0 + q * 4u, t0 + 24u + tk);
        nb0 = load_b(kl, t0 + 16u + tk, col0 + q * 4u); nb1 = load_b(kl, t0 + 24u + tk, col0 + q * 4u);
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
    if (splits > 1u) {
        for (var i = 0u; i < 4u; i++) {
            let row = row0 + lid.y * 4u + i;
            if (row < p.m) {
                let r = rows[i];
                let o = p.x5 + (part * p.m + row) * p.n + col;
                if (col < p.n) { mem[o] = r.x; }
                if (col + 1u < p.n) { mem[o + 1u] = r.y; }
                if (col + 2u < p.n) { mem[o + 2u] = r.z; }
                if (col + 3u < p.n) { mem[o + 3u] = r.w; }
            }
        }
        return;
    }
    for (var i = 0u; i < 4u; i++) {
        let row = row0 + lid.y * 4u + i;
        if (row < p.m) {
            let r = rows[i];
            let o = p.c_off + row * p.c_rs + col;
            if (col < p.n) { mem[o] = finish(r.x, o, col); }
            if (col + 1u < p.n) { mem[o + 1u] = finish(r.y, o + 1u, col + 1u); }
            if (col + 2u < p.n) { mem[o + 2u] = finish(r.z, o + 2u, col + 2u); }
            if (col + 3u < p.n) { mem[o + 3u] = finish(r.w, o + 3u, col + 3u); }
        }
    }
}

// The parts of a split product summed in order and finished, a thread an element of C (2D as
// `clear`).
@compute @workgroup_size(64)
fn matmul_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = row_of(gid) * 4096u + gid.x;
    if (i >= p.m * p.n) { return; }
    var sum = 0.0;
    for (var part = 0u; part < p.x6; part++) {
        sum += mem[p.x5 + part * p.m * p.n + i];
    }
    let row = i / p.n;
    let col = i % p.n;
    let o = p.c_off + row * p.c_rs + col;
    mem[o] = finish(sum, o, col);
}

// The same for big products: a 64x64 tile of C per workgroup of 8x8 threads, 8x8 a thread (rows
// ty*4.. and 32 + ty*4.., columns likewise), k in steps of 8. A's rows are contiguous in k
// (a_cs 1, as every caller's are) and are read along k by 8 neighbouring threads; B likewise
// if b_rs is 1, else along n by 64. Each thread's sums run over k in order, as `matmul`'s.
// (Written out in full: arrays in a thread's own memory are slow on some GPUs.)

// [k][64] a vec4 four rows (columns), each k's 17 apart so writing along k spreads over the banks.
var<workgroup> sa: array<vec4<f32>, 136>;
var<workgroup> sb: array<vec4<f32>, 136>;

// A at (row, t), 0 outside.
fn big_a(row: u32, t: u32) -> f32 {
    let x = mem[p.a_off + min(row, p.m - 1u) * p.a_rs + min(t, p.k - 1u)];
    return select(0.0, x, row < p.m && t < p.k);
}

fn big_b(t: u32, col: u32) -> f32 {
    let x = mem[p.b_off + min(t, p.k - 1u) * p.b_rs + min(col, p.n - 1u) * p.b_cs];
    return select(0.0, x, col < p.n && t < p.k);
}

fn store_big(row: u32, col: u32, r: vec4<f32>) {
    if (row >= p.m) { return; }
    let o = p.c_off + row * p.c_rs + col;
    if (col < p.n) { mem[o] = finish(r.x, o, col); }
    if (col + 1u < p.n) { mem[o + 1u] = finish(r.y, o + 1u, col + 1u); }
    if (col + 2u < p.n) { mem[o + 2u] = finish(r.z, o + 2u, col + 2u); }
    if (col + 3u < p.n) { mem[o + 3u] = finish(r.w, o + 3u, col + 3u); }
}

@compute @workgroup_size(8, 8)
fn matmul_big(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row0 = (wg.y + wg.z * 65535u) * 64u;
    let col0 = wg.x * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 8u + tx;
    // This thread's share of each piece: along k, k tid % 8 of rows (or columns)
    // tid / 8 + 8 j; along n (B with b_rs not 1), column tid of k j.
    let kq = tid % 8u;
    let rq = tid / 8u;
    let along_k = p.b_rs == 1u;
    let bt = select(0u, kq, along_k);
    let bc = select(tid, rq, along_k);
    var c00 = vec4<f32>(0.0);
    var c01 = vec4<f32>(0.0);
    var c10 = vec4<f32>(0.0);
    var c11 = vec4<f32>(0.0);
    var c20 = vec4<f32>(0.0);
    var c21 = vec4<f32>(0.0);
    var c30 = vec4<f32>(0.0);
    var c31 = vec4<f32>(0.0);
    var c40 = vec4<f32>(0.0);
    var c41 = vec4<f32>(0.0);
    var c50 = vec4<f32>(0.0);
    var c51 = vec4<f32>(0.0);
    var c60 = vec4<f32>(0.0);
    var c61 = vec4<f32>(0.0);
    var c70 = vec4<f32>(0.0);
    var c71 = vec4<f32>(0.0);
    var va0 = vec4<f32>(big_a(row0 + rq + 0u, 0u + kq), big_a(row0 + rq + 8u, 0u + kq), big_a(row0 + rq + 16u, 0u + kq), big_a(row0 + rq + 24u, 0u + kq));
    var vb0 = vec4<f32>(big_b(0u + bt + select(0u, 0u, along_k), col0 + bc + select(0u, 0u, along_k)), big_b(0u + bt + select(1u, 0u, along_k), col0 + bc + select(0u, 8u, along_k)), big_b(0u + bt + select(2u, 0u, along_k), col0 + bc + select(0u, 16u, along_k)), big_b(0u + bt + select(3u, 0u, along_k), col0 + bc + select(0u, 24u, along_k)));
    var va1 = vec4<f32>(big_a(row0 + rq + 32u, 0u + kq), big_a(row0 + rq + 40u, 0u + kq), big_a(row0 + rq + 48u, 0u + kq), big_a(row0 + rq + 56u, 0u + kq));
    var vb1 = vec4<f32>(big_b(0u + bt + select(4u, 0u, along_k), col0 + bc + select(0u, 32u, along_k)), big_b(0u + bt + select(5u, 0u, along_k), col0 + bc + select(0u, 40u, along_k)), big_b(0u + bt + select(6u, 0u, along_k), col0 + bc + select(0u, 48u, along_k)), big_b(0u + bt + select(7u, 0u, along_k), col0 + bc + select(0u, 56u, along_k)));
    for (var t0 = 0u; t0 < p.k; t0 += 8u) {
        sa[kq * 17u + (rq + 0u) / 4u][rq % 4u] = va0.x;
        sa[kq * 17u + (rq + 8u) / 4u][rq % 4u] = va0.y;
        sa[kq * 17u + (rq + 16u) / 4u][rq % 4u] = va0.z;
        sa[kq * 17u + (rq + 24u) / 4u][rq % 4u] = va0.w;
        sa[kq * 17u + (rq + 32u) / 4u][rq % 4u] = va1.x;
        sa[kq * 17u + (rq + 40u) / 4u][rq % 4u] = va1.y;
        sa[kq * 17u + (rq + 48u) / 4u][rq % 4u] = va1.z;
        sa[kq * 17u + (rq + 56u) / 4u][rq % 4u] = va1.w;
        if (along_k) {
            sb[kq * 17u + (rq + 0u) / 4u][rq % 4u] = vb0.x;
            sb[kq * 17u + (rq + 8u) / 4u][rq % 4u] = vb0.y;
            sb[kq * 17u + (rq + 16u) / 4u][rq % 4u] = vb0.z;
            sb[kq * 17u + (rq + 24u) / 4u][rq % 4u] = vb0.w;
            sb[kq * 17u + (rq + 32u) / 4u][rq % 4u] = vb1.x;
            sb[kq * 17u + (rq + 40u) / 4u][rq % 4u] = vb1.y;
            sb[kq * 17u + (rq + 48u) / 4u][rq % 4u] = vb1.z;
            sb[kq * 17u + (rq + 56u) / 4u][rq % 4u] = vb1.w;
        } else {
            sb[0u + tid / 4u][tid % 4u] = vb0.x;
            sb[17u + tid / 4u][tid % 4u] = vb0.y;
            sb[34u + tid / 4u][tid % 4u] = vb0.z;
            sb[51u + tid / 4u][tid % 4u] = vb0.w;
            sb[68u + tid / 4u][tid % 4u] = vb1.x;
            sb[85u + tid / 4u][tid % 4u] = vb1.y;
            sb[102u + tid / 4u][tid % 4u] = vb1.z;
            sb[119u + tid / 4u][tid % 4u] = vb1.w;
        }
        workgroupBarrier();
        va0 = vec4<f32>(big_a(row0 + rq + 0u, t0 + 8u + kq), big_a(row0 + rq + 8u, t0 + 8u + kq), big_a(row0 + rq + 16u, t0 + 8u + kq), big_a(row0 + rq + 24u, t0 + 8u + kq));
        vb0 = vec4<f32>(big_b(t0 + 8u + bt + select(0u, 0u, along_k), col0 + bc + select(0u, 0u, along_k)), big_b(t0 + 8u + bt + select(1u, 0u, along_k), col0 + bc + select(0u, 8u, along_k)), big_b(t0 + 8u + bt + select(2u, 0u, along_k), col0 + bc + select(0u, 16u, along_k)), big_b(t0 + 8u + bt + select(3u, 0u, along_k), col0 + bc + select(0u, 24u, along_k)));
        va1 = vec4<f32>(big_a(row0 + rq + 32u, t0 + 8u + kq), big_a(row0 + rq + 40u, t0 + 8u + kq), big_a(row0 + rq + 48u, t0 + 8u + kq), big_a(row0 + rq + 56u, t0 + 8u + kq));
        vb1 = vec4<f32>(big_b(t0 + 8u + bt + select(4u, 0u, along_k), col0 + bc + select(0u, 32u, along_k)), big_b(t0 + 8u + bt + select(5u, 0u, along_k), col0 + bc + select(0u, 40u, along_k)), big_b(t0 + 8u + bt + select(6u, 0u, along_k), col0 + bc + select(0u, 48u, along_k)), big_b(t0 + 8u + bt + select(7u, 0u, along_k), col0 + bc + select(0u, 56u, along_k)));
        for (var k = 0u; k < 8u; k++) {
            let a0 = sa[k * 17u + ty]; let a1 = sa[k * 17u + 8u + ty];
            let b0 = sb[k * 17u + tx]; let b1 = sb[k * 17u + 8u + tx];
            c00 += a0.x * b0; c01 += a0.x * b1;
            c10 += a0.y * b0; c11 += a0.y * b1;
            c20 += a0.z * b0; c21 += a0.z * b1;
            c30 += a0.w * b0; c31 += a0.w * b1;
            c40 += a1.x * b0; c41 += a1.x * b1;
            c50 += a1.y * b0; c51 += a1.y * b1;
            c60 += a1.z * b0; c61 += a1.z * b1;
            c70 += a1.w * b0; c71 += a1.w * b1;
        }
        workgroupBarrier();
    }
    store_big(row0 + ty * 4u + 0u, col0 + tx * 4u, c00);
    store_big(row0 + ty * 4u + 0u, col0 + 32u + tx * 4u, c01);
    store_big(row0 + ty * 4u + 1u, col0 + tx * 4u, c10);
    store_big(row0 + ty * 4u + 1u, col0 + 32u + tx * 4u, c11);
    store_big(row0 + ty * 4u + 2u, col0 + tx * 4u, c20);
    store_big(row0 + ty * 4u + 2u, col0 + 32u + tx * 4u, c21);
    store_big(row0 + ty * 4u + 3u, col0 + tx * 4u, c30);
    store_big(row0 + ty * 4u + 3u, col0 + 32u + tx * 4u, c31);
    store_big(row0 + 32u + ty * 4u + 0u, col0 + tx * 4u, c40);
    store_big(row0 + 32u + ty * 4u + 0u, col0 + 32u + tx * 4u, c41);
    store_big(row0 + 32u + ty * 4u + 1u, col0 + tx * 4u, c50);
    store_big(row0 + 32u + ty * 4u + 1u, col0 + 32u + tx * 4u, c51);
    store_big(row0 + 32u + ty * 4u + 2u, col0 + tx * 4u, c60);
    store_big(row0 + 32u + ty * 4u + 2u, col0 + 32u + tx * 4u, c61);
    store_big(row0 + 32u + ty * 4u + 3u, col0 + tx * 4u, c70);
    store_big(row0 + 32u + ty * 4u + 3u, col0 + 32u + tx * 4u, c71);
}

// ---- Convolutions as matrices -------------------------------------------------------------
// The input: h x1 by w x2 pixels at a_off (row stride a_rs floats a pixel), channels 0..x0.
// The output: m by n pixels. Kernel x3 square, stride x4, padding x5. Columns are
// K = x0 * x3 * x3, in PyTorch's weight order (channel, ky, kx).

// col[pixel][K] at c_off.
@compute @workgroup_size(64)
fn im2col(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kk = p.x3 * p.x3;
    let big_k = p.x0 * kk;
    let e = gid.x;
    let px = row_of(gid);
    if (e >= big_k || px >= p.m * p.n) { return; }
    let c = e / kk;
    let r = e % kk;
    let iy = i32((px / p.n) * p.x4 + r / p.x3) - i32(p.x5);
    let ix = i32((px % p.n) * p.x4 + r % p.x3) - i32(p.x5);
    var v = 0.0;
    if (iy >= 0 && iy < i32(p.x1) && ix >= 0 && ix < i32(p.x2)) {
        v = mem[p.a_off + (u32(iy) * p.x2 + u32(ix)) * p.a_rs + c];
    }
    mem[p.c_off + px * big_k + e] = v;
}

// The other way, gathering: grad_in[pixel][channel] (at c_off, row stride c_rs) += the columns
// at b_off that read it. A thread a (pixel, channel), as `elementwise` (so a few channels
// don't leave most of a workgroup idle), only the taps the stride lands on visited.
@compute @workgroup_size(64)
fn col2im(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = row_of(gid) * 4096u + gid.x;
    if (i >= p.x0 * p.x1 * p.x2) { return; }
    let c = i % p.x0;
    let ipx = i / p.x0;
    let kk = p.x3 * p.x3;
    let big_k = p.x0 * kk;
    let iy = i32(ipx / p.x2) + i32(p.x5);
    let ix = i32(ipx % p.x2) + i32(p.x5);
    let s = i32(p.x4);
    var sum = 0.0;
    for (var ky = iy % s; ky < i32(p.x3) && ky <= iy; ky += s) {
        let oy = (iy - ky) / s;
        if (oy >= i32(p.m)) { continue; }
        for (var kx = ix % s; kx < i32(p.x3) && kx <= ix; kx += s) {
            let ox = (ix - kx) / s;
            if (ox >= i32(p.n)) { continue; }
            sum += mem[p.b_off + (u32(oy) * p.n + u32(ox)) * big_k + c * kk + u32(ky) * p.x3 + u32(kx)];
        }
    }
    mem[p.c_off + ipx * p.c_rs + c] += sum;
}

// ---- Max pooling (3x3) ----------------------------------------------------------------------
// In: x1 by x2 pixels at a_off, row stride a_rs, channels x0; out: m by n at c_off, contiguous;
// the winner's input pixel at b_off (same layout as out). Stride x4, padding x5 (padding never
// wins, as PyTorch's -inf).

@compute @workgroup_size(64)
fn maxpool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let px = row_of(gid);
    if (c >= p.x0 || px >= p.m * p.n) { return; }
    let y0 = i32((px / p.n) * p.x4) - i32(p.x5);
    let x0 = i32((px % p.n) * p.x4) - i32(p.x5);
    var best = -3.4e38;
    var at = 0u;
    for (var dy = 0; dy < 3; dy++) {
        let y = y0 + dy;
        if (y < 0 || y >= i32(p.x1)) { continue; }
        for (var dx = 0; dx < 3; dx++) {
            let x = x0 + dx;
            if (x < 0 || x >= i32(p.x2)) { continue; }
            let i = u32(y) * p.x2 + u32(x);
            let v = mem[p.a_off + i * p.a_rs + c];
            if (v > best) { best = v; at = i; }
        }
    }
    mem[p.c_off + px * p.x0 + c] = best;
    mem[p.b_off + px * p.x0 + c] = f32(at);
}

// grad_in (at c_off, row stride c_rs) += the gradients at a_off of the outputs this input won.
@compute @workgroup_size(64)
fn maxpool_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let ipx = row_of(gid);
    if (c >= p.x0 || ipx >= p.x1 * p.x2) { return; }
    let iy = i32(ipx / p.x2) + i32(p.x5);
    let ix = i32(ipx % p.x2) + i32(p.x5);
    let s = i32(p.x4);
    let lo_y = max((iy - 2 + s - 1) / s, 0);
    let hi_y = min(iy / s, i32(p.m) - 1);
    let lo_x = max((ix - 2 + s - 1) / s, 0);
    let hi_x = min(ix / s, i32(p.n) - 1);
    var sum = 0.0;
    for (var oy = lo_y; oy <= hi_y; oy++) {
        for (var ox = lo_x; ox <= hi_x; ox++) {
            let o = (u32(oy) * p.n + u32(ox)) * p.x0 + c;
            if (u32(mem[p.b_off + o]) == ipx) { sum += mem[p.a_off + o]; }
        }
    }
    mem[p.c_off + ipx * p.c_rs + c] += sum;
}

// ---- Gradient bits ----------------------------------------------------------------------------

// grad (c_off) *= act (a_off) > 0, over m pixels of x0 channels, row stride a_rs.
@compute @workgroup_size(64)
fn relu_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let px = row_of(gid);
    if (c >= p.x0 || px >= p.m) { return; }
    let o = px * p.a_rs + c;
    if (mem[p.a_off + o] <= 0.0) { mem[p.c_off + o] = 0.0; }
}

// Over m floats (2D as `clear`), by x0: 0, the network's input (c_off) from rgb (a_off, 3 a
// pixel) by the scale and offset at b_off (3 each); 1, back: the gradient by rgb (c_off) from
// the input's (a_off); 2, c = a; 3, c += a; 4, c = ReLU(c). (lib.rs EW_*.)
@compute @workgroup_size(64)
fn elementwise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = row_of(gid) * 4096u + gid.x;
    if (i >= p.m) { return; }
    let o = p.c_off + i;
    switch p.x0 {
        case 0u: { mem[o] = mem[p.a_off + i] * mem[p.b_off + i % 3u] + mem[p.b_off + 3u + i % 3u]; }
        case 1u: { mem[o] = mem[p.a_off + i] * mem[p.b_off + i % 3u]; }
        case 2u: { mem[o] = mem[p.a_off + i]; }
        case 3u: { mem[o] += mem[p.a_off + i]; }
        default: { mem[o] = max(mem[o], 0.0); }
    }
}

// mem[c_off + i] = 0 for i < m (2D: x over 64s, rows of 64 * 64).
@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = row_of(gid) * 4096u + gid.x;
    if (i < p.m) { mem[p.c_off + i] = 0.0; }
}

// Two sums over m items, a workgroup's share each, into partials at c_off (2 a workgroup):
// x0 = 0: of a_off's squares (and how many); x0 = 1: of |grad| at a_off (3 a pixel) where the
// lookup at b_off covers the pixel (and 3 for each such pixel); x0 = 2: of the height's
// gradient (shade_back's, 6 a pixel at a_off, the last) likewise; x0 = 3: of that gradient,
// signed; x0 = 4: the largest |a_off| over m (its finish takes the largest too).
var<workgroup> red: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    var s = vec2<f32>(0.0);
    let stride = nwg.x * 256u;
    for (var i = wg.x * 256u + lid.x; i < p.m; i += stride) {
        if (p.x0 == 0u) {
            let v = mem[p.a_off + i];
            s += vec2<f32>(v * v, 1.0);
        } else if (p.x0 == 4u) {
            s = vec2<f32>(max(s.x, abs(mem[p.a_off + i])), 1.0);
        } else if (mem[p.b_off + i * LOOKUP] >= 0.0) {
            if (p.x0 == 2u) {
                s += vec2<f32>(abs(mem[p.a_off + i * 6u + 5u]), 1.0);
            } else if (p.x0 == 3u) {
                s += vec2<f32>(mem[p.a_off + i * 6u + 5u], 1.0);
            } else {
                let g = p.a_off + i * 3u;
                s += vec2<f32>(abs(mem[g]) + abs(mem[g + 1u]) + abs(mem[g + 2u]), 3.0);
            }
        }
    }
    red[lid.x] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid.x < k) {
            if (p.x0 == 4u) { red[lid.x] = max(red[lid.x], red[lid.x + k]); } else { red[lid.x] += red[lid.x + k]; }
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        mem[p.c_off + wg.x * 2u] = red[0].x;
        mem[p.c_off + wg.x * 2u + 1u] = red[0].y;
    }
}

// The partials' totals (m workgroups' worth at a_off) to c_off, c_off + 1 (x0 = 4: the
// largest). One workgroup.
@compute @workgroup_size(256)
fn reduce_finish(@builtin(local_invocation_id) lid: vec3<u32>) {
    var s = vec2<f32>(0.0);
    for (var i = lid.x; i < p.m; i += 256u) {
        if (p.x0 == 4u) {
            s = max(s, vec2<f32>(mem[p.a_off + i * 2u], mem[p.a_off + i * 2u + 1u]));
        } else {
            s += vec2<f32>(mem[p.a_off + i * 2u], mem[p.a_off + i * 2u + 1u]);
        }
    }
    red[lid.x] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid.x < k) {
            if (p.x0 == 4u) { red[lid.x] = max(red[lid.x], red[lid.x + k]); } else { red[lid.x] += red[lid.x + k]; }
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        mem[p.c_off] = red[0].x;
        mem[p.c_off + 1u] = red[0].y;
    }
}

// The objective's gradient: grad (c_off) += act (a_off) / sum of its squares (at b_off), over
// m floats (2D as `clear`): d/d(act) of mean(act^2) / mean(act^2), but for a factor of 2.
@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = row_of(gid) * 4096u + gid.x;
    if (i < p.m) { mem[p.c_off + i] += mem[p.a_off + i] / max(mem[p.b_off], 1e-12); }
}

// ---- The dream layer ----------------------------------------------------------------------
// Surfaces (lib.rs `Surface`), SURFACE floats a pixel of the frame given (x0 by x1 pixels) at
// a_off: page (-1 for none), u, v, du/dx, dv/dx, du/dy, dv/dy, strength. The atlas's pages are
// k texels a side.

// The dream at a point of a page: its colour change (rgb) and how much of the wall it owns (a),
// bilinear.
fn dream_at(page: u32, tx: f32, ty: f32) -> vec4<f32> {
    let r = p.k;
    let t = clamp(vec2<f32>(tx, ty), vec2<f32>(0.0), vec2<f32>(f32(r - 1u)));
    let i0 = min(vec2<u32>(floor(t)), vec2<u32>(r - 2u));
    let f = t - vec2<f32>(i0);
    let base = page * r * r;
    let a = texel(base + i0.y * r + i0.x);
    let b = texel(base + i0.y * r + i0.x + 1u);
    let c = texel(base + (i0.y + 1u) * r + i0.x);
    let d = texel(base + (i0.y + 1u) * r + i0.x + 1u);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn texel(i: u32) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(atlas[i * 2u]), unpack2x16float(atlas[i * 2u + 1u]));
}

fn picture(x: u32, y: u32) -> vec3<f32> {
    return unpack4x8unorm(img[y * p.b_cs + x]).rgb;
}

// The picture blurred: the mean of each SMOOTH x SMOOTH cell (at bias_off, rgb), bilinear at a
// point in the picture's pixels. Where the dream owns the wall it stands in for the picture:
// the light, fog and colour of the moment, without the texture's own pattern.
const SMOOTH: u32 = 8u;

fn smooth_at(x: f32, y: f32) -> vec3<f32> {
    let gw = (p.x0 + SMOOTH - 1u) / SMOOTH;
    let gh = (p.x1 + SMOOTH - 1u) / SMOOTH;
    let t = clamp(vec2<f32>(x, y) / f32(SMOOTH) - 0.5, vec2<f32>(0.0), vec2<f32>(f32(gw - 1u), f32(gh - 1u)));
    let i0 = min(vec2<u32>(floor(t)), vec2<u32>(max(gw, 2u) - 2u, max(gh, 2u) - 2u));
    let f = t - vec2<f32>(i0);
    let o = p.bias_off;
    let a = vec3<f32>(mem[o + (i0.y * gw + i0.x) * 3u], mem[o + (i0.y * gw + i0.x) * 3u + 1u], mem[o + (i0.y * gw + i0.x) * 3u + 2u]);
    let b = vec3<f32>(mem[o + (i0.y * gw + i0.x + 1u) * 3u], mem[o + (i0.y * gw + i0.x + 1u) * 3u + 1u], mem[o + (i0.y * gw + i0.x + 1u) * 3u + 2u]);
    let c = vec3<f32>(mem[o + ((i0.y + 1u) * gw + i0.x) * 3u], mem[o + ((i0.y + 1u) * gw + i0.x) * 3u + 1u], mem[o + ((i0.y + 1u) * gw + i0.x) * 3u + 2u]);
    let d = vec3<f32>(mem[o + ((i0.y + 1u) * gw + i0.x + 1u) * 3u], mem[o + ((i0.y + 1u) * gw + i0.x + 1u) * 3u + 1u], mem[o + ((i0.y + 1u) * gw + i0.x + 1u) * 3u + 2u]);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// The cells' means (one thread a cell) of the x0 by x1 picture, to c_off.
@compute @workgroup_size(16, 16)
fn blur_picture(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gw = (p.x0 + SMOOTH - 1u) / SMOOTH;
    let gh = (p.x1 + SMOOTH - 1u) / SMOOTH;
    if (gid.x >= gw || gid.y >= gh) { return; }
    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var y = gid.y * SMOOTH; y < min((gid.y + 1u) * SMOOTH, p.x1); y++) {
        for (var x = gid.x * SMOOTH; x < min((gid.x + 1u) * SMOOTH, p.x0); x++) {
            sum += picture(x, y);
            n += 1.0;
        }
    }
    let o = p.c_off + (gid.y * gw + gid.x) * 3u;
    let m = sum / max(n, 1.0);
    mem[o] = m.r;
    mem[o + 1u] = m.g;
    mem[o + 2u] = m.b;
}

// The dream's relief: each page's (at a_rs in prepare and show, c_off in raise and clear_pages)
// a RELIEF_CELL-th of its texels a side, -1..1 (sunk in .. stood out), times f1 pages high.
const RELIEF_CELL: u32 = 4u;
const RELIEF_STEPS: u32 = 12u;
// How far in from a page's edge (of its side) its relief comes up.
const RELIEF_EDGE: f32 = 0.12;

fn relief_at(at: u32, page: u32, uv: vec2<f32>) -> f32 {
    let hr = p.k / RELIEF_CELL;
    let t = clamp(uv * f32(hr) - 0.5, vec2<f32>(0.0), vec2<f32>(f32(hr - 1u)));
    let i0 = min(vec2<u32>(floor(t)), vec2<u32>(hr - 2u));
    let f = t - vec2<f32>(i0);
    let o = at + page * hr * hr + i0.y * hr + i0.x;
    // Flat at the page's edges, so neighbouring pages meet on the surface and a line of sight
    // that runs off the page meets it there.
    let e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return mix(mix(mem[o], mem[o + 1u], f.x), mix(mem[o + hr], mem[o + hr + 1u], f.x), f.y) * smoothstep(0.0, RELIEF_EDGE, e);
}

// Where the line of sight through uv (on the flat surface) meets the relief, f1 * strength
// pages high at most: stepped down from that height to as deep, then between the last two steps.
fn on_relief(page: u32, uv: vec2<f32>, slope: vec2<f32>, strength: f32) -> vec2<f32> {
    let depth = p.f1 * strength;
    // Grazing views would look across the whole page; they see the relief as from 70 degrees.
    let len = length(slope);
    let sl = slope * min(1.0, 2.75 / max(len, 1e-6));
    var prev_z = depth;
    var prev_d = depth - depth * relief_at(p.a_rs, page, uv + sl * depth);
    // More steps the deeper it is, so deep relief doesn't band.
    let steps = clamp(u32(f32(RELIEF_STEPS) * depth / 0.1), RELIEF_STEPS, 64u);
    for (var i = 1u; i <= steps; i++) {
        let z = depth * (1.0 - 2.0 * f32(i) / f32(steps));
        let d = z - depth * relief_at(p.a_rs, page, uv + sl * z);
        if (d <= 0.0) {
            let zc = prev_z + (z - prev_z) * prev_d / max(prev_d - d, 1e-9);
            return uv + sl * zc;
        }
        prev_z = z;
        prev_d = d;
    }
    return uv + sl * -depth;
}

// Each touched slot's page (mem[x0 + slot]) has its relief (c_off) moved f0 of the way to its
// dream's brightness (over f1, the dream's limit) times how much the dream owns, averaged over 4 x 4
// texels, 2 apart, around each of the relief's points.
@compute @workgroup_size(16, 16)
fn raise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = p.k;
    let hr = r / RELIEF_CELL;
    if (gid.x >= hr || gid.y >= hr) { return; }
    let page = u32(mem[p.x0 + gid.z]);
    var sum = 0.0;
    for (var dy = 0u; dy < 4u; dy++) {
        let y = u32(clamp(i32(gid.y * RELIEF_CELL + dy * 2u) - 1, 0, i32(r) - 1));
        for (var dx = 0u; dx < 4u; dx++) {
            let x = u32(clamp(i32(gid.x * RELIEF_CELL + dx * 2u) - 1, 0, i32(r) - 1));
            let d = texel(page * r * r + y * r + x);
            sum += clamp((d.r + d.g + d.b) / (3.0 * max(p.f1, 1e-6)), -1.0, 1.0) * clamp(d.a, 0.0, 1.0);
        }
    }
    let o = p.c_off + page * hr * hr + gid.y * hr + gid.x;
    mem[o] = mix(mem[o], sum / 16.0, p.f0);
}

struct Seen {
    page: f32,
    tx: f32,
    ty: f32,
    strength: f32,
    // Where its point of the surface shows in the frame given (its pixels; the pixel's own
    // middle unless the surface is pushed out), and how much of the wall the dream owns there.
    pic: vec2<f32>,
    own: f32,
    // The picture and the dream laid over it, before clamping.
    colour: vec3<f32>,
}

// Where a point of the frame given (in its pixels, fractional) that lies in base pixel (bx, by),
// `off` from its middle, falls on its page (on the relief, with f1 > 0): page, texel x, y, and
// strength in a Seen (colour left black).
fn look(bx: u32, by: u32, off: vec2<f32>) -> Seen {
    let s = p.a_off + (by * p.x0 + bx) * SURFACE;
    var out: Seen;
    out.page = mem[s];
    out.strength = mem[s + 7u];
    out.pic = vec2<f32>(f32(bx), f32(by)) + 0.5 + off;
    if (out.page < 0.0) { return out; }
    out.pic = vec2<f32>(mem[s + 10u], mem[s + 11u]) + off;
    var uv = vec2<f32>(mem[s + 1u], mem[s + 2u]) + off.x * vec2<f32>(mem[s + 3u], mem[s + 4u]) + off.y * vec2<f32>(mem[s + 5u], mem[s + 6u]);
    if (p.f1 > 0.0 && out.strength > 0.0) {
        uv = on_relief(u32(out.page), uv, vec2<f32>(mem[s + 8u], mem[s + 9u]), out.strength);
    }
    out.tx = uv.x * f32(p.k) - 0.5;
    out.ty = uv.y * f32(p.k) - 0.5;
    return out;
}

// The screen layer: the dream grown on the screen (rgb, and how much more of the wall it owns;
// at c_rs, a pixel of the shown frame each, x0 * a_cs by x1 * a_cs), carried along as the view
// moves or laid on the pages. At pixel `at` of the shown frame, or (`block`) the mean over base
// pixel (bx, by)'s.
fn screen_px(at: u32, x: u32, y: u32, sw: u32) -> vec4<f32> {
    let o = at + (y * sw + x) * 4u;
    return vec4<f32>(mem[o], mem[o + 1u], mem[o + 2u], mem[o + 3u]);
}

fn screen_at(bx: u32, by: u32, at: vec2<u32>, block: bool) -> vec4<f32> {
    let up = p.a_cs;
    let sw = p.x0 * up;
    if (!block) {
        return screen_px(p.c_rs, min(at.x, sw - 1u), min(at.y, p.x1 * up - 1u), sw);
    }
    var sum = vec4<f32>(0.0);
    for (var y = by * up; y < (by + 1u) * up; y++) {
        for (var x = bx * up; x < (bx + 1u) * up; x++) {
            sum += screen_px(p.c_rs, x, y, sw);
        }
    }
    return sum / f32(up * up);
}

// ---- The dreamed height --------------------------------------------------------------------
// Each screen layer (at `at`, sw by sh pixels) is followed by its height a pixel: how far (in
// pages) the dream stands off the surface there. Each page has one too, at mem[base], a value
// every HEIGHT_CELL texels a side.
fn height_px(at: u32, x: u32, y: u32, sw: u32, sh: u32) -> f32 {
    return mem[at + 4u * sw * sh + min(y, sh - 1u) * sw + min(x, sw - 1u)];
}

fn page_height(base: u32, page: u32, tx: f32, ty: f32) -> f32 {
    let r = p.k / HEIGHT_CELL;
    let t = clamp(vec2<f32>(tx, ty) / f32(HEIGHT_CELL), vec2<f32>(0.0), vec2<f32>(f32(r - 1u)));
    let i0 = min(vec2<u32>(floor(t)), vec2<u32>(r - 2u));
    let f = t - vec2<f32>(i0);
    let o = base + page * r * r + i0.y * r + i0.x;
    return mix(mix(mem[o], mem[o + 1u], f.x), mix(mem[o + r], mem[o + r + 1u], f.x), f.y);
}

// A surface's change of place on its page (u, v) a pixel of the frame given right and down, as
// the columns of a matrix, over `scale`: from the screen's gradient of the height (per pixel of
// a frame `scale` times the frame given's) to its slope along the page, (du, dv) = M (dx, dy).
fn to_page(bx: u32, by: u32, scale: f32) -> mat2x2<f32> {
    let s = p.a_off + (by * p.x0 + bx) * SURFACE;
    let a = mem[s + 3u] / scale;
    let b = mem[s + 4u] / scale;
    let c = mem[s + 5u] / scale;
    let d = mem[s + 6u] / scale;
    let det = a * d - b * c;
    if (abs(det) < 1e-14) { return mat2x2<f32>(0.0, 0.0, 0.0, 0.0); }
    // The inverse of [[a, b], [c, d]] (rows: d/dx, d/dy of (u, v)), transposed.
    return mat2x2<f32>(d / det, -c / det, -b / det, a / det);
}

// The light a unit normal of slope g along the page takes from direction l (along the page,
// then off it, unit), against the flat surface's: and its derivative by g.
struct Shade {
    s: f32,
    ds: vec2<f32>,
}

fn shade(g: vec2<f32>, l: vec3<f32>, ambient: f32) -> Shade {
    let r = sqrt(1.0 + dot(g, g));
    let d = (l.z - dot(g, l.xy)) / r;
    var out: Shade;
    let k = (1.0 - ambient) / max(l.z, 0.05);
    out.s = ambient + k * max(d, 0.0);
    out.ds = select(vec2<f32>(0.0), k * (-l.xy / r - d * g / (r * r)), d > 0.0);
    return out;
}

// What shows at a point of the frame given (in its pixels, fractional) that lies in base pixel
// (bx, by), `off` from its middle: the picture, or as much of its blur as the dream owns there,
// with the dream's colour added. With SCREEN, the screen layer too (at shown pixel `at`, or
// over the whole base pixel with `block`), owning as much more of the wall as it does, up to f3.
fn seen(bx: u32, by: u32, off: vec2<f32>, at: vec2<u32>, block: bool) -> Seen {
    var out = look(bx, by, off);
    out.colour = picture(bx, by);
    if (out.page < 0.0) { return out; }
    let at_pic = vec2<u32>(clamp(floor(out.pic), vec2<f32>(0.0), vec2<f32>(f32(p.x0 - 1u), f32(p.x1 - 1u))));
    out.colour = picture(at_pic.x, at_pic.y);
    var d = vec4<f32>(0.0);
    if ((p.flags & SCREEN_ONLY) == 0u) {
        d = dream_at(u32(out.page), out.tx, out.ty);
    }
    var own = clamp(d.a, 0.0, 1.0);
    var layer = vec4<f32>(0.0);
    if ((p.flags & SCREEN) != 0u) {
        layer = screen_at(bx, by, at, block);
        own = clamp(min(d.a + layer.a, max(p.f3, d.a)), 0.0, 1.0);
    }
    own *= out.strength;
    out.own = own;
    if (own > 0.0) {
        out.colour = mix(out.colour, smooth_at(out.pic.x, out.pic.y), own);
    }
    // The dream changes the wall's own colour, so it's lit as the wall is.
    let light = light_on(at_pic.x, at_pic.y);
    out.colour += d.rgb * out.strength * light;
    out.colour += layer.rgb * light;
    return out;
}

// The screen layer's height at a pixel of an octave's frame (g0 times the frame given's): the
// mean of the shown pixels it covers (a shown pixel's where it's as fine as the shown frame; at
// most 6 x 6 of them, spread over it). Every shown pixel counts in some octave pixel's, or the
// ones none saw would be pushed by the height's steps with nothing to hold them, and drift into
// lines.
fn net_height(x: u32, y: u32) -> f32 {
    let up = f32(p.a_cs);
    let sw = p.x0 * p.a_cs;
    let sh = p.x1 * p.a_cs;
    let s0 = vec2<f32>(f32(x), f32(y)) * up / p.g0;
    let span = up / p.g0;
    if (span <= 1.0 + 1e-3) { return height_px(p.c_rs, u32(s0.x + 0.5 * span), u32(s0.y + 0.5 * span), sw, sh); }
    let n = min(u32(ceil(span)), 6u);
    var sum = 0.0;
    for (var j = 0u; j < n; j++) {
        for (var i = 0u; i < n; i++) {
            let q = s0 + (vec2<f32>(f32(i), f32(j)) + 0.5) * span / f32(n);
            sum += height_px(p.c_rs, u32(q.x), u32(q.y), sw, sh);
        }
    }
    return sum / f32(n * n);
}

// What an octave's pixel sees at `base` (in the frame given's pixels): as `seen`, at the
// shown pixel there where the octave's as fine as the shown frame, else over the base pixel.
fn seen_at(base: vec2<f32>) -> Seen {
    let bx = min(u32(max(base.x, 0.0)), p.x0 - 1u);
    let by = min(u32(max(base.y, 0.0)), p.x1 - 1u);
    let off = clamp(base - vec2<f32>(f32(bx), f32(by)), vec2<f32>(0.0), vec2<f32>(1.0)) - 0.5;
    let up = f32(p.a_cs);
    let at = vec2<u32>(min(base * up, vec2<f32>(f32(p.x0) * up - 1.0, f32(p.x1) * up - 1.0)));
    return seen(bx, by, off, at, p.g0 < up - 1e-3);
}

// The network's input for one octave: n by m pixels at c_off (HWC, rgb), taken from the
// frame given scaled g0 times (any scale: below 1, each pixel the mean of several looks), from
// (x3, x4) of that, rolled by (x5, x6); and each pixel's lookup at b_off (LOOKUP floats: page,
// texel x, y, strength, which channels aren't clamped).
@compute @workgroup_size(16, 16)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let sx = (gid.x + p.n - p.x5 % p.n) % p.n + p.x3;
    let sy = (gid.y + p.m - p.x6 % p.m) % p.m + p.x4;
    let base = (vec2<f32>(f32(sx), f32(sy)) + 0.5) / p.g0;
    let bx = min(u32(base.x), p.x0 - 1u);
    let by = min(u32(base.y), p.x1 - 1u);
    var s = seen_at(base);
    if (p.g0 < 0.999) {
        let k = min(u32(ceil(1.0 / p.g0)), 4u);
        var sum = vec3<f32>(0.0);
        for (var j = 0u; j < k; j++) {
            for (var i = 0u; i < k; i++) {
                let q = base + ((vec2<f32>(f32(i), f32(j)) + 0.5) / f32(k) - 0.5) / p.g0;
                sum += seen_at(q).colour;
            }
        }
        s.colour = sum / f32(k * k);
    }
    var c = clamp(s.colour, vec3<f32>(0.0), vec3<f32>(1.0));
    if ((p.flags & LAMP) != 0u) {
        // The dream lamp (slot f0 at b_rs):
        // the input shaded by the height's slope, worked out across the network's own pixels,
        // and for shade_back, the colour shaded and how the shading changes with the height's
        // gradient that way. (Only the screen layer's height: while carried along.)
        let lamp = p.b_rs + LAMP_SLOT * u32(p.f0);
        let aux = p.b_rs + LAMP_AUX + (gid.y * p.n + gid.x) * 6u;
        var base = c;
        var dh = vec2<f32>(0.0);
        if (s.page >= 0.0) {
            if (mem[lamp + 5u] > 0.5) { base = vec3<f32>(0.5); }
            let hx = (net_height(sx + 1u, sy) - net_height(max(sx, 1u) - 1u, sy)) / 2.0;
            let hy = (net_height(sx, sy + 1u) - net_height(sx, max(sy, 1u) - 1u)) / 2.0;
            let m = to_page(bx, by, p.g0);
            let sh = shade(m * vec2<f32>(hx, hy), vec3<f32>(mem[lamp], mem[lamp + 1u], mem[lamp + 2u]), mem[lamp + 3u]);
            let amount = mem[lamp + 4u];
            c = base * mix(1.0, sh.s, amount);
            dh = amount * (sh.ds * m);
        }
        mem[aux] = base.r;
        mem[aux + 1u] = base.g;
        mem[aux + 2u] = base.b;
        mem[aux + 3u] = dh.x;
        mem[aux + 4u] = dh.y;
    }
    let o = p.c_off + (gid.y * p.n + gid.x) * 3u;
    mem[o] = c.r;
    mem[o + 1u] = c.g;
    mem[o + 2u] = c.b;
    let free = (s.colour > vec3<f32>(0.0)) & (s.colour < vec3<f32>(1.0));
    let l = p.b_off + (gid.y * p.n + gid.x) * LOOKUP;
    mem[l] = s.page;
    mem[l + 1u] = s.tx;
    mem[l + 2u] = s.ty;
    mem[l + 3u] = s.strength;
    mem[l + 4u] = f32(u32(free.x) | (u32(free.y) << 1u) | (u32(free.z) << 2u));
}

// Each net pixel's gradient (a_off, 3 a pixel), over the mean |gradient| (x0: sum, count),
// times the step f0, into the accumulators of the four texels its lookup (b_off) read, by
// how much it read each. The page's accumulator slot is at mem[x1 + page] (-1: none this
// frame). n by m pixels.
@compute @workgroup_size(16, 16)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let px = gid.y * p.n + gid.x;
    let l = p.b_off + px * LOOKUP;
    let page = mem[l];
    if (page < 0.0) { return; }
    let slot_f = mem[p.x1 + u32(page)];
    if (slot_f < 0.0) { return; } // no accumulator this frame
    let slot = u32(slot_f);
    let free = u32(mem[l + 4u]);
    let mean = mem[p.x0] / max(mem[p.x0 + 1u], 1.0);
    let k = 2.0 * mem[l + 3u] * p.f0 / max(mean, 1e-12);
    let g = vec3<f32>(select(0.0, mem[p.a_off + px * 3u], (free & 1u) != 0u),
                      select(0.0, mem[p.a_off + px * 3u + 1u], (free & 2u) != 0u),
                      select(0.0, mem[p.a_off + px * 3u + 2u], (free & 4u) != 0u)) * k;
    let r = p.k;
    let t = clamp(vec2<f32>(mem[l + 1u], mem[l + 2u]), vec2<f32>(0.0), vec2<f32>(f32(r - 1u)));
    let i0 = min(vec2<u32>(floor(t)), vec2<u32>(r - 2u));
    let f = t - vec2<f32>(i0);
    let w = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let at = array<u32, 4>(i0.y * r + i0.x, i0.y * r + i0.x + 1u, (i0.y + 1u) * r + i0.x, (i0.y + 1u) * r + i0.x + 1u);
    for (var j = 0u; j < 4u; j++) {
        let a = (slot * r * r + at[j]) * 4u;
        atomicAdd(&acc[a], i32(round(g.r * w[j] * FIXED)));
        atomicAdd(&acc[a + 1u], i32(round(g.g * w[j] * FIXED)));
        atomicAdd(&acc[a + 2u], i32(round(g.b * w[j] * FIXED)));
        atomicAdd(&acc[a + 3u], i32(round(w[j] * FIXED)));
    }
}

// Zero the accumulators of m slots.
@compute @workgroup_size(16, 16)
fn clear_acc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = p.k;
    if (gid.x >= r || gid.y >= r) { return; }
    let a = ((gid.z * r + gid.y) * r + gid.x) * 4u;
    atomicStore(&acc[a], 0); atomicStore(&acc[a + 1u], 0); atomicStore(&acc[a + 2u], 0); atomicStore(&acc[a + 3u], 0);
}

// Each touched slot (z) adds its texels' steps to its page (mem[x0 + slot]): the sums and how
// much each texel was seen, spread over a 5x5 binomial within the page where few pixels land
// on it (a page seen from afar, a few pixels to many texels, gets a smooth step), divided. The step's brightness (the
// same in r, g and b) comes from that; its colour (the rest) is spread wider, the binomial's
// taps x2 texels apart, and scaled by f1, so colour comes in soft patches or not at all, not
// in rainbow speckle. First the page decays by mem[x1 + slot]; it's held within +-f0.
fn spread(slot: u32, at: vec2<u32>, gap: i32) -> vec4<f32> {
    let r = p.k;
    let bin = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
    var sum = vec4<f32>(0.0);
    for (var dy = 0; dy < 5; dy++) {
        let y = u32(clamp(i32(at.y) + (dy - 2) * gap, 0, i32(r) - 1));
        for (var dx = 0; dx < 5; dx++) {
            let x = u32(clamp(i32(at.x) + (dx - 2) * gap, 0, i32(r) - 1));
            let a = ((slot * r + y) * r + x) * 4u;
            let v = vec4<f32>(f32(atomicLoad(&acc[a])), f32(atomicLoad(&acc[a + 1u])), f32(atomicLoad(&acc[a + 2u])), f32(atomicLoad(&acc[a + 3u])));
            sum += v * (bin[dy] * bin[dx] / 256.0);
        }
    }
    return sum / FIXED;
}

@compute @workgroup_size(16, 16)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = p.k;
    if (gid.x >= r || gid.y >= r) { return; }
    let slot = gid.z;
    let page = u32(mem[p.x0 + slot]);
    let decay = mem[p.x1 + slot];
    let i = page * r * r + gid.y * r + gid.x;
    var d = texel(i) * decay;
    let near = spread(slot, gid.xy, 1);
    if (near.w > 1e-3) {
        var step = near.xyz / (near.w + 1e-3);
        // Where a pixel or more lands on this texel (up close) its own step is sharp enough;
        // the spread is for texels seen from afar, a few pixels to many texels.
        let a = ((slot * r + gid.y) * r + gid.x) * 4u;
        let own = vec4<f32>(f32(atomicLoad(&acc[a])), f32(atomicLoad(&acc[a + 1u])), f32(atomicLoad(&acc[a + 2u])), f32(atomicLoad(&acc[a + 3u]))) / FIXED;
        if (own.w > 1e-3) {
            step = mix(step, own.xyz / own.w, clamp(own.w, 0.0, 1.0));
        }
        d += vec4<f32>(vec3<f32>((step.r + step.g + step.b) / 3.0), 0.0);
        // Dreaming here takes the wall over, by f2 a step, up to f3.
        d.a = min(d.a + p.f2, max(p.f3, d.a));
    }
    if (p.f1 > 0.0) {
        let wide = spread(slot, gid.xy, i32(max(p.x2, 1u)));
        if (wide.w > 1e-3) {
            let step = wide.xyz / (wide.w + 1e-3);
            d += vec4<f32>(p.f1 * (step - vec3<f32>((step.r + step.g + step.b) / 3.0)), 0.0);
        }
    }
    let rgb = clamp(d.rgb, vec3<f32>(-p.f0), vec3<f32>(p.f0));
    atlas[i * 2u] = pack2x16float(rgb.rg);
    atlas[i * 2u + 1u] = pack2x16float(vec2<f32>(rgb.b, clamp(d.a, 0.0, 1.0)));
}

// Zero m pages listed from mem[x0].
@compute @workgroup_size(16, 16)
fn clear_pages(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = p.k;
    if (gid.x >= r || gid.y >= r) { return; }
    let i = u32(mem[p.x0 + gid.z]) * r * r + gid.y * r + gid.x;
    atlas[i * 2u] = 0u;
    atlas[i * 2u + 1u] = 0u;
    let hr = r / RELIEF_CELL;
    if (gid.x < hr && gid.y < hr) {
        mem[p.c_off + u32(mem[p.x0 + gid.z]) * hr * hr + gid.y * hr + gid.x] = 0.0;
    }
    // The dreamed height (x1).
    let dr = r / HEIGHT_CELL;
    if (gid.x < dr && gid.y < dr) {
        mem[p.x1 + u32(mem[p.x0 + gid.z]) * dr * dr + gid.y * dr + gid.x] = 0.0;
    }
}

// The torch's glint off a surface of normal n (along the page, then off it) seen from e (the
// torch is at the eye, so light, view and half way between are all e), `dist` from it, where
// the dream owns `own` of the wall: GGX with Smith's shadowing, the glint's settings at b_off
// (how bright, the torch's reach, roughness dry and wet, its colour).
fn glint(n: vec3<f32>, e: vec3<f32>, l: vec3<f32>, own: f32, dist: f32) -> vec3<f32> {
    let nv = max(dot(n, e), 1e-3);
    let nl = max(dot(n, l), 0.0);
    let nh = max(dot(n, normalize(e + l)), 0.0);
    let rough = mix(mem[p.b_off + 2u], mem[p.b_off + 3u], clamp(own, 0.0, 1.0));
    let a2 = rough * rough * rough * rough;
    let q = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (3.14159265 * q * q);
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let gv = nv / (nv * (1.0 - k) + k);
    let gl = nl / (nl * (1.0 - k) + k);
    let reach = clamp(1.0 - dist / max(mem[p.b_off + 1u], 1e-3), 0.0, 1.0);
    return vec3<f32>(mem[p.b_off + 4u], mem[p.b_off + 5u], mem[p.b_off + 6u]) * (mem[p.b_off] * d * 0.04 * gv * gl / (4.0 * nv) * reach * reach);
}

// How much brighter the dreamed height of slope g (along the page) makes the wall at base pixel
// (bx, by) than it is flat, by each light (x3: the ambient, how many, then each, the torch
// first; x4: each pixel's point and side) from where it is, the light wrapping round by g2;
// and the torch's direction along the page and its distance, for its glint.
struct Lit {
    gain: vec3<f32>,
    torch: vec3<f32>,
    torch_dist: f32,
    ok: bool,
}

fn lit_by_lights(bx: u32, by: u32, g: vec2<f32>) -> Lit {
    let at = p.x4 + (by * p.x0 + bx) * LIT;
    return lights_at(vec3<f32>(mem[at], mem[at + 1u], mem[at + 2u]), mem[at + 3u], g);
}

// The same at a point `here` of the level, the side seen coded as LIT's (0: none).
fn lights_at(here: vec3<f32>, code: f32, g: vec2<f32>) -> Lit {
    var out: Lit;
    if (code == 0.0) { return out; }
    out.ok = true;
    let axis = u32(abs(code)) - 1u;
    let side = sign(code);
    let nrm = normalize(vec3<f32>(-g, 1.0));
    let lb = p.x3;
    let ambient = vec3<f32>(mem[lb], mem[lb + 1u], mem[lb + 2u]);
    var bumped = ambient;
    var flat = ambient;
    let w = p.g2;
    let count = u32(mem[lb + 3u]);
    for (var j = 0u; j < count; j++) {
        let o = lb + 4u + j * LIGHT;
        let pos = vec3<f32>(mem[o], mem[o + 1u], mem[o + 2u]);
        let r = mem[o + 3u];
        let col = vec3<f32>(mem[o + 4u], mem[o + 5u], mem[o + 6u]);
        var to = pos;
        var weight = col;
        if (j == 0u || r > 0.0) { to = pos - here; }
        let d = max(length(to), 1e-4);
        if (j == 0u) {
            let fall = max(1.0 - d / max(r, 1e-3), 0.0);
            weight = col * fall * fall;
        } else if (r > 0.0) {
            weight = col / (1.0 + (d / r) * (d / r));
        }
        let l = normalize(vec3<f32>(pick(to, axis), to[axis] * side));
        if (j == 0u) {
            out.torch = l;
            out.torch_dist = d;
        }
        bumped += weight * (max(dot(nrm, l), 0.0) + w);
        flat += weight * (max(l.z, 0.0) + w);
    }
    out.gain = bumped / max(flat, vec3<f32>(1e-4));
    return out;
}

// What a shown pixel sees where the shell's march met the relief nearer than any creature
// (globals, lib.rs: 2 its triangles in img (0: no march), 3 its feet's places, light, albedo and
// texel, 4 their rows, 8 the views, 9 the creatures' depth (0: none), 10 the
// picture under the dream the renderer's own): the wall at the hit's foot lit as own_picture
// lights it, from the hit, fogged by its distance, with the dream on its page there and the
// screen layer where the foot shows on the flat screen, as `seen`; and for lighting the
// height, its slope along the page, the eye's direction (along the page, then off it), its
// distance, the hit and the side seen (LIT's code).
struct ShellSeen {
    ok: bool,
    seen: Seen,
    g: vec2<f32>,
    eye: vec3<f32>,
    dist: f32,
    here: vec3<f32>,
    code: f32,
}

// The point of the plane n . x = d at pick's coordinates pf along it.
fn plane_point(pf: vec2<f32>, axis: u32, n: vec3<f32>, d: f32) -> vec3<f32> {
    if (axis == 0u) { return vec3<f32>((d - n.y * pf.x - n.z * pf.y) / n.x, pf.x, pf.y); }
    if (axis == 1u) { return vec3<f32>(pf.x, (d - n.x * pf.x - n.z * pf.y) / n.y, pf.y); }
    return vec3<f32>(pf.x, pf.y, (d - n.x * pf.x - n.y * pf.y) / n.z);
}

fn shell_seen(at: vec2<u32>) -> ShellSeen {
    var out: ShellSeen;
    let tris = bitcast<u32>(mem[2]);
    if (tris == 0u) { return out; }
    let o = at.y * bitcast<u32>(mem[4]) + at.x * 4u;
    let tri = img[tris + o];
    if (tri == 0u) { return out; }
    let t = bitcast<f32>(img[tris + o + 2u]);
    let up = f32(p.x2);
    let views = bitcast<u32>(mem[8]);
    let v = view_at(views);
    let bx = min(at.x / p.x2, p.x0 - 1u);
    let by = min(at.y / p.x2, p.x1 - 1u);
    let creatures = bitcast<u32>(mem[9]);
    if (creatures != 0u) {
        let z = mem[creatures + by * p.x0 + bx];
        if (z < 3e38 && z < t) { return out; }
    }
    out.ok = true;
    let x = v.pos + ray(v, (f32(at.x) + 0.5) / up, (f32(at.y) + 0.5) / up) * t;
    let looks = bitcast<u32>(mem[3]);
    let lk = vec3<u32>(img[looks + o + 1u], img[looks + o + 2u], img[looks + o + 3u]);
    let ti = (tri - 1u) * 4u;
    let n = vec3<f32>(bitcast<f32>(level[ti]), bitcast<f32>(level[ti + 1u]), bitcast<f32>(level[ti + 2u]));
    let an = abs(n);
    var axis = 0u;
    if (an.y > an.x) { axis = 1u; }
    if (an.z > max(an.x, an.y)) { axis = 2u; }
    // The foot: its place on its tile as kept, the tile the hit's over (dropped onto the
    // plane), within half a tile of it.
    let place = unpack2x16unorm(img[looks + o]);
    let plane_d = bitcast<f32>(level[ti + 3u]);
    let dropped = pick(x - n * (dot(n, x) - plane_d), axis);
    var pf = floor(dropped) + place;
    pf -= select(vec2<f32>(0.0), vec2<f32>(1.0), pf - dropped > vec2<f32>(0.5));
    pf += select(vec2<f32>(0.0), vec2<f32>(1.0), pf - dropped < vec2<f32>(-0.5));
    let foot = plane_point(pf, axis, n, plane_d);
    let to_eye = v.pos - x;
    let side = select(-1.0, 1.0, (n * sign(dot(to_eye, n)))[axis] >= 0.0);
    out.here = x;
    out.code = f32(axis + 1u) * side;
    out.eye = normalize(vec3<f32>(pick(to_eye, axis), to_eye[axis] * side));
    out.g = unpack2x16float(img[tris + o + 1u]);
    // The wall, lit as own_picture lights it (the torch where it's held), or the picture given
    // where the foot shows in it.
    let vo = views + 16u;
    let torch = vec3<f32>(mem[vo], mem[vo + 1u], mem[vo + 2u]);
    let reach = mem[vo + 3u];
    let fog_colour = vec3<f32>(mem[vo + 4u], mem[vo + 5u], mem[vo + 6u]);
    let held = v.pos + v.r0 * mem[views + 23u] + v.r1 * mem[views + 24u] + v.r2 * mem[views + 25u];
    let to_torch = held - x;
    let td = max(length(to_torch), 1.0 / 256.0);
    let fall = max(1.0 - td / max(reach, 1e-3), 0.0);
    let lambert = max(dot(n, to_torch) / td, 0.0);
    let light = unpack4x8unorm(lk.x).rgb * 2.0 + unpack4x8unorm(lk.y).rgb * 2.0 * torch * fall * fall * lambert;
    let pic = project(v, foot);
    let at_pic = vec2<u32>(clamp(floor(pic.xy), vec2<f32>(0.0), vec2<f32>(f32(p.x0 - 1u), f32(p.x1 - 1u))));
    out.dist = length(to_eye);
    let fog = clamp((out.dist - v.fog.x) / max(v.fog.y - v.fog.x, 1e-6), 0.0, 1.0);
    var wall = picture(at_pic.x, at_pic.y);
    if (mem[10] > 0.0) {
        wall = mix(min(unpack4x8unorm(lk.z).rgb * light, vec3<f32>(1.0)), fog_colour, fog);
    }
    var s: Seen;
    s.page = -1.0;
    s.strength = 1.0 - fog;
    s.pic = pic.xy;
    s.colour = wall;
    let page_bits = img[tris + o + 3u] & 0x0fffffffu;
    let page = select(page_bits, NONE, page_bits == 0x0fffffffu);
    if (page != NONE) {
        s.page = f32(page);
        s.tx = place.x * f32(p.k) - 0.5;
        s.ty = place.y * f32(p.k) - 0.5;
        var d = vec4<f32>(0.0);
        if ((p.flags & SCREEN_ONLY) == 0u) {
            d = dream_at(page, s.tx, s.ty);
        }
        var own = clamp(d.a, 0.0, 1.0);
        var layer = vec4<f32>(0.0);
        if ((p.flags & SCREEN) != 0u) {
            let sw = p.x0 * p.a_cs;
            let q = vec2<u32>(clamp(floor(pic.xy * f32(p.a_cs)), vec2<f32>(0.0), vec2<f32>(f32(sw - 1u), f32(p.x1 * p.a_cs - 1u))));
            layer = screen_px(p.c_rs, q.x, q.y, sw);
            own = clamp(min(d.a + layer.a, max(p.f3, d.a)), 0.0, 1.0);
        }
        own *= s.strength;
        s.own = own;
        if (own > 0.0) {
            s.colour = mix(s.colour, smooth_at(pic.x, pic.y), own);
        }
        // The dream lit as the wall is (light_on's).
        var lit = vec3<f32>(1.0);
        if (bitcast<u32>(mem[0]) != 0u && mem[1] > 0.0) { lit = mix(vec3<f32>(1.0), light, mem[1]); }
        s.colour += (d.rgb * s.strength + layer.rgb) * lit;
    }
    out.seen = s;
    return out;
}

// The frame shown: the frame given scaled up x2 times (n by m pixels, RGBA8 at img[c_off]) with
// the dream laid over; f0 = how sharp the dream is, 0 drawing it at the frame given's own
// pixels, 1 at the shown ones.
@compute @workgroup_size(16, 16)
fn show(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let up = p.x2;
    let bx = min(gid.x / up, p.x0 - 1u);
    let by = min(gid.y / up, p.x1 - 1u);
    let off = ((vec2<f32>(f32(gid.x % up), f32(gid.y % up)) + 0.5) / f32(up) - 0.5) * p.f0;
    // What the shell's march met (shell_seen), or else the base pixel's surface.
    let hit = shell_seen(gid.xy);
    var s: Seen;
    if (hit.ok) { s = hit.seen; } else { s = seen(bx, by, off, gid.xy, false); }
    var c = clamp(s.colour, vec3<f32>(0.0), vec3<f32>(1.0));
    // The dreamed height lit by the torch at the eye: its diffuse light against the flat
    // wall's by f2 (or the height alone, so lit, in grey), and its glint (b_off: the light's
    // settings, below), GGX, wet where the dream owns the wall.
    let light = p.b_off != 0u;
    if ((p.f2 > 0.0 || light || (p.flags & SHOW_HEIGHT) != 0u) && s.page >= 0.0) {
        var g = vec2<f32>(0.0);
        if (hit.ok) {
            g = hit.g;
        } else if ((p.flags & SCREEN) != 0u) {
            let sw = p.n;
            let sh = p.m;
            let hx = (height_px(p.c_rs, gid.x + 1u, gid.y, sw, sh) - height_px(p.c_rs, max(gid.x, 1u) - 1u, gid.y, sw, sh)) / 2.0;
            let hy = (height_px(p.c_rs, gid.x, gid.y + 1u, sw, sh) - height_px(p.c_rs, gid.x, max(gid.y, 1u) - 1u, sw, sh)) / 2.0;
            g = to_page(bx, by, f32(up)) * vec2<f32>(hx, hy);
        }
        let sf = p.a_off + (by * p.x0 + bx) * SURFACE;
        var eye = normalize(vec3<f32>(mem[sf + 8u], mem[sf + 9u], 1.0));
        if (hit.ok) { eye = hit.eye; }
        // Matte, the light wrapping round by g2 (half-Lambert at 1), against the flat wall's
        // light the same way: a slope turned from the torch darkens and one turned to it
        // brightens, but by a little where the wall's seen glancing (where the flat wall's
        // own light is small, plain Lambert's ratio runs from black to many times as bright).
        let r = sqrt(1.0 + dot(g, g));
        let d = max((eye.z - dot(g, eye.xy)) / r, 0.0);
        var gain = vec3<f32>((d + p.g2) / (eye.z + p.g2));
        var torch = eye;
        var torch_dist = mem[sf + 12u];
        if (hit.ok) { torch_dist = hit.dist; }
        if (p.x3 != 0u) {
            var by_lights: Lit;
            if (hit.ok) { by_lights = lights_at(hit.here, hit.code, g); } else { by_lights = lit_by_lights(bx, by, g); }
            if (by_lights.ok) {
                gain = by_lights.gain;
                torch = by_lights.torch;
                torch_dist = by_lights.torch_dist;
            }
        }
        if ((p.flags & SHOW_HEIGHT) != 0u) {
            c = vec3<f32>(clamp(0.5 * dot(gain, vec3<f32>(1.0 / 3.0)), 0.0, 1.0));
        } else {
            c = c * mix(vec3<f32>(1.0), gain, p.f2);
            if (light) {
                c += glint(normalize(vec3<f32>(-g, 1.0)), eye, torch, s.own, torch_dist) * s.strength;
            }
            c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }
    // The shell's hit alone (globals 2..7: its triangles in img, its looks, their rows, the
    // height's limit, ..., shown alone): grey by the hit's height above its plane, lit by its
    // slope; dark red for none.
    let shell_at = bitcast<u32>(mem[2]);
    if (shell_at != 0u && mem[7] > 0.0) {
        let o = gid.y * bitcast<u32>(mem[4]) + gid.x * 4u;
        let tri = img[shell_at + o];
        if (tri == 0u) {
            c = vec3<f32>(0.25, 0.0, 0.0);
        } else {
            let v = view_at(bitcast<u32>(mem[8]));
            let up = f32(p.x2);
            let x = v.pos + ray(v, (f32(gid.x) + 0.5) / up, (f32(gid.y) + 0.5) / up) * bitcast<f32>(img[shell_at + o + 2u]);
            let ti = (tri - 1u) * 4u;
            let n = vec3<f32>(bitcast<f32>(level[ti]), bitcast<f32>(level[ti + 1u]), bitcast<f32>(level[ti + 2u]));
            let h = dot(n, x) - bitcast<f32>(level[ti + 3u]);
            let g = unpack2x16float(img[shell_at + o + 1u]);
            let lit = 0.55 + 0.45 * dot(normalize(vec3<f32>(-g, 1.0)), normalize(vec3<f32>(0.45, 0.55, 0.7)));
            c = vec3<f32>(clamp(0.5 + 0.5 * h / max(mem[5], 1e-6), 0.0, 1.0) * lit);
        }
    }
    img[p.c_off + gid.y * p.b_rs + gid.x] = pack4x8unorm(vec4<f32>(c, 1.0));
}

// ---- The screen layer ---------------------------------------------------------------------
// While the view stays still the dream grows on the screen, at the shown frame's pixels, not on
// the pages: sharp, and with no G-buffer to make. When the view moves it's laid on the pages
// (bake, then apply) and cleared.

// An octave's gradient (a_off, 3 a pixel, n by m, rolled by (x5, x6)) at a point of the octave's
// crop, unrolled, bilinear.
fn grad_at(u: vec2<f32>) -> vec3<f32> {
    let t = clamp(u, vec2<f32>(0.0), vec2<f32>(f32(p.n - 1u), f32(p.m - 1u)));
    let i0 = min(vec2<u32>(floor(t)), vec2<u32>(max(p.n, 2u) - 2u, max(p.m, 2u) - 2u));
    let f = t - vec2<f32>(i0);
    var g = array<vec3<f32>, 4>();
    for (var j = 0u; j < 4u; j++) {
        let x = (i0.x + (j & 1u) + p.x5) % p.n;
        let y = (i0.y + (j >> 1u) + p.x6) % p.m;
        let o = p.a_off + (y * p.n + x) * 3u;
        g[j] = vec3<f32>(mem[o], mem[o + 1u], mem[o + 2u]);
    }
    return mix(mix(g[0], g[1], f.x), mix(g[2], g[3], f.x), f.y);
}

// One step of an octave (its sizes, crop, roll and scale g0 as prepare's; lookup at b_off) onto
// the screen layer (c_rs; a_rs by b_rs pixels, a_cs times the frame given's): each shown pixel
// the octave covers takes the gradient where it falls, over the mean |gradient| (x0), times
// 2 f0 times the strength: its brightness, and f1 of its colour spread over the binomial's taps
// k of the frame given's pixels apart (k g0 of the octave's). Held within +-f2; the wall owned
// f3 more.
@compute @workgroup_size(16, 16)
fn screen_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let scale = p.g0 / f32(p.a_cs);
    let origin = vec2<u32>(vec2<f32>(f32(p.x3), f32(p.x4)) / scale);
    let at = origin + gid.xy;
    if (f32(gid.x) >= f32(p.n) / scale || f32(gid.y) >= f32(p.m) / scale || at.x >= p.a_rs || at.y >= p.b_rs) { return; }
    let u = (vec2<f32>(at) + 0.5) * scale - 0.5 - vec2<f32>(f32(p.x3), f32(p.x4));
    let near = vec2<u32>(clamp(round(u), vec2<f32>(0.0), vec2<f32>(f32(p.n - 1u), f32(p.m - 1u))));
    let px = ((near.y + p.x6) % p.m) * p.n + (near.x + p.x5) % p.n;
    let l = p.b_off + px * LOOKUP;
    if (mem[l] < 0.0) { return; }
    let free = u32(mem[l + 4u]);
    let mean = mem[p.x0] / max(mem[p.x0 + 1u], 1.0);
    // Faded out over g1 of the octave's pixels towards a piece's edges (the fine octave's),
    // where the net sees past what its step can reach: else each piece leaves its outline, in
    // the colour and, where the next steps take that for an edge, in the height.
    var fade = 1.0;
    if (p.g1 > 0.0) {
        let d = min(min(u.x + 0.5, f32(p.n) - 0.5 - u.x), min(u.y + 0.5, f32(p.m) - 0.5 - u.y));
        fade = smoothstep(0.0, p.g1, d);
    }
    let k = 2.0 * mem[l + 3u] * p.f0 * fade / max(mean, 1e-12);
    let g = grad_at(u);
    var step = vec3<f32>((g.r + g.g + g.b) / 3.0);
    if (p.f1 > 0.0) {
        let bin = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
        var wide = vec3<f32>(0.0);
        for (var dy = 0; dy < 5; dy++) {
            for (var dx = 0; dx < 5; dx++) {
                wide += grad_at(u + vec2<f32>(f32(dx - 2), f32(dy - 2)) * (f32(p.k) * p.g0)) * (bin[dy] * bin[dx] / 256.0);
            }
        }
        step += p.f1 * (wide - vec3<f32>((wide.r + wide.g + wide.b) / 3.0));
    }
    step *= k;
    step = vec3<f32>(select(0.0, step.r, (free & 1u) != 0u), select(0.0, step.g, (free & 2u) != 0u), select(0.0, step.b, (free & 4u) != 0u));
    let o = p.c_rs + (at.y * p.a_rs + at.x) * 4u;
    let v = clamp(vec3<f32>(mem[o], mem[o + 1u], mem[o + 2u]) + step, vec3<f32>(-p.f2), vec3<f32>(p.f2));
    mem[o] = v.r;
    mem[o + 1u] = v.g;
    mem[o + 2u] = v.b;
    mem[o + 3u] = min(mem[o + 3u] + p.f3, 1.0);
    // The height too (bias_off: the dream lamp's slots, this octave's x1, and shade_back's
    // gradient after them), over its own mean |gradient| (c_off), less its signed mean (x2, if
// not 0: rising and sinking alike), held within +-the most.
    if (p.bias_off != 0u) {
        let lamp = p.bias_off + LAMP_SLOT * p.x1;
        let mean_h = mem[p.c_off] / max(mem[p.c_off + 1u], 1.0);
        let kh = 2.0 * mem[l + 3u] * mem[lamp + 6u] * fade / max(mean_h, 1e-12);
        let h = p.c_rs + 4u * p.a_rs * p.b_rs + at.y * p.a_rs + at.x;
        // Pushes outward shrink as the height nears the most (lamp + 7), so it eases into it
        // rather than piling up flat against it; and a share (lamp + 8) lets go each step, so it
        // settles where the network's push holds it.
        let most = max(mem[lamp + 7u], 1e-6);
        let was = mem[h];
        var centre = 0.0;
        if (p.x2 != 0u) { centre = mem[p.x2] / max(mem[p.x2 + 1u], 1.0); }
        let push = kh * (height_grad_at(u, p.bias_off + LAMP_AUX) - centre);
        let room = select(1.0, max(1.0 - abs(was) / most, 0.0), push * was > 0.0);
        mem[h] = clamp((was + push * room) * (1.0 - mem[lamp + 8u] * fade), -most, most);
    }
}

// shade_back's height gradient (6 floats a pixel at `aux`, the last) as grad_at reads.
fn height_grad_at(u: vec2<f32>, aux: u32) -> f32 {
    let t = clamp(u, vec2<f32>(0.0), vec2<f32>(f32(p.n - 1u), f32(p.m - 1u)));
    let i0 = min(vec2<u32>(floor(t)), vec2<u32>(max(p.n, 2u) - 2u, max(p.m, 2u) - 2u));
    let f = t - vec2<f32>(i0);
    var g = array<f32, 4>();
    for (var j = 0u; j < 4u; j++) {
        let x = (i0.x + (j & 1u) + p.x5) % p.n;
        let y = (i0.y + (j >> 1u) + p.x6) % p.m;
        g[j] = mem[aux + (y * p.n + x) * 6u + 5u];
    }
    return mix(mix(g[0], g[1], f.x), mix(g[2], g[3], f.x), f.y);
}

// The network's input gradient (a_off, 3 a pixel, n by m) back through the dream lamp's shading
// (prepare's, 6 floats a pixel at bias_off + LAMP_AUX: the colour shaded and the shading's change by
// the height's gradient across and down) to the height of each pixel, into the last float:
// each pixel's height counts in its neighbours' central differences.
fn shade_q(px: u32) -> f32 {
    let a = p.bias_off + LAMP_AUX + px * 6u;
    let g = p.a_off + px * 3u;
    return mem[g] * mem[a] + mem[g + 1u] * mem[a + 1u] + mem[g + 2u] * mem[a + 2u];
}

// The height's gradient (the last of the lamp's work, 6 floats a pixel at bias_off + LAMP_AUX,
// n by m, rolled by x5, x6) blurred one way (x0: 0 across, 1 down; to the picture's edges) by a
// Gaussian of f0 pixels, from float x1 of each pixel's work to float x2: so the height grows in
// broad shapes, not a pixel's speckle.
@compute @workgroup_size(16, 16)
fn blur_height(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let aux = p.bias_off + LAMP_AUX;
    let r = i32(ceil(2.0 * p.f0));
    var sum = 0.0;
    var wsum = 0.0;
    // Neighbours in the picture's own places (unrolled by x5, x6), stopping at its edges: the
    // roll moves every step, and blurring round it would streak the seam wherever it fell.
    let roll = vec2<i32>(i32(p.x5 % p.n), i32(p.x6 % p.m));
    let size = vec2<i32>(i32(p.n), i32(p.m));
    let own = (vec2<i32>(gid.xy) - roll + size) % size;
    for (var k = -r; k <= r; k++) {
        var q = own;
        if (p.x0 == 0u) { q.x += k; } else { q.y += k; }
        if (any(q < vec2<i32>(0)) || any(q >= size)) { continue; }
        let at = vec2<u32>((q + roll) % size);
        let w = exp(-0.5 * f32(k * k) / max(p.f0 * p.f0, 1e-6));
        sum += w * mem[aux + (at.y * p.n + at.x) * 6u + p.x1];
        wsum += w;
    }
    // Times f1 (0: 1), added to what's there if x3.
    let o = aux + (gid.y * p.n + gid.x) * 6u + p.x2;
    let v = select(1.0, p.f1, p.f1 > 0.0) * sum / wsum;
    if (p.x3 != 0u) { mem[o] += v; } else { mem[o] = v; }
}

// The height's step (the lamp's work's last float, bias_off + LAMP_AUX, n by m) over its mean
// size (results at x1) mixed with the brightness of the network's colour step (a_off, 3 a
// pixel) over its own (x0): f0 of the latter (negative: the height sinks where it brightens).
@compute @workgroup_size(16, 16)
fn height_mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let px = gid.y * p.n + gid.x;
    let a = p.bias_off + LAMP_AUX + px * 6u + 5u;
    let g = p.a_off + px * 3u;
    let mean_c = mem[p.x0] / max(mem[p.x0 + 1u], 1.0);
    let mean_h = mem[p.x1] / max(mem[p.x1 + 1u], 1.0);
    let bright = (mem[g] + mem[g + 1u] + mem[g + 2u]) / 3.0;
    mem[a] = (1.0 - abs(p.f0)) * mem[a] / max(mean_h, 1e-12) + p.f0 * bright / max(mean_c, 1e-12);
}

// The network's input gradient (a_off, 3 a pixel, n by m) blurred one way (x0: 0 across, 1
// down; clamped at the edges) by a Gaussian of f0 pixels, into c_off.
@compute @workgroup_size(16, 16)
fn blur_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let r = i32(ceil(2.0 * p.f0));
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var k = -r; k <= r; k++) {
        let w = exp(-0.5 * f32(k * k) / max(p.f0 * p.f0, 1e-6));
        var x = i32(gid.x);
        var y = i32(gid.y);
        if (p.x0 == 0u) { x = clamp(x + k, 0, i32(p.n) - 1); } else { y = clamp(y + k, 0, i32(p.m) - 1); }
        let o = p.a_off + (u32(y) * p.n + u32(x)) * 3u;
        sum += w * vec3<f32>(mem[o], mem[o + 1u], mem[o + 2u]);
        wsum += w;
    }
    let o = p.c_off + (gid.y * p.n + gid.x) * 3u;
    mem[o] = sum.x / wsum;
    mem[o + 1u] = sum.y / wsum;
    mem[o + 2u] = sum.z / wsum;
}

@compute @workgroup_size(16, 16)
fn shade_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let aux = p.bias_off + LAMP_AUX;
    // Neighbours in the picture's own places (unrolled by x5, x6, as prepare took its slopes),
    // not across the roll's seam.
    let roll = vec2<i32>(i32(p.x5 % p.n), i32(p.x6 % p.m));
    let size = vec2<i32>(i32(p.n), i32(p.m));
    let own = (vec2<i32>(gid.xy) - roll + size) % size;
    var g = 0.0;
    let steps = array<vec3<i32>, 4>(vec3<i32>(-1, 0, 1), vec3<i32>(1, 0, -1), vec3<i32>(0, -1, 1), vec3<i32>(0, 1, -1));
    for (var i = 0u; i < 4u; i++) {
        let o = own + steps[i].xy;
        if (any(o < vec2<i32>(0)) || any(o >= size)) { continue; }
        let at = vec2<u32>((o + roll) % size);
        let q = at.y * p.n + at.x;
        g += f32(steps[i].z) * shade_q(q) * mem[aux + q * 6u + 3u + i / 2u];
    }
    mem[aux + (gid.y * p.n + gid.x) * 6u + 5u] = 0.5 * g;
}

// The screen layer (c_rs, x0 * a_cs by x1 * a_cs pixels; its rows from x5 up to m) into the
// accumulators of the texels each shown pixel falls on, by the surfaces it was grown over (as
// show looks them up), for apply to add to the pages or store to put in their place. A page's
// accumulator slot is at mem[b_off + page] (-1: none). x6: the colour (0), how much of the wall
// it owns (1), the height (2).
@compute @workgroup_size(16, 16)
fn bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let up = p.a_cs;
    let at = vec2<u32>(gid.x, gid.y + p.x5);
    if (at.x >= p.x0 * up || at.y >= min(p.x1 * up, p.m)) { return; }
    let bx = at.x / up;
    let by = at.y / up;
    let off = (vec2<f32>(f32(at.x % up), f32(at.y % up)) + 0.5) / f32(up) - 0.5;
    let s = look(bx, by, off);
    if (s.page < 0.0) { return; }
    let slot_f = mem[p.b_off + u32(s.page)];
    if (slot_f < 0.0) { return; }
    let slot = u32(slot_f);
    // The pages' dream shows times the strength; the screen layer's as it is. x6 = 1: how
    // much of the wall it owns instead, in the first channel.
    let v = screen_at(bx, by, at, false);
    var g = v.rgb / max(s.strength, 0.05);
    if (p.x6 == 1u) {
        g = vec3<f32>(v.a, 0.0, 0.0);
    } else if (p.x6 == 2u) {
        g = vec3<f32>(height_px(p.c_rs, at.x, at.y, p.x0 * up, p.x1 * up), 0.0, 0.0);
    }
    let r = p.k;
    let t = clamp(vec2<f32>(s.tx, s.ty), vec2<f32>(0.0), vec2<f32>(f32(r - 1u)));
    let i0 = min(vec2<u32>(floor(t)), vec2<u32>(r - 2u));
    let f = t - vec2<f32>(i0);
    let w = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let texels = array<u32, 4>(i0.y * r + i0.x, i0.y * r + i0.x + 1u, (i0.y + 1u) * r + i0.x, (i0.y + 1u) * r + i0.x + 1u);
    for (var j = 0u; j < 4u; j++) {
        let a = (slot * r * r + texels[j]) * 4u;
        atomicAdd(&acc[a], i32(round(g.r * w[j] * FIXED)));
        atomicAdd(&acc[a + 1u], i32(round(g.g * w[j] * FIXED)));
        atomicAdd(&acc[a + 2u], i32(round(g.b * w[j] * FIXED)));
        atomicAdd(&acc[a + 3u], i32(round(w[j] * FIXED)));
    }
}

// Where each pixel of the frame given was in the last (bias_off: MOTION floats a pixel, x0 by x1:
// where its middle was in the last frame given's pixels, and how that changes a pixel right and
// a pixel down; x below -1e8 for nowhere: off the screen or hidden then): the screen layer
// carried there from the last frame's (b_rs) to this one's (c_rs), a shown pixel (a_cs times
// the frame given's) each, Catmull-Rom (held within its middle four's range, so it doesn't
// ring); where it wasn't seen, nothing, or (FILL) the pages' dream there (surfaces at a_off, as
// look). Its height likewise (the pages' at c_off).
const MOTION: u32 = 6u;

fn cubic(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let up = p.a_cs;
    let sw = p.x0 * up;
    let sh = p.x1 * up;
    if (gid.x >= sw || gid.y >= sh) { return; }
    let o = p.c_rs + (gid.y * sw + gid.x) * 4u;
    let bx = gid.x / up;
    let by = gid.y / up;
    let off = (vec2<f32>(f32(gid.x % up), f32(gid.y % up)) + 0.5) / f32(up) - 0.5;
    let m = p.bias_off + (by * p.x0 + bx) * MOTION;
    var v = vec4<f32>(0.0);
    var h = 0.0;
    if (mem[m] <= -1e8 && (p.flags & FILL) != 0u) {
        let s = look(bx, by, off);
        if (s.page >= 0.0) {
            let d = dream_at(u32(s.page), s.tx, s.ty);
            v = vec4<f32>(d.rgb * s.strength, d.a);
            h = page_height(p.c_off, u32(s.page), s.tx, s.ty);
        }
    }
    if (mem[m] > -1e8) {
        let was = vec2<f32>(mem[m], mem[m + 1u]) + off.x * vec2<f32>(mem[m + 2u], mem[m + 3u]) + off.y * vec2<f32>(mem[m + 4u], mem[m + 5u]);
        let t = was * f32(up) - 0.5;
        if (t.x > -0.5 && t.y > -0.5 && t.x < f32(sw) - 0.5 && t.y < f32(sh) - 0.5) {
            let i0 = vec2<i32>(floor(t));
            let f = t - vec2<f32>(i0);
            let wx = cubic(f.x);
            let wy = cubic(f.y);
            var lo = vec4<f32>(1e9);
            var hi = vec4<f32>(-1e9);
            var hlo = 1e9;
            var hhi = -1e9;
            for (var j = 0; j < 4; j++) {
                let y = u32(clamp(i0.y + j - 1, 0, i32(sh) - 1));
                for (var i = 0; i < 4; i++) {
                    let x = u32(clamp(i0.x + i - 1, 0, i32(sw) - 1));
                    let s = screen_px(p.b_rs, x, y, sw);
                    let hs = height_px(p.b_rs, x, y, sw, sh);
                    v += s * (wx[i] * wy[j]);
                    h += hs * (wx[i] * wy[j]);
                    if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
                        lo = min(lo, s);
                        hi = max(hi, s);
                        hlo = min(hlo, hs);
                        hhi = max(hhi, hs);
                    }
                }
            }
            v = clamp(v, lo, hi);
            h = clamp(h, hlo, hhi);
        }
    }
    mem[o] = v.r;
    mem[o + 1u] = v.g;
    mem[o + 2u] = v.b;
    mem[o + 3u] = v.a;
    mem[p.c_rs + 4u * sw * sh + gid.y * sw + gid.x] = h;
}

// Each touched slot's page (mem[x0 + slot]) takes the screen layer's dream baked into its
// accumulators in place of its own, where the screen covers it (spread as in apply, for texels
// seen from afar): its colour (x2 = 0, held within +-f0), how much of the wall it owns (x2 = 1),
// or its height (x2 = 2, into the pages' heights at c_off).
// The accumulators of a 16 x 16 block of texels and the two round it, read once for its
// spreads (as `spread` at gap 1 reads them, clamped at the page's edges).
var<workgroup> acc_block: array<vec4<f32>, 400>;

@compute @workgroup_size(16, 16)
fn store(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = p.k;
    let slot = gid.z;
    for (var c = lid.y * 16u + lid.x; c < 400u; c += 256u) {
        let x = u32(clamp(i32(wg.x * 16u + c % 20u) - 2, 0, i32(r) - 1));
        let y = u32(clamp(i32(wg.y * 16u + c / 20u) - 2, 0, i32(r) - 1));
        let a = ((slot * r + y) * r + x) * 4u;
        acc_block[c] = vec4<f32>(f32(atomicLoad(&acc[a])), f32(atomicLoad(&acc[a + 1u])), f32(atomicLoad(&acc[a + 2u])), f32(atomicLoad(&acc[a + 3u])));
    }
    workgroupBarrier();
    if (gid.x >= r || gid.y >= r) { return; }
    // The height's kept at every HEIGHT_CELL-th texel only.
    if (p.x2 == 2u && (gid.x % HEIGHT_CELL != 0u || gid.y % HEIGHT_CELL != 0u)) { return; }
    let i = u32(mem[p.x0 + slot]) * r * r + gid.y * r + gid.x;
    let bin = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
    var sum = vec4<f32>(0.0);
    for (var dy = 0u; dy < 5u; dy++) {
        for (var dx = 0u; dx < 5u; dx++) {
            sum += acc_block[(lid.y + dy) * 20u + lid.x + dx] * (bin[dy] * bin[dx] / 256.0);
        }
    }
    let near = sum / FIXED;
    if (near.w <= 1e-3) { return; }
    var v = near.xyz / near.w;
    let own = acc_block[(lid.y + 2u) * 20u + lid.x + 2u] / FIXED;
    if (own.w > 1e-3) {
        v = mix(v, own.xyz / own.w, clamp(own.w, 0.0, 1.0));
    }
    if (p.x2 == 2u) {
        // The height (at c_off), at every HEIGHT_CELL-th texel.
        if (gid.x % HEIGHT_CELL == 0u && gid.y % HEIGHT_CELL == 0u) {
            let dr = r / HEIGHT_CELL;
            mem[p.c_off + u32(mem[p.x0 + slot]) * dr * dr + (gid.y / HEIGHT_CELL) * dr + gid.x / HEIGHT_CELL] = v.x;
        }
        return;
    }
    var d = texel(i);
    if (p.x2 == 0u) {
        d = vec4<f32>(clamp(v, vec3<f32>(-p.f0), vec3<f32>(p.f0)), d.a);
    } else {
        d.a = clamp(v.x, 0.0, 1.0);
    }
    atlas[i * 2u] = pack2x16float(d.rg);
    atlas[i * 2u + 1u] = pack2x16float(vec2<f32>(d.b, d.a));
}

// ---- The G-buffer of the level drawn ---------------------------------------------------------
// Each pixel's Surface (at a_off) and Motion (b_off) and its 1/z (c_off; the last frame's at
// c_rs), from the triangle it shows (drawn: its index + 1, 0 for none), the
// point of the level's own surface it shows (pushed out by the height, perhaps) and the view
// depth of where it's shown (point_at), and the view (mem[bias_off], VIEW floats: position,
// axes as rows, focal length, fog's ends, width; the last view's after it), as the caller's
// CPU G-buffer would for a flat level (game/src/dream_atlas.rs `surfaces`). Its face is looked
// up in the face table (level[x2], x3 + 1 entries, keys counted from the cell (a_rs, a_cs,
// b_rs)) for its page (level[x4 + face]), and marked seen in its band of rows (n rows a band,
// m u32s a band). Where a creature (its depth at mem[x6], unless NONE; f32 max for none) is
// nearer, no surface. With MOTION_FROM_LAST in flags, where the point was in the last view, if
// it was in sight there: on its screen and as near as the nearest thing it saw there.
const VIEW: u32 = 28u;
// Floats of each pixel's point of the level and which side of it is seen (lib.rs LIT), and of
// a light (lib.rs LIGHT).
const LIT: u32 = 8u;

// mem's first floats (lib.rs GLOBALS): where each pixel's point and light are (LIT floats a
// pixel of the frame given: the point, its side, the light on it), and how much the dream is lit
// by it (0: not at all, its colour its own; 1: wholly, as the wall's own colour).
fn light_on(x: u32, y: u32) -> vec3<f32> {
    let at = bitcast<u32>(mem[0]);
    let k = mem[1];
    if (at == 0u || k <= 0.0) { return vec3<f32>(1.0); }
    let o = at + (y * p.x0 + x) * LIT + 4u;
    return mix(vec3<f32>(1.0), vec3<f32>(mem[o], mem[o + 1u], mem[o + 2u]), k);
}
const LIGHT: u32 = 8u;

// Where the lights and each pixel's point are (after this view and the last).
fn lights_block() -> u32 { return bitcast<u32>(mem[p.bias_off + 2u * VIEW]); }
fn lit_block() -> u32 { return bitcast<u32>(mem[p.bias_off + 2u * VIEW + 1u]); }

// Where the torch is held: along the view's axes from the eye.
fn torch_held(v: View) -> vec3<f32> {
    let o = p.bias_off + 23u;
    return v.pos + v.r0 * mem[o] + v.r1 * mem[o + 1u] + v.r2 * mem[o + 2u];
}
// gbuffer: the picture under the dream is the renderer's own (own_picture): each pixel's own.
const OWN_PICTURE: u32 = 2u;
const MOTION_FROM_LAST: u32 = 1u;
const NEAR: f32 = 0.02;

struct View {
    pos: vec3<f32>,
    r0: vec3<f32>,
    r1: vec3<f32>,
    r2: vec3<f32>,
    focal: f32,
    fog: vec2<f32>,
    width: f32,
}

fn view_at(o: u32) -> View {
    var v: View;
    v.pos = vec3<f32>(mem[o], mem[o + 1u], mem[o + 2u]);
    v.r0 = vec3<f32>(mem[o + 3u], mem[o + 4u], mem[o + 5u]);
    v.r1 = vec3<f32>(mem[o + 6u], mem[o + 7u], mem[o + 8u]);
    v.r2 = vec3<f32>(mem[o + 9u], mem[o + 10u], mem[o + 11u]);
    v.focal = mem[o + 12u];
    v.fog = vec2<f32>(mem[o + 13u], mem[o + 14u]);
    v.width = mem[o + 15u];
    return v;
}

fn ray(v: View, x: f32, y: f32) -> vec3<f32> {
    let d = vec3<f32>((x - v.width / 2.0) / v.focal, (y - f32(p.x1) / 2.0) / v.focal, 1.0);
    return v.r0 * d.x + v.r1 * d.y + v.r2 * d.z;
}

// Where q is on v's screen, and how far ahead.
fn project(v: View, q: vec3<f32>) -> vec3<f32> {
    let d = q - v.pos;
    let z = max(dot(v.r2, d), 1e-6);
    return vec3<f32>(v.width / 2.0 + v.focal * dot(v.r0, d) / z, f32(p.x1) / 2.0 + v.focal * dot(v.r1, d) / z, z);
}

fn face_hash(k0: u32, k1: u32) -> u32 {
    var h = (k0 * 0x9e3779b1u) ^ (k1 * 0x85ebca77u);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    return h ^ (h >> 12u);
}

// The face of orientation `orient` in cell `c`, or NONE.
fn face_of(orient: u32, c: vec3<i32>) -> u32 {
    let r = c - vec3<i32>(i32(p.a_rs), i32(p.a_cs), i32(p.b_rs));
    if (any(r < vec3<i32>(0)) || r.x >= (1 << 29u) || r.y >= 65536 || r.z >= 65536) { return NONE; }
    let k0 = orient | (u32(r.x) << 3u);
    let k1 = (u32(r.y) << 16u) | u32(r.z);
    var h = face_hash(k0, k1) & p.x3;
    for (var i = 0u; i <= p.x3; i++) {
        let e = p.x2 + h * 3u;
        if (level[e] == NONE) { return NONE; }
        if (level[e] == k0 && level[e + 1u] == k1) { return level[e + 2u]; }
        h = (h + 1u) & p.x3;
    }
    return NONE;
}

fn pick(v: vec3<f32>, axis: u32) -> vec2<f32> {
    if (axis == 0u) { return v.yz; }
    if (axis == 1u) { return v.xz; }
    return v.xy;
}

// The flat point and view depth at pixel (x, y) (k + y * row + 4 x), and whether it's on the
// plane (normal n, through q) the pixel's own is on.
fn point_at(x: u32, y: u32) -> vec4<f32> {
    let o = p.k + y * (((16u * p.x0 + 255u) / 256u) * 64u) + x * 4u;
    return vec4<f32>(mem[o], mem[o + 1u], mem[o + 2u], mem[o + 3u]);
}

// What the level drawn left at pixel (x, y) (img[x5], four u32s a pixel, rows padded as
// point_at's): its triangle (index + 1, 0 for none), the texture times the level's light and
// times the torch's albedo (each RGBA8, halved).
fn drawn(x: u32, y: u32) -> vec4<u32> {
    let o = p.x5 + y * (((16u * p.x0 + 255u) / 256u) * 64u) + x * 4u;
    return vec4<u32>(img[o], img[o + 1u], img[o + 2u], img[o + 3u]);
}

fn on_plane(x: u32, y: u32, n: vec3<f32>, q: vec3<f32>) -> bool {
    if (x >= p.x0 || y >= p.x1 || drawn(x, y).x == 0u) { return false; }
    let a = point_at(x, y).xyz;
    return abs(dot(a - q, n)) < 1e-3;
}

@compute @workgroup_size(16, 16)
fn gbuffer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = p.x0;
    if (gid.x >= w || gid.y >= p.x1) { return; }
    let i = gid.y * w + gid.x;
    let s = p.a_off + i * SURFACE;
    let m = p.b_off + i * MOTION;
    for (var k = 0u; k < SURFACE; k++) { mem[s + k] = 0.0; }
    mem[s] = -1.0;
    mem[m] = -1e9;
    mem[m + 1u] = -1e9;
    for (var k = 2u; k < MOTION; k++) { mem[m + k] = 0.0; }
    let lit = lit_block() + i * LIT;
    mem[lit + 3u] = 0.0;
    mem[lit + 4u] = 1.0;
    mem[lit + 5u] = 1.0;
    mem[lit + 6u] = 1.0;
    var nearest = 0.0;
    let tri = drawn(gid.x, gid.y).x;
    let v = view_at(p.bias_off);
    var shown = tri != 0u;
    let here = point_at(gid.x, gid.y);
    // The point of the level's own surface the pixel shows (pushed out from), and where it
    // shows it.
    let pf = here.xyz;
    let x = f32(gid.x) + 0.5;
    let y = f32(gid.y) + 0.5;
    let p0 = v.pos + ray(v, x, y) * here.w;
    var n = vec3<f32>(0.0);
    if (shown) {
        let t = (tri - 1u) * 4u;
        n = vec3<f32>(bitcast<f32>(level[t]), bitcast<f32>(level[t + 1u]), bitcast<f32>(level[t + 2u]));
        nearest = 1.0 / max(here.w, 1e-6);
    }
    if (p.x6 != NONE) {
        let z = mem[p.x6 + i];
        if (z < 3e38 && 1.0 / max(z, 1e-6) > nearest) {
            nearest = 1.0 / max(z, 1e-6);
            shown = false;
        }
    }
    mem[p.c_off + i] = nearest;
    if (!shown) { return; }
    let an = abs(n);
    var axis = 0u;
    if (an.y > an.x) { axis = 1u; }
    if (an.z > max(an.x, an.y)) { axis = 2u; }
    // How the flat point moves a pixel right and down: by the next pixel's on the same plane
    // (or the last's), else as the flat plane's rays would.
    let to = dot(pf - v.pos, n);
    var dx = vec3<f32>(0.0);
    var dy = vec3<f32>(0.0);
    var qx = vec3<f32>(0.0);
    var qy = vec3<f32>(0.0);
    if (on_plane(gid.x + 1u, gid.y, n, pf)) {
        dx = point_at(gid.x + 1u, gid.y).xyz - pf;
        qx = v.pos + ray(v, x + 1.0, y) * point_at(gid.x + 1u, gid.y).w;
    } else if (gid.x > 0u && on_plane(gid.x - 1u, gid.y, n, pf)) {
        dx = pf - point_at(gid.x - 1u, gid.y).xyz;
        qx = p0 + (p0 - (v.pos + ray(v, x - 1.0, y) * point_at(gid.x - 1u, gid.y).w));
    } else {
        let d = ray(v, x + 1.0, y);
        let dn = dot(d, n);
        dx = v.pos + d * (to / select(dn, select(-1e-6, 1e-6, dn >= 0.0), abs(dn) < 1e-6)) - pf;
        qx = p0 + dx;
    }
    if (on_plane(gid.x, gid.y + 1u, n, pf)) {
        dy = point_at(gid.x, gid.y + 1u).xyz - pf;
        qy = v.pos + ray(v, x, y + 1.0) * point_at(gid.x, gid.y + 1u).w;
    } else if (gid.y > 0u && on_plane(gid.x, gid.y - 1u, n, pf)) {
        dy = pf - point_at(gid.x, gid.y - 1u).xyz;
        qy = p0 + (p0 - (v.pos + ray(v, x, y - 1.0) * point_at(gid.x, gid.y - 1u).w));
    } else {
        let d = ray(v, x, y + 1.0);
        let dn = dot(d, n);
        dy = v.pos + d * (to / select(dn, select(-1e-6, 1e-6, dn >= 0.0), abs(dn) < 1e-6)) - pf;
        qy = p0 + dy;
    }
    let cell = vec3<i32>(floor(pf - n * 0.01));
    let orient = axis * 2u + select(0u, 1u, n[axis] > 0.0);
    let face = face_of(orient, cell);
    var page = -1.0;
    if (face != NONE) {
        page = bitcast<f32>(level[p.x4 + face]);
        let band = min(gid.y / p.n, 15u);
        atomicOr(&faces_seen[band * p.m + face / 32u], 1u << (face % 32u));
    }
    let uv = pick(pf - floor(pf), axis);
    let dist = length(p0 - v.pos);
    let fog = clamp((dist - v.fog.x) / max(v.fog.y - v.fog.x, 1e-6), 0.0, 1.0);
    let eye = v.pos - p0;
    let rise = max(abs(dot(eye, n)), 1e-4);
    let slope = pick(eye, axis) / rise;
    let duv_dx = pick(dx, axis);
    let duv_dy = pick(dy, axis);
    var pic = project(v, pf);
    if ((p.flags & OWN_PICTURE) != 0u) { pic = vec3<f32>(x, y, 0.0); }
    mem[s] = page;
    mem[s + 1u] = clamp(uv.x, 0.0, 0.9999);
    mem[s + 2u] = clamp(uv.y, 0.0, 0.9999);
    mem[s + 3u] = duv_dx.x;
    mem[s + 4u] = duv_dx.y;
    mem[s + 5u] = duv_dy.x;
    mem[s + 6u] = duv_dy.y;
    mem[s + 7u] = 1.0 - fog;
    mem[s + 8u] = slope.x;
    mem[s + 9u] = slope.y;
    mem[s + 10u] = pic.x;
    mem[s + 11u] = pic.y;
    mem[s + 12u] = dist;
    // The point, and the side seen: its axis + 1, negative where the side seen faces down it.
    let seen_n = n * sign(dot(eye, n));
    mem[lit] = p0.x;
    mem[lit + 1u] = p0.y;
    mem[lit + 2u] = p0.z;
    mem[lit + 3u] = f32(axis + 1u) * select(-1.0, 1.0, seen_n[axis] >= 0.0);
    if ((p.flags & MOTION_FROM_LAST) != 0u) {
        let last = view_at(p.bias_off + VIEW);
        let was = project(last, p0);
        let xi = floor(was.x);
        let yi = floor(was.y);
        if (was.z > NEAR && xi >= 0.0 && yi >= 0.0 && xi < f32(w) && yi < f32(p.x1)) {
            let near = mem[p.c_rs + u32(yi) * w + u32(xi)];
            if (abs(near - 1.0 / was.z) <= 0.02 / was.z) {
                let a = project(last, qx);
                let b = project(last, qy);
                mem[m] = was.x;
                mem[m + 1u] = was.y;
                mem[m + 2u] = a.x - was.x;
                mem[m + 3u] = a.y - was.y;
                mem[m + 4u] = b.x - was.x;
                mem[m + 5u] = b.y - was.y;
            }
        }
    }
}

// Each vertex of the subdivided level (x0 of them; level[x1], VERT u32s each: its place,
// normal, faces' list) pushed out along its normal by the mean of the dreamed height (c_off,
// pages k texels a side) of each face it's on, at its place on that face (none for a face
// with no page), times f0: into level[x2] (MOVED floats a vertex: there, then where it was).
// Every face that meets at a point gives the same mean there, so the faces' edges meet.
const VERT: u32 = 8u;
const LISTED: u32 = 5u;

@compute @workgroup_size(64)
fn displace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if (i >= p.x0) { return; }
    let r = p.x1 + i * VERT;
    let pos = vec3<f32>(bitcast<f32>(level[r]), bitcast<f32>(level[r + 1u]), bitcast<f32>(level[r + 2u]));
    let nrm = vec3<f32>(bitcast<f32>(level[r + 3u]), bitcast<f32>(level[r + 4u]), bitcast<f32>(level[r + 5u]));
    let list = level[r + 6u];
    let count = level[r + 7u];
    var sum = 0.0;
    for (var j = 0u; j < count; j++) {
        let e = list + j * LISTED;
        let page = bitcast<f32>(level[p.x4 + level[e]]);
        if (page < 0.0) { continue; }
        let cell = vec3<f32>(f32(bitcast<i32>(level[e + 2u])), f32(bitcast<i32>(level[e + 3u])), f32(bitcast<i32>(level[e + 4u])));
        let uv = clamp(pick(pos - cell, level[e + 1u] / 2u), vec2<f32>(0.0), vec2<f32>(1.0));
        sum += page_height(p.c_off, u32(page), uv.x * f32(p.k) - 0.5, uv.y * f32(p.k) - 0.5);
    }
    let h = sum / f32(max(count, 1u)) * p.f0;
    let o = p.x2 + i * 8u;
    let q = pos + nrm * h;
    level[o] = bitcast<u32>(q.x);
    level[o + 1u] = bitcast<u32>(q.y);
    level[o + 2u] = bitcast<u32>(q.z);
}

// The renderer's own picture of the level, in place of the frame given's (rows b_cs apart)
// where it shows the level and nothing's in front of it (as gbuffer): the texture times the
// level's light (drawn) and the torch at the eye (its colour and reach in the view at
// bias_off) on the texture times its albedo, falling off
// and turned away as the torch's light does from where the pixel's point now is (point_at,
// k), then fogged (its colour in the view too). x0 by x1 pixels, the triangles at img[x5], the
// creatures' depth at mem[x6] (or NONE).
@compute @workgroup_size(16, 16)
fn own_picture(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.x0 || gid.y >= p.x1) { return; }
    let px = gid.y * p.b_cs + gid.x;
    let d = drawn(gid.x, gid.y);
    let tri = d.x;
    if (tri == 0u) { return; }
    let here = point_at(gid.x, gid.y);
    if (p.x6 != NONE) {
        let z = mem[p.x6 + gid.y * p.x0 + gid.x];
        if (z < 3e38 && z < here.w) { return; }
    }
    let v = view_at(p.bias_off);
    let o = p.bias_off + 16u;
    let torch = vec3<f32>(mem[o], mem[o + 1u], mem[o + 2u]);
    let reach = mem[o + 3u];
    let fog_colour = vec3<f32>(mem[o + 4u], mem[o + 5u], mem[o + 6u]);
    let at = v.pos + ray(v, f32(gid.x) + 0.5, f32(gid.y) + 0.5) * here.w;
    let t = (tri - 1u) * 4u;
    let n = vec3<f32>(bitcast<f32>(level[t]), bitcast<f32>(level[t + 1u]), bitcast<f32>(level[t + 2u]));
    let to_eye = v.pos - at;
    // The torch where it's held.
    let to_torch = torch_held(v) - at;
    let dist = max(length(to_torch), 1.0 / 256.0);
    let fall = max(1.0 - dist / max(reach, 1e-3), 0.0);
    let lambert = max(dot(n, to_torch) / dist, 0.0);
    // The light on the wall: the level's, and the torch's by its albedo; then the texture so lit,
    // and the light kept for the dream (seen's light_on).
    let light = unpack4x8unorm(d.y).rgb * 2.0 + unpack4x8unorm(d.z).rgb * 2.0 * torch * fall * fall * lambert;
    let texel = unpack4x8unorm(d.w).rgb;
    var c = min(texel * light, vec3<f32>(1.0));
    let lo = lit_block() + (gid.y * p.x0 + gid.x) * LIT + 4u;
    mem[lo] = light.r;
    mem[lo + 1u] = light.g;
    mem[lo + 2u] = light.b;
    let fog = clamp((length(to_eye) - v.fog.x) / max(v.fog.y - v.fog.x, 1e-6), 0.0, 1.0);
    c = mix(c, fog_colour, fog);
    img[px] = pack4x8unorm(vec4<f32>(c, 1.0));
}
