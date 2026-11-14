// Kernels for the dream renderer (lib.rs): GoogLeNet forward and backward to its input, and the
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
// Fixed point of `acc`.
const FIXED: f32 = 4096.0;
// Floats a pixel of the surfaces given (lib.rs Surface) and of a net pixel's lookup.
const SURFACE: u32 = 10u;
const LOOKUP: u32 = 6u;

// The flat index of a 2D dispatch of rows (y, spilling into z).
fn row_of(gid: vec3<u32>) -> u32 {
    return gid.y + gid.z * 65535u;
}

// ---- Matrix multiply ----------------------------------------------------------------------
// C[i][j] (=, or += with ACCUMULATE) act(sum_t A[i][t] * B[t][j] + bias[j]) for i < m, j < n,
// t < k. A[i][t] = mem[a_off + i*a_rs + t*a_cs], B likewise, C[i][j] = mem[c_off + i*c_rs + j];
// bias_off = NONE for none. A 32x32 tile of C per workgroup of 8x8 threads, 4x4 a thread, k in
// steps of 16 with the next prefetched (the engine's gpu_net.wgsl `matmul`, with ReLU).

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
    let kl = p.k;
    let row0 = (wg.y + wg.z * 65535u) * 32u;
    let col0 = wg.x * 32u;
    let tid = lid.y * 8u + lid.x;
    let q = tid % 8u;
    let tk = tid / 8u;
    var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
    var na0 = load_a(kl, row0 + q * 4u, tk); var na1 = load_a(kl, row0 + q * 4u, tk + 8u);
    var nb0 = load_b(kl, tk, col0 + q * 4u); var nb1 = load_b(kl, tk + 8u, col0 + q * 4u);
    for (var t0 = 0u; t0 < kl; t0 += 16u) {
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
// at b_off that read it.
@compute @workgroup_size(64)
fn col2im(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let ipx = row_of(gid);
    if (c >= p.x0 || ipx >= p.x1 * p.x2) { return; }
    let kk = p.x3 * p.x3;
    let big_k = p.x0 * kk;
    let iy = i32(ipx / p.x2) + i32(p.x5);
    let ix = i32(ipx % p.x2) + i32(p.x5);
    let s = i32(p.x4);
    var sum = 0.0;
    for (var ky = 0; ky < i32(p.x3); ky++) {
        let ty = iy - ky;
        if (ty < 0 || ty % s != 0) { continue; }
        let oy = ty / s;
        if (oy >= i32(p.m)) { continue; }
        for (var kx = 0; kx < i32(p.x3); kx++) {
            let tx = ix - kx;
            if (tx < 0 || tx % s != 0) { continue; }
            let ox = tx / s;
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

// mem[c_off + i] = 0 for i < m (2D: x over 64s, rows of 64 * 64).
@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = row_of(gid) * 4096u + gid.x;
    if (i < p.m) { mem[p.c_off + i] = 0.0; }
}

// Two sums over m items, a workgroup's share each, into partials at c_off (2 a workgroup):
// x0 = 0: of a_off's squares (and how many); x0 = 1: of |grad| at a_off (3 a pixel) where the
// lookup at b_off covers the pixel (and 3 for each such pixel).
var<workgroup> red: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    var s = vec2<f32>(0.0);
    let stride = nwg.x * 256u;
    for (var i = wg.x * 256u + lid.x; i < p.m; i += stride) {
        if (p.x0 == 0u) {
            let v = mem[p.a_off + i];
            s += vec2<f32>(v * v, 1.0);
        } else if (mem[p.b_off + i * LOOKUP] >= 0.0) {
            let g = p.a_off + i * 3u;
            s += vec2<f32>(abs(mem[g]) + abs(mem[g + 1u]) + abs(mem[g + 2u]), 3.0);
        }
    }
    red[lid.x] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid.x < k) { red[lid.x] += red[lid.x + k]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        mem[p.c_off + wg.x * 2u] = red[0].x;
        mem[p.c_off + wg.x * 2u + 1u] = red[0].y;
    }
}

// The partials' totals (m workgroups' worth at a_off) to c_off, c_off + 1. One workgroup.
@compute @workgroup_size(256)
fn reduce_finish(@builtin(local_invocation_id) lid: vec3<u32>) {
    var s = vec2<f32>(0.0);
    for (var i = lid.x; i < p.m; i += 256u) {
        s += vec2<f32>(mem[p.a_off + i * 2u], mem[p.a_off + i * 2u + 1u]);
    }
    red[lid.x] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid.x < k) { red[lid.x] += red[lid.x + k]; }
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
    for (var i = 1u; i <= RELIEF_STEPS; i++) {
        let z = depth * (1.0 - 2.0 * f32(i) / f32(RELIEF_STEPS));
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
    if (out.page < 0.0) { return out; }
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

// What shows at a point of the frame given (in its pixels, fractional) that lies in base pixel
// (bx, by), `off` from its middle: the picture, or as much of its blur as the dream owns there,
// with the dream's colour added. With SCREEN, the screen layer too (at shown pixel `at`, or
// over the whole base pixel with `block`), owning as much more of the wall as it does, up to f3.
fn seen(bx: u32, by: u32, off: vec2<f32>, at: vec2<u32>, block: bool) -> Seen {
    var out = look(bx, by, off);
    out.colour = picture(bx, by);
    if (out.page < 0.0) { return out; }
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
    if (own > 0.0) {
        out.colour = mix(out.colour, smooth_at(f32(bx) + 0.5 + off.x, f32(by) + 0.5 + off.y), own);
    }
    out.colour += d.rgb * out.strength;
    out.colour += layer.rgb;
    return out;
}

// The network's input for one octave: n by m pixels at c_off (HWC, 2 rgb - 1), taken from the
// frame given scaled up x2 times, from (x3, x4) of that, rolled by (x5, x6); and each pixel's
// lookup at b_off (LOOKUP floats: page, texel x, y, strength, which channels aren't clamped).
@compute @workgroup_size(16, 16)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n || gid.y >= p.m) { return; }
    let sx = (gid.x + p.n - p.x5 % p.n) % p.n + p.x3;
    let sy = (gid.y + p.m - p.x6 % p.m) % p.m + p.x4;
    let up = p.x2;
    let bx = min(sx / up, p.x0 - 1u);
    let by = min(sy / up, p.x1 - 1u);
    let off = (vec2<f32>(f32(sx % up), f32(sy % up)) + 0.5) / f32(up) - 0.5;
    let s = seen(bx, by, off, vec2<u32>(sx, sy), up < p.a_cs);
    let c = clamp(s.colour, vec3<f32>(0.0), vec3<f32>(1.0));
    let o = p.c_off + (gid.y * p.n + gid.x) * 3u;
    mem[o] = 2.0 * c.r - 1.0;
    mem[o + 1u] = 2.0 * c.g - 1.0;
    mem[o + 2u] = 2.0 * c.b - 1.0;
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
    let c = clamp(seen(bx, by, off, gid.xy, false).colour, vec3<f32>(0.0), vec3<f32>(1.0));
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

// One step of an octave (its sizes, crop, roll and scale x2 as prepare's; lookup at b_off) onto
// the screen layer (c_rs; a_rs by b_rs pixels, a_cs times the frame given's): each shown pixel
// the octave covers takes the gradient where it falls, over the mean |gradient| (x0), times
// 2 f0 times the strength: its brightness, and f1 of its colour spread over the binomial's taps
// k of the frame given's pixels apart (k x2 of the octave's). Held within +-f2; the wall owned
// f3 more.
@compute @workgroup_size(16, 16)
fn screen_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let scale = f32(p.x2) / f32(p.a_cs);
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
    let k = 2.0 * mem[l + 3u] * p.f0 / max(mean, 1e-12);
    let g = grad_at(u);
    var step = vec3<f32>((g.r + g.g + g.b) / 3.0);
    if (p.f1 > 0.0) {
        let bin = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
        var wide = vec3<f32>(0.0);
        for (var dy = 0; dy < 5; dy++) {
            for (var dx = 0; dx < 5; dx++) {
                wide += grad_at(u + vec2<f32>(f32(dx - 2), f32(dy - 2)) * f32(p.k * p.x2)) * (bin[dy] * bin[dx] / 256.0);
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
}

// The screen layer (c_rs, x0 * a_cs by x1 * a_cs pixels; its rows from x5 up to m) into the
// accumulators of the texels each shown pixel falls on, by the surfaces it was grown over (as
// show looks them up), for apply to add to the pages or store to put in their place. A page's
// accumulator slot is at mem[b_off + page] (-1: none).
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
// look).
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
    if (mem[m] <= -1e8 && (p.flags & FILL) != 0u) {
        let s = look(bx, by, off);
        if (s.page >= 0.0) {
            let d = dream_at(u32(s.page), s.tx, s.ty);
            v = vec4<f32>(d.rgb * s.strength, d.a);
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
            for (var j = 0; j < 4; j++) {
                let y = u32(clamp(i0.y + j - 1, 0, i32(sh) - 1));
                for (var i = 0; i < 4; i++) {
                    let x = u32(clamp(i0.x + i - 1, 0, i32(sw) - 1));
                    let s = screen_px(p.b_rs, x, y, sw);
                    v += s * (wx[i] * wy[j]);
                    if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
                        lo = min(lo, s);
                        hi = max(hi, s);
                    }
                }
            }
            v = clamp(v, lo, hi);
        }
    }
    mem[o] = v.r;
    mem[o + 1u] = v.g;
    mem[o + 2u] = v.b;
    mem[o + 3u] = v.a;
}

// Each touched slot's page (mem[x0 + slot]) takes the screen layer's dream baked into its
// accumulators in place of its own, where the screen covers it (spread as in apply, for texels
// seen from afar): its colour (x2 = 0, held within +-f0), or how much of the wall it owns (x2 = 1).
@compute @workgroup_size(16, 16)
fn store(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = p.k;
    if (gid.x >= r || gid.y >= r) { return; }
    let slot = gid.z;
    let i = u32(mem[p.x0 + slot]) * r * r + gid.y * r + gid.x;
    let near = spread(slot, gid.xy, 1);
    if (near.w <= 1e-3) { return; }
    var v = near.xyz / near.w;
    let a = ((slot * r + gid.y) * r + gid.x) * 4u;
    let own = vec4<f32>(f32(atomicLoad(&acc[a])), f32(atomicLoad(&acc[a + 1u])), f32(atomicLoad(&acc[a + 2u])), f32(atomicLoad(&acc[a + 3u]))) / FIXED;
    if (own.w > 1e-3) {
        v = mix(v, own.xyz / own.w, clamp(own.w, 0.0, 1.0));
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
// c_rs), from the triangle it shows (img[x5 + y * b_cs + x], its index + 1, 0 for none) and the
// view (mem[bias_off], VIEW floats: position, axes as rows, focal length, fog's ends, width;
// the last view's after it), as the caller's CPU G-buffer would (game/src/dream_atlas.rs
// `surfaces`): where the ray through the pixel's middle meets the triangle's plane, and the
// rays a pixel right and down. Its face is looked up in the face table (level[x2], x3 + 1
// entries, keys counted from the cell (a_rs, a_cs, b_rs)) for its page (level[x4 + face]),
// and marked seen in its band of rows (n rows a band, m u32s a band). Where a creature (its
// depth at mem[x6], unless NONE; f32 max for none) is nearer, no surface. With MOTION_FROM_LAST
// in flags, where the point was in the last view, if it was in sight there: on its screen and
// as near as the nearest thing it saw there.
const VIEW: u32 = 16u;
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
    var nearest = 0.0;
    let tri = img[p.x5 + gid.y * p.b_cs + gid.x];
    let v = view_at(p.bias_off);
    var shown = tri != 0u;
    var p0 = vec3<f32>(0.0);
    var n = vec3<f32>(0.0);
    var to = 0.0;
    if (shown) {
        let t = (tri - 1u) * 4u;
        n = vec3<f32>(bitcast<f32>(level[t]), bitcast<f32>(level[t + 1u]), bitcast<f32>(level[t + 2u]));
        to = bitcast<f32>(level[t + 3u]) - dot(v.pos, n);
        let x = f32(gid.x) + 0.5;
        let y = f32(gid.y) + 0.5;
        let d = ray(v, x, y);
        let dn = dot(d, n);
        p0 = v.pos + d * (to / select(dn, select(-1e-6, 1e-6, dn >= 0.0), abs(dn) < 1e-6));
        nearest = 1.0 / max(dot(v.r2, p0 - v.pos), 1e-6);
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
    let x = f32(gid.x) + 0.5;
    let y = f32(gid.y) + 0.5;
    let dx = ray(v, x + 1.0, y);
    let dy = ray(v, x, y + 1.0);
    let dnx = dot(dx, n);
    let dny = dot(dy, n);
    let px = v.pos + dx * (to / select(dnx, select(-1e-6, 1e-6, dnx >= 0.0), abs(dnx) < 1e-6));
    let py = v.pos + dy * (to / select(dny, select(-1e-6, 1e-6, dny >= 0.0), abs(dny) < 1e-6));
    let cell = vec3<i32>(floor(p0 - n * 0.01));
    let orient = axis * 2u + select(0u, 1u, n[axis] > 0.0);
    let face = face_of(orient, cell);
    var page = -1.0;
    if (face != NONE) {
        page = bitcast<f32>(level[p.x4 + face]);
        let band = min(gid.y / p.n, 15u);
        atomicOr(&faces_seen[band * p.m + face / 32u], 1u << (face % 32u));
    }
    let uv = pick(p0 - floor(p0), axis);
    let dist = length(p0 - v.pos);
    let fog = clamp((dist - v.fog.x) / max(v.fog.y - v.fog.x, 1e-6), 0.0, 1.0);
    let eye = v.pos - p0;
    let rise = max(abs(dot(eye, n)), 1e-4);
    let slope = pick(eye, axis) / rise;
    let duv_dx = pick(px - p0, axis);
    let duv_dy = pick(py - p0, axis);
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
    if ((p.flags & MOTION_FROM_LAST) != 0u) {
        let last = view_at(p.bias_off + VIEW);
        let was = project(last, p0);
        let xi = floor(was.x);
        let yi = floor(was.y);
        if (was.z > NEAR && xi >= 0.0 && yi >= 0.0 && xi < f32(w) && yi < f32(p.x1)) {
            let near = mem[p.c_rs + u32(yi) * w + u32(xi)];
            if (abs(near - 1.0 / was.z) <= 0.02 / was.z) {
                let a = project(last, px);
                let b = project(last, py);
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
