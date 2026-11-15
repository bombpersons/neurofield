// The pages' heights smoothed (shell.rs Shell::mip_pages, smooth_faces), appended to
// shell.wgsl: each page averaged down, two by two cells, to two cells a side (its mips), then
// each face's page rebuilt at full size from a level of them by a cubic B-spline, across its
// tile's edges from the neighbouring tiles' pages in its plane (so no seam where they meet):
// `near` at the smoothing asked (u.pos.w levels, fractions mixing two), `far` two levels
// smoother (when u.r0.w, the smoothing with distance, is on).

@group(1) @binding(0) var<storage, read_write> mips: array<f32>;
@group(1) @binding(1) var<storage, read_write> near: array<f32>;
@group(1) @binding(2) var<storage, read_write> far: array<f32>;
// The level's faces: orientation and cell.
@group(1) @binding(3) var<storage, read> face_list: array<vec4<i32>>;
// The largest |height| smoothed, its f32 bits (non-negative floats order as their bits).
@group(1) @binding(4) var<storage, read_write> bound_out: array<atomic<u32>>;
// The smoothed heights again, for the march to read four at a time (shell.rs Shell::heights).
@group(1) @binding(5) var heights_out: texture_storage_2d<r32float, write>;
// shell.wgsl's height functions read it; unused here.
@group(1) @binding(8) var<storage, read> toward: array<f32>;

fn cells_a_side() -> u32 {
    return u32(u.o.w / HEIGHT_CELL);
}

// Floats of a page's mips (shell.rs mip_size), where level l (1 up) starts in them, and the
// coarsest level.
fn mip_size(r: u32) -> u32 {
    return mip_at(r, top_level(r) + 1u);
}

fn mip_at(r: u32, l: u32) -> u32 {
    var s = 0u;
    for (var m = 1u; m < l; m++) {
        let n = r >> m;
        s += n * n;
    }
    return s;
}

fn top_level(r: u32) -> u32 {
    var l = 0u;
    while ((r >> (l + 1u)) >= 2u) { l++; }
    return l;
}

@compute @workgroup_size(256)
fn mip_pages(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let page = wg.x + wg.y * 65535u;
    let r = cells_a_side();
    let size = mip_size(r);
    if (page >= arrayLength(&mips) / size) { return; }
    let base = page * size;
    var n = r >> 1u;
    for (var c = li; c < n * n; c += 256u) {
        let x = c % n;
        let y = c / n;
        let o = u.t.w + page * r * r + 2u * y * r + 2u * x;
        mips[base + c] = 0.25 * (mem[o] + mem[o + 1u] + mem[o + r] + mem[o + r + 1u]);
    }
    var off = 0u;
    while (n >= 4u) {
        storageBarrier();
        let m = n >> 1u;
        for (var c = li; c < m * m; c += 256u) {
            let x = c % m;
            let y = c / m;
            let o = base + off + 2u * y * n + 2u * x;
            mips[base + off + n * n + c] = 0.25 * (mips[o] + mips[o + 1u] + mips[o + n] + mips[o + n + 1u]);
        }
        off += n * n;
        n = m;
    }
}

// The pages of the tile's 3 x 3 neighbourhood in its plane (-1 for none), the tile itself in
// the middle; and the level being read, with the two cells round it, from them.
var<workgroup> around: array<i32, 9>;
const PATCH: u32 = 4096u;
var<workgroup> tile_cells: array<f32, 4096>;

fn bspline(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let s = 1.0 - t;
    return vec4<f32>(s * s * s, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
}

// Cell (qx, qy) of the level at `off` in each page's mips (n cells a side) in the tile's grid,
// from the neighbouring tile's page past its edges, or the tile's own edge where there's none.
fn tap(off: u32, size: u32, n: i32, qx: i32, qy: i32, own: i32) -> f32 {
    let dx = select(select(0, 1, qx >= n), -1, qx < 0);
    let dy = select(select(0, 1, qy >= n), -1, qy < 0);
    var page = around[(dy + 1) * 3 + dx + 1];
    var x = qx - dx * n;
    var y = qy - dy * n;
    if (page < 0) {
        page = own;
        x = clamp(qx, 0, n - 1);
        y = clamp(qy, 0, n - 1);
    }
    return mips[u32(page) * size + off + u32(y) * u32(n) + u32(x)];
}

// Weight `w` of level l added to each cell of the tile's page in near (far: in far), or put
// there (first): the cell itself at level 0, else the cubic B-spline of that level's cells
// round it (cell q's middle at ((q + 1/2) 2^l - 1/2) cells), read from the patch when it fits.
fn add_level(l: u32, w: f32, to_far: bool, first: bool, own: i32, li: u32) {
    let r = cells_a_side();
    let size = mip_size(r);
    let off = mip_at(r, l);
    let n = i32(r >> l);
    let side = u32(n) + 4u;
    let fits = l > 0u && side * side <= PATCH;
    if (fits) {
        workgroupBarrier();
        for (var c = li; c < side * side; c += 256u) {
            tile_cells[c] = tap(off, size, n, i32(c % side) - 2, i32(c / side) - 2, own);
        }
        workgroupBarrier();
    }
    let s = f32(1u << l);
    for (var c = li; c < r * r; c += 256u) {
        let i = i32(c % r);
        let j = i32(c / r);
        var v = 0.0;
        if (l == 0u) {
            v = mem[u.t.w + u32(own) * r * r + c];
        } else {
            let g = (vec2<f32>(f32(i), f32(j)) + 0.5) / s - 0.5;
            let g0 = vec2<i32>(floor(g));
            let wx = bspline(g.x - floor(g.x));
            let wy = bspline(g.y - floor(g.y));
            for (var b = 0; b < 4; b++) {
                var row = 0.0;
                for (var a = 0; a < 4; a++) {
                    let qx = g0.x - 1 + a;
                    let qy = g0.y - 1 + b;
                    if (fits) {
                        row += wx[a] * tile_cells[u32(qy + 2) * side + u32(qx + 2)];
                    } else {
                        row += wx[a] * tap(off, size, n, qx, qy, own);
                    }
                }
                v += wy[b] * row;
            }
        }
        let o = u32(own) * r * r + c;
        if (to_far) {
            far[o] = select(far[o], 0.0, first) + w * v;
        } else {
            near[o] = select(near[o], 0.0, first) + w * v;
        }
    }
}

// Level l (fractions mixing two) added as near (or far).
fn smooth_level(l: f32, to_far: bool, own: i32, li: u32) {
    let lc = clamp(l, 0.0, f32(top_level(cells_a_side())));
    let fr = lc - floor(lc);
    add_level(u32(floor(lc)), 1.0 - fr, to_far, true, own, li);
    if (fr > 0.0) { add_level(u32(floor(lc)) + 1u, fr, to_far, false, own, li); }
}

@compute @workgroup_size(256)
fn smooth_faces(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let fi = wg.x + wg.y * 65535u;
    if (fi >= arrayLength(&face_list)) { return; }
    let f = face_list[fi];
    let fo = u32(f.x);
    // A face with no page has nothing to smooth.
    let own_face = face_of(fo, f.yzw);
    if (own_face == NONE || bitcast<f32>(level[u.t.z + own_face]) < 0.0) { return; }
    if (li < 9u) {
        let dx = i32(li % 3u) - 1;
        let dy = i32(li / 3u) - 1;
        let ax = axes_of(fo / 2u);
        let face = face_of(fo, vec3<i32>(f.yzw) + vec3<i32>(ax[0]) * dx + vec3<i32>(ax[1]) * dy);
        var pg = -1;
        if (face != NONE) {
            let p = bitcast<f32>(level[u.t.z + face]);
            if (p >= 0.0) { pg = i32(p); }
        }
        around[li] = pg;
    }
    let own = workgroupUniformLoad(&around[4]);
    smooth_level(u.pos.w, false, own, li);
    let far_on = u.r0.w > 0.0;
    if (far_on) { smooth_level(u.pos.w + 2.0, true, own, li); }
    // The far-off smoothing mixed into near by each cell's view depth (none at the eye, all
    // of it 4 / u.r0.w tiles off; the cell's middle on the plane, which for a sloping face is
    // taken as the face's cell's level along its axis); and the largest |height| (each
    // invocation reads the cells it wrote).
    let r = cells_a_side();
    let axis = fo / 2u;
    let ax = axes_of(axis);
    var origin = vec3<f32>(f.yzw);
    origin[axis] += f32(fo % 2u);
    var most = 0.0;
    for (var c = li; c < r * r; c += 256u) {
        let o = u32(own) * r * r + c;
        var v = near[o];
        if (far_on) {
            let x = origin + ax[0] * ((f32(c % r) + 0.25) / f32(r)) + ax[1] * ((f32(c / r) + 0.25) / f32(r));
            let k = clamp(u.r0.w * dot(u.r2.xyz, x - u.pos.xyz) / 4.0, 0.0, 1.0);
            v = mix(v, far[o], k);
            near[o] = v;
        }
        most = max(most, abs(v));
        textureStore(heights_out, vec2<u32>(u32(own) % HEIGHTS_ROW, u32(own) / HEIGHTS_ROW) * r + vec2<u32>(c % r, c / r), vec4<f32>(v, 0.0, 0.0, 0.0));
    }
    atomicMax(&bound_out[0], bitcast<u32>(most));
}

fn four_cells(at: u32, pg: u32, local: vec2<u32>) -> vec4<f32> {
    return cells_from(at, cells_a_side());
}

// shell.wgsl's fs reads it; unused here.
@group(1) @binding(9) var inner: texture_depth_2d;
@group(1) @binding(10) var<storage, read> homes: array<vec4<i32>>;
