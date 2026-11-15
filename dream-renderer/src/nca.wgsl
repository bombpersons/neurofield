// Neural cellular automata grown on the pages (nca.rs, docs/nca_walls.md): each page a square of
// R x R cells of 16 channels (colour 0..2, height 6, the rest the automaton's own, as MeshNCA's
// models keep them), each cell seeing its 8 neighbours, across a page's edges into the next
// tile's page of the same wall (its own edge where there is none). A step reads one state
// buffer and writes the other; `paint` lays the colour and height on the atlas's pages.

struct NcaParams {
    // Cells a page side, the atlas's texels a page side, pages in the list, and of them the
    // first `stepped` stepped (the rest copied as they are, so both buffers keep them).
    r: u32,
    page_res: u32,
    pages: u32,
    stepped: u32,
    seed: u32,
    // Where the pages' heights are in mem, and height cells a page side.
    height_at: u32,
    hr: u32,
    // The colour painted each cell's alone (the PS1's chunky texels), not bilinear.
    nearest: u32,
    // The chance a cell updates a step, the height's depth (tiles) and what's taken off it
    // (so walls sink as well as rise).
    fire: f32,
    depth: f32,
    lift: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> q: NcaParams;
// 16 channels a cell as 8 u32s of two halves, page by page, rows of R.
@group(0) @binding(1) var<storage, read> state_in: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> state_out: array<vec4<u32>>;
// Each model: its first layer, 128 rows of 21 vec4s (the 80 inputs, the bias, padding), then its
// second, 129 rows (the 128 hidden, the bias) of 4 vec4s (the 16 outputs).
@group(0) @binding(3) var<storage, read> weights: array<vec4<f32>>;
// Each page (TABLE i32s): its model (-1: none), its axis, then for each edge (-x, +x, -y, +y) the
// page beyond (-1: none) and where its cells land there (nca.rs edges).
@group(0) @binding(4) var<storage, read> table: array<i32>;
@group(0) @binding(5) var<storage, read> list: array<u32>;
@group(0) @binding(6) var<storage, read_write> atlas: array<u32>;
@group(0) @binding(7) var<storage, read_write> mem: array<f32>;

const W1_ROW: u32 = 21u;
const TABLE: u32 = 32u;
const MODEL: u32 = 128u * 21u + 129u * 4u;

fn hash(a: u32, b: u32, c: u32) -> u32 {
    var h = (a * 0x9e3779b1u) ^ (b * 0x85ebca77u) ^ (c * 0xc2b2ae3du);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    h *= 0x297a2d39u;
    return h ^ (h >> 15u);
}

// The cell at (x, y) of `page`'s square, past its edges onto the page the surface goes on onto
// (round corners too, nca.rs edges), or held at its own edge where there is none: its index.
fn cell_of(page: u32, xi: i32, yi: i32) -> u32 {
    let r = i32(q.r);
    var pg = page;
    var x = xi;
    var y = yi;
    for (var k = 0; k < 2; k++) {
        var e = -1;
        if (x < 0) { e = 0; } else if (x >= r) { e = 1; } else if (y < 0) { e = 2; } else if (y >= r) { e = 3; }
        if (e < 0) { break; }
        let t = pg * TABLE + 2u + u32(e) * 7u;
        let n = table[t];
        if (n < 0) { break; }
        let nx = table[t + 1u] * x + table[t + 2u] * y + table[t + 5u];
        let ny = table[t + 3u] * x + table[t + 4u] * y + table[t + 6u];
        pg = u32(n);
        x = nx;
        y = ny;
    }
    x = clamp(x, 0, r - 1);
    y = clamp(y, 0, r - 1);
    return pg * q.r * q.r + u32(y) * q.r + u32(x);
}

struct Cell {
    v: array<vec4<f32>, 4>,
}

fn load(i: u32) -> Cell {
    let a = state_in[i * 2u];
    let b = state_in[i * 2u + 1u];
    var c: Cell;
    c.v[0] = vec4<f32>(unpack2x16float(a.x), unpack2x16float(a.y));
    c.v[1] = vec4<f32>(unpack2x16float(a.z), unpack2x16float(a.w));
    c.v[2] = vec4<f32>(unpack2x16float(b.x), unpack2x16float(b.y));
    c.v[3] = vec4<f32>(unpack2x16float(b.z), unpack2x16float(b.w));
    return c;
}

fn store(i: u32, c: Cell) {
    state_out[i * 2u] = vec4<u32>(pack2x16float(c.v[0].xy), pack2x16float(c.v[0].zw), pack2x16float(c.v[1].xy), pack2x16float(c.v[1].zw));
    state_out[i * 2u + 1u] = vec4<u32>(pack2x16float(c.v[2].xy), pack2x16float(c.v[2].zw), pack2x16float(c.v[3].xy), pack2x16float(c.v[3].zw));
}

// The world's directions along a page's two axes (shell.wgsl axes_of).
fn axes_of(ax: i32) -> array<vec3<f32>, 2> {
    if (ax == 0) { return array<vec3<f32>, 2>(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
    if (ax == 1) { return array<vec3<f32>, 2>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
    return array<vec3<f32>, 2>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
}

// One step: perception as MeshNCA's (16 channels x [laplacian, the gradient's world y, z and x
// parts, the cell itself], the gradient a sum of the unit directions to the neighbours times
// their differences), scaled from the icosphere's 6 neighbours to the grid's 8; then 80 -> 128
// (ReLU) -> 16, added to the cell with chance `fire`.
@compute @workgroup_size(8, 8)
fn nca_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= q.r || gid.y >= q.r || gid.z >= q.pages) { return; }
    let page = list[gid.z];
    let here = page * q.r * q.r + gid.y * q.r + gid.x;
    let me = load(here);
    let model = table[page * TABLE];
    if (gid.z >= q.stepped || model < 0 || (hash(here, q.seed, 7u) & 0xffffu) >= u32(q.fire * 65536.0)) {
        store(here, me);
        return;
    }
    var lap: array<vec4<f32>, 4>;
    var gu: array<vec4<f32>, 4>;
    var gv: array<vec4<f32>, 4>;
    for (var k = 0u; k < 4u; k++) {
        lap[k] = vec4<f32>(0.0);
        gu[k] = vec4<f32>(0.0);
        gv[k] = vec4<f32>(0.0);
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let n = load(cell_of(page, x + dx, y + dy));
            let d = vec2<f32>(f32(dx), f32(dy)) / length(vec2<f32>(f32(dx), f32(dy)));
            for (var k = 0u; k < 4u; k++) {
                let diff = n.v[k] - me.v[k];
                lap[k] += diff;
                gu[k] += d.x * diff;
                gv[k] += d.y * diff;
            }
        }
    }
    // The 8 neighbours' sums against 6 evenly round: the laplacian's 3 h^2 against 1.5, the
    // gradient's (2 + 2 sqrt 2) h against 3.
    let gk = 3.0 / (2.0 + 2.0 * sqrt(2.0));
    let ax = axes_of(table[page * TABLE + 1u]);
    var xin: array<vec4<f32>, 21>;
    for (var k = 0u; k < 4u; k++) {
        let u = gu[k] * gk;
        let v = gv[k] * gk;
        xin[k] = lap[k] * 0.5;
        xin[4u + k] = ax[0].y * u + ax[1].y * v;
        xin[8u + k] = ax[0].z * u + ax[1].z * v;
        xin[12u + k] = ax[0].x * u + ax[1].x * v;
        xin[16u + k] = me.v[k];
    }
    xin[20] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let base = u32(model) * MODEL;
    let second = base + 128u * W1_ROW;
    var out: array<vec4<f32>, 4>;
    for (var k = 0u; k < 4u; k++) {
        out[k] = weights[second + 128u * 4u + k];
    }
    for (var j = 0u; j < 128u; j++) {
        let row = base + j * W1_ROW;
        var h = 0.0;
        for (var i = 0u; i < 21u; i++) {
            h += dot(xin[i], weights[row + i]);
        }
        h = max(h, 0.0);
        if (h > 0.0) {
            let o = second + j * 4u;
            for (var k = 0u; k < 4u; k++) {
                out[k] += h * weights[o + k];
            }
        }
    }
    var next = me;
    for (var k = 0u; k < 4u; k++) {
        next.v[k] = me.v[k] + out[k];
    }
    store(here, next);
}

// The colour (channels 0..2, from -0.5 up, as MeshNCA's albedo) and height (channel 6,
// (h + 0.5)^2 as MeshNCA's, times `depth`, less `lift`) at a point of a page in cells, bilinear
// across its edges, from state_in (the step's last written, bound as it).
fn sample(page: u32, c: vec2<f32>) -> vec4<f32> {
    let f0 = floor(c);
    let f = c - f0;
    let i = vec2<i32>(f0);
    var s = array<vec4<f32>, 4>();
    for (var k = 0; k < 4; k++) {
        let at = cell_of(page, i.x + (k & 1), i.y + (k >> 1u));
        let a = state_in[at * 2u];
        let b = state_in[at * 2u + 1u];
        s[k] = vec4<f32>(unpack2x16float(a.x), unpack2x16float(a.y).x, unpack2x16float(a.w).x);
    }
    return mix(mix(s[0], s[1], f.x), mix(s[2], s[3], f.x), f.y);
}

@compute @workgroup_size(16, 16)
fn paint(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pr = q.page_res;
    if (gid.x >= pr || gid.y >= pr || gid.z >= q.stepped) { return; }
    let page = list[gid.z];
    if (table[page * TABLE] < 0) { return; }
    let cells = (vec2<f32>(gid.xy) + 0.5) * f32(q.r) / f32(pr) - 0.5;
    let s = sample(page, cells);
    var c = s.rgb;
    if (q.nearest != 0u) {
        c = sample(page, floor(cells + 0.5)).rgb;
    }
    let rgb = clamp(c + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
    let i = page * pr * pr + gid.y * pr + gid.x;
    atlas[i * 2u] = pack2x16float(rgb.rg);
    atlas[i * 2u + 1u] = pack2x16float(vec2<f32>(rgb.b, 1.0));
    let cell = pr / q.hr;
    if (gid.x % cell == 0u && gid.y % cell == 0u) {
        let h = clamp(s.w + 0.5, 0.0, 1.0);
        mem[q.height_at + page * q.hr * q.hr + (gid.y / cell) * q.hr + gid.x / cell] = q.depth * (h * h - q.lift);
    }
}
