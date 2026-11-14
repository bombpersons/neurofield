// A flesh automaton's step (nca.rs; train/nca.py has the same, and trained it): each cell of a
// wrapping SIZE x SIZE grid of CHANNELS perceives itself and its neighbours (identity, Sobel x
// and y over 8, Laplacian over 16, filter-major), and where a coin comes up adds
// w2 relu(w1 perceived + b1).

const CHANNELS: u32 = 12u;
const HIDDEN: u32 = 96u;
const PERCEIVED: u32 = 48u;

struct Params {
    size: u32,
    // Which step this is, for the coins; how often they come up (FIRE).
    step: u32,
    fire: f32,
    // How tall a unit of height channel stands, for `pack` (the normal's slope).
    _pad: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
// w1 [HIDDEN, PERCEIVED], b1 [HIDDEN], w2 [CHANNELS, HIDDEN], as nca.bin.
@group(0) @binding(1) var<storage, read> weights: array<f32>;
// [SIZE * SIZE * CHANNELS], cell-major.
@group(0) @binding(2) var<storage, read> state: array<f32>;
@group(0) @binding(3) var<storage, read_write> next: array<f32>;
// The colour and height as RGBA8, a u32 a cell.
@group(0) @binding(4) var<storage, read_write> packed: array<u32>;

fn at(x: i32, y: i32, c: u32) -> f32 {
    let n = i32(p.size);
    let xx = u32((x % n + n) % n);
    let yy = u32((y % n + n) % n);
    return state[(yy * p.size + xx) * CHANNELS + c];
}

fn hash(a: u32) -> u32 {
    var x = a;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

@compute @workgroup_size(8, 8)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.size || id.y >= p.size {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    var seen: array<f32, 48>;
    for (var c = 0u; c < CHANNELS; c++) {
        let tl = at(x - 1, y - 1, c);
        let t = at(x, y - 1, c);
        let tr = at(x + 1, y - 1, c);
        let l = at(x - 1, y, c);
        let m = at(x, y, c);
        let r = at(x + 1, y, c);
        let bl = at(x - 1, y + 1, c);
        let b = at(x, y + 1, c);
        let br = at(x + 1, y + 1, c);
        seen[c] = m;
        seen[CHANNELS + c] = ((tr + 2.0 * r + br) - (tl + 2.0 * l + bl)) / 8.0;
        seen[2u * CHANNELS + c] = ((bl + 2.0 * b + br) - (tl + 2.0 * t + tr)) / 8.0;
        seen[3u * CHANNELS + c] = ((tl + tr + bl + br) + 2.0 * (t + l + r + b) - 12.0 * m) / 16.0;
    }
    let cell = id.y * p.size + id.x;
    let fires = f32(hash(cell * 2654435761u ^ hash(p.step)) & 0xffffffu) / 16777216.0 < p.fire;
    var dx: array<f32, 12>;
    if fires {
        let b1 = HIDDEN * PERCEIVED;
        let w2 = b1 + HIDDEN;
        for (var h = 0u; h < HIDDEN; h++) {
            var s = weights[b1 + h];
            for (var i = 0u; i < PERCEIVED; i++) {
                s += weights[h * PERCEIVED + i] * seen[i];
            }
            s = max(s, 0.0);
            for (var c = 0u; c < CHANNELS; c++) {
                dx[c] += weights[w2 + c * HIDDEN + h] * s;
            }
        }
    }
    for (var c = 0u; c < CHANNELS; c++) {
        next[cell * CHANNELS + c] = seen[c] + dx[c];
    }
}

// The colour (channels 0-2, + 0.5) and height (3, + 0.5), 0..1, as RGBA8.
@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell = id.x;
    if cell >= p.size * p.size {
        return;
    }
    let v = vec4(state[cell * CHANNELS], state[cell * CHANNELS + 1u], state[cell * CHANNELS + 2u], state[cell * CHANNELS + 3u]) + 0.5;
    packed[cell] = pack4x8unorm(clamp(v, vec4(0.0), vec4(1.0)));
}
