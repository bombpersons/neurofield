// The heights at the creases (shell.rs Shell::creases), appended to shell.wgsl: at each of the
// points along each crease of each triangle, half the difference between the wall beyond's
// height and the triangle's own there, which the march (height_at) eases its height by near
// the crease, so both walls meet at their mean.

@group(1) @binding(0) var<storage, read_write> toward: array<f32>;
// Each crease (four vec4s): its start, the points along it (bits); its end, where they start in
// toward (bits); the triangle's plane (normal, distance); the wall beyond's.
@group(1) @binding(1) var<storage, read> crease_list: array<vec4<f32>>;
fn four_cells(at: u32, pg: u32, local: vec2<u32>) -> vec4<f32> {
    return cells_from(at, u32(u.o.w / HEIGHT_CELL));
}

// The heights smoothed (smooth.wgsl).
@group(1) @binding(2) var<storage, read> near: array<f32>;
// Each triangle's home tile and its page (tri_homes).
@group(1) @binding(4) var<storage, read_write> homes: array<vec4<i32>>;

@compute @workgroup_size(64)
fn crease_heights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.y + gid.z * 65535u;
    if (c >= arrayLength(&crease_list) / 4u) { return; }
    let a = crease_list[c * 4u];
    let b = crease_list[c * 4u + 1u];
    let points = bitcast<u32>(a.w);
    if (gid.x >= points) { return; }
    let mine = crease_list[c * 4u + 2u];
    let theirs = crease_list[c * 4u + 3u];
    own_cell = vec3<i32>(-2147483647);
    other_cell = own_cell;
    home_cell = own_cell;
    let e = mix(a.xyz, b.xyz, f32(gid.x) / f32(max(points, 2u) - 1u));
    let h = raw_height(e, plane_of(theirs.xyz, theirs.w, 1u)) - raw_height(e, plane_of(mine.xyz, mine.w, 0u));
    toward[bitcast<u32>(b.w) + gid.x] = 0.5 * h;
}

// shell.wgsl's fs reads it; unused here.
@group(1) @binding(9) var inner: texture_depth_2d;

// Each triangle's home tile (the cell its middle's in, as page_at finds cells) and that tile's
// page (-1 for none), for the march to keep to hand.
@compute @workgroup_size(64)
fn tri_homes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= arrayLength(&homes)) { return; }
    let a = corners[t * TRI];
    let b = corners[t * TRI + 1u];
    let c = corners[t * TRI + 2u];
    let m = vec3<f32>(a.w, b.w, c.w);
    let pl = plane_of(m, dot(a.xyz, m), 0u);
    let cell = vec3<i32>(floor((a.xyz + b.xyz + c.xyz) / 3.0 - m * 0.01));
    homes[t] = vec4<i32>(cell, bitcast<i32>(page_of_face(pl.orient, cell)));
}
