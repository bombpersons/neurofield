// The level's triangles drawn as which one each pixel shows (level.rs Raster): its index + 1.

struct View {
    pos: vec4<f32>,
    // The view's axes as rows: right, down, forward.
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    // Focal length in pixels, half the width and height, the fog's far end.
    k: vec4<f32>,
}

@group(0) @binding(0) var<uniform> v: View;

// level.rs NEAR.
const NEAR: f32 = 0.02;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) z: f32,
    @location(1) @interpolate(flat) tri: u32,
}

@vertex
fn vs(@location(0) p: vec3<f32>, @builtin(vertex_index) i: u32) -> Out {
    let d = p - v.pos.xyz;
    let q = vec3<f32>(dot(v.r0.xyz, d), dot(v.r1.xyz, d), dot(v.r2.xyz, d));
    var out: Out;
    // Screen x = w/2 + focal x/z, y = h/2 + focal y/z (y down); depth NEAR/z, cut where z < NEAR.
    out.clip = vec4<f32>(v.k.x * q.x / v.k.y, -v.k.x * q.y / v.k.z, NEAR, q.z);
    out.z = q.z;
    out.tri = i / 3u + 1u;
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) u32 {
    if (in.z > v.k.w) { discard; }
    return in.tri;
}
