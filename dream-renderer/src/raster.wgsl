// The level's triangles, subdivided and pushed out by the height (level.rs Raster), drawn as
// which one each pixel shows (its index + 1), and the point of the level's own surface each
// pixel's came from, with its view depth; and the level's light, the torch's albedo (both
// halved, to hold up to twice the texture) and the texture, beside the triangle.

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
// Each vertex where it's pushed out to, then where it was (level.rs MOVED).
@group(0) @binding(1) var<storage, read> moved: array<vec4<f32>>;
@group(0) @binding(2) var textures: texture_2d_array<f32>;
@group(0) @binding(3) var nearest: sampler;

// level.rs NEAR.
const NEAR: f32 = 0.02;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) z: f32,
    @location(1) flat_point: vec3<f32>,
    @location(2) @interpolate(flat) tri: u32,
    @location(3) @interpolate(flat) layer: u32,
    @location(4) uv: vec2<f32>,
    @location(5) light: vec3<f32>,
    @location(6) albedo: vec3<f32>,
}

// A corner (level.rs Corner): its vertex, the level triangle it's of and its texture, and its
// place on the texture, light and torch albedo.
@vertex
fn vs(@location(0) corner: vec3<u32>, @location(1) uv: vec2<f32>, @location(2) light: vec3<f32>, @location(3) albedo: vec3<f32>) -> Out {
    let d = moved[corner.x * 2u].xyz - v.pos.xyz;
    let q = vec3<f32>(dot(v.r0.xyz, d), dot(v.r1.xyz, d), dot(v.r2.xyz, d));
    var out: Out;
    // Screen x = w/2 + focal x/z, y = h/2 + focal y/z (y down); depth NEAR/z, cut where z < NEAR.
    out.clip = vec4<f32>(v.k.x * q.x / v.k.y, -v.k.x * q.y / v.k.z, NEAR, q.z);
    out.z = q.z;
    out.flat_point = moved[corner.x * 2u + 1u].xyz;
    out.tri = corner.y + 1u;
    out.layer = corner.z;
    out.uv = uv;
    out.light = light;
    out.albedo = albedo;
    return out;
}

// The triangle, the light and the torch's albedo (each RGBA8, halved) and the texture; and the
// point with its depth.
struct Drawn {
    @location(0) tri: vec4<u32>,
    @location(1) point: vec4<f32>,
}

@fragment
fn fs(in: Out) -> Drawn {
    if (in.z > v.k.w) { discard; }
    var out: Drawn;
    out.point = vec4<f32>(in.flat_point, in.z);
    let texel = textureSample(textures, nearest, clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0)), in.layer).rgb;
    out.tri = vec4<u32>(in.tri, pack4x8unorm(vec4<f32>(0.5 * in.light, 1.0)), pack4x8unorm(vec4<f32>(0.5 * in.albedo, 1.0)), pack4x8unorm(vec4<f32>(texel, 1.0)));
    return out;
}
