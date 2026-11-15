// The shell's pixels laid into a camera's own frame (DreamRenderer::encode_psx): where the march
// met the relief (its triangle at `tris`, 4 u32s a pixel, `tri_rows` a row), the frame shown
// (`show`'s, RGBA8 at `shown`, `rows` a row); elsewhere what the camera drew is left.

struct Blit {
    shown: u32,
    rows: u32,
    tris: u32,
    tri_rows: u32,
}

@group(0) @binding(0) var<storage, read> img: array<u32>;
@group(0) @binding(1) var<uniform> b: Blit;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(p.x);
    let y = u32(p.y);
    if (img[b.tris + y * b.tri_rows + x * 4u] == 0u) {
        discard;
    }
    return unpack4x8unorm(img[b.shown + y * b.rows + x]);
}
