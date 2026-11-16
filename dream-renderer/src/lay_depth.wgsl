// A camera's depth laid into another's of the same view, stretched over it (DreamRenderer::
// lay_depth): the creatures' camera starts from the PS1 camera's depth, the shell's bulges in it.

@group(0) @binding(0) var given: texture_depth_2d;
// The target's size, in pixels.
@group(0) @binding(1) var<uniform> size: vec4<u32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @builtin(frag_depth) f32 {
    let have = textureDimensions(given);
    let at = min(vec2<u32>(floor(p.xy * vec2<f32>(have) / vec2<f32>(size.xy))), have - 1u);
    return textureLoad(given, at, 0);
}
