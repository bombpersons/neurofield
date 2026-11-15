// The march's hits finished (shell.rs Shell::resolve), appended to shell.wgsl for the march's
// module: a pixel at a time, once the march's depth test has kept only the nearest of the
// tetrahedra's hits, its slope, page and look worked out (shell.wgsl `finish`), rather than
// for every tetrahedron that meets the relief along the line of sight.

@group(2) @binding(0) var marched: texture_2d<u32>;

// A triangle over the whole target.
@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn resolve(@builtin(position) at: vec4<f32>) -> Hit {
    let found = textureLoad(marched, vec2<i32>(at.xy), 0);
    if (found.x == 0u) { discard; }
    let tri = found.x - 1u;
    begin(tri);
    let b1 = bitcast<f32>(found.y);
    let b2 = bitcast<f32>(found.z);
    return finish(tri, vec4<f32>(1.0 - b1 - b2, b1, b2, 0.0), bitcast<f32>(found.w));
}
