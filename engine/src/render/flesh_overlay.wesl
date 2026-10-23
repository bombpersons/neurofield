// Lays the creatures over the finished PS1 screen: each screen pixel averages
// its 2x2 flesh samples that a creature covered, over the screen by how many
// were.

import bevy_sprite_render::mesh2d::vertex_output::VertexOutput;

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var flesh: texture_2d<f32>;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
    return select(hi, lo, c <= vec3(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = vec2<i32>(in.position.xy) * 2;
    var sum = vec3(0.0);
    var n = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let s = textureLoad(flesh, at + vec2(x, y), 0);
            if s.a > 0.5 {
                sum += s.rgb;
                n += 1.0;
            }
        }
    }
    if n == 0.0 {
        discard;
    }
    // Encoded colour, as the screen holds it when it composites in sRGB.
    var rgb = clamp(sum / n, vec3(0.0), vec3(1.0));
    @if(!SRGB_OUTPUT)
    rgb = srgb_to_linear(rgb);
    return vec4(rgb, n / 4.0);
}
