// Creatures drawn some other way (the dream net's) laid into the flesh image at their own
// depth, so the level and the flesh in front of them hide them (see flesh.rs).

import bevy_pbr::render::mesh_view_bindings::view;

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var colour: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var near: texture_2d<f32>;

struct Vertex {
    // Already in clip space: the quad covers the screen.
    @location(0) position: vec3<f32>,
};

@vertex
fn vertex(v: Vertex) -> @builtin(position) vec4<f32> {
    return vec4(v.position.xy, 0.5, 1.0);
}

struct Out {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> Out {
    // The pictures cover the screen, whatever their size: the game's are the screen's, the
    // editor's 240 lines however big its view (the flesh image has two samples a screen
    // pixel along each axis).
    let size = vec2<i32>(textureDimensions(colour));
    let at = min(vec2<i32>(position.xy / view.viewport.zw * vec2<f32>(size)), size - 1);
    let c = textureLoad(colour, at, 0);
    if c.a <= 0.0 {
        discard;
    }
    // Their depth is in tiles, which are the view's units.
    let d = textureLoad(near, at, 0).r;
    let clip = view.clip_from_view * vec4(0.0, 0.0, -d, 1.0);
    var out: Out;
    out.colour = c;
    out.depth = clip.z / clip.w;
    return out;
}
