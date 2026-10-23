// The creatures in full colour (see flesh.rs). Works in the engine's world
// (y down, its units) and in encoded 0..1
// colour, which the flesh camera's float target keeps as it is.

import bevy_pbr::render::view_transformations::{position_world_to_clip, position_world_to_view};

struct FleshUniform {
    // The eye in world units; w: one tile in them.
    eye: vec4<f32>,
    // Torch colour, radius.
    torch: vec4<f32>,
    // Fog near, far.
    fog: vec4<f32>,
    fog_colour: vec4<f32>,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> u: FleshUniform;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var colour_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var normal_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var skin_sampler: sampler;

struct Vertex {
    @location(0) position: vec3<f32>,
    // Normal, blend weight of the first skin.
    @location(1) nw: vec4<f32>,
    // Per skin: wrapped uv, cap uv; tangent, wrap weight; bitangent, layer.
    @location(2) uvcap0: vec4<f32>,
    @location(3) tw0: vec4<f32>,
    @location(4) bs0: vec4<f32>,
    @location(5) uvcap1: vec4<f32>,
    @location(6) tw1: vec4<f32>,
    @location(7) bs1: vec4<f32>,
    // Cap axes: skin 0 u, v, skin 1 u, v, packed.
    @location(8) ax0: vec4<f32>,
    @location(9) ax1: vec4<f32>,
    @location(10) ax2: vec4<f32>,
    // Baked light, linear SH rows (mean, then x, y, z gradient), packed.
    @location(11) sh0: vec4<f32>,
    @location(12) sh1: vec4<f32>,
    @location(13) sh2: vec4<f32>,
};

struct Varying {
    @builtin(position) position: vec4<f32>,
    // Datagen position; view depth in its units.
    @location(0) pz: vec4<f32>,
    @location(1) nw: vec4<f32>,
    @location(2) uvcap0: vec4<f32>,
    @location(3) tw0: vec4<f32>,
    @location(4) bs0: vec4<f32>,
    @location(5) uvcap1: vec4<f32>,
    @location(6) tw1: vec4<f32>,
    @location(7) bs1: vec4<f32>,
    @location(8) ax0: vec4<f32>,
    @location(9) ax1: vec4<f32>,
    @location(10) ax2: vec4<f32>,
    @location(11) sh0: vec4<f32>,
    @location(12) sh1: vec4<f32>,
    @location(13) sh2: vec4<f32>,
};

@vertex
fn vertex(v: Vertex) -> Varying {
    // Meshes are built in the world (identity transform).
    var out: Varying;
    out.position = position_world_to_clip(v.position);
    let t = u.eye.w;
    let depth = -position_world_to_view(v.position).z * t;
    out.pz = vec4(v.position.x * t, -v.position.y * t, -v.position.z * t, depth);
    out.nw = v.nw;
    out.uvcap0 = v.uvcap0;
    out.tw0 = v.tw0;
    out.bs0 = v.bs0;
    out.uvcap1 = v.uvcap1;
    out.tw1 = v.tw1;
    out.bs1 = v.bs1;
    out.ax0 = v.ax0;
    out.ax1 = v.ax1;
    out.ax2 = v.ax2;
    out.sh0 = v.sh0;
    out.sh1 = v.sh1;
    out.sh2 = v.sh2;
    return out;
}

struct Sample {
    colour: vec3<f32>,
    wet: f32,
    n: vec3<f32>,
};

// One projection of a skin: its texel and normal-mapped normal.
fn project(n0: vec3<f32>, uv: vec2<f32>, tt: vec3<f32>, bb: vec3<f32>, layer: i32, ddx: vec2<f32>, ddy: vec2<f32>) -> Sample {
    var t = tt - n0 * dot(n0, tt);
    if dot(t, t) < 1e-8 {
        t = cross(n0, select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(n0.x) < 0.9));
    }
    t = normalize(t);
    var b = cross(n0, t);
    if dot(b, bb) < 0.0 {
        b = -b;
    }
    let c = textureSampleGrad(colour_tex, skin_sampler, uv, layer, ddx, ddy);
    let tn = textureSampleGrad(normal_tex, skin_sampler, uv, layer, ddx, ddy).xyz * 2.0 - 1.0;
    var s: Sample;
    s.colour = c.rgb;
    s.wet = c.a;
    s.n = normalize(t * tn.x + b * tn.y + n0 * tn.z);
    return s;
}

fn luma(c: vec3<f32>) -> f32 {
    return 0.3 * c.x + 0.59 * c.y + 0.11 * c.z;
}

fn lobes(c: f32) -> f32 {
    let x = max(c, 1e-6);
    return 1.6 * pow(x, 120.0) + 0.2 * pow(x, 14.0);
}

@fragment
fn fragment(in: Varying) -> @location(0) vec4<f32> {
    let p = in.pz.xyz;
    let z = in.pz.w;
    let n0 = normalize(in.nw.xyz);
    let w = clamp(in.nw.w, 0.0, 1.0);
    let cap_u = array<vec3<f32>, 2>(in.ax0.xyz, vec3(in.ax1.zw, in.ax2.x));
    let cap_v = array<vec3<f32>, 2>(vec3(in.ax0.w, in.ax1.xy), in.ax2.yzw);
    let uvcap = array<vec4<f32>, 2>(in.uvcap0, in.uvcap1);
    let tw = array<vec4<f32>, 2>(in.tw0, in.tw1);
    let bs = array<vec4<f32>, 2>(in.bs0, in.bs1);
    let ws = array<f32, 2>(w, 1.0 - w);
    // Gradients outside the loop, where control flow is uniform.
    let d_uv = array<vec4<f32>, 2>(dpdx(in.uvcap0), dpdx(in.uvcap1));
    let e_uv = array<vec4<f32>, 2>(dpdy(in.uvcap0), dpdy(in.uvcap1));

    // Each skin's albedo, normal-mapped normal and gloss, blended by weight;
    // each wrapped round the axis, cross-faded to a projection onto the ends.
    var alb = vec3(0.0);
    var nsum = vec3(0.0);
    var wet = 0.0;
    for (var s = 0; s < 2; s++) {
        let layer = i32(round(bs[s].w));
        let wrap = clamp(tw[s].w, 0.0, 1.0);
        let a = project(n0, uvcap[s].xy, tw[s].xyz, bs[s].xyz, layer, d_uv[s].xy, e_uv[s].xy);
        let c = project(n0, uvcap[s].zw, cap_u[s], cap_v[s], layer, d_uv[s].zw, e_uv[s].zw);
        let ka = ws[s] * wrap;
        let kc = ws[s] * (1.0 - wrap);
        alb += ka * a.colour + kc * c.colour;
        nsum += ka * a.n + kc * c.n;
        wet += ka * a.wet + kc * c.wet;
    }
    let n = normalize(nsum);
    let sh_mean = in.sh0.xyz;
    let sh_x = vec3(in.sh0.w, in.sh1.xy);
    let sh_y = vec3(in.sh1.zw, in.sh2.x);
    let sh_z = in.sh2.yzw;

    let to_eye = u.eye.xyz - p;
    let dist = max(length(to_eye), 1.0);
    let v = to_eye / dist;
    let nv = max(dot(n, v), 0.0);

    // Baked light, exactly as the part's token describes it.
    let irr = max(sh_mean + sh_x * n.x + sh_y * n.y + sh_z * n.z, 0.05 * sh_mean);
    // Its gradient doubles as a key light for highlights.
    let g = vec3(luma(sh_x), luma(sh_y), luma(sh_z));
    let key = sqrt(sh_x * sh_x + sh_y * sh_y + sh_z * sh_z);
    // The torch sits at the eye, so its half vector is the view vector.
    let fall = pow(max(1.0 - dist / u.torch.w, 0.0), 2.0);
    let wrap_light = max((nv + 0.3) / 1.3, 0.0);
    let spec_torch = lobes(nv) * fall;
    var spec_key = 0.0;
    if dot(g, g) > 1e-8 {
        spec_key = lobes(max(dot(n, normalize(normalize(g) + v)), 0.0));
    }
    // A sheen from overhead (y is down), so evenly lit rooms still catch the wet.
    let spec_top = lobes(max(dot(n, normalize(vec3(v.x, v.y - 1.0, v.z))), 0.0));
    let fresnel = 0.04 + 0.96 * pow(1.0 - nv, 5.0);

    let f = (z - u.fog.x) / max(u.fog.y - u.fog.x, 1.0);
    let fog = sqrt(1.0 - clamp(f, 0.0, 1.0));
    let diffuse = alb * (irr + u.torch.rgb * fall * wrap_light);
    let gloss = wet * (spec_torch * u.torch.rgb + 2.5 * spec_key * key + (0.5 * spec_top + 1.2 * fresnel) * sh_mean);
    let rgb = (diffuse + gloss) * fog + u.fog_colour.rgb * (1.0 - fog);
    return vec4(rgb, 1.0);
}
