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
// What the parts of flesh that moves are doing (world::dreamed::Moving::state): first the
// time and the heartbeat's period, then each part's numbers, which its corners point at.
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var<storage, read> parts: array<vec4<f32>>;

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
    // How far the surface has grown in; below 1 it has a ragged edge.
    @location(14) grow: f32,
    // Flesh that moves: how (Moving::how), where its part's numbers start, two of its own.
    @if(MOVING)
    @location(15) moves: vec4<f32>,
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
    @location(14) grow: f32,
};

// How the parts move: as world::dreamed's BALL, LID, VEIN, MOUTH and FILMED.
const BALL: i32 = 1;
const LID: i32 = 2;
const VEIN: i32 = 3;
const MOUTH: i32 = 4;
const FILMED: i32 = 5;

// How much of a heartbeat is passing, x beats after the last: world::dreamed::beat.
fn beat(x: f32) -> f32 {
    let f = x - floor(x);
    let a = (f - 0.1) / 0.06;
    let b = (f - 0.34) / 0.07;
    return min(exp(-a * a) + 0.6 * exp(-b * b), 1.0);
}

// `v` turned by `angle` round the unit `axis`.
fn turn(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    return v * c + cross(axis, v) * sin(angle) + axis * dot(axis, v) * (1.0 - c);
}

@vertex
fn vertex(v: Vertex) -> Varying {
    // Meshes are built in the world (identity transform), in tiles with y and z turned
    // round; parts move in the engine's own world.
    var out: Varying;
    let tile = u.eye.w;
    var p = vec3(v.position.x, -v.position.y, -v.position.z) * tile;
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
    var shown = true;
    @if(MOVING)
    {
        let how = i32(round(v.moves.x));
        let at = u32(round(v.moves.y));
        let first = parts[at];
        if how == BALL {
            // Turned in its socket to look at the viewer, with everything that points.
            shown = first.w > 0.0;
            let turned = mat3x3<f32>(first.xyz, parts[at + 1u].xyz, parts[at + 2u].xyz);
            let centre = parts[at + 3u].xyz;
            p = centre + turned * (p - centre);
            out.nw = vec4(turned * v.nw.xyz, v.nw.w);
            out.tw0 = vec4(turned * v.tw0.xyz, v.tw0.w);
            out.bs0 = vec4(turned * v.bs0.xyz, v.bs0.w);
            out.tw1 = vec4(turned * v.tw1.xyz, v.tw1.w);
            out.bs1 = vec4(turned * v.bs1.xyz, v.bs1.w);
            let cu0 = turned * v.ax0.xyz;
            let cv0 = turned * vec3(v.ax0.w, v.ax1.xy);
            let cu1 = turned * vec3(v.ax1.zw, v.ax2.x);
            let cv1 = turned * v.ax2.yzw;
            out.ax0 = vec4(cu0, cv0.x);
            out.ax1 = vec4(cv0.yz, cu1.xy);
            out.ax2 = vec4(cu1.z, cv1);
        } else if how == LID {
            // Further down the eyeball along its own line of longitude, the more shut it is.
            shown = first.w > 0.0;
            let side = v.moves.w;
            let centre = parts[at + 3u].xyz;
            let up = parts[at + 4u].xyz * side;
            let more = select(parts[at + 2u].w, parts[at + 1u].w, side > 0.0) * v.moves.z;
            let axis = cross(up, p - centre);
            if dot(axis, axis) > 1e-12 {
                let k = normalize(axis);
                p = centre + turn(p - centre, k, more);
                out.nw = vec4(turn(v.nw.xyz, k, more), v.nw.w);
            }
        } else if how == VEIN {
            // Fattened where the beat has got to, and flushed red with it.
            shown = first.x > 0.0;
            let b = beat(parts[0].x / parts[0].y - v.moves.z);
            p += v.nw.xyz * v.moves.w * b;
            out.nw = vec4(v.nw.xyz, 1.0 - b);
        } else if how == MOUTH {
            // Drawn shut towards the middle and bulging, the outside of the rim staying put
            // (world::dreamed::mesh_mouth).
            let rim = 1.9;
            let shut = 0.06;
            let out_of_wall = first.w;
            shown = out_of_wall > 0.0;
            let across = parts[at + 1u];
            let up = parts[at + 2u];
            let n = parts[at + 3u].xyz;
            let clench = parts[at + 3u].w;
            let pull = 1.0 + clench * (rim / (rim - 1.0 + shut) - 1.0);
            let d = p - first.xyz;
            let x = dot(d, across.xyz) / across.w;
            let y = dot(d, up.xyz) / up.w;
            let z = dot(d, n);
            let far = max(length(vec2(x, y)), 1e-4);
            let to = select(far, max(rim - (rim - far) * pull, shut * 0.3), far < rim);
            let rise = max(clench, 0.0) * 0.35 * up.w * (1.0 - min(to / rim, 1.0));
            let k = to / far * out_of_wall;
            p = first.xyz + across.xyz * (x * k * across.w) + up.xyz * (y * k * up.w) + n * ((z + rise) * out_of_wall);
        } else if how == FILMED {
            // The frame of the film and the gaze it's on.
            shown = first.y > 0.0;
            out.bs0 = vec4(v.bs0.xyz, first.x);
        }
    }
    let world = vec3(p.x, -p.y, -p.z) / tile;
    out.position = position_world_to_clip(world);
    if !shown {
        // Out of sight: nothing of it is drawn.
        out.position = vec4(0.0, 0.0, -2.0, 1.0);
    }
    let depth = -position_world_to_view(world).z * tile;
    out.pz = vec4(p, depth);
    out.sh0 = v.sh0;
    out.sh1 = v.sh1;
    out.sh2 = v.sh2;
    out.grow = v.grow;
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

fn hash(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3)) * 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let edge = array<f32, 4>(
        mix(hash(i), hash(i + vec3(1.0, 0.0, 0.0)), s.x),
        mix(hash(i + vec3(0.0, 1.0, 0.0)), hash(i + vec3(1.0, 1.0, 0.0)), s.x),
        mix(hash(i + vec3(0.0, 0.0, 1.0)), hash(i + vec3(1.0, 0.0, 1.0)), s.x),
        mix(hash(i + vec3(0.0, 1.0, 1.0)), hash(i + vec3(1.0, 1.0, 1.0)), s.x),
    );
    return mix(mix(edge[0], edge[1], s.y), mix(edge[2], edge[3], s.y), s.z);
}

// The pattern that a growing edge follows, about 0..1: a fixed thing at each place.
fn ragged(p: vec3<f32>) -> f32 {
    return 0.55 * value_noise(p) + 0.3 * value_noise(p * 2.1 + 7.0) + 0.15 * value_noise(p * 4.3 + 13.0);
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
    // Where the surface is still growing in, it stops along a ragged line with a dark wet rim.
    if in.grow < 1.0 {
        let cover = in.grow * 2.2 - 0.9 + (ragged(p / u.eye.w * 1.6) - 0.5) * 2.0;
        if cover <= 0.0 {
            discard;
        }
        let rim = 1.0 - smoothstep(0.0, 0.22, cover);
        alb = mix(alb, vec3(0.1, 0.015, 0.02), rim * 0.92);
        wet += rim * 0.7;
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
