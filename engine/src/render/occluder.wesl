// The level in the flesh camera: depth only. Writes what its clear does.

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4(0.0);
}
