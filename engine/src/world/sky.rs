//! The sky over the waking world (`level::Sky`): one 64x64 16-colour texture, its rows
//! running from overhead (row 0) down to the horizon (row 63), tiling across, which the
//! renderer wraps round a dome about the camera.

use crate::world::level::Sky;
use crate::world::tex::{self, Texture, SIZE};

/// Cloud cover thins to nothing over the last rows, so the horizon is the fog's colour.
const CLEAR_ROWS: usize = 6;

/// The sky's texture.
pub fn texture(sky: &Sky, seed: u64) -> Texture {
    let mut px = Vec::with_capacity(SIZE * SIZE);
    for y in 0..SIZE {
        let down = y as f32 / (SIZE - 1) as f32;
        let k = down.powf(1.6);
        let clear = [0, 1, 2].map(|c| sky.top[c] + (sky.horizon[c] - sky.top[c]) * k);
        let fade = ((SIZE - 1 - y) as f32 / CLEAR_ROWS as f32).min(1.0);
        for x in 0..SIZE {
            // Clouds stretch out flat toward the horizon.
            let n = tex::fbm(x, y, 4, 4, seed);
            let cover = ((n - (1.0 - sky.clouds)) / 0.12).clamp(0.0, 1.0) * fade;
            // Lit tops, greyer undersides.
            let shade = 0.82 + 0.18 * tex::fbm(x, (y + 2) % SIZE, 8, 2, seed ^ 0x51);
            let cloud = sky.cloud.map(|c| c * shade);
            px.push([0, 1, 2].map(|c| 255.0 * (clear[c] + (cloud[c] - clear[c]) * cover).clamp(0.0, 1.0)));
        }
    }
    crate::world::photo::quantise(&px)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_horizon_is_the_fogs_colour_and_overhead_the_tops() {
        let sky = Sky { clouds: 0.6, ..Sky::default() };
        let t = texture(&sky, 7);
        let at = |x: usize, y: usize| t.pal[t.idx[y * SIZE + x] as usize].map(|c| c as f32 / 255.0);
        let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|c| (a[c] - b[c]).abs() < 0.06);
        assert!((0..SIZE).all(|x| near(at(x, SIZE - 1), sky.horizon)), "horizon row");
        let clear_top = (0..SIZE).filter(|&x| near(at(x, 0), sky.top)).count();
        let cloudy = (0..SIZE * SIZE / 2).filter(|&i| near(at(i % SIZE, i / SIZE), sky.cloud)).count();
        assert!(clear_top > 0 && cloudy > 0, "{clear_top} clear overhead, {cloudy} cloud texels");
        // No clouds asked for, none drawn.
        let t = texture(&Sky { clouds: 0.0, ..sky.clone() }, 7);
        let none = (0..SIZE * SIZE).all(|i| !near(t.pal[t.idx[i] as usize].map(|c| c as f32 / 255.0), sky.cloud));
        assert!(none);
    }
}
