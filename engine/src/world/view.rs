//! How a camera sees: the PS1 screen's size and the GTE's world-to-view
//! rotation, which `views.npy` records and the renderer (`render`) follows.

use crate::world::scene::Camera;
use psx_gte::math::{mul_matrix, rot_x, rot_y, rot_z};
use psx_gte::Matrix;

pub const W: u32 = 320;
pub const H: u32 = 240;

/// World → view rotation (1.3.12). Row 2 is the view direction in world space.
pub fn view_matrix(cam: &Camera) -> Matrix {
    mul_matrix(&rot_z(cam.roll), &mul_matrix(&rot_x(cam.pitch), &rot_y(cam.yaw)))
}
