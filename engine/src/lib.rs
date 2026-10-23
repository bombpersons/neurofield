//! neurofield's engine, shared by the game, the editor and the training data:
//!
//! - `world`: dungeons (generated or hand-built), their props, doors and
//!   creatures, textures and light, and walking around them;
//! - `render`: the world drawn on the GPU with the PS1 look (psx-bevy), creatures
//!   in full colour over it, in a window or headless;
//! - `neural`: the trained network drawing the world instead.

pub mod neural;
pub mod render;
pub mod world;
