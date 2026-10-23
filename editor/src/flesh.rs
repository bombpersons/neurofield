//! Creatures in full colour in the 3D view, as the game draws them. Sculpting one
//! takes 30-400 ms, too slow to redo on every edit, so each is sculpted in the
//! background and kept by its shape (`Creature::shape_hash`). Until its sculpt is
//! ready a creature is drawn through the PS1 pipeline like the level, then it
//! swaps over; moving a light only bakes the kept sculpt's light again.

use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future::poll_once, AsyncComputeTaskPool, Task};
use engine::render::Gfx;
use engine::world::flesh::Flesh;
use engine::world::scene::Scene;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// How long the level must stay unchanged before sculpting starts, so a drag
/// doesn't sculpt every step of the way.
const SETTLE_SECS: f64 = 0.2;
/// Sculpts kept for creatures no longer in the level (for undo, re-rolls back).
const SPARE: usize = 24;

#[derive(Resource, Default)]
pub struct Sculpts {
    /// Finished sculpts by shape, with the tick they were last shown.
    done: HashMap<u64, (Flesh, u64)>,
    pending: HashMap<u64, Task<Flesh>>,
    /// Full-colour creatures in the view, by shape and baked light.
    shown: HashMap<(u64, u64), Entity>,
    tick: u64,
}

impl Sculpts {
    pub fn is_done(&self, shape: u64) -> bool {
        self.done.contains_key(&shape)
    }

    /// Sculpts still being made for the level.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Show the sculpted ones of `shapes` (the scene's creatures, in order) in full
    /// colour with their light baked from `scene`, and nothing else.
    pub fn show(&mut self, commands: &mut Commands, gfx: &mut Gfx, scene: &Scene, shapes: &[u64]) {
        self.tick += 1;
        let mut want = HashSet::new();
        for shape in shapes {
            let Some((flesh, used)) = self.done.get_mut(shape) else { continue };
            *used = self.tick;
            let lights: Vec<_> = flesh.light_at.iter().map(|&p| scene.light_sh(p)).collect();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            lights.iter().flatten().flatten().for_each(|c| c.to_bits().hash(&mut h));
            let key = (*shape, h.finish());
            if want.insert(key) && !self.shown.contains_key(&key) {
                flesh.lights = lights;
                if let Some(e) = gfx.spawn_flesh(commands, flesh) {
                    self.shown.insert(key, e);
                }
            }
        }
        self.shown.retain(|key, e| {
            let keep = want.contains(key);
            if !keep {
                commands.entity(*e).despawn();
            }
            keep
        });
        // Forget the longest-unused sculpts of creatures that have gone.
        let current: HashSet<u64> = shapes.iter().copied().collect();
        let mut spare: Vec<(u64, u64)> =
            self.done.iter().filter(|(s, _)| !current.contains(s)).map(|(&s, &(_, used))| (used, s)).collect();
        if spare.len() > SPARE {
            spare.sort();
            for (_, s) in &spare[..spare.len() - SPARE] {
                self.done.remove(s);
            }
        }
    }
}

/// Collect finished sculpts, drop those for creatures that have gone, and start
/// the missing ones once the level has settled.
pub fn sculpt(time: Res<Time>, built: Res<crate::view3d::Built>, mut sculpts: ResMut<Sculpts>) {
    let s = &mut *sculpts;
    let mut finished = Vec::new();
    for (&k, task) in &mut s.pending {
        if let Some(f) = block_on(poll_once(task)) {
            finished.push((k, f));
        }
    }
    for (k, f) in finished {
        s.pending.remove(&k);
        s.done.insert(k, (f, s.tick));
    }
    if !built.full_colour {
        s.pending.clear();
        return;
    }
    // Dropping a task cancels it if it hasn't started.
    let wanted: HashSet<u64> = built.shapes.iter().copied().collect();
    s.pending.retain(|k, _| wanted.contains(k));
    if time.elapsed_secs_f64() - built.changed_at < SETTLE_SECS {
        return;
    }
    for (c, &shape) in built.scene.creatures.iter().zip(&built.shapes) {
        if !s.done.contains_key(&shape) && !s.pending.contains_key(&shape) {
            let c = c.clone();
            s.pending.insert(shape, AsyncComputeTaskPool::get().spawn(async move { c.sculpt() }));
        }
    }
}
