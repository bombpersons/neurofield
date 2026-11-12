//! Levels joined at their links (`engine::world::link`), walked as one place: the levels a
//! level links to are loaded on a worker thread while the player is in it, drawn where they
//! stand against it, and taken over as the player steps through. What the player leaves
//! behind in a level (doors, chests, levers, keys, dropped things) stays as it was while
//! they're away (`Kept`); its creatures come back. No Bevy here.

use crate::drops::{self, Dropped};
use engine::neural::Level;
use engine::world::link::{LinkAt, Placement};
use engine::world::rng::Rng;
use engine::world::things::ThingsState;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Mutex;

/// A level file loaded: the level, what lies about in it as built, and its links.
pub struct Loaded {
    /// The file, whole (`whole`).
    pub file: PathBuf,
    pub level: Level,
    pub lying: Vec<Dropped>,
    pub links: Vec<LinkAt>,
    /// Which loaded level this is, for its meshes (`meshes`), and the meshes when they
    /// were made as it loaded.
    pub meshes: u64,
    pub prepared: Option<engine::render::Prepared>,
}

/// A new number for a loaded level's meshes.
pub fn meshes() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A level file as a path to compare and save: whole where it can be.
pub fn whole(file: &Path) -> PathBuf {
    file.canonicalize().unwrap_or_else(|_| file.to_path_buf())
}

/// The file link `l` of the level `from` joins (beside it).
pub fn joins(from: &Path, l: &LinkAt) -> PathBuf {
    whole(&from.parent().unwrap_or(Path::new(".")).join(&l.to))
}

/// Load `file`, the walker at `start`, else at its respawn point if `respawn`, else at its
/// player start.
pub fn load(file: &Path, start: Option<([f32; 3], f32)>, respawn: bool, rng: &mut Rng) -> Result<Loaded, String> {
    let level = engine::world::level::Level::load(file)?;
    let start = start.or_else(|| if respawn { level.respawn() } else { None });
    let mut built = Level::from_file(&level, start, rng).map_err(|e| format!("{}: {e}", file.display()))?;
    // The level's own fog, so levels that join fog alike (see `Sim::beyond`).
    built.walker.set_fog(level.fog);
    let lying = drops::of_level(&built.scene, &level);
    Ok(Loaded { file: whole(file), lying, links: level.links(), level: built, meshes: meshes(), prepared: None })
}

/// What stays of a level while the player is away: its doors, chests and levers, the keys
/// taken from it, and what was dropped there.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Kept {
    pub things: ThingsState,
    #[serde(default)]
    pub keys: Vec<(u32, String)>,
    #[serde(default)]
    pub drops: Vec<Dropped>,
}

/// A level beside the one being walked: joined to it by the walked level's link `through`
/// and its own `back`, and placed against it.
pub struct Beside {
    pub loaded: Loaded,
    pub through: u32,
    pub back: u32,
    pub place: Placement,
}

impl Beside {
    /// `loaded` beside the level `here` through its link `l`, if `loaded` has a link to
    /// answer it: named the same and joining `here`.
    pub fn meet(here: &Path, l: &LinkAt, loaded: Loaded) -> Result<Beside, String> {
        let back = loaded.links.iter().find(|o| o.name == l.name && joins(&loaded.file, o) == here);
        let Some(back) = back else {
            return Err(format!("{} has no link {:?} back to {}", loaded.file.display(), l.name, here.display()));
        };
        let (back, place) = (back.id, Placement::between(l, back));
        Ok(Beside { loaded, through: l.id, back, place })
    }

    /// Whether the level beside is of the waking world.
    pub fn plain(&self) -> bool {
        self.loaded.level.scene.sky.is_some()
    }
}

/// Levels being loaded on worker threads, for the walked level's links.
#[derive(Default)]
pub struct Loading {
    /// (The receivers in a mutex only so the game's state can be shared between threads.)
    pending: Vec<(u32, PathBuf, Mutex<Receiver<Result<Loaded, String>>>)>,
}

impl Loading {
    /// Start loading `file` for link `id`, unless it is already; its creatures to be moved
    /// about if `moving` (see `Scene::set_creatures_move`; the waking world's always are).
    pub fn start(&mut self, id: u32, file: PathBuf, seed: u64, moving: bool) {
        if self.pending.iter().any(|p| p.0 == id) {
            return;
        }
        let (tx, rx) = channel();
        let f = file.clone();
        std::thread::spawn(move || {
            let mut rng = Rng::new(seed);
            // Its meshes are made here too, to keep the game from stopping for them.
            let loaded = load(&f, None, false, &mut rng).map(|mut l| {
                let scene = &mut l.level.scene;
                let plain = scene.sky.is_some();
                scene.set_creatures_move(moving || plain);
                l.prepared = Some(engine::render::prepare(&l.level.scene));
                l
            });
            let _ = tx.send(loaded);
        });
        self.pending.push((id, file, Mutex::new(rx)));
    }

    pub fn pending(&self, id: u32) -> bool {
        self.pending.iter().any(|p| p.0 == id)
    }

    /// Forget all but the links `keep`.
    pub fn keep(&mut self, keep: impl Fn(u32, &Path) -> bool) {
        self.pending.retain(|p| keep(p.0, &p.1));
    }

    /// The levels loaded since last asked, by link; `wait` for link `id`'s, if it's loading.
    pub fn done(&mut self, wait: Option<u32>) -> Vec<(u32, Result<Loaded, String>)> {
        let mut out = Vec::new();
        self.pending.retain_mut(|(id, file, rx)| {
            let rx = rx.get_mut().unwrap();
            let got = if wait == Some(*id) { rx.recv().ok() } else { rx.try_recv().ok() };
            match got {
                Some(r) => {
                    out.push((*id, r));
                    false
                }
                // A worker that died (a panic) gives nothing.
                None if matches!(rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    eprintln!("loading {} failed", file.display());
                    false
                }
                None => true,
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two levels written side by side, each with a link on the edge of its room facing out.
    fn pair(dir: &Path) -> (PathBuf, PathBuf) {
        use engine::world::level::{Entity, EntityKind, Level};
        let make = |name: &str, to: &str, link_at: [f32; 2], yaw: f32| {
            let mut l = Level::new(name);
            l.set_box([1, 0, 1], [6, 8, 6], true);
            let c = |v: f32| v * 256.0;
            l.entities.push(Entity { id: 1, kind: EntityKind::PlayerStart, pos: [c(3.5), 0.0, c(3.5)], yaw: 0.0 });
            l.entities.push(Entity { id: 2, kind: EntityKind::Link { to: to.into(), name: "gate".into(), fade: 0.0 }, pos: [c(link_at[0]), 0.0, c(link_at[1])], yaw });
            l
        };
        std::fs::create_dir_all(dir).unwrap();
        let (a, b) = (dir.join("a.ron"), dir.join("b.ron"));
        // a's on the south edge of its room facing south; b's on the west edge facing west.
        make("a", "b.ron", [3.5, 6.0], 0.0).save(&a).unwrap();
        make("b", "a.ron", [1.0, 3.5], std::f32::consts::FRAC_PI_2).save(&b).unwrap();
        (a, b)
    }

    #[test]
    fn a_level_loads_beside_another_and_meets_it_at_the_link() {
        let dir = std::env::temp_dir().join(format!("neurofield-links-test-{}", std::process::id()));
        let (a, b) = pair(&dir);
        let mut rng = Rng::new(1);
        let here = load(&a, None, false, &mut rng).unwrap();
        assert_eq!(here.links.len(), 1);
        let l = &here.links[0];
        assert_eq!(joins(&here.file, l), whole(&b));
        let mut loading = Loading::default();
        loading.start(l.id, joins(&here.file, l), 3, false);
        let got = loading.done(Some(l.id));
        assert_eq!(got.len(), 1);
        let loaded = got.into_iter().next().unwrap().1.unwrap();
        let beside = Beside::meet(&here.file, l, loaded).unwrap();
        // b's link stands where a's does.
        let there = &beside.loaded.links[0];
        let at = beside.place.point(there.pos);
        assert!((0..3).all(|k| (at[k] - l.pos[k]).abs() < 1e-3));
        // Nothing of a is drawn past its edge (the wall there is gone, the strip past it isn't
        // drawn), but the strip can be walked on; the doorway is noted for drawing through.
        assert!(here.level.scene.tris.iter().all(|t| t.v.iter().any(|v| l.past(v.p.map(|c| c as f32)) < 1.0)));
        let s = l.strip();
        assert!(here.level.scene.node_at((s.0 as f32 + 0.5) * 256.0, -10.0, (s.1 as f32 + 0.5) * 256.0).is_some());
        assert_eq!(here.level.scene.doorways.len(), 1);
        // A link that isn't answered is said to be.
        let mut lone = l.clone();
        lone.name = "other".into();
        let mut rng = Rng::new(2);
        assert!(Beside::meet(&here.file, &lone, load(&b, None, false, &mut rng).unwrap()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_town_the_keep_and_the_cellars_meet() {
        let levels = Path::new(env!("CARGO_MANIFEST_DIR")).join("../levels");
        let mut rng = Rng::new(4);
        for name in ["castle.ron", "castle_inside.ron", "dungeon1.ron"] {
            let here = load(&levels.join(name), None, false, &mut rng).unwrap();
            assert!(!here.links.is_empty(), "{name} links nowhere");
            for l in &here.links {
                let there = load(&joins(&here.file, l), None, false, &mut rng).unwrap();
                let b = Beside::meet(&here.file, l, there).unwrap_or_else(|e| panic!("{name}: {e}"));
                // Both sides of the edge are open, at the same floor, and each level's strip is
                // the other's doorway cell.
                let back = b.loaded.links.iter().find(|o| o.id == b.back).unwrap();
                let open = |scene: &engine::world::scene::Scene, c: (i32, i32), y: f32| scene.node_at((c.0 as f32 + 0.5) * 256.0, y - 10.0, (c.1 as f32 + 0.5) * 256.0).is_some();
                assert!(open(&here.level.scene, l.doorway(), l.pos[1]), "{name}'s doorway {} is shut", l.name);
                assert!(open(&b.loaded.level.scene, back.doorway(), back.pos[1]), "the far side of {name}'s {} is shut", l.name);
                let strip = l.strip();
                let theirs = b.place.point([(back.doorway().0 as f32 + 0.5) * 256.0, back.pos[1], (back.doorway().1 as f32 + 0.5) * 256.0]);
                assert_eq!(((theirs[0] / 256.0).floor() as i32, (theirs[2] / 256.0).floor() as i32), strip, "{name}'s {}", l.name);
                // The holes are alike, else one side sees past the other's into what's solid.
                let hole = |scene: &engine::world::scene::Scene, id: u32| scene.doorways.iter().find(|d| d.link.id == id).map(|d| (d.floor, d.ceiling)).unwrap();
                let (mine, far) = (hole(&here.level.scene, l.id), hole(&b.loaded.level.scene, back.id));
                let up = b.place.point([0.0; 3])[1];
                assert!((mine.0 - far.0 - up).abs() < 1.0 && (mine.1 - far.1 - up).abs() < 1.0, "{name}'s {} is {mine:?} high, the far side's {far:?}", l.name);
            }
        }
    }
}
