//! Saving and loading. There are two saves: the one the player makes in town by talking to
//! whoever keeps the ledger (`TOWN`), and the one quitting in a dungeon leaves, to carry on
//! from where they stood next time (`RESUME`). Dying goes back to the town's save, the resume
//! going with it, and quitting somewhere of the waking world drops the resume too. They're
//! RON files in the user's data directory (`dir`). No Bevy here.

use crate::drops::Dropped;
use crate::player::Player;
use engine::world::things::ThingsState;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const TOWN: &str = "town.ron";
pub const RESUME: &str = "resume.ron";
/// The town the player comes round in after dying with nothing saved, from where the game
/// is run.
pub const TOWN_LEVEL: &str = "levels/castle.ron";

/// Where the player is and what they have.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Save {
    /// The level file.
    pub level: PathBuf,
    /// Where their feet were and which way they faced (radians); without it they come round
    /// at the level's respawn point, the level as it was built.
    pub at: Option<([f32; 3], f32)>,
    pub player: Player,
    /// How the level stood, with `at`: the keys held (ids and names), its doors, chests and
    /// levers, what lay dropped about, and which creatures were dead.
    #[serde(default)]
    pub keys: Vec<(u32, String)>,
    #[serde(default)]
    pub things: Option<ThingsState>,
    #[serde(default)]
    pub drops: Vec<Dropped>,
    #[serde(default)]
    pub dead: Vec<usize>,
    /// How the other levels the player has been to stand (`links::Kept`), by file.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub kept: std::collections::BTreeMap<PathBuf, crate::links::Kept>,
}

impl Save {
    /// A new player in `level`, at its respawn point.
    pub fn fresh(level: PathBuf) -> Save {
        Save { level, at: None, player: Player::new(), keys: Vec::new(), things: None, drops: Vec::new(), dead: Vec::new(), kept: Default::default() }
    }
}

/// The user's data directory for the game: `$XDG_DATA_HOME/neurofield`, else
/// `~/.local/share/neurofield`, or `%APPDATA%\neurofield` on Windows.
pub fn dir() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        var("APPDATA")?
    } else {
        var("XDG_DATA_HOME").or_else(|| var("HOME").map(|h| h.join(".local/share")))?
    };
    Some(base.join("neurofield"))
}

/// Write `save` as `name` in `dir`, whole or not at all.
pub fn write(dir: &Path, name: &str, save: &Save) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let text = ron::ser::to_string_pretty(save, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?;
    let (path, part) = (dir.join(name), dir.join(format!("{name}.part")));
    std::fs::write(&part, text).map_err(|e| format!("{}: {e}", part.display()))?;
    std::fs::rename(&part, &path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The save `name` in `dir`, if there is one that reads.
pub fn read(dir: &Path, name: &str) -> Option<Save> {
    let path = dir.join(name);
    let text = std::fs::read_to_string(&path).ok()?;
    ron::from_str(&text).map_err(|e| eprintln!("{}: {e}", path.display())).ok()
}

pub fn remove(dir: &Path, name: &str) {
    let path = dir.join(name);
    if let Err(e) = std::fs::remove_file(&path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            eprintln!("{}: {e}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::Item;

    #[test]
    fn a_save_reads_back_as_it_was_written() {
        let dir = std::env::temp_dir().join(format!("neurofield-save-test-{}", std::process::id()));
        let mut save = Save::fresh(PathBuf::from("levels/x.ron"));
        save.at = Some(([1.0, 2.0, 3.0], 0.5));
        save.player.hurt(7);
        save.player.add(Item::Sword, 1);
        save.keys.push((3, "iron key".into()));
        save.things = Some(ThingsState { pulled: vec![4], ..Default::default() });
        save.dead = vec![0, 2];
        assert_eq!(read(&dir, RESUME), None);
        write(&dir, RESUME, &save).unwrap();
        assert_eq!(read(&dir, RESUME), Some(save));
        remove(&dir, RESUME);
        assert_eq!(read(&dir, RESUME), None);
        remove(&dir, RESUME);
        let _ = std::fs::remove_dir(&dir);
    }
}
