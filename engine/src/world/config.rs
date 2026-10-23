//! Where the game, editor and datagen find their data: `neurofield.ron` at the
//! repository root (or any directory above the working one). Paths in it are
//! relative to the file. Command-line options override it.
//!
//! ```ron
//! Config(
//!     model: Some("runs/model.safetensors"),
//!     photos: Some("data/skins_gen"),
//!     strange: Some("data/strange"),
//!     level_textures: None,
//!     levels: "levels",
//! )
//! ```

use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const FILE: &str = "neurofield.ron";

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The exported network, .safetensors (see `model.sh`).
    pub model: Option<PathBuf>,
    /// Creature skin library, DIR/<motif>/*.png (see `photo::load`).
    pub photos: Option<PathBuf>,
    /// Level textures at every strangeness step, DIR/<step>/<slot>.png (see `tex::load_strange`).
    pub strange: Option<PathBuf>,
    /// Level textures alone, DIR/<slot>.png, when there is no `strange` (see `tex::load_level`).
    pub level_textures: Option<PathBuf>,
    /// Where hand-built levels live.
    pub levels: Option<PathBuf>,
}

impl Config {
    /// The nearest `neurofield.ron` at or above the working directory, with its
    /// paths made absolute; the defaults if there is none.
    pub fn load() -> Config {
        let Some(path) = find() else { return Config::default() };
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut c: Config = ron::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let dir = path.parent().unwrap_or(Path::new("."));
        for p in [&mut c.model, &mut c.photos, &mut c.strange, &mut c.level_textures, &mut c.levels].into_iter().flatten() {
            *p = dir.join(&*p);
        }
        c
    }

    /// Load the textures and skins it names (an option from the command line, by
    /// the same name, wins).
    pub fn load_textures(&self, opts: &std::collections::HashMap<String, String>) {
        let pick = |k: &str, c: &Option<PathBuf>| opts.get(k).map(PathBuf::from).or_else(|| c.clone());
        if let Some(dir) = pick("photos", &self.photos) {
            crate::world::photo::load(&dir);
        }
        if let Some(dir) = pick("strange", &self.strange) {
            crate::world::tex::load_strange(&dir);
        } else if let Some(dir) = pick("level", &self.level_textures) {
            crate::world::tex::load_level(&dir);
        }
    }
}

fn find() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let p = dir.join(FILE);
        if p.is_file() {
            return Some(p);
        }
        if !dir.pop() {
            return None;
        }
    }
}
