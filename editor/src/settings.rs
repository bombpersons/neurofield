//! The editor's own settings, kept between sessions: recent files, the panel
//! layout and the view. In `$XDG_CONFIG_HOME/neurofield/editor.ron` (or ~/.config).

use crate::ui::Tab;
use crate::view3d::ViewMode;
use egui_dock::DockState;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Most recent first.
    pub recent: Vec<PathBuf>,
    pub dock: DockState<Tab>,
    pub mode: ViewMode,
    /// Creatures in full colour, as the game draws them.
    pub full_colour: bool,
    /// The agent server (Tools > Agents): whether it runs, its port and token.
    pub agents_on: bool,
    pub agents_port: u16,
    pub agents_token: String,
    /// Whether `save` writes them; not for runs driven by the developer MCP, which
    /// start from the defaults and leave the user's settings alone.
    #[serde(skip)]
    pub persist: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            recent: Vec::new(),
            dock: crate::ui::default_dock(),
            mode: ViewMode::Plain,
            full_colour: true,
            agents_on: false,
            agents_port: 7801,
            agents_token: String::new(),
            persist: true,
        }
    }
}

fn path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("neurofield").join("editor.ron"))
}

impl Settings {
    pub fn load() -> Settings {
        path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| ron::from_str(&t).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let Some(p) = path().filter(|_| self.persist) else { return };
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::new()) {
            let _ = std::fs::write(p, text);
        }
    }

    /// Put `p` first in the recent files.
    pub fn opened(&mut self, p: &std::path::Path) {
        let p = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        self.recent.retain(|r| *r != p);
        self.recent.insert(0, p);
        self.recent.truncate(10);
        self.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_settings_are_saved_again() {
        let text = ron::ser::to_string(&Settings { persist: false, ..Settings::default() }).unwrap();
        let back: Settings = ron::from_str(&text).unwrap();
        assert!(back.persist);
    }
}
