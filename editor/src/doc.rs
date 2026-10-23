//! The level being edited: its file, unsaved changes and undo history.
//!
//! Edits change `level` in place. Once nothing is being dragged or typed into,
//! `settle` records the state before them as one undo step, so a whole drag or
//! a typed number undoes at once.

use engine::world::level::Level;
use std::path::{Path, PathBuf};

const MAX_UNDO: usize = 200;

pub struct Doc {
    pub level: Level,
    pub path: Option<PathBuf>,
    /// The level as last saved (or loaded), to tell whether there are unsaved changes.
    saved: Level,
    /// The level as of the last undo step.
    settled: Level,
    undo: Vec<Level>,
    redo: Vec<Level>,
    /// Bumped on every change, so views know to rebuild.
    pub version: u64,
    seen: Level,
}

impl Doc {
    pub fn new(level: Level, path: Option<PathBuf>) -> Doc {
        Doc {
            saved: level.clone(),
            settled: level.clone(),
            seen: level.clone(),
            level,
            path,
            undo: Vec::new(),
            redo: Vec::new(),
            version: 0,
        }
    }

    pub fn open(path: &Path) -> Result<Doc, String> {
        Ok(Doc::new(Level::load(path)?, Some(path.to_path_buf())))
    }

    pub fn save(&mut self, path: &Path) -> Result<(), String> {
        self.level.save(path).map_err(|e| format!("{}: {e}", path.display()))?;
        self.path = Some(path.to_path_buf());
        self.saved = self.level.clone();
        Ok(())
    }

    pub fn dirty(&self) -> bool {
        self.level != self.saved
    }

    pub fn name(&self) -> String {
        self.path.as_ref().and_then(|p| p.file_name()).map_or("untitled".into(), |n| n.to_string_lossy().into())
    }

    /// Notice changes made since the last call (bumping `version`), and when
    /// `idle` (no drag or text edit under way) make them one undo step.
    pub fn settle(&mut self, idle: bool) {
        if self.level != self.seen {
            self.seen = self.level.clone();
            self.version += 1;
        }
        if idle && self.level != self.settled {
            self.undo.push(std::mem::replace(&mut self.settled, self.level.clone()));
            if self.undo.len() > MAX_UNDO {
                self.undo.remove(0);
            }
            self.redo.clear();
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty() || self.level != self.settled
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self) {
        // An edit still under way is undone first.
        self.settle(true);
        if let Some(prev) = self.undo.pop() {
            self.redo.push(std::mem::replace(&mut self.level, prev.clone()));
            self.settled = prev;
        }
    }

    pub fn redo(&mut self) {
        if let Some(next) = self.redo.pop() {
            self.undo.push(std::mem::replace(&mut self.level, next.clone()));
            self.settled = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_steps_are_settled_edits() {
        let mut d = Doc::new(Level::new("t"), None);
        d.level.open(0, 0, 0, 4);
        d.settle(false); // dragging
        d.level.open(1, 0, 0, 4);
        d.settle(true); // released: one step
        d.level.open(2, 0, 0, 4);
        d.settle(true);
        assert_eq!(d.level.columns.len(), 3);
        d.undo();
        assert_eq!(d.level.columns.len(), 2);
        d.undo();
        assert_eq!(d.level.columns.len(), 0);
        assert!(!d.dirty());
        d.redo();
        d.redo();
        assert_eq!(d.level.columns.len(), 3);
        d.level.close(2, 0, 0, 4);
        d.settle(true);
        assert!(!d.can_redo());
    }
}
