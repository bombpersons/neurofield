//! The Issues window's checks (docs/editor.md, Validation): what would stop the
//! level playing as built, or the network drawing it well.
//!
//! The walk from the player start is solved with the doors: a locked door opens
//! once its key or a lever that moves it has been reached, so keys behind their
//! own door show up as well as places nobody can get to.

use engine::neural::{features, tokens};
use engine::world::level::{EntityKind, Level};
use engine::world::scene::{Camera, Node, Scene, MAX_STEP, STEP, T};
use engine::world::walker::{EYE, H};
use std::collections::{HashMap, HashSet, VecDeque};

const TF: f32 = T as f32;
/// A view that sees more triangles than this is drawn badly: the network keeps
/// only the nearest `MAX_TRIS`.
pub const TOO_MANY: usize = tokens::MAX_TRIS * 3 / 2;
/// Views sampled for the triangle budget, at most.
const BUDGET_SAMPLES: usize = 160;
/// Unreachable places listed one by one; the rest are counted together.
const PLACES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// The level can't be played as meant.
    Error,
    /// Worth a look; it may be what was meant.
    Warning,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    Start,
    Reach,
    Doors,
    Keys,
    Entities,
    Creatures,
    Budget,
}

impl Group {
    pub fn title(self) -> &'static str {
        match self {
            Group::Start => "Player start",
            Group::Reach => "Reaching places",
            Group::Doors => "Doors",
            Group::Keys => "Keys and levers",
            Group::Entities => "Entities",
            Group::Creatures => "Creatures",
            Group::Budget => "Triangle budget",
        }
    }

    /// A short name for agents.
    pub fn name(self) -> &'static str {
        match self {
            Group::Start => "start",
            Group::Reach => "reach",
            Group::Doors => "doors",
            Group::Keys => "keys",
            Group::Entities => "entities",
            Group::Creatures => "creatures",
            Group::Budget => "budget",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Issue {
    pub group: Group,
    pub severity: Severity,
    pub text: String,
    /// Where to look, world units (y down).
    pub at: Option<[f32; 3]>,
    /// The entity it is about.
    pub entity: Option<u32>,
}

fn issue(group: Group, severity: Severity, text: String, at: Option<[f32; 3]>, entity: Option<u32>) -> Issue {
    Issue { group, severity, text, at, entity }
}

/// Every check, sorted by group, errors first within each.
pub fn check_all(l: &Level, scene: &Scene) -> Vec<Issue> {
    let (mut out, budget) = check(l, scene);
    if let Some(b) = budget {
        out.extend(b.run());
    }
    out
}

/// Every check but the triangle budget, sorted by group, errors first within each;
/// and the budget's views to count, which takes longer.
pub fn check(l: &Level, scene: &Scene) -> (Vec<Issue>, Option<Budget>) {
    let mut budget_views = None;
    let mut out = Vec::new();
    let gates = doors(l, scene, &mut out);
    let starts: Vec<_> = l.entities.iter().filter(|e| e.kind == EntityKind::PlayerStart).collect();
    match starts.len() {
        0 => out.push(issue(Group::Start, Severity::Error, "no player start".into(), None, None)),
        1 => {}
        n => out.push(issue(Group::Start, Severity::Warning, format!("{n} player starts; Play uses the first"), Some(starts[1].pos), Some(starts[1].id))),
    }
    for e in &l.entities {
        // Doors sit on cell edges and are checked on their own.
        if !matches!(e.kind, EntityKind::Door { .. }) && scene.node_at(e.pos[0], e.pos[1], e.pos[2]).is_none() {
            let (g, sev) = if e.kind == EntityKind::PlayerStart { (Group::Start, Severity::Error) } else { (Group::Entities, Severity::Error) };
            out.push(issue(g, sev, format!("{} #{} is inside rock", crate::ui::kind_name(&e.kind), e.id), Some(e.pos), Some(e.id)));
        }
        let key = match &e.kind {
            EntityKind::Door { key, .. } | EntityKind::Chest { key, .. } => *key,
            _ => None,
        };
        if let Some(k) = key {
            if !l.entities.iter().any(|o| o.id == k && matches!(o.kind, EntityKind::Key { .. })) {
                out.push(issue(Group::Keys, Severity::Error, format!("{} #{} needs key #{k}, which isn't a key", crate::ui::kind_name(&e.kind), e.id), Some(e.pos), Some(e.id)));
            }
        }
        if let EntityKind::Lever { targets } = &e.kind {
            for t in targets {
                if !l.entities.iter().any(|o| o.id == *t && matches!(o.kind, EntityKind::Door { .. })) {
                    out.push(issue(Group::Keys, Severity::Warning, format!("Lever #{} moves #{t}, which isn't a door", e.id), Some(e.pos), Some(e.id)));
                }
            }
        }
    }
    if let Some(s) = starts.first() {
        if let Some(start) = scene.node_at(s.pos[0], s.pos[1], s.pos[2]) {
            let solved = solve(l, scene, start, &gates);
            keys(l, scene, &solved, start, &gates, &mut out);
            places(l, scene, &solved, &gates, &mut out);
            budget_views = budget(l, scene, &solved.reach);
        }
    }
    creatures(l, scene, &mut out);
    out.sort_by_key(|i| (i.group, i.severity));
    (out, budget_views)
}

/// The middle of span `n`'s floor.
fn floor_at(scene: &Scene, n: Node) -> [f32; 3] {
    let (x, z) = ((n.0 as f32 + 0.5) * TF, (n.1 as f32 + 0.5) * TF);
    [x, scene.node_floor(n, [x, z]), z]
}

/// Each door's edges, both ways round: the step between two spans it stands in.
type Gates = HashMap<(Node, Node), u32>;

/// Check that every door stands between two open cells, and say which edges it shuts.
fn doors(l: &Level, scene: &Scene, out: &mut Vec<Issue>) -> Gates {
    let mut gates = Gates::new();
    for e in &l.entities {
        let EntityKind::Door { width, .. } = e.kind else { continue };
        let [x, y, z] = e.pos;
        let (s, c) = e.yaw.sin_cos();
        let (fwd, along) = ([-s, c], [c, s]);
        let w = width.max(1);
        let mut rock = false;
        let mut stuck = None;
        for i in 0..w {
            let l = (i as f32 - (w - 1) as f32 / 2.0) * TF;
            let (px, pz) = (x + along[0] * l, z + along[1] * l);
            let side = |k: f32| scene.node_at(px + fwd[0] * k * TF / 2.0, y, pz + fwd[1] * k * TF / 2.0);
            let (Some(a), Some(b)) = (side(-1.0), side(1.0)) else {
                rock = true;
                continue;
            };
            if (a.0 - b.0).abs() + (a.1 - b.1).abs() != 1 {
                stuck = Some("isn't on the edge between two cells");
                continue;
            }
            match scene.rise(a, b) {
                Some(r) if r <= MAX_STEP => {}
                Some(_) => stuck = Some("stands over a step too tall to climb"),
                None => stuck = Some("opens onto a ceiling too low to pass under, or a prop"),
            }
            gates.insert((a, b), e.id);
            gates.insert((b, a), e.id);
        }
        if rock {
            out.push(issue(Group::Doors, Severity::Error, format!("Door #{} opens onto rock", e.id), Some(e.pos), Some(e.id)));
        } else if let Some(why) = stuck {
            out.push(issue(Group::Doors, Severity::Error, format!("Door #{} {why}", e.id), Some(e.pos), Some(e.id)));
        }
    }
    gates
}

/// The walk from the start once every door that can be opened has been.
struct Solved {
    /// Every span reached, with the one it was reached from.
    reach: HashMap<Node, Node>,
    /// Doors still shut at the end.
    shut: HashSet<u32>,
}

/// Spans reachable from `start` without passing the doors in `shut`, with their parents.
fn walk(scene: &Scene, start: Node, gates: &Gates, shut: &HashSet<u32>) -> HashMap<Node, Node> {
    let mut seen = HashMap::from([(start, start)]);
    let mut queue = VecDeque::from([start]);
    while let Some(c) = queue.pop_front() {
        for m in scene.neighbours(c) {
            if gates.get(&(c, m)).is_some_and(|d| shut.contains(d)) || seen.contains_key(&m) {
                continue;
            }
            seen.insert(m, c);
            queue.push_back(m);
        }
    }
    seen
}

/// Can someone in `reach` get to the entity at `pos`? Props, chests and creatures
/// block their own span, so standing next to it counts.
fn reached(scene: &Scene, reach: &HashMap<Node, Node>, pos: [f32; 3]) -> bool {
    let Some(n) = scene.node_at(pos[0], pos[1], pos[2]) else { return false };
    if reach.contains_key(&n) {
        return true;
    }
    let floor = scene.node_floor(n, [pos[0], pos[2]]);
    [(1, 0), (-1, 0), (0, 1), (0, -1)].into_iter().any(|(dx, dz)| {
        (0..scene.spans(n.0 + dx, n.1 + dz).len()).any(|k| {
            let m = (n.0 + dx, n.1 + dz, k);
            reach.contains_key(&m) && (floor_at(scene, m)[1] - floor).abs() <= MAX_STEP + TF / 2.0
        })
    })
}

/// Doors that start shut and stay so until their key, or a lever that moves them, is reached.
fn locked(l: &Level) -> HashSet<u32> {
    l.entities
        .iter()
        .filter(|e| matches!(e.kind, EntityKind::Door { locked: true, open: false, .. }))
        .map(|e| e.id)
        .collect()
}

fn solve(l: &Level, scene: &Scene, start: Node, gates: &Gates) -> Solved {
    let mut shut = locked(l);
    loop {
        let reach = walk(scene, start, gates, &shut);
        let before = shut.len();
        for e in &l.entities {
            match &e.kind {
                EntityKind::Key { .. } if reached(scene, &reach, e.pos) => {
                    shut.retain(|d| l.entities.iter().find(|o| o.id == *d).is_none_or(|o| !matches!(o.kind, EntityKind::Door { key: Some(k), .. } if k == e.id)));
                }
                EntityKind::Lever { targets } if reached(scene, &reach, e.pos) => shut.retain(|d| !targets.contains(d)),
                _ => {}
            }
        }
        if shut.len() == before {
            return Solved { reach, shut };
        }
    }
}

/// Locked doors and chests that can't be opened.
fn keys(l: &Level, scene: &Scene, solved: &Solved, start: Node, gates: &Gates, out: &mut Vec<Issue>) {
    let key_pos = |k: u32| l.entities.iter().find(|o| o.id == k && matches!(o.kind, EntityKind::Key { .. })).map(|o| o.pos);
    let levers = |d: u32| l.entities.iter().any(|o| matches!(&o.kind, EntityKind::Lever { targets } if targets.contains(&d)));
    for e in &l.entities {
        match &e.kind {
            EntityKind::Door { key, .. } if solved.shut.contains(&e.id) => {
                // Doors nobody can get to are part of somewhere unreachable.
                let at_door = gates.iter().any(|((a, _), d)| *d == e.id && solved.reach.contains_key(a));
                if !at_door {
                    continue;
                }
                let text = match key.and_then(key_pos) {
                    Some(p) => {
                        let mut open = solved.shut.clone();
                        open.remove(&e.id);
                        let k = key.unwrap();
                        if reached(scene, &walk(scene, start, gates, &open), p) {
                            format!("Door #{}'s key #{k} is behind the door itself", e.id)
                        } else {
                            format!("Door #{}'s key #{k} can't be reached", e.id)
                        }
                    }
                    None if key.is_some() => continue,
                    None if levers(e.id) => format!("Door #{} is locked, and the lever that moves it can't be reached", e.id),
                    None => format!("Door #{} is locked with no key and no lever", e.id),
                };
                out.push(issue(Group::Keys, Severity::Error, text, Some(e.pos), Some(e.id)));
            }
            EntityKind::Chest { locked: true, key: Some(k), .. } if reached(scene, &solved.reach, e.pos) => {
                if key_pos(*k).is_some_and(|p| !reached(scene, &solved.reach, p)) {
                    out.push(issue(Group::Keys, Severity::Error, format!("Chest #{}'s key #{k} can't be reached", e.id), Some(e.pos), Some(e.id)));
                }
            }
            EntityKind::Chest { locked: true, key: None, .. } => {
                out.push(issue(Group::Keys, Severity::Warning, format!("Chest #{} is locked with no key", e.id), Some(e.pos), Some(e.id)));
            }
            _ => {}
        }
    }
}

/// Floor nobody can get to, place by place, and what cuts each off.
fn places(l: &Level, scene: &Scene, solved: &Solved, gates: &Gates, out: &mut Vec<Issue>) {
    // In column order, so each place starts from the same cell every time.
    let lost: std::collections::BTreeSet<Node> = scene.floor_cells().into_iter().filter(|&c| !scene.span(c).blocked && !solved.reach.contains_key(&c)).collect();
    // Places: lost spans that join up on foot.
    let mut done = HashSet::new();
    let mut found: Vec<Vec<Node>> = Vec::new();
    for &c in &lost {
        if !done.insert(c) {
            continue;
        }
        let mut place = vec![c];
        let mut i = 0;
        while i < place.len() {
            for m in scene.neighbours(place[i]) {
                if lost.contains(&m) && done.insert(m) {
                    place.push(m);
                }
            }
            i += 1;
        }
        found.push(place);
    }
    found.sort_by_key(|p| (std::cmp::Reverse(p.len()), p[0]));
    for place in found.iter().take(PLACES) {
        let cells: HashSet<_> = place.iter().copied().collect();
        // What it holds that the player needs.
        let needed = l.entities.iter().find(|e| {
            matches!(e.kind, EntityKind::Key { .. } | EntityKind::Lever { .. } | EntityKind::Chest { .. })
                && scene.node_at(e.pos[0], e.pos[1], e.pos[2]).is_some_and(|n| cells.contains(&n))
        });
        // The lowest step between it and somewhere reached, and any door shut across the way.
        let mut step: Option<(f32, Node, Node)> = None;
        let mut door = None;
        for &c in place {
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                for k in 0..scene.spans(c.0 + dx, c.1 + dz).len() {
                    let m = (c.0 + dx, c.1 + dz, k);
                    if !solved.reach.contains_key(&m) {
                        continue;
                    }
                    if let Some(d) = gates.get(&(c, m)).filter(|d| solved.shut.contains(d)) {
                        door = Some(*d);
                    }
                    if let Some(r) = scene.rise(c, m).filter(|&r| r > MAX_STEP) {
                        if step.is_none_or(|s| r < s.0) {
                            step = Some((r, c, m));
                        }
                    }
                }
            }
        }
        let n = place.len();
        let cells_text = if n == 1 { "1 floor cell".to_string() } else { format!("{n} floor cells") };
        let (text, at) = match (door, step) {
            (Some(d), _) => (format!("{cells_text} behind locked door #{d}"), floor_at(scene, place[0])),
            (None, Some((r, a, b))) => {
                let (pa, pb) = (floor_at(scene, a), floor_at(scene, b));
                let edge = [(pa[0] + pb[0]) / 2.0, pa[1].min(pb[1]), (pa[2] + pb[2]) / 2.0];
                (format!("{cells_text} cut off by a step of {} layers (at most {} climbs)", layers(r), layers(MAX_STEP)), edge)
            }
            (None, None) => (format!("{cells_text} can't be reached from the start"), floor_at(scene, place[0])),
        };
        // Somewhere empty may be meant to be out of reach, as a pit's floor is.
        let (text, severity) = match needed {
            Some(e) => (format!("{text}, with {} #{} in it", crate::ui::kind_name(&e.kind), e.id), Severity::Error),
            None => (text, Severity::Warning),
        };
        out.push(issue(Group::Reach, severity, text, Some(at), needed.map(|e| e.id)));
    }
    if found.len() > PLACES {
        let rest: usize = found[PLACES..].iter().map(|p| p.len()).sum();
        out.push(issue(Group::Reach, Severity::Warning, format!("{} more places ({rest} cells) can't be reached", found.len() - PLACES), None, None));
    }
}

/// A height in layers, briefly.
fn layers(units: f32) -> String {
    let l = units / STEP as f32;
    if (l - l.round()).abs() < 0.05 { format!("{}", l.round()) } else { format!("{l:.1}") }
}

/// Creatures pressed against the ceiling or reaching into the walls.
fn creatures(l: &Level, scene: &Scene, out: &mut Vec<Issue>) {
    // The scene grows a creature for every creature entity that isn't in rock, in order.
    let placed = l.entities.iter().filter(|e| matches!(e.kind, EntityKind::Creature { .. }) && scene.node_at(e.pos[0], e.pos[1], e.pos[2]).is_some());
    for (e, c) in placed.zip(&scene.creatures) {
        let (r, top, bottom) = c.extent();
        let n = scene.node_at(e.pos[0], e.pos[1], e.pos[2]).unwrap();
        let (x, z) = (c.base[0], c.base[2]);
        let floor = scene.node_floor(n, [x, z]);
        if !scene.span(n).sky {
            let ceiling = scene.node_ceiling(n, [x, z]);
            if bottom - top > 0.9 * (floor - ceiling - 24.0) {
                out.push(issue(Group::Creatures, Severity::Warning, format!("Creature #{} is pressed against the ceiling, so it was shrunk to fit", e.id), Some(e.pos), Some(e.id)));
                continue;
            }
        }
        // A ring around it at half its height, a little inside its reach.
        let mid = (top + bottom) / 2.0;
        let pokes = (0..12).any(|i| {
            let a = i as f32 / 12.0 * std::f32::consts::TAU;
            let (px, pz) = (x + a.cos() * r * 0.8, z + a.sin() * r * 0.8);
            match scene.node_at(px, mid, pz) {
                None => true,
                Some(m) => {
                    let s = scene.span(m);
                    let f = scene.node_floor(m, [px, pz]);
                    mid > f || (!s.sky && mid < scene.node_ceiling(m, [px, pz]))
                }
            }
        });
        if pokes {
            out.push(issue(Group::Creatures, Severity::Warning, format!("Creature #{} reaches into the wall", e.id), Some(e.pos), Some(e.id)));
        }
    }
}

/// The triangle budget's views, to count away from the frame: the scene's
/// triangle features and, for each place sampled, a few views from it.
pub struct Budget {
    tris: Vec<f32>,
    places: Vec<([f32; 3], Vec<Vec<f32>>)>,
    fog_far: f32,
}

/// Views from the walkable floor, to see whether any sees far too many triangles.
fn budget(l: &Level, scene: &Scene, reach: &HashMap<Node, Node>) -> Option<Budget> {
    let mut tris = Vec::new();
    features::tri_features(scene, &mut tris);
    if tris.len() / features::TRI_FEATURES <= TOO_MANY {
        return None;
    }
    let mut cells: Vec<Node> = reach.keys().copied().collect();
    cells.sort();
    let stride = cells.len().div_ceil(BUDGET_SAMPLES).max(1);
    let fog = (l.fog.0 * TF, l.fog.1 * TF);
    let places = cells
        .iter()
        .step_by(stride)
        .map(|&n| {
            let p = floor_at(scene, n);
            // Four ways round: the lens is about 80° across, so these see nearly all of it.
            let views = (0..4)
                .map(|i| {
                    let cam = Camera {
                        pos: [p[0] as i32, (p[1] - EYE) as i32, p[2] as i32],
                        yaw: i * 1024,
                        pitch: 0,
                        roll: 0,
                        h: H,
                        fog_near: fog.0 as i32,
                        fog_far: fog.1 as i32,
                        torch: [0.0; 3],
                        torch_radius: 0,
                    };
                    let mut v = Vec::new();
                    features::view_features(scene, 0, &cam, engine::world::view::W, &mut v);
                    v
                })
                .collect();
            (p, views)
        })
        .collect();
    Some(Budget { tris, places, fog_far: l.fog.1 })
}

impl Budget {
    /// Count what each view sees: the worst few places, a few tiles apart.
    pub fn run(self) -> Vec<Issue> {
        let raw = features::TRI_FEATURES;
        let mut over: Vec<(usize, [f32; 3])> = Vec::new();
        let mut near = Vec::new();
        for (p, views) in &self.places {
            // Only triangles within the fog (and a margin for the lens's corners) can count.
            let eye = [p[0] / TF, (p[1] - EYE) / TF, p[2] / TF];
            let reach = self.fog_far * 1.7 + 2.0;
            near.clear();
            for t in self.tris.chunks_exact(raw) {
                let close = (0..3).any(|j| (0..3).map(|c| (t[3 * j + c] - eye[c]).powi(2)).sum::<f32>() < reach * reach);
                if close {
                    near.extend_from_slice(t);
                }
            }
            let most = views.iter().map(|v| tokens::visible(&near, v)).max().unwrap_or(0);
            if most > TOO_MANY {
                over.push((most, *p));
            }
        }
        over.sort_by_key(|o| std::cmp::Reverse(o.0));
        let mut out = Vec::new();
        let mut shown: Vec<[f32; 3]> = Vec::new();
        for (most, p) in &over {
            if shown.len() == 3 {
                break;
            }
            if shown.iter().any(|s| (s[0] - p[0]).hypot(s[2] - p[2]) < 4.0 * TF) {
                continue;
            }
            shown.push(*p);
            let text = format!("a view here sees {most} triangles; the network keeps the nearest {}, so the far end may vanish", tokens::MAX_TRIS);
            out.push(issue(Group::Budget, Severity::Warning, text, Some(*p), None));
        }
        if over.len() > shown.len() {
            let text = format!("{} of {} places sampled see too many triangles", over.len(), self.places.len());
            out.push(issue(Group::Budget, Severity::Warning, text, None, None));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::world::level::{Entity, Hinge};

    /// A corridor four tiles long running +z, with a door across the middle.
    fn corridor(locked: bool, key_at: f32) -> Level {
        let mut l = Level::new("corridor");
        for z in 0..4 {
            l.open(0, z, 0, 12);
        }
        l.entities.push(Entity { id: 1, kind: EntityKind::PlayerStart, pos: [0.5 * TF, 0.0, 0.5 * TF], yaw: 0.0 });
        l.entities.push(Entity {
            id: 2,
            kind: EntityKind::Door { width: 1, height: 8, hinge: Hinge::Left, open: false, locked, key: Some(3) },
            pos: [0.5 * TF, 0.0, 2.0 * TF],
            yaw: 0.0,
        });
        l.entities.push(Entity { id: 3, kind: EntityKind::Key { name: "iron key".into() }, pos: [0.5 * TF, 0.0, key_at * TF], yaw: 0.0 });
        l
    }

    fn texts(l: &Level) -> Vec<String> {
        check_all(l, &l.to_scene()).into_iter().map(|i| i.text).collect()
    }

    #[test]
    fn a_key_in_reach_opens_its_door() {
        assert_eq!(texts(&corridor(true, 0.5)), Vec::<String>::new());
    }

    #[test]
    fn a_key_behind_its_own_door() {
        let t = texts(&corridor(true, 3.5));
        assert!(t.iter().any(|t| t == "Door #2's key #3 is behind the door itself"), "{t:?}");
        // The far side is cut off, with the key in it.
        assert!(t.iter().any(|t| t.starts_with("2 floor cells behind locked door #2, with Key #3")), "{t:?}");
    }

    #[test]
    fn levers_open_locked_doors() {
        let mut l = corridor(true, 3.5);
        l.entities.push(Entity { id: 4, kind: EntityKind::Lever { targets: vec![2] }, pos: [0.5 * TF, 0.0, 1.0 * TF], yaw: 0.0 });
        assert_eq!(texts(&l), Vec::<String>::new());
    }

    #[test]
    fn doors_onto_rock() {
        let mut l = corridor(false, 0.5);
        l.entities[1].yaw = std::f32::consts::FRAC_PI_2;
        let t = texts(&l);
        assert!(t.iter().any(|t| t == "Door #2 opens onto rock"), "{t:?}");
    }

    #[test]
    fn tall_steps_cut_places_off() {
        let mut l = corridor(false, 0.5);
        // Raise the far end two layers: more than a step.
        for z in 3..4 {
            l.close(0, z, 0, 12);
            l.open(0, z, 2, 14);
        }
        let t = texts(&l);
        assert!(t.iter().any(|t| t == "1 floor cell cut off by a step of 2 layers (at most 1.2 climbs)"), "{t:?}");
        // One layer is a step.
        l.close(0, 3, 0, 14);
        l.open(0, 3, 1, 13);
        assert_eq!(texts(&l), Vec::<String>::new());
    }

    #[test]
    fn no_start_and_missing_keys() {
        let mut l = corridor(true, 0.5);
        l.entities.remove(0);
        l.entities.retain(|e| e.id != 3);
        let t = texts(&l);
        assert_eq!(t, vec!["no player start".to_string(), "Door #2 needs key #3, which isn't a key".into()]);
    }

    #[test]
    fn squashed_creatures() {
        let mut l = corridor(false, 0.5);
        // Two layers of headroom over the start's cell's neighbour.
        l.close(0, 1, 2, 12);
        l.entities.push(Entity { id: 9, kind: EntityKind::Creature { kind: engine::world::creature::Kind::Humanoid, size: 1.0, seed: 1 }, pos: [0.5 * TF, 0.0, 1.5 * TF], yaw: 0.0 });
        let t = texts(&l);
        assert!(t.iter().any(|t| t.starts_with("Creature #9 is pressed against the ceiling")), "{t:?}");
    }

    #[test]
    fn big_halls_overflow_the_budget() {
        let mut l = Level::new("hall");
        l.set_box([0, 0, 0], [24, 24, 24], true);
        // A forest of pillars.
        let mut id = 2;
        for x in (1..24).step_by(2) {
            for z in (1..24).step_by(2) {
                let kind = EntityKind::Prop { shape: engine::world::level::PropShape::Pillar, size: [60.0, 0.0, 60.0], tex: 7, tint: [0.8; 3] };
                l.entities.push(Entity { id, kind, pos: [(x as f32 + 0.5) * TF, 0.0, (z as f32 + 0.5) * TF], yaw: 0.3 });
                id += 1;
            }
        }
        l.entities.push(Entity { id: 1, kind: EntityKind::PlayerStart, pos: [12.0 * TF, 0.0, 12.0 * TF], yaw: 0.0 });
        // Fog far enough off to see across it.
        l.fog = (6.0, 30.0);
        let scene = l.to_scene();
        let t = std::time::Instant::now();
        let (mut all, budget) = check(&l, &scene);
        eprintln!("checked in {:?}", t.elapsed());
        let t = std::time::Instant::now();
        all.extend(budget.expect("enough triangles to count").run());
        eprintln!("{} triangles, checked in {:?}", scene.tris.len(), t.elapsed());
        assert!(all.iter().any(|i| i.group == Group::Budget), "{all:?}");
    }

    #[test]
    fn demo_issues() {
        let l = Level::demo();
        let all = check_all(&l, &l.to_scene());
        for i in &all {
            eprintln!("{:?} {:?} {}", i.group, i.severity, i.text);
        }
        assert!(all.iter().all(|i| i.severity == Severity::Warning), "the demo plays");
    }
}
