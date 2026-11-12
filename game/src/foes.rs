//! The creatures as foes. Each notices the player when they come near and runs after them,
//! strikes when in reach (some cast spells from further off), and gives up when the player
//! gets well away. A strike or spell that comes from in front of a raised shield is blocked
//! and the creature reels back, at a cost in stamina; one that takes the last of it breaks the
//! guard. Blows that land tire the player too. The sword hurts them, and kills them in
//! whatever way suits their bodies. The motions themselves are `engine::world::animate`'s; nothing here needs
//! Bevy.

use engine::world::animate::{Animator, Cue, Strike};
use engine::world::creature::Creature;
use engine::world::rng::Rng;
use engine::world::walker::EYE;

const TF: f32 = engine::world::scene::T as f32;
/// How near the player has to come to be noticed, and how far off to be given up on.
const SIGHT: f32 = 5.0 * TF;
const LOSE: f32 = 8.0 * TF;
/// Further above or below than this, the player is on another floor and goes unnoticed.
const OTHER_FLOOR: f32 = 1.5 * TF;
/// The player's own girth, added to a creature's reach.
const PLAYER_R: f32 = 0.3 * TF;
/// How far past its reach a strike still lands, the player having stepped back.
const GRAZE: f32 = 0.25 * TF;
/// How far the sword reaches past the player, and how far either side of straight ahead.
pub const SWORD_REACH: f32 = 1.1 * TF;
pub const SWORD_ARC: f32 = 1.0;
/// Casters cast between these distances, now and then.
const CAST_NEAR: f32 = 1.5 * TF;
const CAST_FAR: f32 = 5.0 * TF;
const CAST_CHANCE: f32 = 0.6;
const CASTERS: f32 = 0.3;
/// How far either side of straight ahead (radians) a raised shield blocks.
pub const SHIELD_ARC: f32 = 1.2;

/// What a strike does to the player: health lost, affliction added, how it's told, and the
/// stamina it costs to block.
fn harm(s: Strike) -> (i32, f32, &'static str, f32) {
    match s {
        Strike::Swipe => (4, 0.0, "it claws you", 9.0),
        Strike::Lash => (3, 0.02, "it lashes you", 7.0),
        Strike::Stab => (5, 0.0, "it runs you through", 11.0),
        Strike::Lunge => (4, 0.01, "it bites you", 9.0),
        Strike::Slam => (6, 0.0, "it crushes you", 14.0),
    }
}

/// A spell's harm, and the stamina it costs to block.
const SPELL_HP: i32 = 2;
const SPELL_AFFLICTION: f32 = 0.05;
const SPELL_BLOCK: f32 = 6.0;
/// The stamina a blow or spell that gets through knocks out of the player, and the
/// affliction any blow that lands adds, on top of its own.
pub const HIT_TIRE: f32 = 6.0;
pub const HIT_AFFLICTION: f32 = 0.01;

/// How far `at` is from straight ahead of a player at `feet` facing `yaw` (radians), level.
pub fn off_ahead(feet: [f32; 3], yaw: f32, at: [f32; 3]) -> f32 {
    let d = [at[0] - feet[0], at[2] - feet[2]];
    let far = d[0].hypot(d[1]).max(1.0);
    ((-yaw.sin() * d[0] + yaw.cos() * d[1]) / far).clamp(-1.0, 1.0).acos()
}

struct Foe {
    hp: i32,
    caster: bool,
    hunting: bool,
    /// Seconds until it may strike or cast again.
    cooldown: f32,
}

/// What a tick of fighting did to the player.
#[derive(Default, Debug, PartialEq)]
pub struct Outcome {
    pub hurt: i32,
    pub afflict: f32,
    /// Stamina used up.
    pub tired: f32,
    /// Something was blocked, and the last of it broke the guard.
    pub blocked: bool,
    pub broken: bool,
    pub said: Option<String>,
}

/// The player as the foes see them this tick.
pub struct Target {
    /// Where their feet are, and which way they face (radians, forward (-sin, cos)).
    pub feet: [f32; 3],
    pub yaw: f32,
    /// A shield is up, and the stamina left to hold it.
    pub guarding: bool,
    pub stamina: f32,
    /// A blade has just swept through, with this much attack.
    pub struck: Option<i32>,
    pub defence: i32,
}

#[derive(Default)]
pub struct Foes {
    version: Option<u64>,
    all: Vec<Foe>,
}

impl Foes {
    /// One tick of `dt` seconds for the creatures of level `version`: `cues` are the strikes
    /// and spells their animators said went off since the last tick.
    pub fn tick(&mut self, version: u64, animators: &mut [Animator], creatures: &[Creature], cues: &[(usize, Cue)], player: &Target, dt: f32, rng: &mut Rng) -> Outcome {
        let mut out = Outcome::default();
        if animators.len() != creatures.len() {
            return out;
        }
        if self.version != Some(version) || self.all.len() != creatures.len() {
            self.all = animators
                .iter()
                .map(|a| Foe { hp: (8.0 + 6.0 * a.reach() / TF).round() as i32, caster: rng.chance(CASTERS), hunting: false, cooldown: 0.0 })
                .collect();
            self.version = Some(version);
        }
        let feet = player.feet;
        let chest = [feet[0], feet[1] - 0.6 * EYE, feet[2]];
        let away = |c: &Creature| {
            let d = [feet[0] - c.base[0], feet[1] - c.base[1], feet[2] - c.base[2]];
            (d[0].hypot(d[2]), d[1].abs(), d)
        };
        let ahead = [-player.yaw.sin(), player.yaw.cos()];
        // How far off straight ahead the creature is, radians.
        let off = |far: f32, d: [f32; 3]| (-(d[0] * ahead[0] + d[2] * ahead[1]) / far.max(1.0)).clamp(-1.0, 1.0).acos();
        for &(i, cue) in cues {
            let (Some(a), Some(c)) = (animators.get_mut(i), creatures.get(i)) else { continue };
            let (far, up, d) = away(c);
            let shielded = player.guarding && !out.broken && off(far, d) < SHIELD_ARC;
            let (hp, affliction, what, block) = match cue {
                Cue::Strike(s) if far <= a.reach() + PLAYER_R + GRAZE && up < OTHER_FLOOR => harm(s),
                Cue::Spell if far <= LOSE && up < OTHER_FLOOR => (SPELL_HP, SPELL_AFFLICTION, "a curse washes over you", SPELL_BLOCK),
                _ => continue,
            };
            let blocked = shielded && (cue == Cue::Spell || a.blocked());
            if blocked {
                out.tired += block;
                out.blocked = true;
                out.broken = player.stamina - out.tired <= 0.0;
                out.said = Some(if out.broken { "your guard is broken" } else { "your shield takes the blow" }.to_string());
            } else {
                out.hurt += (hp - player.defence / 2).max(1);
                out.afflict += affliction + HIT_AFFLICTION;
                out.tired += HIT_TIRE;
                out.said = Some(what.to_string());
            }
        }
        // The sword finds the nearest creature in front of the player.
        if let Some(attack) = player.struck {
            let hit = (0..creatures.len())
                .filter(|&i| !animators[i].dead())
                .filter_map(|i| {
                    let (far, up, d) = away(&creatures[i]);
                    let body = (animators[i].reach() - 0.3 * TF).max(0.0) * 0.5;
                    (far < SWORD_REACH + body && up < OTHER_FLOOR && (far < body || off(far, d) < SWORD_ARC)).then_some((i, far))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((i, _)) = hit {
                let foe = &mut self.all[i];
                foe.hp -= attack;
                foe.hunting = true;
                if foe.hp <= 0 {
                    let deaths = animators[i].deaths();
                    animators[i].die(rng.pick(&deaths));
                    out.said = Some("it dies".to_string());
                } else {
                    out.said = Some("you strike it".to_string());
                }
            }
        }
        for (i, (foe, a)) in self.all.iter_mut().zip(animators.iter_mut()).enumerate() {
            if a.dead() {
                continue;
            }
            foe.cooldown -= dt;
            let (far, up, _) = away(&creatures[i]);
            if far < SIGHT && up < OTHER_FLOOR {
                foe.hunting = true;
            } else if far > LOSE || up >= OTHER_FLOOR {
                foe.hunting = false;
            }
            a.chase(foe.hunting.then_some(feet));
            if !foe.hunting || a.busy() || foe.cooldown > 0.0 {
                continue;
            }
            if far <= a.reach() + PLAYER_R {
                let strikes = a.strikes();
                a.attack(rng.pick(&strikes), chest);
                foe.cooldown = rng.fr(1.0, 2.2);
            } else if foe.caster && far > CAST_NEAR + a.reach() && far < CAST_FAR && rng.chance(CAST_CHANCE * dt) {
                a.cast(chest);
                foe.cooldown = rng.fr(4.0, 7.0);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::world::animate::Herd;

    fn scene(seed: u64) -> engine::world::scene::Scene {
        engine::world::walkable_scene(&mut Rng::new(seed), 30).0
    }

    #[test]
    fn foes_come_for_the_player_strike_and_die() {
        let (mut struck, mut killed) = (0, 0);
        for seed in 1..6 {
            let mut scene = scene(seed);
            let mut herd = Herd::default();
            let mut foes = Foes::default();
            let mut rng = Rng::new(seed);
            let mut cues = herd.update(&mut scene, 0, 1.0 / 30.0);
            // The player stands just in front of the first creature, sword out.
            let Some(c) = scene.creatures.first() else { continue };
            let ahead = [-c.yaw.sin(), c.yaw.cos()];
            let feet = [c.base[0] + ahead[0] * 1.2 * TF, c.base[1], c.base[2] + ahead[1] * 1.2 * TF];
            let yaw = (ahead[0]).atan2(-ahead[1]);
            let mut hp = 100;
            for frame in 0..600 {
                let target = Target { feet, yaw, guarding: false, stamina: 40.0, struck: (frame % 20 == 0).then_some(5), defence: 0 };
                let out = foes.tick(0, herd.animators(), &scene.creatures, &cues, &target, 1.0 / 30.0, &mut rng);
                hp -= out.hurt;
                if out.hurt > 0 {
                    struck += 1;
                }
                cues = herd.update(&mut scene, 0, 1.0 / 30.0);
            }
            assert!(hp <= 100);
            killed += herd.animators().first().is_some_and(|a| a.dead()) as usize;
        }
        assert!(killed > 0, "the sword never killed anything");
        assert!(struck > 0, "nothing ever struck the player");
    }

    /// The player stands just in front of the first creature, facing it (or turned away),
    /// shield up with `stamina`, for ten seconds; gives what happened each tick.
    fn stand_guard(seed: u64, facing: bool, stamina: f32) -> Vec<Outcome> {
        let mut scene = scene(seed);
        let mut herd = Herd::default();
        let mut foes = Foes::default();
        let mut rng = Rng::new(seed);
        let mut cues = herd.update(&mut scene, 0, 1.0 / 30.0);
        let Some(c) = scene.creatures.first() else { return Vec::new() };
        let ahead = [-c.yaw.sin(), c.yaw.cos()];
        let feet = [c.base[0] + ahead[0] * 0.8 * TF, c.base[1], c.base[2] + ahead[1] * 0.8 * TF];
        let yaw = ahead[0].atan2(-ahead[1]) + if facing { 0.0 } else { std::f32::consts::PI };
        let mut outs = Vec::new();
        for frame in 0..300 {
            let target = Target { feet, yaw, guarding: true, stamina, struck: None, defence: 0 };
            if frame == 1 {
                foes.all.iter_mut().for_each(|f| f.caster = false);
            }
            outs.push(foes.tick(0, herd.animators(), &scene.creatures, &cues, &target, 1.0 / 30.0, &mut rng));
            cues = herd.update(&mut scene, 0, 1.0 / 30.0);
        }
        outs
    }

    #[test]
    fn a_raised_shield_blocks_what_comes_from_in_front() {
        let mut blocked = 0;
        for seed in 1..6 {
            for out in stand_guard(seed, true, 1000.0) {
                assert_eq!(out.hurt, 0, "seed {seed}: a blow got past the shield");
                assert!(!out.broken);
                blocked += out.blocked as usize;
            }
        }
        assert!(blocked > 0, "nothing was ever blocked");
    }

    #[test]
    fn a_shield_does_not_cover_the_back() {
        let hit = (1..6).flat_map(|seed| stand_guard(seed, false, 1000.0)).filter(|o| o.hurt > 0 && o.tired > 0.0).count();
        assert!(hit > 0, "nothing struck the player's back");
    }

    #[test]
    fn the_last_of_the_stamina_breaks_the_guard() {
        let outs: Vec<Outcome> = (1..6).flat_map(|seed| stand_guard(seed, true, 1.0)).filter(|o| o.blocked).collect();
        assert!(!outs.is_empty());
        assert!(outs.iter().all(|o| o.broken && o.hurt == 0 && o.said.as_deref() == Some("your guard is broken")));
    }
}
