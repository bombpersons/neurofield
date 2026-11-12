//! What people wear (`person`): everyday clothes, and every piece of armour the player can
//! put on (`for_item`), so anyone can be dressed in what the player finds.
//!
//! A garment is a set of `Cover`s, each a shell over a stretch of one of the body's parts
//! (both of a pair), a little larger than the part, in a cloth (its texture) and a colour.
//! A cover can run on past its part's end and flare out as it goes, which makes skirts. What
//! a garment covers from end to end is no longer drawn beneath it, and a covered head's hair
//! keeps only what hangs below the cover's rim.

use crate::world::rng::Rng;
use crate::world::tex::{self, Texture, SIZE};
use serde::{Deserialize, Serialize};

/// A part of the body, both of a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bone {
    Pelvis,
    Torso,
    Neck,
    Head,
    UpperArm,
    Forearm,
    Hand,
    Thigh,
    Shin,
    Foot,
}

/// A shell over part of a bone: along it from `from` to `to` (-1 its joint end, 1 its far
/// end; past 1 it goes on along the same line), standing `puff` (a share of the part's
/// radius) off it, and widening by `flare` of the part's widest radius toward `to`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cover {
    pub bone: Bone,
    pub from: f32,
    pub to: f32,
    pub puff: f32,
    pub flare: f32,
}

const fn cover(bone: Bone, from: f32, to: f32, puff: f32) -> Cover {
    Cover { bone, from, to, puff, flare: 0.0 }
}

/// What a garment is made of: which texture it wears.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cloth {
    Weave,
    Knit,
    Stripes,
    Checks,
    Leather,
    Metal,
}

pub const CLOTHS: [Cloth; 6] = [Cloth::Weave, Cloth::Knit, Cloth::Stripes, Cloth::Checks, Cloth::Leather, Cloth::Metal];

impl Cloth {
    /// Its texture's id (see `tex::CLOTH_BASE`).
    pub fn tex(self) -> u8 {
        tex::CLOTH_BASE + CLOTHS.iter().position(|&c| c == self).unwrap() as u8
    }
}

/// Every garment there is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Shirt,
    LongShirt,
    Trousers,
    Shorts,
    Skirt,
    LongSkirt,
    Shoes,
    Boots,
    // Armour the player can find and wear (`for_item`).
    LeatherCap,
    LeatherVest,
    WorkGloves,
    CopperRing,
}

/// Something worn, in its colour (or the garment's own when it has one).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Wear {
    pub kind: Kind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colour: Option<[f32; 3]>,
}

impl Wear {
    pub fn of(kind: Kind) -> Wear {
        Wear { kind, colour: None }
    }
}

/// A garment: its covers, cloth, what it's worn over (higher layers sit outside lower ones)
/// and its own colour.
pub struct Garment {
    pub covers: Vec<Cover>,
    pub cloth: Cloth,
    pub layer: u8,
    pub colour: [f32; 3],
}

const LEATHER: [f32; 3] = [0.55, 0.36, 0.22];

impl Kind {
    pub fn garment(self) -> Garment {
        use Bone::*;
        let (covers, cloth, layer, colour) = match self {
            Kind::Shirt => (vec![cover(Torso, -1.0, 0.95, 0.12), cover(Pelvis, -1.0, -0.2, 0.16), cover(UpperArm, -1.0, 0.3, 0.2)], Cloth::Weave, 0, [0.9, 0.9, 0.85]),
            Kind::LongShirt => (vec![cover(Torso, -1.0, 0.95, 0.12), cover(Pelvis, -1.0, -0.2, 0.16), cover(UpperArm, -1.0, 1.0, 0.2), cover(Forearm, -1.0, 0.75, 0.25)], Cloth::Knit, 0, [0.5, 0.6, 0.8]),
            Kind::Trousers => (vec![cover(Pelvis, -1.0, 1.0, 0.1), cover(Thigh, -1.0, 1.0, 0.12), cover(Shin, -1.0, 0.8, 0.15)], Cloth::Weave, 0, [0.35, 0.3, 0.28]),
            Kind::Shorts => (vec![cover(Pelvis, -1.0, 1.0, 0.1), cover(Thigh, -1.0, 0.1, 0.2)], Cloth::Weave, 0, [0.4, 0.45, 0.6]),
            Kind::Skirt => (vec![cover(Pelvis, -1.0, 1.0, 0.12), Cover { bone: Pelvis, from: -0.2, to: 4.5, puff: 0.15, flare: 0.5 }], Cloth::Checks, 1, [0.8, 0.35, 0.4]),
            Kind::LongSkirt => (vec![cover(Pelvis, -1.0, 1.0, 0.12), Cover { bone: Pelvis, from: -0.2, to: 9.5, puff: 0.15, flare: 0.7 }], Cloth::Weave, 1, [0.4, 0.3, 0.55]),
            Kind::Shoes => (vec![cover(Foot, -1.0, 1.0, 0.15)], Cloth::Leather, 1, [0.3, 0.2, 0.15]),
            Kind::Boots => (vec![cover(Shin, 0.0, 1.0, 0.3), cover(Foot, -1.0, 1.0, 0.2)], Cloth::Leather, 1, [0.35, 0.24, 0.16]),
            // Over the crown, clear of the brows; the hair under it stays hidden.
            Kind::LeatherCap => (vec![cover(Head, 0.32, 1.0, 0.24)], Cloth::Leather, 2, LEATHER),
            Kind::LeatherVest => (vec![cover(Torso, -0.9, 0.95, 0.12), cover(Pelvis, -1.0, -0.2, 0.14)], Cloth::Leather, 2, LEATHER),
            Kind::WorkGloves => (vec![cover(Hand, -1.0, 1.0, 0.3), cover(Forearm, 0.45, 1.0, 0.25)], Cloth::Leather, 2, [0.62, 0.48, 0.3]),
            Kind::CopperRing => (vec![cover(Hand, -0.55, -0.25, 0.25)], Cloth::Metal, 3, [0.85, 0.5, 0.3]),
        };
        Garment { covers, cloth, layer, colour }
    }

    /// The garment for an item the player can wear (by its name, `level::ITEMS`), if it's
    /// worn on the body (not held).
    pub fn for_item(name: &str) -> Option<Kind> {
        match name {
            "leather cap" => Some(Kind::LeatherCap),
            "leather vest" => Some(Kind::LeatherVest),
            "work gloves" => Some(Kind::WorkGloves),
            "copper ring" => Some(Kind::CopperRing),
            _ => None,
        }
    }
}

/// An everyday outfit: a top, something on the legs, something on the feet, in colours
/// that go together.
pub fn outfit(rng: &mut Rng) -> Vec<Wear> {
    let colours = [
        [0.92, 0.9, 0.84],
        [0.85, 0.35, 0.38],
        [0.35, 0.5, 0.8],
        [0.45, 0.65, 0.45],
        [0.95, 0.75, 0.35],
        [0.6, 0.45, 0.75],
        [0.3, 0.3, 0.35],
        [0.55, 0.4, 0.3],
        [0.95, 0.65, 0.7],
        [0.45, 0.75, 0.8],
    ];
    let top = if rng.chance(0.5) { Kind::Shirt } else { Kind::LongShirt };
    let top = Wear { kind: top, colour: Some(rng.pick(&colours)) };
    let legs = rng.pick(&[Kind::Trousers, Kind::Trousers, Kind::Shorts, Kind::Skirt, Kind::LongSkirt]);
    let legs = Wear { kind: legs, colour: Some(rng.pick(&colours)) };
    let feet = Wear { kind: if rng.chance(0.5) { Kind::Shoes } else { Kind::Boots }, colour: None };
    vec![top, legs, feet]
}

/// A cloth's texture, to be tinted with the garment's colour (neutral is near white).
pub fn texture(cloth: Cloth) -> Texture {
    let mut pal = [[0u8; 3]; 16];
    for (i, c) in pal.iter_mut().enumerate() {
        let v = (70.0 + 185.0 * i as f32 / 15.0) as u8;
        *c = [v, v, v];
    }
    let noise = |x: usize, y: usize, s: f32| tex::value_noise(x as f32 / s, y as f32 / s, (SIZE as f32 / s) as i32, 31);
    let idx = (0..SIZE * SIZE)
        .map(|i| {
            let (x, y) = (i % SIZE, i / SIZE);
            let v: f32 = match cloth {
                // A fine weave, threads crossing.
                Cloth::Weave => 0.82 + 0.06 * ((x + y) % 2) as f32 - 0.05 * (x % 4 == 0) as u8 as f32 + 0.06 * noise(x, y, 16.0),
                // Ribs running down in little V's.
                Cloth::Knit => 0.78 + 0.12 * (((x % 4) as f32 - 1.5).abs() / 1.5) * (((y + x % 4 / 2) % 3 != 0) as u8 as f32),
                Cloth::Stripes => (if (y / 8) % 2 == 0 { 0.92 } else { 0.6 }) - 0.04 * ((x + y) % 2) as f32,
                Cloth::Checks => {
                    let (a, b) = ((x / 8) % 2, (y / 8) % 2);
                    0.55 + 0.2 * a as f32 + 0.2 * b as f32
                }
                // Mottled hide, stitched along its edges.
                Cloth::Leather => {
                    let stitch = (y == 3 || y == 60) && x % 4 < 2;
                    if stitch { 1.0 } else { 0.7 + 0.18 * noise(x, y, 8.0) - 0.08 * !(2..=61).contains(&y) as u8 as f32 }
                }
                // Bright, with a gleam across and a row of rivets.
                Cloth::Metal => {
                    let rivet = (y % 32 == 6) && (x % 16 == 7);
                    if rivet { 0.4 } else { 0.75 + 0.25 * (-((y as f32 - 20.0) / 6.0).powi(2)).exp() }
                }
            };
            (v.clamp(0.0, 1.0) * 15.0).round() as u8
        })
        .collect();
    Texture { idx, pal }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_item_worn_on_the_body_has_a_garment() {
        for name in crate::world::level::ITEMS {
            let worn = matches!(name, "leather cap" | "leather vest" | "work gloves" | "copper ring");
            assert_eq!(Kind::for_item(name).is_some(), worn, "{name}");
        }
        for cloth in CLOTHS {
            let t = texture(cloth);
            assert!(t.idx.iter().any(|&i| i != t.idx[0]), "{cloth:?} is flat");
        }
    }
}
