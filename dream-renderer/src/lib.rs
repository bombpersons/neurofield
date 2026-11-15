//! DeepDream that grows on surfaces.
//!
//! GoogLeNet (ImageNet, as the original DeepDream) looks at a picture; the gradient of how
//! excited some of its layers are, with respect to the picture, says how each pixel should
//! change to excite them more. A little of that is added each frame to a *dream layer* kept not
//! on the screen but on the surfaces the pixels show: pages of RGB, one per surface patch, laid
//! over what's drawn. So the dream deepens frame by frame, stays where it grew when the view
//! moves, and grows deepest where the view rests.
//!
//! The caller draws its picture as it would anyway and says, for each of its pixels, which page
//! it shows and where on it (`Surface`); everything else runs on the GPU in one submission a
//! frame (`DreamRenderer::frame`): two octaves of dreaming (the picture at its own size every
//! frame, and a quarter of it at `up` times the size, a different quarter each frame), and the
//! frame shown, `up` times the picture's size, the dream sharp over the picture's own pixels.
//!
//! The network is run as compute shaders (dream.wgsl): convolutions as matrix multiplies
//! (im2col for the bigger kernels), max pooling keeping each window's winner, and the backward
//! pass to the input only (no weight gradients): ReLU masks from the kept activations,
//! convolutions with the weights transposed, pooling's gradient to the winners. Weights are
//! torchvision's GoogLeNet with BatchNorm folded in (train/export_googlenet.py in neurofield).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod level;
pub mod nca;
mod profile;
mod shell;
use shell::Shell;
pub use shell::{edges, neighbours, pushed, Edge, Pushed, TETS};
use level::{Feedback, Level, Raster, BANDS, VIEW};
pub use level::{Face, LevelTri, View};

/// A light of the level, for lighting the dreamed height from where it is (`set_lights`): a
/// point light (where it is, in pages, and its radius: it gives 1 / (1 + (distance / radius)^2)
/// of its colour) or, with `radius` 0, a light from far off in direction `pos` (towards it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    pub pos: [f32; 3],
    pub colour: [f32; 3],
    pub radius: f32,
}

/// The most of the level's lights `set_lights` keeps (the nearest).
pub const MAX_LIGHTS: usize = 16;
/// Floats of a light in `mem`, and of each pixel's point and side (dream.wgsl LIT).
const LIGHT: usize = 8;
const LIT: u32 = 8;
/// Floats at the start of `mem` for dream.wgsl's globals (`light_on`).
const GLOBALS: usize = 64;

pub type Result<T> = std::result::Result<T, String>;

/// One pixel of the caller's picture: the page it shows (`NO_PAGE` for none: nothing dreams
/// there), where on the page (0..1 each way) and how that changes a pixel right and a pixel down,
/// how strongly the dream shows there (0..1: fog, say), and how far along the page the line of
/// sight moves for each unit it rises off the surface towards the eye (the point toward the eye,
/// minus the point, along the page's two directions, over its distance off the surface; in pages,
/// so the relief's height is in pages too): for the dream's relief. And where in the picture
/// that point of the surface is drawn (its pixels, a pixel's middle at +0.5: the pixel's own
/// middle, unless the surface shown is pushed out from it), and how far it is from the eye.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Surface {
    pub page: f32,
    pub uv: [f32; 2],
    pub duv_dx: [f32; 2],
    pub duv_dy: [f32; 2],
    pub strength: f32,
    pub slope: [f32; 2],
    pub pic: [f32; 2],
    pub dist: f32,
}

pub const NO_PAGE: f32 = -1.0;

/// What a shown pixel's line of sight met in the shell (`DreamRenderer::read_shell`): the
/// level's triangle (index + 1, 0 for nothing), the point on it the hit maps to (its foot), the
/// hit's height above the triangle's plane, the hit's view depth and its face's page (u32::MAX
/// for none).
#[derive(Clone, Copy, Debug)]
pub struct ShellHit {
    pub tri: u32,
    pub foot: [f32; 3],
    pub height: f32,
    pub depth: f32,
    pub page: u32,
}

/// Where one pixel's point of the picture was in the last picture (for `frame_reprojected`), in
/// its pixels (0..width, 0..height, a pixel's middle at +0.5), and how that changes a pixel right
/// and a pixel down; `Motion::NONE` where it wasn't seen (off the screen, or hidden).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Motion {
    pub from: [f32; 2],
    pub dfrom_dx: [f32; 2],
    pub dfrom_dy: [f32; 2],
}

impl Motion {
    pub const NONE: Motion = Motion { from: [-1e9; 2], dfrom_dx: [0.0; 2], dfrom_dy: [0.0; 2] };
}

impl Surface {
    pub const NONE: Surface = Surface { page: NO_PAGE, uv: [0.0; 2], duv_dx: [0.0; 2], duv_dy: [0.0; 2], strength: 0.0, slope: [0.0; 2], pic: [0.0; 2], dist: 0.0 };
}

/// Fixed when the renderer is made.
#[derive(Clone, Debug)]
pub struct Config {
    /// The networks to dream with (`Step::net` picks one).
    pub nets: Vec<NetConfig>,
    /// The widest picture and its height, in pixels.
    pub max_width: u32,
    pub height: u32,
    /// The fine octave's and the shown frame's scale.
    pub up: u32,
    /// Texels along a page's side.
    pub page_res: u32,
    pub max_pages: u32,
    /// Pages that can be dreamed on in one frame (each needs its own accumulator).
    pub max_touched: u32,
    /// The level drawn (`set_level`) is cut into triangles no bigger than this a side (in
    /// pages), to bulge where the dream's height pushes it out.
    pub spacing: f32,
}

/// A network to dream with.
#[derive(Clone, Debug)]
pub struct NetConfig {
    /// GoogLeNet's weights (train/export_googlenet.py) or a ResNet-50's (train/export_resnet.py).
    pub weights: PathBuf,
    /// The layers each octave dreams with, several summed, each counting alike: GoogLeNet's
    /// `conv1`, ..., `inception4c` (or just `4c`), ...; ResNet's `conv1`, `maxpool`,
    /// `layer1.0`, ..., `layer3.5` (or `layer3`, `l3`: its last block), .... None: `4c` or `l3`.
    pub layers: Vec<String>,
    pub fine_layers: Vec<String>,
}

impl NetConfig {
    pub fn new(weights: impl Into<PathBuf>) -> NetConfig {
        NetConfig { weights: weights.into(), layers: Vec::new(), fine_layers: Vec::new() }
    }
}

impl Config {
    pub fn new(weights: impl Into<PathBuf>) -> Config {
        Config {
            nets: vec![NetConfig::new(weights)],
            max_width: 448,
            height: 240,
            up: 3,
            page_res: 256,
            max_pages: 512,
            max_touched: 128,
            spacing: 0.125,
        }
    }
}

/// How a frame dreams and shows.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    /// Colour added a step, at the picture's size and at `up` times it; 0 to not dream.
    pub lr: f32,
    pub lr_fine: f32,
    /// Share of the dream let go a frame.
    pub decay: f32,
    /// The most the dream may add or take away.
    pub limit: f32,
    /// Random roll of the picture before the network, at each octave, in its pixels.
    pub jitter: u32,
    pub jitter_fine: u32,
    /// 0: the dream shown at the picture's own pixels; 1: sharp at the shown frame's.
    pub sharpness: f32,
    /// How much of each step's colour (apart from its brightness) is kept: 0 dreams in light
    /// and shade only, over the picture's own colours; 1 keeps it all (rainbow speckle). The
    /// colour that is kept is spread `colour_blur` times wider than the brightness.
    pub colour: f32,
    pub colour_blur: u32,
    /// The network's gradient blurred by a Gaussian this wide (in its pixels; 0 for none)
    /// before it's added: the robust ResNet's otherwise grows a fine grating.
    pub grad_blur: f32,
    /// How much more of the surface the dream owns each time it dreams there, and the most it
    /// may own: where it owns the surface, the picture's own pattern is smoothed away (to the
    /// picture's local mean, keeping its light and colour) and the dream draws in its place.
    /// 0 lays the dream over the picture as a change only.
    pub take_over: f32,
    pub own_max: f32,
    /// The dream's relief: how far (in pages) the brightest of the dream stands off the surface
    /// and the darkest sinks in, where the dream owns it; each pixel looks along its line of
    /// sight for where it meets the relief. 0 keeps the surface flat. The relief follows the
    /// dream's brightness, blurred (a quarter of the page's texels a side), by `relief_rate`
    /// of the way each frame dreamed, so it rises slowly.
    pub relief: f32,
    pub relief_rate: f32,
    /// While the dream is carried along on the screen (`frame_reprojected`), it's put on the
    /// pages a 1/`remember_every` band of the screen a frame, and what comes into sight starts
    /// from the pages' dream: so it's remembered where it grew. 0 forgets what goes out of sight.
    pub remember_every: u32,
    /// The dreamed height (while carried along): how far (in pages) the dream stands off the
    /// surface, grown a step at a time as the colour is, by `height` a step (0: none), held
    /// within +-`height_max`. The network sees it through a *dream lamp*: its input shaded by
    /// the height's slope, `lamp` of the way (0..1), with `lamp_ambient` of the light coming
    /// from everywhere and the rest from a direction off the surface chosen afresh each step,
    /// so a shape only pays where it would show under any light; every `shade_only_every`th
    /// step (0: never) it sees the shading alone, on grey, and the colour doesn't step.
    pub height: f32,
    pub height_max: f32,
    /// Each step of the height blurred by a Gaussian of `height_blur` of the octave's pixels
    /// (0: not at all), so it raises broad shapes.
    pub height_blur: u32,
    /// The height's step blurred at `height_spread` sizes, `height_blur` doubling each time,
    /// summed with weights growing as the size to the power `height_power`: a slope's push on
    /// the height is a difference of neighbours, all but blind to broad shapes, and this
    /// integrates it back (1 is half way, 2 a Poisson solve), so whole features rise.
    pub height_spread: u32,
    pub height_power: f32,
    /// How much of the height's step is the network's colour step's brightness (each over its
    /// own mean size) rather than the lamp's: the height rises where the dream brightens, in
    /// the shapes it paints (negative: sinks).
    pub height_from_colour: f32,
    /// Share of the height let go each step (where an octave dreams), so it settles where the
    /// network's push holds it instead of filling up to the most.
    pub height_decay: f32,
    /// Each step of the height taken about its own mean over the octave, so the shapes rise
    /// and sink alike and the wall as a whole neither swells out nor sinks in.
    pub height_centred: bool,
    pub lamp: f32,
    pub lamp_ambient: f32,
    pub shade_only_every: u32,
    /// The frame shown shaded by the height as the eye's own light lights it, `show_shade` of
    /// the way; or (`show_height`) the height alone, so lit, in grey.
    pub show_shade: f32,
    /// How far the torch's light on the height wraps round (0: Lambert, hard, black where a
    /// slope turns from it; 1: half-Lambert, soft and matte).
    pub show_wrap: f32,
    /// The height lit by the torch where it's held and the level's lights (`set_lights`) from
    /// where they are, not by a light at the eye.
    pub lights: bool,
    /// How much the dream's colour is lit as the wall's own colour is (by the level's light and
    /// the torch; 0: it glows, its own light).
    pub dream_lit: f32,
    pub show_height: bool,
    /// The level drawn (`encode_view`) pushed out by the height, `displace` of it, subdivided to
    /// bulge; 0 draws the level's own triangles flat, so the height never moves where the
    /// dream's steps land (docs/shell_relief.md).
    pub displace: f32,
    /// The relief shell-mapped (docs/shell_relief.md): each shown pixel's line of sight marched
    /// through prisms `height_max` out and in over the flat level, against the pages' height
    /// times this (0: no march).
    pub shell: f32,
    /// How far in from an edge the wall doesn't go on past (tiles) the shell's height tapers
    /// from nothing (at any corner or open edge), so nothing's cut off there; at least twice the
    /// height, so the slope stays gentle.
    pub shell_taper: f32,
    /// The shell's height smoothed, in levels each twice the last's size (0: as dreamed; 1: over
    /// two height cells, 2: four, ..., fractions between), so the relief rises in broader
    /// shapes than the dream's detail.
    pub shell_smooth: f32,
    /// And smoothed more with the distance, up to two levels more: all of it at 4 / this tiles
    /// off (0: none), where the dream's laid thinner.
    pub shell_far: f32,
    /// The march stops halving where the stretch of the line of sight it's narrowed the
    /// relief to is under this much of a pixel's width there (0: always all 8 halvings, the
    /// finest).
    pub shell_pixel: f32,
    /// The relief leaned towards where it's been looked at from (0: straight off the walls; 1:
    /// all the way along the lines of sight it was last seen along, where it shows no depth from
    /// there): each corner of the level keeps the way it's been looked at from, which turns
    /// towards the eye `shell_follow` of the way a frame while it's in sight (lean.wgsl), so the
    /// relief stands out the way the dream painted it and swings round as the view moves on.
    pub shell_lean: f32,
    pub shell_follow: f32,
    /// The shell's hit shown alone, in grey by its height and lit by its slope (red: none).
    pub show_shell: bool,
    /// The pages grown by neural cellular automata (nca.rs, `DreamRenderer::set_nca`) instead:
    /// their height's depth in tiles (0: none, the dream as before), steps a frame (their speed,
    /// up to nca::MAX_STEPS; 0: still), and how much of the height's range is taken off it (so
    /// walls sink as well as rise). Their colour is the wall's own, lit as the level lights it.
    pub nca: f32,
    pub nca_steps: u32,
    pub nca_lift: f32,
    /// Drawn as the PS1 drew: the automata's cells sampled nearest (chunky texels), and the
    /// frame shown in 15-bit colour with the PS1's 4x4 ordered dither.
    pub psx: bool,
    /// The torch (at the eye)'s glint off the surface: how bright (0: none), how far it
    /// reaches (pages), and how rough the surface is where the dream doesn't own it and where
    /// it does (0..1: wet and glossy towards 0).
    pub glint: f32,
    pub torch_reach: f32,
    pub rough_dry: f32,
    pub rough_wet: f32,
    /// The level drawn (`encode_view`) in its own picture under the dream (textured, lit by
    /// the level's light and the torch, fogged), not the picture given.
    pub own_picture: bool,
    /// How big the dream's things are: the coarse octave sees the picture `scale` times its
    /// size (0.25..1.5: smaller sees it smaller, so dreams bigger things), the fine one
    /// `fine_scale` times (1..`Config::up`).
    pub scale: f32,
    pub fine_scale: f32,
    /// Which of `Config::nets` dreams.
    pub net: usize,
    /// The ImageNet class the network dreams towards (its logit's gradient; < 0: its layers,
    /// as `NetConfig::layers`), where it has a classifier (the robust ResNet's): recognisable
    /// things, a dog, a sliding door, brain coral, rather than strokes and patterns.
    pub class: i32,
    /// The coarse octave dreams every `coarse_every` frames; the fine one a 1/`fine_split` by
    /// 1/`fine_split` piece of the frame at `up` times the picture's size (a different piece
    /// each time) every `fine_every` frames, `fine_offset` frames after the coarse one (so
    /// that, both every 2, they take turns).
    pub coarse_every: u32,
    pub fine_split: u32,
    pub fine_every: u32,
    pub fine_offset: u32,
}

impl Default for Step {
    fn default() -> Step {
        Step { lr: 0.012, lr_fine: 0.008, decay: 0.002, limit: 0.6, jitter: 4, jitter_fine: 6, sharpness: 1.0, colour: 0.25, colour_blur: 3, grad_blur: 0.0, take_over: 0.01, own_max: 1.0, relief: 0.0, relief_rate: 0.05, remember_every: 4, height: 0.0, height_max: 0.15, height_blur: 3, height_spread: 1, height_power: 1.0, height_from_colour: 0.0, height_decay: 0.0, height_centred: true, lamp: 1.0, lamp_ambient: 0.3, shade_only_every: 3, show_shade: 1.0, show_wrap: 0.5, lights: true, dream_lit: 1.0, show_height: false, displace: 0.0, shell: 0.0, shell_taper: 0.15, shell_smooth: 0.0, shell_far: 0.0, shell_pixel: 0.0, shell_lean: 0.0, shell_follow: 0.02, show_shell: false, nca: 0.0, nca_steps: 1, nca_lift: 0.3, psx: false, glint: 0.0, torch_reach: 6.0, rough_dry: 0.6, rough_wet: 0.2, own_picture: true, scale: 1.0, fine_scale: 3.0, net: 0, class: -1, coarse_every: 1, fine_split: 2, fine_every: 1, fine_offset: 0 }
    }
}

const SLOT: u64 = 256;
const MAX_OPS: usize = 2048;
const NONE: u32 = u32::MAX;
const RELU: u32 = 1;
const ACCUMULATE: u32 = 2;
/// dream.wgsl's SCREEN: prepare and show lay the screen layer over the dream.
const SCREEN: u32 = 4;
/// dream.wgsl's SCREEN_ONLY (the screen layer is the whole dream) and FILL (warp starts what
/// wasn't seen from the pages).
const SCREEN_ONLY: u32 = 8;
const FILL: u32 = 16;
/// dream.wgsl's SHOW_HEIGHT, LAMP and HEIGHT_CELL.
const SHOW_HEIGHT: u32 = 32;
const LAMP: u32 = 64;
/// dream.wgsl's CLASS: seed's gradient a class's logit's (its row of the classifier).
const CLASS: u32 = 128;
/// dream.wgsl's PSX: show draws in 15-bit colour, dithered as the PS1 did.
const PSX: u32 = 256;
const HEIGHT_CELL: u32 = 2;
/// The results slot of the height's mean |gradient|.
const HEIGHT_MEAN: u32 = RESULTS - 4;
/// The results slot of the height's step's signed sum (its mean taken off: `height_centred`).
const HEIGHT_SIGNED: u32 = RESULTS - 6;
/// The results slot of the largest |height| on any page (how far the shell's prisms reach).
const HEIGHT_BOUND: u32 = RESULTS - 8;
/// The results slot of how sure the network was of the class at the last coarse octave
/// (dream.wgsl confidence), which the fine octave's steps go by.
const SURE: u32 = RESULTS - 10;
/// dream.wgsl's LAMP_SLOT and LAMP_AUX: floats of a dream lamp's slot, and where its work starts.
const LAMP_SLOT: usize = 12;
const LAMP_AUX: u32 = 24;
const SURFACE: u32 = 13;
const MOTION: u32 = 6;
/// dream.wgsl's RELIEF_CELL: texels of a page a side to each of its relief's.
const RELIEF_CELL: u32 = 4;
const LOOKUP: u32 = 6;
const PARTIALS: u32 = 256;
/// Result slots of the reductions: two each.
const RESULTS: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    m: u32,
    k: u32,
    n: u32,
    flags: u32,
    a_off: u32,
    a_rs: u32,
    a_cs: u32,
    b_off: u32,
    b_rs: u32,
    b_cs: u32,
    c_off: u32,
    c_rs: u32,
    bias_off: u32,
    x0: u32,
    x1: u32,
    x2: u32,
    x3: u32,
    x4: u32,
    x5: u32,
    x6: u32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
    g0: f32,
    g1: f32,
    g2: f32,
    g3: f32,
}

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Matmul,
    Im2col,
    Col2im,
    Maxpool,
    MaxpoolBack,
    ReluBack,
    Clear,
    Reduce,
    ReduceFinish,
    Seed,
    Prepare,
    Scatter,
    ClearAcc,
    Apply,
    ClearPages,
    Show,
    Smooth,
    Raise,
    ScreenStep,
    Bake,
    Warp,
    Store,
    Gbuffer,
    ShadeBack,
    BlurHeight,
    Displace,
    OwnPicture,
    Elementwise,
    BlurGrad,
    HeightMix,
    MatmulBig,
    MatmulSum,
    Confidence,
}
const KERNELS: [&str; 33] = [
    "matmul", "im2col", "col2im", "maxpool", "maxpool_back", "relu_back", "clear", "reduce", "reduce_finish", "seed",
    "prepare", "scatter", "clear_acc", "apply", "clear_pages", "show", "blur_picture", "raise", "screen_step", "bake", "warp", "store", "gbuffer", "shade_back", "blur_height", "displace", "own_picture", "elementwise", "blur_grad", "height_mix", "matmul_big", "matmul_sum", "confidence",
];
/// dream.wgsl's `elementwise` modes.
const EW_INPUT: u32 = 0;
const EW_INPUT_BACK: u32 = 1;
const EW_COPY: u32 = 2;
const EW_ADD: u32 = 3;
const EW_RELU: u32 = 4;
/// dream.wgsl's OWN_PICTURE (gbuffer).
const OWN_PICTURE: u32 = 2;
/// dream.wgsl's MOTION_FROM_LAST: gbuffer works out where each pixel was in the last view.
const MOTION_FROM_LAST: u32 = 1;

/// What a frame is given to say what each pixel shows.
#[derive(Clone, Copy)]
enum Given<'a> {
    /// Nothing: the view hasn't moved.
    Still,
    /// The caller's surfaces, and where each pixel was if the dream's carried along.
    Cpu(&'a [Surface], Option<&'a [Motion]>),
    /// The level drawn from this view, its G-buffer made on the GPU; carried along or not, with
    /// the creatures' depth in `mem` or not.
    Gpu { view: View, carry: bool, creatures: bool },
}
/// dream.wgsl's SMOOTH: the blurred picture's cells.
const SMOOTH: u32 = 8;

/// A convolution's weights in `mem` ([cout][cin][k][k]) and shape.
#[derive(Clone, Copy, Debug)]
struct Conv {
    w: u32,
    b: u32,
    cin: u32,
    cout: u32,
    k: u32,
    stride: u32,
    pad: u32,
}

#[derive(Clone, Copy, Debug)]
struct Inception {
    b1: Conv,
    b2: [Conv; 2],
    b3: [Conv; 2],
    b4: Conv,
}

/// A ResNet bottleneck: 1x1, 3x3 (strided), 1x1 with no ReLU, added to the input (or to its
/// strided 1x1 `downsample`), then ReLU.
#[derive(Clone, Copy, Debug)]
struct Bottleneck {
    c: [Conv; 3],
    down: Option<Conv>,
}

/// A step of a network, as its layers are named.
#[derive(Clone, Copy, Debug)]
enum Block {
    /// A convolution and ReLU.
    Conv(Conv),
    /// A size x size max pool, PyTorch's ceil_mode (GoogLeNet) or not (ResNet).
    Pool { size: u32, stride: u32, pad: u32, ceil: bool },
    Inception(Inception),
    Bottleneck(Bottleneck),
}

/// A network to dream with: its steps by name, where its input's per-channel scale and offset
/// are in `mem` (the network sees rgb * scale + offset), and the layers each octave dreams with
/// (places in `order`).
#[derive(Clone, Debug)]
struct Net {
    order: Vec<(String, Block)>,
    input: u32,
    layers: Vec<usize>,
    fine_layers: Vec<usize>,
    /// Its classifier, where it has one (the robust ResNet's `fc`): each class's weights over
    /// the last block's channels (their mean over the pixels), and where in `mem` the class
    /// dreamed towards has its row (`Step::class`); and where all of it (each class's row) and
    /// its biases are in `mem`, for how sure it is (dream.wgsl confidence).
    classes: Option<Classifier>,
}

impl Net {
    /// The layer named, as its place in `order`. GoogLeNet's `4c` is `inception4c`; ResNet's
    /// `layer3` (or `l3`) is its last block, `layer3.5`.
    fn index(&self, name: &str) -> Result<usize> {
        let mut full = name.to_string();
        if name.len() == 2 && name.as_bytes()[0].is_ascii_digit() {
            full = format!("inception{name}");
        }
        if let Some(n) = name.strip_prefix('l').filter(|n| n.len() == 1) {
            full = format!("layer{n}");
        }
        if let Some(i) = self.order.iter().position(|(l, _)| *l == full) {
            return Ok(i);
        }
        let prefix = format!("{full}.");
        self.order
            .iter()
            .rposition(|(l, _)| l.starts_with(&prefix))
            .ok_or(format!("no layer {name} (one of {:?})", self.order.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>()))
    }

    fn indices(&self, names: &[impl AsRef<str>]) -> Result<Vec<usize>> {
        names.iter().map(|l| self.index(l.as_ref())).collect()
    }
}

#[derive(Clone, Debug)]
struct Classifier {
    w: Vec<f32>,
    row: u32,
    all: u32,
    bias: u32,
    count: u32,
}

/// HWC planes, by their place relative to the activations' (and gradients') start.
#[derive(Clone, Copy, Debug)]
struct T {
    off: u32,
    h: u32,
    w: u32,
    c: u32,
}

#[derive(Clone, Copy, Debug)]
enum Layer {
    /// `out`'s channels from `c0` are conv(inp), after ReLU if `relu` (or added to what's there).
    Conv { conv: Conv, inp: T, out: T, c0: u32, relu: bool },
    Pool { inp: T, out: T, idx: u32, size: u32, stride: u32, pad: u32 },
    /// `out` = `inp`, the same shape (a residual's shortcut).
    Copy { inp: T, out: T },
    /// `t` = ReLU(`t`), in place.
    Relu { t: T },
    /// The input `inp` (rgb, 0..1) as the network sees it, in `out`, by the scale and offset at
    /// `at` in `mem`.
    Input { inp: T, out: T, at: u32 },
}

/// The dispatches of one octave: the network forward, the objective, and back to its input.
struct Plan {
    ops: Vec<(Kernel, Params, [u32; 3])>,
    input: T,
    /// The last block's output, where it dreams towards a class.
    last: Option<T>,
}

/// Where a set of slot tables is in `mem`: each page's slot (-1 for none), each slot's page,
/// how much of it is kept (its decay), and ones.
#[derive(Clone, Copy, Debug, Default)]
struct SlotSet {
    of: u32,
    pages: u32,
    decay: u32,
    ones: u32,
}
const SETS: usize = 6;

/// Rows of RGBA8 copied between buffers and textures are whole 256s of bytes: a row's u32s.
fn padded(width: u32) -> u32 {
    (width * 4).div_ceil(256) * 64
}

/// The same for the level drawn's points (four floats a pixel; dream.wgsl `point_at`).
fn point_rows(width: u32) -> u32 {
    (width * 16).div_ceil(256) * 64
}

/// Where things are in `mem`.
#[derive(Clone, Copy, Debug)]
struct Layout {
    /// The surfaces given, twice (this view's and the last's, which laying the screen layer on
    /// the pages still needs while this frame's are written).
    surfaces: u32,
    surface_size: u32,
    smooth: u32,
    /// Each page's relief, (page_res / RELIEF_CELL) squared floats, and its dreamed height,
    /// (page_res / HEIGHT_CELL) squared.
    relief: u32,
    page_height: u32,
    /// The dream lamp's two slots (one an octave, LAMP_SLOT floats), then its work a net pixel (6).
    shade: u32,
    /// The screen layer, twice (this frame's and the last's, for carrying it along): rgb and
    /// ownership a pixel of the shown frame, at its biggest.
    screen: u32,
    screen_size: u32,
    /// Each pixel's `Motion`.
    motion: u32,
    /// The level drawn: each pixel's point of the level's own surface and its view depth (4
    /// floats, rows padded for texture copies); and the torch's glint's settings.
    points: u32,
    light: u32,
    /// This view and the last (VIEW floats each), the creatures' depth a pixel, and each
    /// pixel's 1/z twice (this view's and the last's), for the G-buffer made on the GPU.
    views: u32,
    /// The lights (the ambient, how many, then each, the torch first), and each pixel's point
    /// of the level and which side of it is seen.
    lights: u32,
    lit: u32,
    creatures: u32,
    nearest: u32,
    /// The pages to clear this frame.
    page_list: u32,
    /// Sets of slot tables: one for the frame's dreaming, the rest for laying the screen on
    /// the pages, a set to each group of up to max_touched pages. (A frame's writes all land
    /// before any of its work, so no two of its uses may share one.)
    sets: [SlotSet; SETS],
    lookup: u32,
    partials: u32,
    results: u32,
    acts: u32,
    grads: u32,
    col: u32,
    gcol: u32,
    /// The parts of a product split along k (`matmul` with x6 > 1), before they're summed.
    split: u32,
    floats: u32,
}

/// Where a frame drawn on the GPU goes: an RGBA8 texture of its own (the frame shown), or a
/// camera's frame and depth (`encode_psx`).
#[derive(Clone, Copy)]
enum Out<'a> {
    Texture(&'a wgpu::Texture),
    Psx { colour: &'a wgpu::TextureView, format: wgpu::TextureFormat, depth: &'a wgpu::TextureView },
}

pub struct DreamRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    mem: wgpu::Buffer,
    atlas: wgpu::Buffer,
    acc: wgpu::Buffer,
    img: wgpu::Buffer,
    /// The level drawn, if given: its buffer (level.rs Level), the faces each band of rows
    /// showed, the rasteriser and the faces-seen reports coming back.
    level: Option<Level>,
    level_buf: wgpu::Buffer,
    seen_buf: wgpu::Buffer,
    raster: Raster,
    feedback: Option<Feedback>,
    levels: u64,
    /// Where what the level drawn left at each pixel goes in `img` (u32s; dream.wgsl `drawn`).
    ids_at: u32,
    /// The shell-mapped relief (shell.rs), and where its targets are copied in img: each
    /// shown pixel's triangle (4 u32s), then its height, light, albedo and texel (shell.rs
    /// `looks`), rows padded.
    shell: Shell,
    shell_at: u32,
    /// The shell's marched this frame (shown from), and the creatures' depth was last given.
    shell_live: bool,
    creatures_drawn: bool,
    /// The pages given to faces, each page's face, the frame each was last seen, and the pages
    /// in each band of rows by the last report.
    page_of: HashMap<Face, u32>,
    owner: Vec<Option<Face>>,
    page_seen: Vec<u64>,
    band_pages: Vec<Vec<usize>>,
    /// The last view drawn and its width, and which of the two 1/z is its.
    last_view: Option<(View, u32)>,
    nearest_flip: bool,
    /// Steps the dream lamp has lit.
    lamp_steps: u64,
    /// The level's ambient light and lights (`set_lights`), and whether this frame's pixels
    /// have their points of the level for lighting them (the G-buffer made on the GPU).
    ambient: [f32; 3],
    lights: Vec<Light>,
    lit_ready: bool,
    readback: wgpu::Buffer,
    cfg: Config,
    at: Layout,
    nets: Vec<Net>,
    /// Each octave's plan by its size, whether it's the fine one, the network and whether it
    /// dreams towards a class.
    plans: HashMap<(u32, u32, bool, usize, bool), Plan>,
    /// The class whose row of each network's classifier is in `mem` (Net::classes).
    class_rows: Vec<Option<u32>>,
    /// Frame count, and the frame each page was last dreamed on.
    frames: u64,
    last: Vec<u64>,
    shown: Vec<u8>,
    /// Where the frame shown starts in `img` (u32s).
    shown_at: u32,
    /// Which of the two surfaces is the view's.
    view_flip: bool,
    /// This frame's work so far, the pages to clear, and the next free set of slot tables.
    ops: Vec<(Kernel, Params, [u32; 3])>,
    clears: Vec<u32>,
    /// The automata growing the pages (`set_nca`), their models' names, the model for walls and
    /// for floors and ceilings, and whether the pages' table needs writing again.
    nca: Option<nca::Nca>,
    nca_names: Vec<String>,
    nca_models: [i32; 2],
    nca_dirty: bool,
    /// The pipeline laying the shell's pixels into a camera's frame (`encode_psx`), by the
    /// frame's format: it, its bind group and its uniform.
    blit: Option<(wgpu::TextureFormat, wgpu::RenderPipeline, wgpu::BindGroup, wgpu::Buffer)>,
    next_set: usize,
    /// Milliseconds the last frame took, start to finish, waiting for the GPU.
    pub last_ms: f32,
    /// GPU time by kernel (`GAME_DREAM_PROFILE=1`, profile.rs).
    profiler: std::sync::Mutex<Option<profile::Profiler>>,
    /// Pages in sight last frame (only `Config::max_touched` of them are dreamed on a frame).
    pub pages_seen: usize,
    rng: u64,
    /// The view the surfaces last given show: its width and every page in it.
    view_width: u32,
    view_pages: Vec<usize>,
    /// Whether the screen layer has anything on it, and how much more of the wall it owns.
    screen_live: bool,
    screen_own: f32,
    /// Which of the two screen layers is this frame's.
    flip: bool,
    /// The screen layer is being carried along, and is the whole dream (not a change to the
    /// pages').
    reprojecting: bool,
    /// The pages in each row of the view.
    view_rows: Vec<Vec<usize>>,
}

fn align(n: u32) -> u32 {
    n.div_ceil(64) * 64
}

/// A dispatch's name for profiling, with its sizes if `by_shape`.
fn op_label(kernel: Kernel, p: &Params, by_shape: bool) -> String {
    let dir = if p.bias_off == NONE { "back" } else { "fwd" };
    match kernel {
        Kernel::Matmul | Kernel::MatmulBig | Kernel::MatmulSum if by_shape => format!("{kernel:?} {dir} {}x{}x{}", p.m, p.k, p.n),
        Kernel::Matmul | Kernel::MatmulBig => format!("matmul {dir}"),
        Kernel::Im2col | Kernel::Col2im | Kernel::Maxpool | Kernel::MaxpoolBack if by_shape => format!("{kernel:?} {}x{} c{} k{} s{}", p.x1, p.x2, p.x0, p.x3, p.x4),
        _ => format!("{kernel:?}"),
    }
}

/// Whether an m x n product goes to `matmul_big` (64 x 64 tiles) rather than `matmul` (32 x 32):
/// when it has enough of the bigger tiles to keep the GPU busy (on an RX 580 the bigger ones
/// are faster from about 200; fewer and the smaller tiles' numbers win).
fn big_matmul(m: u32, n: u32) -> bool {
    m.div_ceil(64) * n.div_ceil(64) >= 200
}

/// Workgroups a product split along k aims for (on an RX 580, 160 to 640 do about as well).
const SPLIT_GROUPS: u32 = 160;
/// Floats for a split product's parts: only products of fewer than SPLIT_GROUPS tiles are split,
/// into at most SPLIT_GROUPS / tiles + 1 parts, so the parts cover fewer than 2 SPLIT_GROUPS tiles.
const SPLIT_ROOM: u32 = 2 * SPLIT_GROUPS * 32 * 32;

/// How many parts to split a product of `tiles` 32 x 32 tiles along its k into, so the GPU
/// has enough workgroups to keep busy, each part at least 64 of k.
fn split_k(tiles: u32, k: u32) -> u32 {
    SPLIT_GROUPS.div_ceil(tiles).min(k / 64).max(1)
}

fn rows(n: u32) -> [u32; 2] {
    [n.min(65535), n.div_ceil(65535)]
}

fn conv_out(n: u32, k: u32, s: u32, pad: u32) -> u32 {
    (n + 2 * pad - k) / s + 1
}

/// PyTorch's ceil_mode size for a k x k pool, whose last window must start inside the input.
fn pool_out(n: u32, k: u32, s: u32, pad: u32) -> u32 {
    let mut o = (n + 2 * pad - k).div_ceil(s) + 1;
    if (o - 1) * s >= n + pad {
        o -= 1;
    }
    o
}

/// Builds an octave's plan: activations laid out from 0 (relative to `Layout::acts`, and the
/// same places relative to `Layout::grads`), with the biggest im2col it needs.
struct Builder {
    at: Layout,
    next: u32,
    col_max: u32,
    layers: Vec<Layer>,
    ops: Vec<(Kernel, Params, [u32; 3])>,
}

impl Builder {
    fn alloc(&mut self, h: u32, w: u32, c: u32) -> T {
        let t = T { off: self.next, h, w, c };
        self.next += align(h * w * c);
        t
    }

    fn act(&self, t: T) -> u32 {
        self.at.acts + t.off
    }

    fn grad(&self, t: T) -> u32 {
        self.at.grads + t.off
    }

    fn push(&mut self, k: Kernel, p: Params, g: [u32; 3]) {
        self.ops.push((k, p, g));
    }

    fn matmul(&mut self, p: Params) {
        if big_matmul(p.m, p.n) {
            let r = rows(p.m.div_ceil(64));
            self.push(Kernel::MatmulBig, p, [p.n.div_ceil(64), r[0], r[1]]);
        } else {
            let r = rows(p.m.div_ceil(32));
            let tiles = p.n.div_ceil(32);
            let splits = split_k(tiles * p.m.div_ceil(32), p.k);
            if splits > 1 {
                assert!(splits * p.m * p.n <= SPLIT_ROOM, "a split product's parts fit their room");
                let p = Params { x5: self.at.split, x6: splits, ..p };
                self.push(Kernel::Matmul, p, [tiles * splits, r[0], r[1]]);
                let r = rows((p.m * p.n).div_ceil(4096));
                self.push(Kernel::MatmulSum, p, [64, r[0], r[1]]);
            } else {
                self.push(Kernel::Matmul, p, [tiles, r[0], r[1]]);
            }
        }
    }

    fn conv(&mut self, conv: Conv, inp: T, out: T, c0: u32) {
        self.conv_with(conv, inp, out, c0, true, false);
    }

    fn conv_with(&mut self, conv: Conv, inp: T, out: T, c0: u32, relu: bool, add: bool) {
        assert_eq!(conv.cin, inp.c, "a convolution reads all of its input's channels");
        self.layers.push(Layer::Conv { conv, inp, out, c0, relu });
        let px = out.h * out.w;
        let flags = if relu { RELU } else { 0 } | if add { ACCUMULATE } else { 0 };
        let base = Params { m: px, n: conv.cout, c_off: self.act(out) + c0, c_rs: out.c, bias_off: conv.b, b_off: conv.w, flags, ..Default::default() };
        if conv.k == 1 && conv.stride == 1 {
            self.matmul(Params { k: conv.cin, a_off: self.act(inp), a_rs: inp.c, a_cs: 1, b_rs: 1, b_cs: conv.cin, ..base });
        } else {
            let big_k = conv.cin * conv.k * conv.k;
            self.col_max = self.col_max.max(px * big_k);
            let r = rows(px);
            let p = Params {
                m: out.h,
                n: out.w,
                a_off: self.act(inp),
                a_rs: inp.c,
                c_off: self.at.col,
                x0: conv.cin,
                x1: inp.h,
                x2: inp.w,
                x3: conv.k,
                x4: conv.stride,
                x5: conv.pad,
                ..Default::default()
            };
            self.push(Kernel::Im2col, p, [big_k.div_ceil(64), r[0], r[1]]);
            self.matmul(Params { k: big_k, a_off: self.at.col, a_rs: big_k, a_cs: 1, b_rs: 1, b_cs: big_k, ..base });
        }
    }

    fn conv_new(&mut self, conv: Conv, inp: T) -> T {
        let out = self.alloc(conv_out(inp.h, conv.k, conv.stride, conv.pad), conv_out(inp.w, conv.k, conv.stride, conv.pad), conv.cout);
        self.conv(conv, inp, out, 0);
        out
    }

    fn pool(&mut self, inp: T, size: u32, stride: u32, pad: u32, ceil: bool) -> T {
        let out_of = |n| if ceil { pool_out(n, size, stride, pad) } else { conv_out(n, size, stride, pad) };
        let out = self.alloc(out_of(inp.h), out_of(inp.w), inp.c);
        let idx = self.alloc(out.h, out.w, out.c).off;
        self.layers.push(Layer::Pool { inp, out, idx, size, stride, pad });
        let r = rows(out.h * out.w);
        let p = Params {
            m: out.h,
            n: out.w,
            a_off: self.act(inp),
            a_rs: inp.c,
            b_off: self.at.acts + idx,
            c_off: self.act(out),
            x0: inp.c,
            x1: inp.h,
            x2: inp.w,
            x3: size,
            x4: stride,
            x5: pad,
            ..Default::default()
        };
        self.push(Kernel::Maxpool, p, [inp.c.div_ceil(64), r[0], r[1]]);
        out
    }

    fn inception(&mut self, b: &Inception, x: T) -> T {
        let (c1, c2, c3, c4) = (b.b1.cout, b.b2[1].cout, b.b3[1].cout, b.b4.cout);
        let out = self.alloc(x.h, x.w, c1 + c2 + c3 + c4);
        self.conv(b.b1, x, out, 0);
        let t2 = self.conv_new(b.b2[0], x);
        self.conv(b.b2[1], t2, out, c1);
        let t3 = self.conv_new(b.b3[0], x);
        self.conv(b.b3[1], t3, out, c1 + c2);
        let pooled = self.pool(x, 3, 1, 1, true);
        self.conv(b.b4, pooled, out, c1 + c2 + c3);
        out
    }

    /// An elementwise step over `n` floats (dream.wgsl `elementwise`).
    fn elementwise(&mut self, mode: u32, a: u32, b: u32, c: u32, n: u32) {
        let r = rows(n.div_ceil(4096));
        self.push(Kernel::Elementwise, Params { m: n, a_off: a, b_off: b, c_off: c, x0: mode, ..Default::default() }, [64, r[0], r[1]]);
    }

    fn input(&mut self, inp: T, at: u32) -> T {
        let out = self.alloc(inp.h, inp.w, inp.c);
        self.layers.push(Layer::Input { inp, out, at });
        self.elementwise(EW_INPUT, self.act(inp), at, self.act(out), inp.h * inp.w * inp.c);
        out
    }

    fn bottleneck(&mut self, b: &Bottleneck, x: T) -> T {
        let t1 = self.conv_new(b.c[0], x);
        let t2 = self.conv_new(b.c[1], t1);
        let out = self.alloc(t2.h, t2.w, b.c[2].cout);
        match b.down {
            Some(d) => self.conv_with(d, x, out, 0, false, false),
            None => {
                self.layers.push(Layer::Copy { inp: x, out });
                self.elementwise(EW_COPY, self.act(x), 0, self.act(out), x.h * x.w * x.c);
            }
        }
        self.conv_with(b.c[2], t2, out, 0, false, true);
        self.layers.push(Layer::Relu { t: out });
        self.elementwise(EW_RELU, 0, 0, self.act(out), out.h * out.w * out.c);
        out
    }

    fn clear(&mut self, at: u32, n: u32) {
        let r = rows(n.div_ceil(4096));
        self.push(Kernel::Clear, Params { m: n, c_off: at, ..Default::default() }, [64, r[0], r[1]]);
    }

    /// Sum of squares (`x0` 0) or of |grad| over covered pixels (1) into results slot `slot`.
    fn reduce(&mut self, mode: u32, a: u32, n: u32, slot: u32) {
        self.push(Kernel::Reduce, Params { m: n, a_off: a, b_off: self.at.lookup, c_off: self.at.partials, x0: mode, ..Default::default() }, [PARTIALS, 1, 1]);
        self.push(Kernel::ReduceFinish, Params { m: PARTIALS, a_off: self.at.partials, c_off: self.at.results + slot * 2, ..Default::default() }, [1, 1, 1]);
    }

    /// Back from the gradients at the layers' outputs to the input's.
    fn backward(&mut self) {
        for layer in self.layers.clone().into_iter().rev() {
            match layer {
                Layer::Conv { conv, inp, out, c0, relu } => {
                    let px = out.h * out.w;
                    let r = rows(px);
                    let g_out = self.grad(out) + c0;
                    if relu {
                        self.push(Kernel::ReluBack, Params { m: px, a_off: self.act(out) + c0, c_off: g_out, a_rs: out.c, x0: conv.cout, ..Default::default() }, [conv.cout.div_ceil(64), r[0], r[1]]);
                    }
                    let base = Params { m: px, k: conv.cout, a_off: g_out, a_rs: out.c, a_cs: 1, b_off: conv.w, bias_off: NONE, ..Default::default() };
                    if conv.k == 1 && conv.stride == 1 {
                        self.matmul(Params { n: conv.cin, b_rs: conv.cin, b_cs: 1, c_off: self.grad(inp), c_rs: inp.c, flags: ACCUMULATE, ..base });
                    } else {
                        let big_k = conv.cin * conv.k * conv.k;
                        self.matmul(Params { n: big_k, b_rs: big_k, b_cs: 1, c_off: self.at.gcol, c_rs: big_k, ..base });
                        let r = rows((inp.h * inp.w * conv.cin).div_ceil(4096));
                        let p = Params {
                            m: out.h,
                            n: out.w,
                            b_off: self.at.gcol,
                            c_off: self.grad(inp),
                            c_rs: inp.c,
                            x0: conv.cin,
                            x1: inp.h,
                            x2: inp.w,
                            x3: conv.k,
                            x4: conv.stride,
                            x5: conv.pad,
                            ..Default::default()
                        };
                        self.push(Kernel::Col2im, p, [64, r[0], r[1]]);
                    }
                }
                Layer::Pool { inp, out, idx, size, stride, pad } => {
                    let r = rows(inp.h * inp.w);
                    let p = Params {
                        m: out.h,
                        n: out.w,
                        a_off: self.grad(out),
                        b_off: self.at.acts + idx,
                        c_off: self.grad(inp),
                        c_rs: inp.c,
                        x0: inp.c,
                        x1: inp.h,
                        x2: inp.w,
                        x3: size,
                        x4: stride,
                        x5: pad,
                        ..Default::default()
                    };
                    self.push(Kernel::MaxpoolBack, p, [inp.c.div_ceil(64), r[0], r[1]]);
                }
                Layer::Copy { inp, out } => {
                    self.elementwise(EW_ADD, self.grad(out), 0, self.grad(inp), out.h * out.w * out.c);
                }
                Layer::Relu { t } => {
                    let px = t.h * t.w;
                    let r = rows(px);
                    self.push(Kernel::ReluBack, Params { m: px, a_off: self.act(t), c_off: self.grad(t), a_rs: t.c, x0: t.c, ..Default::default() }, [t.c.div_ceil(64), r[0], r[1]]);
                }
                Layer::Input { inp, out, at } => {
                    self.elementwise(EW_INPUT_BACK, self.grad(out), at, self.grad(inp), inp.h * inp.w * inp.c);
                }
            }
        }
    }
}

fn take(t: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    t.remove(name).ok_or(format!("no {name} in the weights"))
}

fn read_safetensors(path: &Path) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
    Ok(st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let values = view.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            (name, (view.shape().to_vec(), values))
        })
        .collect())
}

/// A network's weights into `data` (each aligned), and its layers.
fn load_net(cfg: &NetConfig, data: &mut Vec<f32>) -> Result<Net> {
    let mut t = read_safetensors(&cfg.weights)?;
    let mut push = |v: &[f32]| {
        data.resize(align(data.len() as u32) as usize, 0.0);
        let at = data.len() as u32;
        data.extend_from_slice(v);
        at
    };
    let mut conv = |t: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str, stride: u32| -> Result<Conv> {
        let (shape, w) = take(t, &format!("{name}.weight"))?;
        let (_, b) = take(t, &format!("{name}.bias"))?;
        let k = shape[2] as u32;
        Ok(Conv { w: push(&w), b: push(&b), cin: shape[1] as u32, cout: shape[0] as u32, k, stride, pad: k / 2 })
    };
    let mut order = Vec::new();
    let resnet = t.contains_key("layer1.0.conv1.weight");
    if resnet {
        order.push(("conv1".to_string(), Block::Conv(conv(&mut t, "conv1", 2)?)));
        order.push(("maxpool".to_string(), Block::Pool { size: 3, stride: 2, pad: 1, ceil: false }));
        for i in 1..=4 {
            for j in 0.. {
                let n = format!("layer{i}.{j}");
                if !t.contains_key(&format!("{n}.conv1.weight")) {
                    break;
                }
                // torchvision's ResNet v1.5: the 3x3 strides, at each layer's first block but the first's.
                let stride = if j == 0 && i > 1 { 2 } else { 1 };
                let c = [conv(&mut t, &format!("{n}.conv1"), 1)?, conv(&mut t, &format!("{n}.conv2"), stride)?, conv(&mut t, &format!("{n}.conv3"), 1)?];
                let down = if t.contains_key(&format!("{n}.downsample.weight")) { Some(conv(&mut t, &format!("{n}.downsample"), stride)?) } else { None };
                order.push((n, Block::Bottleneck(Bottleneck { c, down })));
            }
        }
    } else {
        order.push(("conv1".to_string(), Block::Conv(conv(&mut t, "conv1", 2)?)));
        order.push(("maxpool1".to_string(), Block::Pool { size: 3, stride: 2, pad: 0, ceil: true }));
        order.push(("conv2".to_string(), Block::Conv(conv(&mut t, "conv2", 1)?)));
        order.push(("conv3".to_string(), Block::Conv(conv(&mut t, "conv3", 1)?)));
        order.push(("maxpool2".to_string(), Block::Pool { size: 3, stride: 2, pad: 0, ceil: true }));
        for name in ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"] {
            let n = format!("inception{name}");
            // Older exports stop at 4e (5a and 5b are only for dreaming towards a class).
            if !t.contains_key(&format!("{n}.branch1.weight")) {
                break;
            }
            if name == "5a" {
                order.push(("maxpool4".to_string(), Block::Pool { size: 2, stride: 2, pad: 0, ceil: true }));
            }
            let block = Inception {
                b1: conv(&mut t, &format!("{n}.branch1"), 1)?,
                b2: [conv(&mut t, &format!("{n}.branch2.0"), 1)?, conv(&mut t, &format!("{n}.branch2.1"), 1)?],
                b3: [conv(&mut t, &format!("{n}.branch3.0"), 1)?, conv(&mut t, &format!("{n}.branch3.1"), 1)?],
                b4: conv(&mut t, &format!("{n}.branch4.1"), 1)?,
            };
            order.push((n, Block::Inception(block)));
            if name == "3b" {
                order.push(("maxpool3".to_string(), Block::Pool { size: 3, stride: 2, pad: 0, ceil: true }));
            }
        }
    }
    drop(conv);
    // The input as the network wants it: GoogLeNet's transform_input of the ImageNet-normalised
    // picture is just 2 rgb - 1.
    let scale = t.remove("input.scale").map(|v| v.1).unwrap_or(vec![2.0; 3]);
    let offset = t.remove("input.offset").map(|v| v.1).unwrap_or(vec![-1.0; 3]);
    let input = push(&[scale, offset].concat());
    let classes = match (t.remove("fc.weight"), t.remove("fc.bias")) {
        (Some((shape, w)), Some((_, b))) if shape[1] <= 2048 && shape[0] <= 1024 => {
            let (row, all, bias) = (push(&vec![0.0; shape[1]]), push(&w), push(&b));
            Some(Classifier { w, row, all, bias, count: shape[0] as u32 })
        }
        _ => None,
    };
    let mut net = Net { order, input, layers: Vec::new(), fine_layers: Vec::new(), classes };
    let default = if resnet { "layer3" } else { "inception4c" };
    let layers = if cfg.layers.is_empty() { vec![default.to_string()] } else { cfg.layers.clone() };
    let fine = if cfg.fine_layers.is_empty() { layers.clone() } else { cfg.fine_layers.clone() };
    net.layers = net.indices(&layers)?;
    net.fine_layers = net.indices(&fine)?;
    if net.layers.len().max(net.fine_layers.len()) as u32 >= RESULTS / 2 {
        return Err("too many layers".into());
    }
    Ok(net)
}

impl DreamRenderer {
    /// The largest storage buffer binding the renderer needs, for the device's limits.
    pub fn max_binding(cfg: &Config) -> u64 {
        let atlas = cfg.max_pages as u64 * (cfg.page_res * cfg.page_res) as u64 * 8;
        let acc = cfg.max_touched as u64 * (cfg.page_res * cfg.page_res) as u64 * 16;
        // The network's floats at the biggest octave, generously.
        let (h, w) = (cfg.height * cfg.up / 2, cfg.max_width * cfg.up / 2);
        let net = (h * w) as u64 * 1200 * 4;
        atlas.max(acc).max(net)
    }

    pub fn new(device: wgpu::Device, queue: wgpu::Queue, cfg: Config) -> Result<DreamRenderer> {
        let mut data: Vec<f32> = vec![0.0; GLOBALS];
        let nets = cfg.nets.iter().map(|n| load_net(n, &mut data)).collect::<Result<Vec<_>>>()?;
        if nets.is_empty() {
            return Err("no network to dream with".into());
        }
        let weights_end = align(data.len() as u32);

        // Where everything goes: the network's room comes from a dry run at the biggest octave.
        let (w0, h0, up) = (cfg.max_width, cfg.height, cfg.up);
        let big = ((h0 * up / 2).max(h0), (w0 * up / 2).max(w0));
        let max_px = big.0 * big.1;
        let mut at = Layout {
            surfaces: weights_end,
            smooth: 0,
            relief: 0,
            page_height: 0,
            shade: 0,
            screen: 0,
            screen_size: 0,
            motion: 0,
            points: 0,
            light: 0,
            views: 0,
            lights: 0,
            lit: 0,
            creatures: 0,
            nearest: 0,
            surface_size: 0,
            page_list: 0,
            sets: [SlotSet::default(); SETS],
            lookup: 0,
            partials: 0,
            results: 0,
            acts: 0,
            grads: 0,
            col: 0,
            gcol: 0,
            split: 0,
            floats: 0,
        };
        at.surface_size = align(w0 * h0 * SURFACE);
        at.smooth = at.surfaces + 2 * at.surface_size;
        at.relief = at.smooth + align(3 * w0.div_ceil(SMOOTH) * h0.div_ceil(SMOOTH));
        at.page_height = at.relief + align(cfg.max_pages * (cfg.page_res / RELIEF_CELL).pow(2));
        at.screen = at.page_height + align(cfg.max_pages * (cfg.page_res / HEIGHT_CELL).pow(2));
        // rgb and ownership, then the height.
        at.screen_size = align(5 * w0 * up * h0 * up);
        at.motion = at.screen + 2 * at.screen_size;
        at.points = at.motion + align(w0 * h0 * MOTION);
        at.light = at.points + align(point_rows(w0) * h0);
        at.views = at.light + align(8);
        at.lights = at.views + align(2 * VIEW);
        at.lit = at.lights + align((4 + LIGHT * (MAX_LIGHTS + 1)) as u32);
        at.creatures = at.lit + align(w0 * h0 * LIT);
        at.nearest = at.creatures + align(w0 * h0);
        at.page_list = at.nearest + 2 * align(w0 * h0);
        let mut next = at.page_list + align(cfg.max_pages);
        for set in &mut at.sets {
            set.of = next;
            set.pages = set.of + align(cfg.max_pages);
            set.decay = set.pages + align(cfg.max_touched);
            set.ones = set.decay + align(cfg.max_touched);
            next = set.ones + align(cfg.max_touched);
        }
        at.lookup = next;
        at.shade = at.lookup + align(max_px * LOOKUP);
        at.partials = at.shade + align(LAMP_AUX + max_px * 6);
        at.results = at.partials + align(PARTIALS * 2);
        at.acts = at.results + align(RESULTS * 2);
        let (mut room, mut col) = (0, 0);
        for net in &nets {
            let (b, _) = Self::build_net(net, at, big.0, big.1, net.order.len() - 1);
            (room, col) = (room.max(align(b.next)), col.max(align(b.col_max)));
        }
        at.grads = at.acts + room;
        at.col = at.grads + room;
        at.gcol = at.col + col;
        at.split = at.gcol + col;
        at.floats = at.split + SPLIT_ROOM;

        let mem_bytes = at.floats as u64 * 4;
        let atlas_bytes = cfg.max_pages as u64 * (cfg.page_res * cfg.page_res) as u64 * 8;
        let acc_bytes = cfg.max_touched as u64 * (cfg.page_res * cfg.page_res) as u64 * 16;
        // The picture given, then the frame shown, each with rows padded for texture copies,
        // then the triangle each pixel shows, likewise.
        let shown_at = padded(w0) * h0;
        let ids_at = shown_at + padded(w0 * up) * h0 * up;
        let shell_at = (ids_at + point_rows(w0) * h0).div_ceil(64) * 64;
        let img_bytes = (shell_at + 2 * point_rows(w0 * up) * h0 * up) as u64 * 4;
        let limit = device.limits().max_storage_buffer_binding_size;
        for (what, b) in [("network", mem_bytes), ("atlas", atlas_bytes), ("accumulators", acc_bytes)] {
            if b > limit {
                return Err(format!("the {what} needs {b} bytes bound, the device allows {limit} (ask for DreamRenderer::max_binding)"));
            }
        }
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let mem = buffer("dream mem", mem_bytes, storage);
        let atlas = buffer("dream atlas", atlas_bytes, storage);
        let acc = buffer("dream accumulators", acc_bytes, storage);
        let img = buffer("dream pictures", img_bytes, storage);
        let params = buffer("dream params", SLOT * MAX_OPS as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let readback = buffer("dream readback", (w0 * up * h0 * up).max(max_px * 3) as u64 * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        queue.write_buffer(&mem, 0, bytemuck::cast_slice(&data));
        // Zeroed pages: a dream that hasn't started.
        let zeros = vec![0u8; (atlas_bytes as usize).min(64 << 20)];
        let mut o = 0;
        while o < atlas_bytes {
            let n = (atlas_bytes - o).min(zeros.len() as u64);
            queue.write_buffer(&atlas, o, &zeros[..n as usize]);
            o += n;
        }

        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let rw = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dream"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                    },
                ),
                entry(1, rw),
                entry(2, rw),
                entry(3, rw),
                entry(4, rw),
                entry(5, rw),
                entry(6, rw),
            ],
        });
        let level_buf = buffer("dream level", 256, storage);
        let seen_buf = buffer("dream faces seen", 256, storage);
        let bind = Self::bind_group(&device, &layout, [&params, &mem, &atlas, &acc, &img, &level_buf, &seen_buf]);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("dream.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("dream.wgsl").into()) });
        let pipelines = KERNELS
            .iter()
            .map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let last = vec![0; cfg.max_pages as usize];
        let raster = Raster::new(&device, w0, h0);
        let profiler = profile::Profiler::new(&device, &queue);
        let mut shell = Shell::new(&device, w0 * up, h0 * up, cfg.max_pages, cfg.page_res / HEIGHT_CELL);
        shell.set_textures(&device, raster.textures.as_ref());
        Ok(DreamRenderer {
            pipelines,
            layout,
            bind,
            params,
            mem,
            atlas,
            acc,
            img,
            level: None,
            level_buf,
            seen_buf,
            raster,
            feedback: None,
            levels: 0,
            ids_at,
            shell,
            shell_at,
            shell_live: false,
            creatures_drawn: false,
            page_of: HashMap::new(),
            owner: vec![None; cfg.max_pages as usize],
            page_seen: vec![0; cfg.max_pages as usize],
            band_pages: vec![Vec::new(); BANDS as usize],
            last_view: None,
            nearest_flip: false,
            lamp_steps: 0,
            ambient: [0.0; 3],
            lights: Vec::new(),
            lit_ready: false,
            device,
            queue,
            readback,
            at,
            class_rows: vec![None; nets.len()],
            nets,
            plans: HashMap::new(),
            frames: 0,
            last,
            shown: Vec::new(),
            shown_at,
            view_flip: false,
            ops: Vec::new(),
            clears: Vec::new(),
            nca: None,
            nca_names: Vec::new(),
            nca_models: [-1, -1],
            nca_dirty: false,
            blit: None,
            next_set: 1,
            last_ms: 0.0,
            profiler: std::sync::Mutex::new(profiler),
            pages_seen: 0,
            rng: 0x9e3779b97f4a7c15,
            view_width: 0,
            view_pages: Vec::new(),
            screen_live: false,
            screen_own: 0.0,
            flip: false,
            reprojecting: false,
            view_rows: Vec::new(),
            cfg,
        })
    }

    /// The network on an h x w input (rgb), up to layer `deepest` of its order, and each layer's
    /// output by its place in the order.
    fn build_net(net: &Net, at: Layout, h: u32, w: u32, deepest: usize) -> (Builder, Vec<T>) {
        let mut b = Builder { at, next: 0, col_max: 0, layers: Vec::new(), ops: Vec::new() };
        let rgb = b.alloc(h, w, 3);
        let mut x = b.input(rgb, net.input);
        let mut outs = Vec::new();
        for (_, block) in net.order.iter().take(deepest + 1) {
            x = match block {
                Block::Conv(c) => b.conv_new(*c, x),
                Block::Pool { size, stride, pad, ceil } => b.pool(x, *size, *stride, *pad, *ceil),
                Block::Inception(i) => b.inception(i, x),
                Block::Bottleneck(k) => b.bottleneck(k, x),
            };
            outs.push(x);
        }
        (b, outs)
    }

    /// One octave's dispatches for an h x w input dreaming with network `net` and its `layers`
    /// (places in its order), or towards the class whose row is in `mem` (`class`: the gradient
    /// of its logit, which at the last block is that row at every pixel, the classifier reading
    /// their mean).
    fn plan(&self, net: usize, h: u32, w: u32, layers: &[usize], class: bool) -> (Plan, Vec<T>) {
        let last = [self.nets[net].order.len() - 1];
        let class_at = self.nets[net].classes.as_ref().map(|c| c.row).filter(|_| class);
        let layers = if class_at.is_some() { &last[..] } else { layers };
        let deepest = *layers.iter().max().unwrap();
        let (mut b, outs) = Self::build_net(&self.nets[net], self.at, h, w, deepest);
        let input = T { off: 0, h, w, c: 3 };
        let room = b.next;
        b.clear(self.at.grads, room);
        for (i, &l) in layers.iter().enumerate() {
            let t = outs[l];
            let n = t.h * t.w * t.c;
            if let Some(at) = class_at {
                let r = rows(n.div_ceil(4096));
                b.push(Kernel::Seed, Params { m: n, a_off: at, k: t.c, flags: CLASS, c_off: b.grad(t), ..Default::default() }, [64, r[0], r[1]]);
                continue;
            }
            b.reduce(0, b.act(t), n, i as u32);
            let r = rows(n.div_ceil(4096));
            let p = Params { m: n, a_off: b.act(t), b_off: self.at.results + i as u32 * 2, c_off: b.grad(t), ..Default::default() };
            b.push(Kernel::Seed, p, [64, r[0], r[1]]);
        }
        b.backward();
        let last = class_at.map(|_| outs[layers[0]]);
        (Plan { ops: b.ops, input, last }, outs)
    }

    fn random(&mut self) -> u32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 32) as u32
    }

    /// Forget the dream on these pages (when the caller gives a page to another surface).
    /// Done first thing next frame.
    pub fn clear_pages(&mut self, pages: &[u32]) -> Result<()> {
        self.clears.extend_from_slice(pages);
        Ok(())
    }

    /// The device and queue it runs on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The view's surfaces in `mem`.
    fn surfaces(&self) -> u32 {
        self.at.surfaces + self.view_flip as u32 * self.at.surface_size
    }

    /// A set of slot tables of its own for laying the screen on the pages, if one's left.
    fn take_set(&mut self) -> Option<SlotSet> {
        let set = self.at.sets.get(self.next_set).copied();
        self.next_set += 1;
        if set.is_none() {
            eprintln!("dream renderer: more pages to lay the screen on than slot tables this frame; some wait");
        }
        set
    }

    /// Dream a frame and show it. `picture` is RGBA8, `width` by `Config::height`; `surfaces`
    /// one a pixel. Returns the frame shown, RGBA8, `up` times the size each way.
    pub fn frame(&mut self, picture: &[u8], width: u32, surfaces: &[Surface], step: &Step) -> Result<&[u8]> {
        self.frame_cpu(picture, width, Given::Cpu(surfaces, None), step)
    }

    /// A frame whose view has moved, with the dream kept on the screen and carried along to
    /// where each pixel's point was (`motion`, one a pixel) instead of laid on the pages: sharp
    /// as the screen, but forgetting what goes out of sight.
    pub fn frame_reprojected(&mut self, picture: &[u8], width: u32, surfaces: &[Surface], motion: &[Motion], step: &Step) -> Result<&[u8]> {
        if motion.len() != surfaces.len() {
            return Err(format!("{} motions for {} surfaces", motion.len(), surfaces.len()));
        }
        self.frame_cpu(picture, width, Given::Cpu(surfaces, Some(motion)), step)
    }

    /// The frame's work with the picture from the CPU and the frame shown back to it.
    fn frame_cpu(&mut self, picture: &[u8], width: u32, given: Given, step: &Step) -> Result<&[u8]> {
        let h = self.cfg.height;
        if width > self.cfg.max_width || picture.len() != (width * h * 4) as usize {
            return Err(format!("a {width} x {h} frame wants {} bytes of picture", width * h * 4));
        }
        let t0 = std::time::Instant::now();
        let sw = width * self.cfg.up;
        let ops = self.run(width, given, step, width, sw)?;
        self.queue.write_buffer(&self.img, 0, picture);
        let shown = self.submit_timed(&ops, Some((&self.img.clone(), self.shown_at as u64 * 4, (sw * h * self.cfg.up) as u64 * 4)), true)?;
        self.shown = shown;
        self.last_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(&self.shown)
    }

    /// The frame's work recorded into `encoder`, all on the GPU, for a caller whose own frame
    /// it's part of (a game engine's on the same device): the picture from `picture` (RGBA8,
    /// at least `width` by `Config::height`) and the frame shown into `out` (RGBA8, at least `up`
    /// times that each way). `surfaces` None for a still view (as `frame_still`); `motion` to
    /// carry the dream along (as `frame_reprojected`). The renderer's queue writes must land
    /// before the encoder's work: submit it after this returns, on the same queue.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        picture: &wgpu::Texture,
        width: u32,
        surfaces: Option<&[Surface]>,
        motion: Option<&[Motion]>,
        step: &Step,
        out: &wgpu::Texture,
    ) -> Result<()> {
        let given = match surfaces {
            Some(s) => Given::Cpu(s, motion),
            None => Given::Still,
        };
        self.encode_given(encoder, Some(picture), width, given, step, Out::Texture(out))
    }

    /// As `encode`, with the G-buffer made on the GPU from the level given (`set_level`) as
    /// `view` sees it (None: the view hasn't moved, as `frame_still`): `creatures` the depth of
    /// whatever's drawn in front of the level a pixel (its view z, f32::MAX where there's
    /// nothing; `width` by `Config::height`), left out of the dream; `carry` to carry the dream
    /// along on the screen.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        picture: &wgpu::Texture,
        width: u32,
        view: Option<&View>,
        creatures: Option<&[f32]>,
        carry: bool,
        step: &Step,
        out: &wgpu::Texture,
    ) -> Result<()> {
        let h = self.cfg.height;
        if self.level.is_none() {
            return Err("no level given".into());
        }
        if let Some(c) = creatures.filter(|c| c.len() != (width * h) as usize) {
            return Err(format!("{} creature depths for a {width} x {h} frame", c.len()));
        }
        let given = match view {
            Some(&view) => {
                if let Some(c) = creatures {
                    self.queue.write_buffer(&self.mem, self.at.creatures as u64 * 4, bytemuck::cast_slice(c));
                }
                if let Some(bits) = self.feedback.as_mut().and_then(|f| f.poll(&self.device)) {
                    self.take_report(&bits);
                }
                Given::Gpu { view, carry, creatures: creatures.is_some() }
            }
            None => Given::Still,
        };
        self.encode_given(encoder, Some(picture), width, given, step, Out::Texture(out))
    }

    /// The level's walls drawn into a camera's own frame, as part of its pass (the PS1
    /// renderer's: `Config::up` 1, the frame `width` by `Config::height`): the shell marched
    /// against `depth` (Depth32Float, reversed: NEAR / z, as the march's own), what's drawn
    /// there already in front of it hiding it and the march's depth left for what's drawn
    /// after, and its pixels shaded as `show` shades them laid into `colour` (of `format`),
    /// the rest left as the camera drew them. As `encode_view` otherwise, with no picture
    /// under the dream.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_psx(&mut self, encoder: &mut wgpu::CommandEncoder, width: u32, view: &View, step: &Step, colour: &wgpu::TextureView, format: wgpu::TextureFormat, depth: &wgpu::TextureView) -> Result<()> {
        if self.level.is_none() {
            return Err("no level given".into());
        }
        if self.cfg.up != 1 {
            return Err(format!("drawn into a camera's frame at {} times its size, not 1", self.cfg.up));
        }
        if step.shell <= 0.0 {
            return Err("drawn into a camera's frame with no shell".into());
        }
        if let Some(bits) = self.feedback.as_mut().and_then(|f| f.poll(&self.device)) {
            self.take_report(&bits);
        }
        self.encode_given(encoder, None, width, Given::Gpu { view: *view, carry: false, creatures: false }, step, Out::Psx { colour, format, depth })
    }

    /// The shell's pixels laid into a camera's frame (psx_blit.wgsl), its pipeline made for the
    /// frame's format when first asked for.
    fn blit(&mut self, encoder: &mut wgpu::CommandEncoder, width: u32, colour: &wgpu::TextureView, format: wgpu::TextureFormat) {
        let h = self.cfg.height;
        if self.blit.as_ref().is_none_or(|b| b.0 != format) {
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("psx_blit.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("psx_blit.wgsl").into()) });
            let uniform = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream blit"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dream blit"),
                entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dream blit"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.img.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: uniform.as_entire_binding() }],
            });
            let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("dream blit"),
                layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("dream blit"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 })),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(format.into())] }),
                multiview_mask: None,
                cache: None,
            });
            self.blit = Some((format, pipeline, bind, uniform));
        }
        let Some((_, pipeline, bind, uniform)) = &self.blit else { return };
        let rows = point_rows(width);
        self.queue.write_buffer(uniform, 0, bytemuck::cast_slice(&[self.shown_at, padded(width), self.shell_at, rows]));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dream blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: colour, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, width as f32, h as f32, 0.0, 1.0);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..3, 0..1);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_given(&mut self, encoder: &mut wgpu::CommandEncoder, picture: Option<&wgpu::Texture>, width: u32, given: Given, step: &Step, out: Out) -> Result<()> {
        let (h, up) = (self.cfg.height, self.cfg.up);
        if width > self.cfg.max_width {
            return Err(format!("a {width} wide frame, {} at most", self.cfg.max_width));
        }
        if let Some(picture) = picture.filter(|p| p.width() < width || p.height() < h) {
            return Err(format!("a {width} x {h} frame from a {} x {} picture", picture.width(), picture.height()));
        }
        if let Out::Texture(out) = out
            && (out.width() < width * up || out.height() < h * up)
        {
            return Err(format!("a {width} x {h} frame into {} x {}", out.width(), out.height()));
        }
        let t0 = std::time::Instant::now();
        if let Some(p) = self.profiler.get_mut().unwrap().as_mut() {
            p.begin(encoder);
        }
        let sw = width * up;
        // The level drawn: from this view, or (still, for its own picture) the last.
        let drawn = match given {
            Given::Gpu { view, .. } => Some(view),
            Given::Still if step.own_picture => self.last_view.filter(|&(_, w)| w == width).map(|(v, _)| v),
            _ => None,
        };
        if let Given::Gpu { creatures, .. } = given {
            self.creatures_drawn = creatures;
        }
        self.shell_live = drawn.is_some() && self.level.is_some() && step.shell > 0.0;
        let ops = self.run(width, given, step, padded(width), padded(sw))?;
        if step.nca > 0.0 {
            self.record_nca(encoder, step);
        }
        let copy = |rows: u32, offset: u64| wgpu::TexelCopyBufferInfo {
            buffer: &self.img,
            layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(rows * 4), rows_per_image: None },
        };
        let size = |width: u32, height: u32| wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        if let Some(picture) = picture {
            encoder.copy_texture_to_buffer(picture.as_image_copy(), copy(padded(width), 0), size(width, h));
        }
        let outside = match out {
            Out::Psx { depth, .. } => Some(depth),
            Out::Texture(_) => None,
        };
        // Marching into a camera's depth, the march's hits are its size.
        let (mw, mh) = if outside.is_some() { (width, h) } else { (self.cfg.max_width * up, h * up) };
        self.shell.fit_marched(&self.device, mw, mh);
        if let (Some(view), Some(level)) = (drawn, self.level.as_ref()) {
            // The level drawn (pushed out by the height first, if it is), then the frame's work.
            let displaced = step.displace > 0.0;
            let n = level.verts;
            let displace = Params { x0: n, x1: level.verts_at, x2: level.moved_at, x4: level.pages_at, c_off: self.at.page_height, k: self.cfg.page_res, f0: step.displace, ..Default::default() };
            let groups = n.div_ceil(64);
            let first = displaced.then_some((Kernel::Displace, displace, [groups.min(65535), groups.div_ceil(65535), 1]));
            let split = first.is_some() as usize;
            let all: Vec<_> = first.into_iter().chain(ops).collect();
            // The shell marched after the dream's steps (the height grown this frame), before
            // `show`.
            let marched = self.shell_live;
            let show = all.iter().rposition(|(k, _, _)| matches!(k, Kernel::Show)).unwrap_or(all.len());
            let splits = if marched { vec![split, show] } else { vec![split] };
            let (sw, sh) = (width * self.cfg.up, h * self.cfg.up);
            self.record_split(encoder, &all, &splits, &mut |enc, part| {
                if part == 2 {
                    let mut stamp = |enc: &mut wgpu::CommandEncoder, label: &str| {
                        if let Some(p) = self.profiler.lock().unwrap().as_mut() {
                            p.stamp(enc, label);
                        }
                    };
                    self.shell.record(&self.queue, enc, &view, self.cfg.up, sw, sh, level, step.height_max * step.shell, step.shell, self.at.results + HEIGHT_BOUND * 2, step.shell_taper, self.at.page_height, self.cfg.page_res, step.shell_smooth, step.shell_far, step.shell_pixel, step.shell_lean, step.shell_follow, outside, &mut stamp);
                    let rows = point_rows(sw);
                    for (k, t) in [&self.shell.tris, &self.shell.looks].into_iter().enumerate() {
                        enc.copy_texture_to_buffer(t.as_image_copy(), copy(rows, (self.shell_at + k as u32 * rows * sh) as u64 * 4), size(sw, sh));
                    }
                    return;
                }
                self.raster.record(&self.queue, enc, &view, width, h, displaced);
                enc.copy_texture_to_buffer(self.raster.ids.as_image_copy(), copy(point_rows(width), self.ids_at as u64 * 4), size(width, h));
                let points = wgpu::TexelCopyBufferInfo {
                    buffer: &self.mem,
                    layout: wgpu::TexelCopyBufferLayout { offset: self.at.points as u64 * 4, bytes_per_row: Some(point_rows(width) * 4), rows_per_image: None },
                };
                enc.copy_texture_to_buffer(self.raster.points.as_image_copy(), points, size(width, h));
                enc.clear_buffer(&self.seen_buf, 0, None);
                if let Some(p) = self.profiler.lock().unwrap().as_mut() {
                    p.stamp(enc, "raster");
                }
            })?;
        } else {
            self.record(encoder, &ops)?;
        }
        if let (Given::Gpu { .. }, Some(f)) = (given, self.feedback.as_mut()) {
            f.copy(encoder, &self.seen_buf);
        }
        match out {
            Out::Texture(out) => encoder.copy_buffer_to_texture(copy(padded(sw), self.shown_at as u64 * 4), out.as_image_copy(), size(sw, h * up)),
            Out::Psx { colour, format, .. } => self.blit(encoder, width, colour, format),
        }
        if let Some(p) = self.profiler.get_mut().unwrap().as_mut() {
            p.stamp(encoder, "copies");
            p.end(encoder);
        }
        self.last_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    fn bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, bufs: [&wgpu::Buffer; 7]) -> wgpu::BindGroup {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: bufs[0], offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }),
        }];
        for (i, b) in bufs.iter().enumerate().skip(1) {
            entries.push(wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dream"), layout, entries: &entries })
    }

    /// The level to draw (`encode_view`): its triangles (a face is a unit square of a
    /// triangle's plane, by its main axis). Given again whenever they change (doors move, the
    /// light changes): faces keep their pages, by where they are.
    pub fn set_level(&mut self, tris: &[LevelTri]) -> Result<()> {
        let level = level::build(tris, self.cfg.spacing);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buffer = |label: &str, size: u64| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.div_ceil(4) * 4, usage: storage, mapped_at_creation: false });
        self.level_buf = buffer("dream level", level.data.len() as u64 * 4);
        let words = (level.faces.len() as u32).div_ceil(32).max(1);
        let seen_bytes = (BANDS * words) as u64 * 4;
        self.seen_buf = buffer("dream faces seen", seen_bytes);
        self.queue.write_buffer(&self.level_buf, 0, bytemuck::cast_slice(&level.data));
        self.raster.set_level(&self.device, &self.queue, &level.mesh, &level.flat, &self.level_buf, level.moved_at, level.verts + level.flat_verts);
        self.shell.set_level(&self.device, &self.queue, tris, &level.faces, &self.level_buf, &self.mem, self.cfg.page_res / HEIGHT_CELL);
        self.bind = Self::bind_group(&self.device, &self.layout, [&self.params, &self.mem, &self.atlas, &self.acc, &self.img, &self.level_buf, &self.seen_buf]);
        self.levels += 1;
        self.feedback = Some(Feedback::new(&self.device, seen_bytes, self.levels));
        self.band_pages = vec![Vec::new(); BANDS as usize];
        self.level = Some(level);
        self.write_page_table();
        Ok(())
    }

    /// The textures `LevelTri::texture` counts in: `size` texels a side each, RGBA8 in the
    /// encoding the picture's in (shown nearest, clamped).
    /// The level's lights, for lighting the dreamed height from where they are (with the
    /// torch): the light from everywhere and the rest (the nearest MAX_LIGHTS to the view are
    /// kept each frame). Their light on the level itself is the level's own (`LevelTri::light`).
    pub fn set_lights(&mut self, ambient: [f32; 3], lights: &[Light]) {
        self.ambient = ambient;
        self.lights = lights.to_vec();
    }

    /// This frame's lights into `mem`: the ambient, how many, then the torch (where it's held,
    /// its reach, its colour) and the level's nearest (where, radius, colour; radius 0 for one
    /// from far off).
    fn write_lights(&self, view: &View) {
        let held: [f32; 3] = std::array::from_fn(|i| view.pos[i] + (0..3).map(|a| view.axes[a][i] * view.torch_at[a]).sum::<f32>());
        let mut near: Vec<&Light> = self.lights.iter().collect();
        let d2 = |l: &Light| if l.radius <= 0.0 { 0.0 } else { (0..3).map(|i| (l.pos[i] - view.pos[i]).powi(2)).sum::<f32>() / (l.radius * l.radius) };
        near.sort_by(|a, b| d2(a).total_cmp(&d2(b)));
        near.truncate(MAX_LIGHTS);
        let mut out = vec![self.ambient[0], self.ambient[1], self.ambient[2], (near.len() + 1) as f32];
        out.extend([held[0], held[1], held[2], view.torch[3], view.torch[0], view.torch[1], view.torch[2], 0.0]);
        for l in near {
            out.extend([l.pos[0], l.pos[1], l.pos[2], l.radius, l.colour[0], l.colour[1], l.colour[2], 0.0]);
        }
        self.queue.write_buffer(&self.mem, self.at.lights as u64 * 4, bytemuck::cast_slice(&out));
    }

    pub fn set_textures(&mut self, size: u32, layers: &[Vec<u8>]) -> Result<()> {
        if layers.iter().any(|l| l.len() != (size * size * 4) as usize) {
            return Err(format!("textures of {size} x {size} RGBA8 wanted"));
        }
        self.raster.set_textures(&self.device, Some(&self.queue), size, layers);
        self.shell.set_textures(&self.device, self.raster.textures.as_ref());
        Ok(())
    }

    /// Each face's page, for gbuffer.
    fn write_page_table(&self) {
        let Some(level) = &self.level else { return };
        let table: Vec<f32> = level.faces.iter().map(|f| self.page_of.get(f).map_or(-1.0, |&p| p as f32)).collect();
        if !table.is_empty() {
            self.queue.write_buffer(&self.level_buf, level.pages_at as u64 * 4, bytemuck::cast_slice(&table));
        }
    }

    /// A report of the faces seen (BANDS bitsets) come back: pages for those without, and the
    /// pages in each band of rows.
    fn take_report(&mut self, bits: &[u32]) {
        let Some(level) = &self.level else { return };
        let words = level.faces.len().div_ceil(32).max(1);
        let mut seen: Vec<(usize, Face)> = Vec::new();
        for band in 0..BANDS as usize {
            for (w, &word) in bits[band * words..].iter().take(words).enumerate() {
                let mut b = word;
                while b != 0 {
                    let f = w * 32 + b.trailing_zeros() as usize;
                    b &= b - 1;
                    if let Some(&face) = level.faces.get(f) {
                        seen.push((band, face));
                    }
                }
            }
        }
        let mut bands = vec![Vec::new(); BANDS as usize];
        let mut in_band = vec![u32::MAX; self.cfg.max_pages as usize];
        let mut changed = false;
        for (band, face) in seen {
            let page = match self.page_of.get(&face) {
                Some(&p) => p,
                None => match self.allocate(face) {
                    Some(p) => {
                        changed = true;
                        p
                    }
                    None => continue,
                },
            };
            self.page_seen[page as usize] = self.frames;
            if std::mem::replace(&mut in_band[page as usize], band as u32) != band as u32 {
                bands[band].push(page as usize);
            }
        }
        self.band_pages = bands;
        if changed {
            self.write_page_table();
        }
    }

    /// A page for a face: a free one, or the one longest unseen (not seen lately: the reports
    /// come late).
    fn allocate(&mut self, face: Face) -> Option<u32> {
        let p = match self.owner.iter().position(Option::is_none) {
            Some(p) => p,
            None => {
                let p = (0..self.owner.len()).filter(|&p| self.page_seen[p] + 8 < self.frames).min_by_key(|&p| self.page_seen[p])?;
                if let Some(old) = self.owner[p].take() {
                    self.page_of.remove(&old);
                }
                self.clears.push(p as u32);
                p
            }
        };
        self.owner[p] = Some(face);
        self.page_of.insert(face, p as u32);
        if let Some(n) = self.nca.as_mut() {
            n.reset(p as u32);
        }
        self.nca_dirty = true;
        Some(p as u32)
    }

    /// Grow the pages with the neural cellular automata in `path` (nca::read_models; `r` cells a
    /// page side) while `Step::nca` is above 0: returns their names, for `set_nca_models`.
    pub fn set_nca(&mut self, path: &Path, r: u32) -> Result<Vec<String>> {
        let models = nca::read_models(path)?;
        self.nca_names = models.iter().map(|m| m.0.clone()).collect();
        self.nca = Some(nca::Nca::new(&self.device, &self.queue, &models, r, self.cfg.max_pages, &self.atlas, &self.mem));
        self.nca_dirty = true;
        Ok(self.nca_names.clone())
    }

    /// The automata (by index in `set_nca`'s names, -1 for none) for walls and for floors and
    /// ceilings; the pages changing start afresh.
    pub fn set_nca_models(&mut self, walls: i32, floors: i32) {
        if self.nca_models == [walls, floors] {
            return;
        }
        self.nca_models = [walls, floors];
        if let Some(n) = self.nca.as_mut() {
            n.reset_all();
        }
        self.nca_dirty = true;
    }

    /// The automata's steps on the pages in sight (the rest kept), and their colour and height
    /// laid on the pages.
    fn record_nca(&mut self, encoder: &mut wgpu::CommandEncoder, step: &Step) {
        let Some(n) = self.nca.as_mut() else { return };
        if self.nca_dirty {
            let models = self.nca_models;
            n.set_pages(&self.queue, &self.owner, &self.page_of, |f| models[(f[0] / 2 == 1) as usize]);
            self.nca_dirty = false;
        }
        let active: Vec<u32> = self.view_pages.iter().map(|&p| p as u32).collect();
        let mut all: Vec<u32> = self.page_of.values().copied().collect();
        all.sort();
        let r = self.cfg.page_res;
        n.record(&self.queue, encoder, &active, &all, step.nca_steps, 0.5, step.nca, step.nca_lift, step.psx, r, self.at.page_height, r / HEIGHT_CELL);
        if let Some(p) = self.profiler.get_mut().unwrap().as_mut() {
            p.stamp(encoder, "nca");
        }
    }

    /// The face a page is given to (with `encode_view`), if any.
    pub fn face_of_page(&self, page: u32) -> Option<Face> {
        self.owner.get(page as usize).copied().flatten()
    }

    /// Pages given to faces (with `encode_view`).
    pub fn pages_in_use(&self) -> usize {
        self.page_of.len()
    }

    /// Forget every page's dream and the screen layer's, and which face has which page.
    pub fn wake(&mut self) -> Result<()> {
        self.clears.extend(0..self.cfg.max_pages);
        if let Some(n) = self.nca.as_mut() {
            n.reset_all();
        }
        self.nca_dirty = true;
        self.page_of.clear();
        self.owner.iter_mut().for_each(|o| *o = None);
        self.write_page_table();
        self.clear_screen()
    }

    /// `n` floats of `mem` from `off`, waiting for the GPU (in pieces the size of the readback
    /// buffer).
    fn read_mem(&self, off: u32, n: u32) -> Result<Vec<f32>> {
        let piece = (self.readback.size() / 4) as u32;
        let mut out = Vec::with_capacity(n as usize);
        let mut at = 0;
        while at < n {
            let k = piece.min(n - at);
            let b = self.submit(&[], Some((&self.mem, (off + at) as u64 * 4, k as u64 * 4)))?;
            out.extend_from_slice(bytemuck::cast_slice(&b));
            at += k;
        }
        Ok(out)
    }

    /// For checks: the last frame's surfaces and motion (once its work has been submitted),
    /// waiting for the GPU.
    pub fn read_gbuffer(&self) -> Result<(Vec<Surface>, Vec<Motion>)> {
        let n = self.view_width * self.cfg.height;
        let s = self.read_mem(self.surfaces(), n * SURFACE)?;
        let m = self.read_mem(self.at.motion, n * MOTION)?;
        Ok((bytemuck::cast_slice(&s).to_vec(), bytemuck::cast_slice(&m).to_vec()))
    }

    /// This frame's screen layer.
    fn screen(&self) -> u32 {
        self.at.screen + self.flip as u32 * self.at.screen_size
    }

    /// A frame whose view hasn't moved since the last (so its surfaces are the last frame's):
    /// the dream grows on the screen instead of the pages, at the shown frame's pixels, sharp
    /// and with no surfaces to make. The next `frame` lays it on the pages first.
    pub fn frame_still(&mut self, picture: &[u8], width: u32, step: &Step) -> Result<&[u8]> {
        if width != self.view_width {
            return Err(format!("a still frame {width} wide after a view {} wide", self.view_width));
        }
        self.frame_cpu(picture, width, Given::Still, step)
    }

    /// Forget the screen layer (with `clear_pages` of every page, to wake from the dream).
    pub fn clear_screen(&mut self) -> Result<()> {
        if self.screen_live {
            self.screen_live = false;
            self.screen_own = 0.0;
            let n = 2 * self.at.screen_size;
            let r = rows(n.div_ceil(4096));
            self.ops.push((Kernel::Clear, Params { m: n, c_off: self.at.screen, ..Default::default() }, [64, r[0], r[1]]));
        }
        Ok(())
    }

    /// Lay the screen layer on the pages of the view it grew over (whose surfaces are still in
    /// `mem`), as many at a time as there are accumulators, and clear it.
    fn bake(&mut self, step: &Step) -> Result<()> {
        let (w, h, up, r) = (self.view_width, self.cfg.height, self.cfg.up, self.cfg.page_res);
        let pages = self.view_pages.clone();
        for chunk in pages.chunks(self.cfg.max_touched as usize) {
            let Some(set) = self.take_set() else { break };
            let mut slot_of = vec![-1.0f32; self.cfg.max_pages as usize];
            for (i, &p) in chunk.iter().enumerate() {
                slot_of[p] = i as f32;
            }
            let slot_pages: Vec<f32> = chunk.iter().map(|&p| p as f32).collect();
            self.queue.write_buffer(&self.mem, set.of as u64 * 4, bytemuck::cast_slice(&slot_of));
            self.queue.write_buffer(&self.mem, set.pages as u64 * 4, bytemuck::cast_slice(&slot_pages));
            self.queue.write_buffer(&self.mem, set.ones as u64 * 4, bytemuck::cast_slice(&vec![1.0f32; chunk.len()]));
            let n = chunk.len() as u32;
            let bake = Params { a_off: self.surfaces(), a_rs: self.at.relief, a_cs: up, b_off: set.of, c_rs: self.screen(), k: r, x0: w, x1: h, m: h * up, f1: step.relief, ..Default::default() };
            // Laid on as it is: no decay, all its colour, spread no wider; the wall owned as
            // much more as the screen layer owned it.
            let apply = Params { k: r, x0: set.pages, x1: set.ones, x2: 1, f0: step.limit, f1: 1.0, f2: self.screen_own, f3: step.own_max, ..Default::default() };
            self.ops.extend([
                (Kernel::ClearAcc, Params { k: r, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), n]),
                (Kernel::Bake, bake, [(w * up).div_ceil(16), (h * up).div_ceil(16), 1]),
                (Kernel::Apply, apply, [r.div_ceil(16), r.div_ceil(16), n]),
            ]);
        }
        self.clear_screen()
    }

    /// Put the screen layer's dream (the whole dream, carried along) on the pages in place of
    /// theirs, for these rows of the view (whose surfaces are still in `mem`).
    fn remember(&mut self, step: &Step, rows: std::ops::Range<u32>) -> Result<()> {
        let (w, h, up, r) = (self.view_width, self.cfg.height, self.cfg.up, self.cfg.page_res);
        let rows = rows.start.min(h)..rows.end.min(h);
        let mut pages = Vec::new();
        let mut taken = vec![false; self.cfg.max_pages as usize];
        for row in &self.view_rows[rows.start as usize..rows.end.min(self.view_rows.len() as u32) as usize] {
            for &p in row {
                if !std::mem::replace(&mut taken[p], true) {
                    pages.push(p);
                }
            }
        }
        for chunk in pages.chunks(self.cfg.max_touched as usize) {
            let Some(set) = self.take_set() else { break };
            let mut slot_of = vec![-1.0f32; self.cfg.max_pages as usize];
            for (i, &p) in chunk.iter().enumerate() {
                slot_of[p] = i as f32;
            }
            let slot_pages: Vec<f32> = chunk.iter().map(|&p| p as f32).collect();
            self.queue.write_buffer(&self.mem, set.of as u64 * 4, bytemuck::cast_slice(&slot_of));
            self.queue.write_buffer(&self.mem, set.pages as u64 * 4, bytemuck::cast_slice(&slot_pages));
            let n = chunk.len() as u32;
            let mut ops = Vec::new();
            // Its colour, how much of the wall it owns, and its height.
            for channel in 0..3 {
                let bake = Params { a_off: self.surfaces(), a_cs: up, b_off: set.of, c_rs: self.screen(), k: r, x0: w, x1: h, m: rows.end * up, x5: rows.start * up, x6: channel, ..Default::default() };
                ops.push((Kernel::ClearAcc, Params { k: r, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), n]));
                ops.push((Kernel::Bake, bake, [(w * up).div_ceil(16), (rows.len() as u32 * up).div_ceil(16), 1]));
                ops.push((Kernel::Store, Params { k: r, x0: set.pages, x2: channel, f0: step.limit, c_off: self.at.page_height, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), n]));
            }
            self.ops.extend(ops);
        }
        Ok(())
    }

    /// A frame's work: the picture is in `img` (rows `picture_rows` u32s apart), the frame shown
    /// goes after it (at `shown_at`, rows `shown_rows` apart).
    fn run(&mut self, width: u32, given: Given, step: &Step, picture_rows: u32, shown_rows: u32) -> Result<Vec<(Kernel, Params, [u32; 3])>> {
        let (h, up, r) = (self.cfg.height, self.cfg.up, self.cfg.page_res);
        if let Given::Cpu(s, m) = given
            && (s.len() != (width * h) as usize || m.is_some_and(|m| m.len() != (width * h) as usize))
        {
            return Err(format!("a {width} x {h} frame wants {} surfaces and motions", width * h));
        }
        if matches!(given, Given::Still) && width != self.view_width {
            return Err(format!("a still frame {width} wide after a view {} wide", self.view_width));
        }
        let surfaces = !matches!(given, Given::Still);
        let carry = matches!(given, Given::Cpu(_, Some(_)) | Given::Gpu { carry: true, .. });
        // (Work queued since the last frame, as `clear_screen`, goes first.)
        self.next_set = 1;
        if !self.clears.is_empty() {
            let list: Vec<f32> = self.clears.drain(..).map(|p| p as f32).collect();
            self.queue.write_buffer(&self.mem, self.at.page_list as u64 * 4, bytemuck::cast_slice(&list));
            self.ops.push((Kernel::ClearPages, Params { k: r, x0: self.at.page_list, c_off: self.at.relief, x1: self.at.page_height, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), list.len() as u32]));
        }
        let s0 = self.at.sets[0];
        // While carried along, a band of the screen's dream goes on the pages each frame
        // (through the last view's surfaces, still in mem); all of it when that stops.
        if self.reprojecting && self.screen_live {
            let h = self.cfg.height;
            if surfaces && !carry {
                self.remember(step, 0..h)?;
            } else if step.remember_every > 0 {
                let k = step.remember_every;
                let band = (self.frames % k as u64) as u32;
                self.remember(step, band * h / k..(band + 1) * h / k)?;
            }
        }
        if surfaces && !carry {
            if self.reprojecting {
                self.clear_screen()?;
                self.reprojecting = false;
            } else if self.screen_live {
                self.bake(step)?;
            }
        }
        // Starting to carry the dream along: whatever's on the screen goes on the pages, and
        // the screen starts from them.
        let none;
        let mut given = given;
        // Where each pixel was is unknown: the screen starts afresh.
        let mut fresh = false;
        if carry && (!self.reprojecting || self.view_width != width) {
            if self.screen_live && !self.reprojecting {
                self.bake(step)?;
            }
            self.clear_screen()?;
            self.reprojecting = true;
            if let Given::Cpu(s, _) = given {
                none = vec![Motion::NONE; (width * h) as usize];
                given = Given::Cpu(s, Some(&none));
            }
            fresh = true;
        }
        self.frames += 1;
        // Dreaming on the screen layer, not the pages.
        let still = !surfaces || carry;
        let mut pages = Vec::new();
        let mut gbuffer = Vec::new();
        if surfaces {
            self.view_rows.resize(h as usize, Vec::new());
            if let Given::Cpu(surfaces, _) = given {
                // The pages seen.
                let mut slot_of = vec![-1.0f32; self.cfg.max_pages as usize];
                for s in surfaces {
                    if s.page >= 0.0 {
                        let p = s.page as usize;
                        if p >= slot_of.len() {
                            return Err(format!("page {p} of {}", self.cfg.max_pages));
                        }
                        if slot_of[p] < 0.0 {
                            slot_of[p] = 0.0;
                            pages.push(p);
                        }
                    }
                }
                let mut row_of_page = vec![u32::MAX; self.cfg.max_pages as usize];
                for (y, row) in self.view_rows.iter_mut().enumerate() {
                    row.clear();
                    for s in &surfaces[y * width as usize..][..width as usize] {
                        if s.page >= 0.0 && std::mem::replace(&mut row_of_page[s.page as usize], y as u32) != y as u32 {
                            row.push(s.page as usize);
                        }
                    }
                }
            } else {
                // The pages by the last report of the faces seen.
                let mut taken = vec![false; self.cfg.max_pages as usize];
                for band in &self.band_pages {
                    for &p in band {
                        if !std::mem::replace(&mut taken[p], true) {
                            pages.push(p);
                        }
                    }
                }
                let per_band = h.div_ceil(BANDS);
                for (y, row) in self.view_rows.iter_mut().enumerate() {
                    row.clone_from(&self.band_pages[(y as u32 / per_band).min(BANDS - 1) as usize]);
                }
            }
            self.pages_seen = pages.len();
            self.view_width = width;
            self.view_pages = pages.clone();
            self.view_flip = !self.view_flip;
            match given {
                Given::Cpu(surfaces, _) => {
                    self.lit_ready = false;
                    self.queue.write_buffer(&self.mem, self.surfaces() as u64 * 4, bytemuck::cast_slice(surfaces));
                }
                Given::Gpu { view, creatures, .. } => {
                    self.lit_ready = true;
                    let level = self.level.as_ref().ok_or("no level given")?;
                    let last = self.last_view.filter(|&(_, w)| w == width && carry && !fresh);
                    let mut views = view.floats(width).to_vec();
                    views.extend(last.map_or([0.0; VIEW as usize], |(v, w)| v.floats(w)));
                    // Where the lights and each pixel's point are, for gbuffer, own_picture and show.
                    views.extend([f32::from_bits(self.at.lights), f32::from_bits(self.at.lit)]);
                    self.write_lights(&view);
                    self.queue.write_buffer(&self.mem, self.at.views as u64 * 4, bytemuck::cast_slice(&views));
                    let size = align(self.cfg.max_width * h);
                    let (now, before) = if self.nearest_flip { (self.at.nearest + size, self.at.nearest) } else { (self.at.nearest, self.at.nearest + size) };
                    self.nearest_flip = !self.nearest_flip;
                    self.last_view = Some((view, width));
                    let words = (level.faces.len() as u32).div_ceil(32).max(1);
                    let o = level.origin.map(|c| c as u32);
                    let p = Params {
                        a_off: self.surfaces(),
                        b_off: self.at.motion,
                        c_off: now,
                        c_rs: before,
                        bias_off: self.at.views,
                        x0: width,
                        x1: h,
                        x2: level.table_at,
                        x3: level.table_mask,
                        x4: level.pages_at,
                        x5: self.ids_at,
                        b_cs: picture_rows,
                        x6: if creatures { self.at.creatures } else { NONE },
                        a_rs: o[0],
                        a_cs: o[1],
                        b_rs: o[2],
                        k: self.at.points,
                        m: words,
                        n: h.div_ceil(BANDS),
                        flags: if last.is_some() { MOTION_FROM_LAST } else { 0 } | if step.own_picture { OWN_PICTURE } else { 0 },
                        ..Default::default()
                    };
                    let groups = [width.div_ceil(16), h.div_ceil(16), 1];
                    gbuffer = vec![(Kernel::Gbuffer, p, groups)];
                    if step.own_picture {
                        gbuffer.push((Kernel::OwnPicture, p, groups));
                    }
                }
                Given::Still => {}
            }
        } else if step.own_picture && self.level.is_some() && self.last_view.is_some_and(|(_, w)| w == width) {
            // Still: the level drawn again from the last view (its view still in mem), for
            // its own picture under the dream.
            let p = Params { x0: width, x1: h, x5: self.ids_at, b_cs: picture_rows, k: self.at.points, bias_off: self.at.views, x6: NONE, ..Default::default() };
            gbuffer.push((Kernel::OwnPicture, p, [width.div_ceil(16), h.div_ceil(16), 1]));
        }
        let motion = match given {
            Given::Cpu(_, m) => m,
            _ => None,
        };
        if still {
            pages.clear();
        } else {
            // An accumulator for each page seen, up to max_touched: when more are in sight,
            // those dreamed on longest ago first, so every page gets its turn.
            let mut slot_of = vec![-1.0f32; self.cfg.max_pages as usize];
            for &p in &pages {
                slot_of[p] = 0.0;
            }
            if pages.len() > self.cfg.max_touched as usize {
                pages.sort_by_key(|&p| (self.last[p], p));
                for &p in &pages[self.cfg.max_touched as usize..] {
                    slot_of[p] = -1.0;
                }
                pages.truncate(self.cfg.max_touched as usize);
            }
            for (i, &p) in pages.iter().enumerate() {
                slot_of[p] = i as f32;
            }
            let keep = 1.0 - step.decay;
            let decay: Vec<f32> = pages.iter().map(|&p| keep.powf((self.frames - self.last[p].min(self.frames - 1)) as f32)).collect();
            let slot_pages: Vec<f32> = pages.iter().map(|&p| p as f32).collect();
            let ones = vec![1.0f32; pages.len()];
            self.queue.write_buffer(&self.mem, s0.of as u64 * 4, bytemuck::cast_slice(&slot_of));
            if !pages.is_empty() {
                self.queue.write_buffer(&self.mem, s0.pages as u64 * 4, bytemuck::cast_slice(&slot_pages));
                self.queue.write_buffer(&self.mem, s0.decay as u64 * 4, bytemuck::cast_slice(&decay));
                self.queue.write_buffer(&self.mem, s0.ones as u64 * 4, bytemuck::cast_slice(&ones));
            }
        }

        let mut ops = std::mem::take(&mut self.ops);
        ops.extend(gbuffer);
        let n_slots = pages.len() as u32;
        if carry {
            if let Some(motion) = motion {
                self.queue.write_buffer(&self.mem, self.at.motion as u64 * 4, bytemuck::cast_slice(motion));
            }
            let from = self.screen();
            self.flip = !self.flip;
            let fill = if step.remember_every > 0 { FILL } else { 0 };
            let p = Params { bias_off: self.at.motion, a_off: self.surfaces(), b_rs: from, c_rs: self.screen(), c_off: self.at.page_height, a_cs: up, k: r, x0: width, x1: h, flags: fill, ..Default::default() };
            ops.push((Kernel::Warp, p, [(width * up).div_ceil(16), (h * up).div_ceil(16), 1]));
        }
        // The relief: a_rs where it is, f1 how high. The screen layer: c_rs where it is, a_cs
        // the shown frame's scale, f2 how much more of the wall it owns (up to f3).
        let flags = if still || self.screen_live { SCREEN } else { 0 } | if self.reprojecting { SCREEN_ONLY } else { 0 };
        // The picture's rows: b_cs.
        let frame = Params {
            a_off: self.surfaces(),
            b_cs: picture_rows,
            bias_off: self.at.smooth,
            a_rs: self.at.relief,
            c_rs: self.screen(),
            a_cs: up,
            flags,
            k: r,
            x0: width,
            x1: h,
            f1: step.relief,
            f3: step.own_max,
            ..Default::default()
        };
        let (gw, gh) = (width.div_ceil(SMOOTH), h.div_ceil(SMOOTH));
        ops.push((Kernel::Smooth, Params { c_off: self.at.smooth, ..frame }, [gw.div_ceil(16), gh.div_ceil(16), 1]));
        // The two octaves: (scale, crop, size, roll, step, layers, decay).
        let mut octaves = Vec::new();
        if step.lr > 0.0 && self.frames % step.coarse_every.max(1) as u64 == 0 {
            // A whole 8 wide at the scale asked for, near enough.
            let nw = ((width as f32 * step.scale.clamp(0.25, 1.5) / 8.0).round() as u32 * 8).max(32);
            let scale = nw as f32 / width as f32;
            let nh = ((h as f32 * scale).round() as u32).max(32);
            let j = step.jitter;
            let roll = if j > 0 { [self.random() % (2 * j + 1) + nw - j, self.random() % (2 * j + 1) + nh - j] } else { [0, 0] };
            octaves.push((scale, [0, 0], [nw, nh], roll, step.lr, false, s0.decay));
        }
        let every = step.fine_every.max(1) as u64;
        if step.lr_fine > 0.0 && (self.frames + step.fine_offset as u64) % every == 0 {
            let split = step.fine_split.max(1);
            let scale = step.fine_scale.clamp(1.0, up as f32);
            let (sw, sh) = ((width as f32 * scale) as u32, (h as f32 * scale) as u32);
            // Whole 16s, so the network's sizes stay few.
            let (fw, fh) = ((sw / split) / 16 * 16, (sh / split) / 16 * 16);
            let q = ((self.frames / every) % (split * split) as u64) as u32;
            let jx = fw / 4;
            let jy = fh / 4;
            let x0 = ((q % split) * fw + self.random() % (jx + 1)).saturating_sub(jx / 2).min(sw - fw);
            let y0 = ((q / split) * fh + self.random() % (jy + 1)).saturating_sub(jy / 2).min(sh - fh);
            let j = step.jitter_fine;
            let roll = if j > 0 { [self.random() % (2 * j + 1) + fw - j, self.random() % (2 * j + 1) + fh - j] } else { [0, 0] };
            // The first octave of the frame decays the pages.
            let decay_at = if octaves.is_empty() { s0.decay } else { s0.ones };
            octaves.push((scale, [x0, y0], [fw, fh], roll, step.lr_fine, true, decay_at));
        }
        let dreamed = !octaves.is_empty();
        // The pages' decay is owed until they're next dreamed on; on the screen layer they're
        // dreamed on in its stead, and don't decay.
        if dreamed {
            for &p in if still { &self.view_pages } else { &pages } {
                self.last[p] = self.frames;
            }
        }
        // The dreamed height grows while the dream's carried along: each octave's dream lamp
        // (its slot: direction along the page and off it, ambient, how much, shading alone,
        // the height's step and most).
        let lamp = step.height > 0.0 && self.reprojecting && still;
        let mut shading_alone = [false; 2];
        if lamp && dreamed {
            let mut slots = [0.0f32; 2 * LAMP_SLOT];
            for (k, slot) in slots.chunks_exact_mut(LAMP_SLOT).enumerate() {
                let turn = self.random() as f32 / u32::MAX as f32 * std::f32::consts::TAU;
                let rise = 0.35 + self.random() as f32 / u32::MAX as f32 * 0.85;
                self.lamp_steps += 1;
                shading_alone[k] = step.shade_only_every > 0 && self.lamp_steps % step.shade_only_every as u64 == 0;
                slot[..9].copy_from_slice(&[rise.cos() * turn.cos(), rise.cos() * turn.sin(), rise.sin(), step.lamp_ambient, step.lamp, shading_alone[k] as u32 as f32, step.height, step.height_max, step.height_decay]);
            }
            self.queue.write_buffer(&self.mem, self.at.shade as u64 * 4, bytemuck::cast_slice(&slots));
        }
        if n_slots > 0 || (still && !self.view_pages.is_empty()) {
            for (oi, (scale, crop, size, roll, lr, fine, decay_at)) in octaves.into_iter().enumerate() {
                let net = step.net.min(self.nets.len() - 1);
                let class = self.class_row(net, step.class);
                let key = (size[1], size[0], fine, net, class);
                if !self.plans.contains_key(&key) {
                    let layers = if fine { self.nets[net].fine_layers.clone() } else { self.nets[net].layers.clone() };
                    let (plan, _) = self.plan(net, size[1], size[0], &layers, class);
                    self.plans.insert(key, plan);
                }
                let plan = &self.plans[&key];
                let x = self.at.acts + plan.input.off;
                let gx = self.at.grads + plan.input.off;
                if !still {
                    ops.push((Kernel::ClearAcc, Params { k: r, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), n_slots]));
                }
                let mut p = Params { m: size[1], n: size[0], c_off: x, b_off: self.at.lookup, g0: scale, x3: crop[0], x4: crop[1], x5: roll[0], x6: roll[1], ..frame };
                if lamp {
                    (p.flags, p.b_rs, p.f0) = (p.flags | LAMP, self.at.shade, oi as f32);
                }
                ops.push((Kernel::Prepare, p, [size[0].div_ceil(16), size[1].div_ceil(16), 1]));
                ops.extend(plan.ops.iter().copied());
                if step.grad_blur > 0.0 {
                    // Across into the im2col room (free once the network's run), then down back.
                    let g = [size[0].div_ceil(16), size[1].div_ceil(16), 1];
                    let blur = Params { m: size[1], n: size[0], f0: step.grad_blur, ..Default::default() };
                    ops.push((Kernel::BlurGrad, Params { a_off: gx, c_off: self.at.col, x0: 0, ..blur }, g));
                    ops.push((Kernel::BlurGrad, Params { a_off: self.at.col, c_off: gx, x0: 1, ..blur }, g));
                }
                let mean = RESULTS - 2;
                let px = size[0] * size[1];
                ops.push((Kernel::Reduce, Params { m: px, a_off: gx, b_off: self.at.lookup, c_off: self.at.partials, x0: 1, ..Default::default() }, [PARTIALS, 1, 1]));
                ops.push((Kernel::ReduceFinish, Params { m: PARTIALS, a_off: self.at.partials, c_off: self.at.results + mean * 2, ..Default::default() }, [1, 1, 1]));
                if let (true, Some(c), Some(t)) = (class, &self.nets[net].classes, plan.last) {
                    let p = Params { m: t.h * t.w, k: t.c, a_off: self.at.acts + t.off, b_off: c.all, x0: c.bias, x1: self.class_rows[net].unwrap_or(0), x2: c.count, x3: self.at.results + SURE * 2, x4: fine as u32, c_off: self.at.results + mean * 2, ..Default::default() };
                    ops.push((Kernel::Confidence, p, [1, 1, 1]));
                }
                if lamp {
                    let g = [size[0].div_ceil(16), size[1].div_ceil(16), 1];
                    ops.push((Kernel::ShadeBack, Params { m: size[1], n: size[0], a_off: gx, bias_off: self.at.shade, x5: roll[0], x6: roll[1], ..Default::default() }, g));
                    let height_reduce = |ops: &mut Vec<(Kernel, Params, [u32; 3])>| {
                        ops.push((Kernel::Reduce, Params { m: px, a_off: self.at.shade + LAMP_AUX, b_off: self.at.lookup, c_off: self.at.partials, x0: 2, ..Default::default() }, [PARTIALS, 1, 1]));
                        ops.push((Kernel::ReduceFinish, Params { m: PARTIALS, a_off: self.at.partials, c_off: self.at.results + HEIGHT_MEAN * 2, ..Default::default() }, [1, 1, 1]));
                    };
                    if step.height_from_colour != 0.0 {
                        height_reduce(&mut ops);
                        let p = Params {
                            m: size[1],
                            n: size[0],
                            a_off: gx,
                            bias_off: self.at.shade,
                            x0: self.at.results + mean * 2,
                            x1: self.at.results + HEIGHT_MEAN * 2,
                            f0: step.height_from_colour.clamp(-1.0, 1.0),
                            ..Default::default()
                        };
                        ops.push((Kernel::HeightMix, p, g));
                    }
                    // The lamp's work a pixel is free after shade_back but the height's step
                    // (5): each size blurred across into 3, then down into 4, or added into 0.
                    let blur = Params { m: size[1], n: size[0], bias_off: self.at.shade, x5: roll[0], x6: roll[1], ..Default::default() };
                    // Only the coarse octave: the fine one's pieces would wrap big blurs round inside each.
                    if step.height_spread > 1 && !fine {
                        for i in 0..step.height_spread.min(6) {
                            let size = (step.height_blur.max(1) << i) as f32;
                            let p = Params { f0: size, f1: size.powf(step.height_power), ..blur };
                            ops.push((Kernel::BlurHeight, Params { x0: 0, x1: 5, x2: 3, ..p }, g));
                            ops.push((Kernel::BlurHeight, Params { x0: 1, x1: 3, x2: 0, x3: (i > 0) as u32, ..p }, g));
                        }
                        ops.push((Kernel::BlurHeight, Params { x0: 0, x1: 0, x2: 5, f0: 0.0, ..blur }, g));
                    } else if step.height_blur > 0 {
                        let p = Params { f0: step.height_blur as f32, ..blur };
                        ops.push((Kernel::BlurHeight, Params { x0: 0, x1: 5, x2: 3, ..p }, g));
                        ops.push((Kernel::BlurHeight, Params { x0: 1, x1: 3, x2: 5, ..p }, g));
                    }
                    height_reduce(&mut ops);
                    if step.height_centred {
                        ops.push((Kernel::Reduce, Params { m: px, a_off: self.at.shade + LAMP_AUX, b_off: self.at.lookup, c_off: self.at.partials, x0: 3, ..Default::default() }, [PARTIALS, 1, 1]));
                        ops.push((Kernel::ReduceFinish, Params { m: PARTIALS, a_off: self.at.partials, c_off: self.at.results + HEIGHT_SIGNED * 2, ..Default::default() }, [1, 1, 1]));
                    }
                }
                if still {
                    let p = Params {
                        m: size[1],
                        n: size[0],
                        a_off: gx,
                        b_off: self.at.lookup,
                        c_rs: self.screen(),
                        a_rs: width * up,
                        b_rs: h * up,
                        a_cs: up,
                        k: step.colour_blur,
                        x0: self.at.results + mean * 2,
                        g0: scale,
                        x3: crop[0],
                        x4: crop[1],
                        x5: roll[0],
                        x6: roll[1],
                        f0: if shading_alone[oi.min(1)] { 0.0 } else { lr },
                        f1: step.colour,
                        f2: step.limit,
                        f3: step.take_over,
                        ..Default::default()
                    };
                    // The fine octave's pieces fade out over a sixth of their size at their edges,
                    // the coarse octave's over a twelfth at the screen's (else what grows lopsided
                    // there is carried in as the view turns, a seam where the edge was).
                    let fade = size[0].min(size[1]) as f32 / if fine { 6.0 } else { 12.0 };
                    let p = Params { g1: fade, ..p };
                    let signed = if step.height_centred { self.at.results + HEIGHT_SIGNED * 2 } else { 0 };
                    let p = if lamp { Params { bias_off: self.at.shade, x1: oi as u32, c_off: self.at.results + HEIGHT_MEAN * 2, x2: signed, ..p } } else { p };
                    let (cw, ch) = ((size[0] * up) as f32 / scale, (size[1] * up) as f32 / scale);
                    let (cw, ch) = (cw.ceil() as u32 + 1, ch.ceil() as u32 + 1);
                    ops.push((Kernel::ScreenStep, p, [cw.div_ceil(16), ch.div_ceil(16), 1]));
                    continue;
                }
                let p = Params { m: size[1], n: size[0], k: r, a_off: gx, b_off: self.at.lookup, x0: self.at.results + mean * 2, x1: s0.of, f0: lr, ..Default::default() };
                ops.push((Kernel::Scatter, p, [size[0].div_ceil(16), size[1].div_ceil(16), 1]));
                let p = Params { k: r, x0: s0.pages, x1: decay_at, x2: step.colour_blur, f0: step.limit, f1: step.colour, f2: step.take_over, f3: step.own_max, ..Default::default() };
                ops.push((Kernel::Apply, p, [r.div_ceil(16), r.div_ceil(16), n_slots]));
            }
            if dreamed && !still && step.relief > 0.0 {
                let hr = r / RELIEF_CELL;
                let p = Params { k: r, x0: s0.pages, c_off: self.at.relief, f0: step.relief_rate, f1: step.limit, ..Default::default() };
                ops.push((Kernel::Raise, p, [hr.div_ceil(16), hr.div_ceil(16), n_slots]));
            }
            if dreamed && still {
                self.screen_live = true;
                self.screen_own = (self.screen_own + step.take_over).min(1.0);
            }
        }
        let (sw, sh) = (width * up, h * up);
        // The frame shown's rows: b_rs.
        if step.glint > 0.0 {
            let light = [step.glint, step.torch_reach, step.rough_dry, step.rough_wet, 1.0, 0.82, 0.6, 0.0];
            self.queue.write_buffer(&self.mem, self.at.light as u64 * 4, bytemuck::cast_slice(&light));
        }
        // dream.wgsl's globals (lands before any of the frame's work): the light on each pixel,
        // for lighting the dream, while the G-buffer's made on the GPU; and the shell's hits for
        // `show` (dream.wgsl shell_seen).
        let rows = point_rows(sw);
        let (marched, creatures) = (self.shell_live, self.creatures_drawn);
        let globals = [
            f32::from_bits(if self.lit_ready { self.at.lit } else { 0 }),
            step.dream_lit,
            f32::from_bits(if marched { self.shell_at } else { 0 }),
            f32::from_bits(self.shell_at + rows * sh),
            f32::from_bits(rows),
            step.height_max * step.shell,
            (step.nca > 0.0 && self.nca.is_some()) as u32 as f32,
            step.show_shell as u32 as f32,
            f32::from_bits(self.at.views),
            f32::from_bits(if creatures { self.at.creatures } else { 0 }),
            step.own_picture as u32 as f32,
        ];
        self.queue.write_buffer(&self.mem, 0, bytemuck::cast_slice(&globals));
        let p = Params {
            b_off: if step.glint > 0.0 { self.at.light } else { 0 },
            m: sh,
            n: sw,
            x2: up,
            c_off: self.shown_at,
            b_rs: shown_rows,
            f0: step.sharpness,
            f2: if self.reprojecting || (step.nca > 0.0 && self.nca.is_some()) { step.show_shade } else { 0.0 },
            g2: step.show_wrap,
            // The lights, where each pixel's lit from where they are (else the eye's).
            x3: if self.lit_ready && step.lights { self.at.lights } else { 0 },
            x4: self.at.lit,
            flags: frame.flags | if step.show_height && self.reprojecting { SHOW_HEIGHT } else { 0 } | if step.psx { PSX } else { 0 },
            ..frame
        };
        if self.shell_live {
            // The largest |height| on any page, for the shell's prisms to reach just past.
            let cells = self.cfg.max_pages * (self.cfg.page_res / HEIGHT_CELL).pow(2);
            ops.push((Kernel::Reduce, Params { m: cells, a_off: self.at.page_height, c_off: self.at.partials, x0: 4, ..Default::default() }, [PARTIALS, 1, 1]));
            ops.push((Kernel::ReduceFinish, Params { m: PARTIALS, a_off: self.at.partials, c_off: self.at.results + HEIGHT_BOUND * 2, x0: 4, ..Default::default() }, [1, 1, 1]));
        }
        ops.push((Kernel::Show, p, [sw.div_ceil(16), sh.div_ceil(16), 1]));
        Ok(ops)
    }

    /// Record `ops` into `enc`, their parameters written to the queue (to land before it runs).
    fn record(&self, enc: &mut wgpu::CommandEncoder, ops: &[(Kernel, Params, [u32; 3])]) -> Result<()> {
        self.record_split(enc, ops, &[], &mut |_, _| {})
    }

    /// The same, with `between(enc, k)` recorded before the ops from `splits[k - 1]` on.
    fn record_split(&self, enc: &mut wgpu::CommandEncoder, ops: &[(Kernel, Params, [u32; 3])], splits: &[usize], between: &mut dyn FnMut(&mut wgpu::CommandEncoder, usize)) -> Result<()> {
        if ops.len() > MAX_OPS {
            return Err(format!("{} dispatches, room for {MAX_OPS}", ops.len()));
        }
        let mut slots = vec![0u8; ops.len() * SLOT as usize];
        for (i, (_, p, _)) in ops.iter().enumerate() {
            slots[i * SLOT as usize..][..std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        self.queue.write_buffer(&self.params, 0, &slots);
        let bounds: Vec<usize> = std::iter::once(0).chain(splits.iter().copied()).chain(std::iter::once(ops.len())).collect();
        for (part, range) in bounds.windows(2).map(|b| b[0]..b[1].max(b[0])).enumerate() {
            if part > 0 {
                between(enc, part);
            }
            if range.is_empty() {
                continue;
            }
            let mut pass = enc.begin_compute_pass(&Default::default());
            for i in range {
                let (kernel, _, groups) = &ops[i];
                pass.set_pipeline(&self.pipelines[*kernel as usize]);
                pass.set_bind_group(0, &self.bind, &[(i as u64 * SLOT) as u32]);
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                if let Some(p) = self.profiler.lock().unwrap().as_mut() {
                    p.stamp_pass(&mut pass, &op_label(*kernel, &ops[i].1, p.by_shape));
                }
            }
        }
        Ok(())
    }

    /// Run `ops` and, with `read`, copy (buffer, byte offset, bytes) back.
    fn submit(&self, ops: &[(Kernel, Params, [u32; 3])], read: Option<(&wgpu::Buffer, u64, u64)>) -> Result<Vec<u8>> {
        self.submit_timed(ops, read, false)
    }

    /// `submit`, as a frame for the profiler if `timed`.
    fn submit_timed(&self, ops: &[(Kernel, Params, [u32; 3])], read: Option<(&wgpu::Buffer, u64, u64)>, timed: bool) -> Result<Vec<u8>> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        let profiler = || self.profiler.lock().unwrap();
        if timed && let Some(p) = profiler().as_mut() {
            p.begin(&mut enc);
        }
        self.record(&mut enc, ops)?;
        if timed && let Some(p) = profiler().as_mut() {
            p.end(&mut enc);
        }
        let Some((buf, off, size)) = read else {
            self.queue.submit([enc.finish()]);
            return Ok(Vec::new());
        };
        enc.copy_buffer_to_buffer(buf, off, &self.readback, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        let out = slice.get_mapped_range().map_err(|e| e.to_string())?.to_vec();
        self.readback.unmap();
        Ok(out)
    }

    /// For checks: one coarse octave (no roll) of the picture (RGBA8, `width` wide) shown on
    /// `surfaces`, carried along with no colour yet and the given heights (a shown pixel each)
    /// under the dream lamp `lamp` (a slot: direction along the page and off it, ambient, how
    /// much, shading alone): the objective (half the log of each layer's sum of squares,
    /// summed) and its gradient by each pixel's height as shade_back works it out (a pixel of
    /// the picture's each, by the mean of its shown pixels' heights).
    pub fn lamp_gradient(&mut self, picture: &[u8], width: u32, surfaces: &[Surface], heights: &[f32], lamp: [f32; 6]) -> Result<(f64, Vec<f32>)> {
        let (h, up, r) = (self.cfg.height, self.cfg.up, self.cfg.page_res);
        let (sw, sh) = (width * up, h * up);
        if surfaces.len() != (width * h) as usize || heights.len() != (sw * sh) as usize || picture.len() != (width * h * 4) as usize {
            return Err("sizes".into());
        }
        self.queue.write_buffer(&self.img, 0, picture);
        self.queue.write_buffer(&self.mem, self.surfaces() as u64 * 4, bytemuck::cast_slice(surfaces));
        let mut layer = vec![0.0f32; (4 * sw * sh) as usize];
        layer.extend_from_slice(heights);
        self.queue.write_buffer(&self.mem, self.screen() as u64 * 4, bytemuck::cast_slice(&layer));
        let mut slot = [0.0f32; LAMP_SLOT];
        slot[..6].copy_from_slice(&lamp);
        self.queue.write_buffer(&self.mem, self.at.shade as u64 * 4, bytemuck::cast_slice(&slot));
        let key = (h, width, false, 0, false);
        if !self.plans.contains_key(&key) {
            let (plan, _) = self.plan(0, h, width, &self.nets[0].layers.clone(), false);
            self.plans.insert(key, plan);
        }
        let plan = &self.plans[&key];
        let frame = Params {
            a_off: self.surfaces(),
            b_cs: width,
            bias_off: self.at.smooth,
            a_rs: self.at.relief,
            c_rs: self.screen(),
            a_cs: up,
            flags: SCREEN | SCREEN_ONLY | LAMP,
            k: r,
            x0: width,
            x1: h,
            f3: 1.0,
            ..Default::default()
        };
        let mut ops = vec![(Kernel::Smooth, Params { c_off: self.at.smooth, ..frame }, [width.div_ceil(SMOOTH).div_ceil(16), h.div_ceil(SMOOTH).div_ceil(16), 1])];
        let p = Params { m: h, n: width, c_off: self.at.acts + plan.input.off, b_off: self.at.lookup, g0: 1.0, b_rs: self.at.shade, f0: 0.0, ..frame };
        ops.push((Kernel::Prepare, p, [width.div_ceil(16), h.div_ceil(16), 1]));
        ops.extend(plan.ops.iter().copied());
        ops.push((Kernel::ShadeBack, Params { m: h, n: width, a_off: self.at.grads + plan.input.off, bias_off: self.at.shade, ..Default::default() }, [width.div_ceil(16), h.div_ceil(16), 1]));
        self.submit(&ops, None)?;
        let sums = self.read_mem(self.at.results, self.nets[0].layers.len() as u32 * 2)?;
        let objective = sums.chunks_exact(2).map(|s| 0.5 * (s[0] as f64).ln()).sum();
        let aux = self.read_mem(self.at.shade + LAMP_AUX, width * h * 6)?;
        Ok((objective, aux.chunks_exact(6).map(|a| a[5]).collect()))
    }

    /// For checks: every page given to a face takes the height `height(page)` all over.
    pub fn fill_heights(&self, height: impl Fn(u32) -> f32) {
        let cells = (self.cfg.page_res / HEIGHT_CELL).pow(2) as usize;
        for &p in self.page_of.values() {
            let v = vec![height(p); cells];
            self.queue.write_buffer(&self.mem, (self.at.page_height as u64 + p as u64 * cells as u64) * 4, bytemuck::cast_slice(&v));
        }
    }

    /// For checks: every page given to a face takes `height(page, i, j)` at its height cell
    /// (i, j) (`height_cells` a side, cell i's middle (HEIGHT_CELL i + 0.5) / page_res across).
    pub fn fill_height_cells(&self, height: impl Fn(u32, u32, u32) -> f32) {
        let r = self.height_cells();
        for &p in self.page_of.values() {
            let v: Vec<f32> = (0..r * r).map(|c| height(p, c % r, c / r)).collect();
            self.queue.write_buffer(&self.mem, (self.at.page_height as u64 + (p * r * r) as u64) * 4, bytemuck::cast_slice(&v));
        }
    }

    pub fn height_cells(&self) -> u32 {
        self.cfg.page_res / HEIGHT_CELL
    }

    /// For checks: what the shell's march met at each shown pixel of the last frame drawn
    /// (`Step::shell` on), waiting for the GPU.
    pub fn read_shell(&self) -> Result<Vec<ShellHit>> {
        let (Some(level), Some((view, width))) = (&self.level, self.last_view) else { return Err("no level drawn".into()) };
        let (up, sh) = (self.cfg.up, self.cfg.height * self.cfg.up);
        let sw = width * up;
        let rows = point_rows(sw);
        let mut words = Vec::with_capacity((2 * rows * sh) as usize);
        let total = 2 * rows * sh;
        let chunk = (self.readback.size() / 4) as u32;
        let mut at = 0;
        while at < total {
            let n = chunk.min(total - at);
            let b = self.submit(&[], Some((&self.img, (self.shell_at + at) as u64 * 4, n as u64 * 4)))?;
            words.extend_from_slice(bytemuck::cast_slice::<u8, u32>(&b));
            at += n;
        }
        let looks = (rows * sh) as usize;
        let f = view.focal * up as f32;
        Ok((0..sh)
            .flat_map(|y| (0..sw).map(move |x| (x, y)))
            .map(|(x, y)| {
                let i = (y * rows + 4 * x) as usize;
                let (tri, depth, place) = (words[i], f32::from_bits(words[i + 2]), words[looks + i]);
                let (mut foot, mut height) = ([0.0; 3], 0.0);
                if tri > 0 {
                    // As dream.wgsl's shell_seen: the foot's place on its tile, the tile the hit's
                    // over, within half a tile.
                    let a = view.axes;
                    let dv = [(x as f32 + 0.5 - sw as f32 / 2.0) / f, (y as f32 + 0.5 - sh as f32 / 2.0) / f, 1.0];
                    let t = (tri - 1) as usize * 4;
                    let n: [f32; 3] = std::array::from_fn(|k| f32::from_bits(level.data[t + k]));
                    let d = f32::from_bits(level.data[t + 3]);
                    let hit: [f32; 3] = std::array::from_fn(|k| view.pos[k] + (a[0][k] * dv[0] + a[1][k] * dv[1] + a[2][k] * dv[2]) * depth);
                    height = (0..3).map(|k| n[k] * hit[k]).sum::<f32>() - d;
                    let axis = (0..3).max_by(|&p, &q| n[p].abs().total_cmp(&n[q].abs())).unwrap();
                    let along = [[1, 2], [0, 2], [0, 1]][axis];
                    let unorm = [place & 0xffff, place >> 16].map(|q| q as f32 / 65535.0);
                    for (j, &k) in along.iter().enumerate() {
                        let dropped = hit[k] - n[k] * height;
                        let mut c = dropped.floor() + unorm[j];
                        if c - dropped > 0.5 {
                            c -= 1.0;
                        } else if c - dropped < -0.5 {
                            c += 1.0;
                        }
                        foot[k] = c;
                    }
                    foot[axis] = (d - n[along[0]] * foot[along[0]] - n[along[1]] * foot[along[1]]) / n[axis];
                }
                let page = words[i + 3] & 0x0fff_ffff;
                ShellHit { tri, foot, height, depth, page: if page == 0x0fff_ffff { u32::MAX } else { page } }
            })
            .collect())
    }

    /// For checks: the cells of `page`'s height as the shell smoothed them last frame
    /// (`Step::shell_smooth`), and as smoothed far off (`Step::shell_far`), waiting for the GPU.
    pub fn read_shell_smoothed(&self, page: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = (self.height_cells() * self.height_cells()) as u64;
        let read = |b: &wgpu::Buffer| self.submit(&[], Some((b, page as u64 * n * 4, n * 4))).map(|b| bytemuck::cast_slice::<u8, f32>(&b).to_vec());
        Ok((read(&self.shell.near)?, read(&self.shell.far)?))
    }

    /// For checks: how far the shell's prisms reached last frame (the largest |height| on any
    /// page, or smoothed the largest smoothed one, times `shell`, a hair past it, within the
    /// limit), as shell.wgsl's `reach`.
    pub fn shell_reach(&self, step: &Step) -> Result<f32> {
        let b = if step.shell_smooth > 0.0 || step.shell_far > 0.0 {
            f32::from_bits(bytemuck::cast_slice::<u8, u32>(&self.submit(&[], Some((&self.shell.smooth_bound, 0, 4)))?)[0])
        } else {
            self.read_mem(self.at.results + HEIGHT_BOUND * 2, 1)?[0]
        };
        Ok(shell::reach(b, step.shell, step.height_max * step.shell))
    }

    /// For checks: the last frame shown (RGBA8 a pixel, the picture's width times `up` by its
    /// height times `up`), waiting for the GPU.
    pub fn read_shown(&self) -> Result<Vec<u32>> {
        let (sw, sh) = (self.view_width * self.cfg.up, self.cfg.height * self.cfg.up);
        let rows = padded(sw);
        let b = self.submit(&[], Some((&self.img, self.shown_at as u64 * 4, (rows * sh) as u64 * 4)))?;
        let all: &[u32] = bytemuck::cast_slice(&b);
        Ok((0..sh).flat_map(|y| (0..sw).map(move |x| (y * rows + x) as usize)).map(|i| all[i]).collect())
    }

    /// For checks: the triangle each pixel of the last frame drawn showed (its index + 1, 0 for
    /// none), waiting for the GPU.
    pub fn read_triangles(&self) -> Result<Vec<u32>> {
        let (w, h) = (self.view_width, self.cfg.height);
        let rows = point_rows(w);
        let b = self.submit(&[], Some((&self.img, self.ids_at as u64 * 4, (rows * h) as u64 * 4)))?;
        let all: &[u32] = bytemuck::cast_slice(&b);
        Ok((0..h).flat_map(|y| (0..w).map(move |x| (y * rows + 4 * x) as usize)).map(|i| all[i]).collect())
    }

    /// For profiling: each dispatch of an octave's plan (h x w, `layers`) run and waited on by
    /// itself, its milliseconds totalled by kernel (by shape too with PROFILE_SHAPES set).
    pub fn profile(&mut self, h: u32, w: u32, layers: &[&str]) -> Result<Vec<(String, f32, usize)>> {
        let layers = self.nets[0].indices(layers)?;
        let (plan, _) = self.plan(0, h, w, &layers, false);
        self.submit(&plan.ops, None)?;
        let mut total: HashMap<String, (f32, usize)> = HashMap::new();
        let by_shape = std::env::var_os("PROFILE_SHAPES").is_some();
        for op in &plan.ops {
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
            let t0 = std::time::Instant::now();
            self.submit(std::slice::from_ref(op), None)?;
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            let e = total.entry(op_label(op.0, &op.1, by_shape)).or_default();
            e.0 += ms;
            e.1 += 1;
        }
        let mut out: Vec<_> = total.into_iter().map(|(k, (ms, n))| (k, ms, n)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(out)
    }

    /// For checks: the gradient of the objective (each of `layers`' mean square over its own
    /// value, summed) with respect to `rgb` (HWC, 0..1, h x w), and the activations of
    /// `wanted` layers, HWC.
    pub fn input_gradient(&mut self, rgb: &[f32], h: u32, w: u32, layers: &[&str], wanted: &[&str]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let layers = self.nets[0].indices(layers)?;
        let (plan, outs) = self.plan(0, h, w, &layers, false);
        self.queue.write_buffer(&self.mem, (self.at.acts + plan.input.off) as u64 * 4, bytemuck::cast_slice(rgb));
        self.submit(&plan.ops, None)?;
        let g = self.read_mem(self.at.grads + plan.input.off, h * w * 3)?;
        let mut acts = Vec::new();
        for name in wanted {
            let t = outs[self.nets[0].index(name)?];
            acts.push(self.read_mem(self.at.acts + t.off, t.h * t.w * t.c)?);
        }
        // The objective's mean square has a factor 2 the seed leaves out.
        Ok((g.iter().map(|v| v * 2.0).collect(), acts))
    }

    /// For checks: the gradient of `class`'s logit with respect to `rgb` (HWC, 0..1, h x w), and
    /// how sure of it the network is (dream.wgsl confidence's softmax).
    pub fn class_gradient(&mut self, rgb: &[f32], h: u32, w: u32, class: u32) -> Result<(Vec<f32>, f32)> {
        if !self.class_row(0, class as i32) {
            return Err(format!("no classifier, or no class {class}"));
        }
        let (plan, outs) = self.plan(0, h, w, &[], true);
        self.queue.write_buffer(&self.mem, (self.at.acts + plan.input.off) as u64 * 4, bytemuck::cast_slice(rgb));
        self.submit(&plan.ops, None)?;
        // The seed leaves out the mean pooling's 1 / pixels.
        let t = outs[self.nets[0].order.len() - 1];
        let g = self.read_mem(self.at.grads + plan.input.off, h * w * 3)?;
        // confidence divides 1 by 1 - p.
        let c = self.nets[0].classes.as_ref().unwrap();
        let slot = self.at.results + (RESULTS - 2) * 2;
        self.queue.write_buffer(&self.mem, slot as u64 * 4, bytemuck::cast_slice(&[1.0f32, 1.0]));
        let p = Params { m: t.h * t.w, k: t.c, a_off: self.at.acts + t.off, b_off: c.all, x0: c.bias, x1: class, x2: c.count, x3: self.at.results + SURE * 2, c_off: slot, ..Default::default() };
        self.submit(&[(Kernel::Confidence, p, [1, 1, 1])], None)?;
        let sure = 1.0 - 1.0 / self.read_mem(slot, 1)?[0];
        Ok((g.iter().map(|v| v / (t.h * t.w) as f32).collect(), sure))
    }

    /// Whether network `net` dreams towards `class` (< 0: none), its row of the classifier put
    /// in `mem` if it isn't already.
    fn class_row(&mut self, net: usize, class: i32) -> bool {
        let Some(Classifier { w, row: at, count, .. }) = &self.nets[net].classes else { return false };
        // A row a class, a weight for each of the last block's channels.
        let channels = w.len() / *count as usize;
        if class < 0 || class as u32 >= *count {
            return false;
        }
        if self.class_rows[net] != Some(class as u32) {
            let row = &w[class as usize * channels..(class as usize + 1) * channels];
            self.queue.write_buffer(&self.mem, *at as u64 * 4, bytemuck::cast_slice(row));
            self.class_rows[net] = Some(class as u32);
        }
        true
    }
}
