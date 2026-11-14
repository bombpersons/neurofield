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
use level::{Feedback, Level, Raster, BANDS, VIEW};
pub use level::{Face, View};

pub type Result<T> = std::result::Result<T, String>;

/// One pixel of the caller's picture: the page it shows (`NO_PAGE` for none: nothing dreams
/// there), where on the page (0..1 each way) and how that changes a pixel right and a pixel down,
/// how strongly the dream shows there (0..1: fog, say), and how far along the page the line of
/// sight moves for each unit it rises off the surface towards the eye (the point toward the eye,
/// minus the point, along the page's two directions, over its distance off the surface; in pages,
/// so the relief's height is in pages too): for the dream's relief.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Surface {
    pub page: f32,
    pub uv: [f32; 2],
    pub duv_dx: [f32; 2],
    pub duv_dy: [f32; 2],
    pub strength: f32,
    pub slope: [f32; 2],
}

pub const NO_PAGE: f32 = -1.0;

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
    pub const NONE: Surface = Surface { page: NO_PAGE, uv: [0.0; 2], duv_dx: [0.0; 2], duv_dy: [0.0; 2], strength: 0.0, slope: [0.0; 2] };
}

/// Fixed when the renderer is made.
#[derive(Clone, Debug)]
pub struct Config {
    /// GoogLeNet's weights (train/export_googlenet.py).
    pub weights: PathBuf,
    /// The layers each octave dreams with: `conv1`, ..., `inception4c` (or just `4c`), ...;
    /// several are summed, each counting alike.
    pub layers: Vec<String>,
    pub fine_layers: Vec<String>,
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
}

impl Config {
    pub fn new(weights: impl Into<PathBuf>) -> Config {
        Config {
            weights: weights.into(),
            layers: vec!["inception4c".into()],
            fine_layers: vec!["inception4c".into()],
            max_width: 448,
            height: 240,
            up: 3,
            page_res: 256,
            max_pages: 512,
            max_touched: 128,
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
        Step { lr: 0.012, lr_fine: 0.008, decay: 0.002, limit: 0.6, jitter: 4, jitter_fine: 6, sharpness: 1.0, colour: 0.25, colour_blur: 3, take_over: 0.01, own_max: 1.0, relief: 0.0, relief_rate: 0.05, remember_every: 4, coarse_every: 1, fine_split: 2, fine_every: 1, fine_offset: 0 }
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
const SURFACE: u32 = 10;
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
}
const KERNELS: [&str; 23] = [
    "matmul", "im2col", "col2im", "maxpool", "maxpool_back", "relu_back", "clear", "reduce", "reduce_finish", "seed",
    "prepare", "scatter", "clear_acc", "apply", "clear_pages", "show", "blur_picture", "raise", "screen_step", "bake", "warp", "store", "gbuffer",
];
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

/// Layers in order, as torchvision's GoogLeNet runs them.
const ORDER: [&str; 13] = [
    "conv1", "maxpool1", "conv2", "conv3", "maxpool2", "inception3a", "inception3b", "maxpool3", "inception4a", "inception4b",
    "inception4c", "inception4d", "inception4e",
];

/// The layer named, as its place in `ORDER` (`4c` for `inception4c` will do).
fn layer_index(name: &str) -> Result<usize> {
    let full = if name.len() == 2 && name.as_bytes()[0].is_ascii_digit() { format!("inception{name}") } else { name.to_string() };
    ORDER.iter().position(|l| *l == full).ok_or(format!("no layer {name} (one of {ORDER:?})"))
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
    /// `out`'s channels from `c0` are conv(inp), after ReLU.
    Conv { conv: Conv, inp: T, out: T, c0: u32 },
    Pool { inp: T, out: T, idx: u32, stride: u32, pad: u32 },
}

/// The dispatches of one octave: the network forward, the objective, and back to its input.
struct Plan {
    ops: Vec<(Kernel, Params, [u32; 3])>,
    input: T,
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

/// Where things are in `mem`.
#[derive(Clone, Copy, Debug)]
struct Layout {
    /// The surfaces given, twice (this view's and the last's, which laying the screen layer on
    /// the pages still needs while this frame's are written).
    surfaces: u32,
    surface_size: u32,
    smooth: u32,
    /// Each page's relief, (page_res / RELIEF_CELL) squared floats.
    relief: u32,
    /// The screen layer, twice (this frame's and the last's, for carrying it along): rgb and
    /// ownership a pixel of the shown frame, at its biggest.
    screen: u32,
    screen_size: u32,
    /// Each pixel's `Motion`.
    motion: u32,
    /// This view and the last (VIEW floats each), the creatures' depth a pixel, and each
    /// pixel's 1/z twice (this view's and the last's), for the G-buffer made on the GPU.
    views: u32,
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
    floats: u32,
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
    /// Where the triangle each pixel shows goes in `img` (u32s).
    ids_at: u32,
    /// The pages given to faces, each page's face, the frame each was last seen, and the pages
    /// in each band of rows by the last report.
    page_of: HashMap<Face, u32>,
    owner: Vec<Option<Face>>,
    page_seen: Vec<u64>,
    band_pages: Vec<Vec<usize>>,
    /// The last view drawn and its width, and which of the two 1/z is its.
    last_view: Option<(View, u32)>,
    nearest_flip: bool,
    readback: wgpu::Buffer,
    cfg: Config,
    at: Layout,
    stem: [Conv; 3],
    blocks: Vec<Inception>,
    layers: Vec<usize>,
    fine_layers: Vec<usize>,
    plans: HashMap<(u32, u32, bool), Plan>,
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
    next_set: usize,
    /// Milliseconds the last frame took, start to finish, waiting for the GPU.
    pub last_ms: f32,
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

fn rows(n: u32) -> [u32; 2] {
    [n.min(65535), n.div_ceil(65535)]
}

fn conv_out(n: u32, k: u32, s: u32, pad: u32) -> u32 {
    (n + 2 * pad - k) / s + 1
}

/// PyTorch's ceil_mode size for a 3x3 pool, whose last window must start inside the input.
fn pool_out(n: u32, s: u32, pad: u32) -> u32 {
    let mut o = (n + 2 * pad - 3).div_ceil(s) + 1;
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
        let r = rows(p.m.div_ceil(32));
        self.push(Kernel::Matmul, p, [p.n.div_ceil(32), r[0], r[1]]);
    }

    fn conv(&mut self, conv: Conv, inp: T, out: T, c0: u32) {
        assert_eq!(conv.cin, inp.c, "a convolution reads all of its input's channels");
        self.layers.push(Layer::Conv { conv, inp, out, c0 });
        let px = out.h * out.w;
        let base = Params { m: px, n: conv.cout, c_off: self.act(out) + c0, c_rs: out.c, bias_off: conv.b, b_off: conv.w, flags: RELU, ..Default::default() };
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

    fn pool(&mut self, inp: T, stride: u32, pad: u32) -> T {
        let out = self.alloc(pool_out(inp.h, stride, pad), pool_out(inp.w, stride, pad), inp.c);
        let idx = self.alloc(out.h, out.w, out.c).off;
        self.layers.push(Layer::Pool { inp, out, idx, stride, pad });
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
        let pooled = self.pool(x, 1, 1);
        self.conv(b.b4, pooled, out, c1 + c2 + c3);
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
                Layer::Conv { conv, inp, out, c0 } => {
                    let px = out.h * out.w;
                    let r = rows(px);
                    let g_out = self.grad(out) + c0;
                    self.push(Kernel::ReluBack, Params { m: px, a_off: self.act(out) + c0, c_off: g_out, a_rs: out.c, x0: conv.cout, ..Default::default() }, [conv.cout.div_ceil(64), r[0], r[1]]);
                    let base = Params { m: px, k: conv.cout, a_off: g_out, a_rs: out.c, a_cs: 1, b_off: conv.w, bias_off: NONE, ..Default::default() };
                    if conv.k == 1 && conv.stride == 1 {
                        self.matmul(Params { n: conv.cin, b_rs: conv.cin, b_cs: 1, c_off: self.grad(inp), c_rs: inp.c, flags: ACCUMULATE, ..base });
                    } else {
                        let big_k = conv.cin * conv.k * conv.k;
                        self.matmul(Params { n: big_k, b_rs: big_k, b_cs: 1, c_off: self.at.gcol, c_rs: big_k, ..base });
                        let r = rows(inp.h * inp.w);
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
                        self.push(Kernel::Col2im, p, [conv.cin.div_ceil(64), r[0], r[1]]);
                    }
                }
                Layer::Pool { inp, out, idx, stride, pad } => {
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
                        x4: stride,
                        x5: pad,
                        ..Default::default()
                    };
                    self.push(Kernel::MaxpoolBack, p, [inp.c.div_ceil(64), r[0], r[1]]);
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
        let mut t = read_safetensors(&cfg.weights)?;
        let mut data: Vec<f32> = Vec::new();
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
        let stem = [conv(&mut t, "conv1", 2)?, conv(&mut t, "conv2", 1)?, conv(&mut t, "conv3", 1)?];
        let mut blocks = Vec::new();
        for name in ["3a", "3b", "4a", "4b", "4c", "4d", "4e"] {
            let n = format!("inception{name}");
            blocks.push(Inception {
                b1: conv(&mut t, &format!("{n}.branch1"), 1)?,
                b2: [conv(&mut t, &format!("{n}.branch2.0"), 1)?, conv(&mut t, &format!("{n}.branch2.1"), 1)?],
                b3: [conv(&mut t, &format!("{n}.branch3.0"), 1)?, conv(&mut t, &format!("{n}.branch3.1"), 1)?],
                b4: conv(&mut t, &format!("{n}.branch4.1"), 1)?,
            });
        }
        drop(conv);
        let weights_end = align(data.len() as u32);
        let layers = cfg.layers.iter().map(|l| layer_index(l)).collect::<Result<Vec<_>>>()?;
        let fine_layers = cfg.fine_layers.iter().map(|l| layer_index(l)).collect::<Result<Vec<_>>>()?;
        if layers.is_empty() || fine_layers.is_empty() {
            return Err("no layers to dream with".into());
        }
        if layers.len().max(fine_layers.len()) as u32 >= RESULTS / 2 {
            return Err("too many layers".into());
        }

        // Where everything goes: the network's room comes from a dry run at the biggest octave.
        let (w0, h0, up) = (cfg.max_width, cfg.height, cfg.up);
        let big = ((h0 * up / 2).max(h0), (w0 * up / 2).max(w0));
        let max_px = big.0 * big.1;
        let mut at = Layout {
            surfaces: weights_end,
            smooth: 0,
            relief: 0,
            screen: 0,
            screen_size: 0,
            motion: 0,
            views: 0,
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
            floats: 0,
        };
        at.surface_size = align(w0 * h0 * SURFACE);
        at.smooth = at.surfaces + 2 * at.surface_size;
        at.relief = at.smooth + align(3 * w0.div_ceil(SMOOTH) * h0.div_ceil(SMOOTH));
        at.screen = at.relief + align(cfg.max_pages * (cfg.page_res / RELIEF_CELL).pow(2));
        at.screen_size = align(4 * w0 * up * h0 * up);
        at.motion = at.screen + 2 * at.screen_size;
        at.views = at.motion + align(w0 * h0 * MOTION);
        at.creatures = at.views + align(2 * VIEW);
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
        at.partials = at.lookup + align(max_px * LOOKUP);
        at.results = at.partials + align(PARTIALS * 2);
        at.acts = at.results + align(RESULTS * 2);
        let (room, col) = {
            let (b, _) = Self::build_net(&stem, &blocks, at, big.0, big.1, ORDER.len() - 1);
            (align(b.next), align(b.col_max))
        };
        at.grads = at.acts + room;
        at.col = at.grads + room;
        at.gcol = at.col + col;
        at.floats = at.gcol + col;

        let mem_bytes = at.floats as u64 * 4;
        let atlas_bytes = cfg.max_pages as u64 * (cfg.page_res * cfg.page_res) as u64 * 8;
        let acc_bytes = cfg.max_touched as u64 * (cfg.page_res * cfg.page_res) as u64 * 16;
        // The picture given, then the frame shown, each with rows padded for texture copies,
        // then the triangle each pixel shows, likewise.
        let shown_at = padded(w0) * h0;
        let ids_at = shown_at + padded(w0 * up) * h0 * up;
        let img_bytes = (ids_at + padded(w0) * h0) as u64 * 4;
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
            page_of: HashMap::new(),
            owner: vec![None; cfg.max_pages as usize],
            page_seen: vec![0; cfg.max_pages as usize],
            band_pages: vec![Vec::new(); BANDS as usize],
            last_view: None,
            nearest_flip: false,
            device,
            queue,
            readback,
            at,
            stem,
            blocks,
            layers,
            fine_layers,
            plans: HashMap::new(),
            frames: 0,
            last,
            shown: Vec::new(),
            shown_at,
            view_flip: false,
            ops: Vec::new(),
            clears: Vec::new(),
            next_set: 1,
            last_ms: 0.0,
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

    /// The network on an h x w input, up to layer `deepest` of `ORDER`, and each layer's
    /// output by its place in ORDER.
    fn build_net(stem: &[Conv; 3], blocks: &[Inception], at: Layout, h: u32, w: u32, deepest: usize) -> (Builder, Vec<T>) {
        let mut b = Builder { at, next: 0, col_max: 0, layers: Vec::new(), ops: Vec::new() };
        let mut x = b.alloc(h, w, 3);
        let mut outs = Vec::new();
        for name in ORDER.iter().take(deepest + 1) {
            x = match *name {
                "conv1" => b.conv_new(stem[0], x),
                "conv2" => b.conv_new(stem[1], x),
                "conv3" => b.conv_new(stem[2], x),
                "maxpool1" | "maxpool2" | "maxpool3" => b.pool(x, 2, 0),
                _ => {
                    let k = ["inception3a", "inception3b", "inception4a", "inception4b", "inception4c", "inception4d", "inception4e"]
                        .iter()
                        .position(|n| n == name)
                        .unwrap();
                    b.inception(&blocks[k], x)
                }
            };
            outs.push(x);
        }
        (b, outs)
    }

    /// One octave's dispatches for an h x w input dreaming with `layers` (places in ORDER).
    fn plan(&self, h: u32, w: u32, layers: &[usize]) -> (Plan, Vec<T>) {
        let deepest = *layers.iter().max().unwrap();
        let (mut b, outs) = Self::build_net(&self.stem, &self.blocks, self.at, h, w, deepest);
        let input = T { off: 0, h, w, c: 3 };
        let room = b.next;
        b.clear(self.at.grads, room);
        for (i, &l) in layers.iter().enumerate() {
            let t = outs[l];
            let n = t.h * t.w * t.c;
            b.reduce(0, b.act(t), n, i as u32);
            let r = rows(n.div_ceil(4096));
            let p = Params { m: n, a_off: b.act(t), b_off: self.at.results + i as u32 * 2, c_off: b.grad(t), ..Default::default() };
            b.push(Kernel::Seed, p, [64, r[0], r[1]]);
        }
        b.backward();
        (Plan { ops: b.ops, input }, outs)
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
        let shown = self.submit(&ops, Some((&self.img.clone(), self.shown_at as u64 * 4, (sw * h * self.cfg.up) as u64 * 4)))?;
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
        self.encode_given(encoder, picture, width, given, step, out)
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
        self.encode_given(encoder, picture, width, given, step, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_given(&mut self, encoder: &mut wgpu::CommandEncoder, picture: &wgpu::Texture, width: u32, given: Given, step: &Step, out: &wgpu::Texture) -> Result<()> {
        let (h, up) = (self.cfg.height, self.cfg.up);
        if width > self.cfg.max_width || picture.width() < width || picture.height() < h || out.width() < width * up || out.height() < h * up {
            return Err(format!("a {width} x {h} frame from a {} x {} picture into {} x {}", picture.width(), picture.height(), out.width(), out.height()));
        }
        let t0 = std::time::Instant::now();
        let sw = width * up;
        let ops = self.run(width, given, step, padded(width), padded(sw))?;
        let copy = |rows: u32, offset: u64| wgpu::TexelCopyBufferInfo {
            buffer: &self.img,
            layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(rows * 4), rows_per_image: None },
        };
        let size = |width: u32, height: u32| wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        encoder.copy_texture_to_buffer(picture.as_image_copy(), copy(padded(width), 0), size(width, h));
        if let Given::Gpu { view, .. } = given {
            self.raster.record(&self.queue, encoder, &view, width, h);
            encoder.copy_texture_to_buffer(self.raster.ids.as_image_copy(), copy(padded(width), self.ids_at as u64 * 4), size(width, h));
            encoder.clear_buffer(&self.seen_buf, 0, None);
        }
        self.record(encoder, &ops)?;
        if let (Given::Gpu { .. }, Some(f)) = (given, self.feedback.as_mut()) {
            f.copy(encoder, &self.seen_buf);
        }
        encoder.copy_buffer_to_texture(copy(padded(sw), self.shown_at as u64 * 4), out.as_image_copy(), size(sw, h * up));
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

    /// The level to draw (`encode_view`): its triangles' corners, in units of a page (a face is
    /// a unit square of a triangle's plane, by its main axis). Given again whenever they change
    /// (doors move, say): faces keep their pages, by where they are.
    pub fn set_level(&mut self, tris: &[[f32; 9]]) -> Result<()> {
        let level = level::build(tris);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buffer = |label: &str, size: u64| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.div_ceil(4) * 4, usage: storage, mapped_at_creation: false });
        self.level_buf = buffer("dream level", level.data.len() as u64 * 4);
        let words = (level.faces.len() as u32).div_ceil(32).max(1);
        let seen_bytes = (BANDS * words) as u64 * 4;
        self.seen_buf = buffer("dream faces seen", seen_bytes);
        self.queue.write_buffer(&self.level_buf, 0, bytemuck::cast_slice(&level.data));
        self.raster.set_level(&self.device, &self.queue, &level.vertices);
        self.bind = Self::bind_group(&self.device, &self.layout, [&self.params, &self.mem, &self.atlas, &self.acc, &self.img, &self.level_buf, &self.seen_buf]);
        self.levels += 1;
        self.feedback = Some(Feedback::new(&self.device, seen_bytes, self.levels));
        self.band_pages = vec![Vec::new(); BANDS as usize];
        self.level = Some(level);
        self.write_page_table();
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
        Some(p as u32)
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
            // Its colour, then how much of the wall it owns.
            for channel in 0..2 {
                let bake = Params { a_off: self.surfaces(), a_cs: up, b_off: set.of, c_rs: self.screen(), k: r, x0: w, x1: h, m: rows.end * up, x5: rows.start * up, x6: channel, ..Default::default() };
                ops.push((Kernel::ClearAcc, Params { k: r, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), n]));
                ops.push((Kernel::Bake, bake, [(w * up).div_ceil(16), (rows.len() as u32 * up).div_ceil(16), 1]));
                ops.push((Kernel::Store, Params { k: r, x0: set.pages, x2: channel, f0: step.limit, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), n]));
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
            self.ops.push((Kernel::ClearPages, Params { k: r, x0: self.at.page_list, c_off: self.at.relief, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), list.len() as u32]));
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
        let mut gbuffer = None;
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
                Given::Cpu(surfaces, _) => self.queue.write_buffer(&self.mem, self.surfaces() as u64 * 4, bytemuck::cast_slice(surfaces)),
                Given::Gpu { view, creatures, .. } => {
                    let level = self.level.as_ref().ok_or("no level given")?;
                    let last = self.last_view.filter(|&(_, w)| w == width && carry && !fresh);
                    let mut views = view.floats(width).to_vec();
                    views.extend(last.map_or([0.0; VIEW as usize], |(v, w)| v.floats(w)));
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
                        m: words,
                        n: h.div_ceil(BANDS),
                        flags: if last.is_some() { MOTION_FROM_LAST } else { 0 },
                        ..Default::default()
                    };
                    gbuffer = Some((Kernel::Gbuffer, p, [width.div_ceil(16), h.div_ceil(16), 1]));
                }
                Given::Still => {}
            }
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
            let p = Params { bias_off: self.at.motion, a_off: self.surfaces(), b_rs: from, c_rs: self.screen(), a_cs: up, k: r, x0: width, x1: h, flags: fill, ..Default::default() };
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
            let j = step.jitter;
            let roll = if j > 0 { [self.random() % (2 * j + 1) + width - j, self.random() % (2 * j + 1) + h - j] } else { [0, 0] };
            octaves.push((1, [0, 0], [width, h], roll, step.lr, false, s0.decay));
        }
        let every = step.fine_every.max(1) as u64;
        if step.lr_fine > 0.0 && (self.frames + step.fine_offset as u64) % every == 0 {
            let split = step.fine_split.max(1);
            // Whole 16s, so the network's sizes stay few.
            let (fw, fh) = ((width * up / split) / 16 * 16, (h * up / split) / 16 * 16);
            let q = ((self.frames / every) % (split * split) as u64) as u32;
            let jx = fw / 4;
            let jy = fh / 4;
            let x0 = ((q % split) * fw + self.random() % (jx + 1)).saturating_sub(jx / 2).min(width * up - fw);
            let y0 = ((q / split) * fh + self.random() % (jy + 1)).saturating_sub(jy / 2).min(h * up - fh);
            let j = step.jitter_fine;
            let roll = if j > 0 { [self.random() % (2 * j + 1) + fw - j, self.random() % (2 * j + 1) + fh - j] } else { [0, 0] };
            // The first octave of the frame decays the pages.
            let decay_at = if octaves.is_empty() { s0.decay } else { s0.ones };
            octaves.push((up, [x0, y0], [fw, fh], roll, step.lr_fine, true, decay_at));
        }
        let dreamed = !octaves.is_empty();
        // The pages' decay is owed until they're next dreamed on; on the screen layer they're
        // dreamed on in its stead, and don't decay.
        if dreamed {
            for &p in if still { &self.view_pages } else { &pages } {
                self.last[p] = self.frames;
            }
        }
        if n_slots > 0 || (still && !self.view_pages.is_empty()) {
            for (scale, crop, size, roll, lr, fine, decay_at) in octaves {
                let key = (size[1], size[0], fine);
                if !self.plans.contains_key(&key) {
                    let layers = if fine { self.fine_layers.clone() } else { self.layers.clone() };
                    let (plan, _) = self.plan(size[1], size[0], &layers);
                    self.plans.insert(key, plan);
                }
                let plan = &self.plans[&key];
                let x = self.at.acts + plan.input.off;
                let gx = self.at.grads + plan.input.off;
                if !still {
                    ops.push((Kernel::ClearAcc, Params { k: r, ..Default::default() }, [r.div_ceil(16), r.div_ceil(16), n_slots]));
                }
                let p = Params { m: size[1], n: size[0], c_off: x, b_off: self.at.lookup, x2: scale, x3: crop[0], x4: crop[1], x5: roll[0], x6: roll[1], ..frame };
                ops.push((Kernel::Prepare, p, [size[0].div_ceil(16), size[1].div_ceil(16), 1]));
                ops.extend(plan.ops.iter().copied());
                let mean = RESULTS - 2;
                let px = size[0] * size[1];
                ops.push((Kernel::Reduce, Params { m: px, a_off: gx, b_off: self.at.lookup, c_off: self.at.partials, x0: 1, ..Default::default() }, [PARTIALS, 1, 1]));
                ops.push((Kernel::ReduceFinish, Params { m: PARTIALS, a_off: self.at.partials, c_off: self.at.results + mean * 2, ..Default::default() }, [1, 1, 1]));
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
                        x2: scale,
                        x3: crop[0],
                        x4: crop[1],
                        x5: roll[0],
                        x6: roll[1],
                        f0: lr,
                        f1: step.colour,
                        f2: step.limit,
                        f3: step.take_over,
                        ..Default::default()
                    };
                    let (cw, ch) = (size[0] * up / scale, size[1] * up / scale);
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
        let p = Params { m: sh, n: sw, x2: up, c_off: self.shown_at, b_rs: shown_rows, f0: step.sharpness, ..frame };
        ops.push((Kernel::Show, p, [sw.div_ceil(16), sh.div_ceil(16), 1]));
        Ok(ops)
    }

    /// Record `ops` into `enc`, their parameters written to the queue (to land before it runs).
    fn record(&self, enc: &mut wgpu::CommandEncoder, ops: &[(Kernel, Params, [u32; 3])]) -> Result<()> {
        if ops.len() > MAX_OPS {
            return Err(format!("{} dispatches, room for {MAX_OPS}", ops.len()));
        }
        let mut slots = vec![0u8; ops.len() * SLOT as usize];
        for (i, (_, p, _)) in ops.iter().enumerate() {
            slots[i * SLOT as usize..][..std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        self.queue.write_buffer(&self.params, 0, &slots);
        let mut pass = enc.begin_compute_pass(&Default::default());
        for (i, (kernel, _, groups)) in ops.iter().enumerate() {
            pass.set_pipeline(&self.pipelines[*kernel as usize]);
            pass.set_bind_group(0, &self.bind, &[(i as u64 * SLOT) as u32]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        Ok(())
    }

    /// Run `ops` and, with `read`, copy (buffer, byte offset, bytes) back.
    fn submit(&self, ops: &[(Kernel, Params, [u32; 3])], read: Option<(&wgpu::Buffer, u64, u64)>) -> Result<Vec<u8>> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.record(&mut enc, ops)?;
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

    /// For profiling: each dispatch of an octave's plan (h x w, `layers`) run and waited on by
    /// itself, its milliseconds totalled by kernel, and the matrix multiplies by shape.
    pub fn profile(&mut self, h: u32, w: u32, layers: &[&str]) -> Result<Vec<(String, f32, usize)>> {
        let layers = layers.iter().map(|l| layer_index(l)).collect::<Result<Vec<_>>>()?;
        let (plan, _) = self.plan(h, w, &layers);
        self.submit(&plan.ops, None)?;
        let mut total: HashMap<String, (f32, usize)> = HashMap::new();
        for op in &plan.ops {
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
            let t0 = std::time::Instant::now();
            self.submit(std::slice::from_ref(op), None)?;
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            let p = &op.1;
            let name = match op.0 {
                Kernel::Matmul => format!("matmul {}", if p.bias_off == NONE { "back" } else { "fwd" }),
                k => format!("{k:?}"),
            };
            let e = total.entry(name).or_default();
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
        let layers = layers.iter().map(|l| layer_index(l)).collect::<Result<Vec<_>>>()?;
        let (plan, outs) = self.plan(h, w, &layers);
        let x: Vec<f32> = rgb.iter().map(|v| 2.0 * v - 1.0).collect();
        self.queue.write_buffer(&self.mem, (self.at.acts + plan.input.off) as u64 * 4, bytemuck::cast_slice(&x));
        self.submit(&plan.ops, None)?;
        let g = self.read_mem(self.at.grads + plan.input.off, h * w * 3)?;
        let mut acts = Vec::new();
        for name in wanted {
            let t = outs[layer_index(name)?];
            acts.push(self.read_mem(self.at.acts + t.off, t.h * t.w * t.c)?);
        }
        // d/d(rgb) = 2 d/dx, and the objective's mean square has a factor 2 the seed leaves out.
        Ok((g.iter().map(|v| v * 4.0).collect(), acts))
    }
}
