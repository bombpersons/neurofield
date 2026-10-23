"""Generate seamless creature-skin textures with SDXL on a Colab GPU.

    training-data-gen/photo/colab_textures.sh [session] [per-motif]   # drives this, see there

One chunk per `colab exec` into a persistent kernel: the first call installs
diffusers and builds the pipeline, later calls reuse it from the kernel's
globals. Every Conv2d in the UNet and VAE pads circularly, so each image wraps
seamlessly. Each chunk generates for MINUTES, always topping up the motif with
the fewest images, and saves the 768x768 original plus a 256x256 8-bit RGB copy
(what datagen's photo library wants) and one line of credits per image. At the
end of a chunk the new 256s and their credits go into /content/skins_gen/
chunks/NNNN.tar for the local driver to download, and the whole lot is copied
to Google Drive too when one is mounted at /content/drive (only the 256s and
credits, ~200 MB: the originals stay on the VM).

SET = "deep" makes wall textures for the dungeon's deep layers instead, in
/content/deep_gen: whole frame-filling walls of teeth, eyes, guts and so on.

The model is RealVisXL V4.0 Lightning, a photoreal SDXL fine-tune (Open
RAIL++-M, like SDXL 1.0), with madebyollin's fp16-safe VAE (MIT). Plain SDXL
with the SDXL-Lightning UNet came out plastic at 1024 and broke down into
psychedelic noise at 512, so this renders at 768 (~4.5 s an image on a T4).
"""

import json
import os
import random
import shutil
import tarfile
import time

MINUTES = 6  # short chunks: a reclaimed VM loses only the chunk in progress
PER_MOTIF = 160
SIZE, OUT_SIZE, BATCH, STEPS, CFG = 768, 256, 4, 6, 1.5
OUT = "/content/skins_gen"
DRIVE = "/content/drive/MyDrive/neurofield_skins_gen"
MODEL = "SG161222/RealVisXL_V4.0_Lightning + madebyollin/sdxl-vae-fp16-fix"
LICENCE = "CreativeML Open RAIL++-M (RealVisXL V4.0, SDXL 1.0); VAE fix MIT"

STYLE = "macro photograph, extreme close-up, {wet}, glossy, harsh clinical ring flash, sharp focus, photoreal"
HAIR = (
    "photograph of {subject}, hundreds of fine thin individual hair strands, hair texture filling the frame, "
    "seen from 30 cm, {wet}, harsh flash, sharp focus, photoreal"
)
NEGATIVE = (
    "person, human face, portrait, hands, text, letters, watermark, logo, signature, background, sky, plate, "
    "table, tray, black background, empty space, cartoon, illustration, painting, drawing, 3d render, cgi, "
    "plastic, toy, abstract, pattern, blurry, bokeh, depth of field, vignette, frame, border, low quality, "
    "whole animal, legs, antennae, animal head"
)
WET = [
    "glistening wet", "dripping with clear slime", "coated in thick mucus", "slick with viscous fluid",
    "oily wet sheen", "beaded with moisture", "smeared with translucent goo", "sticky and moist",
]
COLOURS = [
    "pink", "deep red", "dark purple", "sickly yellow", "greenish grey", "pale white", "dark brown",
    "bluish grey", "mottled red and yellow", "bruised purple and green", "fleshy pink and red", "ochre",
    "grey and pink", "black and red", "pallid grey", "jaundiced yellow and pink",
]
# Surfaces tile edge to edge; things sit centred on flesh.
SURFACE = "{colour}, close-up of the surface texture only, filling the frame, top-down view"
THING = "centred, embedded in {colour} raw flesh filling the frame"
SUBJECTS = {
    "flesh": [
        "raw meat tissue", "exposed muscle fibres", "tripe stomach lining", "intestinal lining",
        "raw liver surface", "tongue papillae", "skinless raw flesh", "raw organ tissue",
        "marbled fatty raw flesh", "mucous membrane", "swollen gum tissue", "blistered raw skin",
        "raw chicken skin with goosebumps", "lumpy tumorous flesh", "stomach rugae folds",
    ],
    "veins": [
        "translucent skin with branching veins", "network of engorged blood vessels over a membrane",
        "bulging veins under wet skin", "dense capillary network on tissue", "placenta surface with vessels",
        "varicose veins under thin skin", "bloodshot membrane covered in red veins",
        "veiny translucent membrane over dark organs",
    ],
    "scales": [
        "snake scales", "lizard scales", "iridescent fish scales", "crocodile scutes", "pangolin scales",
        "overlapping reptile scales", "gecko skin tubercles", "carp scales with slime", "eel skin",
        "shed snake skin over flesh",
    ],
    "chitin": [
        "surface of a beetle carapace", "surface of a crab shell", "overlapping cockroach shell plates",
        "iridescent insect exoskeleton plates", "surface of a lobster shell", "segmented larva skin",
        "woodlouse armour plates", "centipede body segments", "swollen tick abdomen skin", "shrimp shell plates",
    ],
    "hair": [
        "long wet black human hair", "shaggy brown animal fur", "dense dark body hair on pale clammy skin",
        "matted wet grey fur", "long greasy blonde hair", "coarse red hair growing from pink skin",
        "wet curly black hair", "patchy mangy dog fur over raw pink skin", "wet hair clumped over a pink scalp",
        "long stringy grey hair matted with slime", "coarse wiry white hair on wrinkled skin",
        "greasy auburn hair tangled in clumps", "shaggy wet goat fur", "dark matted fur crusted with blood",
    ],
    "wrinkles": [
        "deeply wrinkled skin folds", "wrinkled elephant hide", "wrinkled hairless pink skin",
        "folds of loose saggy skin", "creased rhinoceros hide", "creased wrinkled flesh", "wrinkled knuckle skin",
        "puckered scar tissue", "shriveled wrinkled membrane", "wrinkled walrus hide",
    ],
    "bone": [
        "exposed bone surface", "spongy cancellous bone", "bone with shreds of flesh", "skull suture",
        "glistening cartilage", "cracked ivory", "joint surface with cartilage", "porous bone marrow",
        "fractured bone with tissue", "bone with periosteum membrane",
    ],
    "eyes": [
        "a single bulging wet eyeball", "a cluster of small glistening eyes", "a reptile eye with slit pupil",
        "a goat eye with horizontal pupil", "a bloodshot eyeball", "several eyeballs of different sizes",
        "a milky clouded eye", "a yellow eye with veiny sclera", "an octopus eye", "a compound insect eye",
    ],
    "teeth": [
        "rows of teeth growing from gums", "a single long fang in gum tissue", "a cluster of crooked teeth in raw gums",
        "rows of shark teeth", "rotten yellowed teeth in swollen gums", "needle-like teeth in pink gums",
        "molars erupting from flesh", "a ring of teeth set in gums",
    ],
    "mouths": [
        "a lamprey's circular mouth with rings of teeth", "a puckered round mouth opening",
        "an open mouth with tongue and teeth", "a leech mouth", "a gaping toothy maw",
        "a slit-like mouth with lips", "a mouth with several tongues", "a round sucking mouth with radial teeth",
    ],
    "suckers": [
        "octopus suckers", "squid sucker rings with hooks", "a cluster of suckers on tentacle flesh",
        "cuttlefish tentacle suckers", "rows of suction cups", "a single large sucker",
        "leech-like suckers", "barnacle-like sucking pores",
    ],
}
THINGS = {"eyes", "teeth", "mouths", "suckers"}

SET = "skins"
WALL = "flat-on view of a wall filling the entire frame, no perspective, {wet}, {light}, macro photograph, photoreal"
LIGHT = [
    "harsh clinical flash", "dim red light", "cold surgical light", "raking side light", "sickly green light",
    "warm candlelight", "backlit glow",
]
DEEP = {
    "teeth-wall": [
        "a wall built entirely of real human teeth stacked in rows like bricks, the teeth themselves are the bricks, "
        "wet pink gums as the mortar between them",
        "masonry made of yellowed human molars and incisors laid in courses like bricks, swollen red gum tissue as mortar",
        "a dense wall of hundreds of crooked human teeth packed together like cobblestones, bleeding gums between them",
        "rows of enamel teeth set like bricks into glistening pink gum flesh, a wall of teeth",
    ],
    "flesh-wall": [
        "a wall of wet raw flesh with deep folds and glistening membranes", "a wall of raw muscle and fat, glistening",
        "a wall of swollen pink tissue with folds and mucus", "a wall of skinless red flesh with stretched membranes",
    ],
    "eye-wall": [
        "a wall of many real eyeballs embedded in flesh, all staring", "dozens of bloodshot human eyes set into a fleshy wall",
        "a wall of eyeballs of different sizes and iris colours embedded in wet pink flesh",
        "a fleshy wall studded with staring yellow eyes",
    ],
    "gut-wall": [
        "tightly packed coiled intestines forming a wall, glossy", "a wall of densely coiled pink bowels, glistening",
        "a mass of tangled grey-pink intestines filling a wall", "coiled guts packed together, slick and shiny",
    ],
    "vein-membrane": [
        "a translucent membrane stretched thin with dark branching veins behind it, backlit red",
        "thin stretched skin glowing red from behind, a network of dark veins inside",
        "a backlit membrane of tissue with branching purple blood vessels", "translucent red membrane with a web of veins",
    ],
    "mouth-wall": [
        "a wall covered in human lips and small mouths, some open showing teeth",
        "a fleshy wall of many puckered mouths, some gaping with teeth and tongues",
        "dozens of wet lips and small toothy mouths embedded in a flesh wall",
        "a wall of open mouths with teeth set into pink skin",
    ],
    "psychedelic-flesh": [
        "flesh and veins in saturated magenta, cyan and acid green, fractal folds",
        "fractal folds of raw flesh in neon acid colours, glowing veins",
        "psychedelic iridescent tissue in magenta and acid green with branching cyan veins",
        "swirling fractal flesh folds in saturated ultraviolet colours, wet",
    ],
}
if SET == "deep":
    OUT, DRIVE, SUBJECTS, THINGS = "/content/deep_gen", DRIVE.replace("skins", "deep"), DEEP, set()
    NEGATIVE += ", grid, collage, panels, photo mosaic, multiple photos, perspective, corridor, room, floor"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    if os.system(cmd):
        raise RuntimeError(f"failed: {cmd}")


def circular(model):
    import torch

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.padding_mode = "circular"


def build():
    import torch
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    p = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V4.0_Lightning", vae=vae, torch_dtype=torch.float16)
    p.scheduler = DPMSolverMultistepScheduler.from_config(
        p.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    circular(p.unet)
    circular(p.vae)
    p.vae.enable_slicing()  # a batch of four decoded at once runs out of memory on a T4
    p.set_progress_bar_config(disable=True)
    return p.to("cuda")


def prompt(motif, rng):
    if motif == "hair":  # pulled back from macro, or hair stops reading as hair
        return HAIR.format(subject=rng.choice(SUBJECTS[motif]), wet=rng.choice(WET))
    if SET == "deep":
        return f"{rng.choice(DEEP[motif])}, " + WALL.format(wet=rng.choice(WET), light=rng.choice(LIGHT))
    look = (THING if motif in THINGS else SURFACE).format(colour=rng.choice(COLOURS))
    return f"{rng.choice(SUBJECTS[motif])}, {look}, " + STYLE.format(wet=rng.choice(WET))


def counts():
    return {m: len(os.listdir(f"{OUT}/256/{m}")) for m in SUBJECTS}


if "pipe" not in globals():
    sh("pip uninstall -q -y torchao; pip install -q -U diffusers transformers accelerate safetensors")
    t = time.time()
    pipe = build()
    print(f"pipeline ready in {time.time() - t:.0f}s", flush=True)

import torch
from PIL import Image

for m in SUBJECTS:
    os.makedirs(f"{OUT}/orig/{m}", exist_ok=True)
    os.makedirs(f"{OUT}/256/{m}", exist_ok=True)
os.makedirs(f"{OUT}/chunks", exist_ok=True)

rng = random.Random(time.time_ns())
start, made = time.time(), []
while time.time() - start < MINUTES * 60:
    have = counts()
    motif = min(have, key=have.get)
    if have[motif] >= PER_MOTIF:
        break
    prompts = [prompt(motif, rng) for _ in range(BATCH)]
    negative = NEGATIVE + (", tentacles, worms, noodles, fingers, flesh tubes, macro" if motif == "hair" else "")
    seed = rng.randrange(2**31)
    gen = torch.Generator("cuda").manual_seed(seed)
    t = time.time()
    images = pipe(prompts, negative_prompt=[negative] * BATCH, num_inference_steps=STEPS, guidance_scale=CFG,
                  height=SIZE, width=SIZE, generator=gen).images
    n = max([int(f[:4]) for f in os.listdir(f"{OUT}/256/{motif}")] + [-1]) + 1
    with open(f"{OUT}/credits.jsonl", "a") as f:
        for i, (im, pr) in enumerate(zip(images, prompts)):
            name = f"{motif}/{n + i:04d}.png"
            im.save(f"{OUT}/orig/{name}")
            im.convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS).save(f"{OUT}/256/{name}")
            f.write(json.dumps({"file": name, "model": MODEL, "licence": LICENCE, "prompt": pr,
                                "negative": negative, "seed": seed, "batch_index": i, "steps": STEPS, "cfg": CFG, "size": SIZE}) + "\n")
            made.append(name)
    print(f"{motif:9} {have[motif] + BATCH:4}  {time.time() - t:.1f}s/batch", flush=True)

chunk = len(os.listdir(f"{OUT}/chunks"))
with tarfile.open(f"{OUT}/chunks/{chunk:04d}.tar", "w") as tar:
    for name in made:
        tar.add(f"{OUT}/256/{name}", arcname=name)
    tar.add(f"{OUT}/credits.jsonl", arcname="credits.jsonl")
if os.path.isdir("/content/drive/MyDrive"):
    shutil.copytree(f"{OUT}/256", DRIVE, dirs_exist_ok=True)
    shutil.copy(f"{OUT}/credits.jsonl", DRIVE)
print(f"chunk {chunk:04d}: {len(made)} images in {(time.time() - start) / 60:.1f} min; {counts()}", flush=True)
