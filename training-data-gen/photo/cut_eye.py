"""Cut a phone video of an eye into the frames the sculpted walls show, for `datagen sculpt-walk --eye DIR`.

    python3 training-data-gen/photo/cut_eye.py --video eyes/clip.mp4 --out data/eye

Takes one blink from the video, steadied against the head, and for each of its frames
slides the iris to every gaze: the lids are the footage's own, and what shows between
them is the white with the iris redrawn over it. Writes OUT/<frame>_<gx>_<gy>.png
(256x256 RGB), frame 0 open and the rest the blink in order, gx 0-6 left to right and gy
0-4 top to bottom; the blink only at even gx and gy, since it goes by quickly. The numbers below are marked by hand on the video's frame OPEN, at
360x640.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

W, H = 360, 640
OPEN = 670
BLINK = [673, 674, 675, 677, 684, 694, 695, 696, 697, 698]
# How far the upper lid has come down the eye in each of those, 1 being shut.
SHUT = [0.22, 0.42, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.3, 0.12]
IRIS = (187.0, 300.0, 47.0)
UPPER = [(100, 325), (110, 305), (120, 292), (140, 281), (160, 276), (180, 274), (200, 274), (220, 276), (240, 284), (260, 300), (263, 306)]
LOWER = [(250, 322), (240, 330), (220, 340), (200, 347), (180, 349), (160, 349), (140, 346), (120, 340), (105, 330)]
# Where the eye is cut from: left, top, side.
CROP = (35, 160, 300)
# The box that the head is steadied without, since the eye inside it moves on its own.
EYE_BOX = (60, 215, 325, 400)
GAZE_X = [-48, -32, -16, 0, 16, 32, 48]
GAZE_Y = [-26, -13, 0, 13, 26]
# What the skin is multiplied by, to sit among flesh.
SKIN = np.array([0.95, 0.72, 0.70])
SIZE = 256


def read_frames(video, wanted):
    last = max(wanted)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={W}:{H}", "-frames:v", str(last + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    all_frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    return {i: all_frames[i].astype(np.float32) for i in wanted}


def steady(frame, ref, window):
    """`frame` moved to line up with `ref` outside the eye, its colours matched there."""
    a, b = ref.mean(2), frame.mean(2)
    fa, fb = np.fft.rfft2((a - a.mean()) * window), np.fft.rfft2((b - b.mean()) * window)
    cross = fa * np.conj(fb)
    c = np.fft.irfft2(cross / (np.abs(cross) + 1e-6), s=a.shape)
    y, x = np.unravel_index(np.argmax(gaussian_filter(c, 1.0, mode="wrap")), c.shape)
    moved = np.roll(frame, (y - H if y > H / 2 else y, x - W if x > W / 2 else x), (0, 1))
    keep = window > 0
    gain = ref[keep].mean(0) / moved[keep].mean(0)
    return moved * gain


def opening():
    """How much each pixel of the open frame is between the lids, 0..1."""
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon(UPPER + LOWER, fill=255)
    return gaussian_filter(np.asarray(img, np.float32) / 255, 1.2)


def upper_lid():
    return np.interp(np.arange(W), [p[0] for p in UPPER], [p[1] for p in UPPER])


def lower_lid():
    edge = [UPPER[0]] + LOWER[::-1] + [UPPER[-1]]
    return np.interp(np.arange(W), [p[0] for p in edge], [p[1] for p in edge])


def below_lid(shut):
    """1 below where the upper lid has got to when it is `shut` of the way down."""
    if shut >= 1.0:
        return np.zeros((H, W), np.float32)
    line = upper_lid() + shut * (lower_lid() - upper_lid())
    return np.clip(np.arange(H, dtype=np.float32)[:, None] - line[None] + 0.5, 0, 1)


def white_without_iris(ref, between):
    """The eye's white with the iris taken out, filled in along each row from either side."""
    cx, cy, r = IRIS
    white = ref.copy()
    open_px = between > 0.9
    mean_left = ref[open_px & (np.arange(W)[None] < cx - r - 3)].mean(0)
    mean_right = ref[open_px & (np.arange(W)[None] > cx + r + 3)].mean(0)
    for y in range(int(cy - r - 2), int(cy + r + 3)):
        half = np.sqrt(max(r * r - (y - cy) ** 2, 0.0)) + 4
        lx, rx = int(cx - half), int(cx + half)
        left = ref[y, lx - 2 : lx + 1].mean(0) if open_px[y, lx - 2 : lx + 1].all() else mean_left
        right = ref[y, rx : rx + 3].mean(0) if open_px[y, rx : rx + 3].all() else mean_right
        for x in range(lx, rx + 1):
            t = (x - lx) / max(rx - lx, 1)
            white[y, x] = left * (1 - t) + right * t
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    filled = np.hypot(x - cx, y - cy) < r + 6
    white = np.where(filled[..., None], gaussian_filter(white, (3.0, 1.0, 0.0)), white)
    # The upper lid's shadow falls on what was filled in as on the rest.
    top = upper_lid()
    y = np.arange(H)[:, None]
    return white * (1 - 0.4 * np.exp(-np.maximum(y - top[None], 0) / 7))[..., None]


def iris_layer(ref, between):
    """The iris as a colour image and how much of each pixel it covers, the part under the lid mirrored from below."""
    cx, cy, r = IRIS
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    colour = ref.copy()
    # Lashes and shadow reach a little way below the lid.
    seen = between > 0.95
    seen &= gaussian_filter(seen.astype(np.float32), 2.0) > 0.98
    for py, px in zip(*np.nonzero((np.hypot(x - cx, y - cy) < r + 2) & ~seen)):
        my = int(round(2 * cy - py))
        colour[py, px] = ref[my, px] if seen[my, px] else ref[int(round(cy + abs(py - cy) * 0.6)), px]
    return colour, np.clip(r + 0.5 - np.hypot(x - cx, y - cy), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/eye"))
    a = ap.parse_args()
    frames = read_frames(a.video, [OPEN] + BLINK)
    ref = frames[OPEN]
    window = np.ones((H, W), np.float32)
    x0, y0, x1, y1 = EYE_BOX
    window[y0:y1, x0:x1] = 0
    between = opening()
    white, (iris, cover) = white_without_iris(ref, between), iris_layer(ref, between)
    a.out.mkdir(parents=True, exist_ok=True)
    cx, cy, side = CROP
    for f, (index, shut) in enumerate(zip([OPEN] + BLINK, [0.0] + SHUT)):
        lids = steady(frames[index], ref, window) if index != OPEN else ref
        # What's still uncovered shows the eye; the rest is the footage's own lid.
        vis = (between * below_lid(shut))[..., None]
        skin = gaussian_filter(vis[..., 0], 3.0)[..., None]
        for gx, dx in enumerate(GAZE_X):
            for gy, dy in enumerate(GAZE_Y):
                if f > 0 and (gx % 2 or gy % 2):
                    continue
                c = np.roll(cover, (dy, dx), (0, 1))[..., None]
                eye = white * (1 - c) + np.roll(iris, (dy, dx), (0, 1)) * c
                graded = lids * (SKIN + (1 - SKIN) * skin)
                out = np.clip(graded * (1 - vis) + eye * vis, 0, 255).astype(np.uint8)
                Image.fromarray(out[cy : cy + side, cx : cx + side]).resize((SIZE, SIZE), Image.LANCZOS).save(a.out / f"{f}_{gx}_{gy}.png")
    print("frames", 1 + len(BLINK))


if __name__ == "__main__":
    main()
