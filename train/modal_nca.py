"""Train a flesh automaton (nca.py) on Modal: a few thousand steps on a T4, about 15 minutes.

    .venv/bin/modal run --detach train/modal_nca.py --slot 1 --steps 4000
    .venv/bin/modal volume get neurofield /nca/slot1 runs/nca/

The painted textures go up with the image; the weights, samples and log land on the volume's
/nca/slot<N>. As modal_sculpt.py: spawned, so a `modal run` that ends doesn't cancel it.
"""

import pathlib

import modal

ROOT = pathlib.Path(__file__).parent.parent
VOL = "/vol"

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "torchvision", "numpy", "pillow")
image = image.add_local_file(ROOT / "train/nca.py", "/root/train/nca.py")
for f in sorted((ROOT / "data/painted/7").glob("*.png")):
    image = image.add_local_file(f, f"/root/painted/{f.name}")

app = modal.App("neurofield-nca", image=image)
volume = modal.Volume.from_name("neurofield")


@app.function(gpu="T4", cpu=4, memory=16 * 1024, volumes={VOL: volume}, timeout=2 * 3600)
def train(slot: int, steps: int):
    import sys

    sys.path.insert(0, "/root/train")
    import nca

    out = f"{VOL}/nca/slot{slot}"

    def log(line):
        print(line, flush=True)
        with open(f"{out}/log.jsonl", "a") as f:
            f.write(line + "\n")

    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    nca.train(slot, steps, out, painted="/root/painted", log=log, on_save=volume.commit)
    volume.commit()


@app.local_entrypoint()
def main(slot: int = 1, steps: int = 4000):
    call = train.spawn(slot, steps)
    print(f"training as {call.object_id}")
