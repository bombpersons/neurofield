"""One long training run as a Kaggle script kernel.

    kaggle kernels push -p train/kaggle          # start (needs a phone-verified account)
    kaggle kernels status bombpersons/neurofield-train
    kaggle kernels output bombpersons/neurofield-train -p runs/kaggle1

Clones neurofield and trains on the shards the data kernels drew
(data/push_data.py: Kaggle CPU kernels, mounted as kernel sources, since these
GPUs have no Vulkan for datagen to draw with), from the checkpoint in the
private dataset bombpersons/neurofield-ckpt (if there is one), until HOURS
are up, well before Kaggle's 12-hour limit, so the final checkpoint is
written. With no data kernels mounted it builds datagen and draws its own, on
the CPU. The learning rate decays over the run, and each run
generates new levels (seeded by its start time); a share of its scenes are walks
through the editor levels in levels/random. Views come in every width from 4:3
to 2.4:1. Everything in
/kaggle/working/out comes back as output.
To continue from a finished run, upload its ckpt.pt as a new version of that
dataset (`kaggle datasets version -p DIR -m note`) and push again.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

START = time.time()
HOURS = 9.5  # of Kaggle's 12 (11.25 max, leaving room for the last checkpoint); also keep under the weekly quota left
SCENES, VIEWS, SHARD, LEVEL_SHARE = 4000, 8, 250, 0.3
SEED = int(START)  # fresh levels every run, so no scene is seen in more than one
REPO, DATA, OUT = "/tmp/neurofield", "/tmp/data/train", "/kaggle/working/out"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")


sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; df -h /tmp /kaggle/working | tail -2")
sh(f"git clone -q --recurse-submodules https://github.com/bombpersons/neurofield {REPO}")
sh(f"cd {REPO} && git log --oneline -1")
sh("pip install -q lpips")
shards = sorted(os.path.dirname(v) for v in glob.glob("/kaggle/input/**/shard_*/views.npy", recursive=True))
if shards:
    # Every part's shards side by side; the last is held out for validation.
    os.makedirs(DATA, exist_ok=True)
    for i, d in enumerate(shards):
        os.symlink(d, f"{DATA}/shard_{i:04d}")
    print(f"{len(shards)} shards from the data kernels", flush=True)
else:
    sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q")
    # datagen draws with Bevy: it builds against udev, and needs Vulkan (lavapipe, as these GPUs have none).
    sh("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y pkg-config libudev-dev libvulkan1 mesa-vulkan-drivers > /dev/null")
    sh(f"cd {REPO} && ~/.cargo/bin/cargo build --release -q -p training-data-gen")
    sh(
        f"{REPO}/target/release/datagen gen --out {DATA} --scenes {SCENES} --views {VIEWS} --shard-scenes {SHARD} "
        f"--seed {SEED} --levels {REPO}/levels/random --level-share {LEVEL_SHARE} 2>&1 | grep -v WARN | tail -3"
    )

os.makedirs(OUT, exist_ok=True)
ckpts = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
if ckpts and not os.path.exists(f"{OUT}/ckpt.pt"):
    print(f"starting from {ckpts[0]}", flush=True)
    shutil.copy(ckpts[0], f"{OUT}/ckpt.pt")

sys.path.insert(0, f"{REPO}/train")
from train import Trainer  # noqa: E402

trainer = Trainer(DATA, OUT, batch=32, workers=3)
minutes = HOURS * 60 - (time.time() - START) / 60
print(f"setup took {(time.time() - START) / 60:.1f} min, training for {minutes:.0f} min", flush=True)
trainer.run(minutes=minutes, log_every=100, sample_every=500, decay=True)
