"""The level net on sculpted targets, as a Kaggle script kernel: a pilot, to see whether it
can learn folded flesh, filmed eyes, veins and sphincters before spending a whole run.

    .venv/bin/python train/kaggle/data/push_data.py --sculpt BRANCH    # CPU kernels, first
    kaggle kernels push -p train/kaggle_sculpt                         # once they finish
    kaggle kernels output bombpersons/neurofield-sculpt-train -p runs/sculpt1

Starts from the checkpoint in bombpersons/neurofield-ckpt with a fresh resize head (see
model.UpHead). Everything in /kaggle/working/out comes back as output.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

START = time.time()
HOURS = 2.5  # of this week's quota; a full run is about 11
BRANCH = "game-player"
REPO, DATA, OUT = "/tmp/neurofield", "/tmp/data/train", "/kaggle/working/out"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")


sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; df -h /tmp /kaggle/working | tail -2")
sh(f"git clone -q --recurse-submodules -b {BRANCH} https://github.com/bombpersons/neurofield {REPO}")
sh(f"cd {REPO} && git log --oneline -1")
sh("pip install -q lpips")
shards = sorted(os.path.dirname(v) for v in glob.glob("/kaggle/input/**/shard_*/turning.npy", recursive=True))
assert shards, "no sculpted shards mounted: the neurofield-sculpt-data kernels must finish first"
# Every part's shards side by side; the last is held out for validation.
os.makedirs(DATA, exist_ok=True)
for i, d in enumerate(shards):
    os.symlink(d, f"{DATA}/shard_{i:04d}")
print(f"{len(shards)} shards from the data kernels", flush=True)

os.makedirs(OUT, exist_ok=True)
ckpts = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
if ckpts and not os.path.exists(f"{OUT}/ckpt.pt"):
    print(f"starting from {ckpts[0]}", flush=True)
    shutil.copy(ckpts[0], f"{OUT}/ckpt.pt")

sys.path.insert(0, f"{REPO}/train")
from train import Trainer  # noqa: E402

trainer = Trainer(DATA, OUT, batch=32, workers=3, up="resize")
minutes = HOURS * 60 - (time.time() - START) / 60
print(f"setup took {(time.time() - START) / 60:.1f} min, training for {minutes:.0f} min", flush=True)
trainer.run(minutes=minutes, log_every=100, sample_every=500, decay=True)
