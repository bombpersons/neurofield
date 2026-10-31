"""The creature network's training run (train_creatures.py) as a Kaggle script kernel.

    kaggle kernels push -p train/kaggle_creatures
    kaggle kernels status bombpersons/neurofield-creatures
    kaggle kernels output bombpersons/neurofield-creatures -p runs/creatures1

Like train/kaggle/run.py, it trains on the shards the data kernels drew, mounted as
kernel sources; it needs ones drawn with creature layers (creatures.npy). It
trains until HOURS are up and everything in /kaggle/working/out comes back as
output. To continue from a finished run, upload its ckpt.pt as a private dataset
(say bombpersons/neurofield-creatures-ckpt), add that to dataset_sources in
kernel-metadata.json and push again.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

START = time.time()
HOURS = 10  # of Kaggle's 12, leaving room for the last checkpoint
REPO, DATA, OUT = "/tmp/neurofield", "/tmp/data/train", "/kaggle/working/out"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")


sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc")
sh(f"git clone -q https://github.com/bombpersons/neurofield {REPO}")
sh(f"cd {REPO} && git log --oneline -1")
sh("pip install -q lpips")
shards = sorted(os.path.dirname(v) for v in glob.glob("/kaggle/input/**/shard_*/creatures.npy", recursive=True))
assert len(shards) >= 2, f"need shards with creature layers from the data kernels, found {len(shards)}"
os.makedirs(DATA, exist_ok=True)
for i, d in enumerate(shards):  # the last is held out for validation
    os.symlink(d, f"{DATA}/shard_{i:04d}")
print(f"{len(shards)} shards from the data kernels", flush=True)

os.makedirs(OUT, exist_ok=True)
ckpts = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
if ckpts and not os.path.exists(f"{OUT}/ckpt.pt"):
    print(f"starting from {ckpts[0]}", flush=True)
    shutil.copy(ckpts[0], f"{OUT}/ckpt.pt")

sys.path.insert(0, f"{REPO}/train")
from train_creatures import Trainer  # noqa: E402

trainer = Trainer(DATA, OUT, batch=16, workers=3)
minutes = HOURS * 60 - (time.time() - START) / 60
print(f"setup took {(time.time() - START) / 60:.1f} min, training for {minutes:.0f} min", flush=True)
trainer.run(minutes=minutes, log_every=100, sample_every=1000)
