"""The dream net's training on distilled targets (train_distil.py) as a Kaggle script kernel.

    kaggle kernels push -p train/kaggle_distil
    kaggle kernels status bombpersons/neurofield-distil
    kaggle kernels output bombpersons/neurofield-distil -p runs/distil1

The targets are the output of the targets kernel (kaggle_dream_targets), mounted as a
kernel source. It trains until HOURS are up and everything in /kaggle/working/out comes
back as output. To continue from a finished run, upload its ckpt.pt as a private
dataset, add that to dataset_sources in kernel-metadata.json and push again.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

START = time.time()
HOURS = 10
REPO, OUT = "/tmp/neurofield", "/kaggle/working/out"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")


sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc")
sh("pip install -q lpips")
sh(f"git clone -q https://github.com/bombpersons/neurofield {REPO}")
sh(f"cd {REPO} && git log --oneline -1")
targets = glob.glob("/kaggle/input/**/targets_*.npz", recursive=True)
assert len(targets) >= 2, f"need target shards from the targets kernel, found {len(targets)}"
print(f"{len(targets)} target shards", flush=True)

os.makedirs(OUT, exist_ok=True)
ckpts = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
if ckpts and not os.path.exists(f"{OUT}/ckpt.pt"):
    print(f"starting from {ckpts[0]}", flush=True)
    shutil.copy(ckpts[0], f"{OUT}/ckpt.pt")

sys.path.insert(0, f"{REPO}/train")
from train_distil import Trainer  # noqa: E402

trainer = Trainer("/kaggle/input", OUT, batch=16, workers=3)
minutes = HOURS * 60 - (time.time() - START) / 60
print(f"setup took {(time.time() - START) / 60:.1f} min, training for {minutes:.0f} min", flush=True)
trainer.run(minutes=minutes, log_every=100, sample_every=1000)
