"""One long training run as a Kaggle script kernel.

    kaggle kernels push -p train/kaggle          # start (needs a phone-verified account)
    kaggle kernels status bombpersons/neurofield-train
    kaggle kernels output bombpersons/neurofield-train -p runs/kaggle1

Clones neurofield, builds datagen, generates the training data, then trains
from the checkpoint in the private dataset bombpersons/neurofield-ckpt (if
there is one) until shortly before Kaggle's 12-hour limit, so the final
checkpoint is written. Everything in /kaggle/working/out comes back as output.
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
HOURS = 11.25  # of Kaggle's 12, leaving room for setup overruns and the last checkpoint
SCENES, VIEWS, SHARD = 4000, 8, 250
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
sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q")
sh(f"cd {REPO}/datagen && ~/.cargo/bin/cargo build --release -q")
sh(f"{REPO}/datagen/target/release/datagen gen --out {DATA} --scenes {SCENES} --views {VIEWS} --shard-scenes {SHARD} 2>&1 | tail -2")

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
trainer.run(minutes=minutes, log_every=100, sample_every=500)
