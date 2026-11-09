"""A short pilot of the dream net's retrain as a Kaggle script kernel: 64 distillation targets
(dream_targets.py) from the data kernels' shards, then a few minutes of training on them, to
see that the targets look right before spending the week's quota on all of them.

    kaggle kernels push -p train/kaggle_dream_pilot
    kaggle kernels status bombpersons/neurofield-dream-pilot
    kaggle kernels output bombpersons/neurofield-dream-pilot -p runs/dream_pilot
"""

import glob
import os
import subprocess
import sys
import time

START = time.time()
BRANCH = "dream-retrain"
TARGET_MINUTES, TRAIN_MINUTES = 25, 15
REPO, DATA, OUT = "/tmp/neurofield", "/tmp/data/train", "/kaggle/working/out"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")


sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc")
sh("pip install -q lpips")
sh(f"git clone -q -b {BRANCH} https://github.com/bombpersons/neurofield {REPO}")
sh(f"cd {REPO} && git log --oneline -1")
shards = sorted(os.path.dirname(v) for v in glob.glob("/kaggle/input/**/shard_*/creatures.npy", recursive=True))
assert shards, "no shards from the data kernels"
os.makedirs(DATA, exist_ok=True)
for i, d in enumerate(shards):
    os.symlink(d, f"{DATA}/shard_{i:04d}")
print(f"{len(shards)} shards from the data kernels", flush=True)

os.makedirs(OUT, exist_ok=True)
sh(f"python {REPO}/train/dream_targets.py --data {DATA} --out {OUT} --count 64 --per-shard 32 --device cuda:0 --threads 2 --minutes {TARGET_MINUTES}")
print(f"targets done at {(time.time() - START) / 60:.1f} min", flush=True)

sys.path.insert(0, f"{REPO}/train")
from train_distil import Trainer  # noqa: E402

trainer = Trainer(OUT, f"{OUT}/train", batch=16, workers=2, clip_from=500)
trainer.run(minutes=TRAIN_MINUTES, log_every=50, sample_every=250)
print(f"finished at {(time.time() - START) / 60:.1f} min", flush=True)
