"""Distillation targets for the dream net (dream_targets.py) as a Kaggle script kernel.

    kaggle kernels push -p train/kaggle_dream_targets
    kaggle kernels status bombpersons/neurofield-dream-targets
    kaggle kernels output bombpersons/neurofield-dream-targets -p data/dream_targets

It optimises crops from the data kernels' shards (mounted as kernel sources) against
CLIP (downloaded from Hugging Face), one process per GPU, each taking every n-th crop,
until HOURS are up. The shards in /kaggle/working/out come back as output, and the
distillation kernel (kaggle_distil) mounts this kernel as its input.
"""

import glob
import os
import subprocess
import time

START = time.time()
HOURS = 10  # of the week's 30 GPU hours, leaving the rest for training on the targets
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
shards = sorted(os.path.dirname(v) for v in glob.glob("/kaggle/input/**/shard_*/creatures.npy", recursive=True))
assert shards, "no shards from the data kernels"
os.makedirs(DATA, exist_ok=True)
for i, d in enumerate(shards):
    os.symlink(d, f"{DATA}/shard_{i:04d}")
print(f"{len(shards)} shards from the data kernels", flush=True)

import torch  # noqa: E402

gpus = max(1, torch.cuda.device_count())
os.makedirs(OUT, exist_ok=True)
minutes = HOURS * 60 - (time.time() - START) / 60
print(f"setup took {(time.time() - START) / 60:.1f} min, {gpus} GPUs for {minutes:.0f} min", flush=True)
procs = []
for g in range(gpus):
    log = open(f"{OUT}/log_{g}.txt", "w")
    cmd = ["python", f"{REPO}/train/dream_targets.py", "--data", DATA, "--out", OUT, "--part", str(g), "--parts", str(gpus),
           "--device", f"cuda:{g}", "--minutes", str(minutes - 5), "--threads", "2"]
    procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
while any(p.poll() is None for p, _ in procs):
    time.sleep(600)
    for g in range(gpus):  # the latest progress of each, for the kernel's log
        lines = open(f"{OUT}/log_{g}.txt").read().strip().splitlines()
        print(f"gpu {g}: {lines[-1] if lines else '-'}", flush=True)
for g, (p, log) in enumerate(procs):
    log.close()
    print(f"gpu {g} exited {p.returncode}", flush=True)
print(f"{len(glob.glob(f'{OUT}/targets_*.npz'))} shards written", flush=True)
