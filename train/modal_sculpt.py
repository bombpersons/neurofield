"""The level net on sculpted targets, on Modal: the same pilot as train/kaggle_sculpt, for
when the Kaggle quota is spent. The shards and the starting checkpoint live on the private
volume `neurofield`, put there once. The shards come straight from the neurofield-sculpt-data
kernels' output (`fetch`, which needs the Modal secret `kaggle` holding KAGGLE_API_TOKEN):

    .venv/bin/modal volume create neurofield
    .venv/bin/modal volume put neurofield runs/kaggle7/out/ckpt.pt /ckpt/ckpt.pt
    .venv/bin/modal run train/modal_sculpt.py::fetch

    .venv/bin/modal run --detach train/modal_sculpt.py --hours 2.5 --gpu T4 --name sculpt1
    .venv/bin/modal app logs neurofield-sculpt                          # while it runs
    .venv/bin/modal volume get neurofield /runs/sculpt1 runs/          # afterwards

Training code is this checkout's train/*.py, not the pushed branch. The run writes to
/runs/<name> on the volume; started again with the same name it carries on from there.
"""

import pathlib

import modal

TRAIN = pathlib.Path(__file__).parent
VOL = "/vol"

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "torchvision", "lpips", "numpy", "pillow")
for module in ("train.py", "data.py", "model.py"):
    image = image.add_local_file(TRAIN / module, f"/root/train/{module}")

app = modal.App("neurofield-sculpt", image=image)
volume = modal.Volume.from_name("neurofield")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("kaggle==2.2.4", "six"),  # which it uses but doesn't name
    secrets=[modal.Secret.from_name("kaggle")],
    volumes={VOL: volume},
    cpu=2,
    timeout=3600,
)
def fetch(parts: int = 4):
    """The sculpted shards from the Kaggle data kernels onto the volume, as /sculpt/part<i>."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        to = f"{VOL}/sculpt/part{i}"
        r = subprocess.run(
            ["kaggle", "kernels", "output", f"bombpersons/neurofield-sculpt-data-{i}", "-p", to, "-o"],
            capture_output=True, text=True,
        )
        return i, r.returncode, (r.stdout + r.stderr)[-500:]

    with ThreadPoolExecutor(parts) as pool:
        results = list(pool.map(one, range(parts)))
    volume.commit()
    for i, code, tail in results:
        print(f"part {i}: exit {code}\n{tail}", flush=True)
    print(subprocess.run(f"du -sh {VOL}/sculpt/*; ls -d {VOL}/sculpt/*/data/shard_* | wc -l", shell=True, capture_output=True, text=True).stdout)
    assert all(code == 0 for _, code, _ in results), "a download failed"


@app.function(gpu="L4",cpu=4, memory=16 * 1024, volumes={VOL: volume}, timeout=3 * 3600)
def train(hours: float, name: str):
    import glob
    import os
    import shutil
    import subprocess
    import sys
    import time

    start = time.time()
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc", shell=True, capture_output=True, text=True).stdout, flush=True)
    # Off the volume onto local disk, in the order the Kaggle pilot sees them; the last is
    # held out for validation.
    shards = sorted(os.path.dirname(v) for v in glob.glob(f"{VOL}/sculpt/**/shard_*/turning.npy", recursive=True))
    assert len(shards) >= 2, "no sculpted shards on the volume"
    data = "/tmp/data/train"
    for i, d in enumerate(shards):
        shutil.copytree(d, f"{data}/shard_{i:04d}")
    print(f"{len(shards)} shards copied in {(time.time() - start) / 60:.1f} min", flush=True)

    out = f"{VOL}/runs/{name}"
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/ckpt.pt"):
        print(f"starting from {VOL}/ckpt/ckpt.pt", flush=True)
        shutil.copy(f"{VOL}/ckpt/ckpt.pt", f"{out}/ckpt.pt")

    sys.path.insert(0, "/root/train")
    from train import Trainer

    trainer = Trainer(data, out, batch=32, workers=3, up="resize")
    minutes = hours * 60 - (time.time() - start) / 60
    print(f"setup took {(time.time() - start) / 60:.1f} min, training for {minutes:.0f} min", flush=True)
    trainer.run(minutes=minutes, log_every=100, sample_every=500, decay=True)
    volume.commit()


@app.local_entrypoint()
def main(hours: float = 2.5, name: str = "sculpt1", gpu: str = "L4"):
    # A quarter of an hour over the training time, for setup and the last checkpoint. Spawned,
    # so with --detach it runs on whatever happens to this end: a `modal run` killed while
    # waiting on `.remote` cancels the call, detached or not.
    call = train.with_options(gpu=gpu, timeout=int((hours + 0.25) * 3600)).spawn(hours, name)
    print(f"training as {call.object_id}: .venv/bin/modal app logs neurofield-sculpt")
