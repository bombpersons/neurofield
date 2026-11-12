"""The level net on sculpted targets, on Modal: the same pilot as train/kaggle_sculpt, for
when the Kaggle quota is spent. The shards and the starting checkpoint live on the private
volume `neurofield`, put there once. The shards come straight from the neurofield-sculpt-data
kernels' output (`fetch`, which needs the Modal secret `kaggle` holding KAGGLE_API_TOKEN):

    .venv/bin/modal volume create neurofield
    .venv/bin/modal volume put neurofield runs/kaggle7/out/ckpt.pt /ckpt/ckpt.pt
    .venv/bin/modal run train/modal_sculpt.py::fetch

    .venv/bin/modal run --detach train/modal_sculpt.py --hours 2.5 --gpu T4 --name sculpt1 --up resize
    .venv/bin/modal run --detach train/modal_sculpt.py --hours 2.5 --gpu T4 --name stage1 --deferred --start-from runs/sculpt1/ckpt.pt  # the per-pixel stage
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
for module in ("train.py", "data.py", "model.py", "pixels.py", "gbuffer.py"):
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


@app.function(gpu="L4", cpu=4, memory=16 * 1024, volumes={VOL: volume}, timeout=3 * 3600)
def train(hours: float, name: str, up: str = "", deferred: bool = False, freeze: int = 5000,
          organ_weight: float = 10.0, lpips: float = 0.3, batch: int = 32, launch: str = "", start_from: str = "ckpt/ckpt.pt"):
    """Train `hours` (setup included) into /runs/<name>. With `deferred`, the per-pixel stage
    (docs/deferred_stage.md) with its freeze and organ weight; `lpips` only then (the plain
    runs keep 0.1). `up` empty keeps the checkpoint's head. A new run starts from the volume's
    `start_from` (run 7 unless told, or another run's: runs/sculpt1/ckpt.pt).

    The run's end is fixed when it first starts (runs/<name>/run.json, by `launch`): if Modal
    restarts it (a preempted machine), it resumes from the last checkpoint and trains only until then,
    its learning rate carrying on down the same decay."""
    import glob
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time

    start = time.time()
    out = f"{VOL}/runs/{name}"
    os.makedirs(out, exist_ok=True)
    clock = f"{out}/run.json"
    run = json.load(open(clock)) if os.path.exists(clock) else {}
    if run.get("launch") != launch or not launch:
        run = {"launch": launch, "begun": start, "end": start + hours * 3600}
        json.dump(run, open(clock, "w"))
        volume.commit()
    else:
        print(f"restarted, {(run['end'] - start) / 60:.0f} min left of the run begun {(start - run['begun']) / 60:.0f} min ago", flush=True)
    if run["end"] - start < 10 * 60:
        print("too little time left to be worth starting", flush=True)
        return
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc", shell=True, capture_output=True, text=True).stdout, flush=True)
    # Off the volume onto local disk, in the order the Kaggle pilot sees them; the last is
    # held out for validation.
    shards = sorted(os.path.dirname(v) for v in glob.glob(f"{VOL}/sculpt/**/shard_*/turning.npy", recursive=True))
    assert len(shards) >= 2, "no sculpted shards on the volume"
    data = "/tmp/data/train"
    for i, d in enumerate(shards):
        shutil.copytree(d, f"{data}/shard_{i:04d}")
    print(f"{len(shards)} shards copied in {(time.time() - start) / 60:.1f} min", flush=True)

    if not any(os.path.exists(f"{out}/{f}") for f in ("ckpt.pt", "ckpt.prev.pt")):
        print(f"starting from {VOL}/{start_from}", flush=True)
        shutil.copy(f"{VOL}/{start_from}", f"{out}/ckpt.pt")

    sys.path.insert(0, "/root/train")
    from train import Trainer

    extra = dict(deferred=True, freeze=freeze, organ_weight=organ_weight, lpips_weight=lpips) if deferred else {}
    trainer = Trainer(data, out, batch=batch, workers=3, up=up or None, on_save=volume.commit, **extra)
    trainer.probe()
    # Training stops a couple of minutes short of the end, for the last checkpoint.
    total = (run["end"] - run["begun"]) / 60 - 2
    spent = (time.time() - run["begun"]) / 60
    print(f"setup took {(time.time() - start) / 60:.1f} min, training for {total - spent:.0f} min", flush=True)
    trainer.run(minutes=total - spent, log_every=100, sample_every=500, decay=True, spent=spent, total=total)
    volume.commit()


@app.local_entrypoint()
def main(hours: float = 2.5, name: str = "sculpt1", gpu: str = "L4", up: str = "", deferred: bool = False,
         freeze: int = 5000, organ_weight: float = 10.0, lpips: float = 0.3, batch: int = 32, start_from: str = "ckpt/ckpt.pt"):
    # A quarter of an hour over the training time, for setup and the last checkpoint. Spawned,
    # so with --detach it runs on whatever happens to this end: a `modal run` killed while
    # waiting on `.remote` cancels the call, detached or not.
    import time

    launch = time.strftime("%Y%m%dT%H%M%S")
    call = train.with_options(gpu=gpu, timeout=int((hours + 0.25) * 3600)).spawn(
        hours, name, up, deferred, freeze, organ_weight, lpips, batch, launch, start_from)
    print(f"training as {call.object_id}: .venv/bin/modal app logs neurofield-sculpt")
