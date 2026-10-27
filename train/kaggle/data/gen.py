"""One part of a run's training data as a Kaggle CPU kernel (no GPU quota): clone
neurofield, build datagen, and draw PART's scenes with Mesa's software Vulkan
(lavapipe) into /kaggle/working/data. Written and pushed by push_data.py, which
fills in PART, SEED and the sizes; the training kernel (../run.py) mounts every
part's output.
"""

import subprocess
import time
from pathlib import Path

PART, SEED, SCENES, VIEWS, SHARD, LEVEL_SHARE = 0, 1, 1000, 8, 250, 0.3
REPO, OUT, TEX = "/tmp/neurofield", "/kaggle/working/data", "/tmp/textures"
START = time.time()


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")


sh("nproc; free -g | head -2")
sh(f"git clone -q --recurse-submodules https://github.com/bombpersons/neurofield {REPO} && cd {REPO} && git log --oneline -1")
sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q")
sh("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y pkg-config libudev-dev libvulkan1 mesa-vulkan-drivers > /dev/null")
sh(f"cd {REPO} && ~/.cargo/bin/cargo build --release -q -p training-data-gen")
# The photo textures (data/strange, data/skins_gen) aren't in git: they come from the
# private dataset bombpersons/neurofield-textures, uploaded as one tar, which Kaggle
# may hand over unpacked.
sh("find /kaggle/input -maxdepth 4 | head -30")
tar = next(Path("/kaggle/input").rglob("textures.tar"), None)
if tar and tar.is_file():
    sh(f"mkdir -p {TEX} && tar -xf {tar} -C {TEX}")
else:
    strange = next(p for p in Path("/kaggle/input").rglob("strange") if (p / "0" / "0.png").is_file())
    TEX = str(strange.parent)
sh(f"ls {TEX} {TEX}/strange && find {TEX}/skins_gen -name '*.png' | wc -l")
print(f"built in {(time.time() - START) / 60:.0f} min", flush=True)
sh(
    f"{REPO}/target/release/datagen gen --out {OUT} --scenes {SCENES} --views {VIEWS} --shard-scenes {SHARD} "
    f"--seed {SEED} --levels {REPO}/levels/random --level-share {LEVEL_SHARE} "
    f"--strange {TEX}/strange --photos {TEX}/skins_gen 2>&1 | grep -v WARN | tail -3"
)
sh(f"du -sh {OUT}")
