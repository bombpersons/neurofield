"""Prepare a Colab VM: clone neurofield, build datagen, generate training data.

    colab exec -s neurofield -f train/colab_setup.py

Re-running is safe: it pulls, rebuilds and only generates data that is missing.
"""

import os
import subprocess

SCENES, VIEWS, SHARD = 4000, 8, 250
REPO, DATA = "/content/neurofield", "/content/data/train"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")


sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; df -h /content | tail -1")
if os.path.exists(REPO):
    sh(f"cd {REPO} && git pull -q && git submodule update --init -q")
else:
    sh(f"git clone -q --recurse-submodules https://github.com/bombpersons/neurofield {REPO}")
if not os.path.exists(os.path.expanduser("~/.cargo/bin/cargo")):
    sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q")
sh("pip install -q lpips")
# datagen draws with Bevy: it builds against udev, and needs Vulkan (lavapipe if no GPU driver has it).
sh("apt-get -qq update && apt-get -qq install -y pkg-config libudev-dev libvulkan1 mesa-vulkan-drivers > /dev/null")
sh(f"cd {REPO} && ~/.cargo/bin/cargo build --release -q -p training-data-gen")
if not os.path.exists(f"{DATA}/format.json"):
    sh(f"{REPO}/target/release/datagen gen --out {DATA} --scenes {SCENES} --views {VIEWS} --shard-scenes {SHARD} 2>&1 | tail -2")
sh(f"du -sh {DATA}; ls {DATA} | head -3; ls {DATA} | wc -l")
