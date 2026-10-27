"""Start a run's training data: PARTS Kaggle CPU kernels (bombpersons/neurofield-data-<i>),
each drawing SCENES / PARTS scenes from its own seed, all at once.

    .venv/bin/python train/kaggle/data/push_data.py [--parts 4] [--scenes 4000] [--seed S]
    .venv/bin/kaggle kernels status bombpersons/neurofield-data-0

They clone neurofield from GitHub, so push the code first. When they finish, push
the training kernel (train/kaggle), which mounts their output.
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
KAGGLE = str(HERE.parents[2] / ".venv/bin/kaggle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--scenes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    a = ap.parse_args()
    src = (HERE / "gen.py").read_text()
    for part in range(a.parts):
        consts = f"PART, SEED, SCENES, VIEWS, SHARD, LEVEL_SHARE = {part}, {a.seed + part}, {a.scenes // a.parts}, 8, 250, 0.3"
        code = re.sub(r"^PART, SEED, .*$", consts, src, count=1, flags=re.M)
        slug = f"neurofield-data-{part}"
        meta = {
            "id": f"bombpersons/{slug}", "title": slug, "code_file": "gen.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false",
            "enable_internet": "true", "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": [],
        }
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "gen.py").write_text(code)
            (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            r = subprocess.run([KAGGLE, "kernels", "push", "-p", d], text=True, capture_output=True)
            print(slug, (r.stdout + r.stderr).strip().splitlines()[-1])


if __name__ == "__main__":
    main()
