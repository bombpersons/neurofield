"""A fixture for engine/examples/nca_check.rs: an automaton's weights, a state of the game's
size and the state after one step with every cell updating, raw f32 cell-major.

    python train/nca_fixture.py runs/nca/slot1 OUT
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nca  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
m = nca.Nca()
m.load_state_dict(torch.load(src / "nca.pt", map_location="cpu"))
torch.manual_seed(0)
x = nca.seed(1, 256, "cpu")
with torch.no_grad():
    for _ in range(40):  # somewhere into growing, where the step does something
        x = m.step(x)
    y = m.step(x, fire=1.0)
cell_major = lambda t: t[0].permute(1, 2, 0).contiguous().numpy().astype("<f4")  # noqa: E731
cell_major(x).tofile(out / "before.bin")
cell_major(y).tofile(out / "after.bin")
(src / "nca.bin").read_bytes() and (out / "nca.bin").write_bytes((src / "nca.bin").read_bytes())
