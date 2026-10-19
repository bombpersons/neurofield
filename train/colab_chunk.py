"""One training chunk in the persistent Colab kernel.

    colab exec -s neurofield -f train/colab_chunk.py

The first call builds the Trainer (resuming from /content/out/ckpt.pt if
present); later calls reuse it from the kernel's globals and train on.
"""

import sys

MINUTES = 9

if "/content/neurofield/train" not in sys.path:
    sys.path.insert(0, "/content/neurofield/train")
if "trainer" not in globals():
    from train import Trainer

    trainer = Trainer("/content/data/train", "/content/out", batch=32, workers=2)
trainer.run(minutes=MINUTES, log_every=100, sample_every=1000)
