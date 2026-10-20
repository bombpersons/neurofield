"""One training chunk in the persistent Colab kernel.

    colab exec -s neurofield --timeout 7000 -f train/colab_chunk.py   # in a background shell

Keep the client attached for the whole chunk: free-tier VMs were reclaimed
10-15 minutes after it disconnected, even with the kernel busy training.
The first call builds the Trainer (resuming from /content/out/ckpt.pt if
present); later calls reuse it from the kernel's globals and train on.
"""

import sys

MINUTES = 110

if "/content/neurofield/train" not in sys.path:
    sys.path.insert(0, "/content/neurofield/train")
if "trainer" not in globals():
    from train import Trainer

    trainer = Trainer("/content/data/train", "/content/out", batch=32, workers=2)
trainer.run(minutes=MINUTES, log_every=100, sample_every=500)
