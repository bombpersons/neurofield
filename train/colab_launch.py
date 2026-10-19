"""Start (or restart) training on the Colab VM as a detached process.

    colab restart-kernel -s neurofield        # frees GPU memory held by an in-kernel trainer
    colab exec -s neurofield -f train/colab_launch.py

Resumes from /content/out/ckpt.pt; output goes to /content/out/train.out.
"""

import subprocess

MINUTES = 100
subprocess.run("cd /content/neurofield && git pull -q", shell=True)
subprocess.run("pkill -f 'train.py --data' ; mkdir -p /content/out", shell=True)
subprocess.Popen(
    "cd /content/neurofield/train && nohup python train.py --data /content/data/train --out /content/out "
    f"--minutes {MINUTES} --batch 32 --workers 2 --log-every 100 --sample-every 1000 "
    ">> /content/out/train.out 2>&1 &",
    shell=True,
)
print("launched")
