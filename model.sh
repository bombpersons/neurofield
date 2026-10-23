#!/usr/bin/env bash
# Get the network the game and editor run (neurofield.ron's `model`): export the
# newest checkpoint under runs/, or with none, download the latest release's
# model. Leaves it at runs/model.safetensors.
#
#   ./model.sh                        newest runs/*/ckpt.pt or runs/*/out/ckpt.pt (Kaggle)
#   CKPT=runs/x/ckpt.pt ./model.sh    a particular checkpoint
#
# Re-exports only when the checkpoint or the export code is newer than the export;
# sets up .venv for the export on first use.
set -euo pipefail
cd "$(dirname "$0")"

release=https://github.com/bombpersons/neurofield/releases/latest/download
ckpt=${CKPT:-$(ls -t runs/*/ckpt.pt runs/*/out/ckpt.pt 2>/dev/null | head -n 1 || true)}
[ -n "$ckpt" ] || ckpt=runs/release/ckpt.pt
name=model.safetensors exporter=export_weights.py
model=$(dirname "$ckpt")/$name

if [ ! -f "$ckpt" ]; then
    if [ ! -f "$model" ]; then
        echo "no checkpoint under runs/, downloading $name from the latest release"
        mkdir -p runs/release
        curl -fL --progress-bar -o "$model.part" "$release/$name"
        mv "$model.part" "$model"
    fi
elif [ ! -f "$model" ] || [ "$ckpt" -nt "$model" ] || [ "train/$exporter" -nt "$model" ] || [ train/model.py -nt "$model" ]; then
    if [ ! -x .venv/bin/python ]; then
        echo "setting up .venv for the export"
        python3 -m venv --system-site-packages .venv
    fi
    echo "exporting $ckpt"
    (cd train && ../.venv/bin/python "$exporter" --ckpt "../$ckpt" --out "../$model" 2>&1 | grep '^wrote')
fi
if [ "$model" != "runs/$name" ]; then
    cp "$model" "runs/$name"
fi
echo "runs/$name is $model"
