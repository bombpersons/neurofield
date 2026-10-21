#!/usr/bin/env bash
# Run the realtime viewer with the newest checkpoint under runs/.
#
#   ./view.sh [--seed 7]               newest runs/*/ckpt.pt or runs/*/out/ckpt.pt (Kaggle),
#                                      or with none, the latest GitHub release's model
#   CKPT=runs/x/ckpt.pt ./view.sh      a particular checkpoint
#   ./view.sh --gpu [--seed 7]         the network on the GPU (WGSL shaders through wgpu)
#                                      rather than on the CPU with ONNX Runtime
#   ./view.sh --game [--seed 7]        the Bevy game (game/): mouse look, effects panel
#
# Exports the checkpoint (to model.onnx, or model.safetensors for the GPU) when it or the
# export code is newer than the export, sets up .venv for the export on first use, and
# builds the viewer if needed. Without a checkpoint it downloads the exported model from the
# latest release into runs/release/ instead. Extra arguments go to the viewer.
set -euo pipefail
cd "$(dirname "$0")"

release=https://github.com/bombpersons/neurofield/releases/latest/download
ckpt=${CKPT:-$(ls -t runs/*/ckpt.pt runs/*/out/ckpt.pt 2>/dev/null | head -n 1 || true)}
[ -n "$ckpt" ] || ckpt=runs/release/ckpt.pt
app=viewer
if [ "${1:-}" = --gpu ] || [ "${1:-}" = --game ]; then
    [ "$1" = --game ] && app=game
    shift
    model=$(dirname "$ckpt")/model.safetensors exporter=export_weights.py
else
    model=$(dirname "$ckpt")/model.onnx exporter=export_onnx.py
fi

if [ ! -f "$ckpt" ]; then
    if [ ! -f "$model" ]; then
        echo "no checkpoint under runs/, downloading $(basename "$model") from the latest release"
        mkdir -p runs/release
        curl -fL --progress-bar -o "$model.part" "$release/$(basename "$model")"
        mv "$model.part" "$model"
    fi
elif [ ! -f "$model" ] || [ "$ckpt" -nt "$model" ] || [ "train/$exporter" -nt "$model" ] || [ train/model.py -nt "$model" ]; then
    if [ ! -x .venv/bin/python ]; then
        echo "setting up .venv for ONNX export"
        python3 -m venv --system-site-packages .venv
        .venv/bin/pip install -q onnx onnxscript onnxruntime
    fi
    echo "exporting $ckpt"
    (cd train && ../.venv/bin/python "$exporter" --ckpt "../$ckpt" --out "../$model" 2>&1 | grep '^wrote')
fi

cargo build --release --quiet --manifest-path $app/Cargo.toml
exec $app/target/release/$app --model "$model" "$@"
