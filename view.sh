#!/usr/bin/env bash
# Run the realtime viewer with the newest checkpoint under runs/.
#
#   ./view.sh [--seed 7]               newest runs/*/ckpt.pt or runs/*/out/ckpt.pt (Kaggle)
#   CKPT=runs/x/ckpt.pt ./view.sh      a particular checkpoint
#   ./view.sh --device gpu             the pure-Rust (burn) network on the GPU; --device cpu
#                                      runs it on the CPU (slow: ONNX Runtime is the CPU default)
#
# Exports the checkpoint (to model.onnx, or model.safetensors for burn) when it or the
# export code is newer than the export, sets up .venv for the export on first use, and
# builds the viewer if needed. Extra arguments go to the viewer.
set -euo pipefail
cd "$(dirname "$0")"

ckpt=${CKPT:-$(ls -t runs/*/ckpt.pt runs/*/out/ckpt.pt 2>/dev/null | head -n 1)}
[ -n "$ckpt" ] || { echo "no runs/*/ckpt.pt yet" >&2; exit 1; }
if [[ " $* " == *" --device "* ]]; then
    model=$(dirname "$ckpt")/model.safetensors exporter=export_weights.py
else
    model=$(dirname "$ckpt")/model.onnx exporter=export_onnx.py
fi

if [ ! -x .venv/bin/python ]; then
    echo "setting up .venv for ONNX export"
    python3 -m venv --system-site-packages .venv
    .venv/bin/pip install -q onnx onnxscript onnxruntime
fi
if [ ! -f "$model" ] || [ "$ckpt" -nt "$model" ] || [ "train/$exporter" -nt "$model" ] || [ train/model.py -nt "$model" ]; then
    echo "exporting $ckpt"
    (cd train && ../.venv/bin/python "$exporter" --ckpt "../$ckpt" --out "../$model" 2>&1 | grep '^wrote')
fi

cargo build --release --quiet --manifest-path viewer/Cargo.toml
exec viewer/target/release/viewer --model "$model" "$@"
