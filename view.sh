#!/usr/bin/env bash
# Run the realtime viewer with the newest checkpoint under runs/.
#
#   ./view.sh [--seed 7]               newest runs/*/ckpt.pt
#   CKPT=runs/x/ckpt.pt ./view.sh      a particular checkpoint
#
# Exports the checkpoint to ONNX when it is newer than its model.onnx, sets up
# .venv for the export on first use, and builds the viewer if needed. Extra
# arguments go to the viewer.
set -euo pipefail
cd "$(dirname "$0")"

ckpt=${CKPT:-$(ls -t runs/*/ckpt.pt 2>/dev/null | head -n 1)}
[ -n "$ckpt" ] || { echo "no runs/*/ckpt.pt yet" >&2; exit 1; }
onnx=$(dirname "$ckpt")/model.onnx

if [ ! -x .venv/bin/python ]; then
    echo "setting up .venv for ONNX export"
    python3 -m venv --system-site-packages .venv
    .venv/bin/pip install -q onnx onnxscript onnxruntime
fi
if [ ! -f "$onnx" ] || [ "$ckpt" -nt "$onnx" ]; then
    echo "exporting $ckpt"
    (cd train && ../.venv/bin/python export_onnx.py --ckpt "../$ckpt" --out "../$onnx" 2>&1 | grep '^wrote')
fi

cargo build --release --quiet --manifest-path viewer/Cargo.toml
exec viewer/target/release/viewer --model "$onnx" "$@"
