#!/usr/bin/env bash
# Upload a local checkpoint to /content/out/ckpt.pt on a Colab VM, so the next
# train/colab_chunk.py resumes from it. One-shot uploads of a ~120 MB file fail,
# so it goes up in 20 MB parts that the kernel joins and checks against our md5.
#
#   train/colab_upload_ckpt.sh runs/colab2/ckpt.pt [session]
set -eu
ckpt=${1:?checkpoint}
session=${2:-neurofield}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
split -b 20M -d "$ckpt" "$tmp/ckpt.part."
want=$(md5sum "$ckpt" | cut -d' ' -f1)
echo "import os; os.makedirs('/content/out', exist_ok=True)" >"$tmp/mkdir.py"
colab exec -s "$session" --timeout 60 -f "$tmp/mkdir.py" >/dev/null
for p in "$tmp"/ckpt.part.*; do
    colab upload -s "$session" "$p" "/content/out/$(basename "$p")" | tail -n 1 | cut -c1-80
done
cat >"$tmp/join.py" <<EOF
import glob, hashlib, os
parts = sorted(glob.glob('/content/out/ckpt.part.*'))
with open('/content/out/ckpt.tmp', 'wb') as f:
    for p in parts:
        f.write(open(p, 'rb').read())
for p in parts:
    os.remove(p)
got = hashlib.md5(open('/content/out/ckpt.tmp', 'rb').read()).hexdigest()
if got == '$want':
    os.replace('/content/out/ckpt.tmp', '/content/out/ckpt.pt')
print('md5 ok' if got == '$want' else 'md5 MISMATCH', len(parts), 'parts')
EOF
colab exec -s "$session" --timeout 120 -f "$tmp/join.py" | grep md5
