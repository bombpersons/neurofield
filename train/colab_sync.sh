#!/usr/bin/env bash
# Mirror a Colab training run locally: every few minutes, pull the latest
# sample, log and checkpoint from /content/out. Exits non-zero if the session
# disappears (free-tier VMs get reclaimed), leaving the last checkpoint here.
#
#   train/colab_sync.sh runs/colab1 [session] [seconds]
set -u
dest=${1:?destination dir}
session=${2:-neurofield}
every=${3:-300}
mkdir -p "$dest"
while true; do
    for f in log.jsonl sample_latest.png ckpt.pt; do
        out=$(timeout 300 colab download -s "$session" "/content/out/$f" "$dest/$f.part" 2>&1)
        if grep -q "appears to be lost" <<<"$out"; then
            echo "session lost; last checkpoint kept in $dest"
            exit 1
        fi
        [ -s "$dest/$f.part" ] && mv "$dest/$f.part" "$dest/$f"
    done
    tail -n 1 "$dest/log.jsonl" 2>/dev/null
    sleep "$every"
done
