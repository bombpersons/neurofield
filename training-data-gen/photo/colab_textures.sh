#!/usr/bin/env bash
# Generate creature skins on a Colab T4 and mirror them into data/skins_gen.
# Runs colab_textures.py one chunk at a time with the client attached (free VMs
# get reclaimed soon after it disconnects), downloading each chunk's tar of
# 256x256 PNGs as it lands, then writes OUT/credits.json.
#
#   training-data-gen/photo/colab_textures.sh [session] [per-motif] [out] [set]
#
# set is "skins" (default, data/skins_gen) or "deep" (wall textures, e.g.
# `colab_textures.sh skins 20 data/deep_gen deep`). Re-running resumes: an
# existing session keeps generating where it left off.
set -u
here=$(cd "$(dirname "$0")" && pwd)
colab=${COLAB:-$here/../../.venv/bin/colab}
session=${1:-skins}
per=${2:-160}
out=${3:-data/skins_gen}
set=${4:-skins}
remote=/content/$([ "$set" = deep ] && echo deep_gen || echo skins_gen)
mkdir -p "$out/chunks"
if "$colab" status -s "$session" 2>&1 | grep -qE "not found|No sessions"; then
    "$colab" new -s "$session" --gpu T4 || exit 1
fi
t0=$(date +%s)
while true; do
    log=$("$colab" exec -s "$session" --timeout 3000 \
        -f <(sed -e "s/^PER_MOTIF = .*/PER_MOTIF = $per/" -e "s/^SET = .*/SET = \"$set\"/" "$here/colab_textures.py") \
        </dev/null 2>&1 | tee /dev/stderr)
    grep -qE "appears to be lost|Session .* not found" <<<"$log" && { echo "session lost"; exit 1; }
    chunk=$(grep -oE "^chunk [0-9]{4}" <<<"$log" | tail -n1 | cut -d' ' -f2)
    [ -n "$chunk" ] || { echo "no chunk finished"; exit 1; }
    timeout 600 "$colab" download -s "$session" "$remote/chunks/$chunk.tar" "$out/chunks/$chunk.tar" >/dev/null &&
        tar -xf "$out/chunks/$chunk.tar" -C "$out"
    echo "$(( ($(date +%s) - t0) / 60 )) min elapsed"
    grep -q "^chunk $chunk: 0 images" <<<"$log" && break
done
python3 - "$out" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
lines = [json.loads(l) for l in (out / "credits.jsonl").read_text().splitlines()]
credits = {c.pop("file"): c for c in lines}
(out / "credits.json").write_text(json.dumps(credits, indent=1))
print(len(credits), "credited")
PY
