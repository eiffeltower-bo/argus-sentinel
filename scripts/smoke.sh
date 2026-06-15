#!/usr/bin/env bash
# In-container CPU-only smoke test: prove the image runs peek + track (incl. the ffmpeg render
# path) without a GPU. Generates a tiny synthetic clip, so it needs no data baked in. A blank
# clip yields zero detections by design — we assert the pipeline RUNS and RENDERS, not that it
# finds anything. Run with the entrypoint overridden to bash (see the Makefile smoke-* targets).
set -euo pipefail

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[smoke] python $(python -c 'import platform;print(platform.machine())') · import argus"
python - "$WORK" <<'PY'
import sys, cv2, numpy as np, pathlib
import argus  # cheap, GPU-free
clip = pathlib.Path(sys.argv[1]) / "clip.mp4"
vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
for i in range(15):
    vw.write(np.full((240, 320, 3), (i * 15) % 255, np.uint8))
vw.release()
assert clip.exists(), "failed to write synthetic clip"
print(f"[smoke] wrote {clip}")
PY

echo "[smoke] argus peek"
argus peek "$WORK" --glob 'clip.mp4' --json

echo "[smoke] argus track --render (exercises ffmpeg/libx264)"
argus track "$WORK/clip.mp4" --max-frames 15 --render "$WORK/out.mp4" --json

test -f "$WORK/out.mp4"
echo "[smoke] OK — render exists, CPU pipeline ran with no GPU"
