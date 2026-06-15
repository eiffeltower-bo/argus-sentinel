# argus

On-prem **surveillance-footage analysis & understanding**: read recorded video and make
sense of it — detect objects, track them across frames, and triage which clips are worth a
closer look. Built to extend toward face-ID/re-ID, embeddings, and search (full design in
[context/face-id-design.md](context/face-id-design.md)).

`argus` is a small, model-agnostic, pluggable SDK; the marimo notebooks in `examples/`
exercise it on real surveillance footage.

## Layout

```
argus/
  core/        shared types (Detection, Track) + extension Protocols (Detector, Tracker)
               + COCO taxonomy — dependency-free foundation
  detect/      object detection; detect/backends/ultralytics.py (YOLO11)
  track/       tracking-by-detection; track/backends/bytetrack.py (ByteTrack)
  pipeline/    orchestration: tracking.py (VideoTracker/track_video/TrackingResult),
               peek.py (peek_video/peek_videos/PeekResult)
examples/      marimo notebooks (person tracking, vehicle tracking, folder peek-by-time)
tests/         fast unit suite (no GPU/weights/data needed)
context/       architecture.md (module map + how to extend) + face-id-design.md (v0.3)
```

The extension contracts all live in `argus/core` — implement a `Detector`/`Tracker`
Protocol and drop the file in the matching `*/backends/` folder. See
[context/architecture.md](context/architecture.md).

## Setup

```bash
uv sync
```

Installs torch/ultralytics/opencv directly (no SSH/private repo needed). torch defaults to
the **CPU** wheel — reliable everywhere and what the test suite wants; opt into GPU per the
Docker section below. `yolo11s.pt` auto-downloads on first detector use. Optional extras:
`--extra face` / `--extra face-gpu` (face-ID), `--extra datasets` (sample-data fetching, needs
SSH), `--extra notebooks` (marimo).

## Use the SDK

```python
from argus import track_video, peek_videos

result = track_video("clip.mp4", targets=("person", "vehicle"), device="cuda")
result.metrics()                       # per-track polars DataFrame
result.to_parquet("tracks.parquet", what="tracks")
result.render("annotated.mp4")         # annotated H.264 video

peek_videos(clips, targets=("vehicle",), device="cuda")   # fast folder triage
```

Both axes are pluggable: any object satisfying the `Detector` (`.detect(frame)`) or
`Tracker` (`.update(dets, frame)` / `.reset()`) protocol drops into `VideoTracker`.

## Sample scripts

Runnable CLI examples in `examples/` (each takes `--help`):

```bash
uv run python examples/track_clip.py    clip.mp4 --targets person vehicle --render
uv run python examples/peek_folder.py   /footage --targets vehicle --workers 8
uv run python examples/custom_backend.py clip.mp4   # a custom Detector via the protocol
```

## CLI

Installing the package exposes an `argus` command (a thin wrapper over the facade):

```bash
uv run argus peek  /footage --targets vehicle --workers 8 --json
uv run argus track clip.mp4 --targets person vehicle --render out.mp4 --json
```

## Run a notebook

```bash
uv run marimo edit examples/01_dvr_person_tracking.py
```

Run from the **repo root** so `import argus` resolves.

## Docker

A single multi-stage `Dockerfile` builds both a **CPU** image (multi-arch amd64+arm64) and a
**CUDA** image (amd64); the `Makefile` wraps the buildx/QEMU commands and the CPU-only smoke
test. The image entrypoint is the `argus` CLI; mount your footage and run:

```bash
make build-amd64                                  # or: make build-cpu (multi-arch)
docker run --rm -v /footage:/data argus:amd64 peek /data --json
make smoke-amd64 test-amd64                       # CPU-only smoke + in-container pytest
make build-cuda                                   # CUDA image (cu126); run with --gpus all
```

### Develop in a container

`docker-compose.yml` builds a **dev** image with the full environment and bind-mounts the
working directory over `/app`, so edits on the host reflect immediately inside (argus is
installed editable — no rebuild for code changes):

```bash
docker compose up -d dev            # build + start (first run installs torch etc.)
docker compose exec dev bash        # shell in; then run argus / pytest / python
docker compose exec dev pytest -q
docker compose exec dev argus track clip.mp4 --render out.mp4 --json
docker compose down                 # stop
```

CPU by default; uncomment the CUDA build args + GPU reservation in `docker-compose.yml` for
local GPU work (needs the NVIDIA Container Toolkit).

## Notes

- Displayed videos are H.264 and downscaled to 480p so they play inline; **detection
  always runs at full resolution** (`TrackingResult.render` handles the downscale +
  H.264 transcode — this OpenCV wheel has no H.264 encoder).
- Tests: `uv run pytest` (synthetic fakes; no model/GPU/data needed).
