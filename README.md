# argus

On-prem **surveillance-footage analysis & understanding**: read recorded video and make
sense of it — detect objects, track them across frames, and triage which clips are worth a
closer look. Built to extend toward face-ID/re-ID, embeddings, and search (full design in
[context/implementation-plan.md](context/implementation-plan.md)).

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
context/       architecture.md (module map + how to extend) + implementation-plan.md
```

The extension contracts all live in `argus/core` — implement a `Detector`/`Tracker`
Protocol and drop the file in the matching `*/backends/` folder. See
[context/architecture.md](context/architecture.md).

## Setup

```bash
uv sync
```

Pulls `surveillance-datasets` (from git) + torch/ultralytics/opencv. `yolo11s.pt`
auto-downloads on first detector use.

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

## Run a notebook

```bash
uv run marimo edit examples/01_dvr_person_tracking.py
```

Run from the **repo root** so `import argus` resolves.

## Notes

- Displayed videos are H.264 and downscaled to 480p so they play inline; **detection
  always runs at full resolution** (`TrackingResult.render` handles the downscale +
  H.264 transcode — this OpenCV wheel has no H.264 encoder).
- Tests: `uv run pytest` (synthetic fakes; no model/GPU/data needed).
