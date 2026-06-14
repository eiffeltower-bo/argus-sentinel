# faces

Prototypes for an on-prem, forensic **surveillance face-ID system**: read recorded
video → detect people/vehicles → track them → (later) extract + embed faces and search a
vector index. Full design in [context/implementation-plan.md](context/implementation-plan.md).

This repo is the **exploration stage**: a small, model-agnostic **detection + tracking
SDK** (`faces_cv`) plus marimo notebooks that exercise it on real surveillance footage.

## Layout

```
faces_cv/
  detection.py   Detection + Detector protocol; UltralyticsDetector (YOLO11) backend
  tracking.py    Track + Tracker protocol; ByteTrackTracker backend; COCO class maps
  pipeline.py    VideoTracker / track_video / TrackingResult (metrics, render, export)
examples/
  01_dvr_person_tracking.py    ByteTrack person tracking on DVR proxies (SDK demo)
  02_dvr_vehicle_tracking.py   same, for COCO vehicle classes (car/motorcycle/bus/truck)
tests/             fast unit suite (no GPU/weights/data needed)
context/implementation-plan.md  the system design
```

## Setup

```bash
uv sync
```

Pulls `surveillance-datasets` (from git) + torch/ultralytics/opencv. `yolo11s.pt`
auto-downloads on first detector use.

## Use the SDK

```python
from faces_cv import track_video

result = track_video("clip.mp4", targets=("person", "vehicle"), device="cuda")
result.metrics()                       # per-track polars DataFrame
result.to_parquet("tracks.parquet", what="tracks")
result.render("annotated.mp4")         # annotated H.264 video
```

Both axes are pluggable: any object satisfying the `Detector` (`.detect(frame)`) or
`Tracker` (`.update(dets, frame)` / `.reset()`) protocol drops into `VideoTracker`.

## Run a notebook

```bash
uv run marimo edit examples/01_dvr_person_tracking.py
```

Run from the **repo root** so `import faces_cv` resolves.

## Notes

- Displayed videos are H.264 and downscaled to 480p so they play inline; **detection
  always runs at full resolution** (`TrackingResult.render` handles the downscale +
  H.264 transcode — this OpenCV wheel has no H.264 encoder).
- Tests: `uv run pytest` (synthetic fakes; no model/GPU/data needed).
