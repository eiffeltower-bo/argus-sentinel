# argus — project guide for Claude

On-prem **surveillance-footage analysis & understanding**: detect → track → triage recorded
video, designed to extend toward face-ID/re-ID, embeddings, and search.

- **Start here:** [context/architecture.md](context/architecture.md) (module map + how to
  extend) and [context/face-id-design.md](context/face-id-design.md) (face-ID system design,
  v0.3 — argus-native). The older [context/implementation-plan.md](context/implementation-plan.md)
  is **legacy/superseded**; kept only for its scope/compliance/risk analysis.
- **SDK:** `argus/` is a model-agnostic, subpackaged library. Layers (strict downward deps):
  - `core/` — shared `Detection`/`Track` types, the extension **Protocols** (`Detector`,
    `Tracker`), and the COCO taxonomy. Dependency-free; the place new contracts go.
  - `detect/` — `Detector` backends (`detect/backends/ultralytics.py`).
  - `track/` — `Tracker` backends (`track/backends/bytetrack.py`).
  - `pipeline/` — orchestration: `tracking.py` (`VideoTracker`, `track_video`,
    `TrackingResult`) and `peek.py` (`peek_video`, `peek_videos`, `PeekResult`).
  - Public surface is re-exported from `argus/__init__.py` (`from argus import track_video, …`).
- **Add a backend** by implementing the relevant Protocol from `argus.core` and dropping a
  file in `detect/backends/` or `track/backends/` — no registry needed.
- **Notebooks** are marimo (`.py`) in `examples/`, run from the repo root
  (`uv run marimo edit examples/NN_*.py`) so `import argus` resolves. They are thin demos —
  `examples/01_dvr_person_tracking.py` is the template for video work.
- Always **detect at full resolution, display downscaled**; rendering writes `mp4v` then
  transcodes to H.264 via system `ffmpeg` (this OpenCV wheel has no H.264 encoder).
- Tests: `uv run pytest` — fast synthetic suite, no GPU/weights/data needed.

## marimo notebook rules

@marimo.md
