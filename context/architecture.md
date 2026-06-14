# argus — architecture & how to extend

`argus` is a model-agnostic SDK for surveillance-footage analysis. It's organized as a
strict downward dependency stack so the extension contracts live in one place and new
capabilities drop in as a file, not a rewrite.

```
core/      ← detect/  track/   ← pipeline/   ← argus/__init__.py (facade)
(no deps)     (backends)           (orchestration)
```

## Layers

- **`core/`** — the dependency-free foundation. Import-light (no cv2/ultralytics/polars):
  - `types.py` — `Detection`, `Track` (frozen dataclasses).
  - `protocols.py` — the **extension contracts**: `Detector` (`.detect(frame)`), `Tracker`
    (`.update(dets, frame)` / `.reset()`). This is the one place to look to see what's pluggable.
  - `taxonomy.py` — COCO maps (`COCO_LABELS`, `CATEGORY_BY_CLASS`, `TARGET_CLASSES`) and
    `classes_for(targets)`.
- **`detect/`** — object detection. `backends/ultralytics.py` (`UltralyticsDetector`,
  YOLO11; also an optional `detect_batch` for throughput).
- **`track/`** — tracking-by-detection. `backends/bytetrack.py` (`ByteTrackTracker` driving
  ultralytics' standalone `BYTETracker`).
- **`pipeline/`** — user-facing orchestration. `tracking.py` (decode → detect → track →
  `TrackingResult` with metrics/render/export), `peek.py` (fast clip triage), `_video.py`
  (shared decode/sampling helpers).
- **`argus/__init__.py`** — thin facade re-exporting the public surface.

Rule: a layer only imports from layers below it. `detect` and `track` never import each
other; only `pipeline` imports both. Heavy imports (ultralytics, BYTETracker) stay lazy
inside the backend classes, so `import argus` is cheap.

## How to add new functionality

- **A new detector backend** (ONNX, TensorRT, another model): add
  `detect/backends/<name>.py` with a class implementing `core.Detector` (`.detect(frame) ->
  list[Detection]`; optionally `.detect_batch(frames, *, batch_size)`), then re-export it
  from `detect/__init__.py`. It works in `VideoTracker`/`track_video`/`peek_*` immediately
  via the `detector=` argument.
- **A new tracker backend** (e.g. BoT-SORT): add `track/backends/<name>.py` implementing
  `core.Tracker`, re-export from `track/__init__.py`. Pass it as `tracker=`.

## Reserved drop-in spots (not built yet)

These are designed-for but intentionally unbuilt — add them when first needed:

- **Face stage** — `argus/face/`: `align.py` + `backends/{scrfd,yunet,arcface,adaface}.py`.
  New contracts (`FaceDetector`, `Embedder`) and types (`Face`) go in `core/`, keeping the
  stack flat.
- **Output sinks / storage** — `argus/sinks/` (parquet, pgvector, index) behind a minimal
  `Sink` Protocol added to `core/protocols.py` once the embed stage produces records.
- **Composable stage pipeline** — `pipeline/stages.py` with a small `Stage` Protocol + runner
  to chain detect → track → face → embed; `pipeline/_video.py` is the shared-decode seam.

Keep it minimal: the Protocols + `detector=`/`tracker=` injection are the plug mechanism —
no registries, factories, or empty scaffolding.
