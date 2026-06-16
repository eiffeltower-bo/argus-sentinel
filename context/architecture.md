# argus — architecture & how to extend

`argus` is a model-agnostic SDK for surveillance-footage analysis. It's organized as a
strict downward dependency stack so the extension contracts live in one place and new
capabilities drop in as a file, not a rewrite.

```
core/   →  detect/ track/ face/ embed/ store/  →  pipeline/  →  identity/  →  __init__.py  →  mcp/
(no deps)         (backends + persistence)          (orchestrate)  (search/reID)  (facade)       (server)
```

Each arrow reads "is imported by"; a layer only imports from layers to its left. `detect`,
`track`, `face`, `embed`, and `store` are independent siblings over `core`; `pipeline` composes
them (`ingest_video` chains detect→track→face→embed→`store`); `identity` adds the search/re-ID
read path over a `SearchableStore`; `mcp` is an optional server that consumes only the facade.

## Layers

- **`core/`** — the dependency-free foundation. Import-light (no cv2/ultralytics/polars):
  - `types.py` — frozen dataclasses: `Detection`, `Track`, `FaceDetection`, and the persisted
    face records `Sighting`, `Identity`, `Enrollment`, `SearchHit`, `WatchlistHit`.
  - `protocols.py` — the **extension contracts**: `Detector` (`.detect(frame)`), `Tracker`
    (`.update(dets, frame)` / `.reset()`), `FaceDetector`, `Embedder`, and `Store` /
    `SearchableStore` (write-only ingest vs. the richer search/admin/compliance surface). This
    is the one place to look to see what's pluggable.
  - `taxonomy.py` — COCO maps (`COCO_LABELS`, `CATEGORY_BY_CLASS`, `TARGET_CLASSES`) and
    `classes_for(targets)`.
- **`detect/`** — object detection. `backends/ultralytics.py` (`UltralyticsDetector`,
  YOLO11; also an optional `detect_batch` for throughput).
- **`track/`** — tracking-by-detection. `backends/bytetrack.py` (`ByteTrackTracker` driving
  ultralytics' standalone `BYTETracker`).
- **`face/`** — face detection + chip preparation. `detect/backends/insightface.py`
  (`InsightFaceDetector`, SCRFD), `align.py` (5-point aligned chips), `gate.py` (`QualityGate`).
- **`embed/`** — face embedding. `backends/insightface.py` (`InsightFaceEmbedder`, ArcFace,
  512-d L2-normalized, so cosine == dot product).
- **`store/`** — persistence. `backends/sqlite.py` (`SqliteStore`, the `SearchableStore` over
  SQLite + sqlite-vec: cosine KNN, `embedding_space_id` partition key, camera/ts/quality
  metadata columns for in-scan filtering; identities, enrollments, cluster runs, audit log).
- **`pipeline/`** — user-facing orchestration. `tracking.py` (decode → detect → track →
  `TrackingResult` with metrics/render/export), `peek.py` (fast clip triage), `ingest.py`
  (`ingest_video` → detect→track→face→embed→persist one best-face `Sighting` per track),
  `_video.py` (shared decode/sampling helpers).
- **`identity/`** — the face-ID read path over a `SearchableStore`. `search.py`
  (`search_by_image`, `search_by_sighting` → ranked `SearchHit`s + evidence chips), `admin.py`
  (`enroll`, `merge`, `label_cluster`, `reassign`, `audit_log`, `purge`, `export_case`),
  `cluster.py` (`run_clustering`, HDBSCAN over the embeddings → provisional identities). Every
  search/admin op takes an `actor` and writes an audit row.
- **`mcp/`** — optional MCP server (`server.py`, `FastMCP`) exposing the facade as five HTTP
  tools (`list_clips`, `peek_folder`, `peek_clip`, `track_clip`, `search_face`); `_serialize.py`
  turns result objects into JSON. A pure consumer of the public surface — the MCP analogue of
  the CLI. Run with `argus-mcp`; needs the `mcp` extra. See [mcp-server.md](mcp-server.md).
- **`argus/__init__.py`** — thin facade re-exporting the public surface.

Rule: a layer only imports from layers below it. `detect`/`track`/`face`/`embed`/`store` are
independent siblings (none imports another); `pipeline` composes them; `identity` builds on
`store` (+ `face`/`embed` to embed a probe). Heavy imports (ultralytics, insightface,
sqlite-vec, scikit-learn) stay lazy inside the backend/functions, so `import argus` is cheap
and works without the optional extras installed.

## How to add new functionality

- **A new detector backend** (ONNX, TensorRT, another model): add
  `detect/backends/<name>.py` with a class implementing `core.Detector` (`.detect(frame) ->
  list[Detection]`; optionally `.detect_batch(frames, *, batch_size)`), then re-export it
  from `detect/__init__.py`. It works in `VideoTracker`/`track_video`/`peek_*` immediately
  via the `detector=` argument.
- **A new tracker backend** (e.g. BoT-SORT): add `track/backends/<name>.py` implementing
  `core.Tracker`, re-export from `track/__init__.py`. Pass it as `tracker=`.
- **A new face detector / embedder** (YuNet, AdaFace): add `face/detect/backends/<name>.py`
  (`core.FaceDetector`) or `embed/backends/<name>.py` (`core.Embedder`), re-export, and inject
  via `ingest_video`'s / `search_by_image`'s `face_detector=` / `embedder=` arguments. An
  embedder must set its own `embedding_space_id` so the store never compares across spaces.
- **A new store backend** (e.g. pgvector): implement `core.SearchableStore` in
  `store/backends/<name>.py`. Search/admin/identity code depends only on the Protocol, so it
  works unchanged; the synthetic `FakeStore` in `tests/conftest.py` is the reference for the
  contract (brute-force cosine parity).

## Reserved drop-in spots (not built yet)

The detect → track → face → embed → store → search stack is built; what remains designed-for
but intentionally unbuilt:

- **Composable stage pipeline** — `pipeline/stages.py` with a small `Stage` Protocol + runner to
  chain detect → track → face → embed declaratively; `pipeline/_video.py` is the shared-decode
  seam. Today `ingest.py` wires the stages directly.
- **Auth on the MCP server** — the tools take a caller-supplied `actor`; binding it to an
  authenticated principal (OAuth) is designed in [mcp-oauth-plan.md](mcp-oauth-plan.md).

Keep it minimal: the Protocols + `detector=`/`tracker=`/`face_detector=`/`embedder=` injection
are the plug mechanism — no registries, factories, or empty scaffolding.
