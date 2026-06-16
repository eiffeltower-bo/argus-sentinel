# argus — project guide for Claude

On-prem **surveillance-footage analysis & understanding**: detect → track → triage recorded
video, designed to extend toward face-ID/re-ID, embeddings, and search.

- **Start here:** [context/architecture.md](context/architecture.md) (module map + how to
  extend) and [context/face-id-design.md](context/face-id-design.md) (face-ID system design,
  v0.3 — argus-native). The older [context/implementation-plan.md](context/implementation-plan.md)
  is **legacy/superseded**; kept only for its scope/compliance/risk analysis.
- **SDK:** `argus/` is a model-agnostic, subpackaged library. Layers (strict downward deps):
  - `core/` — shared types (`Detection`/`Track`/`FaceDetection`/`Sighting` + the identity/search
    records `Identity`/`Enrollment`/`SearchHit`/`WatchlistHit`), the extension **Protocols**
    (`Detector`, `Tracker`, `FaceDetector`, `Embedder`, `Store`/`SearchableStore`), and the COCO
    taxonomy. Dependency-free; the place new contracts go.
  - `detect/` — `Detector` backends (`detect/backends/ultralytics.py`).
  - `track/` — `Tracker` backends (`track/backends/bytetrack.py`).
  - `face/` — face detection + chip prep: `face/detect/backends/insightface.py` (`FaceDetector`),
    `align.py` (aligned chips), `gate.py` (`QualityGate`).
  - `embed/` — `Embedder` backends (`embed/backends/insightface.py`, ArcFace, 512-d L2-normed).
  - `store/` — `Store`/`SearchableStore` backends (`store/backends/sqlite.py`, `SqliteStore`
    over sqlite-vec: cosine KNN, partition by embedding space, in-scan metadata filters).
  - `pipeline/` — orchestration: `tracking.py` (`VideoTracker`, `track_video`, `TrackingResult`),
    `peek.py` (`peek_video`, `peek_videos`, `PeekResult`), `ingest.py` (`ingest_video` →
    detect→track→face→embed→persist `Sighting`s).
  - `identity/` — face search / re-ID / compliance over `SearchableStore`: `search.py`
    (`search_by_image`, `search_by_sighting`), `admin.py` (`enroll`, `merge`, `label_cluster`,
    `reassign`, `audit_log`, `purge`, `export_case`), `cluster.py` (`run_clustering`, HDBSCAN).
  - `mcp/` — optional MCP server (`server.py`) exposing the facade as 5 HTTP tools (`list_clips`,
    `peek_folder`, `peek_clip`, `track_clip`, `search_face`); a pure consumer of the public
    surface — the MCP analogue of the CLI. Needs the `mcp` extra; run `argus-mcp`.
  - Public surface is re-exported from `argus/__init__.py` (`from argus import track_video, …`).
- **Add a backend** by implementing the relevant Protocol from `argus.core` and dropping a file
  in `detect/backends/`, `track/backends/`, `face/detect/backends/`, or `embed/backends/` — no
  registry needed.
- **Extras** (lazy, opt-in): `face`/`face-gpu` (insightface + onnxruntime), `store` (sqlite-vec),
  `cluster` (scikit-learn), `mcp`. `import argus` stays cheap — heavy deps load only on use.
- **Notebooks** are marimo (`.py`) in `examples/`, run from the repo root
  (`uv run marimo edit examples/NN_*.py`) so `import argus` resolves. They are thin demos —
  `examples/01_dvr_person_tracking.py` is the template for video work.
- Always **detect at full resolution, display downscaled**; rendering writes `mp4v` then
  transcodes to H.264 via system `ffmpeg` (this OpenCV wheel has no H.264 encoder).
- Tests: `uv run pytest` — fast synthetic suite, no GPU/weights/data needed.

## marimo notebook rules

@marimo.md
