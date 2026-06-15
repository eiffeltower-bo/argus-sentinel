# Face-ID — argus-native design

**Status:** Draft for review · v0.3 · 2026-06-14
**Owner:** jose.laruta@instrumental-inc.com
**Supersedes:** the *architecture* of [implementation-plan.md](implementation-plan.md) (v0.2).
The v0.2 plan's scope, compliance, risk, and SDK-surface thinking still hold; this doc
replaces its "separate `faceid` package + Postgres" architecture with an **argus-native**
one, reflecting that `argus` now exists as the detect→track foundation with reserved
face-ID drop-in spots ([architecture.md](architecture.md#L43-L53)).

> Forensic, on-prem face-ID over recorded surveillance video. Detect → track → **face →
> gate → align → embed → store**, then search a watchlist (exact) + sightings (ANN) and
> cluster unknowns (HDBSCAN). Output is **ranked candidates + evidence for a human to
> adjudicate** — never automated identification. Driven through the `argus` Python SDK.

---

## 1. Settled decisions (this exploration, 2026-06-14)

| Dimension | Decision | Why |
|---|---|---|
| **Packaging** | Face-ID **extends `argus`** (new subpackages + `core` contracts), one SDK | architecture.md already reserves `face/`, `sinks/`; reuses detect→track directly |
| **Pipeline** | A dedicated **`ingest_video()`** orchestrator (not a generic Stage runner yet) | One consumer today → YAGNI on the abstraction; keep stages as testable functions so a `Stage` refactor is mechanical at the *second* consumer |
| **Datastore** | **SQLite + sqlite-vec**, single file (vectors *and* metadata) | No daemon/Docker; trivially air-gapped; native SQL for audit/retention/timeline; brute-force KNN is fine at this scale |
| **Face detection** | Run the face detector **on tracked person crops** | Cheaper (skip empty regions), natural zoom on small/distant faces, faces auto-associate to `track_id` |
| **Models** | **Research weights** (SCRFD + AdaFace) behind Protocols | Best accuracy on mixed footage to prove the system; `embedding_space_id` makes a clean-license swap a batch re-embed |
| **Body re-ID** | **Deferred** to an optional phase | Face-only keeps core, schema, and search surface small; add as a 2nd embedding space if face-unusable footage proves a real gap |
| **Quality gate** | **Proxy metrics** v1 (size / Laplacian blur / pose / det score) + composite score | No extra model; calibrate on real camera samples; upgrade to FIQA (CR-FIQA/SER-FIQ) later if recall suffers |
| **Per-track embed** | **Best face only** — one 512-d vector per track | Minimal storage, simplest search; revisit top-K if single-pick hurts recall |

Carried over unchanged from v0.2: forensic/batch mode, on-prem/air-gapped, ≤1k watchlist
(exact cosine) + open re-ID, single 24 GB consumer GPU, "leads not proof," human-in-the-loop.

---

## 2. Architecture — argus grows a memory

`argus` stops being purely stateless: face-ID adds heavy stages behind Protocols **and** a
persistent `store/`. Strict downward deps preserved (`core` ← stages ← `pipeline`/`identity`).

```
argus/
  core/
    types.py        + Face, FaceDetection (extend existing Detection/Track)
    protocols.py    + FaceDetector, Embedder, Store
  detect/  track/   (exist — person detect + tracks, unchanged)
  face/
    detect/backends/{scrfd,yunet}.py   FaceDetector — runs on person crops
    align.py                            5-pt landmarks → 112×112 similarity transform
    gate.py                             quality gate: hard rejects + composite score
  embed/
    backends/{adaface,arcface}.py       Embedder — embedding_space_id, dim=512, L2-norm
  store/
    backends/sqlite.py                  SQLite + sqlite-vec (the new persistent state)
    schema.sql                          tables + the vec virtual table
  pipeline/
    tracking.py · peek.py (exist)
    ingest.py        ingest_video() — the face-ID orchestrator
  identity/
    search.py        search_by_image, search_by_sighting, identity_timeline
    cluster.py       run_clustering (HDBSCAN over unknowns)
    admin.py         enroll, merge, label_cluster, reassign, audit_log, purge
```

**Heavy imports stay lazy** in backends (onnxruntime, insightface, hdbscan) so `import
argus` stays cheap, consistent with the existing rule.

---

## 3. New `core/` contracts

Types (frozen dataclasses, dependency-free — extend `core/types.py`):

```python
@dataclass(frozen=True)
class FaceDetection:
    x1: float; y1: float; x2: float; y2: float
    score: float
    landmarks: tuple[tuple[float, float], ...]   # 5-pt: eyes, nose, mouth corners

@dataclass(frozen=True)
class Face:
    track_id: int                 # which person track this face belongs to
    frame_idx: int
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    quality: float                # composite gate score
    chip_path: str | None = None  # persisted aligned 112×112 chip
```

Protocols (extend `core/protocols.py`):

```python
class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[FaceDetection]: ...   # box + 5 landmarks

class Embedder(Protocol):
    embedding_space_id: str       # e.g. "adaface_ir101_webface12m_v1"
    dim: int                      # 512
    def embed(self, aligned_chips: list[np.ndarray]) -> np.ndarray: ...  # (N, dim) L2-norm

class Store(Protocol):
    def add_sightings(self, rows: list[Sighting]) -> None: ...
    def search(self, vec: np.ndarray, space_id: str, *, top_k: int, filters=...) -> list[Hit]: ...
    def enroll(self, identity_id: int, vecs, chips, space_id: str) -> None: ...
    # + admin: audit, purge, timeline, cluster bookkeeping
```

**Two load-bearing rules** (from v0.2, still central): embeddings are **not comparable
across models** — every vector carries `embedding_space_id` and searches filter by it;
and **persist aligned chips** so a model swap is a cheap re-embed, not a re-decode.

---

## 4. Ingest pipeline (`pipeline/ingest.py`)

The structural difference from `track_video` is a **granularity shift** mid-pipeline:
frame-level perception → per-track aggregation → persist.

```
FRAME-level stream                  ──►  TRACK-level aggregation       ──►  PERSIST
decode (sample ~4 fps, reuse           on track-end (or EOF):              one sighting
  pipeline/_video.py)                  pick BEST gated face per track,      vector/track →
→ person detect → track                batch-embed (AdaFace),               sqlite-vec +
→ face-detect ON each person crop      tag embedding_space_id               chip on disk
→ align (112×112) → quality gate
→ buffer Face candidates by track_id
```

- **Reuse** the existing detect→track core verbatim; face work hangs off live tracks.
- Buffer per-track face candidates; flush on track-end so memory stays bounded on long video.
- Batch the embed call (GPU-efficient) once per flush.
- Stages (`face.detect`, `face.align`, `face.gate`, `embed`) are **small testable
  functions/classes** — the seam that makes a future `pipeline/stages.py` Stage runner a
  mechanical lift if a second pipeline ever needs it.

Throughput (carried from v0.2): ~25 ms GPU/frame at 4 fps ⇒ ~10× real time; ~40
camera-hours ⇒ ~4 h = overnight on one 24 GB GPU.

---

## 5. Store — SQLite + sqlite-vec (single file)

Relational metadata and vectors in **one file** (`argus.db`). Exact cosine over the ≤1k
watchlist; brute-force KNN over sightings (fast enough at this scale; revisit if sightings
reach many millions). Tables mirror v0.2 §5 minus the Postgres-isms:

```
videos      (id, camera_id, path, duration_s, fps, ingested_at, status)
jobs        (id, video_id, type, status, progress, started_at, finished_at, error)
tracks      (id, video_id, camera_id, start_ts, end_ts, frame_count)
sightings   (id, video_id, camera_id, track_id, frame_idx, ts, bbox,
             chip_path, quality, embedding_space_id, identity_id NULL, cluster_id NULL)
identities  (id, type['known'|'provisional'], label NULL, created_by, created_at, notes)
enrollments (id, identity_id, chip_path, embedding_space_id, source)
cluster_runs(id, algo, params, embedding_space_id, created_at)
audit_log   (id, actor, action, target_type, target_id, query_ref, ts, details)

vec_sightings   -- sqlite-vec virtual table: rowid ↔ sightings.id, vector(512)
vec_enrollments -- sqlite-vec virtual table for the watchlist gallery
```

Indexes: B-tree on `(camera_id, ts)`, `identity_id`, `cluster_id`, `embedding_space_id`.
Vectors are tiny (512-d f32 = 2 KB; 1M sightings ≈ 2 GB); video files dominate storage.

---

## 6. SDK surface (face-ID additions to `argus`)

Mirrors the v0.2 SDK design (§6), re-exported from `argus/__init__.py`. Every search
returns **ranked candidates + evidence crops** (score, camera, ts, quality) for human
adjudication — never an automated yes/no.

```python
from argus import ingest_video, search_by_image, enroll, run_clustering

ingest_video("cam-3", "/footage/clip.mp4")                  # decode→…→store
res = search_by_image("probe.jpg", cameras=["cam-3"],
                      since="2026-06-01", top_k=20, min_quality=0.4)
search_by_sighting(9921)                                     # "find more of this person"
pid = enroll("J. Doe", ["a.jpg", "b.jpg"])                  # watchlist identity
run_clustering()                                             # offline HDBSCAN over unknowns
merge(into=pid, source_cluster=244); label_cluster(244, "Suspect A")
audit_log(actor="jose"); purge(before="2026-01-01")         # compliance
```

---

## 7. Stack (delta from v0.2)

| Layer | Choice |
|---|---|
| Inference | ONNX Runtime (CUDA EP); `insightface` for SCRFD + alignment + ArcFace; PyTorch only for AdaFace load/export |
| Person detect / track | YOLO11 · ByteTrack/BoT-SORT — **already in argus** |
| Face detect / align / embed | SCRFD → YuNet (clean) · 5-pt similarity → 112×112 · ArcFace/**AdaFace** |
| Quality gate | proxy metrics (size/blur/pose/score); FIQA optional later |
| Clustering | HDBSCAN (BSD) |
| **Datastore** | **SQLite + sqlite-vec** (single file) — *replaces Postgres+pgvector+Docker* |
| Job execution | synchronous / background process (no Redis at this scale) |
| Packaging | SDK + model weights as an offline wheel/bundle; **no Docker required** |

---

## 8. Build roadmap

| Phase | Deliverable | Acceptance |
|---|---|---|
| **0 — Contracts** | `core` types + Protocols (`FaceDetector`, `Embedder`, `Store`); `store/sqlite.py` schema + migration; package skeleton | `import argus`; `argus.db` initializes; Protocols stubbed; synthetic-suite green |
| **1 — Ingest PoC** | `ingest_video()` on research weights (SCRFD+AdaFace), face-on-crops, proxy gate, best-face embed, persist sightings **+ chips** | Process a sample clip end-to-end; sightings + chips written; counts/quality logged |
| **2 — Search & watchlist** | `enroll` ≤1k; exact gallery search; `search_by_image`/`search_by_sighting` → evidence crops | Enroll a person, find them in footage with ranked candidates + crops |
| **3 — Open re-ID** | `run_clustering` (HDBSCAN); provisional identities; `merge`/`label_cluster`/`reassign` | Unknowns cluster into stable identities; operator can name/merge/reassign |
| **4 — Compliance & packaging** | `audit_log`, `purge`, `export_case`; offline wheel + weights bundle | Every search audited; retention purges on policy; offline install verified |
| *Optional* | Body re-ID modality; commercial-clean model swap + re-embed migration; top-K per track; FIQA gate; `pipeline/stages.py` if a 2nd pipeline appears | Re-embed job rebuilds index under a new `embedding_space_id` |

---

## 9. Open questions remaining

1. **sqlite-vec at scale** — confirm brute-force KNN latency is acceptable at the expected
   sightings volume (months of footage); if not, the `Store` Protocol lets us swap in
   LanceDB/HNSW without touching the pipeline.
2. **Chip storage layout** — flat dir vs sharded by camera/date; ties to `purge` retention.
3. **AdaFace packaging** — ONNX export vs PyTorch at runtime for the air-gap bundle.
4. Carried from v0.2 §12: jurisdiction/retention (EU CNIL vs US BIPA), video delivery
   (watch-folder vs manual path), enrollment source (ID photos vs footage stills).
</content>
</invoke>
