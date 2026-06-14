# Surveillance Face‑ID & Re‑ID System — Implementation Plan

**Status:** Draft for review · v0.2 · 2026-06-13
**Owner:** jose.laruta@instrumental-inc.com

> A forensic, on‑premise computer‑vision system that reads recorded surveillance video, detects
> pedestrians, extracts and embeds faces, and supports identification via vector search — both
> **watchlist matching** against a known gallery and **open re‑identification** (clustering of
> unknown people). Driven through a **Python SDK** (no REST service for now). Output is **ranked
> candidates with evidence for a human to adjudicate**, not automated identification.

---

## 1. Confirmed scope & constraints

These were settled during design Q&A and drive every decision below.

| Dimension | Decision | Consequence |
|---|---|---|
| Processing mode | **Forensic / batch** (archived files, search later) | Latency‑tolerant. Optimize for accuracy + operator workflow, not real‑time. Enables superior **offline** clustering. |
| Identification goal | **Both** — watchlist + open re‑ID | Two‑part identity model: enrolled `known` identities + `provisional` clusters of unknowns. |
| Deployment | **On‑prem / air‑gapped** | Self‑hostable open models only, no cloud APIs. Compliance (audit, retention, access) is in‑scope. |
| Scale | **Small** — handful of cameras, a few hours/day | Single‑box. No cluster, no Kafka, no autoscaling. |
| Interface | **Python SDK** (no REST API for now) | Operators drive it from scripts/notebooks/CLI. HTTP layer can be added later if needed. |
| Hardware | **One 24 GB consumer GPU** (advised; see §7) | Runs the entire system; ~10× faster than real time. |
| Footage quality | **Mixed** (1080p, variable distance/angle, many small faces) | Strict quality gate mandatory; many faces unusable → "investigative leads, not proof." |
| Watchlist size | **≤ 1k identities** | Exact cosine search for the gallery (instant, most accurate); ANN only for the sightings index. |
| Model licensing | **Undecided → build model‑agnostic** | Pluggable backends; PoC on research weights, commercial‑clean path kept open. |

---

## 2. Architecture overview

Single‑box deployment. Postgres runs via Docker; the `faceid` SDK is pip‑installed. Bundled offline for air‑gap.

```
   operator script / notebook / thin CLI
              │  import faceid
              ▼
   ┌─────────────────────────────────────────────────┐
   │  faceid  (Python SDK)                            │
   │  ingest · search · identity · clustering · admin │
   └───────┬───────────────────────────────┬──────────┘
           │ runs pipeline                  │ queries / writes
           ▼                                ▼
   ┌──────────────────────────┐     ┌────────────────┐
   │ Ingest pipeline           │     │ Postgres 16    │
   │ decode→detect→track→face  │────▶│ + pgvector     │
   │ →gate→align→embed         │     │ (vectors+meta) │
   └───────────▲──────────────┘     └────────────────┘
   video files │                            ▲
   (NAS / dir) │              aligned chips + crops
               │              on local FS / MinIO
   offline jobs (SDK calls): HDBSCAN clustering · re-embed migration
```

**Components**
- **`faceid` Python SDK** — the library operators import; exposes ingest, search, identity management, clustering, and admin/compliance calls. No HTTP/REST service.
- **Ingest pipeline** — the GPU pipeline; invoked via the SDK (or a thin CLI). Runs synchronously or as a background process; writes sightings + persists aligned chips.
- **Postgres 16 + pgvector** — single datastore for vectors *and* metadata (one thing to operate, back up, audit). HNSW index for sightings, exact scan for the ≤1k watchlist.
- **Object store** — local filesystem (or MinIO) for aligned face chips + evidence crops.
- **Job queue (optional, deferred)** — Redis + RQ only if parallel/background ingest is later needed; synchronous is fine at this scale.
- **UI** — out of scope; operators drive the SDK from scripts/notebooks.

---

## 3. Processing pipeline

```
decode (NVDEC, sample ~4 fps)
  → person detect ──▶ track (BoT-SORT)            tracks give continuity + body re-ID fallback
  → face detect (high-res / tiled)
  → align (5-pt landmarks → 112×112 chip)         ⟵ PERSIST chip here
  → QUALITY GATE (size / blur / pose / score)     drop unusable faces; score the rest
  → embed BEST face per track (512-d)             one vector per track, tagged embedding_space_id
  → Postgres + pgvector
        ├─ query path:  probe face → ranked candidates + evidence crops (human reviews)
        └─ offline:     HDBSCAN cluster unknowns → provisional "Person #N" identities
```

| # | Stage | Detail | Notes |
|---|---|---|---|
| 1 | **Decode & sample** | NVDEC GPU decode (PyAV/Decord/DALI). Sample ~4 fps (per‑camera configurable). | Don't process every frame; tracking fills gaps. Decode is the usual bottleneck → keep it on GPU. |
| 2 | **Person detect** | Person class only, confidence threshold. | `PersonDetector` interface. PoC: YOLO11. Commercial‑clean: YOLOX. |
| 3 | **Track** | BoT‑SORT associates boxes into per‑person tracks across frames. | Offline → can use full sequence. Track = the unit we embed once. |
| 4 | **Face detect** | Run at **high resolution / tiled** to catch small faces; emit bbox + 5 landmarks; associate to person track. | `FaceDetector` interface. PoC: SCRFD. Commercial‑clean: YuNet. |
| 5 | **Align** | Similarity transform from 5 landmarks → canonical 112×112 chip. **Persist the chip.** | Persisting chips makes a future model swap a cheap re‑embed (§4). |
| 6 | **Quality gate** | Hard rejects: face size < N px, blur (Laplacian var) low, pose (yaw/pitch) extreme, det score low. Soft: store composite quality score. | Critical for mixed footage. v1 uses proxy metrics; optional upgrade to FIQA (CR‑FIQA/SER‑FIQ). |
| 7 | **Embed** | Best face per track → 512‑d vector, tagged `embedding_space_id`. Optionally keep top‑K best chips per track. | `FaceEmbedder` interface. PoC: ArcFace / **AdaFace (preferred for mixed)**. |
| 8 | **Index/persist** | Write sighting rows + HNSW index (per embedding space). | Vectors are tiny (2 KB each); video files dominate storage. |
| 9 | **Query** (interactive) | Probe image → detect → align → embed → search watchlist (exact) + sightings (HNSW) → ranked candidates + crops. | Returns leads for human review. |
| 10 | **Offline re‑ID** (batch) | HDBSCAN (cosine) over unlabeled sightings → provisional identities; operator merges/labels/reassigns. | Optional body re‑ID signal to link sightings when face is unusable. |

---

## 4. Model abstraction & strategy

Because licensing is undecided, the pipeline is **model‑agnostic**: each model sits behind an interface, selected by config.

```python
class PersonDetector(Protocol):
    def detect(self, frame: NDArray) -> list[Detection]: ...        # boxes + scores

class FaceDetector(Protocol):
    def detect(self, image: NDArray) -> list[FaceDetection]: ...     # box + 5 landmarks + score

class FaceEmbedder(Protocol):
    embedding_space_id: str        # e.g. "arcface_glintr100_v1"
    dim: int                        # 512
    def embed(self, aligned_chips: list[NDArray]) -> NDArray: ...    # (N, dim), L2-normalized

class ReIDExtractor(Protocol):     # optional, body appearance
    def extract(self, person_crops: list[NDArray]) -> NDArray: ...
```

### Licensing posture (pending legal)
High‑accuracy *open* face‑recognition weights are trained on restricted datasets (MS1M/WebFace) and are **non‑commercial**. Two tracks, same interfaces:

| Role | Research‑weights (PoC) | Commercial‑clean (later) |
|---|---|---|
| Person detect | YOLO11 — ⚠️ AGPL‑3.0 | YOLOX / RTMDet — Apache‑2.0 |
| Face detect | SCRFD — ⚠️ non‑commercial | YuNet — Apache‑2.0 |
| Face embed | ArcFace / AdaFace — ⚠️ non‑commercial | Licensed SDK or self‑trained ArcFace |
| Clustering | HDBSCAN — ✅ BSD | (same) |

### Two non‑obvious rules (load‑bearing)
1. **Embeddings are not comparable across models.** ArcFace ≠ AdaFace vector space. Every vector carries an `embedding_space_id`; searches filter by it. Swapping the recognizer = **re‑embed the whole index**.
2. **Persist aligned face chips.** A model swap then = cheap re‑embed of stored chips (a batch job), *not* a full re‑decode/re‑detect of all video. This is what makes "let legal decide later" painless.

---

## 5. Data model (Postgres + pgvector)

Sketch — column lists abbreviated.

```sql
videos        (id, camera_id, path, duration_s, fps, ingested_at, status)
jobs          (id, video_id, type, status, progress, started_at, finished_at, error)
tracks        (id, video_id, camera_id, start_ts, end_ts, frame_count)

sightings     (id, video_id, camera_id, track_id, frame_idx, ts,
               bbox, face_chip_path, quality_score,
               embedding vector(512), embedding_space_id,
               identity_id NULL, cluster_id NULL, created_at)

identities    (id, type ['known'|'provisional'], label NULL,
               created_by, created_at, notes)

enrollments   (id, identity_id, face_chip_path,
               embedding vector(512), embedding_space_id, source)   -- watchlist gallery

cluster_runs  (id, algo, params, embedding_space_id, created_at)
audit_log     (id, actor, action, target_type, target_id, query_ref, ts, details)
```

**Indexing**
- `sightings.embedding` → **HNSW**, partial per `embedding_space_id`.
- `enrollments.embedding` → exact cosine scan (≤1k is instant, most accurate).
- B‑tree on `(camera_id, ts)`, `identity_id`, `cluster_id` for timeline/filter queries.

---

## 6. SDK design (`faceid`)

**Principle:** every search returns **ranked candidates + evidence crops** with scores and camera/time context — for human adjudication. Never an automated yes/no.

```python
import faceid

faceid.connect("postgresql://localhost/faceid")          # or faceid.Session(config)

# --- Ingestion ---
job = faceid.ingest_video("cam-3", "/footage/2026-06-01_cam3.mp4")
faceid.ingest_dir("cam-3", "/footage/incoming/")          # batch a folder
job.status                                                 # progress / state

# --- Investigation ---
res = faceid.search_by_image("probe.jpg",
          cameras=["cam-3"], since="2026-06-01", until="2026-06-02",
          top_k=20, min_quality=0.4)
for c in res.candidates:
    c.identity, c.score, c.evidence       # evidence = crops + camera_id + ts + quality

faceid.search_by_sighting(9921)            # "find more of this person"
faceid.identity_timeline(17)               # sightings over time, per camera
faceid.list_sightings(camera="cam-3", since="2026-06-01", identity=17)

# --- Identity management ---
pid = faceid.enroll("J. Doe", ["a.jpg", "b.jpg"])   # add a watchlist identity
faceid.run_clustering()                              # offline HDBSCAN over unknowns
faceid.merge(into=17, source_cluster=244)            # cluster → known identity
faceid.label_cluster(244, "Suspect A")
faceid.reassign(sighting_id=9921, identity_id=17)    # fix a mis-clustered sighting

# --- Admin / compliance ---
faceid.audit_log(actor="jose", since="2026-06-01")
faceid.purge(before="2026-01-01")                    # retention policy
faceid.export_case([17, 244], "case_001.zip")        # logged
```

| Group | SDK call | Purpose |
|---|---|---|
| Ingestion | `ingest_video(camera, path)` · `ingest_dir(camera, dir)` | Process a file / batch a folder. |
| | `job.status` · `faceid.job(id)` | Progress. |
| Investigation | `search_by_image(img, **filters)` | Probe → ranked candidates (watchlist + sightings). |
| | `search_by_sighting(id)` | "Find more of this person." |
| | `identity_timeline(id)` | Sightings over time. |
| | `list_sightings(**filters)` | Filtered list (camera, time, identity, quality). |
| Identity | `enroll(name, photos)` | Add a known watchlist identity. |
| | `run_clustering()` | Offline HDBSCAN over unlabeled sightings. |
| | `merge(into, source_cluster)` | Merge a provisional cluster into an identity. |
| | `label_cluster(id, name)` | Name a provisional cluster. |
| | `reassign(sighting_id, identity_id)` | Fix a mis‑clustered sighting. |
| Compliance | `audit_log(**filters)` | Audit trail. |
| | `purge(before=...)` | Apply retention policy. |
| | `export_case(ids, path)` | Generate a case file (logged). |

---

## 7. Inference requirements & hardware

**Throughput math** (4 fps sampling): ~25 ms GPU work/frame (person ~8 ms + face detect high‑res ~10 ms + embed ~2 ms/face) ⇒ **~10× faster than real time per stream**. ~40 camera‑hours ⇒ **~4 h compute** = an overnight batch on one GPU.

| Tier | GPU | Verdict |
|---|---|---|
| **Minimum viable** | RTX 4070 12 GB / 4060 Ti 16 GB | Finishes daily volume overnight. |
| **Recommended** | **RTX 4090 24 GB (new) or used RTX 3090 24 GB** | Headroom for AdaFace + high‑res/tiled detection. Used 3090 = cost play. |
| Overkill | A100 / L40S | Unnecessary at this scale. |

- **Host:** 8+ core CPU, 32–64 GB RAM, NVMe SSD. **Use NVDEC** so decode doesn't bottleneck CPU.
- **Inference runtime:** ONNX Runtime (CUDA EP). PyTorch only for AdaFace load/export + any fine‑tune. TensorRT optional, unneeded at this scale.
- **Storage:** video files dominate. Vectors trivial (512‑d f32 = 2 KB; 1M sightings ≈ 2 GB). Aligned chips a few KB each.

---

## 8. Compliance & operations

On‑prem/air‑gapped surveillance biometrics raise this from "nice to have" to **in‑scope**:
- **Audit log** — every search/export recorded with the invoking actor, target, timestamp.
- **Access control** — DB credentials + the audit records who ran what. Multi‑user roles deferred (no HTTP layer yet).
- **Retention/purge** — configurable policy via `faceid.purge(...)`; purge raw video, chips, and embeddings on schedule.
- **Data minimization** — store only what's needed; chips + vectors over raw frames where possible.
- **Air‑gap packaging** — Postgres via Docker; `faceid` SDK + model weights shipped as an offline wheel/bundle.
- **Jurisdiction** — TBD (see §11); EU/CNIL vs US BIPA materially change retention + consent rules.

---

## 9. Expectations & risk register

| Risk | Mitigation |
|---|---|
| **Mixed footage → many unusable faces** | Strict quality gate; set expectation = *investigative leads, not proof*; body re‑ID fallback. |
| **False matches** | Human‑in‑the‑loop review of ranked candidates + evidence; never auto‑identify. |
| **Model licensing** | Model‑agnostic interfaces; PoC on research weights; commercial‑clean path defined. |
| **Model swap invalidates index** | `embedding_space_id` namespacing + persisted chips → cheap re‑embed migration. |
| **Clustering errors (over/under‑merge)** | Operator merge/label/reassign tooling; tune HDBSCAN on sample footage. |
| **Quality thresholds need real data** | Calibrate gate on a labeled sample of the actual cameras before trusting results. |

---

## 10. Build roadmap

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| **0 — Contracts** | Repo skeleton, `faceid` package layout, model interfaces, pgvector schema, Postgres compose, config. | `pip install -e .`; Postgres up via compose; schema migrates; model interfaces stubbed. |
| **1 — Ingest PoC** | `ingest_video` pipeline on research weights; persist sightings **+ chips**; thin CLI to process one file. | Process a sample video end‑to‑end; sightings + chips written; counts/quality logged. |
| **2 — Search & watchlist** | `enroll` ≤1k known; exact search; `search_by_image` / `search_by_sighting` returning evidence crops. | Enroll a person, find them in processed footage with ranked candidates + crops. |
| **3 — Open re‑ID** | `run_clustering` (HDBSCAN); provisional identities; `merge` / `label_cluster` / `reassign`. | Unknowns cluster into stable identities; operator can name + merge; sightings reassignable. |
| **4 — Compliance & packaging** | `audit_log`, `purge`, `export_case`, offline wheel + weights bundle. | Every search audited; retention job purges on policy; offline install verified. |
| *Optional* | Body re‑ID modality; commercial‑clean model swap + re‑embed migration. | Re‑embed job rebuilds index under a new `embedding_space_id`. |

---

## 11. Tech stack summary

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Interface | **`faceid` Python SDK** (+ thin CLI) — no REST service |
| Inference | ONNX Runtime (CUDA EP); PyTorch for AdaFace/export; TensorRT optional |
| Person detect / track | YOLO11 → YOLOX (clean) · BoT‑SORT |
| Face detect / embed | SCRFD → YuNet (clean) · ArcFace/AdaFace (model‑agnostic) |
| Clustering | HDBSCAN |
| Datastore | Postgres 16 + pgvector (vectors + metadata) |
| Job execution | Synchronous or background process; Redis + RQ optional if parallel ingest needed |
| Decode | PyAV / Decord / NVIDIA DALI (NVDEC) |
| Packaging | Postgres via Docker; SDK + weights as an offline pip/wheel bundle |

---

## 12. Open questions for stakeholders

1. **SDK consumers** — Confirm the SDK (driven from scripts/notebooks) is sufficient for now. Do you want a small set of **CLI commands / notebook recipes** shipped alongside, or just the library?
2. **Jurisdiction & retention** — Where does this operate (EU/CNIL vs US/BIPA)? Sets retention windows, consent, and audit requirements.
3. **Video delivery** — How do files arrive (watch folder, manual path, NAS)? Defines the ingest entry point.
4. **Enrollment source** — Where do the ≤1k watchlist photos come from (ID photos vs footage stills)? Affects enrollment quality + workflow.
5. **Body re‑ID** — Include the appearance‑based modality now (valuable on mixed footage) or defer to optional phase?
6. **FIQA** — Use a dedicated face‑quality model (CR‑FIQA/SER‑FIQ) from the start, or proxy metrics (size/blur/pose/score) in v1?
7. **Multi‑user** — Single operator for now, or do we need per‑user separation later (would motivate adding the HTTP layer back)?
