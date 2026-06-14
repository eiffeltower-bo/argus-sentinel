# faces — project guide for Claude

On-prem forensic **surveillance face-ID** prototypes: pedestrian → face detection on
recorded video, heading toward face embedding + vector search.

- **Start here:** [HANDOFF.md](HANDOFF.md) (current state + next task) and
  [context/implementation-plan.md](context/implementation-plan.md) (the system design).
- **Detectors:** `faces_cv/detection.py` — model-agnostic `PersonDetector` / `FaceDetector`
  interfaces with YOLO11 (person) and YuNet/Haar (face) backends.
- **Notebooks** are marimo (`.py`), run from the repo root (`uv run marimo edit NN_*.py`) so
  `import faces_cv` resolves. `03_mot17_video_pipeline.py` is the template for video work.
- Always **detect at full resolution, display downscaled**; use `avc1` (H.264) for `mo.video`.

## marimo notebook rules

@marimo.md
