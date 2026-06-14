# faces — project guide for Claude

On-prem forensic **surveillance face-ID** prototypes: pedestrian → face detection on
recorded video, heading toward face embedding + vector search.

- **Start here:** [HANDOFF.md](HANDOFF.md) (current state + next task) and
  [context/implementation-plan.md](context/implementation-plan.md) (the system design).
- **SDK:** `faces_cv/` is a model-agnostic detection + tracking library —
  `detection.py` (`Detection`, `Detector` protocol, `UltralyticsDetector`),
  `tracking.py` (`Track`, `Tracker` protocol, `ByteTrackTracker`), `pipeline.py`
  (`VideoTracker`, `track_video`, `TrackingResult`). Swap backends via the protocols.
- **Notebooks** are marimo (`.py`) in `examples/`, run from the repo root
  (`uv run marimo edit examples/NN_*.py`) so `import faces_cv` resolves. They are thin
  demos of the SDK — `examples/01_dvr_person_tracking.py` is the template for video work.
- Always **detect at full resolution, display downscaled**; rendering writes `mp4v` then
  transcodes to H.264 via system `ffmpeg` (this OpenCV wheel has no H.264 encoder).
- Tests: `uv run pytest` — fast synthetic suite, no GPU/weights/data needed.

## marimo notebook rules

@marimo.md
