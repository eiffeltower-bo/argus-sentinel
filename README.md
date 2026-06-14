# faces

Prototypes for an on-prem, forensic **surveillance face-ID system**: read recorded
video → detect pedestrians → extract faces → (later) embed + search a vector index.
Full design in [context/implementation-plan.md](context/implementation-plan.md).

This repo is the **exploration stage**: a small model-agnostic detector library plus
marimo notebooks that validate the two-stage **pedestrian → face** pipeline on real
surveillance footage. The next step runs on a GPU workstation — see
[HANDOFF.md](HANDOFF.md).

## Layout

```
faces_cv/detection.py   PersonDetector / FaceDetector interfaces + backends
                        (YOLO11 person; YuNet + Haar face) — model-agnostic
01_pedestrian_detection.py    UCSD-Ped2: single-stage person detection (reactive)
02_two_stage_face.py          MOT17: person -> face on each crop (Haar vs YuNet)
03_mot17_video_pipeline.py    MOT17: plain / ground-truth / prediction videos + fps
context/implementation-plan.md  the system design
models/                 committed YuNet ONNX (face detector weights)
```

## Setup

```bash
uv sync
```

Pulls `surveillance-datasets` (from git) + torch/ultralytics/opencv. `yolo11n.pt`
auto-downloads on first use; the YuNet model is committed under `models/`.

## Run a notebook

```bash
uv run marimo edit 01_pedestrian_detection.py
```

Notebooks live at the repo root so `import faces_cv` resolves when marimo runs them.

## Notes

- Datasets used (UCSD-Ped2 auto-downloads; MOT17 is a manual download referenced by an
  absolute path in notebooks 02/03 — edit `MOT17_PATH` for your machine).
- Displayed videos are H.264 (`avc1`) and downscaled to 480p so they play inline and fit
  marimo's output budget; **detection always runs at full resolution**.
