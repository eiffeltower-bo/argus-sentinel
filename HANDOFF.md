# HANDOFF — continue on the GPU workstation

For a Claude session running **on the workstation** (`server`), with a human driving via
VSCode Remote-SSH + marimo in the browser. Goal: build & run the **DVR inference notebook**
on real footage, using the workstation's **GPU**.

---

## 1. Where things stand

This repo (`eiffeltower-bo/faces`) is the exploration stage of an on-prem forensic
**surveillance face-ID** system. Design: [context/implementation-plan.md](context/implementation-plan.md).

**Built & verified on a Mac (CPU):**
- `faces_cv/detection.py` — model-agnostic interfaces + backends:
  - `PersonDetector` / `YoloPersonDetector` (YOLO11n, COCO person class).
  - `FaceDetector` / `YuNetFaceDetector` (OpenCV YuNet, ~230 KB, returns 5 landmarks) and
    `HaarFaceDetector` (zero-dep fallback).
- `01_pedestrian_detection.py` — UCSD-Ped2 single-stage (faces too distant → ~0 faces; expected).
- `02_two_stage_face.py` — MOT17 person→face on each crop. YuNet ≈ 2× Haar's recall.
- `03_mot17_video_pipeline.py` — renders plain / ground-truth / prediction videos with a
  timed-FPS readout. **Copy this as the template for the DVR notebook.**

Measured CPU throughput (Mac, MOT17-09 @ 1080p): full two-stage ≈ **26 fps / 38 ms/frame**.
On the workstation GPU the person stage should be much faster (YuNet stays on CPU).

---

## 2. Workstation setup

```bash
git clone git@github.com:eiffeltower-bo/faces.git
cd faces
uv sync
```

- `uv sync` installs torch/ultralytics/opencv + `surveillance-datasets` (pulled from git —
  the workstation needs SSH access to `eiffeltower-bo/surveillance_datasets`).
- `yolo11n.pt` auto-downloads on first detector use (needs internet). The YuNet model is
  committed at `models/face_detection_yunet_2023mar.onnx`.
- Verify GPU:
  ```bash
  uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  ```
- Make the person detector use the GPU: `YoloPersonDetector(weights="yolo11n.pt", device="cuda")`
  (or `device=0`). Ultralytics auto-selects CUDA if present, but be explicit. **YuNet runs on
  CPU** (OpenCV), so the face stage won't use the GPU — that's fine.

---

## 3. How the human connects

1. VSCode **Remote-SSH** into `server`, open the cloned `faces/` folder.
2. Launch marimo headless and forward the port:
   ```bash
   uv run marimo edit 04_dvr_inference.py --headless --host 0.0.0.0 -p 2718
   ```
   VSCode auto-forwards port 2718; open the printed `http://localhost:2718/...?access_token=...`
   URL locally. (Headless validation without a browser: `uv run python 04_dvr_inference.py` →
   should exit 0; the heavy cell is gated behind a Build button so it won't run.)

---

## 4. Immediate task: `04_dvr_inference.py` (spec agreed with the user)

**Data — already LOCAL on the workstation** (no transfer needed):
`/home/pepe/dev/eiffel_tower_projects/recordings_analysis/data/channel_1`
- 3,540 `.dav` clips across 23 day-folders (`2026-03-13` … `2026-04-03`), 28 GB total.
- Each clip: **HEVC (H.265), 960×1080, 30 fps, ~49 s, ~8 MB**, with `pcm_alaw` audio.

**Pipeline (decisions the user already made):**
1. **Select 20 random clips across all days**:
   `find <channel_1> -name '*.dav' | shuf -n 20` (set a seed / save the list for reproducibility).
   Copy them into `dvr_raw/` (gitignored). Note: motion-triggered → some clips may have no people.
2. **Transcode HEVC `.dav` → H.264 MP4** with system ffmpeg (OpenCV can't read `.dav`/HEVC
   reliably, and `mo.video` needs H.264):
   `ffmpeg -y -i in.dav -an -c:v libx264 -pix_fmt yuv420p out.mp4`
   This MP4 is both the displayed "original" and the decode source for detection.
3. **Detect every frame** (user chose full fidelity, not subsampled) with the two-stage
   pipeline: `YoloPersonDetector(device="cuda")` → for each person crop, `YuNetFaceDetector`.
4. **Workflow = on-demand per clip**: a `mo.ui.dropdown` of the 20 clips → selecting one
   transcodes (if needed) + detects + renders. Don't batch all 20 up front.
5. **Display**: original vs annotated **side-by-side** (`mo.video`), person=green/face=red with
   confidences, plus the **timed FPS** line — exactly the pattern in `03_mot17_video_pipeline.py`.

**Reuse from `03`:** the full-res-detect / 480p-display split, `avc1` writer, `det_time` FPS
timing, and the run-button gating. The main differences: source is `.dav` (transcode first),
no ground-truth video, GPU device, and a clip dropdown instead of a sequence dropdown.

---

## 5. Gotchas already paid for (don't rediscover)

- **`mo.video` needs H.264.** Use `cv2.VideoWriter_fourcc(*"avc1")` (works here via ffmpeg);
  `mp4v` won't play in the browser.
- **marimo output cap.** Embedded video output is limited (default 10 MB). We downscale
  displayed videos to 480p and set `[tool.marimo.runtime] output_max_bytes` in `pyproject.toml`.
  Keep displayed clips small; **detect at full res, display downscaled**.
- **Detection resolution matters for faces** — downscaling the detection input loses ~17–38%
  of faces. Never downscale the detector input, only the displayed frame.
- **`pi_heif`** is a dependency because importing `ultralytics` monkeypatches `PIL.Image.open`
  for HEIF; without it, PIL-based image reads throw. Already in `pyproject.toml`.
- **`surveillance-datasets`** frame-dir reads filter hidden files (a `.DS_Store` fix landed in
  its `core/media.py`); not relevant to `.dav` work but FYI.
- **marimo single-definition rule**: a variable can be assigned in only one cell. Loop and
  tuple-unpack targets count — name per-frame accumulators uniquely.

---

## 6. Pointers

- Design & constraints: `context/implementation-plan.md`
- Detector API: `faces_cv/detection.py`
- Template to copy: `03_mot17_video_pipeline.py`
- marimo editing rules: `marimo.md` (also pulled in via `CLAUDE.md`)

After the DVR notebook works, the next milestone toward the real system is **stage 3:
align faces via the YuNet landmarks → embed (ArcFace/AdaFace) → populate a vector index**
(see the plan's Phase 2/3).
