import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🎬 MOT17 video pipeline: plain → ground truth → predictions

    1. Pick a MOT17 sequence and render two clips: a **plain** video and one with all
       **ground-truth** pedestrian boxes (yellow).
    2. Play them with `mo.video`.
    3. Decode the rendered *plain* video back and run our two-stage detector on it
       (pedestrian → face), writing per-frame **person (green) + face (red)** boxes with
       **confidences**, and timing the pipeline.
    4. Play the result.

    Detection runs at **full resolution**; the displayed videos are downscaled to 480p
    (so they fit marimo's output budget) with overlays drawn at display scale. Heavy step —
    runs only when you click **Build videos**.
    """)
    return


@app.cell
def _():
    import cv2
    import surveillance_datasets as svd
    from surveillance_datasets import Task
    from faces_cv.detection import (
        HaarFaceDetector,
        YoloPersonDetector,
        YuNetFaceDetector,
    )

    return (
        HaarFaceDetector,
        Task,
        YoloPersonDetector,
        YuNetFaceDetector,
        cv2,
        svd,
    )


@app.cell
def _(Task, svd):
    # Manually-downloaded MOT17 (edit if your path differs).
    MOT17_PATH = "/Users/joselaruta/dev/dani_y_jose/eiffel_tower/data/mot17"
    ds = svd.load("mot17", split="train", path=MOT17_PATH, task=Task.OBJECT_DETECTION)
    return (ds,)


@app.cell
def _(YoloPersonDetector):
    # Loaded once; the person-confidence slider filters its output downstream.
    person_detector = YoloPersonDetector(weights="yolo11n.pt", conf=0.01)
    return (person_detector,)


@app.cell
def _(ds, mo):
    ids = [s.id for s in ds]
    default = "MOT17-09-FRCNN" if "MOT17-09-FRCNN" in ids else ids[0]
    seq = mo.ui.dropdown(ids, value=default, label="sequence")
    max_frames = mo.ui.slider(30, 525, value=120, step=15, label="max frames")
    person_conf = mo.ui.slider(0.05, 0.9, value=0.25, step=0.05, label="person conf")
    backend = mo.ui.dropdown(["yunet", "haar"], value="yunet", label="face detector")
    face_score = mo.ui.slider(0.3, 0.9, value=0.6, step=0.05, label="YuNet score")
    build_btn = mo.ui.run_button(label="Build videos")
    mo.vstack([seq, max_frames, person_conf, backend, face_score, build_btn])
    return backend, build_btn, face_score, max_frames, person_conf, seq


@app.cell
def _(HaarFaceDetector, YuNetFaceDetector, backend, face_score):
    if backend.value == "yunet":
        face_detector = YuNetFaceDetector(score_threshold=face_score.value)
    else:
        face_detector = HaarFaceDetector(min_neighbors=3, min_size=12)
    return (face_detector,)


@app.cell(hide_code=True)
def _(
    backend,
    build_btn,
    cv2,
    ds,
    face_detector,
    max_frames,
    mo,
    person_conf,
    person_detector,
    seq,
):
    mo.stop(
        not build_btn.value,
        mo.md("▶️ Set the options above, then click **Build videos** to render."),
    )

    import time
    from pathlib import Path

    clip = next(s for s in ds if s.id == seq.value)
    FPS = 20
    n = min(max_frames.value, clip.media.num_frames)
    h, w = clip.media.height, clip.media.width

    # Display size: 480p, even dims (H.264 wants even). Detection stays full-res.
    dh = 480
    dw = int(round(w * dh / h)) // 2 * 2
    sx, sy = dw / w, dh / h  # full-res -> display scale, for drawing overlays

    out = Path("out/video")
    out.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 -> plays in the browser
    full_p = out / f"{clip.id}_plain_full.mp4"  # full-res, fed back to the detector
    plain_p = out / f"{clip.id}_plain.mp4"  # downscaled, for display
    gt_p = out / f"{clip.id}_gt.mp4"
    pred_p = out / f"{clip.id}_pred.mp4"

    def _sc(x, y):
        return int(x * sx), int(y * sy)

    # 1) plain (full-res for re-decode + 480p for display) and GT (480p) videos
    vw_full = cv2.VideoWriter(str(full_p), fourcc, FPS, (w, h))
    vw_plain = cv2.VideoWriter(str(plain_p), fourcc, FPS, (dw, dh))
    vw_gt = cv2.VideoWriter(str(gt_p), fourcc, FPS, (dw, dh))
    for t in range(n):
        bgr = cv2.cvtColor(clip.frame(t), cv2.COLOR_RGB2BGR)
        vw_full.write(bgr)
        disp = cv2.resize(bgr, (dw, dh))
        vw_plain.write(disp)
        gt = disp.copy()
        for box in clip.boxes_at(t):
            bx1, by1, bx2, by2 = box.as_xyxy()
            cv2.rectangle(gt, _sc(bx1, by1), _sc(bx2, by2), (0, 255, 255), 1)  # GT yellow
        vw_gt.write(gt)
    vw_full.release()
    vw_plain.release()
    vw_gt.release()

    # 2) predictions: decode the full-res video, detect at full res, draw on the 480p frame.
    #    `det_time` accumulates ONLY the two-stage detection (person + face), not I/O/drawing.
    cap = cv2.VideoCapture(str(full_p))
    vw_pred = cv2.VideoWriter(str(pred_p), fourcc, FPS, (dw, dh))
    summary = []
    det_time = 0.0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        _t0 = time.perf_counter()
        persons = [d for d in person_detector.detect(frame) if d.score >= person_conf.value]
        faces = []
        for d in persons:
            x1, y1, x2, y2 = (max(0, int(v)) for v in d.xyxy)
            crop = frame[y1:y2, x1:x2]  # full-res crop -> best face recall
            if crop.size:
                for fc in face_detector.detect(crop):
                    faces.append((x1 + fc.x1, y1 + fc.y1, x1 + fc.x2, y1 + fc.y2, fc.score))
        det_time += time.perf_counter() - _t0

        disp = cv2.resize(frame, (dw, dh))  # draw overlays at display scale (legible)
        for d in persons:
            cv2.rectangle(disp, _sc(d.x1, d.y1), _sc(d.x2, d.y2), (0, 255, 0), 1)
            tx, ty = _sc(d.x1, d.y1)
            cv2.putText(disp, f"p:{d.score:.2f}", (tx, ty - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        for (a, b, c, e, s) in faces:
            cv2.rectangle(disp, _sc(a, b), _sc(c, e), (0, 0, 255), 1)
            fx, fy = _sc(a, b)
            cv2.putText(disp, f"f:{s:.2f}" if s is not None else "f", (fx, fy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        vw_pred.write(disp)
        summary.append((fi, len(persons), len(faces)))
        fi += 1
    cap.release()
    vw_pred.release()

    det_fps = fi / det_time if det_time > 0 else 0.0
    det_ms = det_time / fi * 1000 if fi else 0.0
    speed_msg = (
        f"⚡ Full two-stage pipeline: **{det_fps:.1f} fps** ({det_ms:.0f} ms/frame) — "
        f"YOLO11n + {backend.value}, detecting at {w}×{h} over {fi} frames"
    )

    plain_bytes = plain_p.read_bytes()
    gt_bytes = gt_p.read_bytes()
    pred_bytes = pred_p.read_bytes()
    return clip, gt_bytes, plain_bytes, pred_bytes, speed_msg, summary


@app.cell(hide_code=True)
def _(clip, gt_bytes, mo, plain_bytes):
    mo.vstack(
        [
            mo.md(f"### {clip.id} — plain vs. ground truth (yellow)"),
            mo.hstack(
                [
                    mo.vstack([mo.md("**plain**"), mo.video(plain_bytes, width=440)]),
                    mo.vstack([mo.md("**ground truth**"), mo.video(gt_bytes, width=440)]),
                ],
                wrap=True,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, pred_bytes, speed_msg, summary):
    tp = sum(p for _, p, _ in summary)
    tf = sum(f for _, _, f in summary)
    mo.vstack(
        [
            mo.md(
                f"### Predictions — {len(summary)} frames · "
                f"{tp} pedestrians (green) · {tf} faces (red)"
            ),
            mo.md(speed_msg),
            mo.video(pred_bytes, width=720),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
