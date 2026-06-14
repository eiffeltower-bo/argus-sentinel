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
    # 🚶→🙂 Two-stage detection: pedestrian, then face

    **Stage 1** detects pedestrians (`YoloPersonDetector`). **Stage 2** runs a face
    detector — **YuNet** (default) or Haar, selectable below — *on each person crop*,
    not the whole frame, so a face gets more relative pixels. Person = green, face = red.

    Source: **MOT17** (loaded from a manual download via `path=`). The street-level
    sequences (**MOT17-09 / -11 / -02**) have the most detectable faces. YuNet finds
    ~2× the faces of Haar here; swap either for SCRFD (same `FaceDetector` interface) for more.
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
    # Loaded once at a low threshold; the person-confidence slider filters downstream.
    person_detector = YoloPersonDetector(weights="yolo11n.pt", conf=0.01)
    return (person_detector,)


@app.cell
def _(ds, mo):
    seq_idx = mo.ui.slider(0, len(ds) - 1, value=3, label="sequence (3 = MOT17-09)")
    n_show = mo.ui.slider(4, 12, value=6, step=1, label="frames to show")
    person_conf = mo.ui.slider(0.05, 0.9, value=0.25, step=0.05, label="person conf")
    backend = mo.ui.dropdown(["yunet", "haar"], value="yunet", label="face detector")
    face_score = mo.ui.slider(
        0.3, 0.9, value=0.6, step=0.05, label="YuNet score threshold"
    )
    mo.vstack([seq_idx, n_show, person_conf, backend, face_score])
    return backend, face_score, n_show, person_conf, seq_idx


@app.cell
def _(HaarFaceDetector, YuNetFaceDetector, backend, face_score):
    # Rebuilt when the backend or threshold changes (both are cheap).
    if backend.value == "yunet":
        face_detector = YuNetFaceDetector(score_threshold=face_score.value)
    else:
        face_detector = HaarFaceDetector(min_neighbors=3, min_size=12)
    return (face_detector,)


@app.cell(hide_code=True)
def _(cv2, ds, face_detector, n_show, person_conf, person_detector, seq_idx):
    clip = ds[seq_idx.value]
    total = clip.media.num_frames
    k = n_show.value
    # k evenly-spaced frame indices across the sequence
    idxs = sorted({int(round(i * (total - 1) / max(1, k - 1))) for i in range(k)})

    results = []
    for t in idxs:
        rgb = clip.frame(t)  # HWC RGB uint8 (from the library)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # detectors expect BGR

        persons = [d for d in person_detector.detect(bgr) if d.score >= person_conf.value]
        face_count = 0
        for p in persons:
            x1, y1, x2, y2 = (max(0, int(v)) for v in p.xyxy)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # person = green

            crop = bgr[y1:y2, x1:x2]  # stage 2 runs on the person crop
            if crop.size == 0:
                continue
            for face in face_detector.detect(crop):
                fx1, fy1, fx2, fy2 = (int(v) for v in face.xyxy)
                # map crop-local coords back to full-frame coords
                cv2.rectangle(
                    rgb, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (255, 0, 0), 2
                )  # face = red
                face_count += 1

        results.append((t, len(persons), face_count, rgb))
    return clip, results


@app.cell(hide_code=True)
def _(clip, mo, results):
    n_persons = sum(p for _, p, _, _ in results)
    n_faces = sum(f for _, _, f, _ in results)
    header = mo.md(
        f"### {clip.id} — {len(results)} frames · "
        f"**{n_persons}** pedestrians · **{n_faces}** faces"
    )
    gallery = mo.hstack(
        [
            mo.vstack([mo.image(im, width=320), mo.md(f"frame {t}: 🚶 {p} · 🙂 {f}")])
            for t, p, f, im in results
        ],
        wrap=True,
    )
    mo.vstack([header, gallery])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
