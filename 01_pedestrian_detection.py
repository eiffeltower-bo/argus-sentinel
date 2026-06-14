import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 🚶 Pedestrian detection on UCSD-Ped2

    Reactive version of `ucsd_ped_detect.py`. Loads UCSD-Ped2 (test) through
    **surveillance-datasets**, runs the `YoloPersonDetector` from `faces_cv`, and
    overlays pedestrian boxes. Move the sliders below to re-run detection live.
    """)
    return


@app.cell
def _():
    import cv2
    import surveillance_datasets as svd
    from faces_cv.detection import YoloPersonDetector

    return YoloPersonDetector, cv2, svd


@app.cell
def _(svd):
    # Downloaded on first use, then cached.
    ds = svd.load("ucsd-ped", split="test", variant="ped2", accept_license=True)
    return (ds,)


@app.cell
def _(YoloPersonDetector):
    # Load the model once at a low threshold; we filter by the confidence slider
    # downstream so moving it doesn't reload the model.
    detector = YoloPersonDetector(weights="yolo11n.pt", conf=0.01)
    return (detector,)


@app.cell
def _(ds, mo):
    clip_idx = mo.ui.slider(0, len(ds) - 1, value=0, label="clip")
    stride = mo.ui.slider(10, 60, value=30, step=10, label="frame stride")
    conf = mo.ui.slider(0.05, 0.9, value=0.25, step=0.05, label="confidence")
    mo.vstack([clip_idx, stride, conf])
    return clip_idx, conf, stride


@app.cell(hide_code=True)
def _(clip_idx, conf, cv2, detector, ds, stride):
    clip = ds[clip_idx.value]
    n_frames = clip.media.num_frames

    results = []
    for t in range(0, n_frames, stride.value):
        rgb = clip.frame(t)  # HWC RGB uint8, straight from the library
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # ultralytics expects BGR
        dets = [d for d in detector.detect(bgr) if d.score >= conf.value]
        for d in dets:
            p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
            cv2.rectangle(rgb, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                rgb, f"{d.score:.2f}", (p1[0], p1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )
        results.append((t, len(dets), rgb))
    return clip, results


@app.cell(hide_code=True)
def _(clip, mo, results):
    total = sum(count for _, count, _ in results)
    header = mo.md(
        f"### {clip.id} — {len(results)} frames sampled, **{total}** pedestrians total"
    )
    gallery = mo.hstack(
        [
            mo.vstack([mo.image(im, width=260), mo.md(f"frame {t}: **{count}**")])
            for t, count, im in results
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
