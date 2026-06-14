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
    # 🚗 DVR vehicle tracking — the `faces_cv` SDK

    Same SDK as `01`, pointed at **vehicles** instead of people. One call switches the
    target — the COCO vehicle classes (car, motorcycle, bus, truck) — and the per-track
    `type` falls out of the metrics for free:

    ```python
    from faces_cv import track_video
    result = track_video(clip, targets=("vehicle",), device="cuda", max_frames=300)
    ```

    `track_video` builds an `UltralyticsDetector` (YOLO11s) restricted to the vehicle
    classes + a `ByteTrackTracker`, and returns a **`TrackingResult`**. From it:

    - `result.track_ids` — distinct vehicles seen.
    - `result.metrics()` — per-track table; its `type` column is the dominant detected
      class per track (so a track that flickers car/truck still gets one label).
    - `result.render(path)` — annotated H.264 clip (`label id score` per box).

    The *min track frames* slider re-filters instantly (no re-tracking). Heavy step —
    runs only when you click **Run tracking**.
    """)
    return


@app.cell
def _():
    import time
    from pathlib import Path
    import random

    import altair as alt
    import polars as pl

    from faces_cv import UltralyticsDetector, peek_video, track_video

    return Path, UltralyticsDetector, alt, peek_video, pl, random, time, track_video


@app.cell
def _(Path, random):
    # Flat folder of already-transcoded H.264 proxies (shared with `01`).
    PROXIES_DIR = Path("/home/pepe/dev/dani_y_jose/sentinel/data/proxies")
    SEED = 42  # fixed -> the same 20 clips every run (matches `01`)
    N_CLIPS = 20

    all_clips = sorted(PROXIES_DIR.glob("*.mp4"))
    selected_clips = random.Random(SEED).sample(all_clips, min(N_CLIPS, len(all_clips)))
    selected_clips = sorted(selected_clips)  # stable dropdown order
    return all_clips, selected_clips


@app.cell
def _(all_clips, mo, selected_clips):
    clip_pick = mo.ui.dropdown(
        {p.name: p for p in selected_clips},
        value=selected_clips[0].name,
        label="clip",
    )
    max_frames = mo.ui.slider(30, 1800, value=300, step=30, label="max frames")
    vehicle_conf = mo.ui.slider(0.05, 0.9, value=0.25, step=0.05, label="vehicle conf")
    # Reactive, post-tracking: re-filters the metrics without re-running the tracker.
    min_track_frames = mo.ui.slider(1, 30, value=5, step=1, label="min track frames")
    run_btn = mo.ui.run_button(label="Run tracking")
    mo.vstack(
        [
            mo.md(f"**{len(selected_clips)}** of {len(all_clips)} proxies selected (seed-fixed)."),
            clip_pick,
            max_frames,
            vehicle_conf,
            run_btn,
        ]
    )
    return clip_pick, max_frames, min_track_frames, run_btn, vehicle_conf


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Peek first

    Most surveillance clips have no vehicles. Before committing to full tracking, **peek**
    scans a sparse, evenly-spaced sample of each clip's frames with a small detector at low
    resolution (`yolo11n` @ `imgsz=320`) — fast triage to spot which proxies have traffic.

    ```python
    from faces_cv import peek_video
    peek_video(clip, targets=("vehicle",), device="cuda").interesting   # -> True / False
    ```
    """)
    return


@app.cell
def _(UltralyticsDetector):
    # One small, low-res detector reused across every peek scan (avoids reloading weights
    # per clip). classes=[2,3,5,7] -> car/motorcycle/bus/truck. Built once; cheap to reuse.
    peek_detector = UltralyticsDetector(
        weights="yolo11n.pt", classes=[2, 3, 5, 7], conf=0.35, device="cuda", imgsz=320
    )
    return (peek_detector,)


@app.cell
def _(mo):
    peek_btn = mo.ui.run_button(label="Peek selected clips")
    peek_btn
    return (peek_btn,)


@app.cell(hide_code=True)
def _(mo, peek_btn, peek_detector, peek_video, pl, selected_clips):
    mo.stop(
        not peek_btn.value,
        mo.md("_Click **Peek selected clips** for a quick activity scan (no tracking)._"),
    )
    peek_rows, skipped = [], []
    for _clip in selected_clips:
        try:
            _pk = peek_video(_clip, targets=("vehicle",), detector=peek_detector)
        except RuntimeError:
            skipped.append(_clip.name)  # unreadable / corrupt proxy (e.g. 0-byte file)
            continue
        peek_rows.append(
            {
                "clip": _clip.name,
                "interesting": _pk.interesting,
                "vehicle": _pk.counts.get("vehicle", 0),
                "frames_hit": _pk.frames_with_hits,
                "sampled": _pk.n_sampled,
            }
        )
    peek_df = pl.DataFrame(peek_rows)
    _n_ok = peek_df.height
    _n_interesting = peek_df.filter(pl.col("interesting")).height if _n_ok else 0
    if _n_ok:
        peek_df = peek_df.sort(["interesting", "vehicle"], descending=True)
    _skip = f" · {len(skipped)} unreadable, skipped" if skipped else ""
    mo.vstack(
        [
            mo.md(
                f"### Peek — {_n_interesting} of {_n_ok} clips look interesting "
                f"(pick one below to track){_skip}"
            ),
            mo.ui.table(peek_df),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    Path,
    clip_pick,
    max_frames,
    mo,
    run_btn,
    time,
    track_video,
    vehicle_conf,
):
    mo.stop(
        not run_btn.value,
        mo.md("▶️ Set the options above, then click **Run tracking** to render."),
    )

    src_path = clip_pick.value

    # The whole SDK pipeline: decode + detect (YOLO11s vehicle classes) + ByteTrack.
    _t0 = time.perf_counter()
    result = track_video(
        src_path,
        targets=("vehicle",),
        weights="yolo11s.pt",
        conf=vehicle_conf.value,
        device="cuda",
        max_frames=max_frames.value,
    )
    elapsed = time.perf_counter() - _t0

    out_path = Path("out/examples") / f"{src_path.stem}_vehicle.mp4"
    result.render(out_path, display_height=480)
    trk_bytes = out_path.read_bytes()

    n_frames = len(result.frames)
    e2e_fps = n_frames / elapsed if elapsed else 0.0
    speed_msg = (
        f"⚡ end-to-end (decode + detect + track): **{e2e_fps:.1f} fps** "
        f"({elapsed / n_frames * 1000:.0f} ms/frame) — YOLO11s (cuda) + ByteTrack at "
        f"{result.width}×{result.height} over {n_frames} frames"
    )
    return result, speed_msg, src_path, trk_bytes


@app.cell(hide_code=True)
def _(mo, result, speed_msg, src_path, trk_bytes):
    peak = max((len(tracks) for _, tracks in result.frames), default=0)
    mo.vstack(
        [
            mo.md(
                f"### {src_path.name} — {len(result.frames)} frames · "
                f"**{len(result.track_ids)} distinct vehicles** tracked · "
                f"peak {peak} on screen at once"
            ),
            mo.md(speed_msg),
            mo.video(trk_bytes, width=720),
        ]
    )
    return


@app.cell
def _(result):
    # Per-track metrics straight from the SDK (computed once per run).
    metrics_all = result.metrics()
    return (metrics_all,)


@app.cell
def _(metrics_all, min_track_frames):
    # Reactive: re-filters when the min-track-frames slider moves (no re-tracking).
    metrics_df = metrics_all.filter(metrics_all["n_frames"] >= min_track_frames.value)
    return (metrics_df,)


@app.cell(hide_code=True)
def _(metrics_all, metrics_df, min_track_frames, mo, pl):
    mo.stop(
        not metrics_all.height,
        mo.md("_Per-track metrics appear here after you run tracking._"),
    )
    by_type = metrics_df.group_by("type").len().sort("len", descending=True)
    breakdown = " · ".join(f"{row['type']}: {row['len']}" for row in by_type.iter_rows(named=True))
    metrics_view = metrics_df.select(
        pl.col("id"),
        pl.col("type"),
        pl.col("first_s").round(2),
        pl.col("last_s").round(2),
        pl.col("duration_s").round(2),
        pl.col("n_frames"),
        pl.col("continuity").round(2),
        pl.col("avg_w").round(0),
        pl.col("avg_h").round(0),
        pl.col("avg_area_pct").round(2),
        pl.col("avg_conf").round(2),
        pl.col("min_conf").round(2),
        pl.col("entry_edge"),
        pl.col("exit_edge"),
    ).sort("first_s")
    mo.vstack(
        [
            mo.md(
                f"### Per-track metrics — {metrics_df.height} of {metrics_all.height} "
                f"tracks (≥ {min_track_frames.value} frames) · {breakdown}"
            ),
            mo.ui.table(metrics_view),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, metrics_all, metrics_df, mo):
    mo.stop(not metrics_all.height, mo.md(""))
    mo.stop(
        not metrics_df.height,
        mo.md("_No tracks pass the min-frames filter — lower the slider._"),
    )
    timeline = (
        alt.Chart(metrics_df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("first_s:Q", title="time in clip (s)"),
            x2="last_s:Q",
            y=alt.Y(
                "id:N",
                title="track id",
                sort=alt.EncodingSortField(field="first_s", op="min", order="ascending"),
            ),
            color=alt.Color("type:N", title="vehicle type"),
            tooltip=[
                "id", "type", "first_s", "last_s", "duration_s", "n_frames",
                "continuity", "avg_area_pct", "avg_conf", "entry_edge", "exit_edge",
            ],
        )
        .properties(
            width=680,
            height=alt.Step(22),
            title="Vehicle track timeline — when each ID was on screen",
        )
    )
    mo.ui.altair_chart(timeline)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### People + vehicles at once

    The detector and tracker handle both categories in a single pass — pass both targets
    and the `category` / `type` columns tell them apart:

    ```python
    result = track_video(clip, targets=("person", "vehicle"), device="cuda")
    result.metrics()  # has `category` ("person"/"vehicle") and `type` (car/truck/...)
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
