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
    # 🪪 Face-ID ingest — the `argus` SDK

    A thin demo of the new **face-ID** stage on surveillance footage. The whole pipeline —
    decode → detect people → track → detect a face on each person crop → align → quality-gate
    → embed the **best face per track** → persist — is one call:

    ```python
    from argus import ingest_video, SqliteStore

    store = SqliteStore("argus.db")
    result = ingest_video(clip, "cam-1", store=store, device="cuda")
    ```

    Each tracked person yields at most **one sighting**: a 512-d face embedding, a quality
    score, and a persisted aligned **112×112 chip**, written to a single SQLite + sqlite-vec
    file. Below we run it on one clip and look at what survived the quality gate — on hard
    CCTV footage many faces are unusable, which is exactly why the gate exists ("investigative
    leads, not proof").

    Needs the optional backends: `uv sync --group face --group store`.
    """)
    return


@app.cell
def _():
    import shutil
    import time
    from pathlib import Path

    import altair as alt
    import cv2
    import polars as pl

    from argus import SqliteStore, ingest_video

    return Path, SqliteStore, alt, cv2, ingest_video, pl, shutil, time


@app.cell
def _(Path):
    DATASET = Path("/home/pepe/data/shoplifting_dataset")
    clips_by_cat = {
        cat: sorted((DATASET / cat).glob("*.mp4"))
        for cat in ("normal", "shoplifting")
    }
    return (clips_by_cat,)


@app.cell
def _(clips_by_cat, mo):
    category = mo.ui.dropdown(
        options=list(clips_by_cat),
        value="shoplifting",
        label="dataset split",
    )
    category
    return (category,)


@app.cell
def _(category, clips_by_cat, mo):
    _clips = clips_by_cat[category.value]
    clip_pick = mo.ui.dropdown(
        {p.name: p for p in _clips},
        value=_clips[0].name,
        label="clip",
    )
    max_frames = mo.ui.slider(60, 1200, value=300, step=60, label="max frames")
    device = mo.ui.dropdown(["cuda", "cpu"], value="cuda", label="device")
    run_btn = mo.ui.run_button(label="Run face-ID ingest")
    mo.vstack(
        [
            mo.md(f"**{len(_clips)}** clips in `{category.value}`."),
            clip_pick,
            max_frames,
            device,
            run_btn,
        ]
    )
    return clip_pick, device, max_frames, run_btn


@app.cell
def _(
    Path,
    SqliteStore,
    clip_pick,
    device,
    ingest_video,
    max_frames,
    mo,
    run_btn,
    shutil,
    time,
):
    mo.stop(
        not run_btn.value,
        mo.md("▶️ Pick a clip and click **Run face-ID ingest**."),
    )

    src = clip_pick.value
    work = Path("out/examples/faceid") / src.stem
    if work.exists():
        shutil.rmtree(work)  # fresh store each run so sightings don't accumulate
    work.mkdir(parents=True, exist_ok=True)

    store = SqliteStore(work / "argus.db")
    _t0 = time.perf_counter()
    result = ingest_video(
        src,
        src.stem,
        store=store,
        max_frames=max_frames.value,
        device=device.value,
    )
    elapsed = time.perf_counter() - _t0
    sightings = store.list_sightings()
    return elapsed, result, sightings, src


@app.cell(hide_code=True)
def _(elapsed, mo, result, src):
    _fps = result.n_frames / elapsed if elapsed else 0.0
    mo.vstack(
        [
            mo.md(f"### {src.name} — {result.summary()}"),
            mo.md(
                f"⚡ {result.n_frames} frames in {elapsed:.1f}s "
                f"(**{_fps:.1f} fps** end-to-end: detect + track + face + embed)"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(cv2, mo, sightings):
    mo.stop(
        not sightings,
        mo.md(
            "_No faces passed the quality gate — common on distant/low-res CCTV. "
            "Try another clip or raise **max frames**._"
        ),
    )

    _cards = []
    for _s in sorted(sightings, key=lambda r: r["quality"], reverse=True):
        _chip = cv2.cvtColor(cv2.imread(_s["chip_path"]), cv2.COLOR_BGR2RGB)
        _cards.append(
            mo.vstack(
                [
                    mo.image(_chip, width=112),
                    mo.md(f"**track {_s['track_id']}** · q {_s['quality']:.2f}"),
                ],
                align="center",
            )
        )
    _rows = [mo.hstack(_cards[i : i + 6], justify="start") for i in range(0, len(_cards), 6)]
    mo.vstack(
        [mo.md(f"### Best face per track — {len(sightings)} sightings (sorted by quality)")]
        + _rows
    )
    return


@app.cell
def _(pl, sightings):
    sightings_df = pl.DataFrame(sightings)
    return (sightings_df,)


@app.cell(hide_code=True)
def _(alt, mo, sightings_df):
    mo.stop(not sightings_df.height, mo.md(""))
    chart = (
        alt.Chart(sightings_df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("track_id:N", title="track id"),
            y=alt.Y("quality:Q", title="best-face quality", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("quality:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["track_id", "frame_idx", "ts", "quality"],
        )
        .properties(width=600, height=260, title="Best-face quality per tracked person")
    )
    mo.ui.altair_chart(chart)
    return


@app.cell(hide_code=True)
def _(mo, pl, sightings_df):
    mo.stop(not sightings_df.height, mo.md(""))
    _view = sightings_df.select(
        pl.col("id"),
        pl.col("track_id"),
        pl.col("frame_idx"),
        pl.col("ts").round(2),
        pl.col("quality").round(3),
        pl.col("embedding_space_id"),
    ).sort("track_id")
    mo.vstack([mo.md("### Sighting rows (one per track)"), mo.ui.table(_view)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### What's next (later phases)

    Each row above is a 512-d vector in the `vec_sightings` index, tagged with an
    `embedding_space_id`. The search/enroll phase turns this into investigation:

    ```python
    faceid.enroll("J. Doe", ["photo.jpg"])      # add a watchlist identity
    faceid.search_by_image("probe.jpg")          # ranked candidates + evidence crops
    ```

    Those calls aren't built yet — this notebook covers ingest only (Phases 0–1). See
    `context/face-id-design.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
