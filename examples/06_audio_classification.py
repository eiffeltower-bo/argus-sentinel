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
    # 🔊 Audio classification — the `argus` SDK

    A thin demo of the **audio** dimension on surveillance footage. The whole pipeline —
    extract a 16 kHz mono WAV via ffmpeg → window into overlapping segments → classify each
    with a HuggingFace audio model — is one call:

    ```python
    from argus import analyze_audio

    result = analyze_audio(clip, model="bioamla/ast-esc50", device="cuda")
    result.summary()      # dominant sound label
    result.metrics()      # per-segment predictions (polars)
    ```

    Pick an **AST** model for fixed labels (ESC-50 environmental sounds) or a **CLAP** model
    for zero-shot classification against your own candidate labels. Adapted from
    `paodanchacon/audio-search`. Needs the optional backend: `uv sync --extra audio`.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import polars as pl

    from argus import analyze_audio

    return Path, alt, analyze_audio, pl


@app.cell
def _(mo):
    folder = mo.ui.text(value="/home/pepe/data", label="footage folder", full_width=True)
    glob = mo.ui.text(value="**/*.mp4", label="glob")
    mo.vstack([folder, glob])
    return folder, glob


@app.cell
def _(Path, folder, glob, mo):
    _clips = sorted(Path(folder.value).glob(glob.value)) if Path(folder.value).exists() else []
    _options = {p.name: p for p in _clips} or {"(no clips found)": None}
    clip_pick = mo.ui.dropdown(_options, value=next(iter(_options)), label="clip")
    model = mo.ui.dropdown(
        [
            "bioamla/ast-esc50",
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            "laion/clap-htsat-unfused",
        ],
        value="bioamla/ast-esc50",
        label="model (clap = zero-shot)",
    )
    overlap = mo.ui.slider(0.0, 4.0, value=1.0, step=0.5, label="overlap (s)")
    segment = mo.ui.slider(1.0, 10.0, value=5.0, step=1.0, label="segment (s)")
    labels_input = mo.ui.text(value="gunshot, glass breaking, scream, speech",
                              label="CLAP candidate labels (comma-separated)", full_width=True)
    device = mo.ui.dropdown(["cuda", "cpu"], value="cuda", label="device")
    run_btn = mo.ui.run_button(label="Classify audio")
    mo.vstack([mo.md(f"**{len(_clips)}** clips found."),
               clip_pick, model, overlap, segment, labels_input, device, run_btn])
    return clip_pick, device, labels_input, model, overlap, run_btn, segment


@app.cell
def _(
    analyze_audio,
    clip_pick,
    device,
    labels_input,
    mo,
    model,
    overlap,
    run_btn,
    segment,
):
    mo.stop(
        not run_btn.value or clip_pick.value is None,
        mo.md("▶️ Pick a clip and click **Classify audio** (first run downloads the model)."),
    )
    _labels = [s.strip() for s in labels_input.value.split(",") if s.strip()] or None
    result = analyze_audio(
        clip_pick.value,
        model=model.value,
        overlap_seconds=overlap.value,
        segment_seconds=segment.value,
        candidate_labels=_labels,
        device=device.value,
    )
    return (result,)


@app.cell(hide_code=True)
def _(mo, result):
    mo.md(f"### {result.summary()}")
    return


@app.cell
def _(result):
    metrics_df = result.metrics()
    return (metrics_df,)


@app.cell(hide_code=True)
def _(alt, metrics_df, mo, pl):
    mo.stop(not metrics_df.height, mo.md("_No segments classified — try a longer clip._"))
    top = metrics_df.filter(pl.col("rank") == 0)
    chart = (
        alt.Chart(top)
        .mark_bar(cornerRadius=3, height=20)
        .encode(
            x=alt.X("start_time:Q", title="time (s)"),
            x2="end_time:Q",
            y=alt.Y("label:N", title="top label", sort="-x"),
            color=alt.Color("confidence:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1])),
            tooltip=["segment_index", "start_time", "end_time", "label", "confidence"],
        )
        .properties(width=640, height=280, title="Top sound label per segment over time")
    )
    mo.ui.altair_chart(chart)
    return


@app.cell(hide_code=True)
def _(metrics_df, mo):
    mo.stop(not metrics_df.height, mo.md(""))
    mo.vstack([mo.md("### Per-segment predictions"), mo.ui.table(metrics_df)])
    return


if __name__ == "__main__":
    app.run()
