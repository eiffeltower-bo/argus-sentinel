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
    # 🔭 Peek select videos — open vocabulary

    Point at a **folder of recordings**, pick which **videos** to scan, type any **object
    description** (e.g. "helmet", "backpack", "uniform"), and fast-scan the selected clips
    using **YOLO-World** — an open-vocabulary detector that needs no retraining for new
    categories.

    Clips are scanned with `peek_videos`, which decodes them in parallel threads and runs
    detection in one batched inference pass.

    ```python
    from argus import OpenVocabularyDetector, peek_videos

    detector = OpenVocabularyDetector(prompt="helmet", device="cuda")
    peek_videos(clips, detector=detector, max_workers=8)
    ```
    """)
    return


@app.cell
def _():
    import time
    from pathlib import Path

    from argus import OpenVocabularyDetector, peek_videos

    return OpenVocabularyDetector, Path, peek_videos, time


@app.cell
def _(mo):
    folder = mo.ui.text(
        value="/home/pepe/dev/dani_y_jose/sentinel/data/proxies",
        label="recordings folder",
        full_width=True,
    )
    folder
    return (folder,)


@app.cell
def _(Path, folder):
    _dir = Path(folder.value).expanduser()
    mp4_files: dict[str, Path] = {}
    if _dir.is_dir():
        for _p in sorted(_dir.glob("*.mp4")):
            mp4_files[_p.name] = _p
    return (mp4_files,)


@app.cell
def _(mo, mp4_files: "dict[str, Path]"):  # noqa: F821 (Path is a marimo-injected cell var)
    file_select = mo.ui.multiselect(
        options={_name: str(_path) for _name, _path in mp4_files.items()},
        value=[],
        label="videos to scan",
        full_width=True,
    )
    prompt = mo.ui.text(value="person", label="prompt", full_width=False)
    workers = mo.ui.slider(1, 16, value=8, step=1, label="peek workers")
    peek_run = mo.ui.run_button(label="Peek selected")
    mo.vstack(
        [
            mo.md(f"**{len(mp4_files)}** mp4 files found in the folder."),
            file_select,
            mo.hstack([prompt, workers], justify="start"),
            peek_run,
        ]
    )
    return file_select, peek_run, prompt, workers


@app.cell(hide_code=True)
def _(file_select, mo):
    mo.stop(
        len(file_select.value) == 0,
        mo.md("_Select one or more videos from the list above._"),
    )
    mo.md(
        f"**{len(file_select.value)}** video(s) selected — click **Peek selected** to scan them."
    )
    return


@app.cell
def _(OpenVocabularyDetector, prompt):
    peek_detector = OpenVocabularyDetector(
        prompt=prompt.value, weights="yolov8s-worldv2.pt", conf=0.35, imgsz=320
    )
    return (peek_detector,)


@app.cell(hide_code=True)
def _(
    Path,
    file_select,
    mo,
    peek_detector,
    peek_run,
    peek_videos,
    prompt,
    time,
    workers,
):
    mo.stop(
        not peek_run.value,
        mo.md("_Pick videos, type a prompt, and click **Peek selected** to scan._"),
    )
    _paths = [Path(p) for p in file_select.value]
    _t0 = time.perf_counter()
    with mo.status.spinner(
        title=f"Peeking {len(_paths)} clips for '{prompt.value}' ({workers.value} workers)..."
    ):
        _peeks = peek_videos(_paths, detector=peek_detector, max_workers=workers.value)
    wall_s = time.perf_counter() - _t0

    results = []
    skipped = []
    for _p, _pk in _peeks.items():
        if _pk is None:
            skipped.append(_p.name)
            continue
        results.append(
            (
                _p.name,
                _pk.interesting,
                dict(_pk.counts),
                _pk.frames_with_hits,
                _pk.n_sampled,
                _pk.elapsed_s,
            )
        )
    return results, skipped, wall_s


@app.cell(hide_code=True)
def _(mo, prompt, results, skipped, wall_s, workers):
    interesting = [r for r in results if r[1]]
    quiet = [r for r in results if not r[1]]

    _per_clip = f" · {wall_s / len(results) * 1000:.0f} ms/clip" if results else ""
    _lines = [
        "## Peek summary",
        f"_looking for **{prompt.value}**_",
        "",
        f"Scanned **{len(results)}** clips · **{len(interesting)} interesting** · "
        f"{len(quiet)} quiet · {len(skipped)} unreadable",
        f"**{wall_s:.1f}s** wall across {workers.value} decode worker(s), batched inference{_per_clip}",
        "",
    ]
    if interesting:
        _lines.append("### ✅ Interesting")
        for _name, _, _counts, _hits, _n, _secs in sorted(interesting, key=lambda r: r[0]):
            _cnt = ", ".join(f"{_k}×{_v}" for _k, _v in _counts.items() if _v)
            _lines.append(
                f"- `{_name}` — {_cnt or 'detections'} — "
                f"hits {_hits}/{_n} — {_secs * 1000:.0f} ms decode"
            )
    if quiet:
        _lines.append("### 😴 Quiet")
        _lines += [f"- `{_name}`" for _name, _, _, _, _, _ in sorted(quiet, key=lambda r: r[0])]
    if skipped:
        _lines += ["", "### ⚠️ Unreadable (skipped)"]
        _lines += [f"- `{_s}`" for _s in skipped]
    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
