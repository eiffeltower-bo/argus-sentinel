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
    # 🔭 Peek a folder by time range

    Point at a **folder of recordings**, pick a **time range**, and fast-scan every clip in
    that window to see which ones contain anything worth tracking — before committing to full
    detection + tracking on any of them.

    The recording time is read from the **file name** (`YYYY-MM-DD_HH-MM-SS.mp4`), so the range
    filter is just string-parsed timestamps — no decoding needed to select clips. The in-range
    clips are scanned with `peek_videos`, which **decodes them in parallel threads** and runs
    detection in one **batched** inference pass; the result is a short **text summary** of the
    interesting clips.

    ```python
    from argus import peek_videos
    peek_videos(clips, targets=("person", "vehicle"), device="cuda", max_workers=8)
    ```
    """)
    return


@app.cell
def _():
    import re
    import time
    from datetime import datetime
    from pathlib import Path

    from argus import UltralyticsDetector, peek_videos

    return Path, UltralyticsDetector, datetime, peek_videos, re, time


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
def _(Path, datetime, folder, re):
    # Parse the recording datetime out of each file name (YYYY-MM-DD_HH-MM-SS).
    _stamp = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})")

    def _parse_dt(name):
        m = _stamp.search(name)
        return datetime(*(int(g) for g in m.groups())) if m else None

    _dir = Path(folder.value).expanduser()
    all_files = []
    if _dir.is_dir():
        for _p in sorted(_dir.glob("*.mp4")):
            _dt = _parse_dt(_p.name)
            if _dt is not None:
                all_files.append((_p, _dt))
    return (all_files,)


@app.cell
def _(all_files, mo):
    _dts = [dt for _, dt in all_files]
    _lo, _hi = (min(_dts), max(_dts)) if _dts else (None, None)
    date_from = mo.ui.datetime(value=_lo, label="from")
    date_to = mo.ui.datetime(value=_hi, label="to")
    target_pick = mo.ui.dropdown(
        {
            "person + vehicle": ("person", "vehicle"),
            "person": ("person",),
            "vehicle": ("vehicle",),
        },
        value="person + vehicle",
        label="looking for",
    )
    workers = mo.ui.slider(1, 16, value=8, step=1, label="peek workers")
    peek_run = mo.ui.run_button(label="Peek range")
    mo.vstack(
        [
            mo.md(f"**{len(all_files)}** timestamped clips found in the folder."),
            mo.hstack([date_from, date_to], justify="start"),
            target_pick,
            workers,
            peek_run,
        ]
    )
    return date_from, date_to, peek_run, target_pick, workers


@app.cell
def _(all_files, date_from, date_to):
    # Reactive: which clips fall in the picked range (cheap — just parsed timestamps).
    if date_from.value and date_to.value:
        in_range = [(p, dt) for (p, dt) in all_files if date_from.value <= dt <= date_to.value]
    else:
        in_range = []
    return (in_range,)


@app.cell(hide_code=True)
def _(in_range, mo):
    mo.md(f"""
    **{len(in_range)}** clips fall in the selected range — click **Peek range** to scan them.
    """)
    return


@app.cell
def _(UltralyticsDetector):
    # One small, low-res detector reused across every peek (person + vehicle classes).
    peek_detector = UltralyticsDetector(
        weights="yolo11n.pt", classes=[0, 2, 3, 5, 7], conf=0.35, device="cuda", imgsz=320
    )
    return (peek_detector,)


@app.cell(hide_code=True)
def _(in_range, mo, peek_detector, peek_run, peek_videos, target_pick, time, workers):
    mo.stop(
        not peek_run.value,
        mo.md("_Pick a range and click **Peek range** to scan (no tracking yet)._"),
    )
    _paths = [p for p, _ in in_range]
    _dt_by_path = {p: dt for p, dt in in_range}
    _t0 = time.perf_counter()
    with mo.status.spinner(title=f"Peeking {len(_paths)} clips ({workers.value} workers)..."):
        _peeks = peek_videos(
            _paths, targets=target_pick.value, detector=peek_detector, max_workers=workers.value
        )
    wall_s = time.perf_counter() - _t0

    results = []  # (name, dt, interesting, counts, frames_with_hits, n_sampled, elapsed_s)
    skipped = []
    for _p, _pk in _peeks.items():
        if _pk is None:
            skipped.append(_p.name)  # unreadable / corrupt (e.g. 0-byte file)
            continue
        results.append(
            (
                _p.name,
                _dt_by_path[_p],
                _pk.interesting,
                dict(_pk.counts),
                _pk.frames_with_hits,
                _pk.n_sampled,
                _pk.elapsed_s,
            )
        )
    return results, skipped, wall_s


@app.cell(hide_code=True)
def _(date_from, date_to, mo, results, skipped, target_pick, wall_s, workers):
    interesting = [r for r in results if r[2]]
    quiet = [r for r in results if not r[2]]

    _rng = (
        f"{date_from.value:%Y-%m-%d %H:%M} → {date_to.value:%Y-%m-%d %H:%M}"
        if date_from.value and date_to.value
        else "selected range"
    )
    _per_clip = f" · {wall_s / len(results) * 1000:.0f} ms/clip" if results else ""
    _lines = [
        f"## Peek summary — {_rng}",
        f"_looking for **{' + '.join(target_pick.value)}**_",
        "",
        f"Scanned **{len(results)}** clips · **{len(interesting)} interesting** · "
        f"{len(quiet)} quiet · {len(skipped)} unreadable",
        f"**{wall_s:.1f}s** wall across {workers.value} decode worker(s), batched inference{_per_clip}",
        "",
    ]
    if interesting:
        _lines.append("### ✅ Interesting")
        for _name, _dt, _, _counts, _hits, _n, _secs in sorted(interesting, key=lambda r: r[1]):
            _cnt = ", ".join(f"{_k}×{_v}" for _k, _v in _counts.items() if _v)
            _lines.append(
                f"- **{_dt:%H:%M:%S}**  `{_name}` — {_cnt or 'detections'} — "
                f"hits {_hits}/{_n} — {_secs * 1000:.0f} ms decode"
            )
    if skipped:
        _lines += ["", "### ⚠️ Unreadable (skipped)"]
        _lines += [f"- `{_s}`" for _s in skipped]
    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
