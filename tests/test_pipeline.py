"""Unit tests for the pipeline: edge_of, colors, metrics, exports, VideoTracker, render."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from argus.core import Track
from argus.track import ByteTrackTracker
from argus.pipeline import (
    TrackingResult,
    VideoTracker,
    edge_of,
    track_color,
    track_video,
)
from conftest import drifting_person  # noqa: E402  (conftest on pythonpath)


def test_edge_of():
    w = h = 100  # margin = 4 px
    assert edge_of((0, 40, 20, 60), w, h) == "left"
    assert edge_of((90, 40, 100, 60), w, h) == "right"
    assert edge_of((40, 0, 60, 2), w, h) == "top"
    assert edge_of((40, 98, 60, 100), w, h) == "bottom"
    assert edge_of((40, 40, 60, 60), w, h) == "interior"


def test_track_color_stable_and_cyclic():
    assert track_color(3) == track_color(3)
    assert track_color(3) == track_color(3 + 12)  # palette length 12


def _result_with_one_track(n=5):
    frames = []
    for fi in range(n):
        # box touches the left edge on the first frame, drifts inward after
        x1 = 2 + fi * 5
        frames.append((fi, [Track(x1, 20, x1 + 60, 180, 0.8,
                                   track_id=1, class_id=0, label="person", category="person")]))
    return TrackingResult(video_path=Path("x.mp4"), fps=10.0, width=200, height=200, frames=frames)


def test_metrics_values():
    m = _result_with_one_track(5)
    df = m.metrics()
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["id"] == 1
    assert row["type"] == "person"
    assert row["category"] == "person"
    assert row["n_frames"] == 5
    assert row["continuity"] == 1.0
    assert row["duration_s"] == pytest.approx(0.5)  # (4-0+1)/10
    assert row["first_s"] == pytest.approx(0.0)
    assert row["entry_edge"] == "left"


def test_tracks_dataframe():
    df = _result_with_one_track(3).tracks_dataframe()
    assert df.height == 3
    assert set(["frame", "id", "category", "label", "area_px"]).issubset(df.columns)


def test_empty_result_has_typed_schema():
    empty = TrackingResult(Path("x.mp4"), 10.0, 100, 100, frames=[(0, []), (1, [])])
    m = empty.metrics()
    assert m.height == 0
    assert m.select("id", "type").shape == (0, 2)  # selecting columns must not raise
    t = empty.tracks_dataframe()
    assert t.height == 0
    assert "id" in t.columns


def test_export_roundtrip(tmp_path):
    res = _result_with_one_track(4)
    pq = res.to_parquet(tmp_path / "m.parquet", what="metrics")
    csvp = res.to_csv(tmp_path / "tracks.csv", what="tracks")
    assert pl.read_parquet(pq).height == 1
    assert pl.read_csv(csvp).height == 4


def test_export_rejects_bad_kind(tmp_path):
    with pytest.raises(ValueError):
        _result_with_one_track().to_parquet(tmp_path / "x.parquet", what="bogus")


def test_video_tracker_run(make_video, scripted_detector):
    n = 8
    video = make_video(n_frames=n, fps=10.0)
    detector = scripted_detector(drifting_person(n))
    result = VideoTracker(detector, ByteTrackTracker()).run(video)
    assert len(result.frames) == n
    assert len(result.track_ids) == 1
    assert result.metrics().height == 1


def test_video_tracker_no_detections(make_video, scripted_detector):
    n = 5
    video = make_video(n_frames=n)
    result = VideoTracker(scripted_detector([[] for _ in range(n)]), ByteTrackTracker()).run(video)
    assert len(result.frames) == n          # every frame still recorded
    assert result.metrics().height == 0     # but no tracks


def test_track_video_bad_target(make_video):
    with pytest.raises(ValueError):
        track_video(make_video(n_frames=2), targets=("alien",))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_produces_file(make_video, scripted_detector, tmp_path):
    n = 6
    video = make_video(n_frames=n, w=320, h=240, fps=10.0)
    result = VideoTracker(scripted_detector(drifting_person(n)), ByteTrackTracker()).run(video)
    out = result.render(tmp_path / "annotated.mp4", display_height=120)
    assert out.exists() and out.stat().st_size > 0


def test_render_empty_raises():
    empty = TrackingResult(Path("x.mp4"), 10.0, 100, 100, frames=[])
    with pytest.raises(RuntimeError):
        empty.render("out.mp4")
