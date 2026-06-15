"""MCP server tests — serialization helpers + the tool functions, no GPU/weights.

The tools are plain functions (``@mcp.tool()`` registers and returns them unchanged), so we call
them directly. Facade calls are monkeypatched to canned results; ``list_clips``/``peek_folder``
globbing runs against real (empty) files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")  # the MCP server stack is an optional extra

from argus import PeekResult, TrackingResult
from argus.core import Track
from argus.mcp import _serialize, server


def _peek(path, *, interesting: bool = True) -> PeekResult:
    return PeekResult(
        video_path=Path(path), fps=10.0, width=320, height=240,
        total_frames=100, n_sampled=24, frames_with_hits=5 if interesting else 0,
        counts={"person": 5}, min_hits=2,
    )


def _tracking(path, frames) -> TrackingResult:
    return TrackingResult(video_path=Path(path), fps=10.0, width=320, height=240, frames=frames)


def _make_clips(dirpath, names) -> list[Path]:
    paths = []
    for n in names:
        p = Path(dirpath) / n
        p.write_bytes(b"\x00")
        paths.append(p)
    return paths


# ---- serialization ---------------------------------------------------------------------

def test_peek_to_dict_is_jsonable():
    d = _serialize.peek_to_dict(_peek("a.mp4"))
    json.dumps(d)  # raises if not JSON-able
    assert d["video_path"] == "a.mp4"
    assert d["interesting"] is True
    assert d["counts"] == {"person": 5}
    assert isinstance(d["summary"], str)


def test_tracking_to_dict_empty():
    d = _serialize.tracking_to_dict(_tracking("x.mp4", []), None)
    json.dumps(d)
    assert d["n_frames"] == 0
    assert d["n_tracks"] == 0
    assert d["tracks"] == []
    assert d["rendered"] is None


def test_tracking_to_dict_populated():
    frames = [
        (i, [Track(100 + i, 200, 160 + i, 360, 0.9, 1,
                   class_id=0, label="person", category="person")])
        for i in range(5)
    ]
    d = _serialize.tracking_to_dict(_tracking("x.mp4", frames), "out/x_tracked.mp4")
    json.dumps(d)
    assert d["n_tracks"] == 1
    assert len(d["tracks"]) == 1
    row = d["tracks"][0]
    assert row["id"] == 1 and row["category"] == "person"
    assert d["rendered"] == "out/x_tracked.mp4"


# ---- tools -----------------------------------------------------------------------------

def test_list_clips_filters_by_glob(tmp_path):
    _make_clips(tmp_path, ["a.mp4", "b.mp4", "notes.txt"])
    out = server.list_clips(str(tmp_path))
    json.dumps(out)
    assert out["n_clips"] == 2
    assert {Path(c["path"]).name for c in out["clips"]} == {"a.mp4", "b.mp4"}
    assert all("size_bytes" in c for c in out["clips"])


def test_peek_folder_counts_interesting_and_unreadable(tmp_path, monkeypatch):
    p1, p2 = _make_clips(tmp_path, ["a.mp4", "b.mp4"])
    monkeypatch.setattr(
        server, "peek_videos",
        lambda clips, **kw: {p1: _peek(p1, interesting=True), p2: None},
    )
    out = server.peek_folder(str(tmp_path))
    json.dumps(out)
    assert out["n_clips"] == 2
    assert out["n_interesting"] == 1
    assert out["n_unreadable"] == 1
    assert out["unreadable"] == [str(p2)]


def test_peek_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "peek_video", lambda path, **kw: _peek(path))
    out = server.peek_clip(str(tmp_path / "a.mp4"))
    json.dumps(out)
    assert out["interesting"] is True


def test_track_clip_no_render(tmp_path, monkeypatch):
    frames = [(0, [Track(100, 200, 160, 360, 0.9, 1,
                         class_id=0, label="person", category="person")])]
    monkeypatch.setattr(server, "track_video", lambda path, **kw: _tracking(path, frames))
    out = server.track_clip(str(tmp_path / "clip.mp4"), render=False)
    json.dumps(out)
    assert out["n_tracks"] == 1
    assert len(out["tracks"]) == 1
    assert out["rendered"] is None
