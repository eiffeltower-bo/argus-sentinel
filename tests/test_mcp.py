"""MCP server tests — serialization helpers + the tool functions, no GPU/weights.

The tools are plain functions (``@mcp.tool()`` registers and returns them unchanged), so we call
them directly. Facade calls are monkeypatched to canned results; ``list_clips``/``peek_folder``
globbing runs against real (empty) files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")  # the MCP server stack is an optional extra

from argus import AudioAnalysis, PeekResult, TrackingResult
from argus.core import AudioPrediction, AudioSegment, SearchHit, Sighting, Track
from argus.mcp import _serialize, server


def _hit(track_id: int, score: float, *, sighting_id: int) -> SearchHit:
    s = Sighting(
        video_id=1,
        camera_id="cam-1",
        track_id=track_id,
        frame_idx=track_id,
        ts=float(track_id),
        bbox=(0.0, 0.0, 1.0, 1.0),
        quality=0.9,
        chip_path=f"/chips/c{track_id}.png",
        embedding_space_id="arcface_w600k_r50_v1",
        embedding=np.zeros(4, np.float32),
        id=sighting_id,
    )
    return SearchHit(sighting=s, distance=1.0 - score, score=score)


def _peek(path, *, interesting: bool = True) -> PeekResult:
    return PeekResult(
        video_path=Path(path),
        fps=10.0,
        width=320,
        height=240,
        total_frames=100,
        n_sampled=24,
        frames_with_hits=5 if interesting else 0,
        counts={"person": 5},
        min_hits=2,
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
        (
            i,
            [
                Track(
                    100 + i,
                    200,
                    160 + i,
                    360,
                    0.9,
                    1,
                    class_id=0,
                    label="person",
                    category="person",
                )
            ],
        )
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
        server,
        "peek_videos",
        lambda clips, **kw: {p1: _peek(p1, interesting=True), p2: None},
    )
    out = server.peek_folder(str(tmp_path))
    json.dumps(out)
    assert out["n_matched"] == 2 and out["n_peeked"] == 2 and out["truncated"] is False
    assert out["n_interesting"] == 1
    assert out["n_unreadable"] == 1
    assert out["unreadable"] == [str(p2)]


def test_peek_folder_limit_caps_clips_peeked(tmp_path, monkeypatch):
    _make_clips(tmp_path, [f"c{i}.mp4" for i in range(5)])
    seen = {}

    def fake_peek_videos(clips, **kw):
        seen["n"] = len(clips)
        return {p: _peek(p, interesting=False) for p in clips}

    monkeypatch.setattr(server, "peek_videos", fake_peek_videos)
    out = server.peek_folder(str(tmp_path), limit=2)
    assert seen["n"] == 2  # only 2 of 5 actually peeked
    assert out["n_matched"] == 5 and out["n_peeked"] == 2 and out["truncated"] is True


def test_peek_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "peek_video", lambda path, **kw: _peek(path))
    out = server.peek_clip(str(tmp_path / "a.mp4"))
    json.dumps(out)
    assert out["interesting"] is True


def test_track_clip_no_render(tmp_path, monkeypatch):
    frames = [
        (0, [Track(100, 200, 160, 360, 0.9, 1, class_id=0, label="person", category="person")])
    ]
    monkeypatch.setattr(server, "track_video", lambda path, **kw: _tracking(path, frames))
    out = server.track_clip(str(tmp_path / "clip.mp4"), render=False)
    json.dumps(out)
    assert out["n_tracks"] == 1
    assert len(out["tracks"]) == 1
    assert out["rendered"] is None


# ---- target routing: fixed YOLO vs open-vocab -----------------------------------------


class _FakeOpenVocab:
    """Stand-in for OpenVocabularyDetector — records prompts/device, never touches ultralytics."""

    def __init__(self, prompt, device=None):
        self.prompt = prompt
        self.device = device


def test_detection_kwargs_defaults_to_fixed_model():
    # None, empty, the default pair, and a case-insensitive subset all stay on the fixed model.
    for targets in (None, [], ["person", "vehicle"], ["Person"], ["vehicle"]):
        kw = server._detection_kwargs(targets, device="cuda")
        assert "detector" not in kw
        assert set(kw["targets"]) <= {"person", "vehicle"}
        assert kw["device"] == "cuda"
    assert server._detection_kwargs(None, None)["targets"] == ("person", "vehicle")


def test_detection_kwargs_custom_targets_use_open_vocab(monkeypatch):
    monkeypatch.setattr(server, "OpenVocabularyDetector", _FakeOpenVocab)
    kw = server._detection_kwargs(["forklift", "hard hat"], device="cuda")
    assert "targets" not in kw and "device" not in kw  # device went to the detector
    det = kw["detector"]
    assert isinstance(det, _FakeOpenVocab)
    assert det.prompt == ["forklift", "hard hat"] and det.device == "cuda"


def test_detection_kwargs_mixed_targets_use_open_vocab(monkeypatch):
    # A known group mixed with an unknown class still needs open-vocab (forwards both prompts).
    monkeypatch.setattr(server, "OpenVocabularyDetector", _FakeOpenVocab)
    kw = server._detection_kwargs(["person", "backpack"], device=None)
    assert kw["detector"].prompt == ["person", "backpack"]


def test_peek_clip_custom_targets_route_detector_to_facade(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "OpenVocabularyDetector", _FakeOpenVocab)
    captured = {}

    def fake_peek(path, **kw):
        captured.update(kw)
        return _peek(path)

    monkeypatch.setattr(server, "peek_video", fake_peek)
    server.peek_clip(str(tmp_path / "a.mp4"), targets=["forklift"])
    assert "targets" not in captured
    assert isinstance(captured["detector"], _FakeOpenVocab)
    assert captured["detector"].prompt == ["forklift"]


def test_track_clip_default_targets_use_fixed_model(tmp_path, monkeypatch):
    captured = {}

    def fake_track(path, **kw):
        captured.update(kw)
        return _tracking(path, [])

    monkeypatch.setattr(server, "track_video", fake_track)
    server.track_clip(str(tmp_path / "clip.mp4"))
    assert captured["targets"] == ("person", "vehicle")
    assert "detector" not in captured


# ---- LAN exposure: DNS-rebinding allow-list ------------------------------------------


def test_transport_security_none_when_local_only():
    # No LAN flags -> leave FastMCP's localhost default untouched.
    assert server._transport_security([], [], insecure=False) is None


def test_transport_security_insecure_disables_protection():
    ts = server._transport_security(["192.168.1.14"], [], insecure=True)
    assert ts.enable_dns_rebinding_protection is False


def test_transport_security_bare_host_gets_wildcard_port_and_keeps_localhost():
    ts = server._transport_security(["192.168.1.14"], [], insecure=False)
    assert ts.enable_dns_rebinding_protection is True
    assert "192.168.1.14:*" in ts.allowed_hosts  # any port on the LAN ip
    assert "http://192.168.1.14:*" in ts.allowed_origins
    assert "127.0.0.1:*" in ts.allowed_hosts  # localhost still works


def test_transport_security_host_with_port_taken_verbatim():
    ts = server._transport_security(["192.168.1.14:8765"], [], insecure=False)
    assert "192.168.1.14:8765" in ts.allowed_hosts
    assert "http://192.168.1.14:8765" in ts.allowed_origins
    assert "192.168.1.14:8765:*" not in ts.allowed_hosts  # no bogus double-port companion


def test_csv_parsing():
    assert server._csv(" a , b ,,c ") == ["a", "b", "c"]
    assert server._csv("") == [] and server._csv(None) == []


def test_search_to_dict_is_jsonable():
    d = _serialize.search_to_dict(
        "/probe.jpg", [_hit(1, 0.99, sighting_id=10), _hit(2, 0.50, sighting_id=11)]
    )
    json.dumps(d)
    assert d["query"] == "/probe.jpg"
    assert d["n_hits"] == 2
    top = d["hits"][0]
    assert top["sighting_id"] == 10 and top["score"] == 0.99
    assert top["chip_path"] == "/chips/c1.png"  # evidence surfaced
    assert top["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_search_face_ranks_forwards_filters_and_closes_store(tmp_path, monkeypatch):
    closed = []
    monkeypatch.setattr(
        server, "_open_store", lambda: type("S", (), {"close": lambda self: closed.append(True)})()
    )
    captured = {}

    def fake_search(image, **kw):
        captured.update(image=image, kw=kw)
        return [_hit(1, 0.95, sighting_id=7)]

    monkeypatch.setattr(server, "search_by_image", fake_search)
    out = server.search_face(
        str(tmp_path / "probe.jpg"),
        top_k=5,
        cameras=["cam-1"],
        since=12.0,
        min_quality=0.4,
        actor="alice",
    )
    json.dumps(out)
    assert out["n_hits"] == 1
    assert out["hits"][0]["sighting_id"] == 7
    assert out["hits"][0]["chip_path"] == "/chips/c1.png"
    # filters/actor forwarded to the search facade, and the store is always closed
    assert captured["kw"]["top_k"] == 5
    assert captured["kw"]["cameras"] == ["cam-1"]
    assert captured["kw"]["since"] == 12.0
    assert captured["kw"]["min_quality"] == 0.4
    assert captured["kw"]["actor"] == "alice"
    assert closed == [True]


def _audio(path) -> AudioAnalysis:
    seg = AudioSegment(0, 0.0, 5.0, (AudioPrediction("siren", 0.8), AudioPrediction("None", 0.0)))
    return AudioAnalysis(
        input_file=Path(path),
        audio_path=Path(path).with_suffix(".wav"),
        input_duration_seconds=5.0,
        model_name="ast-esc50",
        overlap_seconds=1.0,
        segment_seconds=5.0,
        segments=[seg],
    )


def test_audio_to_dict_is_jsonable():
    d = _serialize.audio_to_dict(_audio("clip.mp4"))
    json.dumps(d)
    assert d["model_name"] == "ast-esc50"
    assert d["segments"][0]["predictions"][0] == {"class": "siren", "confidence": 0.8}


def test_classify_audio_forwards_and_serializes(tmp_path, monkeypatch):
    captured = {}

    def fake_analyze(path, **kw):
        captured.update(path=path, kw=kw)
        return _audio(path)

    monkeypatch.setattr(server, "analyze_audio", fake_analyze)
    out = server.classify_audio(
        str(tmp_path / "clip.mp4"),
        model="laion/clap-htsat-unfused",
        overlap_seconds=2.0,
        candidate_labels=["siren", "speech"],
    )
    json.dumps(out)
    assert out["segments"][0]["predictions"][0]["class"] == "siren"
    assert captured["kw"]["model"] == "laion/clap-htsat-unfused"
    assert captured["kw"]["overlap_seconds"] == 2.0
    assert captured["kw"]["candidate_labels"] == ["siren", "speech"]
