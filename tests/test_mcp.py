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

from argus import (
    AudioAnalysis,
    ClusterResult,
    IngestResult,
    PeekResult,
    QualityGate,
    TrackingResult,
)
from argus.core import AudioPrediction, AudioSegment, Identity, SearchHit, Sighting, Track
from argus.mcp import _serialize, auth, server


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


# ---- face-ID store-backed tools --------------------------------------------------------


class _FakeStore:
    """Stub store for the face-ID MCP tools: records close(), serves canned list rows."""

    def __init__(self, *, sightings=None, identities=None):
        self.closed = False
        self._sightings = sightings or []
        self._identities = identities or []

    def list_sightings(self):
        return self._sightings

    def list_identities(self, *, type=None):
        return [i for i in self._identities if type is None or i.type == type]

    def close(self):
        self.closed = True


def _sighting_row(sid, *, camera="cam-1", quality=0.9):
    return {
        "id": sid,
        "camera_id": camera,
        "track_id": sid,
        "ts": float(sid),
        "quality": quality,
        "chip_path": f"/chips/c{sid}.png",
        "identity_id": None,
    }


def test_ingest_clip_forwards_and_closes_store(tmp_path, monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(server, "_open_store", lambda: store)
    captured = {}

    def fake_ingest(path, camera_id, **kw):
        captured.update(path=path, camera_id=camera_id, kw=kw)
        return IngestResult(
            video_id=1,
            video_path=Path(path),
            n_frames=10,
            n_tracks=2,
            n_faces_detected=5,
            n_gated_out=1,
            n_sightings=2,
            avg_quality=0.7,
        )

    monkeypatch.setattr(server, "ingest_video", fake_ingest)
    out = server.ingest_clip(str(tmp_path / "clip.mp4"), "cam-A", stride=2, min_face_px=20.0)
    json.dumps(out)
    assert out["n_sightings"] == 2 and out["video_id"] == 1
    assert captured["camera_id"] == "cam-A"
    assert captured["kw"]["stride"] == 2
    assert captured["kw"]["gate"].min_face_px == 20.0  # override applied to the QualityGate
    assert store.closed is True


def test_ingest_clip_gate_defaults_to_calibrated(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_open_store", lambda: _FakeStore())
    captured = {}
    monkeypatch.setattr(
        server,
        "ingest_video",
        lambda path, camera_id, **kw: (
            captured.update(kw=kw) or IngestResult(1, Path(path), 0, 0, 0, 0, 0, 0.0)
        ),
    )
    server.ingest_clip(str(tmp_path / "c.mp4"), "cam-A")
    assert captured["kw"]["gate"].min_face_px == QualityGate().min_face_px  # null -> calibrated


def test_list_sightings_filters_and_closes(monkeypatch):
    store = _FakeStore(
        sightings=[
            _sighting_row(1, camera="cam-A", quality=0.9),
            _sighting_row(2, camera="cam-B", quality=0.9),
            _sighting_row(3, camera="cam-A", quality=0.3),
        ]
    )
    monkeypatch.setattr(server, "_open_store", lambda: store)
    out = server.list_sightings(cameras=["cam-A"], min_quality=0.5)
    json.dumps(out)
    assert out["n"] == 1 and out["sightings"][0]["id"] == 1
    assert store.closed is True


def test_list_sightings_limit(monkeypatch):
    store = _FakeStore(sightings=[_sighting_row(i) for i in range(5)])
    monkeypatch.setattr(server, "_open_store", lambda: store)
    assert server.list_sightings(limit=2)["n"] == 2


def test_list_identities_filters_by_type(monkeypatch):
    store = _FakeStore(
        identities=[
            Identity(type="known", label="A", id=1),
            Identity(type="provisional", id=2),
        ]
    )
    monkeypatch.setattr(server, "_open_store", lambda: store)
    out = server.list_identities(type="known")
    json.dumps(out)
    assert out["n"] == 1 and out["identities"][0]["label"] == "A"
    assert store.closed is True


def test_search_similar_forwards_and_closes(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(server, "_open_store", lambda: store)
    captured = {}

    def fake(sid, **kw):
        captured.update(sid=sid, kw=kw)
        return [_hit(1, 0.9, sighting_id=7)]

    monkeypatch.setattr(server, "search_by_sighting", fake)
    out = server.search_similar(5, top_k=3, cameras=["cam-B"], min_quality=0.4, actor="bob")
    json.dumps(out)
    assert out["query"] == "sighting:5" and out["hits"][0]["sighting_id"] == 7
    assert captured["sid"] == 5
    assert captured["kw"]["top_k"] == 3 and captured["kw"]["cameras"] == ["cam-B"]
    assert captured["kw"]["actor"] == "bob"
    assert store.closed is True


def test_enroll_identity_forwards_and_closes(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(server, "_open_store", lambda: store)
    captured = {}

    def fake(label, images, **kw):
        captured.update(label=label, images=list(images), kw=kw)
        return 42

    monkeypatch.setattr(server, "enroll", fake)
    out = server.enroll_identity("suspect-A", ["a.png", "b.png"], source="cctv", actor="op")
    json.dumps(out)
    assert out["identity_id"] == 42 and out["n_images"] == 2
    assert captured["label"] == "suspect-A"
    assert [str(p) for p in captured["images"]] == ["a.png", "b.png"]
    assert captured["kw"]["source"] == "cctv" and captured["kw"]["actor"] == "op"
    assert store.closed is True


def test_cluster_sightings_inverts_assigned_and_closes(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(server, "_open_store", lambda: store)
    captured = {}

    def fake(**kw):
        captured.update(kw)
        return ClusterResult(
            n_sightings=10, n_clusters=2, n_noise=3, run_id=1, identity_ids=[5, 6]
        )

    monkeypatch.setattr(server, "run_clustering", fake)
    out = server.cluster_sightings(space_id="fake_v1", min_cluster_size=2, include_assigned=True)
    json.dumps(out)
    assert out["n_clusters"] == 2 and out["identity_ids"] == [5, 6]
    assert captured["space_id"] == "fake_v1" and captured["min_cluster_size"] == 2
    assert captured["only_unassigned"] is False  # include_assigned inverts the default
    assert store.closed is True


def test_audit_log_forwards_and_closes(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(server, "_open_store", lambda: store)
    captured = {}
    rows = [{"ts": "t", "actor": "op", "action": "enroll", "details": "x"}]
    monkeypatch.setattr(server, "_audit_log", lambda **kw: captured.update(kw) or rows)
    out = server.audit_log(actor="op", since="2026-01-01")
    json.dumps(out)
    assert out["n"] == 1 and out["rows"] == rows
    assert captured["actor"] == "op" and captured["since"] == "2026-01-01"
    assert store.closed is True


def test_face_id_serializers_are_jsonable():
    json.dumps(_serialize.ingest_to_dict(IngestResult(1, Path("x.mp4"), 10, 2, 5, 1, 2, 0.7)))
    json.dumps(_serialize.cluster_to_dict(ClusterResult(10, 2, 3, 1, [5, 6])))
    d = _serialize.identities_to_dict([Identity(type="known", label="A", id=1)])
    json.dumps(d)
    assert d["identities"][0]["type"] == "known"
    json.dumps(_serialize.sightings_to_dict([_sighting_row(1)]))
    json.dumps(_serialize.audit_to_dict([{"actor": "op", "action": "enroll"}]))


def test_all_tools_have_scopes():
    """Every registered tool has a scope entry (and vice-versa) — catches a forgotten sync."""
    assert {fn.__name__ for fn in server._TOOLS} == set(auth.TOOL_SCOPES.keys())
