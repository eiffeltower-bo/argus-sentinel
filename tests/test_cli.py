"""CLI tests — argument dispatch, store wiring, and JSON output, no GPU/weights/sqlite-vec.

The CLI is a pure dispatch layer over the facade, so (like the MCP tests) we drive ``cli.main``
with canned argv and monkeypatch the facade functions + ``_open_store`` to fakes. We assert that
flags reach the facade unchanged, the right verb is dispatched, the store is always closed, and
``--json`` emits a machine-readable object.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from argus import ClusterResult, IngestResult, cli
from argus.core import Identity, SearchHit, Sighting


class _FakeStore:
    """Minimal stand-in for SqliteStore — records close(), serves canned list rows."""

    def __init__(self) -> None:
        self.closed = False
        self.videos: list[dict] = []
        self.sightings: list[dict] = []
        self.identities: list[Identity] = []

    def count_vectors(self) -> int:
        return len(self.sightings)

    def list_videos(self) -> list[dict]:
        return self.videos

    def list_sightings(self) -> list[dict]:
        return self.sightings

    def list_identities(self):
        return self.identities

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def store(monkeypatch):
    """Install a FakeStore as the CLI's store and hand it back for assertions."""
    s = _FakeStore()
    monkeypatch.setattr(cli, "_open_store", lambda args: s)
    return s


def _sighting_row(sid: int, *, camera="cam-1", quality=0.9, ts=1.0, identity_id=None) -> dict:
    return {
        "id": sid,
        "camera_id": camera,
        "track_id": sid,
        "ts": ts,
        "quality": quality,
        "chip_path": f"/chips/c{sid}.png",
        "identity_id": identity_id,
    }


def _hit(sid: int, score: float, *, camera="cam-1") -> SearchHit:
    s = Sighting(
        video_id=1,
        camera_id=camera,
        track_id=sid,
        frame_idx=sid,
        ts=float(sid),
        bbox=(0.0, 0.0, 1.0, 1.0),
        quality=0.9,
        chip_path=f"/chips/c{sid}.png",
        embedding_space_id="arcface_w600k_r50_v1",
        embedding=np.zeros(4, np.float32),
        id=sid,
    )
    return SearchHit(sighting=s, distance=1.0 - score, score=score)


# ---- ingest ----------------------------------------------------------------------------


def test_ingest_forwards_args_and_emits_json(store, monkeypatch, capsys):
    captured = {}

    def fake_ingest(video, camera, **kw):
        captured.update(video=video, camera=camera, kw=kw)
        return IngestResult(
            video_id=7,
            video_path=video,
            n_frames=100,
            n_tracks=3,
            n_faces_detected=5,
            n_gated_out=1,
            n_sightings=3,
            avg_quality=0.8,
        )

    monkeypatch.setattr(cli, "ingest_video", fake_ingest)
    rc = cli.main(
        [
            "ingest",
            "clip.mp4",
            "--camera",
            "cam-A",
            "--device",
            "cuda",
            "--stride",
            "2",
            "--max-frames",
            "50",
            "--db",
            "x.db",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["video_id"] == 7 and out["camera_id"] == "cam-A" and out["n_sightings"] == 3
    assert captured["camera"] == "cam-A"
    assert captured["kw"]["device"] == "cuda"
    assert captured["kw"]["stride"] == 2
    assert captured["kw"]["max_frames"] == 50
    assert store.closed is True


def test_ingest_camera_defaults_to_stem(store, monkeypatch, capsys):
    captured = {}
    monkeypatch.setattr(
        cli,
        "ingest_video",
        lambda video, camera, **kw: (
            captured.update(camera=camera) or IngestResult(1, video, 0, 0, 0, 0, 0, 0.0)
        ),
    )
    cli.main(["ingest", "/data/2026-03-15_12-29-38.mp4", "--json"])
    assert captured["camera"] == "2026-03-15_12-29-38"


# ---- ls --------------------------------------------------------------------------------


def test_ls_sightings_filters_by_camera_and_quality(store, capsys):
    store.sightings = [
        _sighting_row(1, camera="cam-A", quality=0.9),
        _sighting_row(2, camera="cam-B", quality=0.9),
        _sighting_row(3, camera="cam-A", quality=0.3),
    ]
    cli.main(["ls", "sightings", "--camera", "cam-A", "--min-quality", "0.5", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == [1]  # cam-B and the low-quality cam-A row filtered out
    assert store.closed is True


def test_ls_defaults_to_sightings(store, capsys):
    store.sightings = [_sighting_row(1)]
    cli.main(["ls", "--json"])
    assert json.loads(capsys.readouterr().out)[0]["id"] == 1


def test_ls_identities_serializes_dataclass(store, capsys):
    store.identities = [Identity(type="known", label="J. Doe", created_by="op", id=1)]
    cli.main(["ls", "identities", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["label"] == "J. Doe" and rows[0]["type"] == "known"


def test_ls_videos(store, capsys):
    store.videos = [{"id": 1, "camera_id": "cam-A", "path": "/data/a.mp4"}]
    cli.main(["ls", "videos", "--json"])
    assert json.loads(capsys.readouterr().out)[0]["path"] == "/data/a.mp4"


# ---- search ----------------------------------------------------------------------------


def test_search_by_image_forwards_filters(store, monkeypatch, capsys):
    captured = {}

    def fake_search(image, **kw):
        captured.update(image=image, kw=kw)
        return [_hit(1, 0.95), _hit(2, 0.40)]

    monkeypatch.setattr(cli, "search_by_image", fake_search)
    cli.main(
        [
            "search",
            "--image",
            "probe.jpg",
            "--top-k",
            "5",
            "--cameras",
            "cam-A",
            "cam-B",
            "--since",
            "12.0",
            "--min-quality",
            "0.4",
            "--device",
            "cuda",
            "--actor",
            "alice",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["n_hits"] == 2 and out["hits"][0]["sighting_id"] == 1
    assert out["hits"][0]["chip_path"] == "/chips/c1.png"  # evidence surfaced
    assert str(captured["image"]) == "probe.jpg"
    assert captured["kw"]["top_k"] == 5
    assert captured["kw"]["cameras"] == ["cam-A", "cam-B"]
    assert captured["kw"]["since"] == 12.0
    assert captured["kw"]["min_quality"] == 0.4
    assert captured["kw"]["device"] == "cuda"
    assert captured["kw"]["actor"] == "alice"
    assert store.closed is True


def test_search_by_sighting_dispatches_without_image(store, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        cli,
        "search_by_sighting",
        lambda sid, **kw: seen.update(sid=sid, kw=kw) or [_hit(2, 0.8)],
    )
    monkeypatch.setattr(
        cli,
        "search_by_image",
        lambda *a, **k: pytest.fail("search_by_image must not be called for --sighting"),
    )
    cli.main(["search", "--sighting", "3", "--top-k", "4", "--json"])
    assert seen["sid"] == 3 and seen["kw"]["top_k"] == 4
    assert "device" not in seen["kw"]  # by-sighting search uses the stored embedding, no device


def test_search_requires_a_probe(store):
    with pytest.raises(SystemExit):  # mutually-exclusive group is required
        cli.main(["search", "--json"])


def test_search_image_and_sighting_are_exclusive(store):
    with pytest.raises(SystemExit):
        cli.main(["search", "--image", "p.jpg", "--sighting", "1"])


# ---- enroll ----------------------------------------------------------------------------


def test_enroll_forwards_label_images_source(store, monkeypatch, capsys):
    captured = {}

    def fake_enroll(label, images, **kw):
        captured.update(label=label, images=list(images), kw=kw)
        return 42

    monkeypatch.setattr(cli, "enroll", fake_enroll)
    cli.main(
        ["enroll", "suspect-A", "a.png", "b.png", "--source", "cctv", "--actor", "op", "--json"]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["identity_id"] == 42 and out["n_images"] == 2
    assert captured["label"] == "suspect-A"
    assert [str(p) for p in captured["images"]] == ["a.png", "b.png"]
    assert captured["kw"]["source"] == "cctv" and captured["kw"]["actor"] == "op"
    assert store.closed is True


# ---- cluster ---------------------------------------------------------------------------


def test_cluster_forwards_params_and_inverts_assigned_flag(store, monkeypatch, capsys):
    captured = {}

    def fake_cluster(**kw):
        captured.update(kw)
        return ClusterResult(
            n_sightings=10, n_clusters=2, n_noise=3, run_id=1, identity_ids=[5, 6]
        )

    monkeypatch.setattr(cli, "run_clustering", fake_cluster)
    cli.main(
        [
            "cluster",
            "--space-id",
            "fake_v1",
            "--min-cluster-size",
            "2",
            "--include-assigned",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["n_clusters"] == 2 and out["identity_ids"] == [5, 6]
    assert captured["space_id"] == "fake_v1"
    assert captured["min_cluster_size"] == 2
    assert captured["only_unassigned"] is False  # --include-assigned inverts the default
    assert store.closed is True


def test_cluster_default_space_id_and_only_unassigned(store, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli,
        "run_clustering",
        lambda **kw: captured.update(kw) or ClusterResult(0, 0, 0, None, []),
    )
    cli.main(["cluster"])
    assert captured["space_id"] == cli._DEFAULT_SPACE_ID
    assert captured["only_unassigned"] is True


# ---- audit -----------------------------------------------------------------------------


def test_audit_forwards_filters(store, monkeypatch, capsys):
    captured = {}
    rows = [{"ts": "t", "actor": "op", "action": "enroll", "details": "label='x'"}]
    monkeypatch.setattr(cli, "audit_log", lambda **kw: captured.update(kw) or rows)
    cli.main(["audit", "--actor", "op", "--since", "2026-01-01", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out == rows
    assert captured["actor"] == "op" and captured["since"] == "2026-01-01"
    assert store.closed is True
