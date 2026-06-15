"""Unit tests for the face-ID ingest pipeline (all backends faked, no GPU/weights)."""

from pathlib import Path

from argus.face import QualityGate
from argus.pipeline import ingest_video
from conftest import (  # noqa: E402  (conftest on pythonpath)
    FakeEmbedder,
    FakeStore,
    FakeTracker,
    NoFaceDetector,
    ScriptedDetector,
    ScriptedFaceDetector,
    drifting_person,
)

# Permissive gate: synthetic solid-color chips have zero blur variance, so the real
# defaults would reject everything. These tests exercise pipeline plumbing, not gating.
_PERMISSIVE = QualityGate(min_blur_var=0.0, min_face_px=0.0, max_yaw_ratio=1.0, min_det_score=0.0)


def _ingest(tmp_path, n=10, *, face_detector=None, gate=_PERMISSIVE):
    store = FakeStore(tmp_path)
    res = ingest_video(
        # ScriptedDetector drives detection; the clip just needs to decode to n frames.
        _make_clip(tmp_path, n),
        "cam-1",
        store=store,
        detector=ScriptedDetector(drifting_person(n)),
        tracker=FakeTracker(),
        # Face scores ramp up so the last frame is unambiguously the best.
        face_detector=face_detector or ScriptedFaceDetector([0.1 * (i + 1) for i in range(n)]),
        embedder=FakeEmbedder(),
        gate=gate,
    )
    return res, store


def _make_clip(tmp_path, n):
    import cv2
    import numpy as np

    path = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for i in range(n):
        vw.write(np.full((240, 320, 3), (i * 7) % 255, np.uint8))
    vw.release()
    return path


def test_best_face_per_track(tmp_path):
    res, store = _ingest(tmp_path, n=10)
    assert res.n_tracks == 1
    assert res.n_sightings == 1
    assert res.n_faces_detected == 10
    assert res.n_gated_out == 0
    # Brightness (=> face score) increases each frame, so the last frame is the best.
    sighting = store.sightings[0]
    assert sighting.frame_idx == 9
    assert sighting.embedding.shape == (8,)
    assert sighting.embedding_space_id == "fake_v1"
    assert 0.0 < sighting.quality <= 1.0
    assert Path(sighting.chip_path).exists()


def test_gated_out_faces_excluded(tmp_path):
    gate = QualityGate(min_blur_var=0.0, min_face_px=0.0, max_yaw_ratio=1.0, min_det_score=0.1)
    faces = ScriptedFaceDetector([0.05, 0.05, 0.05, 0.2, 0.3, 0.4])  # first 3 below 0.1
    res, store = _ingest(tmp_path, n=6, gate=gate, face_detector=faces)
    assert res.n_gated_out == 3
    assert res.n_sightings == 1
    assert store.sightings[0].frame_idx == 5  # best surviving frame is the last


def test_no_faces_yields_no_sightings(tmp_path):
    res, store = _ingest(tmp_path, n=6, face_detector=NoFaceDetector())
    assert res.n_tracks == 1
    assert res.n_faces_detected == 0
    assert res.n_sightings == 0
    assert store.sightings == []


def test_video_registered_with_metadata(tmp_path):
    _res, store = _ingest(tmp_path, n=4)
    assert len(store.videos) == 1
    v = store.videos[0]
    assert v["camera_id"] == "cam-1"
    assert v["width"] == 320 and v["height"] == 240
