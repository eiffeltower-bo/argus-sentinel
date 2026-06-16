"""search_by_image / search_by_sighting / enroll with fakes — no GPU/weights/sqlite-vec.

Exercises the orchestration over a FakeStore + ScriptedFaceDetector + FakeEmbedder, and that no
real model is imported (the heavy backends are only constructed when not injected).
"""

import numpy as np

from argus import enroll, search_by_image, search_by_sighting
from argus.core import Sighting
from conftest import FakeEmbedder, FakeStore, ScriptedFaceDetector  # noqa: E402 (conftest on pythonpath)


def _sighting(store_vid, *, track_id, vec, camera="cam-1", quality=0.9, ts=0.0):
    return Sighting(video_id=store_vid, camera_id=camera, track_id=track_id, frame_idx=track_id,
                    ts=ts, bbox=(0.0, 0.0, 1.0, 1.0), quality=quality,
                    chip_path=f"/chips/c{track_id}.png", embedding_space_id="fake_v1",
                    embedding=np.asarray(vec, dtype=np.float32))


def _probe_image():
    # ScriptedFaceDetector reads landmarks off the image shape, so any HxWx3 array works.
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_search_by_image_ranks_and_audits(tmp_path):
    store = FakeStore(tmp_path)
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=100, height=100)
    emb = FakeEmbedder()
    # Embed a known chip to learn the deterministic vector the probe will produce.
    probe_vec = emb.embed([np.zeros((112, 112, 3), np.uint8)])[0]
    far = np.ones(emb.dim, dtype=np.float32)
    far /= np.linalg.norm(far)
    store.add_sightings([
        _sighting(vid, track_id=1, vec=probe_vec),           # identical to the probe
        _sighting(vid, track_id=2, vec=far, quality=0.9),    # different
    ])

    hits = search_by_image(_probe_image(), store=store, top_k=2,
                           face_detector=ScriptedFaceDetector([0.99]), embedder=FakeEmbedder(),
                           actor="tester")
    assert hits[0].sighting.track_id == 1
    assert hits[0].score >= hits[1].score
    assert hits[0].chip_path == "/chips/c1.png"   # evidence surfaced
    assert store.audit_rows[-1]["action"] == "search_by_image"
    assert store.audit_rows[-1]["actor"] == "tester"


def test_search_by_sighting_excludes_self(tmp_path):
    store = FakeStore(tmp_path)
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=100, height=100)
    emb = FakeEmbedder()
    v = emb.embed([np.full((112, 112, 3), 30, np.uint8)])[0]
    other = emb.embed([np.full((112, 112, 3), 31, np.uint8)])[0]
    store.add_sightings([_sighting(vid, track_id=1, vec=v), _sighting(vid, track_id=2, vec=other)])

    hits = search_by_sighting(1, store=store, top_k=5)
    assert all(h.sighting.id != 1 for h in hits)   # self excluded
    assert hits[0].sighting.id == 2


def test_enroll_creates_known_identity(tmp_path):
    store = FakeStore(tmp_path)
    iid = enroll("J. Doe", [_probe_image(), _probe_image()], store=store,
                 face_detector=ScriptedFaceDetector([0.9, 0.9]), embedder=FakeEmbedder(),
                 actor="op")
    ident = store.get_identity(iid)
    assert ident.type == "known" and ident.label == "J. Doe"
    assert len(store.enrollments) == 2
    assert store.audit_rows[-1]["action"] == "enroll"
