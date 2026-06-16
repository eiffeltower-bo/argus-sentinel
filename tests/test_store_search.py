"""SqliteStore search/enroll tests — real sqlite-vec KNN. Skipped if sqlite-vec is absent."""

import numpy as np
import pytest

pytest.importorskip("sqlite_vec")

from argus.core import Enrollment, Identity, Sighting  # noqa: E402
from argus.store import SqliteStore  # noqa: E402

DIM = 8


def _unit(*idxs):
    v = np.zeros(DIM, dtype=np.float32)
    for i in idxs:
        v[i] = 1.0
    return v / np.linalg.norm(v)


def _sighting(vid, *, track_id, vec, camera="cam-1", quality=0.9, ts=0.0, space="fake_v1"):
    return Sighting(video_id=vid, camera_id=camera, track_id=track_id, frame_idx=track_id,
                    ts=ts, bbox=(0.0, 0.0, 1.0, 1.0), quality=quality,
                    chip_path=f"/chips/c{track_id}.png", embedding_space_id=space, embedding=vec)


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "argus.db", dim=DIM)
    yield s
    s.close()


def test_search_sightings_ranks_by_cosine(store):
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=320, height=240)
    e0 = _unit(0)
    near0 = np.array([0.97, 0.24, 0, 0, 0, 0, 0, 0], np.float32)  # close to e0
    e1 = _unit(1)                                                 # orthogonal to e0
    store.add_sightings([
        _sighting(vid, track_id=1, vec=e0),
        _sighting(vid, track_id=2, vec=near0),
        _sighting(vid, track_id=3, vec=e1),
    ])
    hits = store.search_sightings(e0, "fake_v1", top_k=3)
    assert [h.sighting.track_id for h in hits] == [1, 2, 3]      # exact, near, orthogonal
    assert hits[0].score > hits[1].score > hits[2].score
    assert abs(hits[0].score - 1.0) < 1e-5                       # cosine ~1 for the exact match


def test_search_filters_prefilter_in_scan(store):
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=320, height=240)
    store.add_sightings([
        _sighting(vid, track_id=1, vec=_unit(0), camera="camA", quality=0.9, ts=10.0),
        _sighting(vid, track_id=2, vec=_unit(0), camera="camB", quality=0.9, ts=10.0),
        _sighting(vid, track_id=3, vec=_unit(0), camera="camA", quality=0.2, ts=10.0),
        _sighting(vid, track_id=4, vec=_unit(0), camera="camA", quality=0.9, ts=1.0),
    ])
    ids = [h.sighting.track_id
           for h in store.search_sightings(_unit(0), "fake_v1", top_k=10,
                                           cameras=["camA"], since=5.0, min_quality=0.5)]
    assert ids == [1]   # camB filtered, low-quality filtered, early-ts filtered — survivor returned


def test_search_isolated_by_embedding_space(store):
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=320, height=240)
    store.add_sightings([_sighting(vid, track_id=1, vec=_unit(0), space="space_a")])
    assert store.search_sightings(_unit(0), "space_b", top_k=5) == []
    assert len(store.search_sightings(_unit(0), "space_a", top_k=5)) == 1


def test_enroll_and_watchlist_search(store):
    iid = store.add_identity(Identity(type="known", label="J. Doe"))
    store.add_enrollment(
        Enrollment(identity_id=iid, chip_path="/e.png", embedding_space_id="fake_v1",
                   source="id_photo"),
        _unit(0),
    )
    hits = store.search_enrollments(np.array([0.97, 0.24, 0, 0, 0, 0, 0, 0], np.float32),
                                    "fake_v1", top_k=3)
    assert len(hits) == 1
    assert hits[0].identity.label == "J. Doe"
    assert hits[0].score > 0.9


def test_get_embedding_round_trips(store):
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=320, height=240)
    s = _sighting(vid, track_id=1, vec=_unit(2))
    store.add_sightings([s])
    got = store.get_embedding(s.id)
    assert np.allclose(got, _unit(2))
