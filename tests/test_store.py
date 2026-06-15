"""Round-trip tests for the SQLite + sqlite-vec store. Skipped if sqlite-vec is absent."""

import numpy as np
import pytest

pytest.importorskip("sqlite_vec")

from argus.core import Sighting  # noqa: E402  (after importorskip)
from argus.store import SqliteStore  # noqa: E402


def _sighting(video_id, *, track_id=1, dim=8):
    return Sighting(
        video_id=video_id,
        camera_id="cam-1",
        track_id=track_id,
        frame_idx=5,
        ts=0.5,
        bbox=(1.0, 2.0, 3.0, 4.0),
        quality=0.8,
        chip_path="/chips/c.png",
        embedding_space_id="fake_v1",
        embedding=np.ones(dim, dtype=np.float32),
    )


def test_add_video_and_sightings_round_trip(tmp_path):
    store = SqliteStore(tmp_path / "argus.db", dim=8)
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=320, height=240)
    assert vid == 1

    s = _sighting(vid)
    store.add_sightings([s])
    assert s.id is not None

    rows = store.list_sightings()
    assert len(rows) == 1
    row = rows[0]
    assert row["video_id"] == vid
    assert row["track_id"] == 1
    assert row["embedding_space_id"] == "fake_v1"
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (1.0, 2.0, 3.0, 4.0)

    assert store.count_vectors() == 1
    store.close()


def test_chips_dir_created_beside_db(tmp_path):
    store = SqliteStore(tmp_path / "nested" / "argus.db", dim=8)
    assert store.chips_dir.exists()
    assert store.chips_dir == tmp_path / "nested" / "chips"
    store.close()


def test_multiple_sightings_indexed(tmp_path):
    store = SqliteStore(tmp_path / "argus.db", dim=8)
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=320, height=240)
    store.add_sightings([_sighting(vid, track_id=t) for t in (1, 2, 3)])
    assert len(store.list_sightings()) == 3
    assert store.count_vectors() == 3
    store.close()
