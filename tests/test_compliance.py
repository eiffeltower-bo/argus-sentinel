"""Compliance: purge (retention) + export_case + audit_log, on the real sqlite-vec store."""

import json

import numpy as np
import pytest

pytest.importorskip("sqlite_vec")

from argus import audit_log, export_case, purge  # noqa: E402
from argus.core import Identity, Sighting  # noqa: E402
from argus.store import SqliteStore  # noqa: E402

DIM = 8


def _sighting(vid, *, track_id, chip_path):
    return Sighting(
        video_id=vid,
        camera_id="cam-1",
        track_id=track_id,
        frame_idx=track_id,
        ts=float(track_id),
        bbox=(0.0, 0.0, 1.0, 1.0),
        quality=0.9,
        chip_path=str(chip_path),
        embedding_space_id="fake_v1",
        embedding=np.ones(DIM, dtype=np.float32) / np.sqrt(DIM),
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "argus.db", dim=DIM)
    yield s
    s.close()


def test_purge_removes_rows_vectors_and_chips(store, tmp_path):
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=10, height=10)
    old_chip = tmp_path / "old.png"
    new_chip = tmp_path / "new.png"
    old_chip.write_bytes(b"x")
    new_chip.write_bytes(b"y")
    old = _sighting(vid, track_id=1, chip_path=old_chip)
    new = _sighting(vid, track_id=2, chip_path=new_chip)
    store.add_sightings([old, new])
    # backdate the "old" sighting so it falls before the cutoff
    store.conn.execute(
        "UPDATE sightings SET created_at = '2000-01-01 00:00:00' WHERE id = ?", (old.id,)
    )
    store.conn.commit()
    assert store.count_vectors() == 2

    n = purge(store=store, before="2020-01-01 00:00:00", actor="dpo")
    assert n == 1
    assert [r["track_id"] for r in store.list_sightings()] == [2]  # only the new one remains
    assert store.count_vectors() == 1  # its vector also gone
    assert not old_chip.exists() and new_chip.exists()  # chip file unlinked
    assert any(r["action"] == "purge" for r in store.list_audit())


def test_export_case_bundles_chips_and_manifest(store, tmp_path):
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=10, height=10)
    iid = store.add_identity(Identity(type="known", label="J. Doe"))
    chips = []
    rows = []
    for t in (1, 2):
        cp = tmp_path / f"chip{t}.png"
        cp.write_bytes(b"img")
        chips.append(cp)
        rows.append(_sighting(vid, track_id=t, chip_path=cp))
    store.add_sightings(rows)
    for r in rows:
        store.assign_identity(r.id, iid, actor="op")

    dest = export_case(iid, tmp_path / "case", store=store, actor="op")
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["identity"]["label"] == "J. Doe"
    assert len(manifest["sightings"]) == 2
    assert (dest / "chip1.png").exists() and (dest / "chip2.png").exists()
    assert any(r["action"] == "export" for r in store.list_audit())


def test_audit_log_filters_by_actor(store):
    store.audit(actor="alice", action="search_by_image")
    store.audit(actor="bob", action="search_by_image")
    store.audit(actor="alice", action="enroll")
    assert len(audit_log(store=store)) == 3
    assert {r["action"] for r in audit_log(store=store, actor="alice")} == {
        "search_by_image",
        "enroll",
    }
