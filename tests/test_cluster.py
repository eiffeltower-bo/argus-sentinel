"""Open re-ID: run_clustering + label_cluster / merge / reassign. Needs scikit-learn."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from argus import label_cluster, merge, reassign, run_clustering  # noqa: E402
from argus.core import Identity, Sighting  # noqa: E402
from conftest import FakeStore  # noqa: E402 (conftest on pythonpath)

DIM = 8


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _populate(store):
    """Two tight blobs (around e0 and e4) of 8 each + 2 lone outliers, fixed seed."""
    vid = store.add_video("cam-1", "/x.mp4", fps=10.0, duration_s=1.0, width=10, height=10)
    rng = np.random.default_rng(0)
    a, b = np.zeros(DIM), np.zeros(DIM)
    a[0], b[4] = 1.0, 1.0
    rows, tid = [], 1

    def add(vec):
        nonlocal tid
        rows.append(
            Sighting(
                video_id=vid,
                camera_id="cam-1",
                track_id=tid,
                frame_idx=tid,
                ts=float(tid),
                bbox=(0.0, 0.0, 1.0, 1.0),
                quality=0.9,
                chip_path=f"/c{tid}.png",
                embedding_space_id="fake_v1",
                embedding=_norm(vec),
            )
        )
        tid += 1

    for _ in range(8):
        add(a + rng.normal(0, 0.02, DIM))
    for _ in range(8):
        add(b + rng.normal(0, 0.02, DIM))
    add(_norm([0, 0, 1, 0, 0, 0, 0, 0]))  # outliers (lone points -> noise)
    add(_norm([0, 0, 0, 0, 0, 0, 1, 0]))
    store.add_sightings(rows)


def _cid(store, track_id):
    return next(s.cluster_id for s in store.sightings if s.track_id == track_id)


def test_run_clustering_forms_two_provisional_identities(tmp_path):
    store = FakeStore(tmp_path)
    _populate(store)

    res = run_clustering(store=store, space_id="fake_v1", min_cluster_size=3, actor="op")
    assert res.n_clusters == 2

    # Robust to HDBSCAN's exact noise calls / label numbers: assert each blob is internally
    # one cluster and the two blobs land in different clusters.
    a_clusters = {_cid(store, t) for t in range(1, 9)}
    b_clusters = {_cid(store, t) for t in range(9, 17)}
    assert len(a_clusters) == 1 and None not in a_clusters
    assert len(b_clusters) == 1 and None not in b_clusters
    assert a_clusters != b_clusters
    assert len(store.list_identities(type="provisional")) == 2
    assert store.cluster_runs and store.audit_rows[-1]["action"] == "run_clustering"


def test_label_cluster_merge_reassign(tmp_path):
    store = FakeStore(tmp_path)
    _populate(store)
    run_clustering(store=store, space_id="fake_v1", min_cluster_size=3, actor="op")
    a_cid, b_cid = _cid(store, 1), _cid(store, 9)  # each blob's provisional identity id

    # label_cluster: promote the A blob into a named known identity
    known = label_cluster(a_cid, "Suspect A", store=store, actor="op")
    ident = store.get_identity(known)
    assert ident.type == "known" and ident.label == "Suspect A"
    a_blob = [s for s in store.sightings if s.track_id in range(1, 9)]
    assert all(s.identity_id == known and s.cluster_id is None for s in a_blob)

    # merge: fold the B blob into the same known identity
    n = merge(into=known, source_cluster=b_cid, store=store, actor="op")
    assert n >= 8
    b_blob = [s for s in store.sightings if s.track_id in range(9, 17)]
    assert all(s.identity_id == known and s.cluster_id is None for s in b_blob)

    # reassign: move one sighting out to a fresh identity
    other = store.add_identity(Identity(type="known", label="Suspect B"))
    target = a_blob[0].id
    reassign(target, other, store=store, actor="op")
    assert store.get_sighting(target).identity_id == other
    assert any(r["action"] == "assign_identity" for r in store.audit_rows)
