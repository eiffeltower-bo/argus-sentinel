"""Unit tests for Track, the detections adapter, and the ByteTrack backend.

The ByteTrack backend is exercised with synthetic detections — it needs numpy only,
no model/GPU/weights — so these run in milliseconds.
"""

import numpy as np
import pytest

from faces_cv.detection import Detection
from faces_cv.tracking import ByteTrackTracker, Track, _DetectionsAdapter


def test_track_geometry():
    t = Track(10, 20, 30, 60, 0.8, track_id=3, class_id=0, label="person", category="person")
    assert t.xyxy == (10, 20, 30, 60)
    assert t.center == (20.0, 40.0)


def test_adapter_xywh_and_len():
    dets = [Detection(10, 20, 30, 60, 0.9, class_id=0)]
    a = _DetectionsAdapter.from_detections(dets)
    assert len(a) == 1
    cx, cy, w, h = a.xywh[0]
    assert (cx, cy, w, h) == (20.0, 40.0, 20.0, 40.0)
    assert a.conf[0] == np.float32(0.9)
    assert a.cls[0] == np.float32(0)


def test_adapter_empty():
    a = _DetectionsAdapter.from_detections([])
    assert len(a) == 0
    assert a.xywh.shape == (0, 4)


def test_adapter_boolean_mask():
    dets = [Detection(0, 0, 10, 10, 0.9, class_id=0), Detection(0, 0, 10, 10, 0.2, class_id=2)]
    a = _DetectionsAdapter.from_detections(dets)
    high = a[a.conf >= 0.5]
    assert len(high) == 1
    assert high.cls[0] == np.float32(0)


def test_bytetrack_stable_id_across_frames():
    trk = ByteTrackTracker()
    frame = np.zeros((720, 1280, 3), np.uint8)
    ids = []
    for i in range(6):
        x = 100 + i * 8  # small drift -> boxes overlap -> one track
        tracks = trk.update([Detection(x, 200, x + 60, 360, 0.9, class_id=0)], frame)
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
        assert tracks[0].category == "person"
        assert tracks[0].label == "person"
    assert len(set(ids)) == 1  # same id the whole way


def test_bytetrack_tags_vehicle():
    trk = ByteTrackTracker()
    frame = np.zeros((720, 1280, 3), np.uint8)
    t = trk.update([Detection(300, 300, 460, 420, 0.9, class_id=2)], frame)
    assert t[0].category == "vehicle"
    assert t[0].label == "car"


def test_bytetrack_reset_restarts_ids():
    trk = ByteTrackTracker()
    frame = np.zeros((720, 1280, 3), np.uint8)
    first = trk.update([Detection(100, 100, 160, 260, 0.9, class_id=0)], frame)[0]
    trk.reset()
    after = trk.update([Detection(500, 500, 560, 660, 0.9, class_id=0)], frame)[0]
    assert after.track_id == first.track_id  # counter reset -> same starting id


def test_bytetrack_empty_detections():
    trk = ByteTrackTracker()
    assert trk.update([], np.zeros((480, 640, 3), np.uint8)) == []


def test_bytetrack_output_row_contract():
    # Guards the ultralytics-internal assumptions our adapter relies on: the
    # BYTETracker args namespace, and the output row layout where col 4=track_id,
    # 5=score, 6=cls. If an ultralytics bump changes any of these, this fails loudly.
    trk = ByteTrackTracker()
    frame = np.zeros((720, 1280, 3), np.uint8)
    t = trk.update([Detection(300, 300, 460, 460, 0.9, class_id=7)], frame)[0]
    assert isinstance(t.track_id, int) and t.track_id >= 1  # row[4]
    assert t.score == pytest.approx(0.9, abs=0.05)          # row[5]
    assert t.class_id == 7 and t.label == "truck"           # row[6] -> COCO_LABELS
    assert t.category == "vehicle"
