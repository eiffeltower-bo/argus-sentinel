"""ByteTrack tracker backend (implements the core ``Tracker`` protocol).

Drives ultralytics' bundled ``BYTETracker`` standalone (no ``model.track()``) via a small
numpy adapter, so detector and tracker stay independently pluggable.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ...core import CATEGORY_BY_CLASS, COCO_LABELS, Detection, Track


class _DetectionsAdapter:
    """Wrap a ``Detection`` list as the results-like object ``BYTETracker`` expects.

    ``BYTETracker`` reads ``.xywh`` / ``.xyxy`` / ``.conf`` / ``.cls`` and slices with a
    boolean mask (``results[mask]``); this exposes exactly that over numpy arrays.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy.astype(np.float32).reshape(-1, 4)
        self.conf = conf.astype(np.float32).reshape(-1)
        self.cls = cls.astype(np.float32).reshape(-1)

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> "_DetectionsAdapter":
        if not detections:
            empty = np.empty((0, 4), dtype=np.float32)
            return cls(empty, np.empty(0, np.float32), np.empty(0, np.float32))
        xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
        conf = np.array([d.score for d in detections], dtype=np.float32)
        cls_ids = np.array(
            [d.class_id if d.class_id is not None else -1 for d in detections],
            dtype=np.float32,
        )
        return cls(xyxy, conf, cls_ids)

    @property
    def xywh(self) -> np.ndarray:
        # center-x, center-y, width, height
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_DetectionsAdapter":
        return _DetectionsAdapter(self.xyxy[mask], self.conf[mask], self.cls[mask])


class ByteTrackTracker:
    """ByteTrack backend driving ultralytics' standalone ``BYTETracker``.

    Defaults mirror ``ultralytics/cfg/trackers/bytetrack.yaml``. Pure motion+IoU
    association (no appearance model), so it ignores the frame image — but ``update``
    keeps the ``frame`` arg to satisfy the ``Tracker`` protocol. ``track_buffer`` is how
    many frames a lost track is kept alive before retirement.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
        labels: dict[int, str] | None = None,
        categories: dict[int, str] | None = None,
    ) -> None:
        self.args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self._tracker = None
        self._labels = labels or COCO_LABELS
        self._categories = categories or CATEGORY_BY_CLASS
        self.reset()

    def reset(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker  # lazy import

        self._tracker = BYTETracker(self.args)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        results = _DetectionsAdapter.from_detections(detections)
        rows = self._tracker.update(results, img=frame)
        tracks: list[Track] = []
        for row in rows:  # row = [x1, y1, x2, y2, track_id, score, cls, idx]
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            cls_id = int(row[6])
            tracks.append(
                Track(
                    x1,
                    y1,
                    x2,
                    y2,
                    score=float(row[5]),
                    track_id=int(row[4]),
                    class_id=cls_id,
                    label=self._labels.get(cls_id),
                    category=self._categories.get(cls_id),
                )
            )
        return tracks
