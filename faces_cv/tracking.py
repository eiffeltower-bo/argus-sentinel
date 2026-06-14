"""Tracking — a model-agnostic ``Tracker`` interface plus a ByteTrack backend.

Tracking-by-detection, decoupled from detection: a :class:`Tracker` consumes the
:class:`~faces_cv.detection.Detection` list for one frame and returns :class:`Track`
objects with stable ``track_id``s. Swap the backend by providing another class with
the same ``.update`` / ``.reset`` signature.

The default :class:`ByteTrackTracker` drives ultralytics' bundled ``BYTETracker``
standalone (no ``model.track()``), via a small numpy adapter — so detector and tracker
are independently pluggable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol, runtime_checkable

import numpy as np

from .detection import Detection

# COCO classes we care about, and how they roll up into categories.
COCO_LABELS: dict[int, str] = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CATEGORY_BY_CLASS: dict[int, str] = {
    0: "person", 2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle"
}
TARGET_CLASSES: dict[str, list[int]] = {"person": [0], "vehicle": [2, 3, 5, 7]}


@dataclass(frozen=True)
class Track:
    """A tracked object in one frame: a box plus a stable ``track_id``."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    track_id: int
    class_id: int | None = None
    label: str | None = None
    category: str | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@runtime_checkable
class Tracker(Protocol):
    """Associate per-frame detections into tracks. Frames must be fed in order."""

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]: ...

    def reset(self) -> None: ...


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
    keeps the ``frame`` arg to satisfy the :class:`Tracker` protocol. ``track_buffer``
    is how many frames a lost track is kept alive before retirement.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
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
                    x1, y1, x2, y2,
                    score=float(row[5]),
                    track_id=int(row[4]),
                    class_id=cls_id,
                    label=COCO_LABELS.get(cls_id),
                    category=CATEGORY_BY_CLASS.get(cls_id),
                )
            )
        return tracks
