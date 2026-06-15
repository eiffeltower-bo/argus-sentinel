"""Core data types shared across the SDK: detections, tracks, and faces.

Pure dataclasses with no heavy dependencies, so every layer can import them without
pulling in cv2/ultralytics/polars. (numpy is the one exception — it's universal, and the
persisted face-sighting record carries an embedding vector.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single object detection, in absolute pixel xyxy coordinates.

    ``class_id`` / ``label`` are optional so single-class detectors can emit a bare
    ``Detection(x1, y1, x2, y2, score)``; multi-class detectors fill them in so
    downstream tracking can categorise.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int | None = None
    label: str | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


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


@dataclass(frozen=True)
class FaceDetection:
    """A detected face: an xyxy box, a score, and 5 facial landmarks.

    ``landmarks`` are the canonical 5 points (left eye, right eye, nose, left mouth
    corner, right mouth corner) in absolute pixel coordinates of the image the detector
    was run on — used to align the face to a canonical chip before embedding.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    landmarks: tuple[tuple[float, float], ...] = ()

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Sighting:
    """One persisted face observation: the best face of a person track in a video.

    Carries the embedding vector (so not frozen — ndarrays aren't hashable/comparable)
    and the on-disk path of the aligned chip. ``id`` is assigned by the store on insert.
    ``identity_id`` / ``cluster_id`` stay ``None`` until the search/clustering phases.
    """

    video_id: int
    camera_id: str
    track_id: int
    frame_idx: int
    ts: float
    bbox: tuple[float, float, float, float]
    quality: float
    chip_path: str
    embedding_space_id: str
    embedding: np.ndarray
    identity_id: int | None = None
    cluster_id: int | None = None
    id: int | None = None
