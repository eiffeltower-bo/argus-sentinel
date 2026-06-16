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
    category: str | None = None

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


@dataclass(frozen=True)
class Identity:
    """A person the store knows about: an enrolled watchlist entry or an auto-formed cluster.

    ``type`` is ``"known"`` (manually enrolled, named) or ``"provisional"`` (a clustering run
    grouped unlabeled sightings into a candidate person). ``id`` is assigned on insert.
    """

    type: str
    label: str | None = None
    created_by: str | None = None
    created_at: str | None = None
    notes: str | None = None
    id: int | None = None


@dataclass(frozen=True)
class Enrollment:
    """A reference face for a known ``Identity`` — the watchlist gallery row.

    Holds the chip path + the embedding space it was embedded in; the vector itself lives in
    the store's vector index. ``source`` notes provenance (``"id_photo"``/``"footage_still"``).
    """

    identity_id: int
    chip_path: str
    embedding_space_id: str
    source: str | None = None
    id: int | None = None


@dataclass(frozen=True, eq=False)
class SearchHit:
    """One ranked search result: a matched ``Sighting`` plus its similarity to the probe.

    ``distance`` is the raw vector distance (cosine distance — vectors are L2-normalized);
    ``score`` is ``1 - distance`` (cosine similarity in ``[0, 1]``) for thresholds/UI. The
    chip/camera/ts are surfaced for human adjudication ("evidence").
    """

    sighting: Sighting
    distance: float
    score: float

    @property
    def chip_path(self) -> str:
        return self.sighting.chip_path

    @property
    def camera_id(self) -> str:
        return self.sighting.camera_id

    @property
    def ts(self) -> float:
        return self.sighting.ts


@dataclass(frozen=True)
class WatchlistHit:
    """A ranked watchlist match: which enrolled ``Identity`` a probe face resembles."""

    identity: Identity
    distance: float
    score: float
    chip_path: str


@dataclass(frozen=True)
class AudioPrediction:
    """One ranked class prediction for an audio segment: a label and confidence in ``[0, 1]``."""

    label: str
    confidence: float


@dataclass(frozen=True)
class AudioSegment:
    """One analyzed audio window: its time span and the top-k ranked predictions.

    ``start_time``/``end_time`` are seconds from the start of the audio. ``predictions`` are
    ranked best-first; backends keep the top-k and pad to a fixed width with ``confidence`` 0.0
    so every segment has the same shape.
    """

    segment_index: int
    start_time: float
    end_time: float
    predictions: tuple[AudioPrediction, ...] = ()

    @property
    def top(self) -> AudioPrediction | None:
        return self.predictions[0] if self.predictions else None
