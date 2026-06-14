"""Core data types shared across the SDK: a detection and a track.

Pure frozen dataclasses with no heavy dependencies, so every layer can import them
without pulling in cv2/ultralytics/polars.
"""

from __future__ import annotations

from dataclasses import dataclass


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
