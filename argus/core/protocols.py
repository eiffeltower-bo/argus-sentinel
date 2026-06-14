"""The SDK's extension contracts — implement one of these to add a backend.

These Protocols are the single place new functionality plugs in: a detector, a tracker
(and, on the roadmap, a face detector / embedder / output sink). They depend only on the
core data types, never on a concrete backend.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Detection, Track


@runtime_checkable
class Detector(Protocol):
    """Detect objects in a single BGR frame (H x W x 3, uint8).

    The model-agnostic contract the pipeline depends on: any object with this ``.detect``
    signature is a valid backend, ultralytics or otherwise. A backend MAY also offer an
    optional ``detect_batch(frames, *, batch_size) -> list[list[Detection]]`` for batched
    throughput (used by ``peek_videos`` when present); it is not required by this protocol.
    """

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Associate per-frame detections into tracks. Frames must be fed in order."""

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]: ...

    def reset(self) -> None: ...
