"""The SDK's extension contracts — implement one of these to add a backend.

These Protocols are the single place new functionality plugs in: a detector, a tracker,
a face detector, an embedder, and an output store. They depend only on the core data
types, never on a concrete backend.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Detection, FaceDetection, Sighting, Track


@runtime_checkable
class Detector(Protocol):
    """Detect objects in a single BGR frame (H x W x 3, uint8).

    The model-agnostic contract the pipeline depends on: any object with this ``.detect``
    signature is a valid backend, ultralytics or otherwise. A backend MAY also offer an
    optional ``detect_batch(frames, *, batch_size) -> list[list[Detection]]`` for batched
    throughput (used by ``peek_videos`` when present); it is not required by this protocol.
    """

    @property
    def targets(self) -> tuple[str, ...]: ...

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Associate per-frame detections into tracks. Frames must be fed in order."""

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]: ...

    def reset(self) -> None: ...


@runtime_checkable
class FaceDetector(Protocol):
    """Detect faces in a single BGR image (H x W x 3, uint8).

    The image is typically a person crop, not a full frame (face-ID runs the detector on
    tracked person boxes). Each ``FaceDetection`` carries a box, a score, and the 5
    landmarks needed for alignment.
    """

    def detect(self, image: np.ndarray) -> list[FaceDetection]: ...


@runtime_checkable
class Embedder(Protocol):
    """Turn aligned face chips into L2-normalized embedding vectors.

    ``embedding_space_id`` namespaces the vector space: embeddings from different models
    are NOT comparable, so every persisted vector is tagged with it and searches filter by
    it. Swapping the embedder means re-embedding the index under a new id.
    """

    embedding_space_id: str
    dim: int

    def embed(self, chips: list[np.ndarray]) -> np.ndarray: ...  # (N, dim) float32, L2-normalized


@runtime_checkable
class Store(Protocol):
    """Persistence sink for the ingest pipeline (write-path only, for now).

    Search/enroll/clustering methods are added in later phases when first needed.
    """

    def add_video(
        self,
        camera_id: str,
        path: str,
        *,
        fps: float,
        duration_s: float,
        width: int,
        height: int,
    ) -> int: ...

    def add_sightings(self, rows: list[Sighting]) -> None: ...
