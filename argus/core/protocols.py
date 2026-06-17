"""The SDK's extension contracts — implement one of these to add a backend.

These Protocols are the single place new functionality plugs in: a detector, a tracker,
a face detector, an embedder, and an output store. They depend only on the core data
types, never on a concrete backend.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .types import (
    AudioPrediction,
    Detection,
    Enrollment,
    FaceDetection,
    Identity,
    SearchHit,
    Sighting,
    Track,
    WatchlistHit,
)


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
class AudioClassifier(Protocol):
    """Classify one mono audio segment into ranked ``(label, confidence)`` predictions.

    The model-agnostic contract the audio pipeline depends on: any object with this ``.classify``
    signature is a valid backend (a HuggingFace AST or CLAP pipeline, or otherwise). It operates
    on a single segment of float samples at ``samplerate`` Hz; the pipeline handles decode +
    windowing and feeds segments one at a time, so the model is loaded once and reused across a
    whole clip.

    ``classifier_id`` namespaces the model that produced a prediction (the ``Embedder``
    ``embedding_space_id`` analogue), so a result records its provenance. ``is_zero_shot`` is
    True for CLAP-style models, which honor ``candidate_labels``.
    """

    classifier_id: str
    is_zero_shot: bool

    def classify(
        self,
        samples: np.ndarray,
        samplerate: int,
        *,
        top_k: int = 2,
        candidate_labels: list[str] | None = None,
    ) -> list[AudioPrediction]: ...


@runtime_checkable
class Store(Protocol):
    """Persistence sink for the ingest pipeline — the minimal write-path contract.

    ``ingest_video`` depends only on this. The read path (search, enroll, clustering,
    compliance) is the broader ``SearchableStore`` below; keeping them separate lets
    write-only consumers stay on this narrow contract.
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


@runtime_checkable
class SearchableStore(Store, Protocol):
    """The full datastore contract: write-path (``Store``) plus search / identity / compliance.

    Implemented by ``SqliteStore`` (and the test ``FakeStore``). All vector ops filter by
    ``space_id`` (``embedding_space_id``) — vectors from different embedders are not comparable.
    Search returns ranked candidates with evidence; assignment/compliance ops take an ``actor``
    for the audit trail.
    """

    # --- vector search ---
    def search_sightings(
        self,
        vec: np.ndarray,
        space_id: str,
        *,
        top_k: int,
        cameras: list[str] | None = None,
        since: float | None = None,
        min_quality: float = 0.0,
    ) -> list[SearchHit]: ...

    def search_enrollments(
        self, vec: np.ndarray, space_id: str, *, top_k: int
    ) -> list[WatchlistHit]: ...

    # --- read by id (clustering / search-by-sighting) ---
    def get_sighting(self, sighting_id: int) -> Sighting | None: ...

    def get_embedding(self, sighting_id: int) -> np.ndarray | None: ...

    def iter_sightings(
        self, *, space_id: str, unassigned_only: bool = False
    ) -> Iterator[Sighting]: ...

    # --- identity / enrollment ---
    def add_identity(self, identity: Identity) -> int: ...

    def get_identity(self, identity_id: int) -> Identity | None: ...

    def list_identities(self, *, type: str | None = None) -> list[Identity]: ...

    def add_enrollment(self, enrollment: Enrollment, vec: np.ndarray) -> int: ...

    # --- assignment / bookkeeping ---
    def assign_identity(
        self, sighting_id: int, identity_id: int | None, *, actor: str = "unknown"
    ) -> None: ...

    def assign_cluster(self, sighting_ids: list[int], cluster_id: int) -> None: ...

    def merge_cluster_into_identity(
        self, cluster_id: int, identity_id: int, *, actor: str = "unknown"
    ) -> int: ...

    def add_cluster_run(self, algo: str, params: str, space_id: str) -> int: ...

    # --- compliance ---
    def audit(
        self,
        *,
        actor: str,
        action: str,
        target_type: str | None = None,
        target_id: int | None = None,
        query_ref: str | None = None,
        details: str | None = None,
    ) -> None: ...

    def list_audit(self, *, actor: str | None = None, since: str | None = None) -> list[dict]: ...

    def purge(self, *, before: str, actor: str = "unknown") -> int: ...

    def export_case(self, identity_id: int, dest: Path, *, actor: str = "unknown") -> Path: ...
