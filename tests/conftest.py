"""Shared fixtures + fakes for the fast unit suite (no GPU/weights/data needed)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from dataclasses import replace

from argus.core import (
    AudioPrediction,
    Detection,
    FaceDetection,
    SearchHit,
    Track,
    WatchlistHit,
)


class ScriptedDetector:
    """A fake ``Detector`` that returns a pre-scripted detection list per call.

    Lets tests drive the tracker/pipeline deterministically without a real model.
    """

    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = list(per_frame)
        self._i = 0

    @property
    def targets(self) -> tuple[str, ...]:
        return ()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets = self._per_frame[self._i] if self._i < len(self._per_frame) else []
        self._i += 1
        return dets


def drifting_person(n_frames: int, *, dx: int = 8, score: float = 0.9) -> list[list[Detection]]:
    """One person box sliding right by ``dx`` px per frame (boxes overlap → one track)."""
    return [
        [Detection(100 + i * dx, 200, 160 + i * dx, 360, score, class_id=0, label="person")]
        for i in range(n_frames)
    ]


@pytest.fixture
def scripted_detector():
    return ScriptedDetector


class FakeTracker:
    """A fake ``Tracker`` that promotes every detection to a track under one fixed id.

    Removes ByteTrack's confirmation delay so ingest tests can assert exact per-frame
    behaviour (one stable track => one sighting).
    """

    def __init__(self, track_id: int = 1) -> None:
        self.track_id = track_id

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        return [
            Track(
                d.x1,
                d.y1,
                d.x2,
                d.y2,
                d.score,
                self.track_id,
                class_id=d.class_id,
                label=d.label,
                category="person",
            )
            for d in detections
        ]

    def reset(self) -> None:
        pass


class ScriptedFaceDetector:
    """A fake ``FaceDetector`` returning one face per call with a pre-scripted score.

    Decouples face scores from (lossy) decoded pixels so ingest tests can assert exact
    gating/best-face behaviour. Yields nothing once the script is exhausted.
    """

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self._i = 0

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        if self._i >= len(self._scores):
            return []
        score = self._scores[self._i]
        self._i += 1
        h, w = image.shape[:2]
        lm = (
            (0.35 * w, 0.40 * h),
            (0.65 * w, 0.40 * h),
            (0.50 * w, 0.55 * h),
            (0.40 * w, 0.70 * h),
            (0.60 * w, 0.70 * h),
        )
        return [FaceDetection(0.25 * w, 0.25 * h, 0.75 * w, 0.75 * h, score, landmarks=lm)]


class NoFaceDetector:
    """A fake ``FaceDetector`` that never finds a face."""

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        return []


class FakeEmbedder:
    """A fake ``Embedder``: tiny deterministic L2-normalized vectors from chip stats."""

    embedding_space_id = "fake_v1"
    dim = 8

    def embed(self, chips: list[np.ndarray]) -> np.ndarray:
        if not chips:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(chips), self.dim), dtype=np.float32)
        for i, chip in enumerate(chips):
            base = float(chip.mean()) + 1.0
            out[i] = base * (np.arange(self.dim, dtype=np.float32) + 1.0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)


class ScriptedAudioClassifier:
    """A fake ``AudioClassifier`` with deterministic predictions and no heavy deps.

    Records every ``(n_samples, samplerate)`` it is called with, so tests can assert which
    segments the orchestrator actually fed it (windowing / overlap / tail / <100-sample skip).
    """

    classifier_id = "fake_audio_v1"
    is_zero_shot = False

    def __init__(self, label: str = "speech", confidence: float = 0.9) -> None:
        self.calls: list[tuple[int, int]] = []
        self._label, self._conf = label, confidence

    def classify(self, samples, samplerate, *, top_k: int = 2, candidate_labels=None):
        self.calls.append((len(samples), int(samplerate)))
        preds = [AudioPrediction(self._label, self._conf)]
        while len(preds) < top_k:
            preds.append(AudioPrediction("None", 0.0))
        return preds


class FakeStore:
    """In-memory ``SearchableStore`` for tests — brute-force numpy search, no sqlite-vec.

    Mirrors ``SqliteStore`` semantics closely enough to drive search/enroll/cluster/admin tests
    with fakes. ``purge``/``export_case`` are authoritatively tested against the real store.
    """

    def __init__(self, chips_dir) -> None:
        self.chips_dir = chips_dir
        self.videos: list[dict] = []
        self.sightings: list = []
        self.identities: list = []
        self.enrollments: list = []  # (Enrollment, vec)
        self.cluster_runs: list = []
        self.audit_rows: list[dict] = []

    # --- write path ---
    def add_video(self, camera_id, path, *, fps, duration_s, width, height) -> int:
        self.videos.append(
            {
                "camera_id": camera_id,
                "path": path,
                "fps": fps,
                "duration_s": duration_s,
                "width": width,
                "height": height,
            }
        )
        return len(self.videos)

    def add_sightings(self, rows) -> None:
        for r in rows:
            r.id = len(self.sightings) + 1
            self.sightings.append(r)

    # --- search ---
    def search_sightings(self, vec, space_id, *, top_k, cameras=None, since=None, min_quality=0.0):
        cand = [s for s in self.sightings if s.embedding_space_id == space_id]
        if cameras:
            cand = [s for s in cand if s.camera_id in cameras]
        if since is not None:
            cand = [s for s in cand if s.ts >= since]
        if min_quality > 0.0:
            cand = [s for s in cand if s.quality >= min_quality]
        scored = sorted(
            ((1.0 - float(np.dot(vec, s.embedding)), s) for s in cand), key=lambda x: x[0]
        )
        return [SearchHit(sighting=s, distance=d, score=1.0 - d) for d, s in scored[:top_k]]

    def search_enrollments(self, vec, space_id, *, top_k):
        scored = sorted(
            (
                (1.0 - float(np.dot(vec, v)), e)
                for e, v in self.enrollments
                if e.embedding_space_id == space_id
            ),
            key=lambda x: x[0],
        )
        hits, seen = [], set()
        for d, e in scored[:top_k]:
            if e.identity_id in seen:
                continue
            seen.add(e.identity_id)
            hits.append(
                WatchlistHit(
                    identity=self.get_identity(e.identity_id),
                    distance=d,
                    score=1.0 - d,
                    chip_path=e.chip_path,
                )
            )
        return hits

    def get_sighting(self, sighting_id, *, with_embedding=True):
        return next((s for s in self.sightings if s.id == sighting_id), None)

    def get_embedding(self, sighting_id):
        s = self.get_sighting(sighting_id)
        return None if s is None else s.embedding

    def iter_sightings(self, *, space_id, unassigned_only=False):
        for s in sorted(self.sightings, key=lambda s: s.id):
            if s.embedding_space_id != space_id:
                continue
            if unassigned_only and s.identity_id is not None:
                continue
            yield s

    # --- identity / enrollment ---
    def add_identity(self, identity) -> int:
        new = replace(identity, id=len(self.identities) + 1)
        self.identities.append(new)
        return new.id

    def get_identity(self, identity_id):
        return next((i for i in self.identities if i.id == identity_id), None)

    def list_identities(self, *, type=None):
        return [i for i in self.identities if type is None or i.type == type]

    def add_enrollment(self, enrollment, vec) -> int:
        new = replace(enrollment, id=len(self.enrollments) + 1)
        self.enrollments.append((new, np.asarray(vec, dtype=np.float32)))
        return new.id

    # --- assignment / clustering ---
    def assign_identity(self, sighting_id, identity_id, *, actor="unknown") -> None:
        s = self.get_sighting(sighting_id)
        if s is not None:
            s.identity_id = identity_id
        self.audit(
            actor=actor,
            action="assign_identity",
            target_type="sighting",
            target_id=sighting_id,
            details=f"identity_id={identity_id}",
        )

    def assign_cluster(self, sighting_ids, cluster_id) -> None:
        for s in self.sightings:
            if s.id in sighting_ids:
                s.cluster_id = cluster_id

    def merge_cluster_into_identity(self, cluster_id, identity_id, *, actor="unknown") -> int:
        n = 0
        for s in self.sightings:
            if s.cluster_id == cluster_id:
                s.identity_id, s.cluster_id, n = identity_id, None, n + 1
        self.audit(
            actor=actor,
            action="merge",
            target_type="identity",
            target_id=identity_id,
            details=f"cluster_id={cluster_id} n={n}",
        )
        return n

    def add_cluster_run(self, algo, params, space_id) -> int:
        self.cluster_runs.append({"algo": algo, "params": params, "space_id": space_id})
        return len(self.cluster_runs)

    # --- compliance ---
    def audit(
        self, *, actor, action, target_type=None, target_id=None, query_ref=None, details=None
    ) -> None:
        self.audit_rows.append(
            {
                "actor": actor,
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "query_ref": query_ref,
                "details": details,
            }
        )

    def list_audit(self, *, actor=None, since=None):
        return [r for r in self.audit_rows if actor is None or r["actor"] == actor]

    def purge(
        self, *, before, actor="unknown"
    ) -> int:  # in-memory: real semantics tested on sqlite
        n = len(self.sightings)
        self.sightings.clear()
        self.audit(actor=actor, action="purge", details=f"before={before} n={n}")
        return n

    def export_case(self, identity_id, dest, *, actor="unknown"):
        from pathlib import Path

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self.audit(actor=actor, action="export", target_type="identity", target_id=identity_id)
        return dest


@pytest.fixture
def make_video(tmp_path):
    """Factory writing a tiny synthetic mp4 (mp4v) and returning its path."""

    def _make(n_frames: int = 10, w: int = 320, h: int = 240, fps: float = 10.0):
        path = tmp_path / "clip.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n_frames):
            vw.write(np.full((h, w, 3), (i * 7) % 255, np.uint8))
        vw.release()
        return path

    return _make
