"""Shared fixtures + fakes for the fast unit suite (no GPU/weights/data needed)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from argus.core import Detection, FaceDetection, Track


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
            Track(d.x1, d.y1, d.x2, d.y2, d.score, self.track_id,
                  class_id=d.class_id, label=d.label, category="person")
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
            (0.35 * w, 0.40 * h), (0.65 * w, 0.40 * h), (0.50 * w, 0.55 * h),
            (0.40 * w, 0.70 * h), (0.60 * w, 0.70 * h),
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


class FakeStore:
    """An in-memory ``Store`` capturing writes; ``chips_dir`` points at a real temp dir."""

    def __init__(self, chips_dir) -> None:
        self.chips_dir = chips_dir
        self.videos: list[dict] = []
        self.sightings: list = []

    def add_video(self, camera_id, path, *, fps, duration_s, width, height) -> int:
        self.videos.append(
            {"camera_id": camera_id, "path": path, "fps": fps,
             "duration_s": duration_s, "width": width, "height": height}
        )
        return len(self.videos)

    def add_sightings(self, rows) -> None:
        for r in rows:
            r.id = len(self.sightings) + 1
            self.sightings.append(r)


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
