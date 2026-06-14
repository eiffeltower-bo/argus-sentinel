"""Shared fixtures + fakes for the fast unit suite (no GPU/weights/data needed)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from argus.core import Detection


class ScriptedDetector:
    """A fake ``Detector`` that returns a pre-scripted detection list per call.

    Lets tests drive the tracker/pipeline deterministically without a real model.
    """

    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = list(per_frame)
        self._i = 0

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
