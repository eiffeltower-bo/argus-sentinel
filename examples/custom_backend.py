#!/usr/bin/env python
"""Plug in your own backend — anything matching the Detector/Tracker protocol works.

Shows two things:
  1. The explicit, pluggable form ``VideoTracker(detector, tracker)`` that ``track_video`` wraps.
  2. A tiny custom Detector (a confidence-filtering wrapper) satisfying ``argus.core.Detector`` —
     no base class or registration, just a ``.detect(frame)`` method.

    uv run python examples/custom_backend.py path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `import argus` resolve

import numpy as np

from argus import ByteTrackTracker, OpenVocabularyDetector, VideoTracker
from argus.core import Detection, Detector


class MinConfidence:
    """A Detector that wraps another and drops low-confidence detections.

    Implements ``argus.core.Detector`` (``.detect(frame) -> list[Detection]`` + ``.targets``),
    so it slots into ``VideoTracker`` / ``track_video`` anywhere a detector is expected.
    """

    def __init__(self, inner: Detector, min_score: float = 0.5) -> None:
        self.inner = inner
        self.min_score = min_score

    @property
    def targets(self) -> tuple[str, ...]:
        return self.inner.targets

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [d for d in self.inner.detect(frame) if d.score >= self.min_score]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path, help="path to a video clip")
    ap.add_argument("prompt", help="object description to detect")
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--min-score", type=float, default=0.25)
    args = ap.parse_args()

    base = OpenVocabularyDetector(weights="yolov8s-worldv2.pt", prompt=args.prompt, device=args.device)
    detector = MinConfidence(base, min_score=args.min_score)   # <- custom backend, drops in freely
    tracker = ByteTrackTracker(labels=base.model.names, categories=base.model.names)  # open-vocab: labels == categories

    result = VideoTracker(detector, tracker, max_frames=args.max_frames).run(args.video)
    print(f"{args.video.name}: {len(result.track_ids)} tracks (conf >= {args.min_score})")
    print(result.metrics().select("id", "type", "first_s", "last_s", "n_frames", "avg_conf"))


if __name__ == "__main__":
    main()
