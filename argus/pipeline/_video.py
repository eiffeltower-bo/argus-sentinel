"""Shared video-decode helpers used by the peek pipeline (and available to tracking).

Sampling decodes sequentially and only fully retrieves every ``stride``-th frame —
``grab()`` skips the rest cheaply. Random-access seeking is *slower* for H.264 (it decodes
forward from the nearest keyframe), and keyframes are too few/uneven to sample on.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2


def _resize_longest(frame, longest: int):
    """Downscale so the frame's longest side is ``longest`` px (no upscaling, keeps aspect)."""
    h, w = frame.shape[:2]
    side = max(h, w)
    if side <= longest:
        return frame
    scale = longest / side
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _sample_frames(video_path, *, n_samples: int, sample_width: int | None = None):
    """Decode a clip and return ~``n_samples`` frames evenly spread across it.

    When ``sample_width`` is set, each kept frame is downscaled to that longest side (peek
    only needs category presence, not box coordinates), which bounds memory.

    Returns ``(frames, fps, width, height, total_frames, decode_s)``. Raises ``RuntimeError``
    if the clip can't be opened.
    """
    video_path = Path(video_path)
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Spread ~n_samples across the clip; if frame count is unknown, sample ~1/sec.
    stride = max(1, total // n_samples) if total > 0 else max(1, round(fps))
    frames: list = []
    read_idx = 0
    while len(frames) < n_samples:
        if not cap.grab():
            break
        if read_idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(_resize_longest(frame, sample_width) if sample_width else frame)
        read_idx += 1
    cap.release()
    return frames, fps, w, h, total, time.perf_counter() - started
