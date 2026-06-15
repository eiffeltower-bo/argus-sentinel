"""Face alignment — warp a face to a canonical chip from its 5 landmarks.

A similarity transform (rotation + uniform scale + translation, no shear) maps the
detected 5-point landmarks onto the canonical ArcFace reference template, producing the
fixed-size chip the embedder expects. Implemented with plain OpenCV so alignment carries
no heavy face-SDK dependency and stays unit-testable.

The reference template is the de-facto standard ArcFace 5-point layout for a 112x112 chip
(left eye, right eye, nose tip, left mouth corner, right mouth corner), the same constants
used across insightface/ArcFace pipelines. Scaled proportionally for other ``size`` values.
"""

from __future__ import annotations

import cv2
import numpy as np

# Canonical 5-point landmark positions for a 112x112 aligned chip (x, y), in pixels.
_REFERENCE_112 = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


def reference_landmarks(size: int = 112) -> np.ndarray:
    """The canonical 5-point template scaled to a ``size`` x ``size`` chip."""
    return _REFERENCE_112 * (size / 112.0)


def align_chip(
    image: np.ndarray,
    landmarks,
    *,
    size: int = 112,
) -> np.ndarray:
    """Align a face to a ``size`` x ``size`` BGR chip from its 5 landmarks.

    ``landmarks`` is a sequence of 5 (x, y) points in ``image`` coordinates. Returns the
    warped chip (uint8, H=W=``size``). Raises ``ValueError`` if not given exactly 5 points.
    """
    src = np.asarray(landmarks, dtype=np.float32)
    if src.shape != (5, 2):
        raise ValueError(f"expected 5 (x, y) landmarks, got shape {src.shape}")

    dst = reference_landmarks(size)
    # Partial affine = similarity transform (no shear); robust to single-point noise.
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("could not estimate an alignment transform from the landmarks")
    return cv2.warpAffine(image, matrix, (size, size), borderValue=0.0)
