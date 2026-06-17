"""Quality gate — decide whether an aligned face is worth embedding, and score it.

Mixed surveillance footage yields many unusable faces (tiny, blurry, extreme pose). The
gate applies cheap proxy metrics: hard-rejects faces below thresholds and emits a composite
[0, 1] quality score for the ones that pass (used to pick the best face per track). v1 is
deliberately proxy-only (no FIQA model); thresholds need calibration on real cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GateResult:
    """Outcome of evaluating one face against the gate."""

    passed: bool
    score: float
    face_px: float
    blur_var: float
    yaw_ratio: float
    det_score: float


def _yaw_ratio(landmarks) -> float:
    """Scale-invariant pose proxy in [0, 1): eye-to-nose asymmetry.

    A frontal face has the nose roughly equidistant from both eyes (~0); a profile shifts
    the nose toward one eye (→ larger). Returns 0 if landmarks are unavailable.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (5, 2):
        return 0.0
    left_eye, right_eye, nose = pts[0], pts[1], pts[2]
    d_left = float(np.linalg.norm(nose - left_eye))
    d_right = float(np.linalg.norm(nose - right_eye))
    total = d_left + d_right
    if total <= 1e-6:
        return 0.0
    return abs(d_left - d_right) / total


@dataclass(frozen=True)
class QualityGate:
    """Proxy-metric face-quality gate.

    Hard rejects anything below the thresholds; otherwise returns a composite score that
    blends face size, sharpness (Laplacian variance), frontality, and detector confidence.
    """

    min_face_px: float = 40.0
    min_blur_var: float = 40.0
    max_yaw_ratio: float = 0.35
    min_det_score: float = 0.5
    # References at which each sub-score saturates to 1.0.
    size_ref_px: float = 112.0
    blur_ref_var: float = 200.0

    def evaluate(
        self, chip: np.ndarray, *, det_score: float, face_px: float, landmarks
    ) -> GateResult:
        """Score one aligned ``chip`` and decide whether it passes the hard rejects."""
        gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY) if chip.ndim == 3 else chip
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        yaw = _yaw_ratio(landmarks)

        passed = (
            face_px >= self.min_face_px
            and blur_var >= self.min_blur_var
            and yaw <= self.max_yaw_ratio
            and det_score >= self.min_det_score
        )

        size_s = min(face_px / self.size_ref_px, 1.0)
        blur_s = min(blur_var / self.blur_ref_var, 1.0)
        pose_s = max(0.0, 1.0 - yaw / self.max_yaw_ratio) if self.max_yaw_ratio > 0 else 1.0
        det_s = min(max(det_score, 0.0), 1.0)
        score = float(np.mean([size_s, blur_s, pose_s, det_s]))

        return GateResult(
            passed=passed,
            score=score,
            face_px=face_px,
            blur_var=blur_var,
            yaw_ratio=yaw,
            det_score=det_score,
        )

    def score(self, chip: np.ndarray, *, det_score: float, face_px: float, landmarks) -> float:
        return self.evaluate(chip, det_score=det_score, face_px=face_px, landmarks=landmarks).score

    def passes(self, chip: np.ndarray, *, det_score: float, face_px: float, landmarks) -> bool:
        return self.evaluate(
            chip, det_score=det_score, face_px=face_px, landmarks=landmarks
        ).passed
