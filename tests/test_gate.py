"""Unit tests for the proxy-metric quality gate."""

import numpy as np

from argus.face.gate import QualityGate

# Symmetric (frontal) landmarks for a 112-chip: nose equidistant from both eyes.
_FRONTAL = ((38.0, 51.0), (74.0, 51.0), (56.0, 71.0), (41.0, 92.0), (71.0, 92.0))
# A profile: nose pulled close to the left eye -> high yaw ratio.
_PROFILE = ((38.0, 51.0), (74.0, 51.0), (42.0, 52.0), (41.0, 92.0), (71.0, 92.0))


def _sharp_chip():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)


def test_sharp_frontal_face_passes():
    gate = QualityGate()
    res = gate.evaluate(_sharp_chip(), det_score=0.9, face_px=100, landmarks=_FRONTAL)
    assert res.passed
    assert 0.0 <= res.score <= 1.0


def test_flat_chip_fails_on_blur():
    gate = QualityGate()
    flat = np.zeros((112, 112, 3), dtype=np.uint8)
    res = gate.evaluate(flat, det_score=0.9, face_px=100, landmarks=_FRONTAL)
    assert not res.passed
    assert res.blur_var < gate.min_blur_var


def test_small_face_fails_on_size():
    gate = QualityGate(min_face_px=40)
    assert not gate.passes(_sharp_chip(), det_score=0.9, face_px=20, landmarks=_FRONTAL)


def test_low_detection_score_fails():
    gate = QualityGate(min_det_score=0.5)
    assert not gate.passes(_sharp_chip(), det_score=0.3, face_px=100, landmarks=_FRONTAL)


def test_profile_fails_on_pose():
    gate = QualityGate()
    res = gate.evaluate(_sharp_chip(), det_score=0.9, face_px=100, landmarks=_PROFILE)
    assert res.yaw_ratio > gate.max_yaw_ratio
    assert not res.passed


def test_score_monotonic_in_detection_score():
    gate = QualityGate(min_det_score=0.0)
    chip = _sharp_chip()
    low = gate.score(chip, det_score=0.4, face_px=100, landmarks=_FRONTAL)
    high = gate.score(chip, det_score=0.95, face_px=100, landmarks=_FRONTAL)
    assert high > low
