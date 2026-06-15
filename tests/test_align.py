"""Unit tests for face alignment (pure cv2 similarity transform, no insightface)."""

import numpy as np
import pytest

from argus.face.align import align_chip, reference_landmarks


def test_align_returns_fixed_size_chip():
    img = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
    lm = [(40, 60), (90, 62), (65, 90), (45, 120), (88, 121)]
    chip = align_chip(img, lm)
    assert chip.shape == (112, 112, 3)
    assert chip.dtype == np.uint8


def test_align_with_reference_landmarks_is_near_identity():
    # Feeding the canonical template back in should map the image to (almost) itself.
    img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    chip = align_chip(img, reference_landmarks(112))
    assert chip.shape == img.shape
    assert np.mean(np.abs(chip.astype(int) - img.astype(int))) < 1.0


def test_align_custom_size():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lm = reference_landmarks(64)
    assert align_chip(img, lm, size=64).shape == (64, 64, 3)


def test_align_rejects_wrong_landmark_count():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        align_chip(img, [(10, 10), (20, 20), (30, 30), (40, 40)])
