"""Unit tests for the Detection dataclass (no model needed)."""

from faces_cv.detection import Detection


def test_detection_positional_defaults():
    # Back-compat: single-class detectors still construct positionally.
    d = Detection(1.0, 2.0, 3.0, 4.0, 0.9)
    assert d.xyxy == (1.0, 2.0, 3.0, 4.0)
    assert d.score == 0.9
    assert d.class_id is None
    assert d.label is None


def test_detection_with_class():
    d = Detection(1, 2, 3, 4, 0.5, class_id=2, label="car")
    assert d.class_id == 2
    assert d.label == "car"
