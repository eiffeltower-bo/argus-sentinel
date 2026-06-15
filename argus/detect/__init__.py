"""Object detection: the ``Detector`` contract (from core) + its backends."""

from ..core import Detection, Detector
from .backends.ultralytics import OpenVocabularyDetector, UltralyticsDetector

__all__ = ["Detector", "Detection", "UltralyticsDetector", "OpenVocabularyDetector"]
