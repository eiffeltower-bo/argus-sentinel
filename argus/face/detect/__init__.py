"""Face detection: the ``FaceDetector`` contract + backends."""

from ...core import FaceDetector
from .backends.insightface import InsightFaceDetector

__all__ = ["FaceDetector", "InsightFaceDetector"]
