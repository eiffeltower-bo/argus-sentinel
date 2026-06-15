"""Core foundation: shared data types, extension protocols, and the COCO taxonomy.

Dependency-free and import-light — everything else in the SDK builds on top of this.
"""

from .protocols import Detector, Embedder, FaceDetector, Store, Tracker
from .taxonomy import CATEGORY_BY_CLASS, COCO_LABELS, TARGET_CLASSES, classes_for
from .types import Detection, FaceDetection, Sighting, Track

__all__ = [
    "Detection",
    "Track",
    "FaceDetection",
    "Sighting",
    "Detector",
    "Tracker",
    "FaceDetector",
    "Embedder",
    "Store",
    "COCO_LABELS",
    "CATEGORY_BY_CLASS",
    "TARGET_CLASSES",
    "classes_for",
]
