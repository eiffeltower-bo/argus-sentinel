"""Core foundation: shared data types, extension protocols, and the COCO taxonomy.

Dependency-free and import-light — everything else in the SDK builds on top of this.
"""

from .protocols import (
    AudioClassifier,
    Detector,
    Embedder,
    FaceDetector,
    SearchableStore,
    Store,
    Tracker,
)
from .taxonomy import CATEGORY_BY_CLASS, COCO_LABELS, TARGET_CLASSES, classes_for
from .types import (
    AudioPrediction,
    AudioSegment,
    Detection,
    Enrollment,
    FaceDetection,
    Identity,
    SearchHit,
    Sighting,
    Track,
    WatchlistHit,
)

__all__ = [
    "Detection",
    "Track",
    "FaceDetection",
    "Sighting",
    "Identity",
    "Enrollment",
    "SearchHit",
    "WatchlistHit",
    "AudioPrediction",
    "AudioSegment",
    "Detector",
    "Tracker",
    "FaceDetector",
    "Embedder",
    "AudioClassifier",
    "Store",
    "SearchableStore",
    "COCO_LABELS",
    "CATEGORY_BY_CLASS",
    "TARGET_CLASSES",
    "classes_for",
]
