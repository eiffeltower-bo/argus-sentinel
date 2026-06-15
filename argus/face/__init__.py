"""Face stage: detect faces on person crops, align them, and gate by quality.

    from argus.face import InsightFaceDetector, align_chip, QualityGate

``FaceDetector`` is the pluggable contract (lives in ``argus.core``); ``align_chip`` and
``QualityGate`` are dependency-light helpers usable on their own.
"""

from ..core import FaceDetector
from .align import align_chip, reference_landmarks
from .detect.backends.insightface import InsightFaceDetector
from .gate import GateResult, QualityGate

__all__ = [
    "FaceDetector",
    "InsightFaceDetector",
    "align_chip",
    "reference_landmarks",
    "QualityGate",
    "GateResult",
]
