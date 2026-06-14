"""Person detection — a model-agnostic interface plus a YOLO implementation.

This is the Phase-0 seed of the ``PersonDetector`` contract described in
``context/implementation-plan.md``: the pipeline depends on the protocol, not on
any specific model. Swap the backend by providing another class with the same
``.detect(frame)`` signature (e.g. a commercial-clean YOLOX detector later).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single person detection, in absolute pixel xyxy coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@runtime_checkable
class PersonDetector(Protocol):
    """Detect people in a single BGR frame (H x W x 3, uint8)."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YoloPersonDetector:
    """Ultralytics YOLO person detector (COCO class 0 = 'person')."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO  # lazy: heavy import only when used

        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        dets: list[Detection] = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append(Detection(x1, y1, x2, y2, float(b.conf[0])))
        return dets


# --------------------------------------------------------------------------- faces
@dataclass(frozen=True)
class FaceDetection:
    """A single face detection, in absolute pixel xyxy coordinates.

    ``landmarks`` (when available) is 5 (x, y) points — right eye, left eye, nose,
    right mouth corner, left mouth corner — used later for face alignment.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = None
    landmarks: tuple[tuple[float, float], ...] | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@runtime_checkable
class FaceDetector(Protocol):
    """Detect faces in a single BGR image (H x W x 3, uint8)."""

    def detect(self, image: np.ndarray) -> list[FaceDetection]: ...


class HaarFaceDetector:
    """OpenCV Haar-cascade frontal-face detector.

    Zero-dependency stand-in for the plan's SCRFD/YuNet face detector, behind the
    same ``FaceDetector`` interface. Weak on small, rotated, or non-frontal faces —
    fine for a quick two-stage test, swap for SCRFD/YuNet for real accuracy.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 12,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        cv2 = self._cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        return [
            FaceDetection(float(x), float(y), float(x + w), float(y + h))
            for (x, y, w, h) in faces
        ]


class YuNetFaceDetector:
    """OpenCV YuNet face detector (``cv2.FaceDetectorYN``).

    Lightweight (~230 KB ONNX), Apache-2.0, CPU-fast, and far better than Haar on
    small/angled faces. Returns 5 facial landmarks. The plan's commercial-clean
    face detector; same ``FaceDetector`` interface as ``HaarFaceDetector``.

    Model: opencv_zoo ``face_detection_yunet_2023mar.onnx``.
    """

    def __init__(
        self,
        model_path: str = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, nms_threshold, top_k
        )

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []
        self.detector.setInputSize((w, h))  # must match the image before detect()
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
        out: list[FaceDetection] = []
        for f in faces:
            x, y, fw, fh = f[:4]
            landmarks = tuple((float(f[4 + 2 * i]), float(f[5 + 2 * i])) for i in range(5))
            out.append(
                FaceDetection(
                    float(x), float(y), float(x + fw), float(y + fh),
                    score=float(f[14]), landmarks=landmarks,
                )
            )
        return out
