"""Object detection — a model-agnostic interface plus a YOLO implementation.

The pipeline depends on the ``Detector`` protocol, not on any specific model: swap the
backend by providing another class with the same ``.detect(frame)`` signature (e.g. an
ONNX/TensorRT detector for speed, or a commercial-clean model later).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single object detection, in absolute pixel xyxy coordinates.

    ``class_id`` / ``label`` are optional so single-class detectors can emit a bare
    ``Detection(x1, y1, x2, y2, score)``; multi-class detectors fill them in so
    downstream tracking can categorise.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int | None = None
    label: str | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@runtime_checkable
class Detector(Protocol):
    """Detect objects in a single BGR frame (H x W x 3, uint8).

    The model-agnostic contract the tracking pipeline depends on: any object with
    this ``.detect`` signature is a valid backend, ultralytics or otherwise.
    """

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class UltralyticsDetector:
    """Multi-class ultralytics YOLO detector behind the ``Detector`` protocol.

    Each ``Detection`` carries its ``class_id`` and ``label`` (from ``model.names``) so
    the tracker can tag tracks by category. ``classes`` restricts to a COCO subset (e.g.
    ``[0]`` for person, ``[2, 3, 5, 7]`` for vehicles); ``classes=None`` keeps every class.
    """

    def __init__(
        self,
        weights: str = "yolo11s.pt",
        classes: list[int] | None = None,
        conf: float = 0.25,
        device: str | None = None,
        imgsz: int | None = None,
    ) -> None:
        from ultralytics import YOLO  # lazy: heavy import only when used

        self.model = YOLO(weights)
        self.classes = classes
        self.conf = conf
        self.device = device
        # Inference resolution (square). None -> model default (640). Lower (e.g. 320) is
        # faster but misses smaller objects — used by the fast `peek_video` pre-scan.
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            classes=self.classes,
            conf=self.conf,
            device=self.device,
            verbose=False,
            **({} if self.imgsz is None else {"imgsz": self.imgsz}),
        )
        names = self.model.names
        dets: list[Detection] = []
        for r in results:
            dets.extend(self._boxes_to_detections(r, names))
        return dets

    def detect_batch(
        self, frames: list[np.ndarray], *, batch_size: int = 32
    ) -> list[list[Detection]]:
        """Detect on many frames at once — one ``predict`` per ``batch_size`` chunk.

        Returns one ``list[Detection]`` per input frame, aligned to input order. Ultralytics
        runs a whole list as a single forward pass with no internal sub-batching, so we chunk
        to bound VRAM. Far fewer Python-heavy ``predict`` calls than per-frame ``detect``,
        which is the win for the batched :func:`~faces_cv.pipeline.peek_videos`.
        """
        if not frames:
            return []
        names = self.model.names
        out: list[list[Detection]] = []
        for start in range(0, len(frames), max(1, batch_size)):
            chunk = frames[start : start + max(1, batch_size)]
            results = self.model.predict(
                chunk,
                classes=self.classes,
                conf=self.conf,
                device=self.device,
                verbose=False,
                **({} if self.imgsz is None else {"imgsz": self.imgsz}),
            )
            out.extend(self._boxes_to_detections(r, names) for r in results)
        return out

    @staticmethod
    def _boxes_to_detections(result, names) -> list[Detection]:
        dets: list[Detection] = []
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0])
            dets.append(
                Detection(
                    x1, y1, x2, y2, float(b.conf[0]),
                    class_id=cls_id, label=names.get(cls_id),
                )
            )
        return dets
