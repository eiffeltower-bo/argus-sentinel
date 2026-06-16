"""Ultralytics YOLO detector backends (implement the core ``Detector`` protocol)."""

from __future__ import annotations

import numpy as np

from ...core import CATEGORY_BY_CLASS, Detection


def _boxes_to_detections(result, names, categories: dict[int, str] | None = None) -> list[Detection]:
    dets: list[Detection] = []
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls_id = int(b.cls[0])
        dets.append(
            Detection(
                x1, y1, x2, y2, float(b.conf[0]),
                class_id=cls_id, label=names.get(cls_id),
                category=categories.get(cls_id) if categories else None,
            )
        )
    return dets


class UltralyticsDetector:
    """Multi-class ultralytics YOLO detector behind the ``Detector`` protocol.

    Each ``Detection`` carries its ``class_id``, ``label`` (from ``model.names``), and
    ``category`` (coarse COCO roll-up via ``CATEGORY_BY_CLASS``) so the tracker can tag
    tracks by category. ``classes`` restricts to a COCO subset (e.g. ``[0]`` for person,
    ``[2, 3, 5, 7]`` for vehicles); ``classes=None`` keeps every class.
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
        self.imgsz = imgsz

    @property
    def targets(self) -> tuple[str, ...]:
        if self.classes is None:
            return ()
        cats: set[str] = set()
        for c in self.classes:
            cat = CATEGORY_BY_CLASS.get(c)
            if cat:
                cats.add(cat)
        return tuple(sorted(cats))

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
            dets.extend(_boxes_to_detections(r, names, CATEGORY_BY_CLASS))
        return dets

    def detect_batch(
        self, frames: list[np.ndarray], *, batch_size: int = 32
    ) -> list[list[Detection]]:
        """Detect on many frames at once — one ``predict`` per ``batch_size`` chunk.

        Returns one ``list[Detection]`` per input frame, aligned to input order. Ultralytics
        runs a whole list as a single forward pass with no internal sub-batching, so we chunk
        to bound VRAM. Far fewer Python-heavy ``predict`` calls than per-frame ``detect``,
        which is the win for the batched ``peek_videos``.
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
            out.extend(_boxes_to_detections(r, names, CATEGORY_BY_CLASS) for r in results)
        return out


class OpenVocabularyDetector:
    """Open-vocabulary YOLO-World detector behind the ``Detector`` protocol.

    Detects one or more object categories from free-text prompts rather than a fixed taxonomy.
    ``prompt`` is a single class (``"forklift"``) or a list (``["forklift", "hard hat"]``); each
    ``Detection``'s ``label``/``category`` is the matched prompt.

    Pass the detector to ``peek_video`` / ``track_video`` via the ``detector=`` argument.
    """

    def __init__(
        self,
        prompt: str | list[str],
        weights: str = "yolov8s-worldv2.pt",
        conf: float = 0.25,
        device: str | None = None,
        imgsz: int | None = None,
    ) -> None:
        from ultralytics import YOLOWorld  # lazy: heavy import only when used

        self.prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        self.model = YOLOWorld(weights)
        self.model.set_classes(self.prompts)
        self.model.to(device)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self._categories = {i: p for i, p in enumerate(self.prompts)}

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self.prompts)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
            **({} if self.imgsz is None else {"imgsz": self.imgsz}),
        )
        names = self.model.names
        dets: list[Detection] = []
        for r in results:
            dets.extend(_boxes_to_detections(r, names, self._categories))
        return dets

    def detect_batch(
        self, frames: list[np.ndarray], *, batch_size: int = 32
    ) -> list[list[Detection]]:
        """Detect on many frames at once — one ``predict`` per ``batch_size`` chunk.

        Returns one ``list[Detection]`` per input frame, aligned to input order.
        """
        if not frames:
            return []
        names = self.model.names
        categories = self._categories
        out: list[list[Detection]] = []
        for start in range(0, len(frames), max(1, batch_size)):
            chunk = frames[start : start + max(1, batch_size)]
            results = self.model.predict(
                chunk,
                conf=self.conf,
                device=self.device,
                verbose=False,
                **({} if self.imgsz is None else {"imgsz": self.imgsz}),
            )
            out.extend(_boxes_to_detections(r, names, categories) for r in results)
        return out
