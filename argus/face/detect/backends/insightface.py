"""InsightFace SCRFD face-detector backend (implements the core ``FaceDetector`` protocol).

Loads only the detection module of an InsightFace model pack (``buffalo_l`` by default:
SCRFD), so weights auto-download on first use and the heavy import stays lazy. Returns a
box, a score, and the 5 landmarks alignment needs.
"""

from __future__ import annotations

import numpy as np

from ...._onnx import onnx_providers
from ....core import FaceDetection


class InsightFaceDetector:
    """SCRFD face detector behind the ``FaceDetector`` protocol.

    ``device`` follows the SDK convention (``"cuda"``/``"cuda:0"``/``"cpu"``/``None``);
    anything starting with ``cuda`` uses the GPU execution provider. ``det_size`` is the
    SCRFD input resolution; ``min_score`` drops low-confidence detections.
    """

    def __init__(
        self,
        device: str | None = None,
        *,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        min_score: float = 0.5,
    ) -> None:
        from insightface.app import FaceAnalysis  # lazy: heavy import only when used

        providers, ctx_id = onnx_providers(device)
        self._app = FaceAnalysis(
            name=model_name, allowed_modules=["detection"], providers=providers
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.min_score = min_score

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        dets: list[FaceDetection] = []
        for f in self._app.get(image):
            if float(f.det_score) < self.min_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in f.bbox.tolist())
            landmarks = tuple((float(x), float(y)) for x, y in np.asarray(f.kps))
            dets.append(FaceDetection(x1, y1, x2, y2, float(f.det_score), landmarks=landmarks))
        return dets
