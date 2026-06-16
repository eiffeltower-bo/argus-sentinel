"""Probe image -> face embedding, the shared front of search and enroll.

Mirrors the ingest best-face path: detect faces -> pick the highest-scoring one -> align to a
112x112 chip -> embed. Backends are injectable (tests pass fakes); defaults are lazy so the
heavy face/embed models are only imported when actually used.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..face import align_chip


def image_to_embedding(image, *, face_detector=None, embedder=None, device: str | None = None):
    """Return ``(vec, embedding_space_id, chip)`` for the best face in ``image``.

    ``image`` is a path/str or a BGR ndarray. Raises ``ValueError`` if no face (with 5
    landmarks) is found. ``vec`` is the L2-normalized embedding; ``chip`` is the aligned crop.
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"could not read image: {image}")
    else:
        img = image

    if face_detector is None:
        from ..face import InsightFaceDetector

        face_detector = InsightFaceDetector(device=device)
    if embedder is None:
        from ..embed import InsightFaceEmbedder

        embedder = InsightFaceEmbedder(device=device)

    faces = face_detector.detect(img)
    if not faces:
        raise ValueError("no face detected in probe image")
    face = max(faces, key=lambda f: f.score)
    if len(face.landmarks) != 5:
        raise ValueError("detected face has no 5-point landmarks for alignment")
    chip = align_chip(img, face.landmarks)
    vec = np.asarray(embedder.embed([chip])[0], dtype=np.float32)
    return vec, embedder.embedding_space_id, chip
