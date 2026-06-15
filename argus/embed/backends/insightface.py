"""InsightFace ArcFace embedder backend (implements the core ``Embedder`` protocol).

Loads just the recognition model of an InsightFace pack (``buffalo_l`` by default: the
w600k_r50 ArcFace recognizer) and embeds pre-aligned 112x112 chips. ``FaceAnalysis`` can't
be used here — it hard-requires a detection model — so we resolve the recognition ONNX via
the model zoo directly. Output is L2-normalized, so cosine similarity is a dot product.
"""

from __future__ import annotations

import numpy as np

from ..._onnx import onnx_providers


class InsightFaceEmbedder:
    """ArcFace face embedder behind the ``Embedder`` protocol.

    ``embedding_space_id`` namespaces the vector space (see ``Embedder``): vectors from a
    different recognizer are not comparable, so persisted sightings are tagged with it and
    swapping the model means a re-embed under a new id.
    """

    embedding_space_id = "arcface_w600k_r50_v1"
    dim = 512

    def __init__(self, device: str | None = None, *, model_name: str = "buffalo_l") -> None:
        # lazy: heavy imports only when the backend is actually constructed
        import glob
        import os.path as osp

        from insightface import model_zoo
        from insightface.utils.storage import ensure_available

        providers, ctx_id = onnx_providers(device)
        model_dir = ensure_available("models", model_name)
        rec = None
        for onnx_file in sorted(glob.glob(osp.join(model_dir, "*.onnx"))):
            model = model_zoo.get_model(onnx_file, providers=providers)
            if model is not None and getattr(model, "taskname", None) == "recognition":
                rec = model
                break
            del model  # discard the non-recognition models (detection, landmarks, ...)
        if rec is None:
            raise RuntimeError(f"no recognition model found in {model_dir}")
        rec.prepare(ctx_id=ctx_id)
        self._rec = rec

    def embed(self, chips: list[np.ndarray]) -> np.ndarray:
        if not chips:
            return np.zeros((0, self.dim), dtype=np.float32)
        feats = np.asarray(self._rec.get_feat(chips), dtype=np.float32)
        if feats.ndim == 1:  # single chip → (dim,) ; normalize to (1, dim)
            feats = feats[None, :]
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.clip(norms, 1e-12, None)
