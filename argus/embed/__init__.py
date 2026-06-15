"""Face embedding: the ``Embedder`` contract + backends."""

from ..core import Embedder
from .backends.insightface import InsightFaceEmbedder

__all__ = ["Embedder", "InsightFaceEmbedder"]
