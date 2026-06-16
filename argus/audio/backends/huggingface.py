"""HuggingFace transformers audio-classification backend (implements ``AudioClassifier``).

Ported/adapted from github.com/paodanchacon/audio-search (``src/classify_segments.py``) by
Daniela Chambilla.

Wraps a ``transformers`` audio pipeline: ``audio-classification`` for fixed-label models (e.g.
AST fine-tuned on ESC-50) or ``zero-shot-audio-classification`` for CLAP. The pipeline is loaded
once in ``__init__`` and reused for every segment; heavy imports (torch/transformers) stay lazy
so ``import argus`` and the no-extra test suite are unaffected.
"""

from __future__ import annotations

import numpy as np

from ...core import AudioPrediction


def _resolve_device(device: str | None, torch) -> int | str:
    """Map argus's ``device: str | None`` to what ``transformers.pipeline(device=...)`` wants.

    ``None`` auto-selects cuda:0 -> mps -> cpu. transformers takes an int (``>=0`` cuda ordinal,
    ``-1`` cpu) or a device string; return ``0`` for cuda, ``"mps"``, or ``-1`` for cpu.
    """
    if device is None:
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
        return -1
    d = device.lower()
    if d in ("cpu", "-1"):
        return -1
    if d.startswith("cuda"):
        _, _, ordinal = d.partition(":")
        return int(ordinal) if ordinal else 0
    return device  # e.g. "mps" or an explicit transformers device string


class HuggingFaceAudioClassifier:
    """transformers audio-classification backend behind the ``AudioClassifier`` protocol.

    Picks the task from the model name: ``zero-shot-audio-classification`` when ``"clap"`` is in
    the name (honors ``candidate_labels``), else ``audio-classification`` (fixed labels). Loads
    the pipeline once; ``classify`` runs one segment at a time, keeping top-k and padding to
    ``top_k`` with a ``("None", 0.0)`` prediction so every segment has the same shape.
    """

    def __init__(self, model_name: str = "bioamla/ast-esc50", device: str | None = None) -> None:
        # lazy: heavy imports only when the backend is actually constructed
        import os
        import warnings

        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="The given NumPy array is not writable"
        )
        import torch
        from transformers import pipeline

        self.model_name = model_name
        self.classifier_id = model_name.rsplit("/", 1)[-1]
        self.is_zero_shot = "clap" in model_name.lower()
        task = "zero-shot-audio-classification" if self.is_zero_shot else "audio-classification"
        self._pipe = pipeline(task, model=model_name, device=_resolve_device(device, torch))

    def classify(self, samples, samplerate, *, top_k: int = 2, candidate_labels=None):
        # Feed an in-memory WAV, not a raw array: transformers ffmpeg-decodes it and resamples to
        # the model's sample rate. The zero-shot (CLAP) pipeline rejects the {"array","sampling_rate"}
        # dict and treats a bare ndarray as already-at-model-rate; bytes is correct for both tasks.
        import io

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, np.asarray(samples, dtype=np.float32), int(samplerate), format="WAV")
        wav = buf.getvalue()
        if self.is_zero_shot:
            raw = self._pipe(wav, candidate_labels=candidate_labels or ["sound"])
        else:
            raw = self._pipe(wav, top_k=top_k)
        preds = [AudioPrediction(label=r["label"], confidence=float(r["score"]))
                 for r in raw[:top_k]]
        while len(preds) < top_k:
            preds.append(AudioPrediction(label="None", confidence=0.0))
        return preds
