"""Audio classification: the ``AudioClassifier`` contract + backends + an extraction helper.

    from argus.audio import HuggingFaceAudioClassifier, extract_audio

``AudioClassifier`` is the pluggable contract (lives in ``argus.core``); ``extract_audio`` is a
dependency-light ffmpeg helper (16 kHz mono WAV) usable on its own. The backend's heavy deps
(``transformers``/``torch``, the ``audio`` extra) load lazily, so importing this package is cheap.
"""

from ..core import AudioClassifier
from .backends.huggingface import HuggingFaceAudioClassifier
from .extract import extract_audio, is_video

__all__ = ["AudioClassifier", "HuggingFaceAudioClassifier", "extract_audio", "is_video"]
