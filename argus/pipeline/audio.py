"""Audio classification orchestration: extract -> window -> classify each segment.

Ported/adapted from github.com/paodanchacon/audio-search (``src/classify_segments.py``) by
Daniela Chambilla.

``analyze_audio`` decodes a clip's audio (extracting it from video via ffmpeg when needed),
slices it into overlapping fixed-length windows, and classifies each window with one loaded
``AudioClassifier`` — returning an ``AudioAnalysis``. The model-agnostic windowing math lives in
the pure helper ``_segment_bounds`` (unit-testable with plain ints, no audio decode).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio.extract import extract_audio, is_video
from ..core import AudioClassifier, AudioSegment


@dataclass
class AudioAnalysis:
    """Per-segment audio classification over one clip's audio track."""

    input_file: Path
    audio_path: Path
    input_duration_seconds: float
    model_name: str
    overlap_seconds: float
    segment_seconds: float
    segments: list[AudioSegment] = field(default_factory=list)

    def summary(self) -> str:
        """One-line verdict: duration, segment count, and the most frequent top-1 label."""
        tops = Counter(
            s.top.label for s in self.segments if s.top is not None and s.top.label != "None"
        )
        lead = tops.most_common(1)
        headline = f"{lead[0][0]} (x{lead[0][1]})" if lead else "no predictions"
        return (f"{Path(self.input_file).name}: {self.input_duration_seconds:.1f}s, "
                f"{len(self.segments)} segments · top: {headline}")

    def metrics(self):
        """A polars DataFrame, one row per (segment, prediction rank)."""
        import polars as pl

        rows = [
            {"segment_index": s.segment_index, "start_time": s.start_time,
             "end_time": s.end_time, "rank": rank, "label": p.label, "confidence": p.confidence}
            for s in self.segments
            for rank, p in enumerate(s.predictions)
        ]
        schema = {"segment_index": pl.Int64, "start_time": pl.Float64, "end_time": pl.Float64,
                  "rank": pl.Int64, "label": pl.Utf8, "confidence": pl.Float64}
        return pl.DataFrame(rows, schema=schema)

    def to_dict(self) -> dict:
        """The JSON-able results dict (mirrors classify_segments.py's output shape)."""
        return {
            "input_file": str(self.input_file),
            "audio_path": str(self.audio_path),
            "input_duration_seconds": round(self.input_duration_seconds, 3),
            "model_name": self.model_name,
            "overlap_seconds": self.overlap_seconds,
            "segment_seconds": self.segment_seconds,
            "segments": [
                {"segment_index": s.segment_index,
                 "start_time": round(s.start_time, 2), "end_time": round(s.end_time, 2),
                 "predictions": [{"class": p.label, "confidence": p.confidence}
                                 for p in s.predictions]}
                for s in self.segments
            ],
        }


def _segment_bounds(n_samples, samplerate, *, segment_seconds, overlap_seconds):
    """Geometric window boundaries over ``n_samples`` at ``samplerate``.

    Returns ``[(start_sample, end_sample, start_time, end_time), ...]``. Windows are
    ``segment_seconds`` long stepping by ``segment_seconds - overlap_seconds``; always at least
    one window; a trailing partial window is appended when the remainder is >= 1.0 s. (The
    <100-sample skip is applied by the caller — this is purely geometric.)
    """
    if segment_seconds - overlap_seconds <= 0:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be < segment_seconds ({segment_seconds})"
        )
    duration = n_samples / samplerate if samplerate else 0.0
    step = segment_seconds - overlap_seconds

    bounds: list[tuple[int, int, float, float]] = []
    t = 0.0
    while t + segment_seconds <= duration:
        bounds.append(
            (int(t * samplerate), int((t + segment_seconds) * samplerate), t, t + segment_seconds)
        )
        t += step

    if not bounds:
        bounds.append((0, n_samples, 0.0, duration))
    elif t < duration and duration - t >= 1.0:
        bounds.append((int(t * samplerate), n_samples, t, duration))
    return bounds


def analyze_audio(
    path,
    *,
    classifier: AudioClassifier | None = None,
    model: str = "bioamla/ast-esc50",
    overlap_seconds: float = 1.0,
    segment_seconds: float = 5.0,
    top_k: int = 2,
    candidate_labels: list[str] | None = None,
    device: str | None = None,
    keep_audio: bool = False,
    _samples: tuple | None = None,
) -> AudioAnalysis:
    """Classify the audio of a clip in overlapping windows. Returns an ``AudioAnalysis``.

    ``path`` is an audio or video file (a video's audio is extracted to a temp 16 kHz mono WAV
    first, cleaned up unless ``keep_audio``). Windows are ``segment_seconds`` long stepping by
    ``segment_seconds - overlap_seconds``. If ``classifier`` is None a
    ``HuggingFaceAudioClassifier(model, device=device)`` is built lazily (so the synthetic test
    suite never imports transformers). For CLAP models pass ``candidate_labels``. Needs the
    ``audio`` extra (``transformers`` + ``soundfile``).

    ``_samples`` is a test seam: pass ``(samples_ndarray, samplerate)`` to bypass ffmpeg/soundfile
    decode entirely (the dep-free path).
    """
    path = Path(path)

    tmp_wav: Path | None = None
    if _samples is not None:
        data, samplerate = _samples
        audio_path: Path = path
    else:
        if is_video(path):
            audio_path = extract_audio(path)
            tmp_wav = audio_path
        elif not path.exists():
            raise FileNotFoundError(f"input file not found: {path}")
        else:
            audio_path = path
        import soundfile as sf  # lazy: only the real decode path needs soundfile

        data, samplerate = sf.read(str(audio_path))
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)  # stereo -> mono

    try:
        if classifier is None:
            from ..audio import HuggingFaceAudioClassifier

            classifier = HuggingFaceAudioClassifier(model, device=device)

        data = np.asarray(data)
        bounds = _segment_bounds(
            len(data), samplerate, segment_seconds=segment_seconds, overlap_seconds=overlap_seconds
        )
        segments: list[AudioSegment] = []
        for idx, (s0, s1, t0, t1) in enumerate(bounds):
            seg = data[s0:s1]
            if len(seg) < 100:  # skip extremely short tail segments
                continue
            preds = classifier.classify(
                seg, samplerate, top_k=top_k, candidate_labels=candidate_labels
            )
            segments.append(
                AudioSegment(segment_index=idx, start_time=t0, end_time=t1,
                             predictions=tuple(preds))
            )

        return AudioAnalysis(
            input_file=path,
            audio_path=Path(audio_path),
            input_duration_seconds=len(data) / samplerate if samplerate else 0.0,
            model_name=getattr(classifier, "classifier_id", model),
            overlap_seconds=overlap_seconds,
            segment_seconds=segment_seconds,
            segments=segments,
        )
    finally:
        if tmp_wav is not None and not keep_audio:
            Path(tmp_wav).unlink(missing_ok=True)
