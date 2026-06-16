"""Audio extraction from video via system ffmpeg — a dependency-light helper.

Ported/adapted from github.com/paodanchacon/audio-search (``src/classify_segments.py``) by
Daniela Chambilla.

Reuses argus's ffmpeg-subprocess pattern (see ``pipeline/tracking.py``'s ``render``); ffmpeg is
already a system dependency of argus (rendering transcodes through it), so this adds no new one.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

#: Container suffixes whose audio track must be extracted before classification.
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".ogg"})


def is_video(path) -> bool:
    """Whether ``path``'s suffix is a known video container (so its audio needs extracting)."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def extract_audio(video_path, output_audio_path=None, *, sample_rate: int = 16000) -> Path:
    """Extract a mono WAV (``sample_rate`` Hz, default 16 kHz) from a video via system ffmpeg.

    Writes to ``output_audio_path`` if given, else to a ``NamedTemporaryFile(suffix=".wav")``
    whose path is returned (the caller owns cleanup). Raises ``FileNotFoundError`` if the input
    is missing and ``RuntimeError`` if ffmpeg fails or is not installed.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video file not found: {video_path}")

    if output_audio_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        output_audio_path = Path(tmp.name)
    else:
        output_audio_path = Path(output_audio_path)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-ar", str(sample_rate), "-ac", "1", str(output_audio_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e.stderr}") from e
    except FileNotFoundError as e:  # the ffmpeg binary itself is missing
        raise RuntimeError(
            "ffmpeg executable not found; install ffmpeg and ensure it is on PATH"
        ) from e
    return output_audio_path
