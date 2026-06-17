"""Orchestration: video tracking, fast peek triage, ingest, and audio, plus their result types."""

from .audio import (
    DEFAULT_AUDIO_MODEL,
    DEFAULT_CANDIDATE_LABELS,
    AudioAnalysis,
    analyze_audio,
)
from .ingest import IngestResult, ingest_video
from .peek import PeekResult, peek_video, peek_videos
from .tracking import TrackingResult, VideoTracker, edge_of, track_color, track_video

__all__ = [
    "VideoTracker",
    "TrackingResult",
    "track_video",
    "track_color",
    "edge_of",
    "PeekResult",
    "peek_video",
    "peek_videos",
    "ingest_video",
    "IngestResult",
    "AudioAnalysis",
    "analyze_audio",
    "DEFAULT_AUDIO_MODEL",
    "DEFAULT_CANDIDATE_LABELS",
]
