"""Orchestration: video tracking and fast peek triage, plus their result types."""

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
]
