"""faces_cv — computer-vision building blocks for the surveillance face-ID system.

Public SDK surface for video detection + tracking:

    from faces_cv import track_video
    result = track_video("clip.mp4", targets=("person", "vehicle"), device="cuda")
    result.metrics()            # per-track polars DataFrame
    result.render("out.mp4")    # annotated H.264 video
"""

from .detection import Detection, Detector, UltralyticsDetector
from .pipeline import (
    PeekResult,
    TrackingResult,
    VideoTracker,
    peek_video,
    peek_videos,
    track_video,
)
from .tracking import (
    CATEGORY_BY_CLASS,
    COCO_LABELS,
    TARGET_CLASSES,
    ByteTrackTracker,
    Track,
    Tracker,
)

__all__ = [
    # detection
    "Detection",
    "Detector",
    "UltralyticsDetector",
    # tracking
    "Track",
    "Tracker",
    "ByteTrackTracker",
    "COCO_LABELS",
    "CATEGORY_BY_CLASS",
    "TARGET_CLASSES",
    # pipeline
    "VideoTracker",
    "TrackingResult",
    "track_video",
    "peek_video",
    "peek_videos",
    "PeekResult",
]
