"""argus — on-prem surveillance-footage analysis & understanding.

A model-agnostic, pluggable SDK for making sense of recorded surveillance video. Today it
covers object detection, multi-object tracking, and fast clip triage (peek); the same
extension points are designed to absorb further analysis (face-ID, re-ID, embeddings,
storage sinks). The public surface:

    from argus import track_video, peek_videos
    result = track_video("clip.mp4", targets=("person", "vehicle"), device="cuda")
    result.metrics()                       # per-track polars DataFrame
    result.render("out.mp4")               # annotated H.264 video
    peek_videos(clips, targets=("vehicle",), device="cuda")   # fast folder triage

Architecture (see context/architecture.md):
    core/      shared types + the extension Protocols (Detector, Tracker) + COCO taxonomy
    detect/    Detector backends (detect/backends/ — add ONNX/TensorRT here)
    track/     Tracker backends (track/backends/ — add BoT-SORT here)
    pipeline/  orchestration: tracking + peek

Future drop-in spots (not built yet): a face/ stage (detect->align->embed), a sinks/
package (parquet/pgvector), and pipeline/stages.py for composable multi-stage chains.
"""

from .core import (
    CATEGORY_BY_CLASS,
    COCO_LABELS,
    TARGET_CLASSES,
    Detection,
    Detector,
    Embedder,
    FaceDetection,
    FaceDetector,
    Sighting,
    Store,
    Track,
    Tracker,
)
from .detect import UltralyticsDetector
from .embed import InsightFaceEmbedder
from .face import InsightFaceDetector, QualityGate, align_chip
from .pipeline import (
    IngestResult,
    PeekResult,
    TrackingResult,
    VideoTracker,
    ingest_video,
    peek_video,
    peek_videos,
    track_video,
)
from .store import SqliteStore
from .track import ByteTrackTracker

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
    # face stage
    "FaceDetection",
    "FaceDetector",
    "InsightFaceDetector",
    "align_chip",
    "QualityGate",
    # embedding
    "Embedder",
    "InsightFaceEmbedder",
    # storage
    "Store",
    "Sighting",
    "SqliteStore",
    # pipeline
    "VideoTracker",
    "TrackingResult",
    "track_video",
    "peek_video",
    "peek_videos",
    "PeekResult",
    "ingest_video",
    "IngestResult",
]
