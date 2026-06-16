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

from .audio import HuggingFaceAudioClassifier, extract_audio
from .core import (
    CATEGORY_BY_CLASS,
    COCO_LABELS,
    TARGET_CLASSES,
    AudioClassifier,
    AudioPrediction,
    AudioSegment,
    Detection,
    Detector,
    Embedder,
    Enrollment,
    FaceDetection,
    FaceDetector,
    Identity,
    SearchableStore,
    SearchHit,
    Sighting,
    Store,
    Track,
    Tracker,
    WatchlistHit,
)
from .detect import OpenVocabularyDetector, UltralyticsDetector
from .embed import InsightFaceEmbedder
from .face import InsightFaceDetector, QualityGate, align_chip
from .identity import (
    ClusterResult,
    audit_log,
    enroll,
    export_case,
    label_cluster,
    merge,
    purge,
    reassign,
    run_clustering,
    search_by_image,
    search_by_sighting,
)
from .pipeline import (
    AudioAnalysis,
    IngestResult,
    PeekResult,
    TrackingResult,
    VideoTracker,
    analyze_audio,
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
    "OpenVocabularyDetector",
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
    "SearchableStore",
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
    # identity: search / watchlist / clustering / compliance
    "Identity",
    "Enrollment",
    "SearchHit",
    "WatchlistHit",
    "search_by_image",
    "search_by_sighting",
    "enroll",
    "reassign",
    "merge",
    "label_cluster",
    "run_clustering",
    "ClusterResult",
    "audit_log",
    "purge",
    "export_case",
    # audio
    "AudioClassifier",
    "HuggingFaceAudioClassifier",
    "AudioPrediction",
    "AudioSegment",
    "AudioAnalysis",
    "analyze_audio",
    "extract_audio",
]
