"""Face identity layer: search, watchlist enrollment, clustering, and compliance.

Read/curate side of face-ID, on top of the ingest write-path. Pure orchestration over the
``SearchableStore`` + the face/embed backends; see context/face-id-design.md.
"""

from .admin import audit_log, enroll, export_case, label_cluster, merge, purge, reassign
from .cluster import ClusterResult, run_clustering
from .search import search_by_image, search_by_sighting

__all__ = [
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
]
