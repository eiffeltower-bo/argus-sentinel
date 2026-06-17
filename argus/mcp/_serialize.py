"""Turn argus result objects into JSON-able dicts for MCP structured output.

The only place that knows the result-object shapes; pure functions, unit-testable without a
server. Load-bearing conversions: polars ``DataFrame`` -> ``.to_dicts()`` (a raw frame is not
JSON-serializable), and every ``Path`` -> ``str``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from argus import (
        AudioAnalysis,
        ClusterResult,
        IngestResult,
        Identity,
        PeekResult,
        SearchHit,
        TrackingResult,
    )


def peek_to_dict(r: PeekResult) -> dict:
    """A ``PeekResult`` as a flat JSON-able dict (+ the ``interesting`` verdict and summary)."""
    return {
        "video_path": str(r.video_path),
        "fps": r.fps,
        "width": r.width,
        "height": r.height,
        "total_frames": r.total_frames,
        "n_sampled": r.n_sampled,
        "frames_with_hits": r.frames_with_hits,
        "counts": dict(r.counts),
        "min_hits": r.min_hits,
        "elapsed_s": r.elapsed_s,
        "interesting": r.interesting,
        "summary": r.summary(),
    }


def tracking_to_dict(r: TrackingResult, rendered: str | None) -> dict:
    """A ``TrackingResult`` as a JSON-able dict: counts + per-track metrics rows.

    ``rendered`` is the server-side path of an annotated clip if one was written, else ``None``.
    An empty result yields ``tracks == []`` (metrics() builds from a fixed schema).
    """
    return {
        "video_path": str(r.video_path),
        "fps": r.fps,
        "width": r.width,
        "height": r.height,
        "n_frames": len(r.frames),
        "n_tracks": len(r.track_ids),
        "tracks": r.metrics().to_dicts(),
        "rendered": rendered,
    }


def _hit_to_dict(h: SearchHit) -> dict:
    """One ``SearchHit`` as a JSON-able row: similarity + the matched sighting's evidence."""
    s = h.sighting
    return {
        "sighting_id": s.id,
        "score": h.score,
        "distance": h.distance,
        "camera_id": s.camera_id,
        "ts": s.ts,
        "video_id": s.video_id,
        "track_id": s.track_id,
        "frame_idx": s.frame_idx,
        "bbox": list(s.bbox),
        "quality": s.quality,
        "chip_path": s.chip_path,
        "identity_id": s.identity_id,
        "cluster_id": s.cluster_id,
    }


def search_to_dict(query_ref: str, hits: Sequence[SearchHit]) -> dict:
    """Ranked face-search ``SearchHit``s as a JSON-able dict.

    ``chip_path`` on each hit is the server-side path of the aligned face chip — the evidence an
    operator reviews; results are candidates for human adjudication, never an automated match.
    """
    return {
        "query": str(query_ref),
        "n_hits": len(hits),
        "hits": [_hit_to_dict(h) for h in hits],
    }


def ingest_to_dict(r: IngestResult) -> dict:
    """An ``IngestResult`` as a JSON-able dict: per-run counts + the one-line summary."""
    return {
        "video_id": r.video_id,
        "video_path": str(r.video_path),
        "n_frames": r.n_frames,
        "n_tracks": r.n_tracks,
        "n_faces_detected": r.n_faces_detected,
        "n_gated_out": r.n_gated_out,
        "n_sightings": r.n_sightings,
        "avg_quality": r.avg_quality,
        "summary": r.summary(),
    }


def sightings_to_dict(rows: list[dict]) -> dict:
    """Stored sighting rows (metadata only) as a JSON-able dict.

    ``rows`` are ``store.list_sightings()`` output — plain column dicts (no embedding vector),
    already JSON-able; we just wrap them with a count.
    """
    return {"n": len(rows), "sightings": rows}


def identities_to_dict(idents: Sequence[Identity]) -> dict:
    """Stored ``Identity`` records as a JSON-able dict (dataclass -> dict per row)."""
    return {"n": len(idents), "identities": [dataclasses.asdict(i) for i in idents]}


def cluster_to_dict(r: ClusterResult) -> dict:
    """A ``ClusterResult`` as a JSON-able dict: cluster/noise counts + new identity ids."""
    return {
        "n_sightings": r.n_sightings,
        "n_clusters": r.n_clusters,
        "n_noise": r.n_noise,
        "run_id": r.run_id,
        "identity_ids": list(r.identity_ids),
        "summary": r.summary(),
    }


def audit_to_dict(rows: list[dict]) -> dict:
    """Audit-log rows (already JSON-able column dicts) wrapped with a count."""
    return {"n": len(rows), "rows": rows}


def audio_to_dict(r: AudioAnalysis) -> dict:
    """An ``AudioAnalysis`` as a JSON-able dict: per-segment sound predictions over time.

    Delegates to ``AudioAnalysis.to_dict()`` (Path->str, per-segment top-k ``{class, confidence}``);
    kept here for parity/testability with the other ``*_to_dict`` serializers.
    """
    return r.to_dict()
