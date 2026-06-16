"""Turn argus result objects into JSON-able dicts for MCP structured output.

The only place that knows the result-object shapes; pure functions, unit-testable without a
server. Load-bearing conversions: polars ``DataFrame`` -> ``.to_dicts()`` (a raw frame is not
JSON-serializable), and every ``Path`` -> ``str``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argus import PeekResult, TrackingResult


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
