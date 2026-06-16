"""MCP server exposing argus' triage/track facade as tools (streamable HTTP).

A pure consumer of the public facade — the MCP analogue of ``argus/cli.py``. Four read-mostly
tools let an agent discover footage and run the cheap-then-expensive surveillance workflow
(``list_clips`` -> ``peek_folder``/``peek_clip`` -> ``track_clip``). Run it with:

    argus-mcp --host 0.0.0.0 --port 8000     # serves MCP over HTTP at /mcp

Tool inputs are SERVER-SIDE paths (in the container, footage is mounted read-only at /data).
``device=None`` auto-selects CUDA when GPU torch is installed, so the server inherits the warm
GPU in the CUDA image.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from argus import peek_video, peek_videos, track_video

from ._serialize import peek_to_dict, tracking_to_dict

mcp = FastMCP("argus")

_DEFAULT_TARGETS = ["person", "vehicle"]


@mcp.tool()
def list_clips(directory: str, glob: str = "*.mp4") -> dict[str, Any]:
    """List video clips in a server-side directory without analyzing them.

    Use this first to discover what footage is available before peeking or tracking. Paths are
    server-side/container paths (footage is typically mounted read-only at /data). Returns each
    clip's path and size in bytes.
    """
    base = Path(directory)
    clips = sorted(base.glob(glob))
    return {
        "directory": str(base),
        "glob": glob,
        "n_clips": len(clips),
        "clips": [{"path": str(p), "size_bytes": p.stat().st_size} for p in clips],
    }


@mcp.tool()
def peek_folder(
    directory: str,
    glob: str = "*.mp4",
    targets: list[str] | None = None,
    n_samples: int = 24,
    min_hits: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    """Fast-triage every clip in a folder: which ones contain people/vehicles worth tracking.

    Samples a few frames per clip and runs a small detector in one batched pass — cheap relative
    to ``track_clip``. Run this to narrow a footage dump down to the interesting clips, then
    ``track_clip`` those. Returns counts of interesting/total/unreadable plus a per-clip verdict.
    """
    base = Path(directory)
    clips = sorted(base.glob(glob))
    results = peek_videos(
        clips,
        targets=tuple(targets or _DEFAULT_TARGETS),
        n_samples=n_samples,
        min_hits=min_hits,
        device=device,
    )
    verdicts = [peek_to_dict(r) for _, r in sorted(results.items()) if r is not None]
    unreadable = [str(p) for p, r in sorted(results.items()) if r is None]
    return {
        "directory": str(base),
        "n_clips": len(clips),
        "n_interesting": sum(v["interesting"] for v in verdicts),
        "n_unreadable": len(unreadable),
        "clips": verdicts,
        "unreadable": unreadable,
    }


@mcp.tool()
def peek_clip(
    path: str,
    targets: list[str] | None = None,
    n_samples: int = 24,
    min_hits: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    """Fast-triage a single clip: does it contain target objects (person/vehicle)?

    Samples frames and reports per-category counts, frames-with-hits, and an ``interesting``
    boolean. Much cheaper than ``track_clip`` — use it to decide whether full tracking is worth it.
    """
    r = peek_video(
        Path(path),
        targets=tuple(targets or _DEFAULT_TARGETS),
        n_samples=n_samples,
        min_hits=min_hits,
        device=device,
    )
    return peek_to_dict(r)


@mcp.tool()
def track_clip(
    path: str,
    targets: list[str] | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    render: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Detect and track objects through a single clip, returning per-track metrics.

    Each track row has id, category, type, first_s/last_s, duration_s, n_frames, continuity,
    average size/confidence, and entry/exit edge. This is HEAVY (decodes the whole clip and
    detects every frame) — run ``peek_clip`` first to confirm the clip is interesting, and use
    ``max_frames``/``stride`` to bound work on long clips. Set ``render=true`` to also write an
    annotated H.264 video; its server-side path is returned under ``rendered``.
    """
    r = track_video(
        Path(path),
        targets=tuple(targets or _DEFAULT_TARGETS),
        max_frames=max_frames,
        stride=stride,
        device=device,
    )
    rendered = None
    if render:
        rendered = str(r.render(Path("out") / f"{Path(path).stem}_tracked.mp4"))
    return tracking_to_dict(r, rendered)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="argus-mcp", description="argus MCP server (HTTP)")
    ap.add_argument("--host", default=os.environ.get("ARGUS_MCP_HOST", "0.0.0.0"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("ARGUS_MCP_PORT", "8000"))
    )
    args = ap.parse_args(argv)
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
