"""Thin command-line surface over the argus facade — the container entrypoint.

Two subcommands mirror the public functions:

    argus peek  FOLDER  [--targets ...] [--glob '*.mp4'] [--device cuda] [--workers N] [--json]
    argus track VIDEO   [--targets ...] [--device cuda] [--max-frames N] [--render OUT] [--json]

It is a pure dispatch layer: it imports only the cheap re-exports from ``argus`` (heavy libs
stay lazy inside the backends) and adds no logic of its own, so a future FastAPI/MCP server can
call the same ``peek_videos``/``track_video`` functions. ``--json`` emits one machine-readable
object to stdout (handy for smoke tests and scripting).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from argus import peek_videos, track_video

_TARGETS = ("person", "vehicle")


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--targets", nargs="+", default=list(_TARGETS), choices=list(_TARGETS),
                    help="COCO class groups that count as interesting")
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto-detect)")
    ap.add_argument("--json", action="store_true", help="emit one JSON object to stdout")


def _peek(args: argparse.Namespace) -> int:
    clips = sorted(args.folder.glob(args.glob))
    if not clips:
        raise SystemExit(f"no files matching {args.glob!r} in {args.folder}")

    results = peek_videos(
        clips, targets=tuple(args.targets), device=args.device, max_workers=args.workers
    )
    interesting = {p: r for p, r in results.items() if r and r.interesting}
    unreadable = [p for p, r in results.items() if r is None]

    if args.json:
        print(json.dumps({
            "clips": len(clips),
            "interesting": [
                {"path": str(p), "counts": r.counts, "summary": r.summary()}
                for p, r in sorted(interesting.items())
            ],
            "unreadable": [str(p) for p in unreadable],
        }))
        return 0

    print(f"{len(interesting)} of {len(clips)} clips look interesting:")
    for p, r in sorted(interesting.items()):
        print(f"  {p.name:42s} {r.summary()}")
    if unreadable:
        print(f"{len(unreadable)} unreadable: " + ", ".join(p.name for p in unreadable))
    return 0


def _track(args: argparse.Namespace) -> int:
    result = track_video(
        args.video, targets=tuple(args.targets),
        device=args.device, max_frames=args.max_frames,
    )
    rendered = None
    if args.render is not None:
        rendered = str(result.render(args.render))

    if args.json:
        print(json.dumps({
            "video": str(args.video),
            "n_frames": len(result.frames),
            "n_tracks": len(result.track_ids),
            "tracks": result.metrics().to_dicts(),
            "rendered": rendered,
        }))
        return 0

    print(f"{args.video.name}: {len(result.frames)} frames · {len(result.track_ids)} distinct tracks")
    print(result.metrics().select(
        "id", "category", "type", "first_s", "last_s", "duration_s", "n_frames", "avg_conf"
    ))
    if rendered is not None:
        print(f"rendered -> {rendered}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="argus", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_peek = sub.add_parser("peek", help="fast-triage a folder of clips")
    p_peek.add_argument("folder", type=Path, help="folder of video files")
    p_peek.add_argument("--glob", default="*.mp4", help="which files to scan")
    p_peek.add_argument("--workers", type=int, default=8, help="parallel decode workers")
    _add_common(p_peek)
    p_peek.set_defaults(func=_peek)

    p_track = sub.add_parser("track", help="detect + track one clip")
    p_track.add_argument("video", type=Path, help="path to a video clip")
    p_track.add_argument("--max-frames", type=int, default=None, help="cap frames processed")
    p_track.add_argument("--render", type=Path, default=None,
                         help="also write an annotated H.264 clip to this path")
    _add_common(p_track)
    p_track.set_defaults(func=_track)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
