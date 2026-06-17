"""Thin command-line surface over the argus facade — the container entrypoint.

Subcommands mirror the public functions:

    argus peek  FOLDER  [--targets ... | --prompt TEXT...] [--glob '*.mp4'] [--device cuda] [--workers N] [--json]
    argus track VIDEO   [--targets ... | --prompt TEXT...] [--device cuda] [--max-frames N] [--render OUT] [--json]
    argus audio CLIP    [--model M] [--overlap S] [--segment S] [--labels ...] [--device cuda] [--json]

``--prompt`` swaps the fixed COCO detector for the open-vocabulary YOLO-World detector
(detect one or more free-text classes, e.g. ``--prompt forklift "hard hat"``), overriding
``--targets``.

It is a pure dispatch layer: it imports only the cheap re-exports from ``argus`` (heavy libs
stay lazy inside the backends) and adds no logic of its own, so a future FastAPI/MCP server can
call the same ``peek_videos``/``track_video``/``analyze_audio`` functions. ``--json`` emits one
machine-readable object to stdout (handy for smoke tests and scripting).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from argus import (
    DEFAULT_AUDIO_MODEL,
    OpenVocabularyDetector,
    analyze_audio,
    peek_videos,
    track_video,
)

_TARGETS = ("person", "vehicle")


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--targets",
        nargs="+",
        default=list(_TARGETS),
        choices=list(_TARGETS),
        help="COCO class groups that count as interesting",
    )
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto-detect)")
    ap.add_argument(
        "--prompt",
        nargs="+",
        default=None,
        help="open-vocabulary text class(es) (YOLO-World); overrides --targets",
    )
    ap.add_argument("--json", action="store_true", help="emit one JSON object to stdout")


def _peek(args: argparse.Namespace) -> int:
    clips = sorted(args.folder.glob(args.glob))
    if not clips:
        raise SystemExit(f"no files matching {args.glob!r} in {args.folder}")

    if args.prompt:
        det = OpenVocabularyDetector(args.prompt, device=args.device)
        results = peek_videos(clips, detector=det, max_workers=args.workers)
    else:
        results = peek_videos(
            clips, targets=tuple(args.targets), device=args.device, max_workers=args.workers
        )
    interesting = {p: r for p, r in results.items() if r and r.interesting}
    unreadable = [p for p, r in results.items() if r is None]

    if args.json:
        print(
            json.dumps(
                {
                    "clips": len(clips),
                    "interesting": [
                        {"path": str(p), "counts": r.counts, "summary": r.summary()}
                        for p, r in sorted(interesting.items())
                    ],
                    "unreadable": [str(p) for p in unreadable],
                }
            )
        )
        return 0

    print(f"{len(interesting)} of {len(clips)} clips look interesting:")
    for p, r in sorted(interesting.items()):
        print(f"  {p.name:42s} {r.summary()}")
    if unreadable:
        print(f"{len(unreadable)} unreadable: " + ", ".join(p.name for p in unreadable))
    return 0


def _track(args: argparse.Namespace) -> int:
    if args.prompt:
        det = OpenVocabularyDetector(args.prompt, device=args.device)
        result = track_video(args.video, detector=det, max_frames=args.max_frames)
    else:
        result = track_video(
            args.video,
            targets=tuple(args.targets),
            device=args.device,
            max_frames=args.max_frames,
        )
    rendered = None
    if args.render is not None:
        out = Path(args.render) if args.render else Path("out") / f"{args.video.stem}_tracked.mp4"
        rendered = str(result.render(out))

    if args.json:
        print(
            json.dumps(
                {
                    "video": str(args.video),
                    "n_frames": len(result.frames),
                    "n_tracks": len(result.track_ids),
                    "tracks": result.metrics().to_dicts(),
                    "rendered": rendered,
                }
            )
        )
        return 0

    print(
        f"{args.video.name}: {len(result.frames)} frames · {len(result.track_ids)} distinct tracks"
    )
    print(
        result.metrics().select(
            "id", "category", "type", "first_s", "last_s", "duration_s", "n_frames", "avg_conf"
        )
    )
    if rendered is not None:
        print(f"rendered -> {rendered}")
    return 0


def _audio(args: argparse.Namespace) -> int:
    result = analyze_audio(
        args.clip,
        model=args.model,
        overlap_seconds=args.overlap,
        segment_seconds=args.segment,
        top_k=args.top_k,
        candidate_labels=args.labels,
        device=args.device,
    )
    if args.json:
        print(json.dumps(result.to_dict()))
        return 0

    print(result.summary())
    print(result.metrics())
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
    p_track.add_argument(
        "--render",
        nargs="?",
        const="",
        default=None,
        help="also write an annotated H.264 clip; PATH optional (default: out/<name>_tracked.mp4)",
    )
    _add_common(p_track)
    p_track.set_defaults(func=_track)

    p_audio = sub.add_parser("audio", help="classify the audio track of one clip")
    p_audio.add_argument("clip", type=Path, help="path to an audio or video file")
    p_audio.add_argument(
        "--model",
        default=DEFAULT_AUDIO_MODEL,
        help="HuggingFace audio model (default: zero-shot CLAP)",
    )
    p_audio.add_argument(
        "--overlap",
        type=float,
        default=1.0,
        help="seconds of overlap between adjacent 5s segments",
    )
    p_audio.add_argument("--segment", type=float, default=5.0, help="segment length in seconds")
    p_audio.add_argument("--top-k", type=int, default=2, help="predictions kept per segment")
    p_audio.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="candidate labels for the zero-shot CLAP model "
        "(default: a surveillance-oriented set)",
    )
    p_audio.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    p_audio.add_argument("--json", action="store_true", help="emit one JSON object to stdout")
    p_audio.set_defaults(func=_audio)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
