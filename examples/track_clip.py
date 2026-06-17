#!/usr/bin/env python
"""Track one clip and print per-track metrics (optionally render an annotated video).

uv run python examples/track_clip.py path/to/clip.mp4 --targets person vehicle --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `import argus` resolve

from argus import track_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path, help="path to a video clip")
    ap.add_argument(
        "--targets",
        nargs="+",
        default=["person", "vehicle"],
        choices=["person", "vehicle"],
        help="what to track",
    )
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames processed")
    ap.add_argument("--render", action="store_true", help="also write an annotated H.264 clip")
    args = ap.parse_args()

    result = track_video(
        args.video,
        targets=tuple(args.targets),
        device=args.device,
        max_frames=args.max_frames,
    )
    print(
        f"{args.video.name}: {len(result.frames)} frames · {len(result.track_ids)} distinct tracks"
    )
    print(
        result.metrics().select(
            "id", "category", "type", "first_s", "last_s", "duration_s", "n_frames", "avg_conf"
        )
    )

    if args.render:
        out = Path("out/examples") / f"{args.video.stem}_tracked.mp4"
        result.render(out)
        print(f"rendered -> {out}")


if __name__ == "__main__":
    main()
