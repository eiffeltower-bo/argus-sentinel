#!/usr/bin/env python
"""Fast-triage every clip in a folder with open-vocabulary detection (YOLO-World).

    uv run python examples/peek_open_vocab.py path/to/folder --prompt helmet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `import argus` resolve

from argus import OpenVocabularyDetector, peek_videos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path, help="folder of video files")
    ap.add_argument("--prompt", required=True, help="text description to detect (e.g. 'helmet')")
    ap.add_argument("--glob", default="*.mp4", help="which files to scan")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt", help="YOLO-World weights")
    ap.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=320, help="inference resolution (square)")
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    ap.add_argument("--workers", type=int, default=8, help="parallel decode workers")
    args = ap.parse_args()

    clips = sorted(args.folder.glob(args.glob))
    if not clips:
        raise SystemExit(f"no files matching {args.glob!r} in {args.folder}")

    detector = OpenVocabularyDetector(
        prompt=args.prompt,
        weights=args.weights,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
    )
    results = peek_videos(
        clips, detector=detector, max_workers=args.workers,
    )
    interesting = {p: r for p, r in results.items() if r and r.interesting}
    unreadable = [p for p, r in results.items() if r is None]

    print(f"{len(interesting)} of {len(clips)} clips contain '{args.prompt}':")
    for p, r in sorted(interesting.items()):
        print(f"  {p.name:42s} {r.summary()}")
    if unreadable:
        print(f"{len(unreadable)} unreadable: " + ", ".join(p.name for p in unreadable))


if __name__ == "__main__":
    main()
