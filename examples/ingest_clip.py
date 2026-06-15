#!/usr/bin/env python
"""Run face-ID ingest on one clip: detect→track people, embed the best face per track.

    uv run python examples/ingest_clip.py path/to/clip.mp4 --device cuda

Writes sightings (512-d face vectors + metadata) and aligned face chips into a single
SQLite + sqlite-vec store. Needs the optional backends: `uv sync --group face --group store`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `import argus` resolve

from argus import SqliteStore, ingest_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path, help="path to a video clip")
    ap.add_argument("--camera", default=None, help="camera id (default: clip stem)")
    ap.add_argument("--db", type=Path, default=Path("out/examples/faceid/argus.db"),
                    help="store path (default: out/examples/faceid/argus.db)")
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames processed")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    args = ap.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(args.db)
    result = ingest_video(
        args.video,
        args.camera or args.video.stem,
        store=store,
        device=args.device,
        max_frames=args.max_frames,
        stride=args.stride,
    )

    print(result.summary())
    for s in store.list_sightings():
        print(f"  track {s['track_id']:>3}  frame {s['frame_idx']:>5}  "
              f"t={s['ts']:6.2f}s  q={s['quality']:.3f}  chip={s['chip_path']}")
    print(f"\nstore: {args.db}  ({store.count_vectors()} vectors)")
    store.close()


if __name__ == "__main__":
    main()
