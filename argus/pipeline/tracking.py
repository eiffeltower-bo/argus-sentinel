"""Video tracking pipeline — decode a video, detect + track, summarise and render.

The user-facing tracking layer: ``track_video`` for the common case and ``VideoTracker``
for custom detector/tracker injection. Both return a ``TrackingResult`` that yields
per-frame tracks, a per-track metrics table, file exports, and an annotated render.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import polars as pl

from ..core import Detector, Track, Tracker, classes_for
from ..detect import UltralyticsDetector
from ..track import ByteTrackTracker

# Distinct BGR colors so each track ID keeps the same color across the clip.
TRACK_PALETTE = [
    (66, 135, 245), (245, 66, 66), (66, 245, 102), (245, 209, 66),
    (200, 66, 245), (66, 245, 230), (245, 138, 66), (138, 66, 245),
    (66, 245, 156), (245, 66, 167), (147, 245, 66), (66, 173, 245),
]


# Explicit schemas so empty results still return a typed, selectable DataFrame.
_TRACKS_SCHEMA = {
    "frame": pl.Int64, "time_s": pl.Float64, "id": pl.Int64,
    "category": pl.Utf8, "label": pl.Utf8, "score": pl.Float64,
    "x1": pl.Float64, "y1": pl.Float64, "x2": pl.Float64, "y2": pl.Float64,
    "area_px": pl.Float64,
}
_METRICS_SCHEMA = {
    "id": pl.Int64, "category": pl.Utf8, "type": pl.Utf8,
    "first_s": pl.Float64, "last_s": pl.Float64, "duration_s": pl.Float64,
    "n_frames": pl.Int64, "continuity": pl.Float64,
    "avg_w": pl.Float64, "avg_h": pl.Float64, "avg_area_px": pl.Float64,
    "avg_area_pct": pl.Float64, "min_area_px": pl.Float64, "max_area_px": pl.Float64,
    "avg_conf": pl.Float64, "min_conf": pl.Float64,
    "entry_edge": pl.Utf8, "exit_edge": pl.Utf8,
    "first_frame": pl.Int64, "last_frame": pl.Int64,
}


def track_color(track_id: int) -> tuple[int, int, int]:
    """Stable per-ID color; recycles only after many tracks."""
    return TRACK_PALETTE[int(track_id) % len(TRACK_PALETTE)]


def edge_of(box, w: int, h: int, frac: float = 0.04) -> str:
    """Which frame border a box touches — where a track entered / left from."""
    x1, y1, x2, y2 = box
    mx, my = w * frac, h * frac
    if x1 <= mx:
        return "left"
    if x2 >= w - mx:
        return "right"
    if y1 <= my:
        return "top"
    if y2 >= h - my:
        return "bottom"
    return "interior"


@dataclass
class TrackingResult:
    """The output of a tracking run: per-frame tracks plus derived views."""

    video_path: Path
    fps: float
    width: int
    height: int
    frames: list[tuple[int, list[Track]]] = field(default_factory=list)

    @property
    def track_ids(self) -> set[int]:
        return {t.track_id for _, tracks in self.frames for t in tracks}

    def tracks_dataframe(self) -> pl.DataFrame:
        """Long table: one row per (frame, track)."""
        rows = []
        for fi, tracks in self.frames:
            for t in tracks:
                rows.append({
                    "frame": fi,
                    "time_s": fi / self.fps,
                    "id": t.track_id,
                    "category": t.category,
                    "label": t.label,
                    "score": t.score,
                    "x1": t.x1, "y1": t.y1, "x2": t.x2, "y2": t.y2,
                    "area_px": (t.x2 - t.x1) * (t.y2 - t.y1),
                })
        return pl.DataFrame(rows, schema=_TRACKS_SCHEMA)

    def metrics(self) -> pl.DataFrame:
        """Per-track aggregation: temporal + size + type + entry/exit edge."""
        stats: dict[int, dict] = {}
        for fi, tracks in self.frames:
            for t in tracks:
                area = (t.x2 - t.x1) * (t.y2 - t.y1)
                box = t.xyxy
                r = stats.get(t.track_id)
                if r is None:
                    r = {"first_frame": fi, "first_box": box, "n": 0,
                         "sum_w": 0.0, "sum_h": 0.0, "sum_area": 0.0,
                         "min_area": area, "max_area": area,
                         "sum_conf": 0.0, "min_conf": t.score, "cls_counts": {}}
                    stats[t.track_id] = r
                r["last_frame"] = fi
                r["last_box"] = box
                r["n"] += 1
                r["sum_w"] += t.x2 - t.x1
                r["sum_h"] += t.y2 - t.y1
                r["sum_area"] += area
                r["min_area"] = min(r["min_area"], area)
                r["max_area"] = max(r["max_area"], area)
                r["sum_conf"] += t.score
                r["min_conf"] = min(r["min_conf"], t.score)
                key = (t.category, t.label)
                r["cls_counts"][key] = r["cls_counts"].get(key, 0) + 1

        frame_area = float(self.width * self.height)
        rows = []
        for tid, r in sorted(stats.items()):
            span = r["last_frame"] - r["first_frame"] + 1
            (dom_cat, dom_label) = max(r["cls_counts"], key=r["cls_counts"].get)
            rows.append({
                "id": tid,
                "category": dom_cat,
                "type": dom_label,
                "first_s": r["first_frame"] / self.fps,
                "last_s": r["last_frame"] / self.fps,
                "duration_s": span / self.fps,
                "n_frames": r["n"],
                "continuity": r["n"] / span,
                "avg_w": r["sum_w"] / r["n"],
                "avg_h": r["sum_h"] / r["n"],
                "avg_area_px": r["sum_area"] / r["n"],
                "avg_area_pct": r["sum_area"] / r["n"] / frame_area * 100,
                "min_area_px": r["min_area"],
                "max_area_px": r["max_area"],
                "avg_conf": r["sum_conf"] / r["n"],
                "min_conf": r["min_conf"],
                "entry_edge": edge_of(r["first_box"], self.width, self.height),
                "exit_edge": edge_of(r["last_box"], self.width, self.height),
                "first_frame": r["first_frame"],
                "last_frame": r["last_frame"],
            })
        return pl.DataFrame(rows, schema=_METRICS_SCHEMA)

    def render(self, output_path, *, display_height: int = 480) -> Path:
        """Write an annotated H.264 video: boxes + ``label id score``, colored per ID.

        Replays the stored per-frame tracks over a re-decode of the source. Writes
        ``mp4v`` then transcodes to H.264 via system ffmpeg (OpenCV wheels here have no
        H.264 encoder, and browsers need H.264).
        """
        output_path = Path(output_path)
        if not self.frames:
            raise RuntimeError("nothing to render: no frames were processed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        by_frame = {fi: tracks for fi, tracks in self.frames}
        last_idx = max(by_frame)

        dh = display_height
        dw = int(round(self.width * dh / self.height)) // 2 * 2
        sx, sy = dw / self.width, dh / self.height

        tmp = output_path.with_name(output_path.stem + "_mp4v.mp4")
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (dw, dh))
        cap = cv2.VideoCapture(str(self.video_path))
        fi = 0
        while fi <= last_idx:
            ok, frame = cap.read()
            if not ok:
                break
            disp = cv2.resize(frame, (dw, dh))
            for t in by_frame.get(fi, []):
                color = track_color(t.track_id)
                p1 = (int(t.x1 * sx), int(t.y1 * sy))
                p2 = (int(t.x2 * sx), int(t.y2 * sy))
                cv2.rectangle(disp, p1, p2, color, 2)
                tag = f"{t.label or '?'} {t.track_id} {t.score:.2f}"
                cv2.putText(disp, tag, (p1[0], p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            vw.write(disp)
            fi += 1
        cap.release()
        vw.release()

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp), "-an",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
            check=True, capture_output=True,
        )
        tmp.unlink(missing_ok=True)
        return output_path

    def to_parquet(self, path, *, what: str = "metrics") -> Path:
        path = Path(path)
        self._frame_for(what).write_parquet(path)
        return path

    def to_csv(self, path, *, what: str = "metrics") -> Path:
        path = Path(path)
        self._frame_for(what).write_csv(path)
        return path

    def to_json(self, path, *, what: str = "metrics") -> Path:
        path = Path(path)
        self._frame_for(what).write_json(path)
        return path

    def _frame_for(self, what: str) -> pl.DataFrame:
        if what == "metrics":
            return self.metrics()
        if what == "tracks":
            return self.tracks_dataframe()
        raise ValueError(f"what must be 'metrics' or 'tracks', got {what!r}")


class VideoTracker:
    """Decode a video and run a (detector, tracker) pair frame-by-frame.

    Both components are pluggable: any ``Detector`` and any ``Tracker`` work. Frames are
    fed in order; the tracker is reset at the start of each ``run`` so results are
    independent across calls.
    """

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        *,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.max_frames = max_frames
        self.stride = max(1, stride)

    def run(self, video_path) -> TrackingResult:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.tracker.reset()
        frames: list[tuple[int, list[Track]]] = []
        read_idx = 0
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if read_idx % self.stride == 0:
                dets = self.detector.detect(frame)
                tracks = self.tracker.update(dets, frame)
                frames.append((read_idx, tracks))
                processed += 1
                if self.max_frames is not None and processed >= self.max_frames:
                    break
            read_idx += 1
        cap.release()
        return TrackingResult(video_path=video_path, fps=fps, width=w, height=h, frames=frames)


def track_video(
    video_path,
    *,
    targets: tuple[str, ...] = ("person", "vehicle"),
    weights: str = "yolo11s.pt",
    conf: float = 0.25,
    device: str | None = None,
    tracker: Tracker | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    peek_first: bool = False,
) -> TrackingResult:
    """One-line detection + tracking on a video path.

    ``targets`` selects COCO class groups (``"person"`` and/or ``"vehicle"``). Builds an
    ``UltralyticsDetector`` and a ``ByteTrackTracker`` (unless ``tracker`` is supplied) and
    runs them through ``VideoTracker``.

    ``peek_first`` runs a fast ``peek_video`` pre-scan and, if the clip isn't interesting,
    short-circuits — returning an empty ``TrackingResult`` (no frames) without the full
    decode + track. Useful for skipping dead clips in batch runs.
    """
    if peek_first:
        from .peek import peek_video

        peek = peek_video(video_path, targets=targets, device=device)
        if not peek.interesting:
            return TrackingResult(
                video_path=Path(video_path),
                fps=peek.fps, width=peek.width, height=peek.height, frames=[],
            )
    detector = UltralyticsDetector(
        weights=weights, classes=classes_for(targets), conf=conf, device=device
    )
    tracker = tracker or ByteTrackTracker()
    return VideoTracker(detector, tracker, max_frames=max_frames, stride=stride).run(video_path)
