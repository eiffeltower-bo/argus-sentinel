"""Peek — fast clip triage: is anything worth tracking here, without the full pipeline.

``peek_video`` scans one clip (sample frames -> detect -> tally a verdict); ``peek_videos``
scans many in two phases: parallel decode then one batched inference pass.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..core import Detector, classes_for
from ..detect import UltralyticsDetector
from ._video import _sample_frames


@dataclass
class PeekResult:
    """Verdict from a fast ``peek_video`` pre-scan of a clip."""

    video_path: Path
    fps: float
    width: int
    height: int
    total_frames: int
    n_sampled: int
    frames_with_hits: int
    counts: dict[str, int]  # per-category detection totals across sampled frames
    min_hits: int
    elapsed_s: float = 0.0  # decode time for this clip (excludes detector build / inference)

    @property
    def interesting(self) -> bool:
        """Whether enough sampled frames had a target detection to warrant tracking."""
        return self.frames_with_hits >= self.min_hits

    def summary(self) -> str:
        verdict = "interesting" if self.interesting else "skip"
        by_cat = ", ".join(f"{n} {cat}" for cat, n in self.counts.items())
        return f"{verdict} · {by_cat} in {self.frames_with_hits}/{self.n_sampled} frames"


def _tally(detections, counts: dict[str, int]) -> int:
    """Add per-category detection counts; return 1 if the frame hit a target, else 0."""
    hit = 0
    for d in detections:
        if d.category is not None and d.category in counts:
            counts[d.category] += 1
            hit = 1
    return hit


def _detect_many(detector: Detector, frames: list, batch_size: int) -> list[list]:
    """Detect on many frames, using ``detector.detect_batch`` if available, else per-frame.

    Returns one ``list[Detection]`` per frame, aligned to input order.
    """
    if not frames:
        return []
    batched = getattr(detector, "detect_batch", None)
    if batched is not None:
        return batched(frames, batch_size=batch_size)
    return [detector.detect(f) for f in frames]


def peek_video(
    video_path,
    *,
    targets: tuple[str, ...] | None = None,
    weights: str = "yolo11n.pt",
    conf: float = 0.35,
    imgsz: int = 320,
    device: str | None = None,
    n_samples: int = 24,
    min_hits: int = 2,
    detector: Detector | None = None,
) -> PeekResult:
    """Fast clip triage: sample ``n_samples`` evenly-spaced frames and detect on each.

    A cheap pre-scan to decide whether a clip is worth full tracking. Defaults to a small,
    fast model (``yolo11n``) at reduced inference resolution (``imgsz=320``). Pass a custom
    ``detector`` to plug in another backend (``weights``/``conf``/``imgsz`` are then ignored).

    ``targets`` specify which detection categories to count. When omitted, they are derived
    from ``detector.targets``; if no detector is supplied either, defaults to ``("person",
    "vehicle")``.

    A frame "hits" if it has at least one detection in a target category; the clip is
    ``interesting`` once ``min_hits`` sampled frames hit. ``counts`` totals detections per
    category across all sampled frames.
    """
    import time

    video_path = Path(video_path)
    if detector is None:
        _targets = targets if targets is not None else ("person", "vehicle")
        detector = UltralyticsDetector(
            weights=weights,
            classes=classes_for(_targets),
            conf=conf,
            device=device,
            imgsz=imgsz,
        )
    else:
        _targets = targets if targets is not None else detector.targets

    started = time.perf_counter()
    frames, fps, w, h, total, _decode_s = _sample_frames(
        video_path, n_samples=n_samples, sample_width=None
    )
    counts: dict[str, int] = {t: 0 for t in _targets}
    frames_with_hits = 0
    for _frame in frames:
        frames_with_hits += _tally(detector.detect(_frame), counts)

    return PeekResult(
        video_path=video_path,
        fps=fps,
        width=w,
        height=h,
        total_frames=total,
        n_sampled=len(frames),
        frames_with_hits=frames_with_hits,
        counts=counts,
        min_hits=min_hits,
        elapsed_s=time.perf_counter() - started,
    )


def peek_videos(
    paths,
    *,
    targets: tuple[str, ...] | None = None,
    weights: str = "yolo11n.pt",
    conf: float = 0.35,
    imgsz: int = 320,
    device: str | None = None,
    n_samples: int = 24,
    min_hits: int = 2,
    max_workers: int = 8,
    batch_size: int = 32,
    sample_width: int | None = 320,
    detector: Detector | None = None,
) -> dict[Path, PeekResult | None]:
    """Peek many clips fast and return ``{path: PeekResult | None}``.

    Two phases. **Phase 1** decodes + samples every clip's frames in parallel threads
    (``max_workers``); no detector call happens here. **Phase 2** runs inference on *all*
    sampled frames in one batched pass (``batch_size`` per ``predict``) on the calling
    thread — collapsing the hundreds of Python-heavy per-frame calls that bottlenecked the
    per-clip approach.

    ``targets`` specify which detection categories to count. When omitted, they are derived
    from ``detector.targets``; if no detector is supplied either, defaults to ``("person",
    "vehicle")``.

    Frames are downscaled to ``sample_width`` (longest side) at decode time, which bounds
    host memory (peek needs only category presence, not boxes); pass ``None`` for full-res.
    An unreadable/corrupt clip maps to ``None`` instead of failing the batch. Each result's
    ``elapsed_s`` is that clip's decode time (inference is shared/batched, not per-clip).
    """
    if detector is None:
        _targets = targets if targets is not None else ("person", "vehicle")
        detector = UltralyticsDetector(
            weights=weights,
            classes=classes_for(_targets),
            conf=conf,
            device=device,
            imgsz=imgsz,
        )
    else:
        _targets = targets if targets is not None else detector.targets
    paths = [Path(p) for p in paths]

    # Phase 1: parallel decode + sample (no detector calls -> no lock needed).
    sampled: dict[Path, tuple] = {}
    unreadable: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(_sample_frames, p, n_samples=n_samples, sample_width=sample_width): p
            for p in paths
        }
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                sampled[p] = fut.result()
            except RuntimeError:
                unreadable.append(p)  # unreadable / corrupt clip

    # Phase 2: one flat batched inference pass, then regroup detections per clip.
    order = [p for p in paths if p in sampled]
    flat: list = []
    spans: dict[Path, tuple[int, int]] = {}
    for p in order:
        frames = sampled[p][0]
        spans[p] = (len(flat), len(flat) + len(frames))
        flat.extend(frames)
    detections = _detect_many(detector, flat, batch_size)

    out: dict[Path, PeekResult | None] = {}
    for p in order:
        frames, fps, w, h, total, decode_s = sampled[p]
        start, end = spans[p]
        counts: dict[str, int] = {t: 0 for t in _targets}
        frames_with_hits = 0
        for frame_dets in detections[start:end]:
            frames_with_hits += _tally(frame_dets, counts)
        out[p] = PeekResult(
            video_path=p,
            fps=fps,
            width=w,
            height=h,
            total_frames=total,
            n_sampled=len(frames),
            frames_with_hits=frames_with_hits,
            counts=counts,
            min_hits=min_hits,
            elapsed_s=decode_s,
        )
    for p in unreadable:
        out[p] = None
    return out
