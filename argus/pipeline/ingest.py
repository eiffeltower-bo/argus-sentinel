"""Face-ID ingest pipeline — decode a video into persisted sightings + face chips.

``ingest_video`` reuses the detect→track core, then runs the face stage on each tracked
person crop: detect → align → quality-gate, buffering the best face per track. At end of
stream it batch-embeds one chip per track and writes a sighting (vector + metadata + chip)
to the ``Store``. This is the frame→track "regroup" that makes face-ID different from plain
tracking — we embed the *best* face of a track, so the choice can't be made until the
track has been seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from ..core import Sighting, classes_for
from ..face.align import align_chip
from ..face.gate import QualityGate


@dataclass
class IngestResult:
    """Summary of one ``ingest_video`` run."""

    video_id: int
    video_path: Path
    n_frames: int
    n_tracks: int
    n_faces_detected: int
    n_gated_out: int
    n_sightings: int
    avg_quality: float

    def summary(self) -> str:
        return (
            f"{self.video_path.name}: {self.n_frames} frames, {self.n_tracks} tracks, "
            f"{self.n_faces_detected} faces ({self.n_gated_out} gated out), "
            f"{self.n_sightings} sightings, avg quality {self.avg_quality:.2f}"
        )


def _clamp_box(box, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


def ingest_video(
    video_path,
    camera_id: str,
    *,
    store,
    detector=None,
    tracker=None,
    face_detector=None,
    embedder=None,
    gate: QualityGate | None = None,
    conf: float = 0.25,
    stride: int = 1,
    face_stride: int = 1,
    device: str | None = None,
    max_frames: int | None = None,
) -> IngestResult:
    """Ingest one video: detect→track people, embed the best face per track, persist.

    Components are injectable (for tests / model swaps); any not supplied default to the
    research-weights backends. ``stride`` samples frames for detection; ``face_stride``
    runs the face detector every Nth *processed* frame (faces change slowly across frames).
    Returns an ``IngestResult`` with counts + average quality.
    """
    video_path = Path(video_path)

    # Lazy-construct defaults only when not injected, so tests never import the heavy backends.
    if detector is None:
        from ..detect import UltralyticsDetector

        detector = UltralyticsDetector(classes=classes_for(("person",)), conf=conf, device=device)
    if tracker is None:
        from ..track import ByteTrackTracker

        tracker = ByteTrackTracker()
    if face_detector is None:
        from ..face import InsightFaceDetector

        face_detector = InsightFaceDetector(device=device)
    if embedder is None:
        from ..embed import InsightFaceEmbedder

        embedder = InsightFaceEmbedder(device=device)
    gate = gate or QualityGate()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total / fps if total > 0 else 0.0

    video_id = store.add_video(
        camera_id, str(video_path), fps=fps, duration_s=duration_s, width=width, height=height
    )

    tracker.reset()
    seen_tracks: set[int] = set()
    best: dict[int, dict] = {}  # track_id -> {frame_idx, bbox, quality, chip}
    n_faces_detected = 0
    n_gated_out = 0
    read_idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_idx % max(1, stride) == 0:
            dets = detector.detect(frame)
            tracks = tracker.update(dets, frame)
            for t in tracks:
                seen_tracks.add(t.track_id)
            if processed % max(1, face_stride) == 0:
                for t in tracks:
                    cx1, cy1, cx2, cy2 = _clamp_box(t.xyxy, width, height)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    crop = frame[cy1:cy2, cx1:cx2]
                    faces = face_detector.detect(crop)
                    if not faces:
                        continue
                    n_faces_detected += len(faces)
                    face = max(faces, key=lambda f: f.score)
                    if len(face.landmarks) != 5:
                        n_gated_out += 1
                        continue
                    chip = align_chip(crop, face.landmarks)
                    face_px = min(face.x2 - face.x1, face.y2 - face.y1)
                    res = gate.evaluate(
                        chip, det_score=face.score, face_px=face_px, landmarks=face.landmarks
                    )
                    if not res.passed:
                        n_gated_out += 1
                        continue
                    prev = best.get(t.track_id)
                    if prev is None or res.score > prev["quality"]:
                        best[t.track_id] = {
                            "frame_idx": read_idx,
                            "bbox": (float(cx1 + face.x1), float(cy1 + face.y1),
                                     float(cx1 + face.x2), float(cy1 + face.y2)),
                            "quality": res.score,
                            "chip": chip,
                        }
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
        read_idx += 1
    cap.release()

    # Frame→track regroup: one chip per track, embedded in a single batch.
    track_ids = sorted(best)
    rows: list[Sighting] = []
    if track_ids:
        chips = [best[tid]["chip"] for tid in track_ids]
        embeddings = embedder.embed(chips)
        for tid, emb in zip(track_ids, embeddings):
            b = best[tid]
            chip_path = store.chips_dir / f"v{video_id}_t{tid}_f{b['frame_idx']}.png"
            cv2.imwrite(str(chip_path), b["chip"])
            rows.append(
                Sighting(
                    video_id=video_id,
                    camera_id=camera_id,
                    track_id=tid,
                    frame_idx=b["frame_idx"],
                    ts=b["frame_idx"] / fps,
                    bbox=b["bbox"],
                    quality=b["quality"],
                    chip_path=str(chip_path),
                    embedding_space_id=embedder.embedding_space_id,
                    embedding=emb,
                )
            )
        store.add_sightings(rows)

    avg_quality = sum(r.quality for r in rows) / len(rows) if rows else 0.0
    return IngestResult(
        video_id=video_id,
        video_path=video_path,
        n_frames=processed,
        n_tracks=len(seen_tracks),
        n_faces_detected=n_faces_detected,
        n_gated_out=n_gated_out,
        n_sightings=len(rows),
        avg_quality=avg_quality,
    )
