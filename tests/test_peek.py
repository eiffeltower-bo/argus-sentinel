"""Unit tests for the fast peek_video clip-triage pre-scan.

Exercised with a synthetic video + scripted detector (no model/GPU/weights), so these
run in milliseconds. The ScriptedDetector returns one list per detect() call, in the
order peek samples frames.
"""

from pathlib import Path

from argus.core import Detection
from argus.pipeline import PeekResult, peek_video, peek_videos


def _person(score=0.9):
    return Detection(10, 20, 60, 180, score, class_id=0, label="person", category="person")


def _car(score=0.9):
    return Detection(10, 20, 120, 90, score, class_id=2, label="car", category="vehicle")


class _AlwaysPerson:
    """Stateless, thread-safe fake detector: every frame yields one person.

    No ``detect_batch`` -> exercises the per-frame fallback in ``peek_videos``.
    """

    @property
    def targets(self) -> tuple[str, ...]:
        return ("person",)

    def detect(self, frame):
        return [_person()]


class _AlwaysPersonBatched(_AlwaysPerson):
    """Adds a batched path so ``peek_videos`` uses ``detect_batch``."""

    def detect_batch(self, frames, *, batch_size=32):
        return [[_person()] for _ in frames]


class _PersonIfBright:
    """Content-based fake: a person iff the frame's mean pixel >= 128.

    ``detect`` and ``detect_batch`` agree, so batched results must equal serial ones —
    which only holds if ``detect_batch`` preserves per-frame order across chunks.
    """

    @property
    def targets(self) -> tuple[str, ...]:
        return ("person",)

    def detect(self, frame):
        return [_person()] if frame.mean() >= 128 else []

    def detect_batch(self, frames, *, batch_size=32):
        out = []
        for i in range(0, len(frames), batch_size):
            out.extend(self.detect(f) for f in frames[i : i + batch_size])
        return out


def test_peek_interesting_counts_per_category(make_video, scripted_detector):
    video = make_video(n_frames=10)
    # 10 frames, n_samples=5 -> 5 detect calls; put a person on 3 of them.
    script = [[_person()], [], [_person()], [], [_person()]]
    res = peek_video(
        video, targets=("person",), n_samples=5, min_hits=2, detector=scripted_detector(script)
    )
    assert isinstance(res, PeekResult)
    assert res.n_sampled == 5
    assert res.frames_with_hits == 3
    assert res.counts == {"person": 3}
    assert res.interesting is True


def test_peek_not_interesting_when_empty(make_video, scripted_detector):
    video = make_video(n_frames=10)
    res = peek_video(
        video,
        targets=("person",),
        n_samples=5,
        min_hits=2,
        detector=scripted_detector([[] for _ in range(5)]),
    )
    assert res.frames_with_hits == 0
    assert res.counts == {"person": 0}
    assert res.interesting is False


def test_peek_threshold_boundary(make_video, scripted_detector):
    video = make_video(n_frames=10)
    script = [[_person()], [_person()], [], [], []]  # exactly 2 hits
    at = peek_video(
        video, targets=("person",), n_samples=5, min_hits=2, detector=scripted_detector(script)
    )
    assert at.frames_with_hits == 2 and at.interesting is True

    below = peek_video(
        video, targets=("person",), n_samples=5, min_hits=3, detector=scripted_detector(script)
    )
    assert below.frames_with_hits == 2 and below.interesting is False


def test_peek_samples_capped_at_video_length(make_video, scripted_detector):
    video = make_video(n_frames=4)
    # n_samples larger than the clip -> sample every frame, no more.
    res = peek_video(
        video,
        targets=("person",),
        n_samples=24,
        detector=scripted_detector([[] for _ in range(4)]),
    )
    assert res.total_frames == 4
    assert res.n_sampled == 4


def test_peek_ignores_non_target_category(make_video, scripted_detector):
    video = make_video(n_frames=10)
    # Cars present, but we only peek for people -> no hits.
    script = [[_car()], [_car()], [_car()], [], []]
    res = peek_video(
        video, targets=("person",), n_samples=5, min_hits=1, detector=scripted_detector(script)
    )
    assert res.frames_with_hits == 0
    assert res.counts == {"person": 0}
    assert res.interesting is False


def test_peek_multi_target_counts(make_video, scripted_detector):
    video = make_video(n_frames=10)
    script = [[_person(), _car()], [_car()], [], [_person()], []]
    res = peek_video(
        video,
        targets=("person", "vehicle"),
        n_samples=5,
        min_hits=1,
        detector=scripted_detector(script),
    )
    assert res.counts == {"person": 2, "vehicle": 2}
    assert res.frames_with_hits == 3  # frames 0, 1, 3 had a target hit
    assert res.interesting is True


def test_peek_summary_string(make_video, scripted_detector):
    video = make_video(n_frames=6)
    res = peek_video(
        video,
        targets=("person",),
        n_samples=3,
        min_hits=1,
        detector=scripted_detector([[_person()], [], []]),
    )
    s = res.summary()
    assert "interesting" in s and "person" in s


def test_peek_records_elapsed(make_video, scripted_detector):
    video = make_video(n_frames=6)
    res = peek_video(
        video, targets=("person",), n_samples=3, detector=scripted_detector([[], [], []])
    )
    assert res.elapsed_s >= 0.0


def test_peek_videos_processes_all_clips(make_video):
    # Stateless always-person detector -> deterministic regardless of thread order.
    videos = [make_video(n_frames=8) for _ in range(3)]
    out = peek_videos(
        videos, targets=("person",), min_hits=1, detector=_AlwaysPerson(), max_workers=3
    )
    assert set(out) == {Path(v) for v in videos}
    for res in out.values():
        assert res is not None
        assert res.interesting is True
        assert res.counts["person"] == res.n_sampled  # one person per sampled frame


def test_peek_videos_unreadable_maps_to_none(make_video, tmp_path):
    good = make_video(n_frames=6)
    bad = tmp_path / "empty.mp4"
    bad.write_bytes(b"")  # 0-byte file -> cannot open -> None (not a batch failure)
    out = peek_videos(
        [good, bad], targets=("person",), min_hits=1, detector=_AlwaysPerson(), max_workers=2
    )
    assert out[Path(good)] is not None and out[Path(good)].interesting is True
    assert out[Path(bad)] is None


def test_peek_videos_batched_path(make_video):
    videos = [make_video(n_frames=8) for _ in range(3)]
    out = peek_videos(
        videos,
        targets=("person",),
        min_hits=1,
        detector=_AlwaysPersonBatched(),
        max_workers=3,
        batch_size=4,
    )
    assert set(out) == {Path(v) for v in videos}
    for res in out.values():
        assert res is not None and res.interesting
        assert res.counts["person"] == res.n_sampled


def test_peek_videos_batched_matches_serial(make_video):
    # Content-based detector: batched (chunked) verdict/counts must equal serial peek_video,
    # which only holds if detect_batch keeps per-frame order across chunks.
    video = make_video(n_frames=10)
    serial = peek_video(
        video, targets=("person",), n_samples=5, min_hits=1, detector=_PersonIfBright()
    )
    batched = peek_videos(
        [video],
        targets=("person",),
        n_samples=5,
        min_hits=1,
        detector=_PersonIfBright(),
        batch_size=2,
        sample_width=None,
    )[Path(video)]
    assert batched.counts == serial.counts
    assert batched.frames_with_hits == serial.frames_with_hits
    assert batched.interesting == serial.interesting


def test_peek_videos_falls_back_without_detect_batch(make_video):
    # _AlwaysPerson has no detect_batch -> per-frame detect fallback, still correct.
    video = make_video(n_frames=6)
    res = peek_videos([video], targets=("person",), min_hits=1, detector=_AlwaysPerson())[
        Path(video)
    ]
    assert res is not None and res.interesting
    assert res.counts["person"] == res.n_sampled
