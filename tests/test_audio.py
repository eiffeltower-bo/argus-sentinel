"""Audio dimension: windowing + orchestration with a fake classifier (no transformers/soundfile).

The orchestration is exercised through the numpy-native ``AudioClassifier`` protocol + the
``_samples`` decode-bypass seam on ``analyze_audio``, so the bulk of the suite runs with NO heavy
deps. One integration test is guarded by ``importorskip`` + an env gate (it downloads a model).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from argus import AudioAnalysis, analyze_audio, extract_audio
from argus.audio import is_video
from argus.core import AudioPrediction, AudioSegment
from argus.pipeline.audio import _segment_bounds
from conftest import ScriptedAudioClassifier  # noqa: E402 (conftest on pythonpath)

SR = 1000  # tiny synthetic sample rate so n_samples == seconds * 1000


# ---- windowing geometry (_segment_bounds) ---------------------------------------------

def _starts(bounds):
    return [round(b[2], 3) for b in bounds]


def test_segment_bounds_overlap_starts_and_tail():
    bounds = _segment_bounds(12 * SR, SR, segment_seconds=5.0, overlap_seconds=1.0)
    assert _starts(bounds) == [0.0, 4.0, 8.0]          # step = 5 - 1 = 4
    assert [round(b[3], 3) for b in bounds] == [5.0, 9.0, 12.0]
    assert bounds[-1][:2] == (8 * SR, 12 * SR)         # tail runs to the last sample


def test_segment_bounds_short_clip_single_segment():
    bounds = _segment_bounds(3 * SR, SR, segment_seconds=5.0, overlap_seconds=1.0)
    assert bounds == [(0, 3 * SR, 0.0, 3.0)]


def test_segment_bounds_tail_added_when_at_least_one_second():
    bounds = _segment_bounds(int(11.5 * SR), SR, segment_seconds=5.0, overlap_seconds=0.0)
    assert _starts(bounds) == [0.0, 5.0, 10.0]         # 1.5 s remainder -> tail kept
    assert round(bounds[-1][3], 3) == 11.5


def test_segment_bounds_tail_dropped_when_under_one_second():
    bounds = _segment_bounds(int(10.5 * SR), SR, segment_seconds=5.0, overlap_seconds=0.0)
    assert _starts(bounds) == [0.0, 5.0]               # 0.5 s remainder -> dropped
    assert round(bounds[-1][3], 3) == 10.0


def test_segment_bounds_exact_multiple_no_spurious_tail():
    bounds = _segment_bounds(10 * SR, SR, segment_seconds=5.0, overlap_seconds=0.0)
    assert len(bounds) == 2 and round(bounds[-1][3], 3) == 10.0


def test_segment_bounds_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        _segment_bounds(10 * SR, SR, segment_seconds=5.0, overlap_seconds=5.0)


# ---- orchestration (analyze_audio + ScriptedAudioClassifier) --------------------------

def test_analyze_audio_segments_and_feeds_each_window():
    fake = ScriptedAudioClassifier(label="speech")
    samples = np.zeros(12 * SR, dtype=np.float32)
    res = analyze_audio(Path("clip.wav"), classifier=fake, _samples=(samples, SR),
                        segment_seconds=5.0, overlap_seconds=1.0)
    assert isinstance(res, AudioAnalysis)
    assert [s.segment_index for s in res.segments] == [0, 1, 2]
    assert fake.calls == [(5 * SR, SR), (5 * SR, SR), (4 * SR, SR)]   # last is the 8–12 s tail
    assert res.input_duration_seconds == 12.0
    assert res.model_name == "fake_audio_v1"
    assert all(s.top.label == "speech" for s in res.segments)


def test_analyze_audio_skips_under_100_sample_segment():
    fake = ScriptedAudioClassifier()
    res = analyze_audio(Path("blip.wav"), classifier=fake, _samples=(np.zeros(50, np.float32), SR))
    assert res.segments == [] and fake.calls == []     # the lone 50-sample window is skipped


def test_analyze_audio_to_dict_is_jsonable():
    fake = ScriptedAudioClassifier(label="dog_bark", confidence=0.7)
    res = analyze_audio(Path("clip.wav"), classifier=fake, _samples=(np.zeros(12 * SR, np.float32), SR),
                        segment_seconds=5.0, overlap_seconds=1.0)
    d = res.to_dict()
    json.dumps(d)  # raises if not JSON-able
    assert d["input_file"] == "clip.wav" and d["model_name"] == "fake_audio_v1"
    assert len(d["segments"]) == 3
    assert d["segments"][0]["predictions"][0] == {"class": "dog_bark", "confidence": 0.7}


# ---- result views (summary / metrics), built directly ---------------------------------

def _analysis():
    seg = lambda i, lbl: AudioSegment(i, float(i), float(i + 1),  # noqa: E731
                                      (AudioPrediction(lbl, 0.8), AudioPrediction("None", 0.0)))
    return AudioAnalysis(input_file=Path("a.mp4"), audio_path=Path("a.wav"),
                         input_duration_seconds=3.0, model_name="ast-esc50",
                         overlap_seconds=1.0, segment_seconds=5.0,
                         segments=[seg(0, "siren"), seg(1, "siren"), seg(2, "speech")])


def test_summary_reports_dominant_label():
    s = _analysis().summary()
    assert "3 segments" in s and "siren" in s


def test_metrics_one_row_per_segment_prediction():
    df = _analysis().metrics()
    assert df.shape == (6, 6)   # 3 segments x 2 predictions
    assert set(df.columns) == {"segment_index", "start_time", "end_time", "rank", "label",
                               "confidence"}


# ---- extraction helper (no ffmpeg run) ------------------------------------------------

def test_is_video_suffix_table():
    assert is_video("clip.MP4") and is_video("a.mkv")
    assert not is_video("track.wav") and not is_video("notes.txt")


def test_extract_audio_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "nope.mp4")


# ---- optional integration (downloads a model; opt-in) ---------------------------------

@pytest.mark.skipif(not os.environ.get("ARGUS_RUN_MODEL_TESTS"),
                    reason="set ARGUS_RUN_MODEL_TESTS=1 to run (downloads bioamla/ast-esc50)")
def test_huggingface_backend_classifies_a_tone():
    pytest.importorskip("transformers")
    pytest.importorskip("soundfile")
    from argus.audio import HuggingFaceAudioClassifier

    clf = HuggingFaceAudioClassifier(device="cpu")
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    preds = clf.classify(tone, 16000, top_k=2)
    assert len(preds) == 2 and all(isinstance(p, AudioPrediction) for p in preds)
    assert 0.0 <= preds[0].confidence <= 1.0
