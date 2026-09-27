"""녹음 파일 읽기.

매니저 폰이 무엇으로 녹음하든 받아야 하므로, 형식과 샘플레이트를 가리지 않고
16kHz 모노 float32로 맞춰 나오는지 확인한다.

전사 자체는 SenseVoice 모델이 있어야 해서 여기서 검증하지 않는다.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

from voice_ai.transcribe import SAMPLE_RATE, load_audio


def write_wav(path: Path, seconds: float, sample_rate: int, channels: int = 1) -> None:
    frames = int(seconds * sample_rate)
    tone = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, frames)) * 16000).astype(np.int16)
    if channels == 2:
        tone = np.repeat(tone, 2)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(tone.tobytes())


def test_reads_wav_as_float32_mono(tmp_path):
    source = tmp_path / "sample.wav"
    write_wav(source, seconds=1.0, sample_rate=SAMPLE_RATE)

    audio = load_audio(source)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(SAMPLE_RATE, rel=0.02)
    assert np.abs(audio).max() <= 1.0


def test_resamples_to_16k():
    """폰은 보통 48kHz로 녹음한다. 모델은 16kHz만 받는다."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "high.wav"
        write_wav(source, seconds=2.0, sample_rate=48000)

        audio = load_audio(source)

        assert len(audio) == pytest.approx(2 * SAMPLE_RATE, rel=0.02)


def test_downmixes_stereo_to_mono(tmp_path):
    source = tmp_path / "stereo.wav"
    write_wav(source, seconds=1.0, sample_rate=SAMPLE_RATE, channels=2)

    audio = load_audio(source)

    assert audio.ndim == 1
    assert len(audio) == pytest.approx(SAMPLE_RATE, rel=0.02)


def quiet_at(seconds, total=12.0, gap_ms=200):
    """조용한 구간을 심은 잡음 신호."""
    signal = (np.random.rand(int(total * SAMPLE_RATE)).astype(np.float32) - 0.5) * 0.5
    for second in seconds:
        start = int(second * SAMPLE_RATE)
        signal[start : start + int(gap_ms / 1000 * SAMPLE_RATE)] = 0.0
    return signal


def test_short_segment_is_left_alone():
    from voice_ai.transcribe import _split_long

    signal = quiet_at([], total=3.0)

    assert len(_split_long(signal, 5 * SAMPLE_RATE)) == 1


def test_long_segment_is_split_under_the_limit():
    """VAD의 max_speech_duration만 믿을 수 없다. flush가 긴 덩어리를 뱉는다."""
    from voice_ai.transcribe import _split_long

    pieces = _split_long(quiet_at([4, 8]), 5 * SAMPLE_RATE)

    assert len(pieces) == 3
    assert all(len(piece) <= 5 * SAMPLE_RATE for _, piece in pieces)


def test_split_lands_on_a_quiet_point():
    """단어 한가운데를 자르면 양쪽 조각의 전사가 모두 망가진다."""
    from voice_ai.transcribe import _split_long

    pieces = _split_long(quiet_at([4, 8]), 5 * SAMPLE_RATE)
    offsets = [offset / SAMPLE_RATE for offset, _ in pieces]

    assert offsets[1] == pytest.approx(4.0, abs=0.1)
    assert offsets[2] == pytest.approx(8.0, abs=0.1)


def test_pieces_cover_the_whole_segment():
    from voice_ai.transcribe import _split_long

    signal = quiet_at([4, 8])
    pieces = _split_long(signal, 5 * SAMPLE_RATE)

    assert sum(len(piece) for _, piece in pieces) == len(signal)


class _Recorder:
    """설정 객체가 어떤 인자로 만들어졌는지 기록만 한다."""

    def __init__(self, seen: dict, name: str):
        self._seen = seen
        self._name = name

    def __call__(self, **kwargs):
        self._seen[self._name] = kwargs
        return kwargs


def _fake_sherpa_onnx(seen: dict):
    import types

    module = types.ModuleType("sherpa_onnx")
    for name in (
        "OfflineSpeakerSegmentationModelConfig",
        "OfflineSpeakerSegmentationPyannoteModelConfig",
        "SpeakerEmbeddingExtractorConfig",
        "FastClusteringConfig",
    ):
        setattr(module, name, _Recorder(seen, name))

    def diarization_config(**kwargs):
        return types.SimpleNamespace(validate=lambda: True, **kwargs)

    class Diarization:
        def __init__(self, config):
            pass

        def process(self, audio):
            return types.SimpleNamespace(sort_by_start_time=lambda: [])

    module.OfflineSpeakerDiarizationConfig = diarization_config
    module.OfflineSpeakerDiarization = Diarization
    return module


def test_diarization_uses_every_thread_it_was_given(monkeypatch):
    """분할·임베딩 모델의 num_threads 기본값은 1이다.

    넘기지 않으면 코어 하나로 돌아서, 전사보다 화자분리가 더 오래 걸린다.
    실제로 맥에서 99% CPU만 쓰고 있었다.
    """
    import sys

    from voice_ai.transcribe import diarize

    seen: dict = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(seen))

    diarize(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        segmentation_model=Path("segmentation.onnx"),
        embedding_model=Path("embedding.onnx"),
        num_threads=6,
    )

    assert seen["OfflineSpeakerSegmentationModelConfig"]["num_threads"] == 6
    assert seen["SpeakerEmbeddingExtractorConfig"]["num_threads"] == 6
