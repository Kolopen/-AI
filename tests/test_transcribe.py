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


def _fake_engines(seen: dict):
    import types

    module = types.ModuleType("sherpa_onnx")

    class OfflineRecognizer:
        @staticmethod
        def from_sense_voice(**kwargs):
            seen["sensevoice"] = kwargs

        @staticmethod
        def from_moonshine_v2(**kwargs):
            seen["moonshine"] = kwargs

        @staticmethod
        def from_transducer(**kwargs):
            seen["zipformer"] = kwargs

    module.OfflineRecognizer = OfflineRecognizer
    return module


def _touch(directory: Path, *names: str) -> list[Path]:
    made = []
    for name in names:
        path = directory / name
        path.write_bytes(b"not a real model, only needs to exist")
        made.append(path)
    return made


def test_only_sensevoice_gets_a_language(monkeypatch, tmp_path):
    """SenseVoice는 다국어라 한국어를 못 박아야 한다.

    비워 두면 한국어 진료 녹음이 통째로 중국어로 인식된 적이 있다. 한국어 전용
    모델에는 그 인자가 아예 없으므로 넘기면 깨진다.
    """
    import sys

    from voice_ai.transcribe import build_recognizer

    seen: dict = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_engines(seen))

    model, encoder, decoder, tokens = _touch(
        tmp_path, "m.onnx", "e.ort", "d.ort", "t.txt"
    )
    build_recognizer("sensevoice", model=model, tokens=tokens)
    build_recognizer("moonshine", encoder=encoder, decoder=decoder, tokens=tokens)

    assert seen["sensevoice"]["language"] == "ko"
    assert "language" not in seen["moonshine"]


def test_zipformer_needs_the_joiner(monkeypatch, tmp_path):
    """transducer는 encoder·decoder·joiner 셋이 한 벌이다."""
    import sys

    from voice_ai.transcribe import build_recognizer

    seen: dict = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_engines(seen))

    encoder, decoder, joiner, tokens = _touch(
        tmp_path, "e.onnx", "d.onnx", "j.onnx", "t.txt"
    )
    build_recognizer(
        "zipformer",
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=8,
    )

    assert seen["zipformer"]["joiner"] == str(joiner)
    assert seen["zipformer"]["num_threads"] == 8


def test_unknown_engine_names_the_choices():
    from voice_ai.transcribe import build_recognizer

    with pytest.raises(ValueError, match="zipformer"):
        build_recognizer("paraformer", tokens=Path("t.txt"))


def test_missing_model_file_is_named(tmp_path):
    """onnxruntime은 "Invalid fd was supplied: -1" 만 뱉는다.

    어느 파일이 없는지 말해주지 않아서, 압축이 덜 풀린 걸 알아채기 어려웠다.
    """
    from voice_ai.transcribe import build_recognizer

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("a 0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="encoder_model.ort"):
        build_recognizer(
            "moonshine",
            encoder=tmp_path / "encoder_model.ort",
            decoder=tmp_path / "decoder_model_merged.ort",
            tokens=tokens,
        )


class _EchoRecognizer:
    """받은 오디오 길이를 초 단위로 돌려주는 가짜 인식기."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.lengths: list[int] = []

    def create_stream(self):
        recognizer = self

        class Stream:
            def accept_waveform(self, rate, samples):
                recognizer.lengths.append(len(samples))
                seconds = len(samples) / rate
                self.result = type("R", (), {"text": f"{recognizer.prefix}{seconds:.1f}초"})()

            def __init__(self):
                self.result = type("R", (), {"text": ""})()

        self._stream = Stream()
        return self._stream

    def decode_stream(self, stream):
        pass


def test_long_turns_are_split_before_recognition():
    """moonshine 은 12초 구간에서 예외를 내고 빈 결과를 돌려줬다."""
    from voice_ai.transcribe import transcribe_turns

    audio = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
    recognizer = _EchoRecognizer()

    chunks = transcribe_turns(
        audio, [("speaker_00", 0, 30_000)], recognizer=recognizer, max_chunk_duration=10.0
    )

    assert len(chunks) > 1
    assert max(recognizer.lengths) <= 10 * SAMPLE_RATE
    # 쪼갠 조각의 타임스탬프가 원래 구간 안에서 이어져야 한다.
    assert chunks[0]["start_ms"] == 0
    assert chunks[1]["start_ms"] >= chunks[0]["end_ms"] - 1
    assert chunks[-1]["end_ms"] <= 30_000


def test_sentencepiece_marker_is_stripped():
    """moonshine 출력에 "▁피검사에서는" 처럼 어절 경계 기호가 남는다.

    붙어 있으면 용어 사전이 그 어절을 못 찾는다.
    """
    from voice_ai.transcribe import transcribe_turns

    audio = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)

    chunks = transcribe_turns(
        audio, [("speaker_00", 0, 2_000)], recognizer=_EchoRecognizer(prefix="▁")
    )

    assert chunks[0]["raw_text"] == "2.0초"


def test_non_korean_fragments_are_dropped():
    """1초짜리 맞장구에서 다국어 모델이 일본어를 뱉는다.

    실제 전사에 "ねね。", "や。", "う。" 가 화자 발언으로 들어갔다.
    """
    from voice_ai.transcribe import transcribe_turns

    class _Fixed:
        def __init__(self, text):
            self.text = text

        def create_stream(self):
            outer = self

            class Stream:
                result = _Fixed.Result(outer.text)

                def accept_waveform(self, rate, samples):
                    pass

            return Stream()

        def decode_stream(self, stream):
            pass

        class Result:
            def __init__(self, text):
                self.text = text

    audio = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    turn = [("speaker_00", 0, 2_000)]

    assert transcribe_turns(audio, turn, recognizer=_Fixed("ねね。")) == []
    assert transcribe_turns(audio, turn, recognizer=_Fixed("네네."))[0]["raw_text"] == "네네."
