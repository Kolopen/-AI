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


def test_whisper_gets_a_longer_chunk_by_default():
    # whisper 는 30초 창으로 돈다. 10초로 끊으면 남는 자리를 묵음으로 채우고도
    # 같은 시간을 쓰므로 문맥만 잘리고 느려진다.
    from voice_ai.transcribe import default_max_chunk

    assert default_max_chunk("whisper") == 30.0
    assert default_max_chunk("sensevoice") == 10.0


def test_hotwords_keep_the_dictionary_order():
    # 가나다순으로 자르면 "가래, 간질, 감기..."만 남고 정작 "치매"가 빠진다.
    # 사전 파일은 중요한 것부터 적혀 있으므로 그 순서를 그대로 쓴다.
    from voice_ai.transcribe import build_hotwords

    words = build_hotwords("신경과")

    assert words[0] == "치매"


def test_hotwords_leave_out_the_common_dictionary():
    # "감기", "기침" 같은 말은 모델이 이미 잘 받아 적는다. 짧은 자리를
    # 그런 말에 내주면 진료과 용어가 밀려난다.
    from voice_ai.transcribe import build_hotwords

    assert "감기" not in build_hotwords("신경과", limit=40)


def test_hotwords_are_capped():
    from voice_ai.transcribe import HOTWORD_LIMIT, build_hotwords

    assert len(build_hotwords("신경과")) == HOTWORD_LIMIT
    assert len(build_hotwords("신경과", limit=5)) == 5


def test_faster_whisper_does_not_need_tokens(monkeypatch, capsys):
    # --tokens 를 필수로 두면 토크나이저를 품은 엔진이 아예 못 돈다.
    from voice_ai import transcribe as module

    monkeypatch.setattr(module, "load_audio", lambda path: None)
    with pytest.raises(SystemExit):
        module.main(["x.m4a", "--engine", "faster-whisper", "--model", "어떤/모델"])

    # usage 줄에는 --tokens 가 나오므로 마지막 error 줄만 본다.
    complaint = capsys.readouterr().err.strip().splitlines()[-1]
    assert "--tokens" not in complaint


def test_sensevoice_still_needs_tokens(capsys, tmp_path):
    from voice_ai import transcribe as module

    audio = tmp_path / "x.m4a"
    audio.touch()
    with pytest.raises(SystemExit):
        module.main([str(audio), "--engine", "sensevoice", "--model", "m.onnx",
                     "--vad", "v.onnx"])

    complaint = capsys.readouterr().err.strip().splitlines()[-1]
    assert "--tokens" in complaint


def test_a_private_model_says_what_to_do():
    # HuggingFace 는 비공개 저장소와 없는 저장소를 똑같이 401 로 돌려준다.
    # 역추적 40줄을 읽어도 무엇을 해야 할지는 안 나온다.
    from voice_ai.transcribe import _model_hint

    hint = _model_hint("나/모델", RuntimeError("401 Client Error. Repository Not Found"))

    assert "비공개" in hint
    assert "hf auth login" in hint


def test_an_unknown_failure_keeps_the_original_message():
    from voice_ai.transcribe import _model_hint

    assert "디스크가 가득" in _model_hint("나/모델", RuntimeError("디스크가 가득 찼습니다"))


def test_whisper_skips_short_chunks():
    # whisper 는 30초 창으로 학습돼서 0.3초를 주면 나머지를 묵음으로 채우고,
    # 디코더가 그 빈자리를 학습 데이터에서 본 문장으로 메운다.
    #
    # faster-whisper 는 모델이 내놓는 신호로 거르므로 길이로는 거의 안 자른다.
    # sherpa 쪽 whisper 는 그 신호를 꺼내 쓸 수 없어 길이로만 막는다.
    from voice_ai.transcribe import default_min_chunk

    assert default_min_chunk("faster-whisper") == 0.6
    assert default_min_chunk("whisper") == 2.0
    assert default_min_chunk("sensevoice") == 0.1


def test_the_model_tells_us_when_it_made_something_up():
    # 길이로 거르면 4.37초짜리 환각을 못 막고 짧고 멀쩡한 질문을 버린다.
    # 모델은 스스로 세 가지 신호를 내놓는다.
    from types import SimpleNamespace

    from voice_ai.transcribe import confident

    def segment(no_speech=0.05, logprob=-0.3, compression=1.4):
        return SimpleNamespace(
            no_speech_prob=no_speech, avg_logprob=logprob, compression_ratio=compression
        )

    assert confident(segment())
    assert not confident(segment(no_speech=0.92))   # 묵음인데 글자가 나왔다
    assert not confident(segment(logprob=-1.6))     # 끌어다 맞췄다
    assert not confident(segment(compression=3.9))  # 같은 말을 되풀이했다


def test_a_phrase_said_three_times_is_a_decoder_loop():
    # 사람은 같은 문장을 세 번 잇달아 말하지 않는다.
    from voice_ai.transcribe import looks_repeated

    assert looks_repeated("연기한다고 연기한다고 연기한다고 발표했다")
    assert looks_repeated("그렇죠? 그렇죠? 그렇죠?")
    assert looks_repeated(
        "홍 사장의 발언에 국감장이 술렁이자 조정식의 발언에 국감장이 술렁이자 "
        "조정식의 발언에 국감장이 술렁이자 조정식의 발언이 됐어요"
    )


def test_an_ordinary_sentence_is_not_a_loop():
    from voice_ai.transcribe import looks_repeated

    assert not looks_repeated("네 서로 다른 날짜일 수 있습니다 접수에서 조정을 확인하고 가세요")
    # 말을 더듬어 한 낱말을 두 번 한 것은 구멍이 아니다.
    assert not looks_repeated("익숙 익숙한 곳에서 길을 잃은 적도 있으세요")


def test_short_turns_never_reach_the_recognizer():
    from voice_ai.transcribe import transcribe_turns

    class Boom:
        def create_stream(self):
            raise AssertionError("짧은 구간이 인식기까지 갔습니다")

    audio = np.zeros(16000 * 10, dtype=np.float32)
    turns = [("speaker_00", 0, 1_500)]

    assert transcribe_turns(audio, turns, recognizer=Boom(), min_chunk_duration=2.0) == []


def test_the_measured_thresholds_separate_the_real_recording():
    # 실제 녹음 여섯 구간의 값. 숫자를 안 보고 정했다가 한 번 틀렸다.
    from types import SimpleNamespace

    from voice_ai.transcribe import confident, looks_repeated

    measured = [
        # 확신도가 -0.37 이라 그 기준은 못 넘는다. 낱말 단위 되풀이가 잡는다.
        ("환각", -0.37, 2.26, "홍 사장의 발언에 국감장이 술렁이자 조정식의 발언에 "
                              "국감장이 술렁이자 조정식의 발언에 국감장이 술렁이자 "
                              "조정식의 발언이 됐어요"),
        ("환각", -0.80, 0.79, "고속도로 교통정보고 좋습니다"),
        ("환각", -0.90, 0.86, "애플의 베풍과 이혼은 새 액체에 밥을 먹었습니다"),
        ("정상", -0.10, 0.62, "감사합니다"),
        ("정상", -0.18, 0.74, "결국 어디서 찾았을까요"),
        ("정상", -0.10, 1.37, "지난 검사 점수는 삼십 점 한 점에 이십 육 점이었습니다"),
    ]
    for kind, logprob, compression, text in measured:
        segment = SimpleNamespace(
            no_speech_prob=0.0, avg_logprob=logprob, compression_ratio=compression
        )
        kept = confident(segment) and not looks_repeated(text)
        assert kept == (kind == "정상"), text[:30]


def test_level_is_the_rms_of_the_samples():
    from voice_ai.transcribe import segment_level

    assert segment_level(np.full(100, 0.5, dtype=np.float32)) == pytest.approx(0.5)
    assert segment_level(np.zeros(100, dtype=np.float32)) == 0.0
    assert segment_level(np.array([], dtype=np.float32)) == 0.0


def test_levels_are_measured_per_diarized_turn():
    """조용한 환자와 또렷한 의사가 숫자로 갈려야 쓸모가 있다."""
    from voice_ai.transcribe import SAMPLE_RATE, measure_levels

    audio = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
    audio[: SAMPLE_RATE] = 0.2
    audio[SAMPLE_RATE : 2 * SAMPLE_RATE] = 0.01
    turns = [("speaker_00", 0, 1000), ("speaker_01", 1000, 2000)]

    levels = measure_levels(audio, turns)

    assert [row[0] for row in levels] == ["speaker_00", "speaker_01"]
    assert levels[0][3] == pytest.approx(0.2)
    assert levels[1][3] == pytest.approx(0.01)


def test_noise_floor_is_measured_in_the_gaps_between_turns():
    """말하는 동안이 아니라 아무도 말하지 않는 틈을 재야 잡음이다."""
    from voice_ai.transcribe import SAMPLE_RATE, noise_floor

    audio = np.full(4 * SAMPLE_RATE, 0.01, dtype=np.float32)
    audio[: SAMPLE_RATE] = 0.3
    audio[2 * SAMPLE_RATE : 3 * SAMPLE_RATE] = 0.3
    turns = [("speaker_00", 0, 1000), ("speaker_01", 2000, 3000)]

    assert noise_floor(audio, turns) == pytest.approx(0.01)


def test_a_chair_scrape_in_one_gap_does_not_raise_the_floor():
    """가운데값을 쓰므로 틈 하나가 시끄러워도 흔들리지 않는다."""
    from voice_ai.transcribe import SAMPLE_RATE, noise_floor

    audio = np.full(6 * SAMPLE_RATE, 0.01, dtype=np.float32)
    audio[3 * SAMPLE_RATE : 4 * SAMPLE_RATE] = 0.5
    turns = [("speaker_00", i * 2000, i * 2000 + 1000) for i in range(3)]

    assert noise_floor(audio, turns) == pytest.approx(0.01)


def test_the_floor_cannot_be_measured_without_gaps():
    from voice_ai.transcribe import SAMPLE_RATE, noise_floor

    audio = np.full(2 * SAMPLE_RATE, 0.05, dtype=np.float32)
    assert noise_floor(audio, [("speaker_00", 0, 2000)]) == 0.0


def test_snr_says_how_far_speech_sits_above_the_noise():
    from voice_ai.transcribe import signal_to_noise

    assert signal_to_noise(0.1, 0.01) == pytest.approx(20.0)
    assert signal_to_noise(0.01, 0.01) == pytest.approx(0.0)
    assert signal_to_noise(0.05, 0.0) is None


def test_the_stft_round_trip_is_exact():
    """잡음을 빼기 전에, 아무것도 안 빼면 원본이 그대로 나와야 한다."""
    from voice_ai.transcribe import _istft, _stft

    for length in (16_000, 500, 3):
        samples = (np.random.default_rng(0).standard_normal(length) * 0.1).astype(np.float32)
        restored = _istft(_stft(samples), length)
        assert len(restored) == length
        assert np.max(np.abs(samples - restored)) < 1e-6


def test_the_noise_profile_follows_the_frequencies_it_was_given():
    """에어컨은 낮은 쪽, 형광등은 높은 쪽에 깔린다. 한 숫자로는 못 걷는다."""
    from voice_ai.transcribe import FRAME_SIZE, SAMPLE_RATE, noise_profile

    t = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
    audio = (np.sin(2 * np.pi * 200 * t) * 0.05).astype(np.float32)
    profile = noise_profile(audio, [("speaker_00", 0, 1000)])

    assert profile is not None
    peak = int(np.argmax(profile))
    assert abs(peak * SAMPLE_RATE / FRAME_SIZE - 200) < SAMPLE_RATE / FRAME_SIZE


def test_there_is_no_profile_without_a_gap():
    from voice_ai.transcribe import SAMPLE_RATE, noise_profile

    audio = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    assert noise_profile(audio, [("speaker_00", 0, 2000)]) is None


def test_reducing_noise_lifts_speech_above_it():
    """실제 녹음은 말소리가 잡음보다 5dB 밖에 크지 않았다. 그걸 벌려야 한다."""
    from voice_ai.transcribe import SAMPLE_RATE, noise_profile, reduce_noise, segment_level

    rng = np.random.default_rng(0)
    t = np.arange(6 * SAMPLE_RATE) / SAMPLE_RATE
    voice = sum(np.sin(2 * np.pi * f * t) for f in (180, 360, 720, 1400)) / 4

    audio = (rng.standard_normal(len(t)) * 0.05).astype(np.float32)
    audio[2 * SAMPLE_RATE : 3 * SAMPLE_RATE] += (voice[:SAMPLE_RATE] * 0.08).astype(np.float32)
    turns = [("speaker_00", 2000, 3000)]

    def margin(samples: np.ndarray) -> float:
        speech = segment_level(samples[2 * SAMPLE_RATE : 3 * SAMPLE_RATE])
        quiet = segment_level(samples[4 * SAMPLE_RATE : 5 * SAMPLE_RATE])
        return 20 * float(np.log10(speech / quiet))

    cleaned = reduce_noise(audio, noise_profile(audio, turns))
    assert margin(cleaned) - margin(audio) > 3.0


def test_some_noise_is_left_behind_on_purpose():
    """깨끗이 0 으로 만들면 금속성 잡소리가 생겨 전사가 오히려 나빠진다."""
    from voice_ai.transcribe import SAMPLE_RATE, noise_profile, reduce_noise, segment_level

    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(3 * SAMPLE_RATE) * 0.05).astype(np.float32)
    cleaned = reduce_noise(audio, noise_profile(audio, [("speaker_00", 0, 1000)]))

    assert segment_level(cleaned[2 * SAMPLE_RATE :]) > 0.0


def test_only_the_manager_turns_are_retagged(monkeypatch, tmp_path):
    """합쳐진 덩어리에서 매니저 구간만 떨어져 나와야 한다.

    여기서 범위를 잘못 잡으면 의사 발언이 매니저 발언으로 기록된다.
    """
    from voice_ai import transcribe, voiceprint

    monkeypatch.setattr(voiceprint, "build_extractor", lambda *a, **k: None)
    monkeypatch.setattr(voiceprint, "manager_turns", lambda *a, **k: [False, True, False])

    print_path = tmp_path / "manager.json"
    voiceprint.Voiceprint(name="김매니저", vector=[1.0, 0.0]).save(print_path)

    turns = [("speaker_00", 0, 2000), ("speaker_00", 2000, 4000), ("speaker_00", 4000, 6000)]
    chunks = [
        {"speaker": "speaker_00", "start_ms": 100, "end_ms": 1900, "text": "앞"},
        {"speaker": "speaker_00", "start_ms": 2100, "end_ms": 3900, "text": "가운데"},
        {"speaker": "speaker_00", "start_ms": 4100, "end_ms": 5900, "text": "뒤"},
    ]

    marked = transcribe._mark_manager(
        np.zeros(6 * SAMPLE_RATE, dtype=np.float32), turns, chunks, print_path, tmp_path, 1
    )

    assert [c["speaker"] for c in marked] == ["speaker_00", "manager", "speaker_00"]
    assert [c.get("is_manager", False) for c in marked] == [False, True, False]


def test_nothing_is_retagged_when_no_turn_matches(monkeypatch, tmp_path):
    """매니저가 말을 거의 안 한 진료도 있다. 억지로 배정하지 않는다."""
    from voice_ai import transcribe, voiceprint

    monkeypatch.setattr(voiceprint, "build_extractor", lambda *a, **k: None)
    monkeypatch.setattr(voiceprint, "manager_turns", lambda *a, **k: [False])

    print_path = tmp_path / "manager.json"
    voiceprint.Voiceprint(name="김매니저", vector=[1.0, 0.0]).save(print_path)

    chunks = [{"speaker": "speaker_00", "start_ms": 0, "end_ms": 2000, "text": "네"}]
    marked = transcribe._mark_manager(
        np.zeros(2 * SAMPLE_RATE, dtype=np.float32),
        [("speaker_00", 0, 2000)],
        chunks,
        print_path,
        tmp_path,
        1,
    )

    assert marked == chunks


def test_an_empty_audio_path_says_so(capsys):
    """셸에서 찾은 경로가 비면 그대로 넘어온다. 빈 경로는 현재 폴더로 읽힌다."""
    from voice_ai.transcribe import main

    with pytest.raises(SystemExit):
        main(["", "--engine", "faster-whisper", "--model", "x",
              "--segmentation", "/tmp/a", "--embedding", "/tmp/b"])

    assert "녹음 파일 경로가 비어 있습니다" in capsys.readouterr().err


def test_a_missing_audio_file_says_so(capsys, tmp_path):
    from voice_ai.transcribe import main

    with pytest.raises(SystemExit):
        main([str(tmp_path / "없음.m4a"), "--engine", "faster-whisper", "--model", "x",
              "--segmentation", "/tmp/a", "--embedding", "/tmp/b"])

    assert "녹음 파일이 없습니다" in capsys.readouterr().err
