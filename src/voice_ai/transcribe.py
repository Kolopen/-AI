"""녹음 파일을 SenseVoice로 전사해 analyze가 받는 chunks.json을 만든다.

VAD가 긴 녹음을 발화 구간으로 자르고, 잘린 구간마다 SenseVoice를 돌린다.
20분짜리를 통째로 모델에 넣으면 메모리가 감당하지 못하므로 VAD는 선택이 아니다.

오디오 읽기는 PyAV를 쓴다. ffmpeg 라이브러리를 품고 있어서 m4a를 포함해 대부분의
형식을 외부 바이너리 설치 없이 읽는다. 매니저 폰이 무엇으로 녹음하든 받는다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from types import SimpleNamespace

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """어떤 형식이든 16kHz 모노 float32로 읽는다."""
    import av

    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        blocks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                blocks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            blocks.append(resampled.to_ndarray().reshape(-1))

    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks).astype(np.float32) / 32768.0


# SentencePiece 어절 경계 기호. moonshine 출력에 "▁피검사에서는" 처럼 남는다.
_WORD_BOUNDARY = "\u2581"


_HANGUL = re.compile(r"[가-힣]")


def _clean(text: str) -> str:
    """토크나이저 기호를 지우고, 한국어가 아닌 조각은 버린다.

    기호를 붙여 두면 용어 사전이 "▁콜레스테롤"을 못 찾고 리포트에도 그대로 실린다.

    1초 남짓한 맞장구에서는 다국어 모델이 언어를 헷갈려 "ねね。", "や。", "う。"
    같은 일본어를 뱉는다. 한국어 진료 녹음이므로 한글이 하나도 없으면 버린다.
    """
    cleaned = text.replace(_WORD_BOUNDARY, " ").strip()
    return cleaned if _HANGUL.search(cleaned) else ""


def _quietest_point(samples: np.ndarray, target: int, search: int) -> int:
    """target 부근에서 가장 조용한 지점을 찾는다.

    단어 한가운데를 자르면 양쪽 조각의 전사가 모두 망가진다. 숨을 고르는 순간을
    골라 끊으면 손실이 훨씬 적다.
    """
    window = 400  # 25ms
    low = max(window, target - search)
    high = min(len(samples) - window, target + search)
    if high <= low:
        return min(target, len(samples))

    candidates = range(low, high, 160)  # 10ms 간격
    loudness = np.abs(samples)
    return min(candidates, key=lambda c: float(loudness[c - window : c + window].mean()))


def _split_long(samples: np.ndarray, max_samples: int) -> list[tuple[int, np.ndarray]]:
    """길이 상한을 넘는 구간을 조용한 지점에서 끊어 (시작오프셋, 조각) 으로 돌려준다.

    VAD의 max_speech_duration만 믿을 수 없다. 마지막 flush는 남은 버퍼를 길이와
    무관하게 한 구간으로 뱉기 때문에 실제 녹음에서 58초짜리 덩어리가 나왔다.
    """
    if len(samples) <= max_samples:
        return [(0, samples)]

    search = SAMPLE_RATE  # 자를 지점을 앞뒤 1초 안에서 고른다
    pieces: list[tuple[int, np.ndarray]] = []
    offset = 0
    while len(samples) - offset > max_samples:
        cut = _quietest_point(samples, offset + max_samples, search)
        if cut <= offset:
            cut = offset + max_samples
        pieces.append((offset, samples[offset:cut]))
        offset = cut

    pieces.append((offset, samples[offset:]))
    return pieces


# 엔진별로 필요한 파일이 다르다. CLI 검증과 생성이 같은 표를 보게 한다.
ENGINE_FILES = {
    "faster-whisper": ("model",),
    "sensevoice": ("model", "tokens"),
    "moonshine": ("encoder", "decoder", "tokens"),
    "whisper": ("encoder", "decoder", "tokens"),
    "zipformer": ("encoder", "decoder", "joiner", "tokens"),
}

# faster-whisper 의 --model 은 파일이 아니라 폴더이거나 HuggingFace 이름이다.
# 있는지 미리 확인할 수 없으므로 파일 검사에서 뺀다.
PATHLESS_ENGINES = frozenset({"faster-whisper"})

# 구간이 이보다 길면 조용한 자리에서 끊는다. whisper 는 30초 창으로 돌아가서
# 그보다 짧게 끊으면 남는 자리를 묵음으로 채우고도 같은 시간을 쓴다. 10초로
# 끊으면 문맥만 잘리고 세 배 느려진다.
DEFAULT_MAX_CHUNK = {"whisper": 30.0}
FALLBACK_MAX_CHUNK = 10.0


def default_max_chunk(engine: str) -> float:
    return DEFAULT_MAX_CHUNK.get(engine, FALLBACK_MAX_CHUNK)


def _model_hint(model: str, error: Exception) -> str:
    """모델을 못 불러온 까닭을 한 줄로 알려준다.

    HuggingFace 는 비공개 저장소와 없는 저장소를 똑같이 401 로 돌려준다.
    역추적 40줄을 읽어도 무엇을 해야 할지는 안 나온다.
    """
    text = str(error)
    if "401" in text or "RepositoryNotFound" in text or "Repository Not Found" in text:
        return (
            f"모델을 못 불러왔습니다: {model}\n"
            "저장소가 비공개이거나 이름이 틀렸습니다. HuggingFace 는 둘을 똑같이 401 로\n"
            "돌려주므로 구분이 안 됩니다. 셋 중 하나로 푸세요.\n"
            "  1) 저장소를 public 으로 바꾼다\n"
            "  2) hf auth login 으로 읽기 토큰을 넣는다 (예전 이름은 huggingface-cli login)\n"
            "  3) 받아 둔 폴더 경로를 --model 에 그대로 준다"
        )
    if "ctranslate2" in text.lower() or "model.bin" in text:
        return (
            f"모델을 못 불러왔습니다: {model}\n"
            "CTranslate2 형식이 아닌 것 같습니다. faster-whisper 용으로 변환된\n"
            "저장소여야 합니다(폴더 안에 model.bin 이 있습니다)."
        )
    return f"모델을 못 불러왔습니다: {model}\n{text}"


class _Stream:
    """sherpa-onnx 스트림과 같은 모양. 오디오를 받아 두었다가 한 번에 돌린다."""

    def __init__(self) -> None:
        self.audio: np.ndarray | None = None
        self.result = SimpleNamespace(text="")

    def accept_waveform(self, sample_rate: int, audio) -> None:
        self.audio = np.asarray(audio, dtype=np.float32)


class FasterWhisper:
    """faster-whisper 를 sherpa-onnx 인식기와 같은 모양으로 감싼다.

    CTranslate2 형식이라 HuggingFace 의 한국어 파인튜닝 모델을 변환 없이 바로
    불러온다. ONNX 로 바꾸는 과정이 통째로 없어진다.

    `hotwords` 로 진료과 용어를 디코더에 미리 알려줄 수 있다. 실제 녹음에서
    세 엔진이 모두 "치매"를 못 받아 적었는데, 사후 교정으로는 못 살리는
    자리였다. 오인식이 일어나기 전에 막는 쪽이 맞다.
    """

    def __init__(
        self,
        model: str,
        *,
        language: str = "ko",
        num_threads: int = 4,
        hotwords: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(
                model, device="cpu", compute_type="int8", cpu_threads=num_threads
            )
        except Exception as error:  # noqa: BLE001 - 어느 라이브러리가 던질지 모른다
            raise RuntimeError(_model_hint(model, error)) from error
        self.language = language
        self.hotwords = hotwords

    def create_stream(self) -> _Stream:
        return _Stream()

    def decode_stream(self, stream: _Stream) -> None:
        segments, _ = self.model.transcribe(
            stream.audio,
            language=self.language,
            hotwords=self.hotwords,
            beam_size=5,
            # 구간을 따로따로 먹이므로 앞 구간에 기대면 안 된다. 켜 두면 같은
            # 말을 끝없이 되풀이하는 구멍에 빠진다. moonshine 이 "그렇죠?"를
            # 일곱 번 쓴 것이 그 모양이었다.
            condition_on_previous_text=False,
            # 화자분리로 이미 구간을 끊었다.
            vad_filter=False,
        )
        stream.result.text = " ".join(s.text.strip() for s in segments).strip()


def build_recognizer(
    engine: str,
    *,
    model: Path | None = None,
    encoder: Path | None = None,
    decoder: Path | None = None,
    joiner: Path | None = None,
    tokens: Path,
    num_threads: int = 4,
    language: str = "ko",
    hotwords: str | None = None,
):
    """엔진에 맞는 인식기를 만든다.

    SenseVoice는 다국어라 한국어를 못 박아야 한다. 언어를 비워 두면 자동 감지에
    맡기게 되는데, 한국어 진료 녹음이 통째로 중국어·광둥어로 인식된 적이 있다.
    zipformer-korean과 moonshine-tiny-ko는 한국어 전용이라 그 인자가 없다.
    whisper 는 다국어라 SenseVoice 와 같은 이유로 언어를 못 박는다.
    """
    if engine not in ENGINE_FILES:
        raise ValueError(
            f"모르는 엔진입니다: {engine}. {', '.join(ENGINE_FILES)} 중에 고르세요."
        )

    import sherpa_onnx

    given = {"model": model, "encoder": encoder, "decoder": decoder,
             "joiner": joiner, "tokens": tokens}
    # 없는 파일을 넘기면 onnxruntime이 "Invalid fd was supplied: -1" 로 죽는다.
    # 어느 파일인지 알려주지 않아서, 압축이 덜 풀린 것을 알아채는 데 오래 걸린다.
    missing = [
        str(path)
        for name, path in given.items()
        if engine not in PATHLESS_ENGINES
        and name in ENGINE_FILES[engine]
        and path is not None
        and not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "모델 파일이 없습니다: " + ", ".join(missing) +
            "\n압축이 덜 풀렸을 수 있습니다. tar -tf 로 목록을 먼저 확인하세요."
        )

    if engine == "faster-whisper":
        return FasterWhisper(
            str(model), language=language, num_threads=num_threads, hotwords=hotwords
        )
    if engine == "sensevoice":
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(tokens),
            num_threads=num_threads,
            language=language,
            use_itn=True,
        )
    if engine == "moonshine":
        return sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            num_threads=num_threads,
        )
    if engine == "whisper":
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            num_threads=num_threads,
            language=language,
            task="transcribe",
        )
    if engine == "zipformer":
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=num_threads,
        )
    raise AssertionError(f"엔진 분기가 빠졌습니다: {engine}")  # pragma: no cover


def diarize(
    audio: np.ndarray,
    *,
    segmentation_model: Path,
    embedding_model: Path,
    num_speakers: int = -1,
    cluster_threshold: float = 0.5,
    num_threads: int = 4,
) -> list[tuple[str, int, int]]:
    """누가 언제 말했는지 (화자, 시작ms, 끝ms) 로 돌려준다.

    VAD로 기계적으로 끊으면 구간 경계가 화자 전환점과 어긋나 의사 발언 끝에
    환자 응답이 붙는다. 화자분리로 끊으면 구간이 곧 발언권이 된다.
    """
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_model), window_shift_ratio=0.1
            ),
            # 지정하지 않으면 1이다. 분할은 창을 10%씩 밀며 반복 추론해서
            # 전사보다 오래 걸리는데, 코어 하나만 쓰면 그만큼 더 기다린다.
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_model), num_threads=num_threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=cluster_threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("화자분리 설정이 올바르지 않습니다. 모델 경로를 확인하세요.")

    result = sherpa_onnx.OfflineSpeakerDiarization(config).process(audio).sort_by_start_time()
    return [
        (f"speaker_{turn.speaker:02d}", round(turn.start * 1000), round(turn.end * 1000))
        for turn in result
    ]


def transcribe_turns(
    audio: np.ndarray,
    turns: list[tuple[str, int, int]],
    *,
    recognizer,
    max_chunk_duration: float = 10.0,
) -> list[dict]:
    """화자분리가 잡아준 구간마다 전사한다. 구간이 곧 한 사람의 발언이다.

    한 사람이 길게 말하면 구간도 그만큼 길어진다. moonshine 은 12초짜리 구간에서
    onnxruntime 예외를 내고 빈 결과를 돌려줬고, SenseVoice 는 죽지는 않지만 긴
    구간일수록 "신장"을 "심장"으로 쓰는 식으로 정확도가 떨어졌다. 상한을 두고
    조용한 지점에서 끊는다.
    """
    chunks: list[dict] = []
    max_samples = int(max_chunk_duration * SAMPLE_RATE)

    for speaker, start_ms, end_ms in turns:
        piece = audio[int(start_ms * SAMPLE_RATE / 1000) : int(end_ms * SAMPLE_RATE / 1000)]
        if len(piece) < SAMPLE_RATE // 10:  # 0.1초 미만은 버린다
            continue

        for offset, part in _split_long(piece, max_samples):
            if len(part) < SAMPLE_RATE // 10:
                continue
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, part)
            recognizer.decode_stream(stream)

            raw = _clean(stream.result.text)
            if not raw:
                continue

            part_start = start_ms + round(offset * 1000 / SAMPLE_RATE)
            chunks.append(
                {
                    "index": len(chunks),
                    "speaker": speaker,
                    "start_ms": part_start,
                    "end_ms": part_start + round(len(part) * 1000 / SAMPLE_RATE),
                    "raw_text": raw,
                }
            )
    return chunks


def transcribe(
    audio: np.ndarray,
    *,
    recognizer,
    vad_model: Path,
    min_silence_duration: float = 0.25,
    max_speech_duration: float = 10.0,
) -> list[dict]:
    """VAD로 자르고 구간마다 SenseVoice를 돌려 chunks를 만든다.

    구간 길이가 역할 판정의 해상도를 정한다. SenseVoice 한국어 출력에는 문장 부호가
    거의 붙지 않아 문장 단위로 쪼갤 수가 없고, VAD 구간이 곧 판정 단위가 된다.
    진료 대화는 의사가 길게 말하는 중간에 환자가 짧게 끼어들어 자연스러운 침묵이
    잘 생기지 않으므로, 길이 상한으로 강제로 끊어야 화자가 섞이지 않는다.
    """
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_model)
    config.silero_vad.min_silence_duration = min_silence_duration
    config.silero_vad.max_speech_duration = max_speech_duration
    config.sample_rate = SAMPLE_RATE
    window = config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)

    chunks: list[dict] = []

    max_samples = int(max_speech_duration * SAMPLE_RATE)

    def drain() -> None:
        while not vad.empty():
            segment = vad.front
            samples = np.asarray(segment.samples, dtype=np.float32)
            base_ms = round(segment.start * 1000 / SAMPLE_RATE)
            vad.pop()

            for offset, piece in _split_long(samples, max_samples):
                stream = recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, piece)
                recognizer.decode_stream(stream)

                raw = _clean(stream.result.text)
                if not raw:
                    continue

                start_ms = base_ms + round(offset * 1000 / SAMPLE_RATE)
                chunks.append(
                    {
                        "index": len(chunks),
                        "start_ms": start_ms,
                        "end_ms": start_ms + round(len(piece) * 1000 / SAMPLE_RATE),
                        "raw_text": raw,
                    }
                )

    # accept_waveform은 정확히 window_size 만큼을 받는다. 마지막 자투리는 0으로 채운다.
    for offset in range(0, len(audio), window):
        block = audio[offset : offset + window]
        if len(block) < window:
            block = np.pad(block, (0, window - len(block)))
        vad.accept_waveform(block)
        drain()

    vad.flush()
    drain()
    return chunks


# 디코더 프롬프트에 들어갈 수 있는 길이가 정해져 있다(whisper 는 448 토큰이고
# faster-whisper 가 그 절반까지만 쓴다). 넘치면 뒤에서부터 잘려 나가므로
# 중요한 것을 앞에 둔다. 질환과 검사가 먼저다. 리포트의 진단란에 들어가는
# 말이고, 틀리면 없는 병이 기록에 남는다. 약 이름은 사후 교정으로도 어느
# 정도 잡힌다.
HOTWORD_ORDER = ("conditions", "tests", "drugs")

# 한국어 한 낱말이 토큰 두세 개를 먹으므로 이 언저리가 상한이다. 정확한
# 토큰 수는 모델마다 다르고, 넘친 만큼은 faster-whisper 가 알아서 자른다.
HOTWORD_LIMIT = 60


def build_hotwords(department: str, *, limit: int = HOTWORD_LIMIT) -> list[str]:
    """진료과 사전에서 디코더에 미리 알려줄 말을 고른다.

    자를 수밖에 없으므로 무엇이 잘렸는지 부르는 쪽이 알 수 있게 목록으로
    돌려준다. 조용히 사라지면 왜 안 걸리는지 알 길이 없다.
    """
    from . import terminology

    terms = terminology.load(department)
    words: list[str] = []
    for group in HOTWORD_ORDER:
        words.extend(sorted(getattr(terms, group)))
    return words[:limit]


def _mark_manager(
    audio: np.ndarray,
    turns: list[tuple[str, int, int]],
    chunks: list[dict],
    voiceprint_path: Path,
    embedding_model: Path,
    num_threads: int,
) -> list[dict]:
    """등록된 성문과 닮은 화자를 찾아 매니저로 표시한다.

    못 찾으면 아무것도 바꾸지 않는다. 억지로 배정하면 환자가 매니저로 고정되어
    역할이 통째로 뒤바뀐다.
    """
    from .voiceprint import Voiceprint, build_extractor, find_manager

    voiceprint = Voiceprint.load(voiceprint_path)
    extractor = build_extractor(embedding_model, num_threads=num_threads)
    tag, scores = find_manager(audio, turns, voiceprint, extractor)

    rows = "  ".join(f"{speaker} {score}" for speaker, score in sorted(scores.items()))
    if tag is None:
        print(f"성문 매칭 실패 ({voiceprint.name}). 유사도: {rows}")
        return chunks

    print(f"매니저 확정: {tag} = {voiceprint.name}. 유사도: {rows}")
    return [{**chunk, "is_manager": chunk["speaker"] == tag} for chunk in chunks]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="녹음 파일을 SenseVoice로 전사한다.")
    parser.add_argument("audio", type=Path, help="녹음 파일 (m4a, wav, mp3 등)")
    parser.add_argument(
        "--engine",
        choices=sorted(ENGINE_FILES),
        default="sensevoice",
        help="전사 엔진. sensevoice는 다국어, moonshine과 zipformer는 한국어 전용 모델이 있다.",
    )
    parser.add_argument(
        "--model", type=Path, help="sensevoice: model.onnx / faster-whisper: 폴더 또는 HF 이름"
    )
    parser.add_argument("--encoder", type=Path, help="moonshine/whisper/zipformer: encoder")
    parser.add_argument("--decoder", type=Path, help="moonshine/whisper/zipformer: decoder")
    parser.add_argument("--joiner", type=Path, help="zipformer: joiner")
    # faster-whisper 는 토크나이저를 모델 안에 품고 있어 필요 없다. 엔진마다
    # 무엇이 필요한지는 ENGINE_FILES 가 정하고 아래에서 함께 확인한다.
    parser.add_argument("--tokens", type=Path, help="tokens.txt (faster-whisper 는 불필요)")
    parser.add_argument("--vad", type=Path, help="silero_vad.onnx (화자분리를 안 쓸 때)")
    parser.add_argument(
        "--segmentation", type=Path, help="화자분리 모델. 주면 VAD 대신 화자별로 끊는다."
    )
    parser.add_argument("--embedding", type=Path, help="화자 임베딩 모델")
    parser.add_argument(
        "--department",
        help="faster-whisper 전용. 그 진료과 용어를 디코더에 미리 알려준다(hotwords).",
    )
    parser.add_argument(
        "--hotwords",
        help="디코더에 미리 알려줄 말. 쉼표로 나눈다. --department 대신 직접 주고 싶을 때.",
    )
    parser.add_argument(
        "--manager",
        type=Path,
        help="voice-enroll 로 만든 매니저 성문(.json). 주면 그 화자를 매니저로 확정한다.",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=-1,
        help="화자 수를 알면 지정한다. 진료는 보통 2명(의사·환자) 또는 3명(매니저 포함).",
    )
    parser.add_argument("--out", type=Path, default=Path("chunks.json"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--language",
        default="ko",
        help="전사 언어. auto로 두면 한국어를 중국어로 잘못 잡는 일이 있다.",
    )
    parser.add_argument(
        "--max-chunk",
        type=float,
        help="화자분리 구간이 이보다 길면 조용한 지점에서 끊는다(초). "
        "생략하면 엔진에 맞춘다(whisper 30, 나머지 10).",
    )
    parser.add_argument(
        "--max-speech",
        type=float,
        default=10.0,
        help="발화 구간 최대 길이(초). 짧을수록 화자가 덜 섞인다.",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.25,
        help="이만큼 조용하면 구간을 끊는다(초). 짧을수록 자주 끊는다.",
    )
    args = parser.parse_args(argv)
    if args.max_chunk is None:
        args.max_chunk = default_max_chunk(args.engine)

    if not args.segmentation and not args.vad:
        parser.error("--segmentation(+--embedding) 또는 --vad 중 하나는 있어야 합니다.")
    if args.segmentation and not args.embedding:
        parser.error("--segmentation을 쓰려면 --embedding도 필요합니다.")

    missing = [f"--{name}" for name in ENGINE_FILES[args.engine] if getattr(args, name) is None]
    if missing:
        parser.error(f"--engine {args.engine} 에는 {', '.join(missing)} 이(가) 필요합니다.")

    hotwords = args.hotwords
    if hotwords is None and args.department:
        chosen = build_hotwords(args.department)
        hotwords = ", ".join(chosen)
        print(f"{args.department} 용어 {len(chosen)}개를 디코더에 알려줍니다.")
    if hotwords and args.engine not in PATHLESS_ENGINES:
        print(f"--hotwords 는 {args.engine} 에서는 무시됩니다.", file=sys.stderr)

    recognizer = build_recognizer(
        args.engine,
        model=args.model,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        tokens=args.tokens,
        num_threads=args.threads,
        language=args.language,
        hotwords=hotwords,
    )

    audio = load_audio(args.audio)
    print(f"오디오 {len(audio) / SAMPLE_RATE:.1f}초를 읽었습니다.")

    if args.segmentation:
        print("화자를 나누는 중입니다...")
        turns = diarize(
            audio,
            segmentation_model=args.segmentation,
            embedding_model=args.embedding,
            num_speakers=args.speakers,
            num_threads=args.threads,
        )
        speakers = sorted({speaker for speaker, _, _ in turns})
        print(f"화자 {len(speakers)}명, 발언 {len(turns)}구간을 찾았습니다. 전사를 시작합니다.")
        chunks = transcribe_turns(
            audio, turns, recognizer=recognizer, max_chunk_duration=args.max_chunk
        )
        if args.manager:
            chunks = _mark_manager(audio, turns, chunks, args.manager, args.embedding, args.threads)
    else:
        print("전사를 시작합니다.")
        chunks = transcribe(
            audio,
            recognizer=recognizer,
            vad_model=args.vad,
            min_silence_duration=args.min_silence,
            max_speech_duration=args.max_speech,
        )

    args.out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chunks:
        # 빈 결과를 조용히 저장하면 다음 단계에서야 알게 된다. 전사가 통째로
        # 실패한 것이므로 여기서 멈추고 무엇을 볼지 알려준다.
        print(
            f"전사가 한 구간도 내놓지 못했습니다. {args.out} 는 비어 있습니다.\n"
            "  - 모델이 그 언어를 못 내놓는지: --language 를 확인하세요\n"
            "  - 용어 알려주기가 방해하는지: --department 와 --hotwords 를 빼고 돌려보세요\n"
            "  - 녹음 자체가 비었는지: ffprobe 로 길이와 샘플레이트를 보세요",
            file=sys.stderr,
        )
        return 1
    print(f"발화 {len(chunks)}구간을 {args.out}에 저장했습니다.")
    print(f"다음: python3 -m voice_ai.analyze {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
