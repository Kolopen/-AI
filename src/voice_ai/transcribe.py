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

# 성문으로 떼어낸 매니저 구간에 붙이는 화자 태그. 화자분리가 매니저를 다른
# 사람과 한 덩이로 묶었을 때, 이 태그로 갈라 나온다.
MANAGER_TAG = "manager"


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


def segment_level(samples: np.ndarray) -> float:
    """구간의 소리 크기. 사람 말소리는 보통 0.03~0.1 사이에 든다."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def measure_levels(
    audio: np.ndarray, turns: list[tuple[str, int, int]]
) -> list[tuple[str, int, int, float]]:
    """화자별로 얼마나 크게 담겼는지 잰다.

    전사가 사람마다 갈릴 때 소리 크기 탓인지부터 봐야 한다. 환자가 작게 담긴
    것이라면 녹음기 위치를 바꾸면 되고, 크기가 같은데도 안 들리는 것이라면
    반향이나 잡음 문제라 자리를 옮긴다고 해결되지 않는다.
    """
    return [
        (
            speaker,
            start_ms,
            end_ms,
            segment_level(audio[int(start_ms * SAMPLE_RATE / 1000) : int(end_ms * SAMPLE_RATE / 1000)]),
        )
        for speaker, start_ms, end_ms in turns
    ]


# 잡음 분석에 쓰는 창 크기. 32ms 면 말소리의 울림 한 주기가 들어간다.
FRAME_SIZE = 512
HOP_SIZE = 128
# 잰 잡음보다 이만큼 더 빼낸다. 1.0 이면 덜 빠지고, 너무 키우면 말소리가 깎인다.
OVER_SUBTRACT = 1.5
# 다 빼지 않고 이만큼은 남긴다. 0 으로 만들면 "뮤지컬 노이즈" 라는 금속성
# 잡소리가 생겨서 오히려 전사가 나빠진다.
RESIDUAL_GAIN = 0.1


def gaps(audio: np.ndarray, turns: list[tuple[str, int, int]]) -> list[np.ndarray]:
    """아무도 말하지 않는 틈만 모은다. 거기 있는 소리가 곧 잡음이다."""
    found: list[np.ndarray] = []
    cursor = 0
    for _, start_ms, end_ms in sorted(turns, key=lambda t: t[1]):
        if start_ms > cursor:
            gap = audio[int(cursor * SAMPLE_RATE / 1000) : int(start_ms * SAMPLE_RATE / 1000)]
            if len(gap) >= SAMPLE_RATE // 10:
                found.append(gap)
        cursor = max(cursor, end_ms)

    tail = audio[int(cursor * SAMPLE_RATE / 1000) :]
    if len(tail) >= SAMPLE_RATE // 10:
        found.append(tail)
    return found


def noise_floor(audio: np.ndarray, turns: list[tuple[str, int, int]]) -> float:
    """잡음이 얼마나 큰지.

    말소리 크기만으로는 왜 안 들리는지 알 수 없다. 크게 담겼어도 잡음이 같이
    크면 모델은 묻힌 말을 받아 적지 못한다. 틈을 재면 둘이 갈린다.

    가운데값을 쓴다. 틈에 기침이나 문 닫는 소리가 섞여도 흔들리지 않는다.
    """
    levels = [segment_level(gap) for gap in gaps(audio, turns)]
    return float(np.median(levels)) if levels else 0.0


def _stft(samples: np.ndarray) -> np.ndarray:
    # 앞뒤로 창 하나씩 채운다. 한 쪽 끝의 몇 샘플은 창이 거의 0 이라 겹침이
    # 모자라고, 그대로 두면 구간 머리에서 "툭" 하는 소리가 남는다.
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    padded = np.pad(samples, (FRAME_SIZE, FRAME_SIZE))
    count = 1 + (len(padded) - FRAME_SIZE) // HOP_SIZE
    starts = HOP_SIZE * np.arange(count)[:, None]
    return np.fft.rfft(padded[starts + np.arange(FRAME_SIZE)[None, :]] * window, axis=1)


def _istft(spectrum: np.ndarray, length: int) -> np.ndarray:
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    frames = np.fft.irfft(spectrum, n=FRAME_SIZE, axis=1) * window

    out = np.zeros(length + 4 * FRAME_SIZE, dtype=np.float64)
    weight = np.zeros_like(out)
    for index, frame in enumerate(frames):
        start = index * HOP_SIZE
        out[start : start + FRAME_SIZE] += frame
        weight[start : start + FRAME_SIZE] += window**2

    weight[weight < 1e-8] = 1.0
    return (out / weight)[FRAME_SIZE : FRAME_SIZE + length].astype(np.float32)


def noise_profile(audio: np.ndarray, turns: list[tuple[str, int, int]]) -> np.ndarray | None:
    """주파수마다 잡음이 얼마나 깔려 있는지.

    크기 하나로는 못 걷어낸다. 에어컨은 낮은 쪽에, 형광등은 높은 쪽에 깔리므로
    주파수별로 다르게 빼야 말소리를 덜 깎는다.
    """
    frames = [_stft(gap) for gap in gaps(audio, turns) if len(gap) >= FRAME_SIZE]
    if not frames:
        return None
    return np.median(np.abs(np.concatenate(frames, axis=0)), axis=0)


def reduce_noise(
    samples: np.ndarray,
    profile: np.ndarray,
    *,
    over: float = OVER_SUBTRACT,
    residual: float = RESIDUAL_GAIN,
) -> np.ndarray:
    """잰 잡음을 주파수마다 빼낸다.

    이 녹음은 말소리가 잡음보다 5~10dB 밖에 크지 않았다. 클로바는 같은 녹음을
    받아 적었으니 소리가 없는 것이 아니라 우리 모델이 그 잡음을 못 견디는
    것이다. 깔린 만큼 빼주면 모델이 보는 입력이 학습 때와 가까워진다.
    """
    spectrum = _stft(samples)
    magnitude = np.abs(spectrum)
    kept = np.maximum(magnitude - over * profile[None, :], residual * magnitude)
    return _istft(spectrum * (kept / np.maximum(magnitude, 1e-10)), len(samples))


def signal_to_noise(level: float, floor: float) -> float | None:
    """말소리가 잡음보다 몇 dB 큰지. 틈이 없어 잴 수 없으면 None."""
    if floor <= 0.0 or level <= 0.0:
        return None
    return float(20 * np.log10(level / floor))


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


# 이보다 짧은 구간은 whisper 계열에 먹이지 않는다.
#
# whisper 는 30초 창으로 학습됐다. 0.3초짜리를 주면 나머지 29.7초를 묵음으로
# 채우는데, 디코더가 그 빈자리를 학습 데이터에서 본 문장으로 메운다. 실제
# 녹음에서 한국어 파인튜닝 모델이 이렇게 내놨다.
#
#   0.12초  "홍 사장의 발언에 국감장이 술렁이자 조정식의 발언에..."
#   0.62초  "고속도로 교통정보고 좋습니다"
#   1.62초  "북측에서 폭풍이 불어 닥쳤다"
#
# 녹음에 없는 말이다. 길이로 줄 세우면 2초 미만 18개 중 9개가 이런 환각이고,
# 2초 이상 19개 중에서는 1개였다.
#
# 2초를 넘겨도 맞는 말이 버려진다("감사합니다" 0.86초, "결국 어디서
# 찾았을까요" 1.52초). 그래도 지어낸 문장이 리포트에 들어가는 것보다 낫다.
# 빠진 것은 눈에 보이고 지어낸 것은 보이지 않는다.
# faster-whisper 는 모델이 내놓는 신호로 거르므로 길이로는 거의 안 자른다.
# 0.5초 미만은 화자분리가 남긴 부스러기라 어차피 건질 것이 없다. sherpa 쪽
# whisper 는 그 신호를 꺼내 쓸 수 없어 길이로만 막는다.
MIN_CHUNK_SECONDS = {"faster-whisper": 0.6, "whisper": 2.0}
FALLBACK_MIN_CHUNK = 0.1


def default_min_chunk(engine: str) -> float:
    return MIN_CHUNK_SECONDS.get(engine, FALLBACK_MIN_CHUNK)


# 같은 말이 이보다 많이 되풀이되면 디코더가 구멍에 빠진 것으로 본다.
# moonshine 의 "그렇죠? 그렇죠? 그렇죠?" 와 한국어 모델의 "조정식의 발언에"
# 세 번이 같은 모양이다. 사람은 같은 문장을 세 번 잇달아 말하지 않는다.
MAX_REPEATS = 3


def looks_repeated(text: str, *, limit: int = MAX_REPEATS) -> bool:
    """같은 조각이 되풀이되는지 본다. 길이를 바꿔가며 훑는다."""
    words = text.split()
    for size in range(1, len(words) // limit + 1):
        for start in range(len(words) - size * limit + 1):
            piece = words[start : start + size]
            if all(
                words[start + size * n : start + size * (n + 1)] == piece
                for n in range(1, limit)
            ):
                return True
    return False


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


# 모델이 스스로 내놓는 신호. 길이로 거르는 것보다 정확하다. 짧아도 확신하면
# 남기고, 길어도 지어낸 것이면 버린다. 실제 녹음의 여섯 구간으로 재 봤다.
#
#                        묵음   확신   반복
#   환각 0.12초          0.00  -0.37   2.26
#   환각 0.62초          0.00  -0.80   0.79
#   환각 4.37초          0.00  -0.90   0.86
#   정상 0.86초          0.00  -0.10   0.62
#   정상 1.52초          0.00  -0.18   0.74
#   정상 9.89초          0.00  -0.10   1.37
#
# 가르는 것은 확신도 하나뿐이다. 묵음 확률은 전부 0.00 이었다. 말소리가 실제로
# 있고 모델이 그걸 잘못 받아 적은 것이라 당연하다. 되풀이 비율도 2.26 으로
# 기준 아래였는데, 그 구간은 낱말 단위로 보는 looks_repeated 가 잡는다.
#
# -0.5 로 둔다. 정상 셋이 -0.10 ~ -0.18 이고 환각 둘이 -0.80, -0.90 이라
# 사이가 넓다. 조용한 녹음으로 잰 값이고 진료실은 더 시끄러울 테니 바짝
# 조이지 않는다. 0.12초짜리는 확신도로는 못 걸러도 되풀이로 걸린다.
#
# 묵음 확률과 되풀이 비율은 다른 종류의 실패를 위한 그물로 남겨 둔다.
NO_SPEECH_MAX = 0.6
AVG_LOGPROB_MIN = -0.5
COMPRESSION_MAX = 2.4


class _Stream:
    """sherpa-onnx 스트림과 같은 모양. 오디오를 받아 두었다가 한 번에 돌린다."""

    def __init__(self) -> None:
        self.audio: np.ndarray | None = None
        self.result = SimpleNamespace(text="")

    def accept_waveform(self, sample_rate: int, audio) -> None:
        self.audio = np.asarray(audio, dtype=np.float32)


def confident(segment) -> bool:
    """모델이 스스로 내놓은 신호로 지어낸 구간을 가린다."""
    return (
        segment.no_speech_prob <= NO_SPEECH_MAX
        and segment.avg_logprob >= AVG_LOGPROB_MIN
        and segment.compression_ratio <= COMPRESSION_MAX
    )


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

    def explain(self, audio) -> list[tuple]:
        """구간 하나를 돌려 보고 모델이 내놓은 신호를 그대로 보여준다.

        기준을 손으로 정할 때 쓴다. 숫자를 안 보고 정하면 또 틀린다.
        """
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            hotwords=self.hotwords,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return [
            (s.text.strip(), round(s.no_speech_prob, 3), round(s.avg_logprob, 3),
             round(s.compression_ratio, 2), confident(s))
            for s in segments
        ]

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
        kept = [s for s in segments if confident(s)]
        self.last = [
            (s.text.strip(), round(s.no_speech_prob, 3), round(s.avg_logprob, 3),
             round(s.compression_ratio, 2), confident(s))
            for s in kept
        ]
        stream.result.text = " ".join(s.text.strip() for s in kept).strip()


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
    min_chunk_duration: float = FALLBACK_MIN_CHUNK,
) -> list[dict]:
    """화자분리가 잡아준 구간마다 전사한다. 구간이 곧 한 사람의 발언이다.

    한 사람이 길게 말하면 구간도 그만큼 길어진다. moonshine 은 12초짜리 구간에서
    onnxruntime 예외를 내고 빈 결과를 돌려줬고, SenseVoice 는 죽지는 않지만 긴
    구간일수록 "신장"을 "심장"으로 쓰는 식으로 정확도가 떨어졌다. 상한을 두고
    조용한 지점에서 끊는다.
    """
    chunks: list[dict] = []
    max_samples = int(max_chunk_duration * SAMPLE_RATE)
    min_samples = int(min_chunk_duration * SAMPLE_RATE)

    for speaker, start_ms, end_ms in turns:
        piece = audio[int(start_ms * SAMPLE_RATE / 1000) : int(end_ms * SAMPLE_RATE / 1000)]
        if len(piece) < min_samples:
            continue

        for offset, part in _split_long(piece, max_samples):
            if len(part) < min_samples:
                continue
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, part)
            recognizer.decode_stream(stream)

            raw = _clean(stream.result.text)
            if not raw or looks_repeated(raw):
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
HOTWORD_SECTIONS = ("condition", "test", "drug")

# 실제 녹음으로 재 본 결과, 용어를 알려주는 것이 ghost613 한국어 모델을
# 망가뜨렸다. 기본으로 켜지 않는 까닭이다.
#
#   용어 없이   안녕하세요 지난번 이후 기업력 때문에 불폐한 일이 있었나요
#   용어 60개   <아무것도 안 나옴>
#   용어 12개   "이 아홉 명의 발언에 제기되었습니다" 가 아홉 번. 녹음에 없는 말이다.
#
# 디코더 프롬프트에 무엇을 넣든 모델은 그쪽으로 끌려간다. 의료 용어를 끼워
# 넣으려다 문장 전체를 지어내게 만들면 얻는 것보다 잃는 것이 크다.
#
# 모델마다 다를 수 있으므로 길은 남겨 둔다. 쓰려면 --hotwords 로 직접 주고,
# 짧은 구간으로 먼저 재 봐야 한다.
HOTWORD_LIMIT = 12


def build_hotwords(department: str, *, limit: int = HOTWORD_LIMIT) -> list[str]:
    """진료과 사전에서 디코더에 미리 알려줄 말을 고른다.

    공통 사전은 넣지 않는다. "감기", "기침", "고열" 같은 말은 모델이 이미
    잘 받아 적는다. 거들어야 할 것은 그 진료과에서만 쓰는 말이다.

    가나다순으로 자르면 "가래, 간질, 감기..."만 남고 정작 "치매"가 빠진다.
    사전 파일은 중요한 것부터 적혀 있으므로 그 순서를 그대로 쓴다. 무엇을
    앞에 둘지는 사전을 쓰는 사람이 정하는 것이 맞다.

    자를 수밖에 없으므로 무엇이 잘렸는지 부르는 쪽이 알 수 있게 목록으로
    돌려준다. 조용히 사라지면 왜 안 걸리는지 알 길이 없다.
    """
    from .terminology import TERMS_DIR

    path = TERMS_DIR / f"{department}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"그런 진료과 사전이 없습니다: {path}")

    sections: dict[str, list[str]] = {name: [] for name in HOTWORD_SECTIONS}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            continue
        if current in sections:
            sections[current].append(line)

    words: list[str] = []
    for name in HOTWORD_SECTIONS:
        words.extend(sections[name])
    return words[:limit]


def _mark_manager(
    audio: np.ndarray,
    turns: list[tuple[str, int, int]],
    chunks: list[dict],
    voiceprint_path: Path,
    embedding_model: Path,
    num_threads: int,
) -> list[dict]:
    """등록된 성문과 닮은 구간을 매니저로 떼어낸다.

    구간마다 따로 본다. 덩어리째 맞추면 화자분리가 매니저를 의사 쪽에 합쳐
    버린 녹음에서 의사 발언이 전부 매니저 발언이 된다.

    찾은 구간은 화자 태그를 MANAGER_TAG 로 바꾼다. 합쳐진 덩어리에서 떨어져
    나오므로 뒷단은 화자가 하나 늘어난 것으로만 보면 된다.

    못 찾으면 아무것도 바꾸지 않는다. 억지로 배정하면 역할이 통째로 뒤바뀐다.
    """
    from .voiceprint import Voiceprint, build_extractor, manager_turns

    voiceprint = Voiceprint.load(voiceprint_path)
    extractor = build_extractor(embedding_model, num_threads=num_threads)
    matched = manager_turns(audio, turns, voiceprint, extractor)

    spans = [
        (start_ms, end_ms) for (_, start_ms, end_ms), is_manager in zip(turns, matched) if is_manager
    ]
    if not spans:
        print(f"성문과 닮은 구간을 찾지 못했습니다 ({voiceprint.name}).")
        return chunks

    def from_manager(chunk: dict) -> bool:
        start = chunk.get("start_ms", 0)
        return any(start_ms <= start < end_ms for start_ms, end_ms in spans)

    marked = [
        {**chunk, "speaker": MANAGER_TAG, "is_manager": True} if from_manager(chunk) else chunk
        for chunk in chunks
    ]
    count = sum(1 for chunk in marked if chunk.get("is_manager"))
    print(f"매니저 확정: {voiceprint.name}. {len(spans)}구간 중 전사 {count}개를 떼어냈습니다.")
    return marked



def _print_levels(levels: list[tuple[str, int, int, float]], floor: float) -> None:
    """화자별 소리 크기와 잡음 대비를 사람이 읽게 찍는다."""
    by_speaker: dict[str, list[float]] = {}
    for speaker, _, _, rms in levels:
        by_speaker.setdefault(speaker, []).append(rms)

    print(f"\n잡음 바닥  {floor:.4f}" if floor > 0 else "\n잡음 바닥  잴 수 없음 (쉬는 틈이 없습니다)")
    print(f"\n화자별 ({len(levels)}구간)")
    for speaker, values in sorted(by_speaker.items()):
        level = sum(values) / len(values)
        snr = signal_to_noise(level, floor)
        margin = f"잡음보다 {snr:5.1f}dB 큼" if snr is not None else "잡음 대비 잴 수 없음"
        print(f"  {speaker}  크기 {level:.4f}  {margin}  ({len(values)}구간)")

    print(
        "\n말소리는 보통 0.03~0.1 입니다. 잡음보다 20dB 넘게 크면 깨끗하고,"
        "\n10dB 아래면 묻혀서 모델이 못 받아 적습니다. 크기는 멀쩡한데 대비가"
        "\n낮으면 자리를 옮겨도 안 풀리고, 둘 다 멀쩡한데 안 들리면 반향이거나"
        "\n모델이 그 말투를 못 배운 것입니다."
    )


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
        "--hotwords",
        help="디코더에 미리 알려줄 말(쉼표로 나눈다). 모델을 망가뜨릴 수 있으니 "
        "쓰기 전에 짧은 구간으로 먼저 재 보세요. README 의 '용어 알려주기' 참고.",
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
    parser.add_argument(
        "--denoise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="쉬는 틈에서 잡음을 재서 걷어낸 뒤 전사한다. 깨끗한 녹음이면 뺄 것이 적어 거의 그대로 간다.",
    )
    parser.add_argument(
        "--levels",
        action="store_true",
        help="전사하지 않고 화자별 소리 크기만 잰다. 전사가 사람마다 갈릴 때 쓴다.",
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
        "--min-chunk",
        type=float,
        help="이보다 짧은 구간은 버린다(초). 생략하면 엔진에 맞춘다"
        "(whisper 계열 2.0, 나머지 0.1). whisper 는 짧은 구간에서 없는 말을 지어낸다.",
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
    # 셸에서 찾은 경로를 변수로 넘기다 보면 빈 값이 그대로 들어온다. 빈
    # 경로는 현재 폴더로 읽혀서 "디렉터리입니다" 라는 엉뚱한 예외가 난다.
    if not str(args.audio).strip() or str(args.audio) == ".":
        parser.error("녹음 파일 경로가 비어 있습니다. 파일을 찾았는지 확인하세요.")
    if not args.audio.is_file():
        parser.error(f"녹음 파일이 없습니다: {args.audio}")

    if args.max_chunk is None:
        args.max_chunk = default_max_chunk(args.engine)
    if args.min_chunk is None:
        args.min_chunk = default_min_chunk(args.engine)

    if not args.segmentation and not args.vad:
        parser.error("--segmentation(+--embedding) 또는 --vad 중 하나는 있어야 합니다.")
    if args.segmentation and not args.embedding:
        parser.error("--segmentation을 쓰려면 --embedding도 필요합니다.")

    if args.levels:
        if not args.segmentation:
            parser.error("--levels 에는 --segmentation 과 --embedding 이 필요합니다.")
        audio = load_audio(args.audio)
        print(f"오디오 {len(audio) / SAMPLE_RATE:.1f}초를 읽었습니다.")
        print("화자를 나누는 중입니다...")
        turns = diarize(
            audio,
            segmentation_model=args.segmentation,
            embedding_model=args.embedding,
            num_speakers=args.speakers,
            num_threads=args.threads,
        )
        _print_levels(measure_levels(audio, turns), noise_floor(audio, turns))
        return 0

    missing = [f"--{name}" for name in ENGINE_FILES[args.engine] if getattr(args, name) is None]
    if missing:
        parser.error(f"--engine {args.engine} 에는 {', '.join(missing)} 이(가) 필요합니다.")

    hotwords = args.hotwords
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
        if args.denoise:
            profile = noise_profile(audio, turns)
            if profile is None:
                print("쉬는 틈이 없어 잡음을 잴 수 없습니다. 그대로 전사합니다.", file=sys.stderr)
            else:
                before = noise_floor(audio, turns)
                audio = reduce_noise(audio, profile)
                print(f"잡음 {before:.4f} -> {noise_floor(audio, turns):.4f} 로 걷었습니다.")
        print(f"화자 {len(speakers)}명, 발언 {len(turns)}구간을 찾았습니다. 전사를 시작합니다.")
        chunks = transcribe_turns(
            audio,
            turns,
            recognizer=recognizer,
            max_chunk_duration=args.max_chunk,
            min_chunk_duration=args.min_chunk,
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
