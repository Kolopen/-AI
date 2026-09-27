"""녹음 파일을 SenseVoice로 전사해 analyze가 받는 chunks.json을 만든다.

VAD가 긴 녹음을 발화 구간으로 자르고, 잘린 구간마다 SenseVoice를 돌린다.
20분짜리를 통째로 모델에 넣으면 메모리가 감당하지 못하므로 VAD는 선택이 아니다.

오디오 읽기는 PyAV를 쓴다. ffmpeg 라이브러리를 품고 있어서 m4a를 포함해 대부분의
형식을 외부 바이너리 설치 없이 읽는다. 매니저 폰이 무엇으로 녹음하든 받는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def diarize(
    audio: np.ndarray,
    *,
    segmentation_model: Path,
    embedding_model: Path,
    num_speakers: int = -1,
    cluster_threshold: float = 0.5,
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
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding_model)),
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
    model: Path,
    tokens: Path,
    num_threads: int = 4,
) -> list[dict]:
    """화자분리가 잡아준 구간마다 전사한다. 구간이 곧 한 사람의 발언이다."""
    import sherpa_onnx

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model), tokens=str(tokens), num_threads=num_threads, use_itn=True
    )

    chunks: list[dict] = []
    for speaker, start_ms, end_ms in turns:
        piece = audio[int(start_ms * SAMPLE_RATE / 1000) : int(end_ms * SAMPLE_RATE / 1000)]
        if len(piece) < SAMPLE_RATE // 10:  # 0.1초 미만은 버린다
            continue

        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, piece)
        recognizer.decode_stream(stream)

        raw = stream.result.text.strip()
        if raw:
            chunks.append(
                {
                    "index": len(chunks),
                    "speaker": speaker,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "raw_text": raw,
                }
            )
    return chunks


def transcribe(
    audio: np.ndarray,
    *,
    model: Path,
    tokens: Path,
    vad_model: Path,
    num_threads: int = 4,
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

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model),
        tokens=str(tokens),
        num_threads=num_threads,
        use_itn=True,
    )

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

                raw = stream.result.text.strip()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="녹음 파일을 SenseVoice로 전사한다.")
    parser.add_argument("audio", type=Path, help="녹음 파일 (m4a, wav, mp3 등)")
    parser.add_argument("--model", type=Path, required=True, help="SenseVoice model.onnx")
    parser.add_argument("--tokens", type=Path, required=True, help="SenseVoice tokens.txt")
    parser.add_argument("--vad", type=Path, help="silero_vad.onnx (화자분리를 안 쓸 때)")
    parser.add_argument(
        "--segmentation", type=Path, help="화자분리 모델. 주면 VAD 대신 화자별로 끊는다."
    )
    parser.add_argument("--embedding", type=Path, help="화자 임베딩 모델")
    parser.add_argument(
        "--speakers",
        type=int,
        default=-1,
        help="화자 수를 알면 지정한다. 진료는 보통 2명(의사·환자) 또는 3명(매니저 포함).",
    )
    parser.add_argument("--out", type=Path, default=Path("chunks.json"))
    parser.add_argument("--threads", type=int, default=4)
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

    if not args.segmentation and not args.vad:
        parser.error("--segmentation(+--embedding) 또는 --vad 중 하나는 있어야 합니다.")
    if args.segmentation and not args.embedding:
        parser.error("--segmentation을 쓰려면 --embedding도 필요합니다.")

    audio = load_audio(args.audio)
    print(f"오디오 {len(audio) / SAMPLE_RATE:.1f}초를 읽었습니다.")

    if args.segmentation:
        print("화자를 나누는 중입니다...")
        turns = diarize(
            audio,
            segmentation_model=args.segmentation,
            embedding_model=args.embedding,
            num_speakers=args.speakers,
        )
        speakers = sorted({speaker for speaker, _, _ in turns})
        print(f"화자 {len(speakers)}명, 발언 {len(turns)}구간을 찾았습니다. 전사를 시작합니다.")
        chunks = transcribe_turns(
            audio, turns, model=args.model, tokens=args.tokens, num_threads=args.threads
        )
    else:
        print("전사를 시작합니다.")
        chunks = transcribe(
            audio,
            model=args.model,
            tokens=args.tokens,
            vad_model=args.vad,
            num_threads=args.threads,
            min_silence_duration=args.min_silence,
            max_speech_duration=args.max_speech,
        )

    args.out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"발화 {len(chunks)}구간을 {args.out}에 저장했습니다.")
    print(f"다음: python3 -m voice_ai.analyze {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
