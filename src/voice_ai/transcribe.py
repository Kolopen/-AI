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

    def drain() -> None:
        while not vad.empty():
            segment = vad.front
            start_ms = round(segment.start * 1000 / SAMPLE_RATE)
            end_ms = start_ms + round(len(segment.samples) * 1000 / SAMPLE_RATE)

            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, segment.samples)
            vad.pop()
            recognizer.decode_stream(stream)

            raw = stream.result.text.strip()
            if raw:
                chunks.append(
                    {
                        "index": len(chunks),
                        "start_ms": start_ms,
                        "end_ms": end_ms,
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
    parser.add_argument("--vad", type=Path, required=True, help="silero_vad.onnx")
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

    audio = load_audio(args.audio)
    print(f"오디오 {len(audio) / SAMPLE_RATE:.1f}초를 읽었습니다. 전사를 시작합니다.")

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
