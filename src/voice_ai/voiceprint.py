"""매니저 성문 등록과 매칭.

의사·환자·매니저 셋을 가르는 일에서 유일하게 전사 품질과 무관한 자리다.
나머지는 전부 글자가 깨지면 같이 무너진다. 성문은 오디오만 본다.

매니저는 플랫폼 소속 검증 인력이라 자격 심사 때 음성 샘플을 받아둘 수 있다.
한 번 받아 두면 이후 모든 녹음에서 확정되고, 3명 중 1명이 확정되면 나머지는
"의사 vs 환자" 2택이 되어 난이도가 절반으로 떨어진다.

실제 녹음에서 화자분리가 의사는 깨끗이 갈라냈지만 환자와 매니저를 한 덩이로
묶었다. 목소리가 셋 다 구분되는 녹음이었는데도 그랬다. 그 자리를 이걸로 푼다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# 코사인 유사도가 이보다 높아야 같은 사람으로 본다. eres2net 계열의 통상 범위다.
# 낮추면 환자가 매니저로 고정되어 역할이 통째로 뒤바뀌므로 보수적으로 잡는다.
MATCH_THRESHOLD = 0.55

# 화자 하나를 대표할 오디오 길이. 너무 짧으면 성문이 흔들리고, 길다고 더
# 좋아지지도 않는다. 긴 발언부터 모아 이만큼 채운다.
PROFILE_SECONDS = 8.0


@dataclass
class Voiceprint:
    """등록된 매니저 한 사람의 성문."""

    name: str
    vector: list[float]

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"name": self.name, "vector": self.vector}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Voiceprint":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(name=payload["name"], vector=payload["vector"])


def build_extractor(model: Path, *, num_threads: int = 4):
    import sherpa_onnx

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model), num_threads=num_threads
    )
    if not config.validate():
        raise RuntimeError(f"성문 모델 설정이 올바르지 않습니다: {model}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def embed(audio: np.ndarray, extractor) -> np.ndarray:
    stream = extractor.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    stream.input_finished()
    return np.asarray(extractor.compute(stream), dtype=np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def enroll(samples: list[np.ndarray], extractor, *, name: str) -> Voiceprint:
    """음성 샘플 여러 개를 평균 내어 성문 하나를 만든다.

    한 번 녹음한 것만 쓰면 그날의 목소리 상태에 끌려간다. 여러 번 받아 두면
    평균이 그 사람 쪽으로 모인다.
    """
    if not samples:
        raise ValueError("음성 샘플이 없습니다.")
    vectors = [embed(sample, extractor) for sample in samples]
    averaged = np.mean(vectors, axis=0)
    return Voiceprint(name=name, vector=[float(v) for v in averaged])


def _speaker_audio(
    audio: np.ndarray, turns: list[tuple[str, int, int]], speaker: str
) -> np.ndarray:
    """한 화자의 발언을 긴 것부터 모아 이어 붙인다."""
    mine = sorted(
        (t for t in turns if t[0] == speaker), key=lambda t: t[2] - t[1], reverse=True
    )
    pieces: list[np.ndarray] = []
    collected = 0.0
    for _, start_ms, end_ms in mine:
        piece = audio[int(start_ms * SAMPLE_RATE / 1000) : int(end_ms * SAMPLE_RATE / 1000)]
        if len(piece) < SAMPLE_RATE // 2:  # 0.5초 미만은 성문이 못 미덥다
            continue
        pieces.append(piece)
        collected += len(piece) / SAMPLE_RATE
        if collected >= PROFILE_SECONDS:
            break
    return np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)


def find_manager(
    audio: np.ndarray,
    turns: list[tuple[str, int, int]],
    voiceprint: Voiceprint,
    extractor,
    *,
    threshold: float = MATCH_THRESHOLD,
) -> tuple[str | None, dict[str, float]]:
    """등록된 성문과 가장 닮은 화자를 찾는다.

    (매니저로 본 화자 태그, 화자별 유사도) 를 돌려준다. 기준에 못 미치면
    태그는 None 이다. 억지로 배정하지 않는다. 매니저가 말을 거의 안 한
    진료도 있고, 그때 환자를 매니저로 고정하면 역할이 통째로 뒤바뀐다.
    """
    reference = np.asarray(voiceprint.vector, dtype=np.float32)
    scores: dict[str, float] = {}
    for speaker in sorted({t[0] for t in turns}):
        spoken = _speaker_audio(audio, turns, speaker)
        if len(spoken) == 0:
            continue
        scores[speaker] = round(similarity(embed(spoken, extractor), reference), 3)

    if not scores:
        return None, {}
    best = max(scores, key=lambda tag: scores[tag])
    return (best if scores[best] >= threshold else None), scores
