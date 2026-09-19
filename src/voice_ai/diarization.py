"""STT 결과와 화자 구간을 타임스탬프로 맞붙인다.

CLOVA Speech는 전사와 화자분리를 함께 주지만, 화자분리를 따로 쓰면(pyannote 등)
둘을 직접 정렬해야 한다. 어떤 STT와 어떤 화자분리를 고르든 이 정렬은 필요하다.

CLOVA의 구조적 한계가 여기서 풀린다. CLOVA는 세그먼트마다 화자를 하나만 붙여서
긴 의사 설명에 끼어든 환자의 짧은 응답을 표현할 방법이 없었다. 화자 구간을 따로
받으면 한 전사 구간이 여러 화자에 걸친 것을 알아낼 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Utterance


@dataclass
class Turn:
    """화자 한 명이 말한 시간 구간. 텍스트는 없다."""

    speaker: str
    start_ms: int
    end_ms: int


# 두 번째로 많이 겹친 화자가 전사 구간의 이 비율 이상을 차지하면 혼입을 의심한다.
MULTI_SPEAKER_RATIO = 0.2


def parse_pyannote(payload: dict[str, Any]) -> list[Turn]:
    """pyannote 출력을 Turn 목록으로. 초 단위로 오므로 밀리초로 맞춘다."""
    turns = [
        Turn(
            speaker=f"speaker_{entry['speaker']}",
            start_ms=int(round(float(entry["start"]) * 1000)),
            end_ms=int(round(float(entry["end"]) * 1000)),
        )
        for entry in payload.get("diarization", [])
    ]
    turns.sort(key=lambda t: t.start_ms)
    return turns


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def align(
    segments: list[Utterance], turns: list[Turn]
) -> tuple[list[Utterance], list[str]]:
    """전사 구간마다 가장 많이 겹친 화자를 붙이고, 혼입 의심 구간을 경고한다.

    화자를 바꿔 붙이기만 하고 텍스트를 쪼개지는 않는다. 단어 단위 시각 없이
    문장을 나누면 어느 쪽 화자의 말인지 잘못 가를 수 있다.
    """
    aligned: list[Utterance] = []
    warnings: list[str] = []

    for segment in segments:
        overlaps = sorted(
            (
                (_overlap_ms(segment.start_ms, segment.end_ms, t.start_ms, t.end_ms), t.speaker)
                for t in turns
            ),
            reverse=True,
        )

        if not overlaps or overlaps[0][0] == 0:
            aligned.append(
                Utterance(
                    speaker_tag="speaker_unknown",
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=segment.text,
                    confidence=segment.confidence,
                )
            )
            warnings.append(
                f"{segment.start_ms}ms 구간에 겹치는 화자가 없습니다. "
                "전사와 화자분리의 시각 기준이 어긋났을 수 있습니다."
            )
            continue

        best_overlap, best_speaker = overlaps[0]
        aligned.append(
            Utterance(
                speaker_tag=best_speaker,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=segment.text,
                confidence=segment.confidence,
            )
        )

        duration = max(1, segment.duration_ms)
        runner_up = next((o for o in overlaps[1:] if o[1] != best_speaker), None)
        if runner_up and runner_up[0] / duration >= MULTI_SPEAKER_RATIO:
            warnings.append(
                f"{segment.start_ms}ms 구간이 {best_speaker}와 {runner_up[1]}에 걸쳐 있습니다. "
                "한 전사 구간에 두 사람의 말이 섞였을 수 있습니다."
            )

    return aligned, warnings
