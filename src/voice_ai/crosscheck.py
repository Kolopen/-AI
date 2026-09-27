"""두 엔진의 전사를 맞대어 어긋나는 곳을 찾는다.

엔진마다 다른 자리에서 틀린다. 실제 녹음에서 SenseVoice 는 "신장"을 맞추고
moonshine 이 "심장"으로 썼으며, 숫자를 잇는 말은 반대로 moonshine 이 나았다.
한쪽만 보면 어디가 틀렸는지 알 수 없지만, 둘을 맞대면 어긋나는 자리가 드러난다.

특히 숫자가 중요하다. 검사 수치는 리포트의 핵심인데 틀려도 그럴듯해 보인다.
두 엔진이 독립적으로 같은 숫자를 냈다면 믿을 만하고, 다르면 사람이 들어야 한다.

합치지는 않는다. 두 전사를 섞으면 아무도 말하지 않은 문장이 만들어진다.
한쪽을 본문으로 쓰고, 어긋난 자리만 표시한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Utterance

_NUMBER = re.compile(r"\d+")

# 두 구간이 이만큼 겹치면 같은 대목으로 본다. 화자분리를 공유하면 보통 정확히 겹친다.
_MIN_OVERLAP_MS = 500


@dataclass
class Disagreement:
    start_ms: int
    kind: str  # NUMBER 또는 TERM
    primary: str
    secondary: str


def _overlap(a: Utterance, b: Utterance) -> int:
    return min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms)


def align(primary: list[Utterance], secondary: list[Utterance]) -> list[tuple[Utterance, str]]:
    """본문 구간마다 같은 시간대의 다른 엔진 전사를 붙인다.

    화자분리를 한 번만 돌려 두 엔진에 같은 구간을 먹이면 시간이 그대로 맞는다.
    구간 길이 상한을 다르게 준 경우까지 견디도록 겹침으로 찾는다.
    """
    paired: list[tuple[Utterance, str]] = []
    for utterance in primary:
        matched = [
            other.text
            for other in secondary
            if _overlap(utterance, other) >= _MIN_OVERLAP_MS
        ]
        paired.append((utterance, " ".join(matched)))
    return paired


def cross_check(
    primary: list[Utterance], secondary: list[Utterance], terms: frozenset[str]
) -> list[Disagreement]:
    """숫자와 의학 용어가 엇갈리는 자리를 찾는다."""
    found: list[Disagreement] = []

    for utterance, other_text in align(primary, secondary):
        if not other_text:
            continue

        mine = _NUMBER.findall(utterance.text)
        theirs = _NUMBER.findall(other_text)
        if mine != theirs:
            found.append(
                Disagreement(utterance.start_ms, "NUMBER", ", ".join(mine), ", ".join(theirs))
            )

        my_terms = {t for t in terms if t in utterance.text}
        their_terms = {t for t in terms if t in other_text}
        for term in sorted(my_terms - their_terms):
            found.append(Disagreement(utterance.start_ms, "TERM", term, ""))
        for term in sorted(their_terms - my_terms):
            found.append(Disagreement(utterance.start_ms, "TERM", "", term))

    return found
