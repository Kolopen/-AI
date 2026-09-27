"""두 엔진의 전사를 맞대어 어긋나는 곳을 찾는다.

엔진마다 다른 자리에서 틀린다. 실제 녹음에서 SenseVoice 는 "신장"을 맞추고
moonshine 이 "심장"으로 썼으며, 숫자를 잇는 말은 반대로 moonshine 이 나았다.
한쪽만 보면 어디가 틀렸는지 알 수 없지만, 둘을 맞대면 어긋난 자리가 드러난다.

특히 숫자가 중요하다. 검사 수치는 리포트의 핵심인데 틀려도 그럴듯해 보인다.
두 엔진이 독립적으로 같은 숫자를 냈다면 믿을 만하고, 다르면 사람이 들어야 한다.

두 엔진의 구간 나누기는 서로 다르다. 같은 화자분리를 먹여도 빈 결과를 내는
구간이 다르고, 길이 상한을 달리 준 전사끼리 비교할 일도 있다. 그래서 구간을
짝짓지 않는다. 용어는 녹음 전체에서 한 번씩 보고, 숫자는 시간대로 묶어 본다.

합치지는 않는다. 두 전사를 섞으면 아무도 말하지 않은 문장이 만들어진다.
한쪽을 본문으로 쓰고, 어긋난 자리만 표시한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Utterance

_NUMBER = re.compile(r"\d+")

# 숫자를 이 길이로 묶어 견준다. 구간 경계가 엇갈려도 같은 칸에 들어가도록
# 넉넉히 잡되, 한 칸에 여러 검사 수치가 뭉치지 않을 만큼은 짧게 둔다.
NUMBER_BUCKET_MS = 15_000


@dataclass
class Disagreement:
    start_ms: int
    kind: str  # NUMBER 또는 TERM
    primary: str
    secondary: str


def _first_seen(utterances: list[Utterance], terms: frozenset[str]) -> dict[str, int]:
    """용어마다 처음 나온 시각. 같은 용어를 여러 번 알리지 않기 위한 것."""
    seen: dict[str, int] = {}
    for utterance in utterances:
        for term in terms:
            if term in utterance.text and term not in seen:
                seen[term] = utterance.start_ms
    return seen


def _numbers_by_bucket(utterances: list[Utterance]) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {}
    for utterance in utterances:
        bucket = utterance.start_ms // NUMBER_BUCKET_MS
        buckets.setdefault(bucket, []).extend(_NUMBER.findall(utterance.text))
    return buckets


def _only_in(these: list[str], those: list[str]) -> list[str]:
    """중복까지 헤아려 이쪽에만 있는 값을 돌려준다. 76이 두 번이면 두 번 다 본다."""
    remaining = list(those)
    extra = []
    for value in these:
        if value in remaining:
            remaining.remove(value)
        else:
            extra.append(value)
    return extra


def cross_check(
    primary: list[Utterance], secondary: list[Utterance], terms: frozenset[str]
) -> list[Disagreement]:
    """숫자와 의학 용어가 엇갈리는 자리를 찾는다.

    한쪽이 비어 있으면 비교할 것이 없다. 상대가 통째로 실패한 것을 전부
    엇갈림으로 세면 경고만 쌓이고 쓸모가 없다.
    """
    if not primary or not secondary:
        return []

    found: list[Disagreement] = []

    mine = _numbers_by_bucket(primary)
    theirs = _numbers_by_bucket(secondary)
    for bucket in sorted(set(mine) | set(theirs)):
        at = bucket * NUMBER_BUCKET_MS
        here, there = mine.get(bucket, []), theirs.get(bucket, [])
        missing_there = _only_in(here, there)
        missing_here = _only_in(there, here)
        if missing_there or missing_here:
            found.append(
                Disagreement(at, "NUMBER", ", ".join(missing_there), ", ".join(missing_here))
            )

    my_terms = _first_seen(primary, terms)
    their_terms = _first_seen(secondary, terms)
    for term in sorted(set(my_terms) - set(their_terms)):
        found.append(Disagreement(my_terms[term], "TERM", term, ""))
    for term in sorted(set(their_terms) - set(my_terms)):
        found.append(Disagreement(their_terms[term], "TERM", "", term))

    found.sort(key=lambda d: (d.start_ms, d.kind))
    return found
