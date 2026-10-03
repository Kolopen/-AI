"""점수로 말하는 검사를 뽑는다.

검사 수치는 이름이 먼저 나와야 숫자를 받는다("콜레스테롤이 200"). 그 사전
대조가 환각 숫자를 막는 안전장치다. 그런데 점수는 이름 없이 온다.

    지난 검사 점수는 30점 만점에 26점이었습니다

무슨 검사인지 말하지 않으므로 사전으로는 한 글자도 못 건진다. 두 녹음에서
연속으로 놓쳤고, 둘 다 보호자에게 꼭 전해야 할 숫자였다. 기억력 검사에서는
환자가 29점으로 잘못 기억하고 있어서 의사가 바로잡는 장면이었다.

점수는 `점`이라는 단위 자체가 근거다. 이름 대신 단위를 본다.

숫자의 주인이 누구인지로 누구 말을 받을지 가른다. 검사 점수의 주인은 검사
결과이므로 의사 말만 받는다. 아픈 정도의 주인은 환자 본인이므로 환자 말도
받는다. 의사 말만 보면 통증 점수는 영영 못 잡는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Role, Utterance
from .terminology import Terminology
from .timeline import UNMARKED, group_by_time

# "30점 만점에 26점". 만점 쪽은 값이 아니라 눈금이다.
# 실제 전사가 "만점"을 "만쯤"으로 흘려서 한 글자는 흔들려도 받는다.
# 영문자나 숫자에 붙은 숫자는 이름의 일부다. "비타민 b12 점수"의 12 가
# 12점으로 올라가던 자리다.
_DIGITS = r"(?<![A-Za-z0-9])(\d{1,3})"
# "점수"의 점은 단위가 아니다. 숫자 쪽에도 같은 덫이 있다("비12 점수").
# 전사가 점을 "쩜"으로 흘리는 일이 잦아 둘 다 받는다. 실제 진료 녹음에서
# "삼십 쩜 만쯤에 이십 육 점" 으로 나와 만점 쪽을 통째로 놓쳤다.
_UNIT = r"[점쩜](?!수)"
_OUT_OF = re.compile(
    _DIGITS + r"\s*" + _UNIT + r"\s*만[가-힣]?\s*(?:에|으로)?\s*(?:는)?\s*" + _DIGITS + r"\s*" + _UNIT
)
_SCORE = re.compile(_DIGITS + r"\s*" + _UNIT)

# "영 점, 십 점으로 해서 적으세요"는 눈금 설명이지 측정값이 아니다.
# 올리면 "통증 0점"이라는 없던 기록이 남는다.
_SCALE = re.compile(r"\d{1,3}\s*[점쩜]\s*(?:으로|로)\s*(?:해서|하고|두고|보고|잡|하면)")

# 아픈 정도는 환자 본인만 안다. 이쪽은 환자 말도 받는다.
_SELF_REPORTED = re.compile(r"통증|아프|아픔|불편|저림|가렵|가려")

# 한자어 수. 한국어 파인튜닝 모델은 숫자를 한글로 쓴다("이십 육 점").
# 숫자로 쓰는 모델과 섞어 쓰려면 한쪽으로 맞춰야 한다.
_SINO = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
         "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
_SINO_UNITS = {"십": 10, "백": 100}
_SINO_CHARS = "".join(_SINO) + "".join(_SINO_UNITS)

# 한자어 수는 "점" 이 바로 뒤에 붙었을 때만 읽는다. 단위 없이 읽으면 "검사"의
# "사"가 4가 되고 "지금"의 "이"가 2가 된다. 앞에 한글이 붙어 있어도 낱말의
# 일부이므로 읽지 않는다("검사 점수"의 사).
#
# "점" 뒤에 "수"가 붙으면 단위가 아니라 "점수"라는 낱말이다. 전사가 "비타민
# B12"를 "비타민 비시 이 점수치"로 흘렸는데, 그 "이"가 2점이 됐다.
_SINO_SCORE = re.compile(
    rf"(?<![가-힣])([{_SINO_CHARS}]+(?:\s+[{_SINO_CHARS}]+)*)\s*([점쩜])(?!수)"
)

# 점수라고 알아볼 말. 하나도 없으면 "점"이 다른 뜻일 수 있으므로 받지 않는다.
_SCORE_CONTEXT = re.compile(r"점수|검사|통증|아프|아픔|정도")

# 이름을 앞 구간에서 이어받는 범위. 의사가 "통증 정도는 어땠나요" 하고 물으면
# 환자는 다음 숨에 "사 점이나 오 점" 이라고만 답한다.
CARRY_WINDOW_MS = 30_000

GENERIC_NAME = "검사 점수"
PAIN_NAME = "통증"


@dataclass
class Score:
    """점수로 말한 측정값."""

    name: str
    value: str
    maximum: str | None
    when: str
    quote: str
    start_ms: int
    said_by: Role
    # 이름을 앞 구간에서 이어받았으면 표시한다. 같은 구간에 함께 나왔다는 것
    # 말고는 근거가 없다.
    inferred: bool = False


def _sino_value(text: str) -> int | None:
    """한자어 수 한 덩이를 숫자로 바꾼다. "이십육" -> 26"""
    total, current = 0, 0
    for char in text:
        if char in _SINO:
            current = _SINO[char]
        elif char in _SINO_UNITS:
            total += (current or 1) * _SINO_UNITS[char]
            current = 0
        elif not char.isspace():
            return None
    return total + current


def to_digits(text: str) -> str:
    """한글로 쓴 점수를 숫자로 바꾼다. 나머지는 건드리지 않는다."""

    def swap(found: re.Match[str]) -> str:
        value = _sino_value(found.group(1))
        return found.group() if value is None else f"{value}{found.group(2)}"

    return _SINO_SCORE.sub(swap, text)


def _named(text: str, terms: Terminology) -> str | None:
    compact = text.replace(" ", "").casefold()
    found = next(
        (t for t in sorted(terms.tests, key=len, reverse=True) if t.casefold() in compact), None
    )
    if found:
        return found
    if _SELF_REPORTED.search(text):
        return PAIN_NAME
    if _SCORE_CONTEXT.search(text):
        return GENERIC_NAME
    return None


def _values(text: str) -> list[tuple[str, str | None]]:
    """(점수, 만점) 짝을 뽑는다. 눈금 설명은 버린다."""
    found: list[tuple[str, str | None]] = []
    taken: list[tuple[int, int]] = []

    for match in _OUT_OF.finditer(text):
        if _SCALE.match(text, match.start()):
            continue
        found.append((match.group(2), match.group(1)))
        taken.append(match.span())

    for match in _SCORE.finditer(text):
        if any(start <= match.start() < end for start, end in taken):
            continue
        if _SCALE.match(text, match.start()):
            continue
        found.append((match.group(1), None))
    return found


def extract(
    utterances: list[Utterance], roles: dict[str, Role], terms: Terminology
) -> list[Score]:
    scores: list[Score] = []
    carried: tuple[str, int] | None = None

    for utterance in utterances:
        role = roles.get(utterance.speaker_tag, Role.UNKNOWN)
        previous = carried
        here = _named(utterance.text, terms)
        if here:
            carried = (here, utterance.start_ms)

        # 한글로 쓴 점수를 숫자로 맞춰 둔다. 근거 문장은 원문 그대로 남긴다.
        spoken = to_digits(utterance.text)
        found = _values(spoken)
        if not found:
            continue

        # 앞 구간의 구체적인 이름이 이 문장의 두루뭉술한 이름을 이긴다.
        # "통증 정도는 어땠나요" 다음에 환자는 "4점이나 5점 정도요" 라고만
        # 답하는데, 그 "정도" 때문에 검사 점수로 잡히면 환자 말이라고 버려진다.
        name, inferred = here, False
        usable = (
            previous
            and previous[0] != GENERIC_NAME
            and utterance.start_ms - previous[1] <= CARRY_WINDOW_MS
        )
        if usable and name in (None, GENERIC_NAME):
            name, inferred = previous[0], True
        if name is None:
            continue

        # 숫자의 주인만 그 숫자를 말할 수 있다. 환자가 자기 검사 점수를 잘못
        # 기억하는 일이 실제로 있었다("26점이요? 29점인 줄 알았어요").
        if role is not Role.DOCTOR and not (role is Role.PATIENT and name == PAIN_NAME):
            continue

        buckets = group_by_time(spoken)
        for value, maximum in found:
            when = next(
                (marker for marker, values in buckets.items() if value in values), UNMARKED
            )
            # 의사는 환자가 잘못 기억한 점수를 바로잡으며 같은 값을 다시
            # 말한다("26점입니다"). 시점 없이 되풀이한 값은 새 측정이 아니다.
            if when is UNMARKED and any(
                kept.name == name and kept.value == value for kept in scores
            ):
                continue
            scores.append(
                Score(
                    name=name,
                    value=value,
                    maximum=maximum,
                    when=when,
                    quote=utterance.text.strip(),
                    start_ms=utterance.start_ms,
                    said_by=role,
                    inferred=inferred,
                )
            )
    return scores
