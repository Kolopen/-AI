"""의사 발언에서 사실만 뽑아낸다.

문장을 발췌하면 전사의 깨진 어미가 그대로 따라온다. "40 이상 보다 높 으니까"
같은 줄은 보호자에게 보여줄 수 없고, 매끄럽게 고치려면 생성형이 필요한데
그러면 잘못 전사된 수치가 그럴듯한 다른 값으로 바뀐다.

그래서 문장 대신 값을 뽑는다. 이름표는 우리가 붙이고 값은 원문에서 그대로
가져온다. 지어내는 부분이 없고, 깨진 어미는 값이 아니므로 따라오지 않는다.

숫자가 어느 검사의 것인지는 같은 문장에 함께 나왔다는 것 말고는 근거가 없다.
그래서 근거 문장을 항상 같이 남겨 사람이 맞는지 볼 수 있게 한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Role, Utterance
from .roles import split_sentences
from .terminology import Terminology

_NUMBER = re.compile(r"\d+")

# "술도 줄이셔야", "체중 줄이시면", "허리를 줄이셔야"
# -시- 는 어미와 만나면 "셔"로 줄어든다(줄이시어야 → 줄이셔야). 둘 다 받아야 한다.
_REDUCE = re.compile(r"([가-힣]{1,6}?)\s*(?:도|를|을|은|는)?\s*(?:줄이|끊으|빼)(?:시|셔)")
# "운동 열심히 하셔서", "운동도 열심히"
_EFFORT = re.compile(r"([가-힣]{2,6})\s*(?:도|를|을)?\s*열심히")

# 이상이 없다고 말한 항목. 진단으로 올리면 없는 병이 기록에 남는다.
_NORMAL = re.compile(r"괜찮|이상\s*없|문제\s*없|없으세|없어요")

# 용어 뒤 이만큼 안에서 "괜찮다"는 말이 나오면 그 항목을 가리킨 것으로 본다.
_NORMAL_WINDOW = 25


# 검사 이름을 앞 구간에서 이어받을 수 있는 시간. 의사는 "간 수치가 높죠" 하고
# 다음 숨에 "작년에는 76에 34였고" 를 잇는다. 이보다 멀면 다른 화제로 본다.
CARRY_WINDOW_MS = 30_000


@dataclass
class Measurement:
    test: str
    values: list[str]
    quote: str
    start_ms: int
    # 검사 이름이 이 구간에 없어 앞에서 이어받았다. 사람이 확인해야 한다.
    inferred: bool = False


@dataclass
class Facts:
    measurements: list[Measurement] = field(default_factory=list)
    diagnoses: list[tuple[str, int]] = field(default_factory=list)
    # 의사가 "괜찮다"고 말한 항목. 진단과 섞으면 없는 병이 기록에 남는다.
    normal: list[str] = field(default_factory=list)
    lifestyle: list[str] = field(default_factory=list)


def _doctor_sentences(
    utterances: list[Utterance], roles: dict[str, Role]
) -> list[tuple[Utterance, str]]:
    return [
        (utterance, sentence)
        for utterance in utterances
        if roles.get(utterance.speaker_tag) is Role.DOCTOR
        for sentence in split_sentences(utterance.text)
    ]


def extract(
    utterances: list[Utterance], roles: dict[str, Role], terms: Terminology
) -> Facts:
    """검사 수치·진단·생활 지도를 값으로 뽑는다."""
    facts = Facts()
    seen_tests: set[tuple[str, str]] = set()
    seen_conditions: set[str] = set()
    seen_advice: set[str] = set()
    last_test: str | None = None
    last_at = 0

    # 수치는 구간 단위로 본다. 문장으로 쪼개면 "간 수치가 좀 높죠"와
    # "40이 정상이인데"가 갈라져 검사 이름과 숫자가 서로 다른 조각에 남는다.
    for utterance in utterances:
        if roles.get(utterance.speaker_tag) is not Role.DOCTOR:
            continue
        compact = utterance.text.replace(" ", "")
        named = next(
            (t for t in sorted(terms.tests, key=len, reverse=True) if t in compact), None
        )
        # 이름은 숫자가 없는 구간에서도 기억해 둔다. "간 수치가 좀 높죠" 처럼
        # 이름만 대고 수치는 다음 숨에 말하는 일이 흔하다.
        if named:
            last_test, last_at = named, utterance.start_ms

        numbers = _NUMBER.findall(utterance.text)
        if not numbers:
            continue
        if not named and not (
            last_test and utterance.start_ms - last_at <= CARRY_WINDOW_MS
        ):
            continue

        test = named or last_test
        key = (test, ",".join(numbers))
        if key in seen_tests:
            continue
        seen_tests.add(key)
        facts.measurements.append(
            Measurement(
                test, numbers, utterance.text.strip(), utterance.start_ms, inferred=not named
            )
        )
        last_at = utterance.start_ms

    for utterance, sentence in _doctor_sentences(utterances, roles):
        compact = sentence.replace(" ", "")

        for condition in terms.conditions:
            position = compact.find(condition)
            if position == -1 or condition in seen_conditions:
                continue
            seen_conditions.add(condition)
            after = compact[position + len(condition) : position + len(condition) + _NORMAL_WINDOW]
            if _NORMAL.search(after):
                facts.normal.append(condition)
            else:
                facts.diagnoses.append((condition, utterance.start_ms))

        for match in _REDUCE.finditer(sentence):
            target = match.group(1).strip()
            if len(target) >= 1 and target not in seen_advice:
                seen_advice.add(target)
                facts.lifestyle.append(f"{target} 줄이기")
        for match in _EFFORT.finditer(sentence):
            target = match.group(1).strip()
            if target not in seen_advice:
                seen_advice.add(target)
                facts.lifestyle.append(target)

    facts.diagnoses.sort(key=lambda d: d[1])
    return facts
