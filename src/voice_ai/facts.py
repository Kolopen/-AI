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
from .schedule import mask_broken_numbers, mask_dates, mask_periods
from .scores import Score
from .scores import extract as extract_scores
from .timeline import group_by_time, has_marker
from .terms import TermCorrection, corrected_text, phonetic_similarity

# 영문자에 붙은 숫자는 검사 이름의 일부다. "비타민 B12"의 12 는 수치가 아니다.
# 앞자리를 막으면 뒷자리부터 다시 걸리므로("b12" 의 2) 숫자도 함께 막는다.
_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+")

# "술도 줄이셔야", "체중 줄이시면", "허리를 줄이셔야"
# -시- 는 어미와 만나면 "셔"로 줄어든다(줄이시어야 → 줄이셔야). 둘 다 받아야 한다.
_REDUCE = re.compile(r"([가-힣]{1,6}?)\s*(?:도|를|을|은|는)?\s*(?:줄이|끊으|빼)(?:시|셔)")
# "운동 열심히 하셔서", "운동도 열심히"
_EFFORT = re.compile(r"([가-힣]{2,6})\s*(?:도|를|을)?\s*열심히")

# 복용 주기. 값만 필요하므로 걸린 부분만 떼어낸다. 문장을 통째로 실으면
# "하루에 하나하시면만 드시면 되니까" 처럼 깨진 어미가 따라온다.
_SCHEDULE = re.compile(
    r"하루(?:에)?\s*(?:하나|한\s*개|한\s*알|한\s*번|[한두세네\d]\s*(?:번|알|개|정|캡슐))"
    r"|아침저녁|하루\s*[한두세네\d]\s*끼|식후|식전|자기\s*전|매일"
)

# 처방 기간. 뒤·후가 붙으면 다음 방문까지의 간격이지 처방 기간이 아니다.
# "4주 뒤에 보겠습니다"가 4주치 처방으로 잡히던 자리다.
_DURATION = re.compile(
    r"(?:[\d]+|한|두|세|네|다섯|여섯|열)\s*(?:달|개월|주일|주)\s*(?:분|치)?(?!\s*(?:뒤|후))"
)

# 이상이 없다고 말한 항목. 진단으로 올리면 없는 병이 기록에 남는다.
_NORMAL = re.compile(r"괜찮|이상\s*없|문제\s*없|없으세|없어요")

# 아직 아니라고 말한 항목. "치매라고 진단하지는 않습니다" 를 진단에 올리면
# 받지도 않은 진단이 기록에 남는다. 그렇다고 정상으로 올릴 수도 없다.
# 의사가 한 말은 "아직 아니다" 이지 "괜찮다" 가 아니다.
_UNCONFIRMED = re.compile(r"아니|않")

# 용어 뒤 이만큼 안에서 "괜찮다"는 말이 나오면 그 항목을 가리킨 것으로 본다.
_NORMAL_WINDOW = 25

# 생활 지도 대상을 표준 이름으로 모을 때 쓰는 기준. 전사가 흔들려도 같은 것을
# 가리킨다고 볼 만한 선이다. "허리리"/"허리" 0.80, "운동부"/"운동" 0.857 이고,
# 뜻이 다른 "야식"/"운동" 은 0.18 이라 섞이지 않는다.
LIFESTYLE_SIMILARITY = 0.75


def _canonical(target: str, vocabulary: frozenset[str]) -> str:
    """전사가 흔들린 대상을 표준 이름으로 모은다.

    "허리리 줄이기"와 "허리 줄이기"가 따로 남으면 같은 지도가 둘로 보인다.
    사전에 없는 말은 그대로 둔다. 의사가 무엇을 줄이라 했는지 버릴 수는 없다.
    """
    best, score = None, LIFESTYLE_SIMILARITY
    for word in vocabulary:
        similarity = phonetic_similarity(target, word)
        if similarity >= score:
            best, score = word, similarity
    return best or target


# 검사 이름을 앞 구간에서 이어받을 수 있는 시간. 의사는 "간 수치가 높죠" 하고
# 다음 숨에 "작년에는 76에 34였고" 를 잇는다. 이보다 멀면 다른 화제로 본다.
CARRY_WINDOW_MS = 30_000


@dataclass
class Measurement:
    test: str
    values: list[str]
    # 시점별로 가른 것. {"정상": ["40"], "작년": ["76","34"], "이번": ["67","23"]}
    # 나열만 해서는 좋아졌는지 나빠졌는지 읽을 수 없다.
    by_time: dict[str, list[str]]
    quote: str
    start_ms: int
    # 검사 이름이 이 구간에 없어 앞에서 이어받았다. 사람이 확인해야 한다.
    inferred: bool = False
    # 시점 표현을 다른 엔진 전사에서 빌렸다.
    time_from_alternate: bool = False


@dataclass
class Facts:
    measurements: list[Measurement] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    schedule: list[str] = field(default_factory=list)
    duration: list[str] = field(default_factory=list)
    diagnoses: list[tuple[str, int]] = field(default_factory=list)
    # 의사가 "괜찮다"고 말한 항목. 진단과 섞으면 없는 병이 기록에 남는다.
    normal: list[str] = field(default_factory=list)
    # 의사가 "아직 아니다"라고 말한 항목. 진단도 정상도 아니다.
    unconfirmed: list[str] = field(default_factory=list)
    # 점수로 말한 검사. 이름 없이 오므로 사전이 아니라 "점" 단위로 잡는다.
    scores: list[Score] = field(default_factory=list)
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


def _inside_test(drug: str, compact: str, terms: Terminology) -> bool:
    """약 이름이 같은 자리에 나온 검사 이름에 통째로 들어 있는지 본다."""
    folded = compact.casefold()
    return any(
        test != drug and drug in test and test.casefold() in folded for test in terms.tests
    )


# 수치를 말한 것이지 복용이 아니다. 검사 이름이 사전에 그대로 있으면
# _inside_test 가 잡지만, 전사가 깨지면 그 그물을 빠져나간다. 실제 녹음에서
# "비타민 B12 수치" 가 "비타민이 기십 이 점수치" 로 흘렀고, 남은 "비타민" 이
# 복용 약으로 리포트에 올라갔다. 먹지 않는 약이 진료 기록에 남는 자리다.
#
# 수치·농도·레벨만 본다. "정상" 까지 넣으면 "혈압약은 정상적으로 드세요" 가
# 걸려 진짜 복용이 빠진다.
_LAB_VALUE = re.compile(r"수치|농도|레벨")
_LAB_WINDOW = 14


def is_lab_value(drug: str, compact: str) -> bool:
    """나온 자리가 모두 수치를 말하는 자리인지.

    한 번이라도 복용으로 말했으면 약이다. "비타민 수치는 정상이고 비타민은
    계속 드세요" 를 통째로 버리면 진짜 복용이 빠진다.
    """
    found = False
    for match in re.finditer(re.escape(drug), compact):
        found = True
        after = compact[match.end() : match.end() + _LAB_WINDOW]
        if not _LAB_VALUE.search(after):
            return False
    return found


def extract(
    utterances: list[Utterance],
    roles: dict[str, Role],
    terms: Terminology,
    corrections: list[TermCorrection] | None = None,
    alternate: list[Utterance] | None = None,
) -> Facts:
    """검사 수치·진단·복용·생활 지도를 값으로 뽑는다.

    교정을 먼저 반영한다. "간보제"는 사전에 없어 약품으로 알아보지 못하는데,
    바로 그것이 교정이 잡아낸 오인식이다.
    """
    if corrections:
        utterances = [
            Utterance(
                u.speaker_tag, u.start_ms, u.end_ms, corrected_text(u.text, corrections)
            )
            for u in utterances
        ]
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
        compact = utterance.text.replace(" ", "").casefold()
        named = next(
            (t for t in sorted(terms.tests, key=len, reverse=True) if t.casefold() in compact),
            None,
        )
        # 이름은 숫자가 없는 구간에서도 기억해 둔다. "간 수치가 좀 높죠" 처럼
        # 이름만 대고 수치는 다음 숨에 말하는 일이 흔하다.
        if named:
            last_test, last_at = named, utterance.start_ms

        # 날짜·시각·기간의 숫자는 검사 수치가 아니다. "10월 20일에 오세요"가
        # 10, 20 두 개의 수치로, "4주 뒤에 보겠습니다"가 4로 잡히던 자리다.
        spoken = mask_broken_numbers(mask_periods(mask_dates(utterance.text)))
        numbers = _NUMBER.findall(spoken)
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
        # 시점 표현이 이 전사에 없으면 다른 엔진 것을 본다. 실제 녹음에서
        # SenseVoice 는 "정상"을 "정는"으로 흘렸는데 moonshine 은 "정산"으로
        # 들어 살릴 수 있었다. 어미는 무너져도 시점 단어는 대체로 남는다.
        source, borrowed = spoken, False
        if not has_marker(source) and alternate:
            nearby = " ".join(
                mask_broken_numbers(mask_periods(mask_dates(other.text)))
                for other in alternate
                if min(other.end_ms, utterance.end_ms) - max(other.start_ms, utterance.start_ms) > 0
            )
            if has_marker(nearby):
                source, borrowed = nearby, True

        facts.measurements.append(
            Measurement(
                test,
                numbers,
                group_by_time(source),
                utterance.text.strip(),
                utterance.start_ms,
                inferred=not named,
                time_from_alternate=borrowed,
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
            elif _UNCONFIRMED.search(after):
                facts.unconfirmed.append(condition)
            else:
                facts.diagnoses.append((condition, utterance.start_ms))

        for drug in terms.drugs:
            if drug not in compact or drug in facts.drugs:
                continue
            # "비타민 B12 수치"의 비타민은 약이 아니라 검사 이름의 일부다.
            if _inside_test(drug, compact, terms) or is_lab_value(drug, compact):
                continue
            facts.drugs.append(drug)
        for match in _SCHEDULE.finditer(sentence):
            value = " ".join(match.group().split())
            if value not in facts.schedule:
                facts.schedule.append(value)
        for match in _DURATION.finditer(sentence):
            value = " ".join(match.group().split())
            # "두 달"과 "두 달분"은 같은 말이다. 긴 쪽만 남긴다.
            if any(value.startswith(kept) for kept in facts.duration):
                facts.duration = [k for k in facts.duration if not value.startswith(k)]
            elif any(kept.startswith(value) for kept in facts.duration):
                continue
            facts.duration.append(value)

        for match in _REDUCE.finditer(sentence):
            target = _canonical(match.group(1).strip(), terms.lifestyle)
            if target and target not in seen_advice:
                seen_advice.add(target)
                facts.lifestyle.append(f"{target} 줄이기")
        for match in _EFFORT.finditer(sentence):
            target = _canonical(match.group(1).strip(), terms.lifestyle)
            if target not in seen_advice:
                seen_advice.add(target)
                facts.lifestyle.append(target)

    facts.diagnoses.sort(key=lambda d: d[1])
    facts.scores = extract_scores(utterances, roles, terms)
    return facts
