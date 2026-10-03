"""의사 발언에서 리포트 초안을 뽑는다.

요약을 생성하지 않고 원문 문장을 발췌한다. 의료 기록에서는 이쪽이 안전하다.
전사에 이미 "간 보호제"가 "관보제"로, "신장"이 "심장"으로 잘못 실리는 일이 있는데,
생성형에 넘기면 그런 오류가 그럴듯한 다른 말로 매끄럽게 바뀌어 알아볼 수 없게 된다.
발췌는 틀려도 원문 그대로 틀리므로 운영자가 바로 알아본다.

발췌한 문장에는 타임스탬프를 달아 원문 구간으로 되짚을 수 있게 한다.
`summary`는 채우지 않는다. 자연어 생성이 필요한 유일한 항목이고, 매니저가 쓴다.
"""

from __future__ import annotations

import datetime as dt
import re

from .models import ReportDraft, Role, Utterance
from .roles import split_sentences
from .facts import is_lab_value
from .schedule import find_next_visit, mask_dates

# 검사 수치와 진단. 숫자가 있거나 의학 용어가 있는 의사 문장.
_FINDING = re.compile(r"정상|수치|높|낮|검사|결과|소견|때문에|가능성")
_NUMBER = re.compile(r"\d")

# 진료의 결론. 숫자가 없어서 위 그물을 빠져나가지만 리포트에 가장 필요한
# 말이다. 실제 녹음에서 이런 문장들이 통째로 빠졌다.
#
#   오늘은 기억력에 대한 약을 처방하지 않겠습니다
#   검사 결과를 확인한 뒤 치료 방향을 논의하겠습니다
#   오늘은 MRI 를 예약하고 결과를 보고 추가로 설명드리겠습니다
#   점수 하나만으로 치매라고 진단하지는 않습니다
#
# 의사가 자기 행위를 말하는 자리라 어미까지 묶어서 본다. "진단" 만 보면
# 환자의 "치매 진단이 뭔가요" 가 걸린다.
_PLAN = re.compile(
    r"(?:진단|처방|치료|평가|검사|예약|촬영|복용|조절|추적|관찰|상의|논의|설명)"
    r"[가-힣\s]{0,6}"
    r"(?:하겠|드리겠|보겠|겠습니다|합니다|습니다|하지는|하지\s*않|않겠|해야|하세요|드립니다)"
)

# 복용 주기.
_SCHEDULE = re.compile(
    r"하루(에)?\s*(한|두|세|네|\d)\s*(번|알|개|정|캡슐)"
    r"|매일|아침|점심|저녁|식후|식전|자기\s*전|하나씩|한\s*알씩"
)

# 처방 기간. "일"은 숫자와 분·치가 둘 다 붙어야 센다. "불편한 일이", "잊어버린
# 일" 처럼 관형사 뒤의 의존명사 "일"이 "한 일"로 걸려 들어오기 때문이다.
# 뒤·후가 붙으면 다음 방문까지의 간격이지 처방 기간이 아니다.
_DURATION = re.compile(
    r"(?:\d+|한|두|세|네|다섯|여섯|열)\s*(?:달|개월|주일|주)\s*(?:분|치|씩)?(?!\s*(?:뒤|후))"
    r"|\d+\s*일\s*(?:분|치)"
)

# 다음 방문.
# 한자어 수도 받는다. 전사가 "4주 뒤에" 를 "사주 뒤에" 로 내놓는다. 사주나
# 이주는 다른 뜻이 있지만 뒤·후가 붙으면 기간 말고 읽을 길이 없다.
_NEXT_VISIT = re.compile(
    r"(\d+|한|두|세|네|여섯|일|이|삼|사|오|육|칠|팔|구|십)\s*(달|개월|주|주일|년)\s*(뒤|후|있다가)"
    r"|재검|재진|다시\s*(오|보|와)|다음에\s*(오|뵈|봐)|다음\s*진료|경과\s*(보|관찰)"
)


# findings 는 이미 걸러진 목록인데 _collect 가 같은 목록을 _FINDING 으로 한 번
# 더 거른다. 그래서 "점수 하나만으로 치매라고 진단하지는 않습니다" 처럼
# 검사·결과·수치가 없는 결론이 두 번째 체에서 떨어졌다. 두 그물을 합쳐 넘긴다.
_FINDING_OR_PLAN = re.compile(f"{_FINDING.pattern}|{_PLAN.pattern}")


def _stamp(utterance: Utterance) -> str:
    seconds = utterance.start_ms // 1000
    return f"[{seconds // 60:02d}:{seconds % 60:02d}]"


def _doctor_sentences(
    utterances: list[Utterance], roles: dict[str, Role]
) -> list[tuple[Utterance, str]]:
    return [
        (utterance, sentence)
        for utterance in utterances
        if roles.get(utterance.speaker_tag) is Role.DOCTOR
        for sentence in split_sentences(utterance.text)
    ]


def _collect(
    pairs: list[tuple[Utterance, str]],
    pattern: re.Pattern[str],
    *,
    mask: bool = False,
) -> str:
    """패턴에 걸리는 의사 문장을 시각과 함께 모은다.

    `mask` 를 켜면 날짜와 시각을 지우고 맞춘다. "10월 20일에 오세요"의 20일이
    처방 기간으로 잡히던 자리다. 내보내는 문장은 언제나 원문 그대로다.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for utterance, sentence in pairs:
        target = mask_dates(sentence) if mask else sentence
        if not pattern.search(target) or sentence in seen:
            continue
        seen.add(sentence)
        lines.append(f"{_stamp(utterance)} {sentence}")
    return "\n".join(lines)


def _inside_test(drug: str, spoken: str, test_terms: frozenset[str]) -> bool:
    """약 이름이 같은 진료에 나온 검사 이름에 통째로 들어 있는지 본다.

    "비타민 B12 수치가 정상"의 비타민은 처방한 약이 아니라 검사 이름이다.
    약품란에 올리면 처방하지 않은 약이 리포트에 남는다.
    """
    return any(
        test != drug and drug in test and re.sub(r"\s+", "", test).casefold() in spoken
        for test in test_terms
    )


def build_report_draft(
    utterances: list[Utterance],
    roles: dict[str, Role],
    *,
    drug_terms: frozenset[str] = frozenset(),
    test_terms: frozenset[str] = frozenset(),
    consult_date: dt.date | None = None,
) -> ReportDraft:
    """의사 발언에서 리포트 항목별로 관련 문장을 발췌한다.

    약품란에는 약 이름만 넣는다. 질환명과 검사명이 섞이면 리포트가 못 쓰게 된다.

    `consult_date`를 주면 "10월 20일날 오세요" 같은 문장에서 후속 예약 날짜를
    값으로 뽑아 `next_visit_at`에 넣는다. 말한 날짜에는 연도가 없으므로
    진료일이 있어야 연도를 정할 수 있다.
    """
    pairs = _doctor_sentences(utterances, roles)

    findings = [
        (u, s)
        for u, s in pairs
        if (_FINDING.search(s) and (_NUMBER.search(mask_dates(s)) or "수치" in s))
        or _PLAN.search(s)
    ]

    # 전사는 "간 보호제", 사전은 "간보호제"처럼 띄어쓰기가 어긋나므로 공백을 지우고 맞춘다.
    spoken = re.sub(r"\s+", "", " ".join(sentence for _, sentence in pairs)).casefold()
    names = sorted(
        term
        for term in drug_terms
        if re.sub(r"\s+", "", term).casefold() in spoken
        and not _inside_test(term, spoken, test_terms)
        # 전사가 검사 이름을 깨뜨리면 위 그물을 빠져나간다. "비타민 B12
        # 수치" 가 "비타민이 기십 이 점수치" 로 흘러 비타민이 복용 약으로
        # 올라갔다. 먹지 않는 약이 진료 기록에 남는 자리다.
        and not is_lab_value(re.sub(r"\s+", "", term).casefold(), spoken)
    )

    visit = (
        find_next_visit(pairs, consult_date=consult_date) if consult_date is not None else None
    )
    next_visit_note = _collect(pairs, _NEXT_VISIT)
    if visit is not None:
        # 날짜를 뽑아낸 문장은 메모에도 반드시 남긴다. 매니저가 근거를 봐야 한다.
        line = f"{_stamp(visit.utterance)} {visit.sentence}"
        if line not in next_visit_note:
            next_visit_note = f"{line}\n{next_visit_note}".rstrip()

    return ReportDraft(
        treatment_notes=_collect(findings, _FINDING_OR_PLAN),
        medication_name=", ".join(names),
        medication_schedule_note=_collect(pairs, _SCHEDULE, mask=True),
        medication_notes=_collect(pairs, _DURATION, mask=True),
        next_visit_note=next_visit_note,
        next_visit_at=visit.iso if visit is not None else None,
        # 자연어 생성이 필요한 유일한 항목. 매니저가 확인하며 쓴다.
        summary="",
    )
