"""의사 발언에서 리포트 초안을 뽑는다.

요약을 생성하지 않고 원문 문장을 발췌한다. 의료 기록에서는 이쪽이 안전하다.
전사에 이미 "간 보호제"가 "관보제"로, "신장"이 "심장"으로 잘못 실리는 일이 있는데,
생성형에 넘기면 그런 오류가 그럴듯한 다른 말로 매끄럽게 바뀌어 알아볼 수 없게 된다.
발췌는 틀려도 원문 그대로 틀리므로 운영자가 바로 알아본다.

발췌한 문장에는 타임스탬프를 달아 원문 구간으로 되짚을 수 있게 한다.
`summary`는 채우지 않는다. 자연어 생성이 필요한 유일한 항목이고, 매니저가 쓴다.
"""

from __future__ import annotations

import re

from .models import ReportDraft, Role, Utterance
from .roles import split_sentences

# 검사 수치와 진단. 숫자가 있거나 의학 용어가 있는 의사 문장.
_FINDING = re.compile(r"정상|수치|높|낮|검사|결과|소견|때문에|가능성")
_NUMBER = re.compile(r"\d")

# 복용 주기.
_SCHEDULE = re.compile(
    r"하루(에)?\s*(한|두|세|네|\d)\s*(번|알|개|정|캡슐)"
    r"|매일|아침|점심|저녁|식후|식전|자기\s*전|하나씩|한\s*알씩"
)

# 처방 기간.
_DURATION = re.compile(r"(\d+|한|두|세|네|다섯|여섯|열)\s*(달|개월|주일|주|일)\s*(분|치|씩)?")

# 다음 방문.
_NEXT_VISIT = re.compile(
    r"(\d+|한|두|세|네|여섯)\s*(달|개월|주|주일|년)\s*(뒤|후|있다가)"
    r"|재검|다시\s*(오|보|와)|다음에\s*(오|뵈|봐)|다음\s*진료|경과\s*(보|관찰)"
)


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


def _collect(pairs: list[tuple[Utterance, str]], pattern: re.Pattern[str]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for utterance, sentence in pairs:
        if not pattern.search(sentence) or sentence in seen:
            continue
        seen.add(sentence)
        lines.append(f"{_stamp(utterance)} {sentence}")
    return "\n".join(lines)


def build_report_draft(
    utterances: list[Utterance],
    roles: dict[str, Role],
    *,
    drug_terms: frozenset[str] = frozenset(),
) -> ReportDraft:
    """의사 발언에서 리포트 항목별로 관련 문장을 발췌한다.

    약품란에는 약 이름만 넣는다. 질환명과 검사명이 섞이면 리포트가 못 쓰게 된다.
    """
    pairs = _doctor_sentences(utterances, roles)

    findings = [
        (u, s) for u, s in pairs if _FINDING.search(s) and (_NUMBER.search(s) or "수치" in s)
    ]

    # 전사는 "간 보호제", 사전은 "간보호제"처럼 띄어쓰기가 어긋나므로 공백을 지우고 맞춘다.
    spoken = re.sub(r"\s+", "", " ".join(sentence for _, sentence in pairs))
    names = sorted(term for term in drug_terms if re.sub(r"\s+", "", term) in spoken)

    return ReportDraft(
        treatment_notes=_collect(findings, _FINDING),
        medication_name=", ".join(names),
        medication_schedule_note=_collect(pairs, _SCHEDULE),
        medication_notes=_collect(pairs, _DURATION),
        next_visit_note=_collect(pairs, _NEXT_VISIT),
        # 자연어 생성이 필요한 유일한 항목. 매니저가 확인하며 쓴다.
        summary="",
    )
