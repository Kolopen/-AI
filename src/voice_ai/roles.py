"""화자 역할 판정.

발화 하나하나를 분류하면 "네", "음..." 같은 짧은 응답에서 계속 흔들리므로,
화자별로 발화를 전부 모아 한 번에 판정한다. 규칙 기반 베이스라인이며
온프레미스 LLM이 붙으면 이 점수를 사전 확률로 넘겨준다.
"""

from __future__ import annotations

import re
from itertools import permutations

from .models import Method, Role, SpeakerProfile, SpeakerRole

Signal = tuple[re.Pattern[str], float]

# 의사: 처방·진단 어휘와 지시형 종결이 결정적
DOCTOR_SIGNALS: list[Signal] = [
    (re.compile(r"처방|복용|투약|항생제|주사|시술|수술|염증|소견|진단"), 3.0),
    (re.compile(r"검사|수치|초음파|엑스레이|CT|MRI|재검|경과|추적"), 2.5),
    (re.compile(r"(드시|하시|해보시|드셔보|쉬시)(면|고|는|세요|십시오)"), 2.0),
    (re.compile(r"하시면 됩니다|하셔야|중단하시|끊으시|유지하시"), 2.5),
    (re.compile(r"보시면|결과가|결과를 보니|지금 보니|말씀드리면"), 1.5),
    (re.compile(r"언제부터|어디가|어떻게 아프|불편하신|아프신|드시고 계신"), 2.0),
]

# 매니저: 대리 질문 프레임과 3인칭 환자 지칭
MANAGER_SIGNALS: list[Signal] = [
    (re.compile(r"여쭤|여쭙|여쭈"), 3.5),
    (re.compile(r"보호자(분|님)|가족(분|님)|따님|아드님|자제분"), 3.0),
    (re.compile(r"대신 (여쭤|물어|말씀)|전달(드리|받)|확인 부탁"), 3.0),
    (re.compile(r"어머님|아버님|환자분|할머님|할아버님"), 2.0),
    (re.compile(r"메모|적어|기록|정리해서"), 2.0),
    (re.compile(r"혹시|괜찮으실까요|가능할까요|될까요"), 1.0),
    (re.compile(r"다음 진료|예약|언제 (다시|오면)|접수"), 1.0),
]

# 환자: 1인칭 증상 서술
PATIENT_SIGNALS: list[Signal] = [
    (re.compile(r"제가|저는|저도|저한테"), 2.0),
    (re.compile(r"아파요|아픕니다|아프고|불편해요|어지러|쑤시|저려|답답"), 3.0),
    (re.compile(r"잘 모르겠|그냥|글쎄|좀 그래요"), 1.5),
]

NURSE_SIGNALS: list[Signal] = [
    (re.compile(r"체온|혈압|재겠습니다|수납|접수|대기|성함|들어오세요"), 3.0),
]

SIGNALS: dict[Role, list[Signal]] = {
    Role.DOCTOR: DOCTOR_SIGNALS,
    Role.MANAGER: MANAGER_SIGNALS,
    Role.PATIENT: PATIENT_SIGNALS,
    Role.NURSE: NURSE_SIGNALS,
}

# 한 명씩만 존재한다고 보는 역할. 나머지는 UNKNOWN으로 남긴다.
EXCLUSIVE_ROLES = (Role.DOCTOR, Role.MANAGER, Role.PATIENT, Role.NURSE)

QUESTION_PATTERN = re.compile(r"\?|까요|나요|가요|습니까|ㅂ니까|인가요|은지요")


def question_ratio(profile: SpeakerProfile) -> float:
    if not profile.utterance_count:
        return 0.0
    hits = sum(1 for u in profile.utterances if QUESTION_PATTERN.search(u.text))
    return hits / profile.utterance_count


def lexical_score(profile: SpeakerProfile, role: Role) -> float:
    text = profile.text
    raw = sum(weight * len(pattern.findall(text)) for pattern, weight in SIGNALS[role])
    return raw / max(1, profile.utterance_count)


def score_speaker(profile: SpeakerProfile) -> dict[Role, float]:
    """화자 하나에 대해 역할별 점수를 낸다."""
    scores = {role: lexical_score(profile, role) for role in EXCLUSIVE_ROLES}
    q_ratio = question_ratio(profile)

    # 의사는 설명이 길다. 매니저와 환자는 짧다.
    if profile.mean_chars > 40:
        scores[Role.DOCTOR] += 1.0
    elif profile.mean_chars < 15:
        scores[Role.PATIENT] += 0.5

    # 매니저는 질문 비율이 높고, 환자는 낮다.
    scores[Role.MANAGER] += q_ratio * 2.0
    scores[Role.PATIENT] -= q_ratio * 1.0

    return scores


def _assignment_score(
    profiles: list[SpeakerProfile],
    matrix: dict[str, dict[Role, float]],
    assignment: tuple[Role, ...],
) -> float:
    return sum(matrix[p.speaker_tag][role] for p, role in zip(profiles, assignment))


def classify(
    profiles: list[SpeakerProfile],
    *,
    manager_speaker_tag: str | None = None,
) -> list[SpeakerRole]:
    """화자 목록에 역할을 배정한다.

    manager_speaker_tag가 주어지면(성문 매칭 성공) 그 화자는 매니저로 고정하고
    나머지만 배정한다. 후보가 줄어드는 만큼 정확도가 크게 오른다.
    """
    if not profiles:
        return []

    matrix = {p.speaker_tag: score_speaker(p) for p in profiles}

    locked: list[SpeakerRole] = []
    open_profiles = list(profiles)

    if manager_speaker_tag is not None:
        for profile in list(open_profiles):
            if profile.speaker_tag == manager_speaker_tag:
                locked.append(
                    SpeakerRole(
                        speaker_tag=profile.speaker_tag,
                        role=Role.MANAGER,
                        confidence=1.0,
                        method=Method.VOICEPRINT,
                        scores={r.value: round(s, 3) for r, s in matrix[profile.speaker_tag].items()},
                    )
                )
                open_profiles.remove(profile)

    candidate_roles = [r for r in EXCLUSIVE_ROLES if r is not Role.MANAGER or manager_speaker_tag is None]
    # 화자가 역할 수보다 많으면 남는 자리는 UNKNOWN으로 채운다.
    padded = candidate_roles + [Role.UNKNOWN] * max(0, len(open_profiles) - len(candidate_roles))

    best: tuple[Role, ...] = ()
    best_total = float("-inf")
    runner_up = float("-inf")

    for assignment in set(permutations(padded, len(open_profiles))):
        total = _assignment_score(open_profiles, matrix, assignment)
        if total > best_total:
            runner_up, best_total, best = best_total, total, assignment
        elif total > runner_up:
            runner_up = total

    margin = best_total - runner_up if runner_up > float("-inf") else best_total
    confidence = _confidence_from_margin(margin)

    resolved = [
        SpeakerRole(
            speaker_tag=profile.speaker_tag,
            role=role,
            confidence=confidence,
            method=Method.TEXT_PATTERN,
            scores={r.value: round(s, 3) for r, s in matrix[profile.speaker_tag].items()},
        )
        for profile, role in zip(open_profiles, best)
    ]

    return sorted(locked + resolved, key=lambda s: s.speaker_tag)


def _confidence_from_margin(margin: float) -> float:
    """1·2위 배정안의 점수 차를 0~1 신뢰도로 눌러 담는다."""
    return round(min(0.99, max(0.30, margin / (margin + 2.0))) if margin > 0 else 0.30, 2)
