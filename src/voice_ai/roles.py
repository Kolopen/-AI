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
    (re.compile(r"(드시|드셔|하시|해보시|드셔보|쉬시|주시)(면|고|는|세요|십시오)"), 2.0),
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

# 환자: 1인칭 서술과 수용 반응.
# 재진 상담에서는 환자가 증상을 호소하기보다 설명을 듣고 짧게 반응하는 쪽에 가깝다.
PATIENT_SIGNALS: list[Signal] = [
    (re.compile(r"제가|저는|저도|저한테"), 2.0),
    (re.compile(r"아파요|아픕니다|아프고|불편해요|어지러|쑤시|저려|답답"), 3.0),
    # "같기는 해요"는 의사의 추측 표현이라 자기 행위 서술에서 제외한다.
    (re.compile(r"(?<!같)기는 (해요|합니다|하는데|한데)"), 2.0),
    (re.compile(r"알겠습니다|그렇군요|해야겠네요|그래야겠"), 2.0),
    (re.compile(r"잘 모르겠|그냥|글쎄|좀 그래요"), 1.5),
]

# 비의료인 화자의 기본값. 매니저 화법이 없으면 환자로 수렴시킨다.
PATIENT_PRIOR = 0.5

# 이보다 내용이 적고 어휘 신호가 없으면 텍스트만으로는 판정하지 않는다.
LOW_SIGNAL_CHARS = 60

# 이 아래 신뢰도는 운영자 검수 큐로 올린다. 검수 결과가 학습 데이터가 된다.
REVIEW_THRESHOLD = 0.6

# 이 점수에 못 미치면 의사가 녹음에 없거나 전사가 망가진 것으로 본다.
DOCTOR_MIN_SCORE = 2.0

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


def has_lexical_evidence(profile: SpeakerProfile) -> bool:
    return any(lexical_score(profile, role) > 0 for role in EXCLUSIVE_ROLES)


def score_speaker(profile: SpeakerProfile) -> dict[Role, float]:
    """화자 하나에 대해 역할별 점수를 낸다.

    구조 특징(발화 길이, 질문 비율)은 어휘 근거를 보강만 하고 역할을 새로 만들지는 않는다.
    질문을 많이 한다고 매니저인 것이 아니라 대리 질문 화법이 있어야 매니저다.
    """
    scores = {role: lexical_score(profile, role) for role in EXCLUSIVE_ROLES}
    scores[Role.PATIENT] += PATIENT_PRIOR

    # 의사는 설명이 길다.
    if profile.mean_chars > 40:
        scores[Role.DOCTOR] += 1.0

    # 대리 질문 화법이 이미 잡힌 화자에 한해 질문 비율로 힘을 실어준다.
    if scores[Role.MANAGER] > 0:
        scores[Role.MANAGER] += question_ratio(profile) * 2.0

    return scores


def doctor_score(profile: SpeakerProfile) -> float:
    score = lexical_score(profile, Role.DOCTOR)
    if profile.mean_chars > 40:
        score += 1.0
    return score


def identify_doctor(profiles: list[SpeakerProfile]) -> tuple[str | None, float]:
    """의사를 먼저, 독립적으로 찾는다.

    가장 중요한 판정이므로 다른 화자의 모호함이 여기에 영향을 주면 안 된다.
    배정 최적화에 섞으면 환자·매니저 쪽 점수가 흔들릴 때 의사 판정까지 같이 흔들린다.
    """
    if not profiles:
        return None, 0.0

    ranked = sorted((doctor_score(p), p.speaker_tag) for p in profiles)
    top_score, top_tag = ranked[-1]
    if top_score < DOCTOR_MIN_SCORE:
        return None, 0.0

    runner_up = ranked[-2][0] if len(ranked) > 1 else 0.0
    return top_tag, _margin_to_confidence(top_score - runner_up)


def _non_doctor_roles(manager_locked: bool) -> list[Role]:
    roles = [Role.PATIENT, Role.NURSE] if manager_locked else [Role.MANAGER, Role.PATIENT, Role.NURSE]
    return roles


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
    merge_non_doctor: bool = False,
) -> tuple[list[SpeakerRole], list[str]]:
    """화자 목록에 역할을 배정하고 (배정 결과, 경고) 를 돌려준다.

    의사를 먼저 확정하고 나머지를 배정한다. 의사와 비의사를 가르는 것이
    환자와 매니저를 가르는 것보다 중요하므로 두 판정을 분리한다.

    manager_speaker_tag가 주어지면(성문 매칭 성공) 그 화자는 매니저로 고정한다.

    merge_non_doctor는 화자분리가 한 사람을 여러 명으로 쪼갠 것이 확인됐을 때 쓴다.
    비의사 화자를 한 사람으로 합쳐 판정하고, 합쳐진 태그를 merged_from에 남긴다.
    텍스트만으로 동일인 여부를 알 수는 없으므로 호출자가 판단해 넘겨야 한다.
    """
    if not profiles:
        return [], []

    matrix = {p.speaker_tag: score_speaker(p) for p in profiles}
    resolved: list[SpeakerRole] = []
    warnings: list[str] = []

    doctor_tag, doctor_confidence = identify_doctor(profiles)
    if doctor_tag is None:
        warnings.append(
            "의사를 특정하지 못했습니다. 진료 어휘가 잡히지 않아 녹음 구간이나 "
            "전사 품질을 먼저 확인해야 합니다."
        )
    else:
        resolved.append(
            SpeakerRole(
                speaker_tag=doctor_tag,
                role=Role.DOCTOR,
                confidence=doctor_confidence,
                method=Method.TEXT_PATTERN,
                scores={r.value: round(s, 3) for r, s in matrix[doctor_tag].items()},
                needs_review=doctor_confidence < REVIEW_THRESHOLD,
            )
        )

    rest = [p for p in profiles if p.speaker_tag != doctor_tag]
    if not rest:
        return sorted(resolved, key=lambda s: s.speaker_tag), warnings

    if merge_non_doctor:
        resolved.extend(_classify_merged(rest, manager_speaker_tag))
        return sorted(resolved, key=lambda s: s.speaker_tag), warnings

    locked = [p for p in rest if p.speaker_tag == manager_speaker_tag]
    open_profiles: list[SpeakerProfile] = []

    for profile in rest:
        scores = {r.value: round(s, 3) for r, s in matrix[profile.speaker_tag].items()}
        if profile.speaker_tag == manager_speaker_tag:
            resolved.append(
                SpeakerRole(profile.speaker_tag, Role.MANAGER, 1.0, Method.VOICEPRINT, scores)
            )
        elif not has_lexical_evidence(profile) and profile.char_count < LOW_SIGNAL_CHARS:
            resolved.append(
                SpeakerRole(
                    profile.speaker_tag, Role.UNKNOWN, 0.2, Method.TEXT_PATTERN, scores, needs_review=True
                )
            )
            warnings.append(
                f"{profile.speaker_tag}: 발화가 {profile.char_count}자뿐이고 역할 신호가 없어 "
                "텍스트만으로 판정할 수 없습니다. 다른 비의사 화자와 같은 사람이 "
                "쪼개져 나온 것일 수 있으니 확인이 필요합니다."
            )
        else:
            open_profiles.append(profile)

    if open_profiles:
        candidates = _non_doctor_roles(bool(locked))
        padded = candidates + [Role.UNKNOWN] * max(0, len(open_profiles) - len(candidates))
        best = max(
            set(permutations(padded, len(open_profiles))),
            key=lambda a: _assignment_score(open_profiles, matrix, a),
        )

        for profile, role in zip(open_profiles, best):
            scores = matrix[profile.speaker_tag]
            confidence = _speaker_confidence(scores, role)
            resolved.append(
                SpeakerRole(
                    speaker_tag=profile.speaker_tag,
                    role=role,
                    confidence=confidence,
                    method=Method.TEXT_PATTERN,
                    scores={r.value: round(s, 3) for r, s in scores.items()},
                    needs_review=confidence < REVIEW_THRESHOLD,
                )
            )

    return sorted(resolved, key=lambda s: s.speaker_tag), warnings


def _classify_merged(
    rest: list[SpeakerProfile], manager_speaker_tag: str | None
) -> list[SpeakerRole]:
    """비의사 화자를 한 사람으로 합쳐 판정하고, 원래 태그마다 같은 결과를 돌려준다.

    태그별로 결과를 내야 하위 단계(Q&A 페어링)가 그대로 동작한다.
    """
    if manager_speaker_tag and any(p.speaker_tag == manager_speaker_tag for p in rest):
        return [
            SpeakerRole(p.speaker_tag, Role.MANAGER, 1.0, Method.VOICEPRINT, {})
            for p in rest
        ]

    merged = SpeakerProfile(
        speaker_tag="merged_non_doctor",
        utterances=[u for p in rest for u in p.utterances],
    )
    scores = score_speaker(merged)
    role = max(_non_doctor_roles(False), key=lambda r: scores[r])
    confidence = _speaker_confidence(scores, role)
    tags = sorted(p.speaker_tag for p in rest)

    return [
        SpeakerRole(
            speaker_tag=tag,
            role=role,
            confidence=confidence,
            method=Method.ELIMINATION,
            scores={r.value: round(s, 3) for r, s in scores.items()},
            needs_review=confidence < REVIEW_THRESHOLD,
            merged_from=[t for t in tags if t != tag],
        )
        for tag in tags
    ]


def _speaker_confidence(scores: dict[Role, float], assigned: Role) -> float:
    """배정된 역할과 그 화자의 차순위 역할 사이의 점수 차로 낸다.

    전역 배정 점수 차를 쓰면 판정 불가 화자 하나가 다른 화자의 확신까지 끌어내린다.
    """
    if assigned is Role.UNKNOWN:
        return 0.2
    others = [s for r, s in scores.items() if r is not assigned]
    return _margin_to_confidence(scores.get(assigned, 0.0) - (max(others) if others else 0.0))


def _margin_to_confidence(margin: float) -> float:
    """1위와 2위의 점수 차를 0~1 신뢰도로 눌러 담는다."""
    return round(min(0.99, max(0.30, margin / (margin + 2.0))) if margin > 0 else 0.30, 2)
