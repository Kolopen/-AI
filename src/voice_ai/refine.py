"""화자분리 오류 보정.

CLOVA는 세그먼트마다 화자를 하나만 붙이므로, 긴 의사 설명 중간에 끼어든
환자의 짧은 응답이 의사 발화로 흡수된다. 실제 앵커 녹음에서도
"운동 열심히 해야겠네요", "알겠습니다"가 의사 블록 안에 들어가 있었다.

역할이 확정된 뒤에 역방향으로 훑으면 이런 문장을 텍스트만으로 잡아낼 수 있다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Role, SpeakerProfile
from .roles import SIGNALS

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ForeignSentence:
    """배정된 역할과 어긋나는 화법이 잡힌 문장."""

    speaker_tag: str
    start_ms: int
    sentence: str
    assigned_role: Role
    suspected_role: Role
    score: float


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _sentence_score(sentence: str, role: Role) -> float:
    return sum(weight * len(pattern.findall(sentence)) for pattern, weight in SIGNALS[role])


def flag_foreign_sentences(
    profile: SpeakerProfile, assigned_role: Role, *, candidates: tuple[Role, ...] = ()
) -> list[ForeignSentence]:
    """배정된 역할과 다른 화자의 화법이 잡히는 문장을 표시한다.

    표시만 하고 재배정하지는 않는다. 한 문장으로 화자를 단정하기에는 근거가 얇고,
    잘못 옮기면 의사 발언이 환자 발언으로 뒤바뀌어 리포트가 틀어진다.
    운영자 검수와 성문 매칭이 붙은 뒤에 재배정을 판단한다.
    """
    if assigned_role not in SIGNALS:
        return []

    others = candidates or tuple(r for r in SIGNALS if r is not assigned_role)
    flagged: list[ForeignSentence] = []

    for utterance in profile.utterances:
        for sentence in split_sentences(utterance.text):
            own = _sentence_score(sentence, assigned_role)
            if own > 0:
                continue

            best_role, best_score = None, 0.0
            for role in others:
                score = _sentence_score(sentence, role)
                if score > best_score:
                    best_role, best_score = role, score

            if best_role is not None:
                flagged.append(
                    ForeignSentence(
                        speaker_tag=profile.speaker_tag,
                        start_ms=utterance.start_ms,
                        sentence=sentence,
                        assigned_role=assigned_role,
                        suspected_role=best_role,
                        score=best_score,
                    )
                )

    return flagged
