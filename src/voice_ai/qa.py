"""매니저 질문 → 의사 답변 페어링.

동행 리포트의 실제 산출물이다. 사전 등록 질문이 있으면 추론이 아니라 대조가 되므로
정확도가 크게 오르고, 매니저가 빠뜨린 질문도 함께 잡아낼 수 있다.
"""

from __future__ import annotations

import re

from .models import QAPair, Role, Utterance
from .roles import PROXY_QUESTION, QUESTION_PATTERN

# 답변으로 인정할 최대 간격. 이보다 멀면 다른 화제로 넘어간 것으로 본다.
ANSWER_WINDOW_MS = 60_000

_QUESTION_MARK = re.compile(r"\?")


_NON_WORD = re.compile(r"[^가-힣A-Za-z0-9]")
_MATCH_THRESHOLD = 0.5


def _bigrams(text: str) -> set[str]:
    """음절 바이그램. 조사가 바뀌어도(약이랑/약을) 대부분 겹쳐 한국어에 잘 견딘다."""
    cleaned = _NON_WORD.sub("", text)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _coverage(registered_text: str, utterance_text: str) -> float:
    """등록 질문이 실제 발화에 얼마나 담겼는지.

    대칭 유사도를 쓰면 매니저 특유의 앞머리("보호자분이 여쭤봐 달라고 하셨는데요")가
    점수를 깎으므로, 등록 질문 쪽을 기준으로만 본다.
    """
    reference = _bigrams(registered_text)
    if not reference:
        return 0.0
    return len(reference & _bigrams(utterance_text)) / len(reference)


def match_pre_registered(question: str, registered: dict[str, str]) -> str | None:
    """실제 발화를 사전 등록 질문과 대조해 가장 가까운 항목의 id를 돌려준다."""
    best_id, best_score = None, 0.0
    for question_id, text in registered.items():
        score = _coverage(text, question)
        if score > best_score:
            best_id, best_score = question_id, score
    return best_id if best_score >= _MATCH_THRESHOLD else None


def pair_qa(
    utterances: list[Utterance],
    roles: dict[str, Role],
    *,
    registered_questions: dict[str, str] | None = None,
) -> tuple[list[QAPair], list[str]]:
    """시간순 발화에서 질문-답변 쌍을 뽑는다.

    반환값은 (페어 목록, 묻지 않은 사전질문 id 목록).
    """
    registered = registered_questions or {}
    pairs: list[QAPair] = []
    seq = 0

    # 물음표를 내놓는 전사라면 물음표만 믿는다. 어미로만 가르면 "기억이 잘 안
    # 나요" 가 "-나요" 질문으로 잡혀 엉뚱한 답변이 붙는다. SenseVoice 도
    # moonshine 도 물음표를 찍으므로 대부분 이쪽으로 간다.
    punctuated = any(_QUESTION_MARK.search(u.text) for u in utterances)
    marker = _QUESTION_MARK if punctuated else QUESTION_PATTERN

    def is_question(text: str) -> bool:
        # 대리 질문은 평서문으로 온다. 물음표도 의문 어미도 없지만 보호자가
        # 물은 것이고, 리포트에 가장 필요한 질문이기도 하다.
        return bool(marker.search(text) or PROXY_QUESTION.search(text))

    for index, utterance in enumerate(utterances):
        asker = roles.get(utterance.speaker_tag, Role.UNKNOWN)
        if asker not in (Role.MANAGER, Role.PATIENT):
            continue
        if not is_question(utterance.text):
            continue

        answer_parts: list[Utterance] = []
        for follow_up in utterances[index + 1 :]:
            if follow_up.start_ms - utterance.end_ms > ANSWER_WINDOW_MS:
                break
            speaker_role = roles.get(follow_up.speaker_tag, Role.UNKNOWN)
            if speaker_role is Role.DOCTOR:
                answer_parts.append(follow_up)
            elif answer_parts:
                break

        if not answer_parts:
            continue

        seq += 1
        answer_text = " ".join(part.text for part in answer_parts)
        pairs.append(
            QAPair(
                seq=seq,
                question=utterance.text,
                answer=answer_text,
                question_at_ms=utterance.start_ms,
                answer_at_ms=answer_parts[0].start_ms,
                asked_by=asker,
                answered_by=Role.DOCTOR,
                confidence=_pair_confidence(utterance, answer_parts[0]),
                pre_registered_question_id=match_pre_registered(utterance.text, registered),
            )
        )

    asked_ids = {p.pre_registered_question_id for p in pairs if p.pre_registered_question_id}
    unasked = [qid for qid in registered if qid not in asked_ids]
    return pairs, unasked


def _pair_confidence(question: Utterance, answer: Utterance) -> float:
    """질문 직후에 붙은 답변일수록 신뢰한다."""
    gap_ms = max(0, answer.start_ms - question.end_ms)
    return round(max(0.4, 1.0 - gap_ms / ANSWER_WINDOW_MS), 2)


def explain_qa(
    utterances: list[Utterance], roles: dict[str, Role]
) -> tuple[str, list[tuple[int, str, str, str]]]:
    """질문-답변이 안 붙는 이유를 발화별로 돌려준다.

    짝이 0건일 때 어디서 끊겼는지 알아야 고친다. 역할이 아니어서인지, 질문으로
    안 보여서인지, 뒤에 의사가 없어서인지는 눈으로 봐야 갈린다.

    반환값은 (판정 방식, [(시작 ms, 역할, 판정, 발화)]).
    """
    punctuated = any(_QUESTION_MARK.search(u.text) for u in utterances)
    marker = _QUESTION_MARK if punctuated else QUESTION_PATTERN
    mode = "물음표" if punctuated else "의문 어미"

    rows: list[tuple[int, str, str, str]] = []
    for index, utterance in enumerate(utterances):
        asker = roles.get(utterance.speaker_tag, Role.UNKNOWN)
        if asker not in (Role.MANAGER, Role.PATIENT):
            verdict = f"{asker.value} 라서 건너뜀"
        elif PROXY_QUESTION.search(utterance.text):
            verdict = "대리 질문"
        elif marker.search(utterance.text):
            verdict = "질문"
        elif QUESTION_PATTERN.search(utterance.text):
            verdict = f"의문 어미인데 {mode} 방식이라 놓침"
        else:
            verdict = "질문이 아님"

        if verdict in ("질문", "대리 질문"):
            has_answer = any(
                roles.get(f.speaker_tag, Role.UNKNOWN) is Role.DOCTOR
                for f in utterances[index + 1 :]
                if f.start_ms - utterance.end_ms <= ANSWER_WINDOW_MS
            )
            if not has_answer:
                verdict += " · 뒤에 의사 발화 없음"

        rows.append((utterance.start_ms, asker.value, verdict, utterance.text))
    return mode, rows
