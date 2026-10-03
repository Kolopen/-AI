"""매니저 질문 → 의사 답변 페어링.

동행 리포트의 실제 산출물이다. 사전 등록 질문이 있으면 추론이 아니라 대조가 되므로
정확도가 크게 오르고, 매니저가 빠뜨린 질문도 함께 잡아낼 수 있다.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import AskedState, QAPair, RegisteredQuestion, Role, Utterance
from .roles import PROXY_QUESTION, QUESTION_PATTERN

# 답변으로 인정할 최대 간격. 이보다 멀면 다른 화제로 넘어간 것으로 본다.
ANSWER_WINDOW_MS = 60_000

_QUESTION_MARK = re.compile(r"\?")


_NON_WORD = re.compile(r"[^가-힣A-Za-z0-9]")

# 등록 질문을 발화와 맞출 때 쓰는 두 단. 위를 넘으면 물은 것으로 보고,
# 아래와 사이면 비슷한 말이 있었다고만 적는다. 실제 녹음에서 같은 질문이
# 클로바 전사로는 0.60, 우리 전사로는 0.13~0.20 이 나왔다. 한 단으로
# 자르면 전사가 흔들릴 때마다 "안 물어봤다" 가 되는데, 그건 매니저에게
# 부당한 기록이다.
MATCH_CONFIRMED = 0.5
MATCH_LIKELY = 0.25

# 등록 질문이 발화를 거꾸로 얼마나 덮는지. 답변은 질문에 없던 내용을 들고
# 있으므로 이 값이 낮고, 질문은 등록된 문장 안에 거의 다 들어가므로 높다.
#
#                      정방향  역방향
#   의사 답변 "점수는 26점이었습니다"   0.27    0.14
#   의사 답변 "치매라고 진단하지는"     0.25    0.10
#   의사 답변 "약을 처방하지 않겠습니다" 0.18    0.07
#   잘린 질문 "혈압약이랑 같이 드셔도"  0.35    0.67
#   멀쩡한 질문 "치매로 진단된 건가요"   1.00    0.73
#
# 정방향은 0.27 과 0.35 로 겹쳐서 못 가른다. 역방향은 다섯 배 벌어진다.
REVERSE_MIN = 0.4


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


def question_detector(utterances: list[Utterance]) -> Callable[[str], bool]:
    """이 전사에서 무엇을 질문으로 볼지 정한다.

    물음표를 내놓는 전사라면 물음표만 믿는다. 어미로만 가르면 "기억이 잘 안
    나요" 가 "-나요" 질문으로 잡혀 엉뚱한 답변이 붙는다. SenseVoice 도
    moonshine 도 물음표를 찍으므로 대부분 이쪽으로 간다.
    """
    punctuated = any(_QUESTION_MARK.search(u.text) for u in utterances)
    marker = _QUESTION_MARK if punctuated else QUESTION_PATTERN

    def is_question(text: str) -> bool:
        # 대리 질문은 평서문으로 온다. 물음표도 의문 어미도 없지만 보호자가
        # 물은 것이고, 리포트에 가장 필요한 질문이기도 하다.
        return bool(marker.search(text) or PROXY_QUESTION.search(text))

    return is_question


def _answer_after(
    utterances: list[Utterance],
    index: int,
    roles: dict[str, Role],
    is_question: Callable[[str], bool],
) -> list[Utterance]:
    """질문 뒤에 이어지는 의사 발화를 모은다. 다른 사람이 끼어들면 끊는다."""
    question = utterances[index]
    parts: list[Utterance] = []
    for follow_up in utterances[index + 1 :]:
        if follow_up.start_ms - question.end_ms > ANSWER_WINDOW_MS:
            break
        # 다음 질문이 나오면 답변은 끝났다. 화자분리가 매니저를 의사 쪽에
        # 합쳐 버리면 뒤에 오는 질문까지 답변으로 삼켜 버린다.
        if is_question(follow_up.text):
            break
        if roles.get(follow_up.speaker_tag, Role.UNKNOWN) is Role.DOCTOR:
            parts.append(follow_up)
        elif parts:
            break
    return parts


def _build_pair(
    seq: int,
    utterances: list[Utterance],
    index: int,
    roles: dict[str, Role],
    is_question: Callable[[str], bool],
    *,
    question_id: str | None = None,
) -> QAPair | None:
    answer_parts = _answer_after(utterances, index, roles, is_question)
    if not answer_parts:
        return None

    question = utterances[index]
    return QAPair(
        seq=seq,
        question=question.text,
        answer=" ".join(part.text for part in answer_parts),
        question_at_ms=question.start_ms,
        answer_at_ms=answer_parts[0].start_ms,
        asked_by=roles.get(question.speaker_tag, Role.UNKNOWN),
        answered_by=Role.DOCTOR,
        confidence=_pair_confidence(question, answer_parts[0]),
        pre_registered_question_id=question_id,
    )


def match_registered(
    utterances: list[Utterance],
    roles: dict[str, Role],
    registered: dict[str, str],
) -> tuple[list[RegisteredQuestion], set[int]]:
    """등록 질문마다 가장 가까운 발화를 찾는다. 반환값은 (결과, 쓰인 발화 번호).

    질문을 녹음에서 알아낼 필요가 없다는 것이 요점이다. 보호자가 적어 보낸
    글을 이미 가지고 있으니, 그 글과 가장 비슷한 발화를 찾고 뒤에 붙은 의사
    답변을 가져오면 된다. 환자 목소리가 묻혀도 이쪽은 선다.

    매니저 역할이 아닌 발화도 본다. 실제 녹음에서 화자분리가 매니저를 의사
    쪽에 합쳐 버렸다. 역할만 믿으면 이 기능이 영영 안 걸린다. 다만 그렇게
    찾은 것은 확정하지 않는다.
    """
    results: list[RegisteredQuestion] = []
    used: set[int] = set()
    is_question = question_detector(utterances)

    for question_id, text in registered.items():
        best_index, best_score = None, 0.0
        for index, utterance in enumerate(utterances):
            if index in used:
                continue
            # 답변은 질문의 낱말을 되풀이하므로 겹치는 정도만 보면 가장
            # 비슷한 것이 답변 쪽이 된다. "점수가 얼마나 나왔나요" 에
            # "점수는 26점이었습니다" 가 걸리고 그 뒤 엉뚱한 말이 답변으로
            # 붙는다. 질문 꼴이거나, 등록 질문 안에 거의 다 들어가는 발화만
            # 후보로 둔다. 매니저가 길게 풀어 물으면 뒤쪽이 낮게 나오므로
            # 둘 중 하나만 맞으면 받는다.
            if not is_question(utterance.text) and _coverage(utterance.text, text) < REVERSE_MIN:
                continue
            score = _coverage(text, utterance.text)
            if score > best_score:
                best_index, best_score = index, score

        if best_index is None or best_score < MATCH_LIKELY:
            results.append(
                RegisteredQuestion(question_id, text, AskedState.UNCONFIRMED, round(best_score, 2))
            )
            continue

        asker = roles.get(utterances[best_index].speaker_tag, Role.UNKNOWN)
        confirmed = best_score >= MATCH_CONFIRMED and asker in (Role.MANAGER, Role.PATIENT)
        pair = _build_pair(
            len(results) + 1, utterances, best_index, roles, is_question, question_id=question_id
        )
        used.add(best_index)
        results.append(
            RegisteredQuestion(
                question_id,
                text,
                AskedState.CONFIRMED if confirmed else AskedState.LIKELY,
                round(best_score, 2),
                pair,
            )
        )
    return results, used


def pair_qa(
    utterances: list[Utterance],
    roles: dict[str, Role],
    *,
    registered_questions: dict[str, str] | None = None,
) -> tuple[list[QAPair], list[RegisteredQuestion]]:
    """시간순 발화에서 질문-답변 쌍을 뽑는다.

    두 번 훑는다. 먼저 보호자가 미리 남긴 질문을 맞춰 보고, 그 다음 현장에서
    새로 나온 질문을 줍는다. 반환값은 (현장 질문 쌍, 등록 질문 결과).
    """
    registered = registered_questions or {}
    results, used = match_registered(utterances, roles, registered)

    pairs: list[QAPair] = []
    seq = len([r for r in results if r.pair])

    is_question = question_detector(utterances)

    for index, utterance in enumerate(utterances):
        if index in used:
            continue
        if roles.get(utterance.speaker_tag, Role.UNKNOWN) not in (Role.MANAGER, Role.PATIENT):
            continue
        if not is_question(utterance.text):
            continue

        pair = _build_pair(seq + 1, utterances, index, roles, is_question)
        if pair is None:
            continue
        seq += 1
        pairs.append(pair)

    return pairs, results


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
