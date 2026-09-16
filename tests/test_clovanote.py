"""실제 진료 녹음 1건에서 드러난 실패 패턴을 고정한다.

내용은 합성이지만 구조는 실제 앵커와 같다. 재진 상담이라 환자가 증상을 호소하지
않고 짧게 반응만 하며, 마지막에 인사말만 남긴 화자가 하나 더 잡힌다.
"""

from voice_ai.clova import group_by_speaker
from voice_ai.clovanote import parse
from voice_ai.models import Role
from voice_ai.qa import pair_qa
from voice_ai.roles import classify

# 재진 상담. 환자는 듣는 쪽이고, 자기 상태를 직접 묻는다.
FOLLOW_UP_VISIT = """참석자 1 00:00
피검사에서 신장이랑 빈혈은 괜찮으시고요.
저번에 말씀드린 것처럼 혈당 수치가 좀 높죠.
100이 정상인데 이번에는 118이라 조금씩 높아요.
식습관 때문에 그랬을 가능성이 제일 높습니다.
그래서 일단 야식을 줄이셔야 되고 단 음료는 별로 안 드시잖아요.

참석자 2 00:30
요즘 많이 안 먹기는 해요.

참석자 1 00:32
그래요. 그것도 줄이셔야 되고 운동 열심히 하셔서 체중 줄이시면 좋아지는 경우가 많거든요.
그러면 이 수치는 체중이 빠지면 돌아올 수 있습니다.

참석자 2 00:52
저번이랑 비교해서 많이 나빠진 건가요?

참석자 1 00:57
그런 건 아니고 비슷하세요. 비슷한 정도인데 그때도 약을 두 달분 드셨었잖아요.
매일 하루에 하나씩만 드시면 되니까 그거 드시면 좀 빨리 내려갈 것 같긴 해요.
두 달 분 드렸으니까 잘 챙겨 드시고 또 필요하시면 얘기해 주십시오.

참석자 3 01:30
네 감사합니다. 명절 잘 보내시고요. 감사합니다.
다음에 또."""


def resolve():
    utterances = parse(FOLLOW_UP_VISIT)
    roles, warnings = classify(group_by_speaker(utterances))
    return utterances, {r.speaker_tag: r for r in roles}, warnings


def test_parses_clovanote_blocks():
    utterances = parse(FOLLOW_UP_VISIT)
    assert len(utterances) == 6
    assert utterances[0].speaker_tag == "speaker_1"
    assert utterances[1].start_ms == 30_000
    # 블록 끝 시각은 다음 블록 시작으로 메운다.
    assert utterances[1].end_ms == 32_000


def test_doctor_is_identified_from_clinical_vocabulary():
    _, resolved, _ = resolve()
    assert resolved["speaker_1"].role is Role.DOCTOR
    assert resolved["speaker_1"].confidence > 0.6


def test_question_asking_patient_is_not_mistaken_for_manager():
    """환자도 자기 상태를 묻는다. 질문 비율만으로 매니저라고 하면 안 된다."""
    _, resolved, _ = resolve()
    assert resolved["speaker_2"].role is Role.PATIENT


def test_greeting_only_speaker_is_left_unknown():
    """인사말뿐인 화자를 억지로 배정하지 않고 경고로 넘긴다."""
    _, resolved, warnings = resolve()
    assert resolved["speaker_3"].role is Role.UNKNOWN
    assert any("speaker_3" in w for w in warnings)


def test_patient_question_pairs_with_doctor_answer():
    utterances, resolved, _ = resolve()
    pairs, _ = pair_qa(utterances, {tag: r.role for tag, r in resolved.items()})
    assert len(pairs) == 1
    assert pairs[0].asked_by is Role.PATIENT
    assert "비슷하세요" in pairs[0].answer


def test_low_confidence_speakers_go_to_review_queue():
    """운영자가 전건을 볼 수 없으니 확신이 낮은 판정만 큐에 올린다."""
    _, resolved, _ = resolve()
    assert resolved["speaker_1"].needs_review is False
    assert resolved["speaker_2"].needs_review is True
    assert resolved["speaker_3"].needs_review is True
