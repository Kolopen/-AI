import json
from pathlib import Path

from voice_ai import terminology
from voice_ai.analyze import _diarization_collapsed, analyze_without_speakers
from voice_ai.clova import group_by_speaker, parse_segments
from voice_ai.models import Role, SpeakerProfile, Utterance
from voice_ai.qa import pair_qa
from voice_ai.roles import classify, doctor_score, sentence_role

FIXTURE = Path(__file__).parent / "fixtures" / "sample_clova.json"


def load_utterances():
    return parse_segments(json.loads(FIXTURE.read_text(encoding="utf-8")))


def role_map(manager_speaker_tag=None):
    profiles = group_by_speaker(load_utterances())
    resolved, _ = classify(profiles, manager_speaker_tag=manager_speaker_tag)
    return {r.speaker_tag: r.role for r in resolved}


def test_parses_speaker_and_timestamps():
    utterances = load_utterances()
    assert len(utterances) == 10
    assert utterances[0].speaker_tag == "speaker_1"
    assert utterances[0].start_ms == 0
    assert utterances[-1].end_ms == 54000


def test_classifies_roles_from_text_alone():
    assert role_map() == {
        "speaker_1": Role.MANAGER,
        "speaker_2": Role.DOCTOR,
        "speaker_3": Role.PATIENT,
    }


def test_voiceprint_lock_keeps_manager_fixed():
    roles, _ = classify(group_by_speaker(load_utterances()), manager_speaker_tag="speaker_1")
    resolved = {r.speaker_tag: r for r in roles}
    assert resolved["speaker_1"].role is Role.MANAGER
    assert resolved["speaker_1"].confidence == 1.0
    assert resolved["speaker_2"].role is Role.DOCTOR
    assert resolved["speaker_3"].role is Role.PATIENT


def test_pairs_manager_questions_with_doctor_answers():
    pairs, unasked = pair_qa(load_utterances(), role_map())
    questions = [p.question for p in pairs]
    assert any("혈압약" in q for q in questions)
    assert any("다음 진료" in q for q in questions)
    assert unasked == []

    blood_pressure = next(p for p in pairs if "혈압약" in p.question)
    assert blood_pressure.asked_by is Role.MANAGER
    assert blood_pressure.answered_by is Role.DOCTOR
    assert "식후에 드세요" in blood_pressure.answer


def test_detects_unasked_pre_registered_question():
    registered = {
        "q_bp": "지금 드시는 혈압약이랑 같이 먹어도 되나요?",
        "q_diet": "식사할 때 피해야 할 음식이 있나요?",
    }
    pairs, unasked = pair_qa(load_utterances(), role_map(), registered_questions=registered)

    matched = {p.pre_registered_question_id for p in pairs}
    assert "q_bp" in matched
    assert unasked == ["q_diet"]


def _over_segmented():
    """화자분리가 짧은 맞장구를 화자마다 흩어 놓은 실제 상황.

    의사만 길게 말하고 비의사는 한두 마디씩 여러 태그로 쪼개진다. 실제 녹음에서
    비의사 화자 넷이 각각 8~21자로 나왔다.
    """
    from voice_ai.models import Utterance

    return [
        Utterance("speaker_00", 0, 9_000, "피검사에서는 신장이라든지 빈혈은 괜찮으시고요."),
        Utterance("speaker_01", 9_000, 10_500, "네."),
        Utterance("speaker_00", 10_500, 20_000, "간수치가 정상보다 조금 높으신 상태예요."),
        Utterance("speaker_04", 20_000, 22_000, "그럼 언제까지 하나요?"),
        Utterance("speaker_00", 22_000, 32_000, "두 달 뒤에 다시 보시면 됩니다."),
        Utterance("speaker_07", 32_000, 33_000, "예."),
    ]


def _analyze(merge_non_doctor=False):
    from voice_ai import terminology
    from voice_ai.analyze import analyze_with_speakers

    return analyze_with_speakers(
        _over_segmented(),
        terms=terminology.load("내과"),
        merge_non_doctor=merge_non_doctor,
    )


def test_split_speakers_are_flagged_not_guessed():
    """쪼개진 조각은 UNKNOWN으로 남기고 합치라고 알려준다.

    조각마다 억지로 역할을 붙이면 틀린 화자에게 발언이 귀속된다.
    """
    result = _analyze()

    assert [s.role for s in result.speakers].count(Role.UNKNOWN) == 3
    assert any("--merge-non-doctor" in w for w in result.warnings)


def test_merging_keeps_the_doctor_apart():
    """합치는 것은 비의사끼리만이다. 의사 판정이 흔들리면 안 된다."""
    result = _analyze(merge_non_doctor=True)

    doctors = [s for s in result.speakers if s.role is Role.DOCTOR]
    assert [s.speaker_tag for s in doctors] == ["speaker_00"]
    assert all(s.role is Role.PATIENT for s in result.speakers if s.speaker_tag != "speaker_00")
    assert result.speakers[1].merged_from == ["speaker_04", "speaker_07"]


def test_merging_recovers_a_question_that_was_lost():
    """짧고 어휘 신호가 없는 질문은 화자가 UNKNOWN이면 통째로 버려진다.

    질문자를 특정하지 못하면 Q&A 페어링이 아예 시작되지 않기 때문이다.
    """
    assert _analyze().qa_pairs == []

    pairs = _analyze(merge_non_doctor=True).qa_pairs

    assert len(pairs) == 1
    assert pairs[0].question == "그럼 언제까지 하나요?"
    assert "두 달" in pairs[0].answer


def test_unspaced_transcript_is_called_out():
    """zipformer-korean 은 띄어쓰기 없이 내놓는다.

    용어 교정은 어절을 후보로 삼으므로 한 건도 못 찾고, 리포트는 문장을 못 갈라
    같은 문단을 여러 항목에 넣는다. 결과가 비는 게 아니라 그럴듯하게 틀린다.
    """
    from voice_ai import terminology
    from voice_ai.analyze import analyze_with_speakers
    from voice_ai.models import Utterance

    blob = "그건높긴한데요구땜에약정도아니에요근데저번에말씀드건전수치는높죠근데작년에는"
    result = analyze_with_speakers(
        [
            Utterance("speaker_00", 0, 9_000, blob),
            Utterance("speaker_01", 9_000, 10_000, "네."),
        ],
        terms=terminology.load("내과"),
    )

    assert any("띄어쓰기가 없습니다" in w for w in result.warnings)


def test_spaced_transcript_is_not_warned_about():
    from voice_ai import terminology
    from voice_ai.analyze import analyze_with_speakers
    from voice_ai.models import Utterance

    result = analyze_with_speakers(
        [
            Utterance("speaker_00", 0, 9_000, "피검사에서는 신장이라든지 빈혈은 괜찮으시고요."),
            Utterance("speaker_01", 9_000, 10_000, "네."),
        ],
        terms=terminology.load("내과"),
    )

    assert not any("띄어쓰기" in w for w in result.warnings)


def test_one_speaker_means_diarization_told_us_nothing():
    # 실제 3인 녹음에서 화자분리가 한 명만 내놨다. 화자 단위로 가르면
    # 모든 발화가 같은 역할을 받아 의사와 환자를 구분하지 못한다.
    collapsed = [
        Utterance("speaker_00", 0, 180_000, "두통이 언제부터 있으셨어요? " * 20),
        Utterance("speaker_01", 180_000, 181_000, "네 맞아요 그렇습니다"),
    ]
    assert _diarization_collapsed(collapsed)


def test_two_speaking_speakers_are_kept():
    kept = [
        Utterance("speaker_00", 0, 180_000, "두통이 언제부터 있으셨어요? " * 20),
        Utterance("speaker_01", 180_000, 240_000, "한 달쯤 됐어요 참다가 왔어요. " * 5),
    ]
    assert not _diarization_collapsed(kept)


def test_sentence_mode_hands_back_role_labelled_utterances():
    # 문장 단위로 내려가면 화자 태그가 역할 이름으로 바뀐다. 이걸 돌려주지
    # 않으면 핵심 내용 추출이 입력 태그로 의사 발화를 찾다가 전부 놓친다.
    terms = terminology.load("신경과")
    result = analyze_without_speakers(
        [Utterance("", 0, 9_000, "편두통 양상과 함께 나타날 수 있습니다. 약은 매일 드셔야 합니다.")],
        terms=terms,
    )

    assert result.labelled
    assert Role.DOCTOR.value in {u.speaker_tag for u in result.labelled}


def _doctor_said(count: int) -> str:
    return "검사 결과를 보시면 수치가 높으시니까 약을 처방해 드릴게요. " * count


def test_doctor_score_survives_fine_segmentation():
    """같은 말을 잘게 쪼개도 의사 점수가 달라지면 안 된다.

    실제 녹음에서 같은 의사가 클로바 분할로는 3.33, 우리 화자분리로는 1.45 를
    받아 기준선 아래로 떨어졌고, 그 바람에 의사를 특정하지 못했다. 쪼개는
    방식은 누가 말했는지와 아무 상관이 없다.
    """
    whole = _doctor_said(6)
    coarse = SpeakerProfile("speaker_00", [Utterance("speaker_00", 0, 60_000, whole)])
    pieces = [s.strip() + " " for s in whole.split(". ") if s.strip()]
    fine = SpeakerProfile(
        "speaker_00",
        [
            Utterance("speaker_00", i * 2_000, i * 2_000 + 1_900, piece)
            for i, piece in enumerate(pieces)
        ],
    )

    assert doctor_score(coarse) == doctor_score(fine)


def test_a_short_speaker_quoting_the_doctor_does_not_outscore_the_doctor():
    # 매니저가 "기형 검사", "치매", "확진"처럼 의사 어휘를 옮겨 말하면 짧은
    # 발화 안에서 밀도가 의사만큼 올라간다. 바닥 길이가 이걸 막는다.
    doctor = SpeakerProfile("speaker_00", [Utterance("speaker_00", 0, 60_000, _doctor_said(6))])
    manager = SpeakerProfile(
        "speaker_01",
        [Utterance("speaker_01", 60_000, 64_000, "지난번 검사 결과가 어떠신지 여쭤보셨어요")],
    )

    assert doctor_score(doctor) > doctor_score(manager)


def _conversation(*rows) -> tuple[list[Utterance], dict[str, Role]]:
    utterances = [
        Utterance(tag, i * 10_000, i * 10_000 + 9_000, text)
        for i, (tag, text) in enumerate(rows)
    ]
    return utterances, {"D": Role.DOCTOR, "M": Role.MANAGER, "P": Role.PATIENT}


def test_proxy_questions_are_questions_without_a_question_mark():
    # 대리 질문은 평서문으로 온다. 리포트에 가장 필요한 질문인데 물음표가 없다.
    utterances, roles = _conversation(
        ("M", "보호자분이 MRI와 뇌파 검사가 같은 검사인지도 물어보셨어요."),
        ("D", "서로 다른 검사입니다. 뇌파 검사는 이번에 시행하지 않았습니다."),
    )
    pairs, _ = pair_qa(utterances, roles)

    assert len(pairs) == 1
    assert pairs[0].asked_by is Role.MANAGER
    assert "뇌파 검사는 이번에 시행하지 않았습니다" in pairs[0].answer


def test_a_proxy_question_full_of_medical_words_is_still_the_manager():
    # 매니저가 보호자 질문을 옮기면서 의학 용어를 같이 말한다. 점수로 겨루면
    # 의사가 이겨서 대리 질문이 통째로 사라졌다.
    role, _ = sentence_role(
        "지난번에 말씀하신 편두통과 관련된 건지 여쭤봐 달라고 하셨어요.",
        medical_terms=frozenset({"편두통", "두통"}),
    )

    assert role is Role.MANAGER


def test_a_statement_ending_in_나요_is_not_a_question():
    # "기억이 잘 안 나요"가 "-나요" 질문으로 잡혀 엉뚱한 답변이 붙었다.
    utterances, roles = _conversation(
        ("P", "저는 기억이 잘 안 나요."),
        ("D", "횟수는 정확하지 않은 것으로 기록하겠습니다?"),
    )
    pairs, _ = pair_qa(utterances, roles)

    assert pairs == []


def test_proxy_marker_survives_a_dropped_syllable():
    # whisper 는 "궁금해하셨어요"를 "궁금하셨어요"로 내놨다. 이 녹음에서 제일
    # 중요한 질문이 그 한 글자 때문에 통째로 사라졌다.
    role, _ = sentence_role("보호자분이 치매를 뜻하는지 궁금하셨어요")

    assert role is Role.MANAGER
