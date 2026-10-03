import contextlib
import io
import json
from pathlib import Path

from voice_ai import terminology
from voice_ai.analyze import _diarization_collapsed, analyze_without_speakers, manager_tag
from voice_ai.clova import group_by_speaker, parse_segments
from voice_ai.models import AskedState, Role, SpeakerProfile, Utterance
from voice_ai.qa import explain_qa, pair_qa
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
    pairs, registered = pair_qa(load_utterances(), role_map())
    questions = [p.question for p in pairs]
    assert any("혈압약" in q for q in questions)
    assert any("다음 진료" in q for q in questions)
    assert registered == []

    blood_pressure = next(p for p in pairs if "혈압약" in p.question)
    assert blood_pressure.asked_by is Role.MANAGER
    assert blood_pressure.answered_by is Role.DOCTOR
    assert "식후에 드세요" in blood_pressure.answer


def test_a_guardian_question_finds_its_doctor_answer():
    """보호자 질문은 글로 이미 있다. 녹음에서 알아낼 것은 의사 답변뿐이다."""
    asked = {
        "q_bp": "지금 드시는 혈압약이랑 같이 먹어도 되나요?",
        "q_diet": "식사할 때 피해야 할 음식이 있나요?",
    }
    _, registered = pair_qa(load_utterances(), role_map(), registered_questions=asked)
    found = {r.question_id: r for r in registered}

    assert found["q_bp"].state is AskedState.CONFIRMED
    assert found["q_bp"].pair is not None
    assert "식후에 드세요" in found["q_bp"].pair.answer


def test_a_question_we_cannot_find_is_not_called_unasked():
    """못 찾은 것과 안 물어본 것은 다르다. 약한 고리는 우리 전사다."""
    asked = {"q_diet": "식사할 때 피해야 할 음식이 있나요?"}
    _, registered = pair_qa(load_utterances(), role_map(), registered_questions=asked)

    assert registered[0].state is AskedState.UNCONFIRMED
    assert registered[0].pair is None


def test_a_loose_match_is_only_probable():
    """전사가 흔들려 반쯤만 겹치면 물었다고 단정하지 않는다."""
    utterances, roles = _conversation(
        ("M", "혈압약이랑 같이 드셔도"),
        ("D", "네 괜찮습니다 식후에 드세요"),
    )
    _, registered = pair_qa(
        utterances, roles, registered_questions={"q_bp": "지금 드시는 혈압약이랑 같이 먹어도 되나요?"}
    )

    assert registered[0].state is AskedState.LIKELY
    assert registered[0].pair is not None


def test_a_question_asked_from_the_doctor_cluster_is_only_probable():
    """화자분리가 매니저를 의사 쪽에 합쳐도 찾기는 한다. 다만 확정하지 않는다."""
    utterances, roles = _conversation(
        ("D", "지금 드시는 혈압약이랑 같이 먹어도 되나요"),
        ("D", "네 괜찮습니다 식후에 드세요"),
    )
    _, registered = pair_qa(
        utterances, roles, registered_questions={"q_bp": "지금 드시는 혈압약이랑 같이 먹어도 되나요?"}
    )

    assert registered[0].score > 0.9
    assert registered[0].state is AskedState.LIKELY


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


def test_the_enrolled_manager_is_read_back_from_the_transcript(tmp_path):
    # voice-transcribe --manager 가 붙인 표시를 분석 쪽이 그대로 받는다.
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            [
                {"speaker": "speaker_00", "start_ms": 0, "end_ms": 9000,
                 "raw_text": "약을 드셔 보시고", "is_manager": False},
                {"speaker": "speaker_01", "start_ms": 9000, "end_ms": 18000,
                 "raw_text": "보호자분이 여쭤봐 달라고 하셨어요", "is_manager": True},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert manager_tag(path) == "speaker_01"


def test_no_manager_tag_without_enrolment(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps([{"speaker": "speaker_00", "start_ms": 0, "end_ms": 9000,
                     "raw_text": "약을 드셔 보시고"}]),
        encoding="utf-8",
    )

    assert manager_tag(path) is None


def test_why_qa_names_the_reason_each_utterance_was_skipped():
    """짝이 0건일 때 어디서 끊겼는지 눈으로 갈려야 한다."""
    utterances, roles = _conversation(
        ("D", "어디가 불편하신가요?"),
        ("P", "머리가 아파요"),
        ("M", "검사를 더 해야 하나요?"),
    )
    mode, rows = explain_qa(utterances, roles)

    assert mode == "물음표"
    assert rows[0][2] == "DOCTOR 라서 건너뜀"
    assert rows[1][2] == "질문이 아님"
    assert rows[2][2] == "질문 · 뒤에 의사 발화 없음"


def test_why_qa_catches_a_question_lost_to_the_question_mark_rule():
    """물음표가 한 군데라도 있으면 어미만 있는 질문은 통째로 버려진다."""
    utterances, roles = _conversation(
        ("D", "지난번 이후로 어떠셨어요?"),
        ("P", "약을 계속 먹어야 하나요"),
        ("D", "네 당분간은 드셔야 합니다"),
    )
    mode, rows = explain_qa(utterances, roles)

    assert mode == "물음표"
    assert rows[1][2] == "의문 어미인데 물음표 방식이라 놓침"


def test_the_doctors_answer_is_not_mistaken_for_the_question():
    """답변은 질문의 낱말을 되풀이한다. 겹치는 정도만 보면 답변이 1등이 된다."""
    utterances, roles = _conversation(
        ("D", "지난검사 점수는 삼십 쩜 만쯤에 이십 육 점이었습니다"),
        ("D", "다른 검사 결과도 함께 봐야 합니다"),
    )
    _, registered = pair_qa(
        utterances, roles, registered_questions={"q": "검사 점수가 얼마나 나왔나요?"}
    )

    assert registered[0].state is AskedState.UNCONFIRMED
    assert registered[0].pair is None


def test_an_answer_stops_at_the_next_question():
    """화자분리가 매니저를 의사 쪽에 합치면 뒤 질문까지 답변으로 삼킨다."""
    utterances, roles = _conversation(
        ("M", "혈압약이랑 같이 먹어도 되나요?"),
        ("D", "네 괜찮습니다"),
        ("D", "그러면 운전은 해도 되나요?"),
        ("D", "운전은 당분간 피하세요"),
    )
    pairs, _ = pair_qa(utterances, roles)

    first = next(p for p in pairs if "혈압약" in p.question)
    assert first.answer == "네 괜찮습니다"


def test_both_drafts_name_what_differs():
    """두 방식의 리포트를 나란히 놓고 다른 줄만 짚어준다."""
    from voice_ai.analyze import print_both_drafts
    from voice_ai.models import ReportDraft

    grouped = ReportDraft(treatment_notes="[01:18] 원인을 평가하는 단계입니다")
    split = ReportDraft(treatment_notes="[01:18] 원인을 평가하는 단계입니다\n[01:30] MRI를 예약하겠습니다")

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        print_both_drafts(grouped, split, [])
    out = printed.getvalue()

    assert "구분 안 함에만     [01:30] MRI를 예약하겠습니다" in out
    assert "화자 구분함에만" not in out


def test_both_drafts_say_when_nothing_differs():
    from voice_ai.analyze import print_both_drafts
    from voice_ai.models import ReportDraft

    same = ReportDraft(treatment_notes="[01:18] 같은 줄")

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        print_both_drafts(same, ReportDraft(treatment_notes="[01:18] 같은 줄"), [])

    assert "두 리포트가 같습니다" in printed.getvalue()


def test_a_patients_line_in_the_draft_is_marked():
    """화자분리가 환자 말을 의사 쪽에 섞으면 리포트에 그대로 들어간다."""
    from voice_ai.analyze import doubtful

    assert doubtful("[02:28] 아침에 조금 졸려요 약 때문인지 모르겠는데요", frozenset())
    assert doubtful("[02:32] 보호자분도 졸려 보인다고 말씀하셔서 여쭤봅니다", frozenset())


def test_a_doctors_line_is_not_marked():
    from voice_ai.analyze import doubtful

    assert not doubtful("[03:07] 매일 복용하도록 처방한 약과 필요할 때 복용하는 약은 구분하세요", frozenset())


def test_both_runs_even_when_diarization_collapsed(tmp_path, capsys):
    """두 리포트가 가장 크게 갈리는 자리가 바로 붕괴한 녹음이다."""
    from voice_ai.analyze import main

    transcript = tmp_path / "t.json"
    transcript.write_text(
        json.dumps(
            [
                {"speaker": "speaker_00", "start_ms": 0, "end_ms": 9_000,
                 "text": "매일 복용하도록 처방한 약과 필요할 때 복용하는 약은 구분하세요"},
                {"speaker": "speaker_00", "start_ms": 10_000, "end_ms": 14_000,
                 "text": "아침에 조금 졸려요 약 때문인지 모르겠는데요"},
                {"speaker": "speaker_01", "start_ms": 20_000, "end_ms": 21_000, "text": "네"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert main([str(transcript), "--both"]) == 0
    out = capsys.readouterr().out
    assert "화자 구분함" in out and "화자 구분 안 함" in out
    assert "화자분리가 사실상 한 사람만" in out
