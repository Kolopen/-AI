"""리포트 초안 발췌.

`bodeul.session_reports` 컬럼을 원문 발췌로 채운다. 내용은 합성이지만 말투와
표현은 실제 진료 녹음에서 나온 그대로다.
"""

import datetime as dt

from voice_ai.models import Role, Utterance
from voice_ai.report import build_report_draft

DRUG_TERMS = frozenset({"간보호제"})

ROLES = {"D": Role.DOCTOR, "P": Role.PATIENT}

CONSULTATION = [
    Utterance("D", 0, 12_000, "콜레스테롤이 200이 정상인데 216이니까 약간 높습니다. 간 수치도 좀 높죠."),
    Utterance("P", 12_000, 14_000, "요즘 많이 안 먹기는 해요."),
    Utterance("D", 14_000, 26_000, "지방간 때문에 그럴 가능성이 제일 높습니다. 간 보호제를 매일 하루에 하나씩 드시면 됩니다."),
    Utterance("D", 26_000, 35_000, "두 달 분 드렸으니까 잘 챙겨 드시고요. 세 달 뒤에 재검사 한번 하시죠."),
]


def draft():
    return build_report_draft(CONSULTATION, ROLES, drug_terms=DRUG_TERMS)


def test_extracts_findings_with_timestamps():
    notes = draft().treatment_notes

    assert "콜레스테롤이 200이 정상인데" in notes
    # 원문 구간으로 되짚을 수 있어야 한다.
    assert "[00:00]" in notes


def test_finds_medication_name_from_dictionary():
    assert draft().medication_name == "간보호제"


def test_extracts_dosing_schedule():
    assert "매일 하루에 하나씩" in draft().medication_schedule_note


def test_extracts_prescription_duration():
    assert "두 달 분" in draft().medication_notes


def test_extracts_next_visit():
    note = draft().next_visit_note

    assert "세 달 뒤에 재검사" in note
    assert "[00:26]" in note


def test_patient_speech_is_never_quoted():
    """환자 발언이 의사 소견으로 섞여 들어가면 안 된다."""
    result = draft()
    everything = " ".join(
        [
            result.treatment_notes,
            result.medication_notes,
            result.medication_schedule_note,
            result.next_visit_note,
        ]
    )

    assert "요즘 많이 안 먹기는" not in everything


def test_summary_is_left_for_a_person():
    """자연어 생성이 필요한 유일한 항목이라 비워 둔다."""
    assert draft().summary == ""


# SenseVoice 한국어 출력에는 마침표가 거의 붙지 않는다. 실제 전사에서 가져온 모양이다.
UNPUNCTUATED = (
    "그때도 약을 두 달 드셨었잖아요 근데 두 달만 딱 드시니까 "
    "아직 완전히 좋아지진 않았고 지방간이 회복되는 게 중요할 것 같고요 "
    "그거를 매일 하루에 하나씩 드시면 되니까 빨리 내려갈 것 같긴 해요"
)


def test_clauses_are_split_without_punctuation():
    """마침표가 없으면 구간 하나가 통째로 한 문장이 되어 리포트를 못 쓰게 만든다."""
    from voice_ai.roles import split_sentences

    clauses = split_sentences(UNPUNCTUATED)

    assert len(clauses) > 1
    assert any("매일 하루에 하나씩" in c for c in clauses)
    # 한 절이 문단 전체를 품으면 안 된다.
    assert all(len(c) < 70 for c in clauses)


def test_report_fields_stay_short_without_punctuation():
    unpunctuated = [Utterance("D", 0, 30_000, UNPUNCTUATED)]

    result = build_report_draft(unpunctuated, ROLES, drug_terms=DRUG_TERMS)

    assert "매일 하루에 하나씩" in result.medication_schedule_note
    # 복용 방법란에 진료 설명이 통째로 딸려 오면 안 된다.
    assert "지방간이 회복되는" not in result.medication_schedule_note


def test_후속_예약_날짜를_값으로_뽑는다():
    utterances = [
        Utterance(speaker_tag="1", text="그럼 10월 20일날 오세요.", start_ms=65_000, end_ms=68_000),
    ]
    roles = {"1": Role.DOCTOR}
    draft = build_report_draft(utterances, roles, consult_date=dt.date(2026, 9, 30))
    assert draft.next_visit_at == "2026-10-20"
    # 근거 문장이 메모에 남아야 매니저가 확인할 수 있다.
    assert "10월 20일날 오세요" in draft.next_visit_note
    assert "[01:05]" in draft.next_visit_note


def test_진료일을_안_주면_날짜를_만들지_않는다():
    utterances = [
        Utterance(speaker_tag="1", text="10월 20일날 오세요.", start_ms=0, end_ms=3000),
    ]
    draft = build_report_draft(utterances, {"1": Role.DOCTOR})
    assert draft.next_visit_at is None


def test_기간만_말하면_날짜를_만들지_않는다():
    utterances = [
        Utterance(speaker_tag="1", text="두 달 뒤에 다시 오세요.", start_ms=0, end_ms=3000),
    ]
    draft = build_report_draft(utterances, {"1": Role.DOCTOR}, consult_date=dt.date(2026, 9, 30))
    assert draft.next_visit_at is None
    assert "두 달 뒤에 다시 오세요" in draft.next_visit_note


def test_예약_날짜는_처방_기간이_아니다():
    # "20일"이 20일치 처방으로 잡히던 자리다.
    utterances = [
        Utterance(speaker_tag="1", text="10월 20일날 오세요.", start_ms=0, end_ms=3000),
    ]
    draft = build_report_draft(utterances, {"1": Role.DOCTOR}, consult_date=dt.date(2026, 9, 30))
    assert draft.medication_notes == ""


def test_같은_문장의_처방_기간은_남는다():
    utterances = [
        Utterance(
            speaker_tag="1",
            text="두 달분 드릴 테니 10월 20일에 오세요.",
            start_ms=0,
            end_ms=3000,
        ),
    ]
    draft = build_report_draft(utterances, {"1": Role.DOCTOR}, consult_date=dt.date(2026, 9, 30))
    assert "두 달분" in draft.medication_notes
    assert draft.next_visit_at == "2026-10-20"


def test_관형사_뒤의_일은_처방_기간이_아니다():
    # "불편한 일이"의 "한 일"이 처방 기간으로 잡히던 자리다.
    utterances = [
        Utterance(speaker_tag="1", text="기억력 때문에 불편한 일이 있었나요?", start_ms=0, end_ms=3000),
    ]
    draft = build_report_draft(utterances, {"1": Role.DOCTOR})
    assert draft.medication_notes == ""


def test_검사_이름에_든_약_이름은_약품란에_넣지_않는다():
    utterances = [
        Utterance(
            speaker_tag="1",
            text="혈액검사에서 비타민 B12 수치가 정상 범위였습니다.",
            start_ms=0,
            end_ms=3000,
        ),
    ]
    draft = build_report_draft(
        utterances,
        {"1": Role.DOCTOR},
        drug_terms=frozenset({"비타민"}),
        test_terms=frozenset({"비타민B12", "혈액검사"}),
    )
    assert draft.medication_name == ""


def test_한자어로_말한_간격도_다음_방문이다():
    """전사가 "4주 뒤에"를 "사주 뒤에"로 내놓는다."""
    draft = build_report_draft(
        [Utterance("D", 113_000, 117_000, "사주 뒤에 보겠습니다 재진 날짜를 확인해 주세요")],
        {"D": Role.DOCTOR},
    )

    assert "사주 뒤에" in draft.next_visit_note


def test_수치로_말한_약은_리포트에_올라가지_않는다():
    draft = build_report_draft(
        [Utterance("D", 0, 9_000, "지난번 혈액검사에서 비타민이 기십 이 점수치가 정상범이었습니다")],
        {"D": Role.DOCTOR},
        drug_terms=frozenset({"비타민"}),
        test_terms=frozenset({"비타민B12", "혈액검사"}),
    )

    assert draft.medication_name == ""


def test_숫자_없는_진료_결론도_담는다():
    """실제 녹음에서 진료의 결론이 통째로 빠졌다. 숫자가 없어서였다."""
    draft = build_report_draft(
        [
            Utterance("D", 97_000, 107_000,
                      "오늘은 기억력에 대한 약을 처방하지 않겠습니다 검사 결과를 확인한 뒤 치료 방향을 논의하겠습니다"),
            Utterance("D", 90_000, 95_000, "오늘은 MRI를 예약하고 결과를 보고 추가로 설명 드리겠습니다"),
            Utterance("D", 60_000, 66_000, "점수 하나만으로 치매라고 진단하지는 않습니다"),
        ],
        {"D": Role.DOCTOR},
    )

    notes = draft.treatment_notes
    assert "처방하지 않겠습니다" in notes
    assert "예약하고" in notes
    assert "진단하지는 않습니다" in notes


def test_같은_낱말로_묻는_질문은_담지_않는다():
    """"진단"만 보면 환자의 "치매 진단이 뭔가요"가 진료 내용으로 올라간다."""
    from voice_ai.report import _PLAN

    for question in ("치매 진단이 뭔가요", "검사 언제 받아요", "약 먹으면 졸려요"):
        assert not _PLAN.search(question), question
