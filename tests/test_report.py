"""리포트 초안 발췌.

`bodeul.session_reports` 컬럼을 원문 발췌로 채운다. 내용은 합성이지만 말투와
표현은 실제 진료 녹음에서 나온 그대로다.
"""

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
