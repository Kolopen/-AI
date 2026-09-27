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
