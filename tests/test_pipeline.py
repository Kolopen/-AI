import json
from pathlib import Path

from voice_ai.clova import group_by_speaker, parse_segments
from voice_ai.models import Role
from voice_ai.qa import pair_qa
from voice_ai.roles import classify

FIXTURE = Path(__file__).parent / "fixtures" / "sample_clova.json"


def load_utterances():
    return parse_segments(json.loads(FIXTURE.read_text(encoding="utf-8")))


def role_map(manager_speaker_tag=None):
    profiles = group_by_speaker(load_utterances())
    resolved = classify(profiles, manager_speaker_tag=manager_speaker_tag)
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
    resolved = {
        r.speaker_tag: r for r in classify(group_by_speaker(load_utterances()), manager_speaker_tag="speaker_1")
    }
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
