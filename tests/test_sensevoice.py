"""SenseVoice 출력 정규화.

SenseVoice는 전사에 언어·감정·음향이벤트 태그를 붙여 준다. 본문만 뽑되
언어 태그는 전사 신뢰도 판단에 쓴다.
"""

from voice_ai.models import Role
from voice_ai.roles import sentence_role
from voice_ai.sensevoice import parse_chunks, parse_tags, strip_tags

CHUNKS = [
    {
        "index": 0,
        "start_ms": 0,
        "end_ms": 8000,
        "raw_text": "<|ko|><|NEUTRAL|><|Speech|><|withitn|>간 수치가 정상보다 좀 높으시니까 간 보호제를 드셔야 합니다.",
        "text": "간 수치가 정상보다 좀 높으시니까 간 보호제를 드셔야 합니다.",
    },
    {
        "index": 1,
        "start_ms": 8000,
        "end_ms": 11000,
        "raw_text": "<|ko|><|NEUTRAL|><|Speech|><|withitn|>요즘 많이 안 먹기는 해요.",
        "text": "요즘 많이 안 먹기는 해요.",
    },
]


def test_strips_tags_from_raw_text():
    raw = "<|ko|><|NEUTRAL|><|Speech|><|withitn|>간 수치가 높습니다."

    assert strip_tags(raw) == "간 수치가 높습니다."


def test_reads_language_and_emotion_tags():
    tags = parse_tags("<|ko|><|HAPPY|><|Speech|><|withitn|>안녕하세요.")

    assert tags.language == "ko"
    assert tags.emotion == "HAPPY"
    assert tags.events == ("Speech",)


def test_parses_chunks_into_utterances():
    utterances, warnings = parse_chunks(CHUNKS)

    assert len(utterances) == 2
    assert utterances[0].start_ms == 0
    assert utterances[0].text.startswith("간 수치가")
    # SenseVoice는 화자를 구분하지 않는다.
    assert all(u.speaker_tag == "" for u in utterances)
    assert warnings == []


def test_warns_when_a_chunk_is_recognised_as_another_language():
    """한국어 진료에서 다른 언어가 찍히면 그 구간 전사를 믿을 수 없다."""
    chunks = [
        {"start_ms": 5000, "end_ms": 7000, "raw_text": "<|en|><|NEUTRAL|><|Speech|>Thank you."}
    ]

    utterances, warnings = parse_chunks(chunks)

    assert utterances[0].text == "Thank you."
    assert len(warnings) == 1
    assert "en로 인식" in warnings[0]


def test_roles_can_be_assigned_without_any_speaker_labels():
    """화자분리 없는 출력이 그대로 역할 판정으로 이어진다."""
    utterances, _ = parse_chunks(CHUNKS)

    assert sentence_role(utterances[0].text)[0] is Role.DOCTOR
    assert sentence_role(utterances[1].text)[0] is Role.PATIENT


def test_analyzes_a_transcript_that_has_no_speaker_labels():
    """SenseVoice 출력처럼 화자가 없는 전사도 끝까지 분석된다."""
    from voice_ai.analyze import STARTER_TERMS, analyze_without_speakers

    utterances, _ = parse_chunks(CHUNKS)
    result = analyze_without_speakers(utterances, medical_terms=STARTER_TERMS)

    roles = {s.role for s in result.speakers}
    assert Role.DOCTOR in roles
    assert Role.PATIENT in roles
