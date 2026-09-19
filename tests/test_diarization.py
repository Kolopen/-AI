"""STT 결과와 화자 구간 정렬.

pyannote 같은 별도 화자분리를 쓸 때 필요하다. CLOVA가 구조적으로 표현하지 못하던
"한 전사 구간에 두 사람" 상황을 여기서 잡아낸다.
"""

from voice_ai.diarization import Turn, align, parse_pyannote
from voice_ai.models import Utterance


def test_parses_pyannote_seconds_into_milliseconds():
    turns = parse_pyannote(
        {"diarization": [{"speaker": "SPEAKER_00", "start": 3.045, "end": 5.845}]}
    )

    assert turns == [Turn(speaker="speaker_SPEAKER_00", start_ms=3045, end_ms=5845)]


def test_assigns_the_most_overlapping_speaker():
    segments = [
        Utterance("", 0, 5000, "검사 결과를 보시면 수치가 올라가 있어요."),
        Utterance("", 5000, 6000, "네."),
    ]
    turns = [
        Turn("speaker_A", 0, 5000),
        Turn("speaker_B", 5000, 6000),
    ]

    aligned, warnings = align(segments, turns)

    assert [u.speaker_tag for u in aligned] == ["speaker_A", "speaker_B"]
    assert warnings == []


def test_flags_a_segment_spanning_two_speakers():
    """의사 설명에 환자의 짧은 응답이 끼어든 구간. CLOVA로는 표현조차 못 하던 경우다."""
    segments = [Utterance("", 0, 10_000, "그거 드시면 좀 빨리 내려갈 것 같아요. 알겠습니다.")]
    turns = [Turn("speaker_A", 0, 7_000), Turn("speaker_B", 7_000, 10_000)]

    aligned, warnings = align(segments, turns)

    assert aligned[0].speaker_tag == "speaker_A"
    assert len(warnings) == 1
    assert "speaker_A와 speaker_B에 걸쳐" in warnings[0]


def test_brief_overlap_is_not_flagged():
    """경계가 조금 어긋난 정도로는 경고하지 않는다."""
    segments = [Utterance("", 0, 10_000, "검사 결과를 보시면 수치가 올라가 있어요.")]
    turns = [Turn("speaker_A", 0, 9_500), Turn("speaker_B", 9_500, 12_000)]

    _, warnings = align(segments, turns)

    assert warnings == []


def test_segment_outside_every_turn_is_reported():
    segments = [Utterance("", 60_000, 62_000, "감사합니다.")]
    turns = [Turn("speaker_A", 0, 5_000)]

    aligned, warnings = align(segments, turns)

    assert aligned[0].speaker_tag == "speaker_unknown"
    assert "겹치는 화자가 없습니다" in warnings[0]
