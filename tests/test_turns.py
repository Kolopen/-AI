"""같은 화자의 이어지는 구간 묶기.

화자분리는 숨을 고르는 곳마다 끊으므로 의사 한 사람의 설명이 구간 여럿으로
흩어진다. 클로바노트처럼 화자별로 묶어야 사람이 읽을 수 있다.
"""

from voice_ai.models import Utterance
from voice_ai.turns import group_turns


def test_consecutive_segments_from_one_speaker_become_one_turn():
    utterances = [
        Utterance("speaker_00", 0, 5_000, "술도 줄이셔야 되고"),
        Utterance("speaker_00", 5_000, 10_000, "운동도 열심히 하셔야 합니다."),
    ]

    turns = group_turns(utterances)

    assert len(turns) == 1
    assert turns[0].text == "술도 줄이셔야 되고\n운동도 열심히 하셔야 합니다."
    assert (turns[0].start_ms, turns[0].end_ms) == (0, 10_000)


def test_the_other_speaker_starts_a_new_turn():
    utterances = [
        Utterance("speaker_00", 0, 5_000, "술은 별로 안 드시잖아요?"),
        Utterance("speaker_01", 5_000, 7_000, "요즘 많이 안 먹기는 해요."),
        Utterance("speaker_00", 7_000, 12_000, "그래요."),
    ]

    assert [t.speaker_tag for t in group_turns(utterances)] == [
        "speaker_00",
        "speaker_01",
        "speaker_00",
    ]


def test_a_long_silence_starts_a_new_turn():
    """같은 화자라도 한참 뒤면 다른 발언이다. 상대가 말하고 돌아온 것이다."""
    utterances = [
        Utterance("speaker_00", 0, 5_000, "여기까지 보겠습니다."),
        Utterance("speaker_00", 90_000, 95_000, "두 달분 드렸으니까 잘 챙겨 드세요."),
    ]

    assert len(group_turns(utterances)) == 2


def test_each_segment_keeps_its_own_line():
    """묶되 합치지는 않는다. 어느 대목이 어느 시각인지 남아야 녹음에서 찾는다."""
    utterances = [
        Utterance("speaker_00", 0, 5_000, "첫 줄"),
        Utterance("speaker_00", 5_000, 9_000, "둘째 줄"),
        Utterance("speaker_00", 9_000, 12_000, "셋째 줄"),
    ]

    assert group_turns(utterances)[0].text.split("\n") == ["첫 줄", "둘째 줄", "셋째 줄"]
