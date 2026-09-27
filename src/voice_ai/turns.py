"""이어지는 같은 화자의 구간을 한 발언으로 묶는다.

화자분리는 숨을 고르는 곳마다 구간을 끊으므로, 의사가 한 화제를 말하는 동안에도
구간이 여럿 생긴다. 사람이 읽을 때는 이것이 방해가 된다. 클로바노트가 화자마다
한 덩어리로 보여주는 것도 같은 이유다.

묶기만 하고 합치지는 않는다. 구간마다 줄을 나눠 두면 어느 대목이 어느 시각인지
그대로 남고, 매니저가 녹음에서 그 자리를 찾을 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Utterance

# 같은 화자라도 이만큼 떨어지면 다른 발언으로 본다. 상대가 말하고 돌아온 것이다.
TURN_GAP_MS = 15_000


@dataclass
class Turn:
    speaker_tag: str
    start_ms: int
    end_ms: int
    text: str  # 구간마다 줄바꿈으로 나눠 둔다


def group_turns(utterances: list[Utterance], gap_ms: int = TURN_GAP_MS) -> list[Turn]:
    turns: list[Turn] = []
    for utterance in utterances:
        last = turns[-1] if turns else None
        if (
            last is not None
            and last.speaker_tag == utterance.speaker_tag
            and utterance.start_ms - last.end_ms <= gap_ms
        ):
            last.text += "\n" + utterance.text
            last.end_ms = max(last.end_ms, utterance.end_ms)
            continue
        turns.append(
            Turn(utterance.speaker_tag, utterance.start_ms, utterance.end_ms, utterance.text)
        )
    return turns
