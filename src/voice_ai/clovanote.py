"""클로바노트 내보내기 텍스트 파서.

CLOVA Speech API를 붙이기 전 중간 입력으로 쓴다. 블록 단위 타임스탬프만 있어서
문장별 시각이 없고, 한 블록에 여러 화자가 섞여 들어오는 경우가 있다.
"""

from __future__ import annotations

import re

from .models import Utterance

HEADER = re.compile(r"^참석자\s*(\d+)\s+(?:(\d{1,2}):)?(\d{1,2}):(\d{2})$")

# 마지막 블록은 끝 시각을 알 수 없어 글자 수로 어림잡는다.
_MS_PER_CHAR = 90


def parse(raw: str) -> list[Utterance]:
    blocks: list[dict] = []
    current: dict | None = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        header = HEADER.match(line)
        if header:
            label, hours, minutes, seconds = header.groups()
            start_s = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
            current = {"tag": f"speaker_{label}", "start_ms": start_s * 1000, "lines": []}
            blocks.append(current)
        elif current is not None:
            current["lines"].append(line)

    utterances: list[Utterance] = []
    for index, block in enumerate(blocks):
        text = " ".join(block["lines"]).strip()
        if not text:
            continue
        if index + 1 < len(blocks):
            end_ms = blocks[index + 1]["start_ms"]
        else:
            end_ms = block["start_ms"] + len(text) * _MS_PER_CHAR
        utterances.append(
            Utterance(
                speaker_tag=block["tag"],
                start_ms=block["start_ms"],
                end_ms=end_ms,
                text=text,
            )
        )
    return utterances
