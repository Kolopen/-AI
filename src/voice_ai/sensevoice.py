"""SenseVoice(FunASR) 출력을 Utterance 목록으로 정규화한다.

SenseVoice는 전사와 함께 언어·감정·음향이벤트를 태그로 붙여 준다.

    <|ko|><|NEUTRAL|><|Speech|><|withitn|>간 수치가 좀 높으시네요.

본문만 쓰지만 언어 태그는 버리지 않는다. 한국어 진료 녹음에서 다른 언어가
찍히면 그 구간은 전사를 믿을 수 없다는 뜻이라 경고로 올린다.

SenseVoice는 화자분리를 하지 않는다. 그래서 여기서 나온 발화에는 화자가 없고,
`roles.sentence_role`로 문장별 역할을 가르거나 별도 화자분리를 붙여야 한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import Utterance

_TAG = re.compile(r"<\|([^|>]+)\|>")

# 본문이 아니라 분류 결과인 태그들. 언어 태그는 이 목록에 없는 값으로 판별한다.
_EMOTIONS = {"NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}
_EVENTS = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough", "Event_UNK"}
_ITN = {"withitn", "woitn"}

EXPECTED_LANGUAGE = "ko"


@dataclass
class Tags:
    language: str | None = None
    emotion: str | None = None
    events: tuple[str, ...] = ()


def parse_tags(raw_text: str) -> Tags:
    language = emotion = None
    events: list[str] = []

    for value in _TAG.findall(raw_text):
        if value in _EMOTIONS:
            emotion = value
        elif value in _EVENTS:
            events.append(value)
        elif value in _ITN:
            continue
        elif language is None:
            language = value

    return Tags(language=language, emotion=emotion, events=tuple(events))


def strip_tags(raw_text: str) -> str:
    return _TAG.sub("", raw_text).strip()


def parse_chunks(
    chunks: list[dict[str, Any]], *, expected_language: str = EXPECTED_LANGUAGE
) -> tuple[list[Utterance], list[str]]:
    """SenseVoice 청크 목록을 (발화, 경고) 로 바꾼다.

    청크는 `long_audio_no_vad.py`가 내보내는 모양을 따른다.
    start_ms, end_ms, raw_text 또는 text 를 가진다.
    """
    utterances: list[Utterance] = []
    warnings: list[str] = []

    for chunk in chunks:
        raw = chunk.get("raw_text") or ""
        text = (chunk.get("text") or strip_tags(raw)).strip()
        if not text:
            continue

        start_ms = int(chunk.get("start_ms") or 0)
        tags = parse_tags(raw)

        if tags.language and tags.language != expected_language:
            warnings.append(
                f"{start_ms}ms 구간이 {tags.language}로 인식됐습니다. "
                "한국어 진료 녹음이라면 이 구간의 전사를 믿을 수 없습니다."
            )

        utterances.append(
            Utterance(
                speaker_tag="",  # SenseVoice는 화자를 구분하지 않는다.
                start_ms=start_ms,
                end_ms=int(chunk.get("end_ms") or start_ms),
                text=text,
            )
        )

    utterances.sort(key=lambda u: u.start_ms)
    return utterances, warnings
