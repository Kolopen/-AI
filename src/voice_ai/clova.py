"""CLOVA Speech 긴 문장 인식 응답을 Utterance 목록으로 정규화한다.

CLOVA는 세그먼트 화자를 `speaker.label`로 주기도 하고 `diarization.label`로 주기도 해서
둘 다 받아들인다. 실제 응답을 받으면 tests/fixtures에 넣고 여기를 좁힐 것.
"""

from __future__ import annotations

from typing import Any

from .models import SpeakerProfile, Utterance

_SPEAKER_KEYS = ("speaker", "diarization")


def _speaker_tag(segment: dict[str, Any], index: int) -> str:
    for key in _SPEAKER_KEYS:
        holder = segment.get(key)
        if isinstance(holder, dict):
            label = holder.get("label") or holder.get("name")
            if label is not None:
                return f"speaker_{label}"
        elif isinstance(holder, (str, int)):
            return f"speaker_{holder}"
    return f"speaker_unknown_{index}"


def parse_segments(response: dict[str, Any]) -> list[Utterance]:
    segments = response.get("segments") or []
    utterances: list[Utterance] = []

    for index, segment in enumerate(segments):
        text = (segment.get("textEdited") or segment.get("text") or "").strip()
        if not text:
            continue
        utterances.append(
            Utterance(
                speaker_tag=_speaker_tag(segment, index),
                start_ms=int(segment.get("start") or 0),
                end_ms=int(segment.get("end") or 0),
                text=text,
                confidence=segment.get("confidence"),
            )
        )

    utterances.sort(key=lambda u: u.start_ms)
    return utterances


def group_by_speaker(utterances: list[Utterance]) -> list[SpeakerProfile]:
    profiles: dict[str, SpeakerProfile] = {}
    for utterance in utterances:
        profile = profiles.get(utterance.speaker_tag)
        if profile is None:
            profile = SpeakerProfile(speaker_tag=utterance.speaker_tag)
            profiles[utterance.speaker_tag] = profile
        profile.utterances.append(utterance)
    return sorted(profiles.values(), key=lambda p: -p.char_count)
