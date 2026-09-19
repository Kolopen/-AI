"""의료 용어 오인식 교정.

앵커 녹음에서 "간 보호제"가 같은 대화 안에서 한 번은 맞게, 한 번은 "관보제"로
전사됐다. 사람이 이걸 알아보는 근거는 세 가지다. 올바른 형태가 같은 문서에
이미 나왔고, 발음이 비슷하고, 문맥이 간 이야기였다.

첫 번째가 가장 강하다. 같은 진료 안에서 같은 약을 두 번 말하는 일은 흔하고,
STT가 두 번 다 똑같이 틀리는 일은 드물다.

교정은 제안만 하고 원문을 덮어쓰지 않는다. 약 이름을 잘못 고치면 리포트가
조용히 틀린 채로 나간다. 확정은 운영자가 한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import Utterance

_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

# 겹모음·겹받침을 풀어야 간/관, 갑/값 같은 한 끗 차이가 거리 1로 잡힌다.
_COMPOUND = {
    "ㅘ": "ㅗㅏ", "ㅙ": "ㅗㅐ", "ㅚ": "ㅗㅣ", "ㅝ": "ㅜㅓ", "ㅞ": "ㅜㅔ",
    "ㅟ": "ㅜㅣ", "ㅢ": "ㅡㅣ", "ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ",
    "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ",
    "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ",
}

_TOKEN = re.compile(r"[가-힣]{2,}")
_SPACE = re.compile(r"\s+")

# 이 아래로 비슷하면 다른 단어로 본다. 약 이름을 잘못 고치는 쪽이 더 위험하므로 보수적으로 잡는다.
SIMILARITY_THRESHOLD = 0.78


def to_jamo(text: str) -> str:
    """한글을 자모로 푼다. 발음이 비슷한 오인식을 거리로 재기 위한 것."""
    out: list[str] = []
    for char in text:
        code = ord(char)
        if _HANGUL_BASE <= code <= _HANGUL_LAST:
            offset = code - _HANGUL_BASE
            for jamo in (_CHO[offset // 588], _JUNG[(offset % 588) // 28], _JONG[offset % 28]):
                if jamo != " ":
                    out.append(_COMPOUND.get(jamo, jamo))
        else:
            out.append(char)
    return "".join(out)


def phonetic_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, to_jamo(a), to_jamo(b)).ratio()


@dataclass
class TermCorrection:
    original: str
    corrected: str
    similarity: float
    # IN_DOCUMENT: 올바른 형태가 같은 전사문에 이미 나왔다. 가장 믿을 만하다.
    # DICTIONARY: 사전에만 있다.
    evidence: str
    start_ms: int


def _candidates(text: str) -> set[str]:
    """어절 하나와 인접 두 어절을 붙인 것까지 후보로 본다.

    "간 보호제"처럼 띄어 쓴 경우와 "관보제"처럼 붙은 경우가 섞여 나오기 때문이다.
    """
    words = _TOKEN.findall(text)
    found = set(words)
    found.update(a + b for a, b in zip(words, words[1:]))
    return found


def find_corrections(
    utterances: list[Utterance], dictionary: set[str]
) -> list[TermCorrection]:
    """전사문에서 사전 용어의 오인식으로 보이는 표현을 찾는다."""
    whole = _SPACE.sub("", " ".join(u.text for u in utterances))
    present = {term for term in dictionary if term in whole}

    corrections: list[TermCorrection] = []
    seen: set[tuple[str, str]] = set()

    for utterance in utterances:
        for candidate in _candidates(utterance.text):
            if candidate in dictionary:
                continue

            best_term, best_score = None, 0.0
            for term in dictionary:
                # 한쪽이 다른 쪽을 품고 있으면 오인식이 아니라 조사가 붙었거나
                # 더 큰 말의 일부다. "지방간은", "복부내장지방" 같은 것들.
                if term in candidate or candidate in term:
                    continue
                score = phonetic_similarity(candidate, term)
                if score > best_score:
                    best_term, best_score = term, score

            if best_term is None or best_score < SIMILARITY_THRESHOLD:
                continue
            if (candidate, best_term) in seen:
                continue

            seen.add((candidate, best_term))
            corrections.append(
                TermCorrection(
                    original=candidate,
                    corrected=best_term,
                    similarity=round(best_score, 3),
                    evidence="IN_DOCUMENT" if best_term in present else "DICTIONARY",
                    start_ms=utterance.start_ms,
                )
            )

    corrections.sort(key=lambda c: (c.evidence != "IN_DOCUMENT", -c.similarity))
    return corrections
