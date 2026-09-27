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
#
# 올바른 형태가 같은 전사문에 이미 나왔으면 근거가 훨씬 강하므로 기준을 낮춘다.
# 실제 녹음에서 "간 보호제"가 앞에 정확히 실린 뒤 뒤에서 "감보제"로 잘못 나왔는데,
# 둘의 유사도가 0.75라 하나의 엄격한 기준으로는 놓쳤다.
THRESHOLD_IN_DOCUMENT = 0.70
THRESHOLD_DICTIONARY = 0.80

# 두 글자 말은 우연히 닮기 쉽다. "이제"가 "이뇨제"로, "였고"가 "연고"로 끌려갔다.
# 흔한 부사와 어미가 약 이름으로 바뀌면 리포트가 엉뚱해지므로 사실상 일치를 요구한다.
SHORT_CANDIDATE_LENGTH = 2
THRESHOLD_SHORT = 0.95

# 두 글자 의학 용어는 사전 근거만으로 고치지 않는다. 복통·두통·간염 같은 말은
# 흔한 한국어와 우연히 닮는다. 실제로 "추석 잘 보내시고요"가 쪼개져 나온
# "보고요"가 "복통"으로 바뀌었다(0.833). 인사말이 증상이 되어 리포트에 실린다.
# 올바른 형태가 같은 전사문에 이미 나왔다면(IN_DOCUMENT) 근거가 다르므로 허용한다.
SHORT_TERM_LENGTH = 2


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
            # 사전 용어와 포함 관계면 오인식이 아니다. 양쪽 다 걸러야 한다.
            #
            # 후보가 용어의 일부인 경우: "복부 내장지방"이 띄어 써져 "내장"만 남은 것.
            # 이걸 발음이 비슷한 "신장"으로 고치면 콩팥 이야기로 둔갑한다.
            #
            # 후보가 용어를 품은 경우: "콜레스테롤이"처럼 조사가 붙은 것.
            # 그냥 두면 "콜레스테롤약" 같은 다른 용어로 끌려간다.
            if any(candidate in term or term in candidate for term in dictionary):
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

            if best_term is None:
                continue

            in_document = best_term in present
            if not in_document and len(best_term) <= SHORT_TERM_LENGTH:
                continue
            threshold = THRESHOLD_IN_DOCUMENT if in_document else THRESHOLD_DICTIONARY
            if len(candidate) <= SHORT_CANDIDATE_LENGTH:
                threshold = max(threshold, THRESHOLD_SHORT)
            if best_score < threshold:
                continue
            if (candidate, best_term) in seen:
                continue

            seen.add((candidate, best_term))
            corrections.append(
                TermCorrection(
                    original=candidate,
                    corrected=best_term,
                    similarity=round(best_score, 3),
                    evidence="IN_DOCUMENT" if in_document else "DICTIONARY",
                    start_ms=utterance.start_ms,
                )
            )

    corrections.sort(key=lambda c: (c.evidence != "IN_DOCUMENT", -c.similarity))
    return corrections


# 이 정도로 발음이 닮은 용어끼리만 서로 의심한다. 신장/심장이 0.833이다.
CONFUSABLE_SIMILARITY = 0.75

# 문맥 단어를 이만큼 떨어진 발화까지 본다. 의사는 한 화제를 몇 문장에 걸쳐 말한다.
CONTEXT_WINDOW_MS = 20_000


@dataclass
class Confusion:
    written: str
    suspected: str
    cue: str
    start_ms: int


def find_confusions(
    utterances: list[Utterance], confusable: dict[str, tuple[str, ...]]
) -> list[Confusion]:
    """실재하는 두 용어가 서로 바뀌어 전사된 것을 문맥으로 의심한다.

    "신장"과 "심장"은 발음이 0.833으로 닮았지만 둘 다 사전에 있는 말이라
    find_corrections 가 거른다. 대신 주변에 무엇이 함께 나왔는지를 본다.
    소변 검사 옆의 "심장"은 콩팥일 가능성이 높다.

    고치지는 않는다. 장기 이름을 잘못 바꾸는 쪽이 틀린 채로 두는 것보다 위험하다.
    사람이 녹음을 다시 듣고 정하도록 표시만 한다.
    """
    found: list[Confusion] = []
    seen: set[tuple[str, str, int]] = set()

    for utterance in utterances:
        for written, own_cues in confusable.items():
            if written not in utterance.text:
                continue

            nearby = " ".join(
                other.text
                for other in utterances
                if abs(other.start_ms - utterance.start_ms) <= CONTEXT_WINDOW_MS
            )
            # 제 문맥이 하나라도 있으면 쓰인 대로 믿는다.
            if any(cue in nearby for cue in own_cues):
                continue

            for other, other_cues in confusable.items():
                if other == written:
                    continue
                if phonetic_similarity(written, other) < CONFUSABLE_SIMILARITY:
                    continue
                cue = next((c for c in other_cues if c in nearby), None)
                if cue is None:
                    continue
                key = (written, other, utterance.start_ms)
                if key in seen:
                    continue
                seen.add(key)
                found.append(Confusion(written, other, cue, utterance.start_ms))

    return found


def apply_corrections(text: str, corrections: list[TermCorrection]) -> str:
    """교정을 본문에 넣되 원문을 괄호로 남긴다.

    읽기 좋게 고친 글과, 실제로 무엇이 들렸는지를 한 화면에서 볼 수 있어야 한다.
    원문을 지우면 매니저가 녹음을 다시 듣기 전에는 판단할 근거가 없어진다.

    한 번에 훑고 한 번에 만든다. 하나씩 치환하면 앞서 끼워 넣은 표시 안쪽을
    다음 교정이 또 건드려 "간수치(←간수)치(←..." 처럼 글자가 뭉개진다.
    """
    spans: list[tuple[int, int, str]] = []
    for correction in corrections:
        marked = f"{correction.corrected}(←{correction.original})"
        start = text.find(correction.original)
        while start != -1:
            spans.append((start, start + len(correction.original), marked))
            start = text.find(correction.original, start + 1)

    # 겹치면 긴 쪽을 남긴다. "간수치가"를 놔두고 "간수"를 버려야 말이 된다.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen: list[tuple[int, int, str]] = []
    for span in spans:
        if chosen and span[0] < chosen[-1][1]:
            continue
        chosen.append(span)

    out, cursor = [], 0
    for start, end, marked in chosen:
        out.append(text[cursor:start])
        out.append(marked)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
