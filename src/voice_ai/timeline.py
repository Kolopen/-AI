"""수치를 시점별로 가른다.

"간수치 76, 34, 67, 23, 40" 이라고 나열하면 사람은 읽을 수 없다. 보호자가
알고 싶은 것은 좋아졌는지 나빠졌는지이고, 그건 시점이 붙어야 보인다.

한국어는 두 방향이 섞인다. 기준값은 숫자가 앞에 오고("40이 정상"), 시점은
뒤에 온다("작년에는 76"). 그래서 기준값을 먼저 떼어낸 다음 나머지를 시점에
붙인다. 순서를 바꾸면 "40 이상 보다 높으니까"의 40이 직전 시점으로 딸려간다.

전사가 깨져도 시점 단어는 대체로 살아남는다. 실제 녹음에서 어미가 모두
무너진 "작년는 76 의 34였고 이번에는 67 회 23" 에서도 같은 결과가 나왔다.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"\d+")

# 기준값. 숫자가 앞에 온다. 먼저 떼어내지 않으면 뒤 시점에 섞인다.
#
# "정상" 이 아니라 "정상기준" 으로 적는다. 같은 칸에 저번·이번 같은 시점이
# 들어가는데 "정상 200" 만 보면 시점으로 읽힌다. 실제로 그렇게 읽혔다.
REFERENCE = "정상기준"
_REFERENCE = re.compile(r"(\d+)\s*(?:이|가)?\s*(?:정상|이상|미만|이하|초과)")

# 전사가 "정상"을 흘린 경우. "200이 정산이 216" 처럼 숫자 뒤에 오는 두세 글자를
# 발음으로 견준다. 정산/정상 0.833 은 잡고, 뜻이 다른 정도/정상 0.545 는 거른다.
_LOOSE_REFERENCE = re.compile(r"(\d+)\s*(?:이|가)\s*([가-힣]{2,3})")
REFERENCE_SIMILARITY = 0.8

# 시점 표현. 이 자리부터 다음 표현 전까지의 숫자가 그 시점의 것이다.
_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("작년", re.compile(r"작년|재작년")),
    ("저번", re.compile(r"저번|지난번|지난|예전")),
    ("이번", re.compile(r"이번|올해|오늘|지금|현재")),
]
_ANY_MARKER = re.compile("|".join(p.pattern for _, p in _MARKERS))

# 어느 시점인지 말하지 않은 수치.
UNMARKED = "시점없음"


def _label(word: str) -> str:
    for name, pattern in _MARKERS:
        if pattern.match(word):
            return name
    return UNMARKED  # pragma: no cover


def _looks_like_reference(word: str) -> bool:
    from .terms import phonetic_similarity

    return phonetic_similarity(word, "정상") >= REFERENCE_SIMILARITY


def group_by_time(text: str) -> dict[str, list[str]]:
    """문장의 숫자를 시점별로 묶는다. 시점을 못 찾으면 UNMARKED 에 모은다."""
    grouped: dict[str, list[str]] = {}

    spans: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group(1)) for m in _REFERENCE.finditer(text)
    ]
    for match in _LOOSE_REFERENCE.finditer(text):
        if any(start <= match.start() < end for start, end, _ in spans):
            continue
        if _looks_like_reference(match.group(2)):
            spans.append((match.start(), match.end(), match.group(1)))

    for _, _, value in sorted(spans):
        grouped.setdefault(REFERENCE, []).append(value)

    # 기준값 자리를 공백으로 덮어 뒤 단계에서 두 번 세지 않게 한다.
    rest = list(text)
    for start, end, _ in spans:
        rest[start:end] = " " * (end - start)
    rest = "".join(rest)

    marks = [(m.start(), m.group()) for m in _ANY_MARKER.finditer(rest)]
    head = rest[: marks[0][0]] if marks else rest
    if _NUMBER.search(head):
        grouped.setdefault(UNMARKED, []).extend(_NUMBER.findall(head))

    for index, (position, word) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(rest)
        numbers = _NUMBER.findall(rest[position:end])
        if numbers:
            grouped.setdefault(_label(word), []).extend(numbers)

    return grouped


def has_marker(text: str) -> bool:
    """시점이나 기준값 표현이 있는지. 다른 엔진 전사를 빌릴지 정할 때 쓴다."""
    if _ANY_MARKER.search(text) or _REFERENCE.search(text):
        return True
    return any(_looks_like_reference(m.group(2)) for m in _LOOSE_REFERENCE.finditer(text))
