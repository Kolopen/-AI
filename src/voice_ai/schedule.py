"""후속 예약 날짜를 뽑는다.

의사가 "10월 20일날 오세요"처럼 날짜를 직접 말하면 매니저가 그 자리에서
예약을 잡아야 한다. 문장만 `next_visit_note`에 남기면 매니저가 다시 읽고
손으로 옮겨 적게 되고, 옮기다 숫자를 틀린다. 날짜는 값으로 뽑아 둔다.

연도는 말하지 않는다. "10월 20일"이라고만 하므로 진료일을 기준으로 정한다.
후속 예약은 언제나 진료일 뒤이므로, 앞이면 해를 넘긴 것으로 본다.

"두 달 뒤에 오세요"처럼 기간만 말한 경우는 날짜를 만들지 않는다. 의사가
정한 날이 아니라 어림이다. 없는 날짜를 만들어 넣으면 매니저가 그대로
예약을 잡는다. 이런 문장은 지금처럼 `next_visit_note`에만 남긴다.

숫자는 아라비아 숫자만 읽는다. "이십일"은 21일도 되고 20일도 되므로
한글 숫자를 풀면 틀린 날짜가 나온다. 못 읽으면 문장이 메모에 남는다.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from .models import Utterance

# 방문을 뜻하는 말. 날짜가 예약인지 지난 이야기인지 이걸로 가른다.
_VISIT = re.compile(r"오세요|오시|오라|와서|와야|뵈|봬|예약|잡아|잡을|방문|내원")

# 지난 이야기. "작년 10월 20일에 검사하셨죠"를 예약으로 읽으면 안 된다.
_PAST = re.compile(r"작년|재작년|지난|저번|예전")

# 숫자 뒤에 이게 붙으면 날짜가 아니라 기간이다. "20일분", "3일 뒤", "10일째"
_DURATION_TAIL = re.compile(r"\s*(?:분|치|씩|간|동안|어치|뒤|후|째|정도|만에|지나)")

_MONTH_DAY = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_NEXT_MONTH = re.compile(r"다음\s*달\s*(\d{1,2})\s*일")
_THIS_MONTH = re.compile(r"이번\s*달\s*(\d{1,2})\s*일")
_BARE_DAY = re.compile(r"(\d{1,2})\s*일")
_NEXT_YEAR = re.compile(r"내년|명년")

# "3시간 뒤"의 시는 시각이 아니다.
_TIME = re.compile(r"(오전|오후)?\s*(\d{1,2})\s*시(?!간)\s*(?:(\d{1,2})\s*분)?")

# 오전·오후를 말하지 않았을 때 그대로 믿는 범위. 진료 시간 밖이면 버린다.
CLINIC_OPEN = 8
CLINIC_CLOSE = 18

# 날짜와 시각이 차지하는 자리. "10월 20일 오전 10시"의 10, 20, 10 은 검사 수치도
# 처방 기간도 아니다. 숫자를 세기 전에 이 자리를 지운다.
_DATE_SPAN = re.compile(
    r"(?:다음|이번)\s*달\s*\d{1,2}\s*일"
    r"|\d{1,2}\s*월\s*\d{1,2}\s*일"
    r"|(?:오전|오후)\s*\d{1,2}\s*시(?!간)(?:\s*\d{1,2}\s*분)?"
    r"|\d{1,2}\s*시(?!간)(?:\s*\d{1,2}\s*분)?"
)


def mask_dates(text: str) -> str:
    """날짜와 시각 자리를 공백으로 바꾼다. 길이는 그대로 둬서 위치가 어긋나지 않게 한다."""
    return _DATE_SPAN.sub(lambda found: " " * len(found.group()), text)


@dataclass
class NextVisit:
    """의사가 말한 후속 예약 시점."""

    day: dt.date
    hour: int | None
    minute: int | None
    utterance: Utterance
    sentence: str

    @property
    def iso(self) -> str:
        """`bodeul.session_reports.next_visit_at`에 넣을 값.

        시각을 말하지 않았으면 날짜만 준다. 시간대는 붙이지 않는다.
        진료가 한국에서만 일어나므로 Core API가 KST로 읽으면 된다.
        """
        if self.hour is None:
            return self.day.isoformat()
        return f"{self.day.isoformat()}T{self.hour:02d}:{self.minute or 0:02d}"


def _build(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _shift_month(anchor: dt.date, months: int) -> tuple[int, int]:
    index = anchor.year * 12 + anchor.month - 1 + months
    return index // 12, index % 12 + 1


def resolve_date(sentence: str, consult: dt.date) -> dt.date | None:
    """문장에서 날짜를 읽는다. 연도는 진료일로 채운다."""
    next_year = bool(_NEXT_YEAR.search(sentence))

    found = _MONTH_DAY.search(sentence)
    if found:
        month, day = int(found.group(1)), int(found.group(2))
        year = consult.year + (1 if next_year else 0)
        candidate = _build(year, month, day)
        if candidate is None or next_year or candidate >= consult:
            return candidate
        return _build(year + 1, month, day)

    found = _NEXT_MONTH.search(sentence)
    if found:
        year, month = _shift_month(consult, 1)
        return _build(year, month, int(found.group(1)))

    found = _THIS_MONTH.search(sentence)
    if found:
        return _build(consult.year, consult.month, int(found.group(1)))

    for found in _BARE_DAY.finditer(sentence):
        if _DURATION_TAIL.match(sentence, found.end()):
            continue
        day = int(found.group(1))
        candidate = _build(consult.year, consult.month, day)
        if candidate is not None and candidate >= consult:
            return candidate
        year, month = _shift_month(consult, 1)
        return _build(year, month, day)
    return None


def resolve_time(sentence: str) -> tuple[int, int] | None:
    found = _TIME.search(sentence)
    if found is None:
        return None
    marker, hour_text, minute_text = found.groups()
    hour = int(hour_text)
    minute = int(minute_text) if minute_text else 0
    if hour > 23 or minute > 59:
        return None
    if marker == "오후":
        if hour < 12:
            hour += 12
    elif marker == "오전":
        if hour == 12:
            hour = 0
    elif not CLINIC_OPEN <= hour <= CLINIC_CLOSE:
        # 오전·오후를 말하지 않았는데 진료 시간 밖이다. 어느 쪽인지 알 수 없다.
        return None
    return hour, minute


def find_next_visit(
    pairs: list[tuple[Utterance, str]], *, consult_date: dt.date
) -> NextVisit | None:
    """의사 문장에서 후속 예약 날짜를 찾는다.

    여러 번 나오면 마지막 것을 쓴다. 의사가 말을 고치는 일이 있다.
    """
    latest: NextVisit | None = None
    for utterance, sentence in pairs:
        if _PAST.search(sentence):
            continue
        # 날짜와 방문 표현이 다른 문장으로 갈릴 수 있으므로 발화 전체를 본다.
        if not (_VISIT.search(sentence) or _VISIT.search(utterance.text)):
            continue
        day = resolve_date(sentence, consult_date)
        if day is None:
            continue
        clock = resolve_time(sentence)
        latest = NextVisit(
            day=day,
            hour=clock[0] if clock else None,
            minute=clock[1] if clock else None,
            utterance=utterance,
            sentence=sentence,
        )
    return latest
