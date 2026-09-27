"""운영 지표. 관리자가 보는 쪽이다.

경고는 매니저에게 보내지 않는다. 매니저는 진료실에서 환자 옆에 있고,
"화자분리가 한 사람을 쪼갰을 수 있습니다" 같은 말로 할 수 있는 일이 없다.
사전을 고치고 엔진을 바꾸는 것은 운영하는 쪽의 일이다.

대신 진료 한 건마다 산문을 읽게 하지 않는다. 관리자에게 필요한 것은 여러
건에 걸친 추세다. 어느 진료과에서 판정이 흔들리는지, 어떤 용어가 자주
엇갈리는지는 숫자로 봐야 보인다. 그래서 세부 내용과 함께 세어서 넘긴다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 경고 문구에서 종류를 알아내는 표. 문구가 바뀌면 여기도 바꿔야 하므로,
# 새 경고를 만들 때는 분류를 함께 정한다.
_KINDS: list[tuple[str, re.Pattern[str]]] = [
    ("unresolved_speaker", re.compile(r"역할 신호가 없어")),
    ("split_suspected", re.compile(r"merge-non-doctor")),
    ("unspaced_transcript", re.compile(r"띄어쓰기가 없습니다")),
    ("confusable_term", re.compile(r"일 수 있습니다\. 주변에")),
    ("number_disagreement", re.compile(r"숫자가 엇갈립니다")),
    ("term_disagreement", re.compile(r"다른 엔진")),
]


@dataclass
class Operations:
    counts: dict[str, int] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)


def classify_warning(warning: str) -> str:
    for kind, pattern in _KINDS:
        if pattern.search(warning):
            return kind
    return "other"


def summarize(warnings: list[str], *, term_corrections: int = 0) -> Operations:
    """경고를 종류별로 세고 원문도 남긴다."""
    counts: dict[str, int] = {}
    for warning in warnings:
        kind = classify_warning(warning)
        counts[kind] = counts.get(kind, 0) + 1
    if term_corrections:
        counts["term_correction"] = term_corrections
    return Operations(counts=counts, details=list(warnings))
