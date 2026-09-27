"""진료과별 용어 사전.

용어는 두 곳에 쓰인다. 오인식 교정의 기준이 되고, 리포트의 약품란을 채운다.
그래서 약품과 질환·검사를 나눠 둔다. 섞으면 약품란에 병명이 들어간다.

진료과를 나누는 이유는 사전이 클수록 엉뚱한 교정이 늘기 때문이다. 정형외과
진료에 내과 약 이름이 후보로 끼어들 이유가 없다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TERMS_DIR = Path(__file__).parent / "data" / "terms"

# 진료과를 고르지 않아도 늘 함께 불러오는 사전.
COMMON = "공통"

_SECTION = re.compile(r"^\[(drug|condition|test)\]$")


@dataclass(frozen=True)
class Terminology:
    drugs: frozenset[str] = frozenset()
    conditions: frozenset[str] = frozenset()
    tests: frozenset[str] = frozenset()

    @property
    def all_terms(self) -> frozenset[str]:
        return self.drugs | self.conditions | self.tests

    def merged_with(self, other: Terminology) -> Terminology:
        return Terminology(
            drugs=self.drugs | other.drugs,
            conditions=self.conditions | other.conditions,
            tests=self.tests | other.tests,
        )


def available() -> list[str]:
    """고를 수 있는 진료과 이름."""
    return sorted(p.stem for p in TERMS_DIR.glob("*.txt") if p.stem != COMMON)


def _read(path: Path) -> Terminology:
    buckets: dict[str, set[str]] = {"drug": set(), "condition": set(), "test": set()}
    section: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        header = _SECTION.match(line)
        if header:
            section = header.group(1)
        elif section:
            buckets[section].add(line)

    return Terminology(
        drugs=frozenset(buckets["drug"]),
        conditions=frozenset(buckets["condition"]),
        tests=frozenset(buckets["test"]),
    )


def load(department: str | None = None) -> Terminology:
    """공통 사전에 진료과 사전을 얹어 돌려준다."""
    terminology = _read(TERMS_DIR / f"{COMMON}.txt")
    if not department:
        return terminology

    path = TERMS_DIR / f"{department}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"'{department}' 사전이 없습니다. 있는 것: {', '.join(available()) or '없음'}"
        )
    return terminology.merged_with(_read(path))
