"""진료과별 용어 사전.

사전이 클수록 엉뚱한 교정이 늘어서 진료과로 나눈다. 약품과 질환·검사를
나눠 두는 것은 리포트 약품란에 병명이 들어가지 않게 하기 위해서다.
"""

import pytest

from voice_ai import terminology
from voice_ai.models import Utterance
from voice_ai.terms import find_corrections


def test_common_terms_load_without_a_department():
    terms = terminology.load()

    assert "항생제" in terms.drugs
    assert "피검사" in terms.tests


def test_department_terms_are_added_to_common():
    terms = terminology.load("내과")

    assert "간보호제" in terms.drugs  # 내과
    assert "항생제" in terms.drugs  # 공통
    assert "지방간" in terms.conditions


def test_drugs_and_conditions_stay_separate():
    """약품란에 병명이 들어가면 리포트를 못 쓴다."""
    terms = terminology.load("내과")

    assert "지방간" not in terms.drugs
    assert "콜레스테롤" not in terms.drugs


def test_unknown_department_names_what_exists():
    with pytest.raises(FileNotFoundError, match="내과"):
        terminology.load("존재하지않는과")


def test_confusable_pairs_are_both_present():
    """신장(콩팥)과 심장은 실제 녹음에서 서로 잘못 전사됐다.

    둘 다 사전에 있어야 한쪽을 다른 쪽으로 고치는 일이 생기지 않는다.
    """
    terms = terminology.load("내과")

    assert "신장" in terms.all_terms
    assert "심장" in terms.all_terms


def test_fragment_of_a_known_term_is_not_corrected():
    """"복부 내장지방"이 띄어 써져 "내장"만 남아도 "신장"으로 고치면 안 된다."""
    terms = terminology.load("내과")
    transcript = [
        Utterance("D", 0, 5_000, "피검사에서 신장이랑 빈혈은 괜찮으시고요."),
        Utterance("D", 5_000, 12_000, "복부 내장 지방 때문에 그런 경우가 많거든요."),
    ]

    corrections = find_corrections(transcript, set(terms.all_terms))

    assert not any(c.original == "내장" for c in corrections)
