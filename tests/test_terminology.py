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


def test_particle_suffix_is_not_dragged_to_another_term():
    """'콜레스테롤이'는 조사가 붙었을 뿐인데 '콜레스테롤약'으로 끌려갔었다."""
    terms = terminology.load("내과")
    transcript = [Utterance("D", 0, 6_000, "콜레스테롤이 200이 정상인데 216입니다.")]

    corrections = find_corrections(transcript, set(terms.all_terms))

    assert not any("콜레스테롤" in c.original for c in corrections)


def test_two_letter_words_are_not_turned_into_drug_names():
    """'이제'가 '이뇨제'로, '였고'가 '연고'로 바뀌면 리포트가 엉뚱해진다."""
    terms = terminology.load("내과")
    transcript = [Utterance("D", 0, 6_000, "이제 작년에는 수치가 높았고 올해는 괜찮습니다.")]

    corrections = find_corrections(transcript, set(terms.all_terms))

    assert corrections == []


def test_real_words_swapped_for_each_other_are_flagged_by_context():
    """신장(콩팥)과 심장은 0.833으로 닮았지만 둘 다 실재하는 말이다.

    발음 유사도로는 못 거른다. 실제 녹음에서 두 엔진이 모두 "피검사에서는
    신장이라든지 소변 검사"를 "심장"으로 썼다.
    """
    from voice_ai.terms import find_confusions

    terms = terminology.load("내과")
    transcript = [
        Utterance("D", 0, 9_000, "피검사에서는 심장이라든지 빈혈, 소변 검사 이런 건 괜찮으시고,")
    ]

    found = find_confusions(transcript, terms.confusable)

    assert [(c.written, c.suspected) for c in found] == [("심장", "신장")]


def test_a_term_with_its_own_context_is_left_alone():
    """심전도 이야기 중의 "심장"은 심장이 맞다."""
    from voice_ai.terms import find_confusions

    terms = terminology.load("내과")
    transcript = [
        Utterance("D", 0, 9_000, "심전도 찍어보니 심장은 괜찮으시고 부정맥도 없으세요.")
    ]

    assert find_confusions(transcript, terms.confusable) == []


def test_distant_context_does_not_trigger_a_confusion():
    """20초 넘게 떨어진 말은 다른 화제다."""
    from voice_ai.terms import find_confusions

    terms = terminology.load("내과")
    transcript = [
        Utterance("D", 0, 5_000, "심장은 괜찮으십니다."),
        Utterance("D", 120_000, 125_000, "소변 검사도 해보겠습니다."),
    ]

    assert find_confusions(transcript, terms.confusable) == []
