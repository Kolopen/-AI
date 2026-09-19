"""의료 용어 오인식 교정.

실제 앵커에서 "간 보호제"가 같은 대화 안에서 한 번은 맞게, 한 번은 "관보제"로
전사됐다. 내용은 합성이지만 오인식 패턴은 실제 그대로다.
"""

from voice_ai.models import Utterance
from voice_ai.terms import find_corrections, phonetic_similarity, to_jamo

DICTIONARY = {"간보호제", "고지혈증", "콜레스테롤", "지방간", "내장지방", "항생제"}

# 앞에서는 맞게, 뒤에서는 틀리게 전사된 상황.
TRANSCRIPT = [
    Utterance("speaker_1", 0, 5000, "간 수치가 정상보다 좀 높으시니까 그 간 보호제 있잖아요."),
    Utterance("speaker_2", 5000, 6000, "네."),
    Utterance("speaker_1", 6000, 12000, "저 관보제 드시던 거 있잖아요. 지방간은 체중이 빠지면 돌아옵니다."),
]


def test_decomposes_compound_vowels():
    """겹모음을 풀어야 간과 관이 한 끗 차이로 잡힌다."""
    assert to_jamo("간") == "ㄱㅏㄴ"
    assert to_jamo("관") == "ㄱㅗㅏㄴ"


def test_misrecognized_drug_name_is_caught():
    corrections = find_corrections(TRANSCRIPT, DICTIONARY)

    assert len(corrections) == 1
    assert corrections[0].original == "관보제"
    assert corrections[0].corrected == "간보호제"
    # 올바른 형태가 같은 전사문에 이미 있으므로 가장 믿을 만한 근거다.
    assert corrections[0].evidence == "IN_DOCUMENT"


def test_particles_are_not_treated_as_misrecognition():
    """'지방간은'은 오인식이 아니라 조사가 붙은 것이다."""
    corrections = find_corrections(TRANSCRIPT, DICTIONARY)

    assert not any("지방간" in c.original for c in corrections)


def test_unrelated_words_are_left_alone():
    plain = [Utterance("speaker_1", 0, 3000, "오늘 날씨가 참 좋습니다.")]

    assert find_corrections(plain, DICTIONARY) == []


def test_phonetic_similarity_separates_near_and_far():
    assert phonetic_similarity("관보제", "간보호제") > 0.8
    assert phonetic_similarity("항생제", "콜레스테롤") < 0.4
