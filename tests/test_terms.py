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


def test_in_document_evidence_lowers_the_bar():
    """올바른 형태가 앞에 이미 나왔으면 조금 덜 닮아도 교정한다.

    실제 녹음에서 "간 보호제"가 먼저 실린 뒤 "감보제"로 잘못 나왔는데,
    유사도가 0.75라 하나의 엄격한 기준으로는 놓쳤다.
    """
    transcript = [
        Utterance("D", 0, 8_000, "간 수치가 높으시니까 간 보호제 있잖아요."),
        Utterance("D", 8_000, 16_000, "저 감보제 드시던 거 있잖아요."),
    ]

    corrections = find_corrections(transcript, DICTIONARY)

    caught = {c.original: c for c in corrections}
    assert "감보제" in caught
    assert caught["감보제"].corrected == "간보호제"
    assert caught["감보제"].evidence == "IN_DOCUMENT"


def test_dictionary_only_evidence_stays_strict():
    """사전에만 있는 말은 더 닮아야 교정한다. 약 이름을 잘못 고치는 쪽이 위험하다."""
    transcript = [Utterance("D", 0, 5_000, "저 감보제 드시던 거 있잖아요.")]

    corrections = find_corrections(transcript, DICTIONARY)

    assert not any(c.original == "감보제" for c in corrections)


def test_corrections_keep_the_original_in_view():
    """고친 글과 실제로 들린 것을 한 화면에서 봐야 한다.

    원문을 지우면 매니저가 녹음을 다시 듣기 전에는 판단할 근거가 없다.
    """
    from voice_ai.terms import TermCorrection, apply_corrections

    corrections = [
        TermCorrection("반수치가", "간수치", 0.75, "IN_DOCUMENT", 0),
        TermCorrection("감보제", "간보호제", 0.75, "IN_DOCUMENT", 0),
    ]

    applied = apply_corrections("반수치가 좋아지려면 감보제 드시면 됩니다", corrections)

    assert applied == "간수치(←반수치가) 좋아지려면 간보호제(←감보제) 드시면 됩니다"


def test_longer_originals_are_replaced_first():
    """짧은 것부터 바꾸면 긴 것 안쪽을 먼저 건드려 글자가 깨진다."""
    from voice_ai.terms import TermCorrection, apply_corrections

    corrections = [
        TermCorrection("간수", "간수치", 0.8, "DICTIONARY", 0),
        TermCorrection("간수치가", "간수치", 0.9, "IN_DOCUMENT", 0),
    ]

    applied = apply_corrections("간수치가 높아요", corrections)

    assert applied == "간수치(←간수치가) 높아요"


def test_text_without_corrections_is_untouched():
    from voice_ai.terms import apply_corrections

    assert apply_corrections("간수치가 정상입니다", []) == "간수치가 정상입니다"
