"""의사 발언에서 사실만 뽑기.

문장을 발췌하면 전사의 깨진 어미가 따라온다("40 이상 보다 높 으니까").
값만 뽑으면 그 문제가 없고, 지어내는 부분도 없다.
"""

from voice_ai import terminology
from voice_ai.facts import extract
from voice_ai.models import Role, Utterance

TERMS = terminology.load("내과")
ROLES = {"D": Role.DOCTOR, "P": Role.PATIENT}


def test_values_are_tied_to_the_test_named_with_them():
    found = extract(
        [Utterance("D", 0, 9_000, "콜레스테롤이 200이 정상인데 216이니까")], ROLES, TERMS
    )

    assert [(m.test, m.values) for m in found.measurements] == [("콜레스테롤", ["200", "216"])]
    assert found.measurements[0].inferred is False


def test_the_test_name_carries_to_the_next_breath():
    """의사는 이름을 한 번 대고 수치를 이어 말한다.

    "간 수치가 좀 높죠" 다음 숨에 "작년에는 76에 34였고" 가 온다. 이름이 없다고
    버리면 정작 중요한 값이 사라진다.
    """
    found = extract(
        [
            Utterance("D", 0, 9_000, "간 수치가 좀 높죠"),
            Utterance("D", 9_000, 17_000, "작년에는 76에 34였고 이번에는 67에 23이에요"),
        ],
        ROLES,
        TERMS,
    )

    carried = [m for m in found.measurements if m.inferred]
    assert len(carried) == 1
    assert carried[0].test == "간수치"
    assert carried[0].values == ["76", "34", "67", "23"]


def test_a_distant_number_does_not_borrow_the_test_name():
    found = extract(
        [
            Utterance("D", 0, 9_000, "간 수치가 좀 높죠"),
            Utterance("D", 120_000, 125_000, "두 달분 드렸으니까 2주 뒤에 오세요"),
        ],
        ROLES,
        TERMS,
    )

    assert [m for m in found.measurements if m.inferred] == []


def test_something_the_doctor_called_fine_is_not_a_diagnosis():
    """"빈혈은 괜찮으시고" 를 진단에 올리면 없는 병이 기록에 남는다."""
    found = extract(
        [Utterance("D", 0, 9_000, "빈혈 소변 검사 이런 건 괜찮으시고 지방간이 있으세요")],
        ROLES,
        TERMS,
    )

    assert [name for name, _ in found.diagnoses] == ["지방간"]
    assert found.normal == ["빈혈"]


def test_lifestyle_advice_survives_the_honorific_contraction():
    """-시- 는 어미와 만나면 "셔"로 줄어든다. 줄이시어야 → 줄이셔야."""
    found = extract(
        [
            Utterance("D", 0, 9_000, "허리를 줄이셔야 되고 술도 줄이셔야 되고"),
            Utterance("D", 9_000, 15_000, "운동 열심히 하셔서 체중 줄이시면 좋습니다"),
        ],
        ROLES,
        TERMS,
    )

    assert "허리 줄이기" in found.lifestyle
    assert "술 줄이기" in found.lifestyle
    assert "체중 줄이기" in found.lifestyle
    assert "운동" in found.lifestyle


def test_the_patient_is_not_quoted_as_the_doctor():
    """환자가 말한 수치를 진료 소견으로 올릴 수는 없다."""
    found = extract(
        [Utterance("P", 0, 9_000, "저번에 콜레스테롤이 300이라고 들었어요")], ROLES, TERMS
    )

    assert found.measurements == []
