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


def test_a_shaky_transcription_maps_to_the_standard_name():
    """전사가 "허리리", "운동부" 로 흔들려도 같은 지도로 모여야 한다.

    안 모으면 "운동"과 "운동부"가 따로 남아 지도가 둘로 보인다.
    """
    found = extract(
        [
            Utterance("D", 0, 9_000, "허리리를 줄이셔야 되고"),
            Utterance("D", 9_000, 15_000, "운동 열심히 하시고 체중 줄이시면"),
            Utterance("D", 15_000, 20_000, "운동부 열심히 하시고 하시는게 좋겠네요"),
        ],
        ROLES,
        TERMS,
    )

    assert found.lifestyle == ["허리 줄이기", "체중 줄이기", "운동"]


def test_a_word_outside_the_vocabulary_is_kept_as_heard():
    """의사가 무엇을 줄이라 했는지 버릴 수는 없다."""
    found = extract([Utterance("D", 0, 9_000, "탄산음료 줄이셔야 됩니다")], ROLES, TERMS)

    assert found.lifestyle == ["탄산음료 줄이기"]


def test_schedule_and_duration_come_out_as_values():
    """문장을 통째로 실으면 "하나하시면만 드시면 되니까" 가 따라온다."""
    found = extract(
        [
            Utterance("D", 0, 9_000, "그거를 매일 하루에 하나하시면만 드시면 되니까"),
            Utterance("D", 9_000, 15_000, "두 달 드셨었잖아요 두 달분 드렸으니까"),
        ],
        ROLES,
        TERMS,
    )

    assert found.schedule == ["매일", "하루에 하나"]
    assert found.duration == ["두 달분"]


def test_a_misheard_drug_is_recognised_after_correction():
    """"간보제"는 사전에 없다. 그것이 바로 교정이 잡아낸 오인식이다."""
    from voice_ai.terms import find_corrections

    utterances = [
        Utterance("D", 0, 9_000, "간 보호제 있잖아요 그거를 좀 드셔주시면"),
        Utterance("D", 9_000, 15_000, "저 간보제 드시던 거 있잖아요"),
    ]
    corrections = find_corrections(utterances, set(TERMS.all_terms))

    assert extract(utterances, ROLES, TERMS, corrections).drugs == ["간보호제"]


def test_measurements_carry_their_periods():
    found = extract(
        [
            Utterance("D", 0, 9_000, "간 수치가 좀 높죠 40이 정상이신데"),
            Utterance("D", 9_000, 17_000, "작년에는 76에 34였고 이번에는 67에 23이에요"),
        ],
        ROLES,
        TERMS,
    )

    carried = [m for m in found.measurements if m.inferred][0]
    assert carried.by_time == {"작년": ["76", "34"], "이번": ["67", "23"]}


def test_the_other_engine_supplies_a_period_word_this_one_lost():
    """SenseVoice 가 "정상"을 "정는"으로 흘렸고 moonshine 은 "정산"으로 들었다."""
    ours = [Utterance("D", 0, 9_000, "콜레스테롤이 200이 정는 216 이니까")]
    theirs = [Utterance("D", 0, 9_000, "폴레스테롤이 200이 정산이 216이니까")]

    alone = extract(ours, ROLES, TERMS).measurements[0]
    assert alone.by_time == {"시점없음": ["200", "216"]}
    assert alone.time_from_alternate is False

    borrowed = extract(ours, ROLES, TERMS, alternate=theirs).measurements[0]
    assert borrowed.by_time == {"정상": ["200"], "시점없음": ["216"]}
    assert borrowed.time_from_alternate is True


def test_the_quote_always_stays_from_our_own_transcript():
    """시점만 빌린다. 근거 문장까지 남의 것을 보여주면 검증이 어긋난다."""
    ours = [Utterance("D", 0, 9_000, "콜레스테롤이 200이 정는 216 이니까")]
    theirs = [Utterance("D", 0, 9_000, "폴레스테롤이 200이 정산이 216이니까")]

    assert extract(ours, ROLES, TERMS, alternate=theirs).measurements[0].quote == ours[0].text


def test_예약_날짜는_검사_수치가_아니다():
    # "10월 20일 오전 10시"가 10, 20, 10 세 개의 수치로 잡히던 자리다.
    found = extract(
        [
            Utterance("D", 0, 9_000, "간수치가 40이 정상인데 67이세요"),
            Utterance("D", 60_000, 66_000, "10월 20일날 오전 10시에 오세요. 피검사 다시 하죠."),
        ],
        ROLES,
        TERMS,
    )

    assert [(m.test, m.values) for m in found.measurements] == [("간수치", ["40", "67"])]


def test_한_문장에_날짜와_수치가_같이_있어도_수치만_남는다():
    found = extract(
        [Utterance("D", 0, 9_000, "간수치 67이니까 10월 20일에 다시 봅시다")], ROLES, TERMS
    )

    assert [(m.test, m.values) for m in found.measurements] == [("간수치", ["67"])]
