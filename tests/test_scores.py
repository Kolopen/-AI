"""점수로 말하는 검사.

검사 수치는 이름이 먼저 나와야 숫자를 받는데, 점수는 이름 없이 온다.
두 녹음에서 연속으로 놓쳤고 둘 다 보호자에게 꼭 전해야 할 숫자였다.
"""

from voice_ai import terminology
from voice_ai.models import Role, Utterance
from voice_ai.scores import GENERIC_NAME, PAIN_NAME, extract

TERMS = terminology.load("신경과")
ROLES = {"D": Role.DOCTOR, "P": Role.PATIENT, "M": Role.MANAGER}


def test_만점과_점수를_함께_뽑는다():
    found = extract(
        [Utterance("D", 58_000, 66_000, "지난 검사 점수는 30점 만점에 26점이었습니다")],
        ROLES,
        TERMS,
    )

    assert [(s.name, s.value, s.maximum, s.when) for s in found] == [
        (GENERIC_NAME, "26", "30", "저번")
    ]


def test_만점이_흘려도_받는다():
    # 실제 전사가 "만점"을 "만쯤"으로 흘렸다.
    found = extract(
        [Utterance("D", 0, 9_000, "지난 검사 점수는 30점 만쯤에 26점이었습니다")], ROLES, TERMS
    )

    assert [(s.value, s.maximum) for s in found] == [("26", "30")]


def test_눈금_설명은_측정값이_아니다():
    # "0점, 10점으로 해서 적으세요"를 올리면 없던 통증 기록이 남는다.
    found = extract(
        [
            Utterance(
                "D", 0, 9_000, "전혀 아프지 않은 상태를 0점, 가장 심한 통증을 10점으로 해서 적으세요"
            )
        ],
        ROLES,
        TERMS,
    )

    assert [s.value for s in found] == ["0"]


def test_검사_점수는_의사_말만_받는다():
    # 환자가 자기 점수를 잘못 기억하는 일이 실제로 있었다.
    found = extract(
        [Utterance("P", 0, 9_000, "검사 점수가 29점인 줄 알았어요")], ROLES, TERMS
    )

    assert found == []


def test_통증_점수는_환자_말을_받는다():
    # 아픈 정도의 주인은 환자 본인이다. 의사 말만 보면 영영 못 잡는다.
    found = extract(
        [
            Utterance("D", 0, 9_000, "통증 정도는 이번에 어떠셨어요?"),
            Utterance("P", 10_000, 14_000, "이번에는 4점이나 5점 정도요"),
        ],
        ROLES,
        TERMS,
    )

    assert [(s.name, s.value, s.when, s.inferred) for s in found] == [
        (PAIN_NAME, "4", "이번", True),
        (PAIN_NAME, "5", "이번", True),
    ]


def test_되풀이한_값은_새_측정이_아니다():
    found = extract(
        [
            Utterance("D", 0, 9_000, "지난 검사 점수는 30점 만점에 26점이었습니다"),
            Utterance("D", 12_000, 15_000, "26점입니다. 결과지에도 적어 드릴게요"),
        ],
        ROLES,
        TERMS,
    )

    assert len(found) == 1
    assert found[0].maximum == "30"


def test_이름에_붙은_숫자는_점수가_아니다():
    # "비타민 b12 점수"의 12 가 12점으로 올라가던 자리다.
    found = extract(
        [Utterance("D", 0, 9_000, "비타민 b12 점수 수치가 정상 범위였습니다")], ROLES, TERMS
    )

    assert found == []


def test_점수_문맥이_없으면_받지_않는다():
    found = extract([Utterance("D", 0, 9_000, "그 점 유의해 주세요")], ROLES, TERMS)

    assert found == []


def test_한글로_쓴_점수를_읽는다():
    # 한국어 파인튜닝 모델은 숫자를 한글로 쓴다. 전사가 제일 좋은 모델인데
    # 숫자를 못 읽으면 그 장점이 통째로 날아간다.
    from voice_ai.scores import to_digits

    assert to_digits("이십 육 점이었습니다") == "26점이었습니다"
    assert to_digits("삼십 점 만점") == "30점 만점"
    assert to_digits("통증은 칠 점 정도요") == "통증은 7점 정도요"


def test_단위가_없으면_읽지_않는다():
    # "검사"의 "사"가 4가 되고 "지금"의 "이"가 2가 된다. "점"이 붙었을 때만 읽는다.
    from voice_ai.scores import to_digits

    assert to_digits("검사 점수가 어떻게 되나요") == "검사 점수가 어떻게 되나요"
    assert to_digits("지난번 검사 결과입니다") == "지난번 검사 결과입니다"
    assert to_digits("오늘은 새 약을 처방하지 않겠습니다") == "오늘은 새 약을 처방하지 않겠습니다"


def test_낱말_속_글자는_수가_아니다():
    # "검사 점수"의 사가 4점이 되면 없는 점수가 생긴다.
    from voice_ai.scores import to_digits

    assert "4점" not in to_digits("지난 검사 점수는 어떤가요")


def test_한글_숫자_점수도_만점과_함께_뽑는다():
    found = extract(
        [Utterance("D", 58_000, 68_000, "지난 검사 점수는 삼십 점 만점에 이십 육 점이었습니다")],
        ROLES,
        TERMS,
    )

    assert [(s.value, s.maximum) for s in found] == [("26", "30")]
    # 근거 문장은 들린 그대로 남는다.
    assert "삼십 점" in found[0].quote


def test_점수라는_낱말은_단위가_아니다():
    # 전사가 "비타민 B12"를 "비타민 비시 이 점수치"로 흘렸고 그 "이"가 2점이 됐다.
    from voice_ai.scores import to_digits

    assert to_digits("비타민 비시 이 점수치가 정상범이었습니다") == "비타민 비시 이 점수치가 정상범이었습니다"

    neurology = terminology.load("신경과")
    found = extract(
        [Utterance("D", 0, 9_000, "지난번 검사에서 비타민 비시 이 점수치가 정상이었습니다")],
        ROLES,
        neurology,
    )
    assert found == []


def test_숫자_쪽에도_같은_덫이_있다():
    neurology = terminology.load("신경과")
    found = extract(
        [Utterance("D", 0, 9_000, "혈액검사에서 비12 점수 수치가 정상 범위였습니다")],
        ROLES,
        neurology,
    )

    assert found == []


def test_점을_쩜으로_흘려도_만점을_받는다():
    # 잡음을 걷어낸 실제 녹음이 "삼십 쩜 만쯤에 이십 육 점" 으로 나왔다.
    # 전보다 정확해진 전사인데 단위가 쩜이라 만점을 통째로 놓쳤다.
    found = extract(
        [Utterance("D", 58_000, 66_000, "지난검사 점수는 삼십 쩜 만쯤에 이십 육 점이었습니다")],
        ROLES,
        TERMS,
    )

    assert [(s.value, s.maximum) for s in found] == [("26", "30")]


def test_쩜으로_말한_검사_점수도_받는다():
    found = extract(
        [Utterance("D", 72_000, 75_000, "기억력검사 점수는 이십 육 쩜이었습니다")], ROLES, TERMS
    )

    assert [s.value for s in found] == ["26"]


def test_쩜으로_설명한_눈금은_측정값이_아니다():
    # "0쩜으로 해서 적어보세요"는 눈금 설명이다. 올리면 없던 기록이 남는다.
    assert not extract([Utterance("D", 0, 5_000, "0쩜으로 해서 적어보세요")], ROLES, TERMS)
