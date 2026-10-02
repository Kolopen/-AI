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
