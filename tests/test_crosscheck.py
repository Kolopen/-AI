"""두 엔진 전사 맞대기.

엔진마다 다른 자리에서 틀린다. 실제 녹음에서 SenseVoice 는 "신장"을 맞추고
moonshine 이 "심장"으로 썼다. 한쪽만 보면 알 수 없다.
"""

from voice_ai import terminology
from voice_ai.crosscheck import cross_check
from voice_ai.models import Utterance

TERMS = terminology.load("내과").all_terms


def test_numbers_that_agree_are_not_reported():
    """두 엔진이 독립적으로 같은 숫자를 냈으면 믿을 만하다.

    조용한 것이 곧 검증 결과다. 실제 녹음에서 200과 216이 양쪽 모두 같았다.
    """
    moonshine = [Utterance("D", 0, 9_000, "콜레스테롤이 200이 정산이 216이니까.")]
    sensevoice = [Utterance("D", 0, 9_000, "콜레스테롤이 200이 정는 216 이니까,")]

    assert [d for d in cross_check(moonshine, sensevoice, TERMS) if d.kind == "NUMBER"] == []


def test_numbers_that_differ_are_reported():
    """검사 수치는 틀려도 그럴듯해 보인다. 엇갈리면 사람이 들어야 한다."""
    moonshine = [Utterance("D", 0, 9_000, "간수치가 42인데요.")]
    sensevoice = [Utterance("D", 0, 9_000, "간수치가 40인데요.")]

    numbers = [d for d in cross_check(moonshine, sensevoice, TERMS) if d.kind == "NUMBER"]

    assert len(numbers) == 1
    assert numbers[0].primary == "42"
    assert numbers[0].secondary == "40"


def test_a_term_only_one_engine_heard_is_reported():
    """신장과 심장은 둘 다 실재하는 말이라 발음으로는 못 거른다."""
    moonshine = [Utterance("D", 0, 9_000, "피검사에서는 심장이라든지 빈혈은 괜찮으시고,")]
    sensevoice = [Utterance("D", 0, 9_000, "피검사에서는 신장 이라든지 빈혈은 괜찮으시고")]

    terms = cross_check(moonshine, sensevoice, TERMS)

    assert ("심장", "") in [(d.primary, d.secondary) for d in terms]
    assert ("", "신장") in [(d.primary, d.secondary) for d in terms]


def test_segments_are_matched_by_time_not_by_order():
    """구간 길이 상한을 다르게 준 전사끼리도 맞댈 수 있어야 한다."""
    moonshine = [Utterance("D", 0, 10_000, "간수치가 42입니다.")]
    sensevoice = [
        Utterance("D", 0, 5_000, "간수치가"),
        Utterance("D", 5_000, 10_000, "42입니다."),
    ]

    assert [d for d in cross_check(moonshine, sensevoice, TERMS) if d.kind == "NUMBER"] == []


def test_a_segment_the_other_engine_dropped_is_skipped():
    """상대가 빈 결과를 낸 구간까지 엇갈림으로 셀 수는 없다."""
    moonshine = [Utterance("D", 0, 9_000, "간수치가 42입니다.")]
    sensevoice: list[Utterance] = []

    assert cross_check(moonshine, sensevoice, TERMS) == []


def test_one_long_segment_does_not_repeat_the_same_finding():
    """엔진마다 구간을 다르게 나눈다.

    한쪽의 긴 구간이 상대의 여러 구간과 겹친다고 해서 같은 발견을 그만큼
    반복하면 안 된다. 실제로 '지방간' 하나가 11번 나왔다.
    """
    moonshine = [Utterance("D", i * 8_000, i * 8_000 + 8_000, "간 수치가 좀 높죠.") for i in range(11)]
    sensevoice = [Utterance("D", 0, 88_000, "지방간 때문에 간수치가 좀 높으세요.")]

    found = cross_check(moonshine, sensevoice, TERMS)

    assert [d.secondary for d in found if d.kind == "TERM"].count("지방간") == 1


def test_only_the_numbers_that_differ_are_named():
    """목록 전체를 다시 보여주면 어느 것이 문제인지 알 수 없다."""
    moonshine = [Utterance("D", 0, 9_000, "간수치가 76에 34이고 요산은 7입니다.")]
    sensevoice = [Utterance("D", 0, 9_000, "간수치가 76에 34이고 요산은 9입니다.")]

    numbers = [d for d in cross_check(moonshine, sensevoice, TERMS) if d.kind == "NUMBER"]

    assert len(numbers) == 1
    assert numbers[0].primary == "7"
    assert numbers[0].secondary == "9"


def test_a_repeated_number_is_counted_twice():
    """40이 두 번 나왔는데 한 번만 들었다면 그것도 엇갈림이다."""
    moonshine = [Utterance("D", 0, 9_000, "40에서 40으로 그대로입니다.")]
    sensevoice = [Utterance("D", 0, 9_000, "40 그대로입니다.")]

    numbers = [d for d in cross_check(moonshine, sensevoice, TERMS) if d.kind == "NUMBER"]

    assert numbers[0].primary == "40"
    assert numbers[0].secondary == ""
