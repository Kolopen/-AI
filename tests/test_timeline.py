"""수치를 시점별로 가르기.

"간수치 76, 34, 67, 23, 40" 은 사람이 읽을 수 없다. 보호자가 알고 싶은 것은
좋아졌는지 나빠졌는지이고, 그건 시점이 붙어야 보인다.
"""

from voice_ai.timeline import group_by_time, has_marker


def test_reference_value_comes_before_its_word():
    """한국어는 "40이 정상" 처럼 기준값이 앞에 온다. 시점은 뒤에 온다."""
    assert group_by_time("40이 정상이신데") == {"정상": ["40"]}


def test_numbers_follow_their_time_marker():
    grouped = group_by_time("작년에는 76에 34였고 이번에는 67에 23이에요")

    assert grouped == {"작년": ["76", "34"], "이번": ["67", "23"]}


def test_a_trailing_threshold_does_not_join_the_last_period():
    """기준값을 먼저 떼지 않으면 "40 이상"의 40이 직전 시점으로 딸려간다.

    실제 녹음에서 이번(67, 23)에 40이 섞여 들어갔었다.
    """
    grouped = group_by_time("이번에는 67 회 23 마가 조금씩 높요 40 이상 보다 높 으니까.")

    assert grouped == {"정상": ["40"], "이번": ["67", "23"]}


def test_a_broken_transcript_gives_the_same_grouping():
    """어미가 다 무너져도 시점 단어는 살아남는다."""
    clean = "40이 정상이신데 작년에는 76에 34였고 이번에는 67에 23 마찬가지로 높아요."
    broken = "작년는 76 의 34였고 이번에는 67 회 23 마가 조금씩 높요 40 이상 보다 높 으니까."

    assert group_by_time(clean) == group_by_time(broken)


def test_a_misheard_reference_word_is_still_recognised():
    """moonshine 이 "정상"을 "정산"으로 들었다. 0.833 이면 같은 말로 본다."""
    assert group_by_time("200이 정산이 216이니까") == {"정상": ["200"], "시점없음": ["216"]}


def test_a_different_word_is_not_mistaken_for_the_reference():
    """"정도"는 0.545 다. 기준을 낮추면 이것부터 걸린다."""
    assert group_by_time("혈압이 130 정도 나왔어요") == {"시점없음": ["130"]}


def test_numbers_without_any_marker_stay_unmarked():
    assert group_by_time("혈압은 130에 80입니다") == {"시점없음": ["130", "80"]}


def test_has_marker_decides_whether_to_borrow_from_the_other_engine():
    assert has_marker("작년에는 76이었고")
    assert has_marker("200이 정산이 216")
    assert not has_marker("콜레스테롤이 200이 정는 216 이니까")
