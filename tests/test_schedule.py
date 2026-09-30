"""후속 예약 날짜 추출."""

import datetime as dt

from voice_ai.models import Utterance
from voice_ai.schedule import find_next_visit, resolve_date, resolve_time

CONSULT = dt.date(2026, 9, 30)


def _pairs(*sentences: str) -> list[tuple[Utterance, str]]:
    return [
        (Utterance(speaker_tag="1", text=s, start_ms=i * 1000, end_ms=i * 1000 + 900), s)
        for i, s in enumerate(sentences)
    ]


def test_월일을_읽는다():
    assert resolve_date("10월 20일날 오세요", CONSULT) == dt.date(2026, 10, 20)


def test_진료일보다_앞선_달은_내년이다():
    # 9월 30일에 "1월 10일"이라고 하면 올해 1월일 수 없다.
    assert resolve_date("1월 10일에 오세요", CONSULT) == dt.date(2027, 1, 10)


def test_내년이라고_말하면_그대로_민다():
    assert resolve_date("내년 10월 20일에 보죠", CONSULT) == dt.date(2027, 10, 20)


def test_다음_달():
    assert resolve_date("다음 달 15일에 오세요", CONSULT) == dt.date(2026, 10, 15)


def test_이번_달():
    assert resolve_date("이번 달 30일에 오세요", CONSULT) == dt.date(2026, 9, 30)


def test_날짜만_말하면_지나지_않은_가장_가까운_날이다():
    assert resolve_date("5일에 오세요", CONSULT) == dt.date(2026, 10, 5)


def test_없는_날짜는_버린다():
    assert resolve_date("2월 30일에 오세요", CONSULT) is None


def test_기간은_날짜가_아니다():
    # "20일분"은 20일치 약이지 20일이 아니다.
    assert resolve_date("20일분 드릴게요", CONSULT) is None
    assert resolve_date("3일 뒤에 오세요", CONSULT) is None
    assert resolve_date("10일째 드시면", CONSULT) is None


def test_개월은_월이_아니다():
    assert resolve_date("3개월 뒤에 오세요", CONSULT) is None


def test_시각을_읽는다():
    assert resolve_time("오전 10시에 오세요") == (10, 0)
    assert resolve_time("오후 2시 30분에 오세요") == (14, 30)
    assert resolve_time("오전 12시") == (0, 0)
    assert resolve_time("오후 12시") == (12, 0)


def test_시간은_시각이_아니다():
    assert resolve_time("3시간 뒤에 오세요") is None


def test_오전_오후를_안_말하면_진료시간_밖은_버린다():
    # "2시"는 오후 2시겠지만 말한 그대로는 새벽 2시다. 지어내지 않는다.
    assert resolve_time("2시에 오세요") is None
    assert resolve_time("10시에 오세요") == (10, 0)


def test_지난_이야기는_예약이_아니다():
    pairs = _pairs("작년 10월 20일에 오셨을 때는 괜찮았어요")
    assert find_next_visit(pairs, consult_date=CONSULT) is None


def test_방문_표현이_없으면_예약이_아니다():
    pairs = _pairs("10월 20일에 결과가 나옵니다")
    assert find_next_visit(pairs, consult_date=CONSULT) is None


def test_방문_표현은_같은_발화_안에서_찾는다():
    utterance = Utterance(
        speaker_tag="1",
        text="그럼 10월 20일이요. 그때 오시면 됩니다.",
        start_ms=0,
        end_ms=3000,
    )
    pairs = [(utterance, "그럼 10월 20일이요."), (utterance, "그때 오시면 됩니다.")]
    visit = find_next_visit(pairs, consult_date=CONSULT)
    assert visit is not None
    assert visit.day == dt.date(2026, 10, 20)


def test_말을_고치면_마지막_것을_쓴다():
    pairs = _pairs("10월 20일에 오세요", "아 21일로 하죠 그날 오세요")
    visit = find_next_visit(pairs, consult_date=CONSULT)
    assert visit is not None
    assert visit.day == dt.date(2026, 10, 21)


def test_iso_는_시각이_없으면_날짜만_준다():
    pairs = _pairs("10월 20일날 오세요")
    visit = find_next_visit(pairs, consult_date=CONSULT)
    assert visit is not None
    assert visit.iso == "2026-10-20"


def test_iso_는_시각을_말하면_함께_준다():
    pairs = _pairs("10월 20일 오후 3시 30분에 오세요")
    visit = find_next_visit(pairs, consult_date=CONSULT)
    assert visit is not None
    assert visit.iso == "2026-10-20T15:30"


def test_한글_숫자는_읽지_않는다():
    # "이십일"은 21일도 되고 20일도 된다. 틀린 날짜보다 없는 편이 낫다.
    assert resolve_date("시월 이십일에 오세요", CONSULT) is None
