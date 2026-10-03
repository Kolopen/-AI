"""전사 결과 하나를 받아 리포트 초안까지 돌리는 진입점.

화자분리가 없는 전사(SenseVoice)는 문장 단위로 역할을 가르고,
화자가 붙은 전사(CLOVA, 클로바노트)는 화자 단위로 가른다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from . import clovanote, questions, sensevoice, terminology
from .clova import group_by_speaker
from .models import (
    AnalysisResult,
    AskedState,
    Method,
    RegisteredQuestion,
    ReportDraft,
    Role,
    SpeakerRole,
    Utterance,
)
from .qa import explain_qa, pair_qa
from .report import build_report_draft
from .roles import classify, sentence_role, split_sentences
from .crosscheck import cross_check
from .facts import extract as extract_facts
from .operations import summarize
from .turns import group_turns
from .terms import apply_corrections, find_confusions, find_corrections




def _split_into_sentences(utterances: list[Utterance]) -> list[Utterance]:
    """발화를 문장으로 쪼갠다. 시각은 원래 발화의 것을 물려받는다."""
    out: list[Utterance] = []
    for utterance in utterances:
        for sentence in split_sentences(utterance.text):
            out.append(
                Utterance(
                    speaker_tag=utterance.speaker_tag,
                    start_ms=utterance.start_ms,
                    end_ms=utterance.end_ms,
                    text=sentence,
                )
            )
    return out


def analyze_without_speakers(
    utterances: list[Utterance],
    *,
    terms: terminology.Terminology,
    consult_date: dt.date | None = None,
    questions: dict[str, str] | None = None,
) -> AnalysisResult:
    """화자 라벨이 없는 전사. 문장별로 역할을 가른다."""
    sentences = _split_into_sentences(utterances)
    labelled: list[Utterance] = []
    confidence_by_role: dict[Role, list[float]] = {}

    for sentence in sentences:
        role, confidence = sentence_role(sentence.text, medical_terms=terms.all_terms)
        labelled.append(
            Utterance(
                speaker_tag=role.value,
                start_ms=sentence.start_ms,
                end_ms=sentence.end_ms,
                text=sentence.text,
            )
        )
        confidence_by_role.setdefault(role, []).append(confidence)

    speakers = [
        SpeakerRole(
            speaker_tag=role.value,
            role=role,
            confidence=round(sum(scores) / len(scores), 2),
            method=Method.TEXT_PATTERN,
            needs_review=role is Role.UNKNOWN,
        )
        for role, scores in confidence_by_role.items()
    ]

    roles = {r.speaker_tag: r.role for r in speakers}
    pairs, registered = pair_qa(labelled, roles, registered_questions=questions)
    unknown = len(confidence_by_role.get(Role.UNKNOWN, []))
    warnings = []
    if unknown:
        warnings.append(
            f"{unknown}개 문장은 역할을 가르지 못했습니다. 대부분 짧은 맞장구이며 "
            "리포트에 담을 내용은 없습니다."
        )

    return AnalysisResult(
        speakers=speakers,
        qa_pairs=pairs,
        registered=registered,
        report_draft=build_report_draft(
            labelled, roles, drug_terms=terms.drugs, test_terms=terms.tests, consult_date=consult_date
        ),
        warnings=warnings,
        labelled=labelled,
    )


# 전체 글자 수에서 이 몫을 넘겨야 말한 화자로 센다. 길이로 끊지 않는 이유는
# 녹음 길이가 제각각이어서다. 3분짜리의 50자와 30분짜리의 50자는 뜻이 다르다.
MIN_SPEAKER_SHARE = 0.1


def _diarization_collapsed(utterances: list[Utterance]) -> bool:
    """화자분리가 사실상 한 명만 내놨는지 본다.

    한 명뿐이면 화자 단위 판정은 그 한 명에게 역할 하나를 붙이고 끝난다.
    모든 발화가 같은 역할을 받으므로 의사와 환자를 가르지 못한다. 실제
    3인 녹음에서 둘째 화자가 22자뿐이었고, 그 화자는 판정 불가로 끝났다.
    """
    spoken: dict[str, int] = {}
    for utterance in utterances:
        spoken[utterance.speaker_tag] = spoken.get(utterance.speaker_tag, 0) + len(utterance.text)
    total = sum(spoken.values())
    if not total:
        return True
    return sum(1 for length in spoken.values() if length / total >= MIN_SPEAKER_SHARE) <= 1


# 이보다 많은 비의사 화자가 판정 불가로 남으면 화자분리가 한 사람을 쪼갠 쪽을 의심한다.
_SPLIT_SUSPICION = 2

# 띄어쓰기 없이 이어진 한글이 이보다 길면 엔진이 띄어쓰기를 안 내놓은 것으로 본다.
# 한국어 어절은 열 자를 넘는 일이 드물다.
_UNSPACED_RUN = 25
_UNSPACED = re.compile(rf"[가-힣]{{{_UNSPACED_RUN},}}")


def _spacing_warning(utterances: list[Utterance]) -> str | None:
    """띄어쓰기 없는 전사는 뒤 단계를 조용히 무너뜨린다.

    용어 교정은 어절을 후보로 삼으므로 통째로 붙은 글에서는 한 건도 못 찾고,
    리포트는 문장을 못 갈라 한 문단을 여러 항목에 그대로 집어넣는다. 결과가
    비는 것이 아니라 그럴듯하게 틀리기 때문에 알려야 한다.
    """
    longest = max((len(m.group()) for u in utterances for m in _UNSPACED.finditer(u.text)), default=0)
    if longest < _UNSPACED_RUN:
        return None
    return (
        f"전사에 띄어쓰기가 없습니다(붙어 있는 한글 최대 {longest}자). "
        "용어 교정과 문장 분리가 동작하지 않아 리포트가 부정확해집니다. "
        "띄어쓰기를 내놓는 엔진(sensevoice)을 쓰는 편이 낫습니다."
    )


def _confusion_warnings(
    utterances: list[Utterance], terms: terminology.Terminology
) -> list[str]:
    """실재하는 두 용어가 서로 바뀐 것으로 보이면 알린다. 고치지는 않는다."""
    return [
        f"[{c.start_ms // 60000:02d}:{c.start_ms // 1000 % 60:02d}] '{c.written}'이(가) "
        f"'{c.suspected}'일 수 있습니다. 주변에 '{c.cue}'이(가) 나옵니다. "
        "둘 다 실재하는 말이라 자동으로 고치지 않습니다. 녹음을 확인하세요."
        for c in find_confusions(utterances, terms.confusable)
    ]


def _crosscheck_warnings(
    utterances: list[Utterance],
    other: list[Utterance],
    terms: terminology.Terminology,
) -> list[str]:
    """다른 엔진의 전사와 맞대어 어긋난 자리를 알린다."""
    lines = []
    for d in cross_check(utterances, other, terms.all_terms):
        stamp = f"[{d.start_ms // 60000:02d}:{d.start_ms // 1000 % 60:02d}]"
        if d.kind == "NUMBER":
            parts = []
            if d.primary:
                parts.append(f"'{d.primary}'은(는) 이쪽에만")
            if d.secondary:
                parts.append(f"'{d.secondary}'은(는) 다른 엔진에만")
            lines.append(
                f"{stamp} 숫자가 엇갈립니다. {', '.join(parts)} 있습니다. "
                "검사 수치라면 반드시 확인하세요."
            )
        elif d.primary:
            lines.append(f"{stamp} '{d.primary}'은(는) 다른 엔진에 없습니다.")
        else:
            lines.append(f"{stamp} 다른 엔진은 '{d.secondary}'이라고 들었습니다.")
    return lines


def analyze_with_speakers(
    utterances: list[Utterance],
    *,
    terms: terminology.Terminology,
    merge_non_doctor: bool = False,
    compare_with: list[Utterance] | None = None,
    consult_date: dt.date | None = None,
    manager_speaker_tag: str | None = None,
    questions: dict[str, str] | None = None,
) -> AnalysisResult:
    """화자 라벨이 있는 전사. 화자 단위로 역할을 가른다."""
    speakers, warnings = classify(
        group_by_speaker(utterances),
        merge_non_doctor=merge_non_doctor,
        manager_speaker_tag=manager_speaker_tag,
    )
    spacing = _spacing_warning(utterances)
    if spacing:
        warnings.append(spacing)
    warnings.extend(_confusion_warnings(utterances, terms))
    if compare_with:
        warnings.extend(_crosscheck_warnings(utterances, compare_with, terms))
    roles = {s.speaker_tag: s.role for s in speakers}

    # 판정 불가가 쌓이면 Q&A가 통째로 비게 된다. 질문자를 특정하지 못하기 때문이다.
    unresolved = [s.speaker_tag for s in speakers if s.role is Role.UNKNOWN]
    if not merge_non_doctor and len(unresolved) >= _SPLIT_SUSPICION:
        warnings.append(
            f"비의사 화자 {len(unresolved)}명을 판정하지 못했습니다. 한 사람이 쪼개져 "
            "나온 것이라면 --merge-non-doctor 로 합쳐서 다시 분석하세요. "
            "녹음 단계에서 --speakers 로 인원을 지정하는 편이 더 낫습니다."
        )
    pairs, registered = pair_qa(utterances, roles, registered_questions=questions)
    return AnalysisResult(
        speakers=speakers,
        qa_pairs=pairs,
        registered=registered,
        report_draft=build_report_draft(
            utterances, roles, drug_terms=terms.drugs, test_terms=terms.tests, consult_date=consult_date
        ),
        warnings=warnings,
        labelled=utterances,
    )


def load(path: Path) -> tuple[list[Utterance], list[str]]:
    if path.suffix == ".txt":
        return clovanote.parse(path.read_text(encoding="utf-8")), []

    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload if isinstance(payload, list) else payload.get("chunks", [])
    return sensevoice.parse_chunks(chunks)


def manager_tag(path: Path) -> str | None:
    """전사에 성문으로 확정된 매니저가 있으면 그 화자 태그를 돌려준다.

    `voice-transcribe --manager` 가 붙인 표시다. 없으면 None 이고, 그때는
    역할 판정이 지금까지처럼 어휘로만 돈다.
    """
    if path.suffix == ".txt":
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload if isinstance(payload, list) else payload.get("chunks", [])
    tags = {c.get("speaker") for c in chunks if c.get("is_manager")}
    return next(iter(tags)) if len(tags) == 1 else None


# 리포트 줄 앞의 "[01:18]". 역할을 볼 때는 떼고 본다.
_TIMESTAMP = re.compile(r"^\s*\[\d\d:\d\d\]\s*")


def draft_sections(draft: ReportDraft) -> list[tuple[str, str]]:
    """리포트 초안에서 내용이 있는 항목만 순서대로."""
    sections = [
        ("진료 내용", draft.treatment_notes),
        ("약품", draft.medication_name),
        ("복용 방법", draft.medication_schedule_note),
        ("처방 기간", draft.medication_notes),
        ("다음 방문", draft.next_visit_note),
    ]
    if draft.next_visit_at:
        sections.append(("후속 예약", f"{draft.next_visit_at}  (매니저 확인 후 예약)"))
    return [(label, body) for label, body in sections if body]


def doubtful(line: str, medical_terms: frozenset[str]) -> bool:
    """화자는 의사라는데 문장만 보면 의사 말투가 아닌지.

    걸러내지 않고 표시만 한다. 걸러 보니 진짜 의사 문장 열 개 중 넷이 같이
    날아갔다. "오늘은 MRI 를 예약하고 결과를 보고 설명드리겠습니다" 처럼
    진료의 결론이 빠지는 자리였다.

    반대로 섞여 들어온 쪽은 잘 걸린다. 화자분리가 무너진 녹음에서 환자와
    매니저 말 넷이 모두 여기 잡혔다. 의사 신호는 처방·진단 같은 말인데
    비의사의 서술문에는 그런 말이 없어서다.
    """
    role, _ = sentence_role(_TIMESTAMP.sub("", line), medical_terms=medical_terms)
    return role is not Role.DOCTOR


def _print_draft(
    title: str, draft: ReportDraft, corrections: list, medical_terms: frozenset[str]
) -> int:
    print(f"\n════ {title} ════")
    filled = draft_sections(draft)
    if not filled:
        print("  (비어 있습니다)")
        return 0

    marked = 0
    for label, body in filled:
        print(f"  [{label}]")
        for line in body.splitlines():
            flag = " "
            if doubtful(line, medical_terms):
                flag, marked = "?", marked + 1
            print(f"  {flag} {apply_corrections(line, corrections)}")
    print("    [요약] 매니저가 작성합니다.")
    return marked


def print_both_drafts(
    with_speakers: ReportDraft,
    without_speakers: ReportDraft,
    corrections: list,
    medical_terms: frozenset[str] = frozenset(),
) -> None:
    """같은 전사를 두 방식으로 돌린 리포트를 나란히 놓는다.

    화자를 덩어리로 묶는 쪽은 의사 발언을 놓치지 않는 대신 남의 말을 섞고,
    문장마다 가르는 쪽은 섞지 않는 대신 의사 발언을 흘린다. 어느 쪽이
    나은지는 녹음마다 다르므로 눈으로 보고 고르게 한다.
    """
    left_marked = _print_draft("화자 구분함", with_speakers, corrections, medical_terms)
    right_marked = _print_draft("화자 구분 안 함", without_speakers, corrections, medical_terms)
    if left_marked or right_marked:
        print(
            "\n  ? 는 문장만 보면 의사 말투가 아닌 줄입니다. 화자분리가 환자나 매니저 말을"
            "\n    섞어 넣었을 수 있으니 매니저가 확인해야 합니다. 의사 말인데 표시되는"
            f"\n    경우도 있습니다. 화자 구분함 {left_marked}줄, 구분 안 함 {right_marked}줄."
        )

    left = dict(draft_sections(with_speakers))
    right = dict(draft_sections(without_speakers))
    print("\n════ 차이 ════")
    same = True
    for label in sorted(set(left) | set(right)):
        here, there = left.get(label, ""), right.get(label, "")
        if here == there:
            continue
        same = False
        print(f"  [{label}]")
        for line in sorted(set(here.splitlines()) - set(there.splitlines())):
            print(f"    화자 구분함에만    {line}")
        for line in sorted(set(there.splitlines()) - set(here.splitlines())):
            print(f"    구분 안 함에만     {line}")
    if same:
        print("  두 리포트가 같습니다.")


def _print_registered(registered: list[RegisteredQuestion]) -> None:
    """보호자 질문마다 의사가 뭐라고 답했는지."""
    label = {
        AskedState.CONFIRMED: "물어봄  ",
        AskedState.LIKELY: "아마 물어봄",
        AskedState.UNCONFIRMED: "못 찾음 ",
    }
    confirmed = sum(1 for r in registered if r.state is AskedState.CONFIRMED)
    print(f"\n보호자 질문 ({len(registered)}건 중 {confirmed}건 확인)")

    for item in registered:
        print(f"  {label[item.state]}  [{item.question_id}] {item.text}")
        if item.pair is None:
            print("      녹음에서 이 질문을 찾지 못했습니다. 매니저가 확인해 주세요.")
            continue
        stamp = f"{item.pair.question_at_ms // 60000:02d}:{item.pair.question_at_ms // 1000 % 60:02d}"
        print(f"      물음  [{stamp}] ({item.pair.asked_by.value}) {item.pair.question}")
        print(f"      답변  {item.pair.answer}")
        if item.state is AskedState.LIKELY:
            print(f"      일치도 {item.score} 라 확정하지 않았습니다. 매니저가 확인해 주세요.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="진료 전사를 분석해 리포트 초안을 만든다.")
    parser.add_argument("transcript", type=Path, help="클로바노트 .txt 또는 SenseVoice .json")
    parser.add_argument("--json", action="store_true", help="사람이 읽는 표 대신 JSON으로 출력")
    parser.add_argument(
        "--full",
        action="store_true",
        help="발췌 대신 전체 대화를 화자와 함께 보여준다. 교정은 원문을 남긴 채 반영한다.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="다른 엔진으로 만든 전사. 숫자와 용어가 엇갈리는 자리를 표시한다.",
    )
    parser.add_argument(
        "--merge-non-doctor",
        action="store_true",
        help="비의사 화자를 한 사람으로 합쳐 판정한다. 화자분리가 한 사람을 여러 명으로 쪼갰을 때 쓴다.",
    )
    parser.add_argument(
        "--sentences",
        action="store_true",
        help="화자 라벨을 무시하고 문장 단위로 역할을 가른다. 목소리가 하나뿐인 녹음에 쓴다.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="화자를 구분했을 때와 안 했을 때의 리포트를 나란히 보여준다.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help="보호자가 미리 남긴 질문 파일. 줄마다 하나씩, 'id = 질문' 형식도 받는다.",
    )
    parser.add_argument(
        "--why-qa",
        action="store_true",
        help="질문-답변이 안 붙는 이유를 발화별로 보여준다.",
    )
    parser.add_argument(
        "--date",
        type=dt.date.fromisoformat,
        default=dt.date.today(),
        help="진료를 본 날 (YYYY-MM-DD). 의사가 말한 \"10월 20일\"의 연도를 이걸로 정한다.",
    )
    parser.add_argument(
        "--department",
        help=f"진료과 용어 사전. 있는 것: {', '.join(terminology.available())}",
    )
    args = parser.parse_args(argv)

    terms = terminology.load(args.department)
    asked = questions.load(args.questions) if args.questions else {}
    utterances, warnings = load(args.transcript)
    if not utterances:
        print("전사 내용이 비어 있습니다.", file=sys.stderr)
        return 1

    compare = load(args.compare)[0] if args.compare else None
    has_speakers = any(u.speaker_tag for u in utterances) and not args.sentences
    if has_speakers and _diarization_collapsed(utterances):
        # 화자분리가 사실상 한 명만 내놨다. 화자 단위로 가르면 모든 발화가
        # 같은 역할을 받으므로 판정이 아니라 복사다. 문장 단위로 내려간다.
        has_speakers = False
        warnings.append(
            "화자분리가 사실상 한 사람만 내놓아 문장 단위 판정으로 내려갔습니다. "
            "녹음에 목소리가 하나뿐이거나 화자분리가 실패한 것입니다."
        )
    if args.both:
        # 붕괴 판정으로 내려간 경우에도 둘 다 보여준다. 오히려 그때가 두
        # 리포트가 가장 크게 갈리는 자리라 눈으로 봐야 한다.
        if not any(u.speaker_tag for u in utterances):
            print(
                "이 전사에는 화자 라벨이 없어 비교할 것이 없습니다. "
                "--segmentation 으로 전사해야 합니다.",
                file=sys.stderr,
            )
            return 1
        for line in warnings:
            print(f"  ! {line}")
        grouped = analyze_with_speakers(
            utterances,
            terms=terms,
            merge_non_doctor=args.merge_non_doctor,
            consult_date=args.date,
            manager_speaker_tag=manager_tag(args.transcript),
            questions=asked,
        )
        split = analyze_without_speakers(
            utterances, terms=terms, consult_date=args.date, questions=asked
        )
        print_both_drafts(
            grouped.report_draft,
            split.report_draft,
            find_corrections(utterances, set(terms.all_terms), terms.misheard),
            frozenset(terms.conditions) | frozenset(terms.tests) | frozenset(terms.drugs),
        )
        return 0

    result = (
        analyze_with_speakers(
            utterances,
            terms=terms,
            merge_non_doctor=args.merge_non_doctor,
            compare_with=compare,
            consult_date=args.date,
            manager_speaker_tag=manager_tag(args.transcript),
            questions=asked,
        )
        if has_speakers
        else analyze_without_speakers(
            utterances, terms=terms, consult_date=args.date, questions=asked
        )
    )
    result.warnings = warnings + result.warnings

    # 역할이 붙은 발화로 넘어간다. 문장 단위로 내려간 경우 입력 발화의
    # 화자 태그는 역할과 짝이 맞지 않아 뒤 단계가 전부 비어서 나온다.
    labelled = result.labelled or utterances
    corrections = find_corrections(utterances, set(terms.all_terms), terms.misheard)

    roles_by_tag = {sp.speaker_tag: sp.role for sp in result.speakers}
    facts = extract_facts(
        labelled, roles_by_tag, terms, corrections, alternate=compare
    )

    if args.json:
        # 매니저가 보는 것과 관리자가 보는 것을 나눠 담는다. 경고는 진료실에서
        # 할 수 있는 일이 없으므로 매니저 쪽에 넣지 않는다.
        operations = summarize(result.warnings, term_corrections=len(corrections))
        print(json.dumps({
            "speakers": [asdict(s) for s in result.speakers],
            "qa_pairs": [asdict(p) for p in result.qa_pairs],
            "facts": asdict(facts),
            "report_draft": asdict(result.report_draft),
            "term_corrections": [asdict(c) for c in corrections],
            "operations": asdict(operations),
        }, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"화자 판정 ({'화자 라벨 있음' if has_speakers else '문장 단위'})")
    for speaker in result.speakers:
        flag = "  [검수 필요]" if speaker.needs_review else ""
        print(f"  {speaker.speaker_tag:12} {speaker.role.value:8} 신뢰도 {speaker.confidence}{flag}")

    if args.full:
        roles = {s.speaker_tag: s.role.value for s in result.speakers}
        turns = group_turns(labelled)
        print(f"\n전체 대화 ({len(turns)}발언 / {len(labelled)}구간)")
        for turn in turns:
            stamp = f"{turn.start_ms // 60000:02d}:{turn.start_ms // 1000 % 60:02d}"
            role = roles.get(turn.speaker_tag, "UNKNOWN")
            print(f"\n  {role} {stamp}")
            for line in turn.text.split("\n"):
                print(f"    {apply_corrections(line, corrections)}")
        print()

    if args.why_qa:
        mode, rows = explain_qa(labelled, roles_by_tag)
        print(f"\n질문 판정 ({mode} 방식 · {len(rows)}구간)")
        for start_ms, role, verdict, text in rows:
            stamp = f"{start_ms // 60000:02d}:{start_ms // 1000 % 60:02d}"
            print(f"  [{stamp}] {role:<8} {verdict}")
            print(f"           {text[:70]}")

    if result.registered:
        _print_registered(result.registered)

    print(f"\n현장에서 나온 질문 ({len(result.qa_pairs)}건)")
    for pair in result.qa_pairs:
        print(f"  [{pair.question_at_ms // 1000}초] ({pair.asked_by.value}) {pair.question}")
        print(f"        -> {pair.answer[:80]}")

    if any([facts.measurements, facts.scores, facts.diagnoses, facts.normal,
            facts.unconfirmed, facts.drugs, facts.schedule, facts.duration,
            facts.lifestyle]):
        print("\n핵심 내용")
        if facts.measurements:
            print("  [검사 수치]")
            for m in facts.measurements:
                stamp = f"{m.start_ms // 60000:02d}:{m.start_ms // 1000 % 60:02d}"
                marks = []
                if m.inferred:
                    marks.append("검사명 추정")
                if m.time_from_alternate:
                    marks.append("시점은 다른 엔진")
                suffix = f"  ({', '.join(marks)})" if marks else ""
                print(f"    {m.test}{suffix}")
                for when, values in m.by_time.items():
                    print(f"    {'':4} {when:6} {' / '.join(values)}")
                print(f"    {'':4} └ [{stamp}] {apply_corrections(m.quote, corrections)}")
        if facts.scores:
            print("  [점수]")
            # 한 문장에서 같은 시점으로 나온 점수는 한 줄로 모은다. 전사가
            # "만점"을 흘리면("30점 안에 26점") 눈금과 값이 따로 떨어져 측정이
            # 두 번 있었던 것처럼 보인다. 어느 쪽이 만점인지 모를 때는 둘 다
            # 보여주고 판단은 근거 문장에 맡긴다.
            grouped: dict[tuple, list] = {}
            for sc in facts.scores:
                grouped.setdefault((sc.name, sc.inferred, sc.when, sc.start_ms), []).append(sc)
            for (name, inferred, when, start_ms), group in grouped.items():
                stamp = f"{start_ms // 60000:02d}:{start_ms // 1000 % 60:02d}"
                values = " / ".join(
                    f"{sc.value}점" + (f" ({sc.maximum}점 만점)" if sc.maximum else "")
                    for sc in group
                )
                mark = "  (검사명 추정)" if inferred else ""
                print(f"    {name}{mark}")
                print(f"    {'':4} {when:6} {values}")
                print(f"    {'':4} └ [{stamp}] {apply_corrections(group[0].quote, corrections)}")
        if facts.diagnoses:
            print("  [진단·소견]")
            print(f"    {', '.join(name for name, _ in facts.diagnoses)}")
        if facts.normal:
            print("  [이상 없다고 한 항목]")
            print(f"    {', '.join(facts.normal)}")
        if facts.unconfirmed:
            print("  [아직 아니라고 한 항목]")
            print(f"    {', '.join(facts.unconfirmed)}")
        if facts.drugs or facts.schedule or facts.duration:
            print("  [복용]")
            row = [", ".join(facts.drugs) or "약품 미확인"]
            if facts.schedule:
                row.append(" / ".join(facts.schedule))
            if facts.duration:
                row.append(" / ".join(facts.duration))
            print(f"    {'   '.join(row)}")
        if facts.lifestyle:
            print("  [생활 지도]")
            print(f"    {', '.join(facts.lifestyle)}")

    filled = draft_sections(result.report_draft)
    if filled:
        print("\n리포트 초안")
        for label, body in filled:
            print(f"  [{label}]")
            for line in body.splitlines():
                print(f"    {apply_corrections(line, corrections)}")
        print("  [요약] 매니저가 작성합니다.")

    if corrections:
        print(f"\n용어 교정 후보 ({len(corrections)}건)")
        for c in corrections:
            print(f"  {c.original!r} -> {c.corrected!r}  유사도 {c.similarity}  근거 {c.evidence}")

    if result.warnings:
        operations = summarize(result.warnings, term_corrections=len(corrections))
        print("\n운영 점검 (관리자용 · 매니저 화면에는 내보내지 않는다)")
        print("  " + "  ".join(f"{k} {v}" for k, v in sorted(operations.counts.items())))
        for warning in result.warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
