"""전사 결과 하나를 받아 리포트 초안까지 돌리는 진입점.

화자분리가 없는 전사(SenseVoice)는 문장 단위로 역할을 가르고,
화자가 붙은 전사(CLOVA, 클로바노트)는 화자 단위로 가른다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from . import clovanote, sensevoice, terminology
from .clova import group_by_speaker
from .models import AnalysisResult, Method, Role, SpeakerRole, Utterance
from .qa import pair_qa
from .report import build_report_draft
from .roles import classify, sentence_role, split_sentences
from .crosscheck import cross_check
from .facts import extract as extract_facts
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
    utterances: list[Utterance], *, terms: terminology.Terminology
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
    pairs, unasked = pair_qa(labelled, roles)
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
        unasked_question_ids=unasked,
        report_draft=build_report_draft(labelled, roles, drug_terms=terms.drugs),
        warnings=warnings,
    )


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
) -> AnalysisResult:
    """화자 라벨이 있는 전사. 화자 단위로 역할을 가른다."""
    speakers, warnings = classify(
        group_by_speaker(utterances), merge_non_doctor=merge_non_doctor
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
    pairs, unasked = pair_qa(utterances, roles)
    return AnalysisResult(
        speakers=speakers,
        qa_pairs=pairs,
        unasked_question_ids=unasked,
        report_draft=build_report_draft(utterances, roles, drug_terms=terms.drugs),
        warnings=warnings,
    )


def load(path: Path) -> tuple[list[Utterance], list[str]]:
    if path.suffix == ".txt":
        return clovanote.parse(path.read_text(encoding="utf-8")), []

    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload if isinstance(payload, list) else payload.get("chunks", [])
    return sensevoice.parse_chunks(chunks)


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
        "--department",
        help=f"진료과 용어 사전. 있는 것: {', '.join(terminology.available())}",
    )
    args = parser.parse_args(argv)

    terms = terminology.load(args.department)
    utterances, warnings = load(args.transcript)
    if not utterances:
        print("전사 내용이 비어 있습니다.", file=sys.stderr)
        return 1

    has_speakers = any(u.speaker_tag for u in utterances)
    result = (
        analyze_with_speakers(
            utterances,
            terms=terms,
            merge_non_doctor=args.merge_non_doctor,
            compare_with=load(args.compare)[0] if args.compare else None,
        )
        if has_speakers
        else analyze_without_speakers(utterances, terms=terms)
    )
    result.warnings = warnings + result.warnings

    corrections = find_corrections(utterances, set(terms.all_terms))

    if args.json:
        print(json.dumps({
            "speakers": [asdict(s) for s in result.speakers],
            "qa_pairs": [asdict(p) for p in result.qa_pairs],
            "report_draft": asdict(result.report_draft),
            "term_corrections": [asdict(c) for c in corrections],
            "warnings": result.warnings,
        }, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"화자 판정 ({'화자 라벨 있음' if has_speakers else '문장 단위'})")
    for speaker in result.speakers:
        flag = "  [검수 필요]" if speaker.needs_review else ""
        print(f"  {speaker.speaker_tag:12} {speaker.role.value:8} 신뢰도 {speaker.confidence}{flag}")

    if args.full:
        roles = {s.speaker_tag: s.role.value for s in result.speakers}
        turns = group_turns(utterances)
        print(f"\n전체 대화 ({len(turns)}발언 / {len(utterances)}구간)")
        for turn in turns:
            stamp = f"{turn.start_ms // 60000:02d}:{turn.start_ms // 1000 % 60:02d}"
            role = roles.get(turn.speaker_tag, "UNKNOWN")
            print(f"\n  {role} {stamp}")
            for line in turn.text.split("\n"):
                print(f"    {apply_corrections(line, corrections)}")
        print()

    print(f"\n질문과 답변 ({len(result.qa_pairs)}건)")
    for pair in result.qa_pairs:
        print(f"  [{pair.question_at_ms // 1000}초] ({pair.asked_by.value}) {pair.question}")
        print(f"        -> {pair.answer[:80]}")

    roles_by_tag = {sp.speaker_tag: sp.role for sp in result.speakers}
    facts = extract_facts(utterances, roles_by_tag, terms, corrections)
    if any([facts.measurements, facts.diagnoses, facts.normal, facts.drugs,
            facts.schedule, facts.duration, facts.lifestyle]):
        print("\n핵심 내용")
        if facts.measurements:
            print("  [검사 수치]")
            for m in facts.measurements:
                stamp = f"{m.start_ms // 60000:02d}:{m.start_ms // 1000 % 60:02d}"
                mark = "  (검사명 추정)" if m.inferred else ""
                print(f"    {m.test:8} {', '.join(m.values)}{mark}")
                print(f"    {'':8} └ [{stamp}] {apply_corrections(m.quote, corrections)}")
        if facts.diagnoses:
            print("  [진단·소견]")
            print(f"    {', '.join(name for name, _ in facts.diagnoses)}")
        if facts.normal:
            print("  [이상 없다고 한 항목]")
            print(f"    {', '.join(facts.normal)}")
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

    draft = result.report_draft
    sections = [
        ("진료 내용", draft.treatment_notes),
        ("약품", draft.medication_name),
        ("복용 방법", draft.medication_schedule_note),
        ("처방 기간", draft.medication_notes),
        ("다음 방문", draft.next_visit_note),
    ]
    filled = [(label, body) for label, body in sections if body]
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
        print("\n경고")
        for warning in result.warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
