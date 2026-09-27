"""전사 결과 하나를 받아 리포트 초안까지 돌리는 진입점.

화자분리가 없는 전사(SenseVoice)는 문장 단위로 역할을 가르고,
화자가 붙은 전사(CLOVA, 클로바노트)는 화자 단위로 가른다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import clovanote, sensevoice
from .clova import group_by_speaker
from .models import AnalysisResult, Method, Role, SpeakerRole, Utterance
from .qa import pair_qa
from .report import build_report_draft
from .roles import classify, sentence_role, split_sentences
from .terms import find_corrections

# 앵커 녹음에서 실제로 쓰인 말들. 진료과별로 넓혀야 한다.
# 약품과 질환·검사를 나눠 둔다. 섞으면 리포트의 약품란에 병명이 들어간다.
DRUG_TERMS = frozenset({"간보호제", "항생제", "혈압약", "소염제", "진통제"})

CONDITION_TERMS = frozenset({"고지혈증", "지방간", "내장지방", "빈혈", "염증"})

TEST_TERMS = frozenset({"콜레스테롤", "소변검사", "피검사", "간수치", "혈당", "신장"})

STARTER_TERMS = DRUG_TERMS | CONDITION_TERMS | TEST_TERMS


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
    utterances: list[Utterance], *, medical_terms: frozenset[str]
) -> AnalysisResult:
    """화자 라벨이 없는 전사. 문장별로 역할을 가른다."""
    sentences = _split_into_sentences(utterances)
    labelled: list[Utterance] = []
    confidence_by_role: dict[Role, list[float]] = {}

    for sentence in sentences:
        role, confidence = sentence_role(sentence.text, medical_terms=medical_terms)
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
        report_draft=build_report_draft(labelled, roles, drug_terms=DRUG_TERMS),
        warnings=warnings,
    )


def analyze_with_speakers(
    utterances: list[Utterance], *, medical_terms: frozenset[str] = frozenset()
) -> AnalysisResult:
    """화자 라벨이 있는 전사. 화자 단위로 역할을 가른다."""
    speakers, warnings = classify(group_by_speaker(utterances))
    roles = {s.speaker_tag: s.role for s in speakers}
    pairs, unasked = pair_qa(utterances, roles)
    return AnalysisResult(
        speakers=speakers,
        qa_pairs=pairs,
        unasked_question_ids=unasked,
        report_draft=build_report_draft(utterances, roles, drug_terms=DRUG_TERMS),
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
    args = parser.parse_args(argv)

    utterances, warnings = load(args.transcript)
    if not utterances:
        print("전사 내용이 비어 있습니다.", file=sys.stderr)
        return 1

    has_speakers = any(u.speaker_tag for u in utterances)
    result = (
        analyze_with_speakers(utterances, medical_terms=STARTER_TERMS)
        if has_speakers
        else analyze_without_speakers(utterances, medical_terms=STARTER_TERMS)
    )
    result.warnings = warnings + result.warnings

    corrections = find_corrections(utterances, set(STARTER_TERMS))

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

    print(f"\n질문과 답변 ({len(result.qa_pairs)}건)")
    for pair in result.qa_pairs:
        print(f"  [{pair.question_at_ms // 1000}초] ({pair.asked_by.value}) {pair.question}")
        print(f"        -> {pair.answer[:80]}")

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
                print(f"    {line}")
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
