"""파이프라인 전 구간에서 공유하는 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    DOCTOR = "DOCTOR"
    PATIENT = "PATIENT"
    MANAGER = "MANAGER"
    NURSE = "NURSE"
    UNKNOWN = "UNKNOWN"


class Method(str, Enum):
    """역할을 무엇으로 판정했는지. 리포트에서 신뢰도 표기를 가르는 기준."""

    VOICEPRINT = "VOICEPRINT"
    TEXT_PATTERN = "TEXT_PATTERN"
    ELIMINATION = "ELIMINATION"


@dataclass
class Utterance:
    """CLOVA 세그먼트 하나를 정규화한 단위. 시각은 밀리초."""

    speaker_tag: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class SpeakerProfile:
    """한 화자의 전체 발화를 묶은 것. 역할 판정은 이 단위로 한다."""

    speaker_tag: str
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.utterances)

    @property
    def utterance_count(self) -> int:
        return len(self.utterances)

    @property
    def char_count(self) -> int:
        return sum(len(u.text) for u in self.utterances)

    @property
    def mean_chars(self) -> float:
        return self.char_count / self.utterance_count if self.utterance_count else 0.0


@dataclass
class SpeakerRole:
    speaker_tag: str
    role: Role
    confidence: float
    method: Method
    scores: dict[str, float] = field(default_factory=dict)
    # 운영자 검수 큐에 올릴지. 전건을 볼 수 없으므로 신뢰도 낮은 것부터 본다.
    needs_review: bool = False
    # 화자 과분할로 이 화자와 한 사람으로 합쳐진 다른 태그들.
    merged_from: list[str] = field(default_factory=list)


@dataclass
class QAPair:
    seq: int
    question: str
    answer: str
    question_at_ms: int
    answer_at_ms: int
    asked_by: Role
    answered_by: Role
    confidence: float
    pre_registered_question_id: str | None = None


class MedicationComparison(str, Enum):
    UNSET = ""
    MATCHED = "MATCHED"
    CHANGED = "CHANGED"
    RECHECK_REQUIRED = "RECHECK_REQUIRED"


@dataclass
class ReportDraft:
    """보들 `bodeul.session_reports` 컬럼에 1:1로 대응한다.

    매니저가 손으로 쓰던 리포트의 초안이다. 확정은 매니저가 한다.
    """

    summary: str = ""
    treatment_notes: str = ""
    medication_notes: str = ""
    medication_name: str = ""
    medication_change_summary: str = ""
    medication_schedule_note: str = ""
    medication_comparison_decision_code: MedicationComparison = MedicationComparison.UNSET
    medication_comparison_note: str = ""
    next_visit_at: str | None = None
    next_visit_note: str = ""


@dataclass
class AnalysisResult:
    """온프레미스 AI가 Core API로 돌려주는 최종 산출물."""

    speakers: list[SpeakerRole] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)
    unasked_question_ids: list[str] = field(default_factory=list)
    report_draft: ReportDraft = field(default_factory=ReportDraft)
    warnings: list[str] = field(default_factory=list)
