"""매니저 성문.

의사·환자·매니저를 가르는 일에서 유일하게 전사 품질과 무관한 자리다.
나머지는 글자가 깨지면 같이 무너진다.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from voice_ai.voiceprint import (
    manager_turns,
    MATCH_THRESHOLD,
    Voiceprint,
    _speaker_audio,
    find_manager,
    similarity,
)

SAMPLE_RATE = 16000


class FakeExtractor:
    """오디오 평균값을 성문으로 삼는 가짜 추출기.

    실제 모델 없이 매칭 흐름을 고정한다. 화자마다 다른 상수로 채운 신호를
    주면 성문도 그만큼 갈린다.
    """

    def create_stream(self):
        return self

    def accept_waveform(self, sample_rate, audio):
        self._audio = np.asarray(audio, dtype=np.float32)

    def input_finished(self):
        pass

    def compute(self, stream):
        level = float(np.mean(stream._audio))
        return [level, 1.0 - level, 0.5]


def _tone(level: float, seconds: float) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), level, dtype=np.float32)


def test_similarity_is_one_for_the_same_vector():
    vector = np.array([0.3, 0.7, 0.5], dtype=np.float32)
    assert similarity(vector, vector) == pytest.approx(1.0)


def test_a_zero_vector_does_not_blow_up():
    assert similarity(np.zeros(3), np.ones(3)) == 0.0


def test_the_matching_speaker_is_named():
    extractor = FakeExtractor()
    audio = np.concatenate([_tone(0.9, 10), _tone(0.1, 10)])
    turns = [("speaker_00", 0, 10_000), ("speaker_01", 10_000, 20_000)]
    print_ = Voiceprint(name="김승민", vector=[0.9, 0.1, 0.5])

    tag, scores = find_manager(audio, turns, print_, extractor)

    assert tag == "speaker_00"
    assert scores["speaker_00"] > scores["speaker_01"]


def test_nobody_is_named_when_no_one_is_close_enough():
    # 매니저가 말을 거의 안 한 진료도 있다. 억지로 배정하면 환자가 매니저로
    # 고정되어 역할이 통째로 뒤바뀐다.
    extractor = FakeExtractor()
    audio = _tone(0.5, 10)
    turns = [("speaker_00", 0, 10_000)]

    tag, scores = find_manager(
        audio, turns, Voiceprint(name="김승민", vector=[1.0, -1.0, 0.0]), extractor
    )

    assert tag is None
    assert scores["speaker_00"] < MATCH_THRESHOLD


def test_short_turns_are_skipped():
    # 0.5초짜리 조각에서 뽑은 성문은 믿을 게 못 된다.
    audio = _tone(0.5, 10)
    turns = [("speaker_00", 0, 300), ("speaker_00", 1_000, 9_000)]

    spoken = _speaker_audio(audio, turns, "speaker_00")

    assert len(spoken) == 8 * SAMPLE_RATE


def test_a_voiceprint_survives_a_round_trip(tmp_path: Path):
    path = tmp_path / "manager.json"
    Voiceprint(name="김승민", vector=[0.1, 0.2]).save(path)

    assert Voiceprint.load(path) == Voiceprint(name="김승민", vector=[0.1, 0.2])
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "김승민"


def test_a_merged_cluster_is_split_turn_by_turn():
    """화자분리가 매니저를 의사 덩어리에 합쳐도 구간마다 재면 갈린다."""
    # 셋 다 speaker_00 으로 묶였지만 가운데 2초만 매니저 목소리다.
    audio = np.concatenate([_tone(0.1, 2), _tone(0.9, 2), _tone(0.1, 2)])
    turns = [
        ("speaker_00", 0, 2000),
        ("speaker_00", 2000, 4000),
        ("speaker_00", 4000, 6000),
    ]
    manager = Voiceprint(name="김매니저", vector=[0.9, 0.1, 0.5])

    assert manager_turns(audio, turns, manager, FakeExtractor()) == [False, True, False]


def test_a_short_turn_is_never_called_the_manager():
    """1초 미만은 성문이 사실상 잡음이다. 여기서 틀리면 의사가 매니저가 된다."""
    audio = _tone(0.9, 1)
    manager = Voiceprint(name="김매니저", vector=[0.9, 0.1, 0.5])

    assert manager_turns(audio, [("speaker_00", 0, 500)], manager, FakeExtractor()) == [False]


def test_nothing_is_marked_when_no_turn_is_close_enough():
    """매니저가 말을 거의 안 한 진료도 있다. 억지로 배정하지 않는다."""
    audio = _tone(0.1, 4)
    manager = Voiceprint(name="김매니저", vector=[0.9, 0.1, 0.5])

    assert manager_turns(audio, [("speaker_00", 0, 4000)], manager, FakeExtractor()) == [False]
