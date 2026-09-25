"""녹음 파일 읽기.

매니저 폰이 무엇으로 녹음하든 받아야 하므로, 형식과 샘플레이트를 가리지 않고
16kHz 모노 float32로 맞춰 나오는지 확인한다.

전사 자체는 SenseVoice 모델이 있어야 해서 여기서 검증하지 않는다.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

from voice_ai.transcribe import SAMPLE_RATE, load_audio


def write_wav(path: Path, seconds: float, sample_rate: int, channels: int = 1) -> None:
    frames = int(seconds * sample_rate)
    tone = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, frames)) * 16000).astype(np.int16)
    if channels == 2:
        tone = np.repeat(tone, 2)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(tone.tobytes())


def test_reads_wav_as_float32_mono(tmp_path):
    source = tmp_path / "sample.wav"
    write_wav(source, seconds=1.0, sample_rate=SAMPLE_RATE)

    audio = load_audio(source)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(SAMPLE_RATE, rel=0.02)
    assert np.abs(audio).max() <= 1.0


def test_resamples_to_16k():
    """폰은 보통 48kHz로 녹음한다. 모델은 16kHz만 받는다."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "high.wav"
        write_wav(source, seconds=2.0, sample_rate=48000)

        audio = load_audio(source)

        assert len(audio) == pytest.approx(2 * SAMPLE_RATE, rel=0.02)


def test_downmixes_stereo_to_mono(tmp_path):
    source = tmp_path / "stereo.wav"
    write_wav(source, seconds=1.0, sample_rate=SAMPLE_RATE, channels=2)

    audio = load_audio(source)

    assert audio.ndim == 1
    assert len(audio) == pytest.approx(SAMPLE_RATE, rel=0.02)
