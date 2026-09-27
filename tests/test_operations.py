"""운영 지표는 관리자만 본다.

매니저는 진료실에서 환자 옆에 있다. "화자분리가 한 사람을 쪼갰을 수 있습니다"
같은 말로 그 자리에서 할 수 있는 일이 없다. 사전을 고치고 엔진을 바꾸는 것은
운영하는 쪽의 일이다.
"""

import json
import subprocess
import sys

from voice_ai.operations import classify_warning, summarize


def test_each_warning_gets_a_kind():
    cases = {
        "speaker_01: 발화가 11자뿐이고 역할 신호가 없어 판정할 수 없습니다.": "unresolved_speaker",
        "비의사 화자 4명을 판정하지 못했습니다. --merge-non-doctor 로 합치세요.": "split_suspected",
        "전사에 띄어쓰기가 없습니다(붙어 있는 한글 최대 102자).": "unspaced_transcript",
        "[00:00] '심장'이(가) '신장'일 수 있습니다. 주변에 '소변'이(가) 나옵니다.": "confusable_term",
        "[00:00] 숫자가 엇갈립니다. '40'은(는) 이쪽에만 있습니다.": "number_disagreement",
        "[00:32] '내장지방'은(는) 다른 엔진에 없습니다.": "term_disagreement",
    }

    for warning, expected in cases.items():
        assert classify_warning(warning) == expected, warning


def test_counts_let_an_admin_see_a_trend():
    """진료 한 건마다 산문을 읽을 수는 없다. 여러 건에 걸친 추세는 숫자로 봐야 한다."""
    warnings = [
        "[00:00] 숫자가 엇갈립니다. '40'은(는) 이쪽에만 있습니다.",
        "[00:15] 숫자가 엇갈립니다. '76'은(는) 이쪽에만 있습니다.",
        "[00:32] '내장지방'은(는) 다른 엔진에 없습니다.",
    ]

    operations = summarize(warnings, term_corrections=3)

    assert operations.counts == {
        "number_disagreement": 2,
        "term_disagreement": 1,
        "term_correction": 3,
    }
    assert len(operations.details) == 3


def test_json_keeps_the_two_audiences_apart(tmp_path):
    """매니저가 받는 것과 관리자가 받는 것이 한 덩어리면 플랫폼이 나눌 수 없다."""
    transcript = tmp_path / "chunks.json"
    transcript.write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "speaker": "speaker_00",
                    "start_ms": 0,
                    "end_ms": 9000,
                    "raw_text": "콜레스테롤이 200이 정상인데 216입니다.",
                },
                {
                    "index": 1,
                    "speaker": "speaker_01",
                    "start_ms": 9000,
                    "end_ms": 11000,
                    "raw_text": "네.",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    done = subprocess.run(
        [sys.executable, "-m", "voice_ai.analyze", str(transcript), "--department", "내과", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(done.stdout)

    assert "operations" in payload
    # 경고는 관리자 쪽에만 있어야 한다.
    assert "warnings" not in payload
    for manager_facing in ("speakers", "qa_pairs", "facts", "report_draft"):
        assert manager_facing in payload
