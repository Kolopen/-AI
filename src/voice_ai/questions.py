"""보호자가 미리 남긴 질문 읽기.

보호자는 진료실에 없다. 궁금한 것을 미리 적어 보내고, 매니저가 대신 묻는다.
그래서 질문은 녹음에서 알아내는 것이 아니라 이미 글로 가지고 있다. 알아들어야
하는 것은 의사의 답변뿐이고, 그쪽은 가장 또렷하게 담긴다.

형식은 두 가지를 받는다. 플랫폼이 보내는 것은 앞쪽이고, 손으로 적을 때는
뒤쪽이 편하다.

    q_dizzy = 어지럼증이 약 때문인가요?
    운전을 계속해도 되나요?
"""

from __future__ import annotations

import re
from pathlib import Path

_LABELLED = re.compile(r"^([A-Za-z0-9_-]+)\s*=\s*(.+)$")


def parse(text: str) -> dict[str, str]:
    """줄마다 질문 하나. 빈 줄과 #으로 시작하는 줄은 건너뛴다."""
    questions: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        found = _LABELLED.match(line)
        if found:
            question_id, body = found.group(1), found.group(2).strip()
        else:
            # 번호를 안 붙였으면 순서대로 매긴다. 플랫폼에서 올 때는 id 가
            # 함께 오므로, 손으로 적는 경우를 위한 편의다.
            question_id, body = f"문{len(questions) + 1}", line

        if body:
            questions[question_id] = body
    return questions


def load(path: Path) -> dict[str, str]:
    return parse(path.read_text(encoding="utf-8"))
