"""매니저 음성 샘플을 성문으로 등록한다 (CLI).

    voice-enroll 김승민 샘플1.m4a 샘플2.m4a \
        --embedding 3dspeaker_..._16k.onnx --out 매니저_김승민.json

샘플은 여러 개 주는 편이 낫다. 하나만 쓰면 그날 목소리 상태에 끌려간다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .voiceprint import build_extractor, enroll


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="매니저 음성 샘플로 성문을 만든다.")
    parser.add_argument("name", help="매니저 이름 또는 사번")
    parser.add_argument("samples", type=Path, nargs="+", help="음성 샘플 (m4a, wav 등)")
    parser.add_argument("--embedding", type=Path, required=True, help="화자 임베딩 모델")
    parser.add_argument("--out", type=Path, required=True, help="성문을 저장할 .json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    from .transcribe import load_audio

    missing = [str(p) for p in [*args.samples, args.embedding] if not p.is_file()]
    if missing:
        print("파일이 없습니다: " + ", ".join(missing), file=sys.stderr)
        return 1

    extractor = build_extractor(args.embedding, num_threads=args.threads)
    voiceprint = enroll(
        [load_audio(path) for path in args.samples], extractor, name=args.name
    )
    voiceprint.save(args.out)
    print(f"{args.name}: 샘플 {len(args.samples)}개로 성문을 만들었습니다 -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
