#!/usr/bin/env bash
# 녹음 하나를 두 엔진으로 돌리고 맞대어 분석한다.
#
#   ./scripts/run.sh 녹음.m4a [진료과] [진료일]
#   ./scripts/run.sh 녹음.m4a 내과 2026-10-02
#
# 결과는 녹음 파일 옆에 파일 이름으로 폴더를 만들어 쌓는다. 녹음을 어디
# 두든 결과가 따라가므로 한 건이 한 폴더에 모인다. VOICE_OUT 으로 바꾼다.
#
#   ~/Desktop/bodeul-voice/녹음.m4a  ->  ~/Desktop/bodeul-voice/녹음/
#
# 저장소 안에 두더라도 전사 결과는 .gitignore 가 막는다. 실제 진료 음성이면
# 녹음 자체도 저장소 밖에 두는 편이 안전하다.
#
# 모델 경로는 VOICE_MODELS 로 바꾼다. 기본값은 현재 폴더다.
#   VOICE_MODELS=~/models ./scripts/run.sh 녹음.m4a

set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

AUDIO="$1"
DEPARTMENT="${2:-내과}"
CONSULT_DATE="${3:-$(date +%F)}"

MODELS="${VOICE_MODELS:-.}"
SPEAKERS="${VOICE_SPEAKERS:-3}"
THREADS="${VOICE_THREADS:-8}"
MAX_CHUNK="${VOICE_MAX_CHUNK:-8}"

SENSE="$MODELS/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
MOON="$MODELS/sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27"
SEGMENTATION="$MODELS/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDING="$MODELS/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

# 15초 기다렸다가 파일이 없다고 깨지면 아까우니 먼저 다 확인한다.
missing=0
for path in \
    "$AUDIO" \
    "$SENSE/model.int8.onnx" "$SENSE/tokens.txt" \
    "$MOON/encoder_model.ort" "$MOON/decoder_model_merged.ort" "$MOON/tokens.txt" \
    "$SEGMENTATION" "$EMBEDDING"
do
    if [ ! -f "$path" ]; then
        echo "없음: $path" >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo >&2
    echo "모델 경로가 다르면 VOICE_MODELS 로 알려주세요. 받는 법은 README 의 '전사 준비'." >&2
    exit 1
fi

name="$(basename "${AUDIO%.*}")"
OUT="${VOICE_OUT:-$(cd "$(dirname "$AUDIO")" && pwd)/$name}"
mkdir -p "$OUT"
echo "결과 폴더: $OUT"

echo "[1/3] sensevoice 전사"
voice-transcribe "$AUDIO" \
    --engine sensevoice \
    --model "$SENSE/model.int8.onnx" \
    --tokens "$SENSE/tokens.txt" \
    --segmentation "$SEGMENTATION" \
    --embedding "$EMBEDDING" \
    --speakers "$SPEAKERS" --threads "$THREADS" --max-chunk "$MAX_CHUNK" \
    --out "$OUT/sense.json"

echo "[2/3] moonshine 전사 (맞대기용, 같은 설정)"
voice-transcribe "$AUDIO" \
    --engine moonshine \
    --encoder "$MOON/encoder_model.ort" \
    --decoder "$MOON/decoder_model_merged.ort" \
    --tokens "$MOON/tokens.txt" \
    --segmentation "$SEGMENTATION" \
    --embedding "$EMBEDDING" \
    --speakers "$SPEAKERS" --threads "$THREADS" --max-chunk "$MAX_CHUNK" \
    --out "$OUT/moon.json"

echo "[3/3] 분석"
voice-analyze "$OUT/sense.json" \
    --compare "$OUT/moon.json" \
    --department "$DEPARTMENT" \
    --date "$CONSULT_DATE" \
    | tee "$OUT/report.txt"

voice-analyze "$OUT/sense.json" \
    --compare "$OUT/moon.json" \
    --department "$DEPARTMENT" \
    --date "$CONSULT_DATE" \
    --json > "$OUT/report.json"

echo
echo "전체 대화를 보려면:"
echo "  voice-analyze $OUT/sense.json --compare $OUT/moon.json --department $DEPARTMENT --full"
echo "결과: $OUT/"
