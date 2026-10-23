#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="" DATA_ROOT="" MODEL="$ROOT_DIR/weights" OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! -f "$INPUT" || ! -d "$DATA_ROOT" || ! -d "$MODEL" || -z "$OUTPUT" ]]; then
  echo 'Usage: bash run_inference.sh --input QUESTIONS.jsonl --data-root IMAGES --model WEIGHTS --output RESULT.jsonl' >&2
  exit 2
fi
MODEL="$(cd "$MODEL" && pwd)"
PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT_DIR/dependencies/runtime/bin/python3" ]]; then
  PYTHON="$ROOT_DIR/dependencies/runtime/bin/python3"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export DEC_W=1 DEC_RES=500
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
"$PYTHON" "$ROOT_DIR/inference/infer_v973_core.py" \
  --run "$MODEL/v973" --head best_task_heads.pt \
  --peft_base "$MODEL/base" --adapter "$MODEL/v973_adapter" \
  --input "$INPUT" --data-root "$DATA_ROOT" --size 256 --bs 8 \
  --out "$WORK_DIR/raw.jsonl" --out_areas "$WORK_DIR/raw.areas.json"
"$PYTHON" "$ROOT_DIR/inference/make_submission.py" \
  --input "$INPUT" --raw "$WORK_DIR/raw.jsonl" \
  --areas "$WORK_DIR/raw.areas.json" --calibration "$ROOT_DIR/calibration.json" \
  --out "$OUTPUT"
