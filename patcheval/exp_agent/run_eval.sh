#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <prefix> [runner-output-dir]" >&2
  exit 2
fi

PREFIX="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-${SCRIPT_DIR}/../datasets/patcheval_verified.json}"

if [[ $# -eq 2 ]]; then
  RUNNER_OUTPUT="$2"
else
  RUNNER_OUTPUT="$(find "${SCRIPT_DIR}/agent_runs" -maxdepth 1 -type d -name "*-${PREFIX}" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [[ -z "${RUNNER_OUTPUT:-}" || ! -d "$RUNNER_OUTPUT" ]]; then
  echo "Could not find runner output for prefix '$PREFIX'" >&2
  exit 1
fi

PROCESS_DATA="${SCRIPT_DIR}/eval_inputs/${PREFIX}.jsonl"
python "${SCRIPT_DIR}/process_data.py" \
  --runner-output "$RUNNER_OUTPUT" \
  --process-data-path "$PROCESS_DATA" \
  --test-data-path "$DATASET"

(
  cd "${SCRIPT_DIR}/../evaluation"
  python run_evaluation.py \
    --output "results/${PREFIX}" \
    --patch_file "$PROCESS_DATA" \
    --input_file "$DATASET" \
    --max_workers "${MAX_WORKERS:-4}" \
    --log_level "${LOG_LEVEL:-INFO}"
)
