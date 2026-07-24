#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <codex|opencode|traecli> [prefix]" >&2
  exit 2
fi

AGENT="$1"
PREFIX="${2:-$AGENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_FILE="${SCRIPT_DIR}/agents/${AGENT}.sh"
if [[ ! -f "$AGENT_FILE" ]]; then
  echo "Unknown agent '$AGENT'; expected one of: codex opencode traecli" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$AGENT_FILE"

DATASET="${DATASET:-${SCRIPT_DIR}/../datasets/patcheval_verified.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/agent_runs}"

mount_args=()
for mount in "${AGENT_MOUNTS[@]:-}"; do
  mount_args+=(--mount "$mount")
done

python "${SCRIPT_DIR}/patch_agent_runner.py" \
  --input "$DATASET" \
  --output-dir "$OUTPUT_BASE" \
  --limit "${LIMIT:--1}" \
  --concurrency "${CONCURRENCY:-4}" \
  --run-label "$PREFIX" \
  "${mount_args[@]}" \
  "${AGENT_EXTRA_ARGS[@]}" \
  --agent-command "$AGENT_COMMAND" \
  --agent-timeout "${AGENT_TIMEOUT:-3600}" \
  --container-prefix "patcheval-${AGENT}"
