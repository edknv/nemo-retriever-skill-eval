#!/bin/bash
#
# Usage: ./test_c2_only.sh [--gpu N] [--config batch_1|batch_2]
# --gpu pins the full sweep to a single GPU; otherwise inherits CUDA_VISIBLE_DEVICES.
# --config restricts to one batch; omit to run both. Useful for two-GPU parallel
# sweep (run with --gpu 0 --config batch_1 in one shell and --gpu 1 --config
# batch_2 in another).

ONLY_CONFIG=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu) export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --config) ONLY_CONFIG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_LOG="$SCRIPT_DIR/runs.log"
N_RUNS=5
case "$ONLY_CONFIG" in
  batch_1) CONFIGS=("$SCRIPT_DIR/skill_eval_batch_1.yaml") ;;
  batch_2) CONFIGS=("$SCRIPT_DIR/skill_eval_batch_2.yaml") ;;
  "")      CONFIGS=("$SCRIPT_DIR/skill_eval_batch_1.yaml" "$SCRIPT_DIR/skill_eval_batch_2.yaml") ;;
  *)       echo "--config must be batch_1 or batch_2" >&2; exit 2 ;;
esac

for config in "${CONFIGS[@]}"; do
  config_name=$(basename "$config" .yaml)
  echo "#### config: $config_name" >> "$RUNS_LOG"

  for i in $(seq 1 "$N_RUNS"); do
    echo "==== c2_retriever [$config_name] run $i/$N_RUNS ===="

    log_file=$(mktemp)
    uv sync --extra llm --quiet 2>&1 | tail -2
    uv run --extra llm retriever skill-eval run \
      --config "$config" \
      --conditions c2_retriever \
      --domains vidore_v3_hr,vidore_v3_finance_en,vidore_v3_pharmaceuticals \
      2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}

    session_dir=$(grep -m1 '^Session dir: ' "$log_file" | sed 's/^Session dir: //')
    rm -f "$log_file"

    if [ -n "$session_dir" ]; then
      echo "$session_dir" >> "$RUNS_LOG"
      echo "logged: $session_dir"
    else
      echo "WARN: could not parse Session dir from $config_name run $i (exit=$status); not appended to $RUNS_LOG" >&2
    fi
  done
done
