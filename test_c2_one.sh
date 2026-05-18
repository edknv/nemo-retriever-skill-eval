#!/bin/bash
# test_c2_one.sh — single-(config, domain) c2_retriever iteration.
#
# Usage:
#   ./test_c2_one.sh [--gpu N] <config> <domain>
#
# <config> accepts a shorthand or a full path:
#   batch_1   -> ./skill_eval_batch_1.yaml
#   batch_2   -> ./skill_eval_batch_2.yaml
#   /path/to/whatever.yaml
#
# <domain> is one of: vidore_v3_hr, vidore_v3_finance_en, vidore_v3_pharmaceuticals
#
# --gpu N pins the run to GPU index N (sets CUDA_VISIBLE_DEVICES). Omit to inherit
# whatever CUDA_VISIBLE_DEVICES is in the env (or use all visible GPUs).
#
# Parallelize two pairs across the two GPUs:
#   ./test_c2_one.sh --gpu 0 batch_2 vidore_v3_hr      &
#   ./test_c2_one.sh --gpu 1 batch_2 vidore_v3_finance_en &
#   wait
#
# Appends the resulting artifact dir to runs.log, tagged with `# <config> × <domain>`
# so rotation iterations are easy to scan after the fact.
#
#  ┌─────────────────┬─────────┬──────┬─────────────┐
#  │     domain      │ entries │ PDFs │ total pages │
#  ├─────────────────┼─────────┼──────┼─────────────┤
#  │ pharmaceuticals │ 16      │ 52   │ 2,313       │
#  ├─────────────────┼─────────┼──────┼─────────────┤
#  │ finance_en      │ 15      │ 6    │ 2,942       │
#  ├─────────────────┼─────────┼──────┼─────────────┤
#  │ hr              │ 15      │ 14   │ 1,110       │
#  └─────────────────┴─────────┴──────┴─────────────┘

set -e

if [ "$1" = "--gpu" ]; then
  if [ -z "$2" ]; then
    echo "--gpu requires an index" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$2"
  shift 2
fi

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 [--gpu N] <config> <domain>" >&2
  echo "  <config>: batch_1 | batch_2 | /path/to/config.yaml" >&2
  echo "  <domain>: vidore_v3_hr | vidore_v3_finance_en | vidore_v3_pharmaceuticals" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_LOG="$SCRIPT_DIR/runs.log"

CONFIG_ARG="$1"
DOMAIN="$2"

case "$CONFIG_ARG" in
  batch_1|batch_2) CONFIG_PATH="$SCRIPT_DIR/skill_eval_${CONFIG_ARG}.yaml" ;;
  *)               CONFIG_PATH="$CONFIG_ARG" ;;
esac
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)

if [ ! -f "$CONFIG_PATH" ]; then
  echo "config not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "==== c2_retriever [$CONFIG_NAME × $DOMAIN]  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ===="

# Without the `llm` extra the in-process judge fails with `ModuleNotFoundError: litellm`,
# producing an artifact whose session_summary.md has `judge=—`. Sync is a no-op if it's
# already installed.
uv sync --extra llm --quiet 2>&1 | tail -3

log_file=$(mktemp)
uv run --extra llm retriever skill-eval run \
  --config "$CONFIG_PATH" \
  --conditions c2_retriever \
  --domains "$DOMAIN" \
  2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}

session_dir=$(grep -m1 '^Session dir: ' "$log_file" | sed 's/^Session dir: //')
rm -f "$log_file"

if [ -n "$session_dir" ]; then
  echo "$session_dir  # $CONFIG_NAME × $DOMAIN" >> "$RUNS_LOG"
  echo "logged: $session_dir"
else
  echo "WARN: no Session dir from $CONFIG_NAME × $DOMAIN (exit=$status); not appended to $RUNS_LOG" >&2
  exit "$status"
fi
