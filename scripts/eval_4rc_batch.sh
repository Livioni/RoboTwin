#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Edit eval_4rc_tasks.yml to choose tasks. Extra arguments are passed to eval_4rc.sh.
# Example: bash scripts/eval_4rc_batch.sh --gpu-ids 0,1 --test-num 50
for setting in clean random; do
    echo "[FourRC batch] Starting ${setting} evaluation"
    bash scripts/eval_4rc.sh \
        --config "$ROOT/scripts/eval_4rc_tasks.yml" \
        --setting "$setting" \
        --test-num 100 \
        --execute-horizon 8 \
        "$@" \
        -- --output-dir "$ROOT/eval_result/multitask/4rc_${setting}"
done
