#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Edit eval_4rc_tasks.yml to choose tasks. Extra arguments are passed to eval_4rc.sh.
# Example: bash scripts/eval_4rc_batch.sh --checkpoint /path/to/checkpoint --gpu-ids 0,1 --test-num 50
# Add --resume to skip completed tasks for this checkpoint in each setting.
exit_code=0
for setting in clean random; do
    echo "[FourRC batch] Starting ${setting} evaluation"
    if bash scripts/eval_4rc.sh \
        --config "$ROOT/scripts/eval_4rc_tasks.yml" \
        --setting "$setting" \
        --test-num 10 \
        --execute-horizon 20 \
        "$@" \
        -- --output-dir "$ROOT/eval_result/multitask/4rc_${setting}"; then
        continue
    else
        status=$?
        # Still evaluate the other setting when individual tasks fail.
        if ((status != 1)); then exit "$status"; fi
        exit_code=1
    fi
done
exit "$exit_code"
