#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FOURRC_ROOT="${FOURRC_ROOT:-$(dirname "${ROOT}")/4RC}"
task=click_bell
config=""
setting=clean
gpus=0
checkpoint="${FOURRC_ROOT}/checkpoints/RoboTwin-qity"
count=2
policy_env=4rc
eval_env=RoboTwin
execute_horizon="${FOURRC_EXECUTE_HORIZON:-}"
extra=()
while (($#)); do
    case "$1" in
        --task) task="$2"; shift 2 ;;
        --config) config="$2"; shift 2 ;;
        --setting) setting="$2"; shift 2 ;;
        --gpu-ids) gpus="$2"; shift 2 ;;
        --checkpoint) checkpoint="$2"; shift 2 ;;
        --test-num) count="$2"; shift 2 ;;
        --execute-horizon)
            [[ $# -ge 2 ]] || { echo '--execute-horizon requires an integer from 1 to 16' >&2; exit 2; }
            execute_horizon="$2"
            [[ "$execute_horizon" =~ ^([1-9]|1[0-6])$ ]] || { echo 'execute_horizon must be an integer from 1 to 16' >&2; exit 2; }
            shift 2 ;;
        --viser) export FOURRC_VISER=1; shift ;;
        --viser-auto) export FOURRC_VISER=1 FOURRC_VISER_MANUAL=0; shift ;;
        --viser-port)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]{0,4}$ ]] && (( $2 <= 65535 )) || { echo '--viser-port requires an integer from 1 to 65535' >&2; exit 2; }
            export FOURRC_VISER=1 FOURRC_VISER_PORT="$2"; shift 2 ;;
        --viser-host)
            [[ $# -ge 2 && -n "$2" ]] || { echo '--viser-host requires a host address' >&2; exit 2; }
            export FOURRC_VISER=1 FOURRC_VISER_HOST="$2"; shift 2 ;;
        --policy-conda-env) policy_env="$2"; shift 2 ;;
        --eval-env-conda-env) eval_env="$2"; shift 2 ;;
        --help|-h)
            cat <<'HELP'
Usage: bash scripts/eval_4rc.sh [options] [-- scheduler-options]
  --task NAME                 Default click_bell
  --config PATH               Use an existing task list (overrides --task)
  --setting clean|random      Default clean; calibrated third camera enabled
  --checkpoint PATH           Stage2 directory with config.json/model.safetensors
  --test-num N                Default 2 episodes per task
  --execute-horizon N         Execute first N predictions per inference (1..16; deploy.yml default 8)
  --viser                     Browser UI: click once to infer and execute one chunk
  --viser-auto                Browser visualization with automatic inference
  --viser-port PORT           Viser port on policy host (default 8020; enables viewer)
  --viser-host HOST           Viser bind address (default 127.0.0.1; enables viewer)
  --gpu-ids 0,1               Default 0; one policy server + simulator per GPU
  --policy-conda-env NAME      Default 4rc
  --eval-env-conda-env NAME    Default RoboTwin
  --dry-run                   Validate the schedule without starting processes
  -- [scheduler-options]      E.g. --enable-remote --policy-server-ip HOST --policy-server-port PORT
Environment: FOURRC_ROOT, FOURRC_T5_MODEL, FOURRC_LOG_DIR, FOURRC_EXECUTE_HORIZON
             FOURRC_VISER, FOURRC_VISER_HOST, FOURRC_VISER_PORT, FOURRC_VISER_MANUAL
Use -- --stream-output to see the live viewer URL in the terminal.
For an existing remote server, enable/configure viser when starting that server.
For an existing remote server, configure execute_horizon on the server host.
HELP
            exit 0 ;;
        --) shift; extra+=("$@"); break ;;
        --dry-run) extra+=("$1"); shift ;;
        *) echo "Unknown option: $1 (use --help)" >&2; exit 2 ;;
    esac
done
[[ "$setting" == clean || "$setting" == random ]] || { echo 'setting must be clean or random' >&2; exit 2; }
if [[ -n "$execute_horizon" || ${FOURRC_EXECUTE_HORIZON+x} ]]; then
    [[ "$execute_horizon" =~ ^([1-9]|1[0-6])$ ]] || { echo 'execute_horizon must be an integer from 1 to 16' >&2; exit 2; }
    export FOURRC_EXECUTE_HORIZON="$execute_horizon"
fi
# Run scheduler/config preparation with the simulator environment's dependencies.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$eval_env"
tmp_config="$(mktemp /tmp/fourrc-eval-XXXXXX.yml)"
trap 'rm -f "$tmp_config"' EXIT
python - "$config" "$task" "$gpus" "$tmp_config" "$ROOT/XPolicyLab/policy/FourRC/deploy.yml" <<'PY'
import os, sys, yaml
from pathlib import Path
source,task,gpus,dest,deploy=sys.argv[1:]
horizon=os.environ.get('FOURRC_EXECUTE_HORIZON', yaml.safe_load(Path(deploy).read_text()).get('execute_horizon',8))
print(f'[FourRC] policy-server execute_horizon={horizon}; history=8, prediction=16, time_encoding=index', flush=True)
cfg=yaml.safe_load(Path(source).read_text()) if source else {'tasks':[task]}
cfg.update(gpu_ids=gpus, jobs_per_gpu=1, num_workers=1)
Path(dest).write_text(yaml.safe_dump(cfg))
PY
cd "$ROOT"
# Frequency is shared evaluator metadata only; Stage2 conditions on step indices.
bash scripts/eval_policy.sh multitask \
    --config "$tmp_config" --policy-name FourRC --ckpt-name "$checkpoint" \
    --env-cfg-type arx_x5 --action-type ee --task-config "4rc_${setting}" \
    --policy-conda-env "$policy_env" --eval-env-conda-env "$eval_env" \
    --test-num "$count" --frequency 15 --num-workers 1 "${extra[@]}"
