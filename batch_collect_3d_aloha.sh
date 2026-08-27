#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU_ID="${1:-0}"
TASK_CONFIG="3d_aloha_dataset"
EXPECTED_TASKS=50
EXPECTED_EPISODES=50
OUTPUT_ROOT="data/${TASK_CONFIG}"
LOG_ROOT="${OUTPUT_ROOT}/_logs_gpu${GPU_ID}"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash $0 [gpu_id]" >&2
  exit 2
fi

if [[ ! -f "env_cfg/task_config/${TASK_CONFIG}.yml" ]]; then
  echo "Missing config: env_cfg/task_config/${TASK_CONFIG}.yml" >&2
  exit 1
fi

configured_episodes="$(awk '$1 == "episode_num:" {print $2; exit}' "env_cfg/task_config/${TASK_CONFIG}.yml")"
if [[ "$configured_episodes" != "$EXPECTED_EPISODES" ]]; then
  echo "Expected episode_num=${EXPECTED_EPISODES}, found '${configured_episodes}' in the config." >&2
  exit 1
fi

mapfile -t TASKS < <(
  find envs -maxdepth 1 -type f -name '*.py' -printf '%f\n' \
    | sed 's/\.py$//' \
    | grep -Ev '^(__init__|_base_task|_GLOBAL_CONFIGS)$' \
    | grep -Ev '^_' \
    | sort
)

if [[ "${#TASKS[@]}" -ne "$EXPECTED_TASKS" ]]; then
  echo "Expected ${EXPECTED_TASKS} tasks, found ${#TASKS[@]}." >&2
  echo "Refusing to start so the task scope is not silently changed." >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"

completed=()
skipped=()
failed=()

echo "Starting ${EXPECTED_TASKS} tasks on GPU ${GPU_ID}"
echo "Config: ${TASK_CONFIG}; episodes per task: ${EXPECTED_EPISODES}"
echo "Logs: ${LOG_ROOT}"

for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  task_number=$((index + 1))
  data_dir="${OUTPUT_ROOT}/${task}/aloha_agilex/data"
  episode_count=0

  if [[ -d "$data_dir" ]]; then
    episode_count="$(find "$data_dir" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)"
  fi

  if [[ "$episode_count" -ge "$EXPECTED_EPISODES" ]]; then
    echo "[${task_number}/${EXPECTED_TASKS}] SKIP ${task}: ${episode_count}/${EXPECTED_EPISODES} episodes"
    skipped+=("$task")
    continue
  fi

  echo "[${task_number}/${EXPECTED_TASKS}] START ${task}: ${episode_count}/${EXPECTED_EPISODES} episodes exist"
  if bash collect_data.sh "$task" "$TASK_CONFIG" "$GPU_ID" \
      2>&1 | tee "$LOG_ROOT/${task}.log"; then
    final_count=0
    if [[ -d "$data_dir" ]]; then
      final_count="$(find "$data_dir" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)"
    fi

    if [[ "$final_count" -ge "$EXPECTED_EPISODES" ]]; then
      echo "[${task_number}/${EXPECTED_TASKS}] DONE ${task}: ${final_count}/${EXPECTED_EPISODES} episodes"
      completed+=("$task")
    else
      echo "[${task_number}/${EXPECTED_TASKS}] INCOMPLETE ${task}: ${final_count}/${EXPECTED_EPISODES} episodes" >&2
      failed+=("$task")
    fi
  else
    echo "[${task_number}/${EXPECTED_TASKS}] FAILED ${task}; continuing with the next task" >&2
    failed+=("$task")
  fi
done

echo
echo "Collection summary"
echo "  completed this run: ${#completed[@]}"
echo "  already complete:   ${#skipped[@]}"
echo "  failed/incomplete:  ${#failed[@]}"

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "Failed tasks: ${failed[*]}" >&2
  echo "Rerun the same command to resume them." >&2
  exit 1
fi

echo "All ${EXPECTED_TASKS} tasks contain at least ${EXPECTED_EPISODES} episodes."
