#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU dispatcher for ALOHA camera data collection.
#
# It splits task names across GPUs and starts one worker per GPU. Each worker
# reuses batch_collect_aloha_camera_data.sh, so the raw collection and extracted
# camera-data layout stay identical to the single-GPU script.
#
# Examples:
#   bash batch_collect_aloha_camera_data_multi_gpu.sh
#   bash batch_collect_aloha_camera_data_multi_gpu.sh --gpus "0 1 2 3"
#   bash batch_collect_aloha_camera_data_multi_gpu.sh --gpus "0 1 2 3" beat_block_hammer click_bell
#   SAVE_ROOT=./datasets/robotwin_aloha_4gpu bash batch_collect_aloha_camera_data_multi_gpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash batch_collect_aloha_camera_data_multi_gpu.sh [options] [task_name ...]

Options:
  --gpus IDS          GPU ids separated by spaces or commas (default: "0 1")
  --config NAME       Task config basename under task_config/ (default: custom_aloha)
  --save-root DIR     Final extracted dataset root (default: ./datasets/robotwin_aloha)
  --raw-root DIR      Raw RoboTwin data root (default: ./data)
  --video-fps FPS     FPS for exported MP4 videos (default: 30)
  --overwrite         Re-extract even if a task output directory already exists
  --keep-raw          Keep raw data/<task>/<config> after extraction
  --bulk-extract      Use old behavior: extract all raw episodes after collection
  --no-scan-output    Do not pre-scan extracted episodes for corrupt/incomplete files
  --no-skip-complete  Do not pre-filter tasks that already have episode_num complete extracted episodes
  --dry-run           Print GPU/task assignment without running collection
  -h, --help          Show this help

Environment overrides:
  GPUS, CONFIG, SAVE_ROOT, RAW_ROOT, VIDEO_FPS, KEEP_RAW, OVERWRITE, CAMERAS, INCREMENTAL_EXTRACT,
  SCAN_EXTRACTED_OUTPUT, SKIP_COMPLETED

If no task_name is given, all task files under envs/*.py are collected.
Each GPU runs one task at a time, then continues with the next assigned task.
EOF
}

GPUS="${GPUS:-0 1}"
CONFIG="${CONFIG:-custom_aloha}"
SAVE_ROOT="${SAVE_ROOT:-./datasets/robotwin_aloha}"
RAW_ROOT="${RAW_ROOT:-./data}"
VIDEO_FPS="${VIDEO_FPS:-30}"
KEEP_RAW="${KEEP_RAW:-0}"
OVERWRITE="${OVERWRITE:-0}"
CAMERAS="${CAMERAS:-head_camera left_camera right_camera third_view}"
INCREMENTAL_EXTRACT="${INCREMENTAL_EXTRACT:-1}"
SCAN_EXTRACTED_OUTPUT="${SCAN_EXTRACTED_OUTPUT:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN=0

TASKS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --save-root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    --raw-root)
      RAW_ROOT="$2"
      shift 2
      ;;
    --video-fps)
      VIDEO_FPS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --keep-raw)
      KEEP_RAW=1
      shift
      ;;
    --bulk-extract)
      INCREMENTAL_EXTRACT=0
      shift
      ;;
    --no-scan-output)
      SCAN_EXTRACTED_OUTPUT=0
      shift
      ;;
    --no-skip-complete)
      SKIP_COMPLETED=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      TASKS+=("$@")
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      TASKS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "batch_collect_aloha_camera_data.sh" && ! -f "batch_collect_aloha_camera_data.sh" ]]; then
  echo "ERROR: batch_collect_aloha_camera_data.sh not found under ${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -f "task_config/${CONFIG}.yml" && ! -f "task_config/${CONFIG}.yaml" ]]; then
  echo "ERROR: task_config/${CONFIG}.yml or task_config/${CONFIG}.yaml not found" >&2
  exit 1
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  mapfile -t TASKS < <(
python - <<'PY'
import pathlib
root = pathlib.Path("envs")
ignore = {"__init__", "_base_task", "_GLOBAL_CONFIGS"}
tasks = sorted(
    p.stem for p in root.glob("*.py")
    if p.stem not in ignore and not p.stem.startswith("_")
)
for task in tasks:
    print(task)
PY
  )
fi

GPUS="${GPUS//,/ }"
read -r -a GPU_LIST <<< "$GPUS"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no GPU ids were provided" >&2
  exit 1
fi

EXPECTED_EPISODE_NUM="$(
python - "$CONFIG" <<'PY'
import pathlib
import sys

import yaml

config = sys.argv[1]
for suffix in (".yml", ".yaml"):
    path = pathlib.Path("task_config") / f"{config}{suffix}"
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        print(int(data.get("episode_num", 0)))
        break
else:
    raise SystemExit(f"task config not found: {config}")
PY
)"

if [[ "$EXPECTED_EPISODE_NUM" -le 0 ]]; then
  echo "ERROR: episode_num must be a positive integer in task_config/${CONFIG}.yml/yaml" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
LOG_DIR="${SAVE_ROOT}/_logs"
WORKER_PIDS=()

kill_process_tree() {
  local pid="$1"
  local sig="${2:-TERM}"
  local child

  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  while IFS= read -r child; do
    kill_process_tree "$child" "$sig"
  done < <(pgrep -P "$pid" 2>/dev/null || true)

  kill "-${sig}" "$pid" 2>/dev/null || true
}

cleanup_tmp_dir() {
  rm -rf "$TMP_DIR"
}

cancel_workers() {
  local pid

  trap - INT TERM EXIT
  echo
  echo "Interrupted; stopping GPU workers and pending tasks..." >&2

  for pid in "${WORKER_PIDS[@]}"; do
    kill_process_tree "$pid" TERM
  done

  sleep 2

  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill_process_tree "$pid" KILL
    fi
  done

  wait 2>/dev/null || true
  cleanup_tmp_dir
  echo "Stopped GPU workers. No further tasks will be started." >&2
  exit 130
}

trap cleanup_tmp_dir EXIT
trap cancel_workers INT TERM

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR"
fi

ORIGINAL_TASK_COUNT="${#TASKS[@]}"
TASK_STATUS_FILE="${TMP_DIR}/task_status.tsv"
FILTERED_TASKS_FILE="${TMP_DIR}/filtered_tasks.txt"
TASK_SUMMARY_FILE="${TMP_DIR}/task_summary.tsv"
EFFECTIVE_SKIP_COMPLETED="$SKIP_COMPLETED"

if [[ "$OVERWRITE" == "1" ]]; then
  EFFECTIVE_SKIP_COMPLETED=0
fi

python - "$SAVE_ROOT" "$RAW_ROOT" "$CONFIG" "$EXPECTED_EPISODE_NUM" "$EFFECTIVE_SKIP_COMPLETED" \
  "$TASK_STATUS_FILE" "$FILTERED_TASKS_FILE" "$TASK_SUMMARY_FILE" "${TASKS[@]}" <<'PY'
import pathlib
import re
import sys

save_root = pathlib.Path(sys.argv[1])
raw_root = pathlib.Path(sys.argv[2])
config = sys.argv[3]
expected = int(sys.argv[4])
skip_completed = sys.argv[5] == "1"
status_file = pathlib.Path(sys.argv[6])
filtered_tasks_file = pathlib.Path(sys.argv[7])
summary_file = pathlib.Path(sys.argv[8])
tasks = sys.argv[9:]


def complete_episode_count(out_dir):
    count = 0
    if not out_dir.is_dir():
        return count

    for episode_dir in out_dir.iterdir():
        if not episode_dir.is_dir() or not re.fullmatch(r"episode[0-9]+", episode_dir.name):
            continue
        if (
            (episode_dir / "meta.json").is_file()
            and (episode_dir / "camera_data").is_dir()
            and (episode_dir / "robot_data").is_dir()
        ):
            count += 1
    return count


def count_glob(path, pattern):
    return sum(1 for _ in path.glob(pattern)) if path.is_dir() else 0


rows = []
filtered_tasks = []
summary = {"complete": 0, "partial": 0, "pending": 0}

for task in tasks:
    out_dir = save_root / task
    raw_dir = raw_root / task / config
    complete_count = complete_episode_count(out_dir)
    if complete_count >= expected:
        state = "complete"
    elif complete_count > 0:
        state = "partial"
    else:
        state = "pending"

    summary[state] += 1
    raw_seed = int((raw_dir / "seed.txt").is_file())
    raw_traj_count = count_glob(raw_dir / "_traj_data", "episode*.pkl")
    raw_hdf5_count = (
        count_glob(raw_dir / "data", "episode*.hdf5")
        + count_glob(raw_dir / "data", "episode*.h5")
    )
    rows.append(
        "\t".join(
            [
                task,
                state,
                str(complete_count),
                str(expected),
                str(raw_seed),
                str(raw_traj_count),
                str(raw_hdf5_count),
            ]
        )
    )
    if not (skip_completed and state == "complete"):
        filtered_tasks.append(task)

status_file.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
filtered_tasks_file.write_text("\n".join(filtered_tasks) + ("\n" if filtered_tasks else ""), encoding="utf-8")
summary_file.write_text(
    "\t".join(
        [
            str(len(tasks)),
            str(len(filtered_tasks)),
            str(summary["complete"]),
            str(summary["partial"]),
            str(summary["pending"]),
        ]
    )
    + "\n",
    encoding="utf-8",
)
PY

mapfile -t TASKS < "$FILTERED_TASKS_FILE"
IFS=$'\t' read -r STATUS_TOTAL STATUS_REMAINING STATUS_COMPLETE STATUS_PARTIAL STATUS_PENDING < "$TASK_SUMMARY_FILE"

for worker_idx in "${!GPU_LIST[@]}"; do
  : > "${TMP_DIR}/gpu_${worker_idx}.tasks"
done

for task_idx in "${!TASKS[@]}"; do
  worker_idx=$((task_idx % ${#GPU_LIST[@]}))
  printf '%s\n' "${TASKS[$task_idx]}" >> "${TMP_DIR}/gpu_${worker_idx}.tasks"
done

echo "Task config: ${CONFIG}"
echo "GPUs: ${GPU_LIST[*]}"
echo "Tasks requested: ${ORIGINAL_TASK_COUNT}"
echo "Expected episodes per task: ${EXPECTED_EPISODE_NUM}"
echo "Task status: ${STATUS_COMPLETE} complete, ${STATUS_PARTIAL} partial, ${STATUS_PENDING} pending"
if [[ "$EFFECTIVE_SKIP_COMPLETED" == "1" ]]; then
  echo "Tasks assigned after completion filter: ${#TASKS[@]}"
else
  echo "Tasks assigned after completion filter: ${#TASKS[@]} (filter disabled)"
fi
echo "Cameras: ${CAMERAS}"
echo "Output root: ${SAVE_ROOT}"
echo "Incremental extract: ${INCREMENTAL_EXTRACT}"
echo "Scan extracted output: ${SCAN_EXTRACTED_OUTPUT}"
echo "Logs: ${LOG_DIR}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Task completion status:"
  while IFS=$'\t' read -r task state complete expected raw_seed raw_traj raw_hdf5; do
    printf '  %-32s %8s %3s/%s raw_seed=%s raw_traj=%s raw_hdf5=%s\n' \
      "$task" "$state" "$complete" "$expected" "$raw_seed" "$raw_traj" "$raw_hdf5"
  done < "$TASK_STATUS_FILE"
fi

for worker_idx in "${!GPU_LIST[@]}"; do
  gpu_id="${GPU_LIST[$worker_idx]}"
  task_file="${TMP_DIR}/gpu_${worker_idx}.tasks"
  mapfile -t worker_tasks < "$task_file"

  echo
  echo "[GPU ${gpu_id}] assigned ${#worker_tasks[@]} task(s): ${worker_tasks[*]:-<none>}"

  if [[ "$DRY_RUN" == "1" || ${#worker_tasks[@]} -eq 0 ]]; then
    continue
  fi

  worker_args=(
    --gpu "$gpu_id"
    --config "$CONFIG"
    --save-root "$SAVE_ROOT"
    --raw-root "$RAW_ROOT"
    --video-fps "$VIDEO_FPS"
  )

  if [[ "$OVERWRITE" == "1" ]]; then
    worker_args+=(--overwrite)
  fi
  if [[ "$KEEP_RAW" == "1" ]]; then
    worker_args+=(--keep-raw)
  fi
  if [[ "$INCREMENTAL_EXTRACT" != "1" ]]; then
    worker_args+=(--bulk-extract)
  fi
  if [[ "$SCAN_EXTRACTED_OUTPUT" != "1" ]]; then
    worker_args+=(--no-scan-output)
  fi

  log_file="${LOG_DIR}/gpu_${gpu_id}.log"
  (
    export CAMERAS
    export INCREMENTAL_EXTRACT
    export SCAN_EXTRACTED_OUTPUT
    bash batch_collect_aloha_camera_data.sh "${worker_args[@]}" "${worker_tasks[@]}"
  ) > "$log_file" 2>&1 &

  WORKER_PIDS+=("$!")
  echo "[GPU ${gpu_id}] started worker pid=$!, log=${log_file}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run only; no collection started."
  exit 0
fi

failed=0
for job in $(jobs -p); do
  if ! wait "$job"; then
    failed=1
  fi
done

if [[ "$failed" != "0" ]]; then
  echo "ERROR: one or more GPU workers failed. Check logs under ${LOG_DIR}" >&2
  exit 1
fi

echo "All GPU workers finished. Extracted data saved under: ${SAVE_ROOT}"
