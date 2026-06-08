#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU ALOHA camera test-set collection with fresh seeds.
#
# The base config is copied to a generated task_config name for each run. Since
# collect_data.py stores seeds under data/<task>/<task_config>/seed.txt, using a
# fresh generated config name prevents reusing seeds from older custom_aloha_clean
# collections.
#
# Default behavior:
#   - source config: task_config/custom_aloha_clean.yml
#   - episodes per task: 2
#   - generated config: task_config/_testset_<config>_<episodes>ep_<run_id>.yml
#   - output: SAVE_ROOT/custom_aloha_2ep_clean/<task>/episode*
#
# Examples:
#   bash batch_collect_aloha_camera_test_set_multi_gpu_clean.sh
#   bash batch_collect_aloha_camera_test_set_multi_gpu_clean.sh --gpus "0 1 2 3"
#   bash batch_collect_aloha_camera_test_set_multi_gpu_clean.sh --gpus "0 1 2 3" beat_block_hammer click_bell
#   RUN_ID=seed_v2 SAVE_ROOT=./datasets/robotwin_aloha_testset bash batch_collect_aloha_camera_test_set_multi_gpu_clean.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash batch_collect_aloha_camera_test_set_multi_gpu_clean.sh [options] [task_name ...]

Default recipe:
  Generate 2 episodes per task from task_config/custom_aloha_clean.yml with a fresh
  generated config name, so old data/<task>/custom_aloha_clean/seed.txt is not reused.

Options:
  --gpus IDS           GPU ids separated by spaces or commas (default: "1 2 3")
  --config NAME        Source task config basename or path (default: custom_aloha_clean)
  --episodes N         Episodes per task in the generated config (default: 2)
  --run-id ID          Stable run id for generated config/output names (default: current timestamp)
  --save-root DIR      Test-set root (default: ./datasets/robotwin_aloha_testset)
  --output-name DIR    Output subdir under save-root (default: custom_aloha_2ep_clean; use "." for save-root directly)
  --raw-root DIR       Raw RoboTwin data root (default: ./data)
  --video-fps FPS      FPS for exported MP4 videos (default: 30)
  --overwrite          Re-extract even if a task output directory already exists
  --keep-raw           Keep raw data/<task>/<generated_config> after extraction
  --bulk-extract       Use old behavior: extract all raw episodes after collection
  --no-scan-output     Do not pre-scan extracted episodes for corrupt/incomplete files
  --no-skip-complete   Do not pre-filter tasks that already have complete extracted episodes
  --dry-run            Print GPU/task assignment without running collection
  -h, --help           Show this help

Environment overrides:
  GPUS, CONFIG, EPISODES, RUN_ID, SAVE_ROOT, OUTPUT_NAME, RAW_ROOT, VIDEO_FPS,
  KEEP_RAW, OVERWRITE, CAMERAS, INCREMENTAL_EXTRACT, SCAN_EXTRACTED_OUTPUT,
  SKIP_COMPLETED, GENERATED_CONFIG_PREFIX

If no task_name is given, all task files under envs/*.py are collected.
EOF
}

normalize_config_name() {
  local config="$1"

  config="${config##*/}"
  config="${config%.yml}"
  config="${config%.yaml}"
  printf '%s\n' "$config"
}

config_path_for() {
  local config="$1"

  if [[ -f "task_config/${config}.yml" ]]; then
    printf '%s\n' "task_config/${config}.yml"
    return 0
  fi
  if [[ -f "task_config/${config}.yaml" ]]; then
    printf '%s\n' "task_config/${config}.yaml"
    return 0
  fi

  echo "ERROR: task_config/${config}.yml or task_config/${config}.yaml not found" >&2
  exit 1
}

sanitize_name() {
  local name="$1"
  name="${name//[^a-zA-Z0-9_.-]/_}"
  printf '%s\n' "$name"
}

require_positive_integer() {
  local name="$1"
  local value="$2"

  if [[ -z "$value" || "$value" == *[!0-9]* ]]; then
    echo "ERROR: ${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
  if (( 10#$value <= 0 )); then
    echo "ERROR: ${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

GPUS="${GPUS:-0 1 2 3}"
CONFIG="${CONFIG:-custom_aloha_clean}"
EPISODES="${EPISODES:-2}"
RUN_ID="${RUN_ID:-}"
SAVE_ROOT="${SAVE_ROOT:-./datasets/robotwin_aloha_testset}"
OUTPUT_NAME="${OUTPUT_NAME:-custom_aloha_2ep_clean}"
RAW_ROOT="${RAW_ROOT:-./data}"
VIDEO_FPS="${VIDEO_FPS:-30}"
KEEP_RAW="${KEEP_RAW:-0}"
OVERWRITE="${OVERWRITE:-0}"
INCREMENTAL_EXTRACT="${INCREMENTAL_EXTRACT:-1}"
SCAN_EXTRACTED_OUTPUT="${SCAN_EXTRACTED_OUTPUT:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
GENERATED_CONFIG_PREFIX="${GENERATED_CONFIG_PREFIX:-_testset}"
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
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --save-root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="$2"
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

require_positive_integer "--episodes" "$EPISODES"

CONFIG="$(normalize_config_name "$CONFIG")"
CONFIG_PATH="$(config_path_for "$CONFIG")"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
RUN_ID="$(sanitize_name "$RUN_ID")"

if [[ -z "$GENERATED_CONFIG_PREFIX" ]]; then
  echo "ERROR: GENERATED_CONFIG_PREFIX must not be empty" >&2
  exit 2
fi
GENERATED_CONFIG_PREFIX="$(sanitize_name "$GENERATED_CONFIG_PREFIX")"

GENERATED_CONFIG="${GENERATED_CONFIG_PREFIX}_$(sanitize_name "$CONFIG")_${EPISODES}ep_${RUN_ID}"
GENERATED_CONFIG_PATH="task_config/${GENERATED_CONFIG}.yml"
GENERATED_MARKER="# Generated by batch_collect_aloha_camera_test_set_multi_gpu.sh."
ACTIVE_CHILD_PID=""

if [[ -z "$OUTPUT_NAME" ]]; then
  OUTPUT_NAME="${CONFIG}_${EPISODES}ep_${RUN_ID}"
fi

if [[ ! -x "batch_collect_aloha_camera_data_multi_gpu.sh" && ! -f "batch_collect_aloha_camera_data_multi_gpu.sh" ]]; then
  echo "ERROR: batch_collect_aloha_camera_data_multi_gpu.sh not found under ${ROOT_DIR}" >&2
  exit 1
fi

cleanup_generated_config() {
  local first_line

  if [[ ! -f "$GENERATED_CONFIG_PATH" ]]; then
    return 0
  fi

  IFS= read -r first_line < "$GENERATED_CONFIG_PATH" || first_line=""
  if [[ "$first_line" == "$GENERATED_MARKER" ]]; then
    rm -f "$GENERATED_CONFIG_PATH"
  fi
}

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

cancel_active_collection() {
  trap - INT TERM EXIT
  echo
  echo "Interrupted; stopping active test-set collection..." >&2

  if [[ -n "$ACTIVE_CHILD_PID" ]]; then
    kill_process_tree "$ACTIVE_CHILD_PID" TERM
    sleep 2
    if kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
      kill_process_tree "$ACTIVE_CHILD_PID" KILL
    fi
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
  fi

  cleanup_generated_config
  echo "Stopped. Generated temporary task config was removed." >&2
  exit 130
}

write_episode_config() {
  python - "$CONFIG_PATH" "$GENERATED_CONFIG_PATH" "$EPISODES" "$GENERATED_MARKER" <<'PY'
import pathlib
import sys

import yaml

source_path = pathlib.Path(sys.argv[1])
target_path = pathlib.Path(sys.argv[2])
episodes = int(sys.argv[3])
marker = sys.argv[4]

if target_path.exists():
    first_line = target_path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if first_line != [marker]:
        raise SystemExit(f"refusing to overwrite non-generated config: {target_path}")

data = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
data["episode_num"] = episodes

body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
target_path.write_text(
    f"{marker}\n# Source: {source_path}\n{body}",
    encoding="utf-8",
)
PY
}

trap cleanup_generated_config EXIT
trap cancel_active_collection INT TERM

write_episode_config

if [[ "$OUTPUT_NAME" == "." ]]; then
  PHASE_SAVE_ROOT="$SAVE_ROOT"
else
  PHASE_SAVE_ROOT="${SAVE_ROOT}/${OUTPUT_NAME}"
fi

args=(
  --gpus "$GPUS"
  --config "$GENERATED_CONFIG"
  --save-root "$PHASE_SAVE_ROOT"
  --raw-root "$RAW_ROOT"
  --video-fps "$VIDEO_FPS"
)

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "$KEEP_RAW" == "1" ]]; then
  args+=(--keep-raw)
fi
if [[ "$INCREMENTAL_EXTRACT" != "1" ]]; then
  args+=(--bulk-extract)
fi
if [[ "$SCAN_EXTRACTED_OUTPUT" != "1" ]]; then
  args+=(--no-scan-output)
fi
if [[ "$SKIP_COMPLETED" != "1" ]]; then
  args+=(--no-skip-complete)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ ${#TASKS[@]} -gt 0 ]]; then
  args+=(-- "${TASKS[@]}")
fi

echo "Test-set source config: ${CONFIG_PATH}"
echo "Generated config: ${GENERATED_CONFIG_PATH}"
echo "Episodes per task: ${EPISODES}"
echo "Run id: ${RUN_ID}"
echo "Raw seed path pattern: ${RAW_ROOT}/<task>/${GENERATED_CONFIG}/seed.txt"
echo "Output root: ${PHASE_SAVE_ROOT}"
echo "GPUs: ${GPUS}"

bash batch_collect_aloha_camera_data_multi_gpu.sh "${args[@]}" &
ACTIVE_CHILD_PID="$!"
wait "$ACTIVE_CHILD_PID"
ACTIVE_CHILD_PID=""
