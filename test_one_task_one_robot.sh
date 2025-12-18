#!/usr/bin/env bash
set -euo pipefail

# Test runner: single GPU + single robot + single task
# Pipeline:
#   1) collect raw via collect_data.sh
#   2) extract episodes via read_hdf5_advanced.py
#   3) delete raw directory (optional)
#
# Usage:
#   bash test_one_task_one_robot.sh --gpu 0 --robot piper --task beat_block_hammer
#
# Optional:
#   --config custom_piper      (override inferred config)
#   --save-root ./processed    (default: ./processed_data_test)
#   --raw-root ./data          (default: ./data)
#   --keep-raw                (do not delete raw data)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU_ID=""
ROBOT=""
TASK=""
CONFIG=""
SAVE_ROOT="./extract_data"
RAW_ROOT="./data"
KEEP_RAW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="${2:-}"; shift 2 ;;
    --robot)
      ROBOT="${2:-}"; shift 2 ;;
    --task)
      TASK="${2:-}"; shift 2 ;;
    --config)
      CONFIG="${2:-}"; shift 2 ;;
    --save-root)
      SAVE_ROOT="${2:-}"; shift 2 ;;
    --raw-root)
      RAW_ROOT="${2:-}"; shift 2 ;;
    --keep-raw)
      KEEP_RAW=1; shift 1 ;;
    -h|--help)
      sed -n '1,60p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$GPU_ID" || -z "$ROBOT" || -z "$TASK" ]]; then
  echo "Missing required args. Example:" >&2
  echo "  bash $0 --gpu 0 --robot piper --task beat_block_hammer" >&2
  exit 2
fi

# Infer CONFIG from ROBOT if not provided
if [[ -z "$CONFIG" ]]; then
  case "${ROBOT}" in
    aloha|aloha-agilex)
      CONFIG="custom_aloha" ;;
    ARX-X5|arx|arx-x5)
      CONFIG="custom_arx" ;;
    franka|franka-panda)
      CONFIG="custom_franka" ;;
    ur5|ur5-wsg)
      CONFIG="custom_ur5" ;;
    piper)
      CONFIG="custom_piper" ;;
    *)
      echo "Unknown robot '${ROBOT}'. Please pass --config explicitly." >&2
      exit 2
      ;;
  esac
fi

# Validate config exists (.yml or .yaml)
if [[ ! -f "task_config/${CONFIG}.yml" && ! -f "task_config/${CONFIG}.yaml" ]]; then
  echo "ERROR: task_config/${CONFIG}.yml or task_config/${CONFIG}.yaml not found" >&2
  exit 1
fi

mkdir -p "$SAVE_ROOT"

# Collect raw
echo "[GPU ${GPU_ID}] [${ROBOT}] collecting: ${TASK} (${CONFIG})"
bash collect_data.sh "$TASK" "$CONFIG" "$GPU_ID"

raw_dir="${RAW_ROOT}/${TASK}/${CONFIG}"
episode_dir="${raw_dir}/data"
out_dir="${SAVE_ROOT}/${TASK}/${ROBOT}"

if [[ ! -d "$episode_dir" ]]; then
  echo "ERROR: episode dir not found: $episode_dir" >&2
  exit 1
fi

mkdir -p "$out_dir"

# Extract
echo "[GPU ${GPU_ID}] [${ROBOT}] extracting -> ${out_dir}"
python read_hdf5_advanced.py "$episode_dir" --save-camera-data --output-dir "$out_dir" --overwrite

# Cleanup
if [[ "$KEEP_RAW" -eq 0 ]]; then
  echo "[GPU ${GPU_ID}] [${ROBOT}] deleting raw: ${raw_dir}"
  rm -rf "$raw_dir"
else
  echo "Keeping raw data: ${raw_dir}"
fi

echo "Done. Extracted data: ${out_dir}"
