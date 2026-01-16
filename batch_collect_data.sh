#!/usr/bin/env bash
set -euo pipefail

# 50 tasks (envs/*.py) on a single GPU & single robot
# Pipeline per task:
#   1) collect raw hdf5 via collect_data.sh
#   2) extract per-episode camera+robot data via read_hdf5_advanced.py
#   3) delete raw directory to save space

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# optional: prepare PYTHONPATH etc (no-op if not present)
./script/.update_path.sh > /dev/null 2>&1 || true

# Where to store FINAL extracted data (task/robot/episode*/...)
# SAVE_ROOT="${SAVE_ROOT:-./datasets/aloha-agilex-1}"
# SAVE_ROOT="${SAVE_ROOT:-./datasets/franka-panda-1}"
# SAVE_ROOT="${SAVE_ROOT:-./datasets/arx-x5-1}"
# SAVE_ROOT="${SAVE_ROOT:-./datasets/piper-1}"
SAVE_ROOT="${SAVE_ROOT:-./datasets/ur5-wsg-1}"

# Where RoboTwin writes RAW collected data (task/config/...)
RAW_ROOT="${RAW_ROOT:-./data}"

mkdir -p "$SAVE_ROOT"

# Single-robot settings
# - CONFIG is the task_config name WITHOUT extension; collect_data.py supports both .yml and .yaml now.
# ROBOT="${ROBOT:-aloha-agilex}"
# ROBOT="${ROBOT:-franka-panda}"
# ROBOT="${ROBOT:-arx-x5}"
# ROBOT="${ROBOT:-piper}"
ROBOT="${ROBOT:-ur5-wsg}"

# CONFIG="${CONFIG:-custom_aloha}"
# CONFIG="${CONFIG:-custom_franka}"
# CONFIG="${CONFIG:-custom_arx}"
# CONFIG="${CONFIG:-custom_piper}"
CONFIG="${CONFIG:-custom_ur5}"

GPU_ID="${GPU_ID:-3}"

# Validate config exists (.yml or .yaml)
if [[ ! -f "task_config/${CONFIG}.yml" && ! -f "task_config/${CONFIG}.yaml" ]]; then
  echo "ERROR: task_config/${CONFIG}.yml or task_config/${CONFIG}.yaml not found" >&2
  exit 1
fi

# Collect the 50 task names from envs/*.py
mapfile -t TASKS < <(
python - <<'PY'
import pathlib
root = pathlib.Path('envs')
ignore = {'__init__', '_base_task', '_GLOBAL_CONFIGS'}
tasks = sorted([p.stem for p in root.glob('*.py') if p.stem not in ignore and not p.stem.startswith('_')])
for t in tasks:
    print(t)
PY
)

echo "Total tasks: ${#TASKS[@]}"

for task in "${TASKS[@]}"; do
  echo "[GPU ${GPU_ID}] [${ROBOT}] collecting: ${task} (${CONFIG})"

  # collect raw
  if ! bash collect_data.sh "$task" "$CONFIG" "$GPU_ID"; then
    echo "WARN: collect_data failed for ${task}, skipping to next task"
    continue
  fi

  # extract
  raw_dir="${RAW_ROOT}/${task}/${CONFIG}"
  episode_dir="${raw_dir}/data"
  out_dir="${SAVE_ROOT}/${task}/${ROBOT}"

  if [[ ! -d "$episode_dir" ]]; then
    echo "WARN: episode dir not found for ${task}: $episode_dir. Skipping."
    continue
  fi

  mkdir -p "$out_dir"

  echo "[GPU ${GPU_ID}] [${ROBOT}] extracting -> ${out_dir}"
  python read_hdf5_advanced.py "$episode_dir" --save-camera-data --output-dir "$out_dir" --overwrite

  # delete raw to save space
  echo "[GPU ${GPU_ID}] [${ROBOT}] deleting raw: ${raw_dir}"
  rm -rf "$raw_dir"
done

echo "All done. Extracted data saved under: ${SAVE_ROOT}/${ROBOT}"
