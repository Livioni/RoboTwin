#!/usr/bin/env bash
set -euo pipefail

# Collect RoboTwin data and extract the ALOHA head + wrist + third-view camera streams.
#
# Default behavior:
#   - task config: task_config/custom_aloha.yml
#   - cameras: head_camera, left_camera, right_camera, third_view
#   - extracted data: RGB PNG frames, MP4 videos, depth PNG, intrinsic/extrinsic/cam2world NPY
#
# Examples:
#   bash batch_collect_aloha_camera_data.sh
#   bash batch_collect_aloha_camera_data.sh beat_block_hammer
#   bash batch_collect_aloha_camera_data.sh --gpu 0 --config custom_aloha beat_block_hammer
#   SAVE_ROOT=./datasets/aloha-3cam KEEP_RAW=1 bash batch_collect_aloha_camera_data.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash batch_collect_aloha_camera_data.sh [options] [task_name ...]

Options:
  --config NAME       Task config basename under task_config/ (default: custom_aloha)
  --gpu ID            CUDA_VISIBLE_DEVICES id passed to collect_data.sh (default: 0)
  --save-root DIR     Final extracted dataset root (default: ./datasets/aloha-camera-data)
  --raw-root DIR      Raw RoboTwin data root (default: ./data)
  --video-fps FPS     FPS for exported MP4 videos (default: 30)
  --overwrite         Re-extract even if a task output directory already exists
  --keep-raw          Keep raw data/<task>/<config> after extraction
  -h, --help          Show this help

Environment overrides:
  CONFIG, GPU_ID, SAVE_ROOT, RAW_ROOT, VIDEO_FPS, KEEP_RAW, OVERWRITE, CAMERAS

If no task_name is given, all task files under envs/*.py are collected.
CAMERAS defaults to: "head_camera left_camera right_camera third_view".
EOF
}

CONFIG="${CONFIG:-custom_aloha}"
GPU_ID="${GPU_ID:-0}"
SAVE_ROOT="${SAVE_ROOT:-./datasets/robotwin_aloha}"
RAW_ROOT="${RAW_ROOT:-./data}"
VIDEO_FPS="${VIDEO_FPS:-30}"
KEEP_RAW="${KEEP_RAW:-0}"
OVERWRITE="${OVERWRITE:-0}"
CAMERAS="${CAMERAS:-head_camera left_camera right_camera third_view}"

TASKS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
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

if [[ ! -f "task_config/${CONFIG}.yml" && ! -f "task_config/${CONFIG}.yaml" ]]; then
  echo "ERROR: task_config/${CONFIG}.yml or task_config/${CONFIG}.yaml not found" >&2
  exit 1
fi

if [[ ! -f "read_hdf5_advanced.py" ]]; then
  echo "ERROR: read_hdf5_advanced.py not found under ${ROOT_DIR}" >&2
  exit 1
fi

./script/.update_path.sh > /dev/null 2>&1 || true

mkdir -p "$SAVE_ROOT"

if ! python - <<'PY'
import sys
import warnings

try:
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    import pkg_resources  # noqa: F401
    import sapien.core as sapien  # noqa: F401
    import h5py  # noqa: F401
    import cv2  # noqa: F401
    from PIL import Image  # noqa: F401
except ModuleNotFoundError as exc:
    print(f"ERROR: missing Python module: {exc.name}", file=sys.stderr)
    if exc.name == "pkg_resources":
        print(
            "Hint: pkg_resources is provided by older setuptools. "
            "Try: python -m pip install 'setuptools<81'",
            file=sys.stderr,
        )
    sys.exit(1)
except Exception as exc:
    print(f"ERROR: Python dependency preflight failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY
then
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

read -r -a CAMERA_LIST <<< "$CAMERAS"

echo "Task config: ${CONFIG}"
echo "GPU id: ${GPU_ID}"
echo "Tasks: ${#TASKS[@]}"
echo "Cameras: ${CAMERA_LIST[*]}"
echo "Output root: ${SAVE_ROOT}"

for task in "${TASKS[@]}"; do
  out_dir="${SAVE_ROOT}/${task}"
  raw_dir="${RAW_ROOT}/${task}/${CONFIG}"
  episode_dir="${raw_dir}/data"

  if [[ -d "$out_dir" && "$OVERWRITE" != "1" ]]; then
    echo "[GPU ${GPU_ID}] skipping ${task}: output already exists (${out_dir})"
    continue
  fi

  echo "[GPU ${GPU_ID}] collecting ${task} (${CONFIG})"
  if ! bash collect_data.sh "$task" "$CONFIG" "$GPU_ID"; then
    echo "WARN: collect_data failed for ${task}, skipping." >&2
    continue
  fi

  if [[ ! -d "$episode_dir" ]]; then
    echo "WARN: episode dir not found for ${task}: ${episode_dir}, skipping." >&2
    continue
  fi

  mkdir -p "$out_dir"

  extract_args=(
    "$episode_dir"
    --save-camera-data
    --output-dir "$out_dir"
    --no-auto-select-moving-arm
    --cameras "${CAMERA_LIST[@]}"
    --save-videos
    --video-fps "$VIDEO_FPS"
    --save-both-arms
  )

  if [[ "$OVERWRITE" == "1" ]]; then
    extract_args+=(--overwrite)
  fi

  echo "[GPU ${GPU_ID}] extracting ${task} -> ${out_dir}"
  if ! python read_hdf5_advanced.py "${extract_args[@]}"; then
    echo "WARN: extraction failed for ${task}, raw data kept at ${raw_dir}." >&2
    continue
  fi

  if [[ "$KEEP_RAW" == "1" ]]; then
    echo "[GPU ${GPU_ID}] keeping raw data: ${raw_dir}"
  else
    echo "[GPU ${GPU_ID}] deleting raw data: ${raw_dir}"
    rm -rf "$raw_dir"
  fi
done

echo "All done. Extracted data saved under: ${SAVE_ROOT}"
