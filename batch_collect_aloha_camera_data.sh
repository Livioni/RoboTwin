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
  --bulk-extract      Use old behavior: extract all raw episodes after collection
  --no-scan-output    Do not pre-scan extracted episodes for corrupt/incomplete files
  -h, --help          Show this help

Environment overrides:
  CONFIG, GPU_ID, SAVE_ROOT, RAW_ROOT, VIDEO_FPS, KEEP_RAW, OVERWRITE, CAMERAS, INCREMENTAL_EXTRACT,
  SCAN_EXTRACTED_OUTPUT

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
INCREMENTAL_EXTRACT="${INCREMENTAL_EXTRACT:-1}"
SCAN_EXTRACTED_OUTPUT="${SCAN_EXTRACTED_OUTPUT:-1}"

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
    --bulk-extract)
      INCREMENTAL_EXTRACT=0
      shift
      ;;
    --no-scan-output)
      SCAN_EXTRACTED_OUTPUT=0
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
ACTIVE_CHILD_PID=""

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

cancel_active_task() {
  trap - INT TERM
  echo
  echo "[GPU ${GPU_ID}] interrupted; stopping current task and pending tasks..." >&2

  if [[ -n "$ACTIVE_CHILD_PID" ]]; then
    kill_process_tree "$ACTIVE_CHILD_PID" TERM
    sleep 2
    if kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
      kill_process_tree "$ACTIVE_CHILD_PID" KILL
    fi
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
  fi

  echo "[GPU ${GPU_ID}] stopped. No further tasks will be started." >&2
  exit 130
}

run_collect_data() {
  bash collect_data.sh "$@" &
  ACTIVE_CHILD_PID="$!"
  wait "$ACTIVE_CHILD_PID"
  local status="$?"
  ACTIVE_CHILD_PID=""
  return "$status"
}

trap cancel_active_task INT TERM

episode_output_complete() {
  local episode_output_dir="$1"
  [[ -f "${episode_output_dir}/meta.json" \
    && -d "${episode_output_dir}/camera_data" \
    && -d "${episode_output_dir}/robot_data" ]]
}

has_complete_extracted_episode() {
  local out_dir="$1"
  local episode_output_dir

  if [[ ! -d "$out_dir" ]]; then
    return 1
  fi

  while IFS= read -r episode_output_dir; do
    if episode_output_complete "$episode_output_dir"; then
      return 0
    fi
  done < <(find "$out_dir" -mindepth 1 -maxdepth 1 -type d -name 'episode*' | sort -V)

  return 1
}

scan_extracted_episodes() {
  local out_dir="$1"
  local raw_dir="$2"
  local broken_episode_dirs=()
  local broken_episode_dir

  CORRUPT_EPISODES_REMOVED=0

  if [[ "$SCAN_EXTRACTED_OUTPUT" != "1" || ! -d "$out_dir" ]]; then
    return 0
  fi

  echo "[GPU ${GPU_ID}] scanning extracted episodes: ${out_dir}"
  mapfile -t broken_episode_dirs < <(
    python - "$out_dir" "$raw_dir" "$CAMERAS" <<'PY'
import json
import os
import pathlib
import re
import sys

out_dir = pathlib.Path(sys.argv[1])
raw_dir = pathlib.Path(sys.argv[2])
cameras = sys.argv[3].split()

robot_files = {
    "left_arm_joint_action.npy",
    "left_endpose.npy",
    "left_endpose_gripper.npy",
    "left_gripper_action.npy",
    "right_arm_joint_action.npy",
    "right_endpose.npy",
    "right_endpose_gripper.npy",
    "right_gripper_action.npy",
}


def episode_index(path):
    match = re.fullmatch(r"episode([0-9]+)", path.name)
    return int(match.group(1)) if match else -1


def bad_png(path):
    try:
        if path.stat().st_size < 20:
            return True
        with path.open("rb") as file:
            head = file.read(8)
            file.seek(-12, os.SEEK_END)
            tail = file.read(12)
        return head != b"\x89PNG\r\n\x1a\n" or b"IEND" not in tail
    except OSError:
        return True


def bad_npy(path):
    try:
        if path.stat().st_size <= 6:
            return True
        with path.open("rb") as file:
            return file.read(6) != b"\x93NUMPY"
    except OSError:
        return True


def frame_files(path, pattern):
    return sorted(path.glob(pattern))


def check_episode(ep_dir):
    issues = []

    meta_path = ep_dir / "meta.json"
    if not meta_path.is_file():
        issues.append("missing meta.json")
    else:
        try:
            json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"bad meta.json: {exc}")

    camera_data = ep_dir / "camera_data"
    robot_data = ep_dir / "robot_data"
    if not camera_data.is_dir():
        issues.append("missing camera_data")
    if not robot_data.is_dir():
        issues.append("missing robot_data")

    for camera in cameras:
        counts = {}
        for subdir, pattern, checker in (
            ("images", "frame_*.png", bad_png),
            ("depths", "frame_*.png", bad_png),
            ("extrinsics", "frame_*.npy", bad_npy),
        ):
            cam_dir = camera_data / subdir / camera
            if not cam_dir.is_dir():
                issues.append(f"missing {subdir}/{camera}")
                continue
            files = frame_files(cam_dir, pattern)
            counts[subdir] = len(files)
            if not files:
                issues.append(f"empty {subdir}/{camera}")
                continue
            for file_path in files:
                if checker(file_path):
                    issues.append(f"bad {subdir}/{camera}/{file_path.name}")
                    break

        if counts and len(set(counts.values())) != 1:
            issues.append(f"frame count mismatch {camera}: {counts}")

        intrinsic_path = camera_data / "intrinsics" / camera / "intrinsic.npy"
        if not intrinsic_path.is_file() or bad_npy(intrinsic_path):
            issues.append(f"missing/bad intrinsics/{camera}/intrinsic.npy")

        video_path = camera_data / "videos" / camera / "video.mp4"
        if not video_path.is_file() or video_path.stat().st_size == 0:
            issues.append(f"missing/empty videos/{camera}/video.mp4")

    if robot_data.is_dir():
        for file_name in sorted(robot_files):
            file_path = robot_data / file_name
            if not file_path.is_file() or bad_npy(file_path):
                issues.append(f"missing/bad robot_data/{file_name}")

    return issues


episode_dirs = sorted(
    [
        path
        for path in out_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"episode[0-9]+", path.name)
    ],
    key=episode_index,
)

for ep_dir in episode_dirs:
    issues = check_episode(ep_dir)
    if not issues:
        continue

    idx = episode_index(ep_dir)
    raw_hdf5 = raw_dir / "data" / f"episode{idx}.hdf5"
    raw_h5 = raw_dir / "data" / f"episode{idx}.h5"
    raw_traj = raw_dir / "_traj_data" / f"episode{idx}.pkl"
    seed_state = raw_dir / "seed.txt"
    can_regenerate = raw_hdf5.exists() or raw_h5.exists() or raw_traj.exists() or seed_state.exists()

    print(
        f"WARN: corrupt extracted episode detected: {ep_dir} "
        f"({'; '.join(issues[:6])})",
        file=sys.stderr,
    )
    if len(issues) > 6:
        print(f"WARN: ... {len(issues) - 6} more issue(s)", file=sys.stderr)

    print(str(ep_dir))
    if not can_regenerate:
        print(
            f"WARN: no raw hdf5/traj/seed state found for {ep_dir}; "
            "the task will regenerate it from task config if collection runs.",
            file=sys.stderr,
        )
PY
  )

  if [[ ${#broken_episode_dirs[@]} -eq 0 ]]; then
    echo "[GPU ${GPU_ID}] scan complete: no corrupt extracted episodes found in ${out_dir}"
    return 0
  fi

  for broken_episode_dir in "${broken_episode_dirs[@]}"; do
    echo "[GPU ${GPU_ID}] removing corrupt extracted episode before regeneration: ${broken_episode_dir}"
    rm -rf "$broken_episode_dir"
    CORRUPT_EPISODES_REMOVED=1
  done
}

delete_raw_episode_files() {
  local raw_dir="$1"
  local episode_idx="$2"
  local hdf5_path="$3"

  if [[ "$KEEP_RAW" == "1" ]]; then
    echo "[GPU ${GPU_ID}] keeping raw episode: ${hdf5_path}"
    return
  fi

  echo "[GPU ${GPU_ID}] deleting raw episode files for episode${episode_idx}"
  rm -f "$hdf5_path"
  rm -f "${raw_dir}/video/episode${episode_idx}.mp4"
  rm -f "${raw_dir}/_traj_data/episode${episode_idx}.pkl"
}

extract_one_episode() {
  local hdf5_path="$1"
  local out_dir="$2"
  local raw_dir="$3"
  local base_name episode_idx episode_output_dir

  base_name="$(basename "$hdf5_path")"
  if [[ "$base_name" =~ ^episode([0-9]+)\.h(df5|5)$ ]]; then
    episode_idx="${BASH_REMATCH[1]}"
  else
    echo "WARN: cannot parse episode index from ${hdf5_path}, keeping raw." >&2
    return 1
  fi

  episode_output_dir="${out_dir}/episode${episode_idx}"
  if [[ -d "$episode_output_dir" && -n "$(find "$episode_output_dir" -mindepth 1 -maxdepth 1 -print -quit)" && "$OVERWRITE" != "1" ]]; then
    if episode_output_complete "$episode_output_dir"; then
      echo "[GPU ${GPU_ID}] episode${episode_idx} already extracted: ${episode_output_dir}"
      delete_raw_episode_files "$raw_dir" "$episode_idx" "$hdf5_path"
      return 0
    fi
    echo "[GPU ${GPU_ID}] removing incomplete extracted output before retry: ${episode_output_dir}"
    rm -rf "$episode_output_dir"
  fi

  mkdir -p "$out_dir"

  extract_args=(
    "$hdf5_path"
    --save-camera-data
    --output-dir "$episode_output_dir"
    --no-auto-select-moving-arm
    --cameras "${CAMERA_LIST[@]}"
    --save-videos
    --video-fps "$VIDEO_FPS"
    --save-both-arms
  )

  if [[ "$OVERWRITE" == "1" ]]; then
    extract_args+=(--overwrite)
  fi

  echo "[GPU ${GPU_ID}] extracting episode${episode_idx}: ${hdf5_path} -> ${episode_output_dir}"
  if ! python read_hdf5_advanced.py "${extract_args[@]}"; then
    echo "WARN: extraction failed for ${hdf5_path}, raw episode kept." >&2
    return 1
  fi

  if ! episode_output_complete "$episode_output_dir"; then
    echo "WARN: extraction output is incomplete, raw episode kept: ${episode_output_dir}" >&2
    return 1
  fi

  delete_raw_episode_files "$raw_dir" "$episode_idx" "$hdf5_path"
}

convert_existing_episodes() {
  local episode_dir="$1"
  local out_dir="$2"
  local raw_dir="$3"
  local hdf5_files=()
  local hdf5_path

  if [[ ! -d "$episode_dir" ]]; then
    return 0
  fi

  mapfile -t hdf5_files < <(
    find "$episode_dir" -maxdepth 1 -type f \( -name 'episode*.hdf5' -o -name 'episode*.h5' \) | sort -V
  )

  if [[ ${#hdf5_files[@]} -eq 0 ]]; then
    return 0
  fi

  echo "[GPU ${GPU_ID}] converting ${#hdf5_files[@]} existing raw episode(s) before collection"
  for hdf5_path in "${hdf5_files[@]}"; do
    if ! extract_one_episode "$hdf5_path" "$out_dir" "$raw_dir"; then
      return 1
    fi
  done
}

echo "Task config: ${CONFIG}"
echo "GPU id: ${GPU_ID}"
echo "Tasks: ${#TASKS[@]}"
echo "Cameras: ${CAMERA_LIST[*]}"
echo "Output root: ${SAVE_ROOT}"
echo "Incremental extract: ${INCREMENTAL_EXTRACT}"

for task in "${TASKS[@]}"; do
  out_dir="${SAVE_ROOT}/${task}"
  raw_dir="${RAW_ROOT}/${task}/${CONFIG}"
  episode_dir="${raw_dir}/data"

  if [[ "$INCREMENTAL_EXTRACT" == "1" ]]; then
    mkdir -p "$out_dir"

    CORRUPT_EPISODES_REMOVED=0
    scan_extracted_episodes "$out_dir" "$raw_dir"

    if ! convert_existing_episodes "$episode_dir" "$out_dir" "$raw_dir"; then
      echo "WARN: existing raw episode conversion failed for ${task}; new collection skipped." >&2
      continue
    fi

    if [[ "$CORRUPT_EPISODES_REMOVED" != "1" && ! -f "${raw_dir}/seed.txt" && "$OVERWRITE" != "1" ]] && has_complete_extracted_episode "$out_dir"; then
      echo "[GPU ${GPU_ID}] skipping ${task}: extracted episodes exist and no raw seed state is available"
      continue
    fi

    echo "[GPU ${GPU_ID}] collecting ${task} (${CONFIG}) with per-episode extraction"
    if ! (
      export ROBOTWIN_EXTRACT_EACH_EPISODE=1
      export ROBOTWIN_EXTRACT_OUTPUT_DIR="$out_dir"
      export ROBOTWIN_EXTRACT_CAMERAS="$CAMERAS"
      export ROBOTWIN_EXTRACT_VIDEO_FPS="$VIDEO_FPS"
      export ROBOTWIN_EXTRACT_KEEP_RAW="$KEEP_RAW"
      export ROBOTWIN_EXTRACT_OVERWRITE="$OVERWRITE"
      run_collect_data "$task" "$CONFIG" "$GPU_ID"
    ); then
      echo "WARN: collect_data failed for ${task}, raw data kept for failed/unconverted episodes." >&2
      continue
    fi
    continue
  fi

  if [[ -d "$out_dir" && "$OVERWRITE" != "1" ]]; then
    echo "[GPU ${GPU_ID}] skipping ${task}: output already exists (${out_dir})"
    continue
  fi

  echo "[GPU ${GPU_ID}] collecting ${task} (${CONFIG})"
  if ! run_collect_data "$task" "$CONFIG" "$GPU_ID"; then
    echo "WARN: collect_data failed for ${task}, skipping." >&2
    continue
  fi

  if [[ ! -d "$episode_dir" ]]; then
    echo "WARN: episode dir not found for ${task}: ${episode_dir}, skipping." >&2
    continue
  fi

  mkdir -p "$out_dir"

  if ! convert_existing_episodes "$episode_dir" "$out_dir" "$raw_dir"; then
    echo "WARN: extraction failed for ${task}, raw data kept at ${raw_dir}." >&2
    continue
  fi

  if [[ "$KEEP_RAW" == "1" ]]; then
    echo "[GPU ${GPU_ID}] keeping raw data: ${raw_dir}"
  else
    echo "[GPU ${GPU_ID}] raw episode files converted and removed from: ${raw_dir}"
  fi
done

echo "All done. Extracted data saved under: ${SAVE_ROOT}"
