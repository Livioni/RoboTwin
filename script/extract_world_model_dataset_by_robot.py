#!/usr/bin/env python3
"""Extract DiffSynth world_model_data/dataset into robot-grouped folders.

Input layout:
  dataset/<task>/<robot>_clean_50/data/episode*.hdf5

Output layout:
  robotwin_aloha/ori_set/<robot>/<task>/episode*/...

The actual HDF5 decoding is delegated to read_hdf5_advanced.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = Path(
    "/home/wenchaoxu/phs/DiffSynth-Studio/world_model_data/dataset"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/wenchaoxu/phs/DiffSynth-Studio/world_model_data/robotwin_aloha/ori_set"
)


@dataclass(frozen=True)
class RobotSpec:
    source_dir_name: str
    output_name: str


ROBOT_SPECS = (
    RobotSpec("aloha-agilex_clean_50", "aloha-agilex"),
    RobotSpec("arx-x5_clean_50", "arx-x5"),
    RobotSpec("franka_clean_50", "franka"),
    RobotSpec("piper_clean_50", "piper"),
    RobotSpec("ur5_clean_50", "ur5"),
)

HDF5_SUFFIXES = (".hdf5", ".h5")
COMPLETE_MARKER_NAME = ".extract_complete.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use read_hdf5_advanced.py to extract world_model_data/dataset "
            "into robot-grouped task folders."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Source dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--reader",
        type=Path,
        default=repo_root() / "read_hdf5_advanced.py",
        help="Path to read_hdf5_advanced.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=f"Python executable used to run the reader. Default: {sys.executable}",
    )
    parser.add_argument(
        "--robot",
        action="append",
        choices=[spec.output_name for spec in ROBOT_SPECS],
        help="Only extract this robot. Can be passed multiple times. Default: all 5 robots.",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Only extract this task. Can be passed multiple times. Default: all tasks.",
    )
    parser.add_argument(
        "--mode",
        choices=("3d", "all-cameras"),
        default="3d",
        help=(
            "3d extracts head_camera/left_camera/right_camera/third_view and both arms; "
            "all-cameras extracts every available camera. Default: 3d."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to read_hdf5_advanced.py. This disables task-level resume skips.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Do not skip robot/task folders that already look complete. "
            "Episode-level skipping in read_hdf5_advanced.py still applies unless --overwrite is used."
        ),
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Also save per-camera MP4 files through read_hdf5_advanced.py.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="Video FPS when --save-videos is used. Default: 30.",
    )
    parser.add_argument(
        "--save-cam2world",
        action="store_true",
        help="Also save cam2world_gl matrices.",
    )
    parser.add_argument(
        "--save-depth-arrays",
        action="store_true",
        help="Also save raw depth arrays as .npy files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running extraction.",
    )
    return parser.parse_args()


def discover_tasks(dataset_root: Path, requested_tasks: Iterable[str] | None) -> list[str]:
    if requested_tasks:
        return sorted(set(requested_tasks))
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def selected_robot_specs(requested_robots: Iterable[str] | None) -> list[RobotSpec]:
    if not requested_robots:
        return list(ROBOT_SPECS)
    wanted = set(requested_robots)
    return [spec for spec in ROBOT_SPECS if spec.output_name in wanted]


def is_hdf5_path(path: Path) -> bool:
    return path.suffix.lower() in HDF5_SUFFIXES


def collect_source_episodes(source_data_dir: Path) -> list[Path]:
    return sorted(path for path in source_data_dir.rglob("*") if path.is_file() and is_hdf5_path(path))


def output_episode_dir(source_data_dir: Path, episode_path: Path, output_task_dir: Path) -> Path:
    rel = episode_path.relative_to(source_data_dir)
    parts = list(rel.with_suffix("").parts)
    if "data" in parts:
        parts.remove("data")
    return output_task_dir.joinpath(*parts)


def completion_status(
    source_data_dir: Path,
    output_task_dir: Path,
) -> tuple[int, int, list[Path]]:
    source_episodes = collect_source_episodes(source_data_dir)
    missing_outputs = [
        episode_path
        for episode_path in source_episodes
        if not (output_episode_dir(source_data_dir, episode_path, output_task_dir) / "meta.json").is_file()
    ]
    completed = len(source_episodes) - len(missing_outputs)
    return completed, len(source_episodes), missing_outputs


def is_robot_task_complete(source_data_dir: Path, output_task_dir: Path) -> bool:
    completed, total, missing_outputs = completion_status(source_data_dir, output_task_dir)
    return total > 0 and completed == total and not missing_outputs


def write_complete_marker(
    output_task_dir: Path,
    source_data_dir: Path,
    spec: RobotSpec,
    task: str,
    args: argparse.Namespace,
) -> None:
    completed, total, _ = completion_status(source_data_dir, output_task_dir)
    marker = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "robot": spec.output_name,
        "source_dir_name": spec.source_dir_name,
        "task": task,
        "source_data_dir": str(source_data_dir),
        "output_task_dir": str(output_task_dir),
        "episodes_completed": completed,
        "episodes_total": total,
        "mode": args.mode,
        "save_videos": bool(args.save_videos),
        "save_cam2world": bool(args.save_cam2world),
        "save_depth_arrays": bool(args.save_depth_arrays),
    }
    marker_path = output_task_dir / COMPLETE_MARKER_NAME
    with marker_path.open("w", encoding="utf-8") as fp:
        json.dump(marker, fp, ensure_ascii=False, indent=2)


def build_reader_command(
    args: argparse.Namespace,
    source_data_dir: Path,
    output_task_dir: Path,
) -> list[str]:
    cmd = [
        args.python,
        str(args.reader),
        str(source_data_dir),
        "--output-dir",
        str(output_task_dir),
        "--video-fps",
        str(args.video_fps),
    ]

    if args.mode == "3d":
        cmd.append("--save-3d-data")
    else:
        cmd.extend(["--save-camera-data", "--no-auto-select-moving-arm", "--save-both-arms"])

    if args.overwrite:
        cmd.append("--overwrite")
    if args.save_videos:
        cmd.append("--save-videos")
    if args.save_cam2world:
        cmd.append("--save-cam2world")
    if args.save_depth_arrays:
        cmd.append("--save-depth-arrays")

    return cmd


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    reader = args.reader.resolve()
    args.reader = reader

    if not dataset_root.is_dir():
        print(f"ERROR: dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2
    if not reader.is_file():
        print(f"ERROR: reader script does not exist: {reader}", file=sys.stderr)
        return 2

    tasks = discover_tasks(dataset_root, args.task)
    robot_specs = selected_robot_specs(args.robot)

    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")
    print(f"Reader:       {reader}")
    print(f"Tasks:        {len(tasks)}")
    print(f"Robots:       {', '.join(spec.output_name for spec in robot_specs)}")
    print(f"Mode:         {args.mode}")

    planned = 0
    skipped_complete = 0
    skipped_missing = 0
    failed = 0

    for spec in robot_specs:
        robot_output_root = output_root / spec.output_name
        for task in tasks:
            source_data_dir = dataset_root / task / spec.source_dir_name / "data"
            if not source_data_dir.is_dir():
                print(
                    f"WARN: missing source data, skip: {source_data_dir}",
                    file=sys.stderr,
                )
                skipped_missing += 1
                continue

            output_task_dir = robot_output_root / task
            cmd = build_reader_command(args, source_data_dir, output_task_dir)

            if not args.overwrite and not args.no_resume:
                completed, total, _ = completion_status(source_data_dir, output_task_dir)
                if total > 0 and completed == total:
                    skipped_complete += 1
                    print(f"\n[skip] {spec.output_name}/{task}")
                    print(f"  complete: {completed}/{total} episodes")
                    print(f"  dst: {output_task_dir}")
                    continue
                if completed > 0:
                    print(f"\n[resume] {spec.output_name}/{task}")
                    print(f"  existing: {completed}/{total} episodes; continuing missing episodes")

            planned += 1

            print(f"\n[{planned}] {spec.output_name}/{task}")
            print(f"  src: {source_data_dir}")
            print(f"  dst: {output_task_dir}")
            if args.dry_run:
                print("  cmd: " + " ".join(cmd))
                continue

            output_task_dir.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failed += 1
                print(
                    f"ERROR: extraction failed ({result.returncode}): "
                    f"{spec.output_name}/{task}",
                    file=sys.stderr,
                )
                continue

            if is_robot_task_complete(source_data_dir, output_task_dir):
                write_complete_marker(output_task_dir, source_data_dir, spec, task, args)
            else:
                completed, total, missing_outputs = completion_status(source_data_dir, output_task_dir)
                failed += 1
                print(
                    f"ERROR: extraction incomplete after reader success: "
                    f"{spec.output_name}/{task} ({completed}/{total} episodes, "
                    f"missing {len(missing_outputs)})",
                    file=sys.stderr,
                )

    print("\nDone.")
    print(f"Planned extractions: {planned}")
    print(f"Completed robot-task outputs skipped: {skipped_complete}")
    print(f"Missing robot-task sources skipped: {skipped_missing}")
    print(f"Failed extractions: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
