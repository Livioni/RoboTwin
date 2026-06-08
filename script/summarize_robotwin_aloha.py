#!/usr/bin/env python3
"""Summarize exported RoboTwin ALOHA dataset episodes.

Default behavior is fast: read each episode's meta.json and, if needed,
infer frame count from robot_data/*.npy headers. Use --check-camera-frames to
also count image files for a camera stream, which can be much slower.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - script still works from meta.json
    np = None


EPISODE_RE = re.compile(r"^episode(\d+)$")
FRAME_RE = re.compile(r"frame_(\d+)\.[A-Za-z0-9]+$")
DEFAULT_FPS = 30.0
NPY_FRAME_CANDIDATES = (
    "left_gripper_action.npy",
    "right_gripper_action.npy",
    "left_arm_joint_action.npy",
    "right_arm_joint_action.npy",
    "left_endpose.npy",
    "right_endpose.npy",
)


@dataclass
class EpisodeSummary:
    task: str
    episode: str
    episode_index: int
    frames: int | None
    fps: float | None
    duration_sec: float | None
    success: bool | None
    result: str | None
    frame_source: str
    camera_frame_count: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class TaskSummary:
    task: str
    episodes: list[EpisodeSummary]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def npy_first_dim(path: Path) -> int | None:
    if np is None or not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            elif version == (3, 0):
                shape, _, _ = np.lib.format.read_array_header_3_0(f)
            else:
                return None
        return int(shape[0]) if shape else None
    except Exception:
        return None


def infer_frames_from_robot_data(episode_dir: Path) -> tuple[int | None, str]:
    robot_data = episode_dir / "robot_data"
    for name in NPY_FRAME_CANDIDATES:
        count = npy_first_dim(robot_data / name)
        if count is not None:
            return count, f"robot_data/{name}"

    if np is not None and robot_data.is_dir():
        for npy_path in sorted(robot_data.glob("*.npy")):
            count = npy_first_dim(npy_path)
            if count is not None:
                return count, f"robot_data/{npy_path.name}"

    return None, "missing"


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "success", "succeeded", "1", "yes"}:
            return True
        if lowered in {"false", "fail", "failed", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def episode_success(episode_dir: Path, meta: dict[str, Any] | None) -> tuple[bool | None, str | None]:
    result = None
    success = None

    if meta:
        if "result" in meta:
            result = str(meta["result"])
        success = normalize_bool(meta.get("success"))
        if success is None and result is not None:
            success = normalize_bool(result)

    if success is None:
        if (episode_dir / "success.txt").exists():
            success = True
            result = result or "success"
        elif (episode_dir / "fail.txt").exists():
            success = False
            result = result or "fail"

    return success, result


def count_camera_frames(episode_dir: Path, preferred_camera: str | None) -> tuple[int | None, str | None]:
    rgb_root = episode_dir / "camera_data" / "rgbs"
    depth_root = episode_dir / "camera_data" / "depths"
    search_roots = [rgb_root, depth_root]

    camera_dirs: list[Path] = []
    if preferred_camera:
        for root in search_roots:
            camera_dir = root / preferred_camera
            if camera_dir.is_dir():
                camera_dirs.append(camera_dir)

    if not camera_dirs:
        for root in search_roots:
            if root.is_dir():
                camera_dirs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
        if not camera_dirs:
            return None, None

    camera_dir = camera_dirs[0]
    count = 0
    max_index = -1
    for path in camera_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_RE.match(path.name)
        if not match:
            continue
        count += 1
        max_index = max(max_index, int(match.group(1)))

    if count == 0:
        return None, str(camera_dir.relative_to(episode_dir))
    return count if max_index + 1 == count else max_index + 1, str(camera_dir.relative_to(episode_dir))


def summarize_episode(
    task_name: str,
    episode_dir: Path,
    fallback_fps: float,
    check_camera_frames: bool,
    camera: str | None,
) -> EpisodeSummary:
    match = EPISODE_RE.match(episode_dir.name)
    episode_index = int(match.group(1)) if match else -1
    meta = load_json(episode_dir / "meta.json")
    warnings: list[str] = []

    frames = None
    frame_source = "missing"
    fps = fallback_fps

    if meta is None:
        warnings.append("missing_or_invalid_meta")
    else:
        if isinstance(meta.get("total_frames"), int):
            frames = int(meta["total_frames"])
            frame_source = "meta.total_frames"
        elif isinstance(meta.get("total_frames"), float):
            frames = int(meta["total_frames"])
            frame_source = "meta.total_frames"

        try:
            fps = float(meta.get("video_fps", fps))
        except (TypeError, ValueError):
            warnings.append("invalid_meta_video_fps")

    if frames is None:
        frames, frame_source = infer_frames_from_robot_data(episode_dir)
        if frames is None:
            warnings.append("missing_frame_count")

    success, result = episode_success(episode_dir, meta)

    camera_frame_count = None
    if check_camera_frames:
        preferred_camera = camera
        if preferred_camera is None and meta:
            selected = meta.get("selected_cameras") or meta.get("requested_cameras")
            if isinstance(selected, list) and selected:
                preferred_camera = str(selected[0])

        camera_frame_count, camera_source = count_camera_frames(episode_dir, preferred_camera)
        if camera_frame_count is None:
            warnings.append("missing_camera_frames")
        elif frames is not None and camera_frame_count != frames:
            warnings.append(f"camera_frame_mismatch:{camera_source}={camera_frame_count}")

    duration_sec = None
    if frames is not None and fps and fps > 0:
        duration_sec = frames / fps
    elif frames is not None:
        warnings.append("missing_duration_fps")

    return EpisodeSummary(
        task=task_name,
        episode=episode_dir.name,
        episode_index=episode_index,
        frames=frames,
        fps=fps,
        duration_sec=duration_sec,
        success=success,
        result=result,
        frame_source=frame_source,
        camera_frame_count=camera_frame_count,
        warnings=warnings,
    )


def discover_tasks(root: Path) -> list[Path]:
    tasks = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        has_episode = any(p.is_dir() and EPISODE_RE.match(p.name) for p in path.iterdir())
        if has_episode:
            tasks.append(path)
    return tasks


def discover_episodes(task_dir: Path) -> list[Path]:
    episodes = [p for p in task_dir.iterdir() if p.is_dir() and EPISODE_RE.match(p.name)]
    return sorted(episodes, key=lambda p: int(EPISODE_RE.match(p.name).group(1)))  # type: ignore[union-attr]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def task_row(summary: TaskSummary) -> dict[str, Any]:
    episodes = summary.episodes
    frame_values = [e.frames for e in episodes if e.frames is not None]
    duration_values = [e.duration_sec for e in episodes if e.duration_sec is not None]
    success_count = sum(1 for e in episodes if e.success is True)
    fail_count = sum(1 for e in episodes if e.success is False)
    unknown_count = len(episodes) - success_count - fail_count
    warnings = sum(len(e.warnings) for e in episodes)

    return {
        "task": summary.task,
        "episodes": len(episodes),
        "success": success_count,
        "fail": fail_count,
        "unknown": unknown_count,
        "avg_frames": mean([float(v) for v in frame_values]),
        "min_frames": min(frame_values) if frame_values else None,
        "max_frames": max(frame_values) if frame_values else None,
        "total_frames": sum(frame_values),
        "avg_duration_sec": mean([float(v) for v in duration_values]),
        "total_duration_sec": sum(duration_values),
        "warnings": warnings,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "task",
        "episodes",
        "success",
        "fail",
        "unknown",
        "avg_frames",
        "min_frames",
        "max_frames",
        "total_frames",
        "avg_sec",
        "total_sec",
        "warnings",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["task"],
                str(row["episodes"]),
                str(row["success"]),
                str(row["fail"]),
                str(row["unknown"]),
                fmt(row["avg_frames"]),
                fmt(row["min_frames"]),
                fmt(row["max_frames"]),
                str(row["total_frames"]),
                fmt(row["avg_duration_sec"]),
                fmt(row["total_duration_sec"]),
                str(row["warnings"]),
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in table_rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "episodes",
        "success",
        "fail",
        "unknown",
        "avg_frames",
        "min_frames",
        "max_frames",
        "total_frames",
        "avg_duration_sec",
        "total_duration_sec",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, summaries: list[TaskSummary], rows: list[dict[str, Any]]) -> None:
    payload = {
        "tasks": rows,
        "episodes": [
            {
                "task": e.task,
                "episode": e.episode,
                "frames": e.frames,
                "fps": e.fps,
                "duration_sec": e.duration_sec,
                "success": e.success,
                "result": e.result,
                "frame_source": e.frame_source,
                "camera_frame_count": e.camera_frame_count,
                "warnings": e.warnings,
            }
            for summary in summaries
            for e in summary.episodes
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def print_episode_rows(summaries: list[TaskSummary]) -> None:
    print()
    print("Episodes with warnings or unknown frame count:")
    any_printed = False
    for summary in summaries:
        for episode in summary.episodes:
            if episode.warnings or episode.frames is None:
                any_printed = True
                warning_text = ",".join(episode.warnings) if episode.warnings else "-"
                print(
                    f"{episode.task}/{episode.episode}: "
                    f"frames={fmt(episode.frames)} fps={fmt(episode.fps)} "
                    f"duration_sec={fmt(episode.duration_sec)} warnings={warning_text}"
                )
    if not any_printed:
        print("none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/robotwin_aloha"),
        help="Dataset root, default: datasets/robotwin_aloha",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=DEFAULT_FPS,
        help="FPS used when an episode has no meta video_fps, default: 30",
    )
    parser.add_argument(
        "--check-camera-frames",
        action="store_true",
        help="Count frame_*.png files in a camera stream and compare with metadata.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Camera name to check with --check-camera-frames, e.g. head_camera.",
    )
    parser.add_argument(
        "--show-episodes",
        action="store_true",
        help="Print every episode instead of only warning episodes.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Write per-task CSV summary.")
    parser.add_argument("--json", type=Path, default=None, help="Write JSON summary with task and episode rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    summaries: list[TaskSummary] = []
    for task_dir in discover_tasks(root):
        episodes = [
            summarize_episode(
                task_name=task_dir.name,
                episode_dir=episode_dir,
                fallback_fps=args.fallback_fps,
                check_camera_frames=args.check_camera_frames,
                camera=args.camera,
            )
            for episode_dir in discover_episodes(task_dir)
        ]
        summaries.append(TaskSummary(task=task_dir.name, episodes=episodes))

    rows = [task_row(summary) for summary in summaries]
    total_episodes = sum(row["episodes"] for row in rows)
    total_frames = sum(row["total_frames"] for row in rows)
    total_duration = sum(row["total_duration_sec"] for row in rows)
    all_frame_values = [
        float(episode.frames)
        for summary in summaries
        for episode in summary.episodes
        if episode.frames is not None
    ]

    print(f"Dataset root: {root}")
    print(f"Tasks: {len(rows)}")
    print(f"Episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Average frames/episode: {fmt(mean(all_frame_values))}")
    print(f"Total duration: {fmt(total_duration)} sec ({fmt(total_duration / 3600.0)} hours)")
    print()
    print_table(rows)

    if args.show_episodes:
        print()
        print("Episodes:")
        for summary in summaries:
            for episode in summary.episodes:
                warning_text = ",".join(episode.warnings) if episode.warnings else "-"
                print(
                    f"{episode.task}/{episode.episode}: "
                    f"frames={fmt(episode.frames)} fps={fmt(episode.fps)} "
                    f"duration_sec={fmt(episode.duration_sec)} success={episode.success} "
                    f"source={episode.frame_source} warnings={warning_text}"
                )
    else:
        print_episode_rows(summaries)

    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nWrote CSV: {args.csv}")
    if args.json:
        write_json(args.json, summaries, rows)
        print(f"Wrote JSON: {args.json}")


if __name__ == "__main__":
    main()
