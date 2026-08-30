#!/usr/bin/env python3
"""Export RoboTwin 3D ALOHA trajectories as per-frame 4D training data.

The source is the XPolicyLab HDF5 layout produced by RoboTwin.  Each output
episode has the following structure::

    <output_root>/<task>/episode_XXXXXXX/
        images/<view>/000000.png
        depths/<view>/000000.png
        intrinsics/<view>.npy       # (3, 3)
        extrinsics/<view>.npy       # (T, 3, 4)
        robot_state.npy             # (T, 14)
        robot_action.npy            # (T, 14)
        metadata.json

RGB PNG files use the standard RGB channel interpretation.  Depth PNG files
are lossless uint16 images in millimetres, with zero denoting invalid depth.
By default, extrinsics are OpenCV world-to-camera matrices, which is also the
camera-from-world convention expected by VGGT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np
from tqdm import tqdm


CAMERA_MAP = {
    "cam_left_wrist": "left_view",
    "cam_right_wrist": "right_view",
    "cam_head": "head_view",
    "cam_third_view": "third_views",
}

ROBOT_FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)

EPISODE_PATTERN = re.compile(r"^episode_(\d+)\.hdf5$")


@dataclass(frozen=True)
class ConversionJob:
    source_file: str
    output_dir: str
    task: str
    embodiment: str
    extrinsics_convention: str
    existing: str
    png_compression: int


def _episode_number(path: Path) -> int:
    match = EPISODE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path}")
    return int(match.group(1))


def _decode_rgb(encoded: Any, field_name: str) -> np.ndarray:
    """Decode the repo's JPEG bytes into its canonical logical RGB array."""
    if isinstance(encoded, np.ndarray):
        encoded = encoded.tobytes()
    else:
        encoded = bytes(encoded)
    # Fixed-width HDF5 byte strings are padded with NUL bytes.
    encoded = encoded.rstrip(b"\0")
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode {field_name}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"{field_name} must decode to HxWx3 uint8, got {image.shape} {image.dtype}"
        )
    # RoboTwin deliberately round-trips logical RGB arrays through OpenCV
    # without swapping.  The returned channel values are therefore RGB even
    # though OpenCV normally calls its decoded representation BGR.
    return image


def _write_rgb_png(path: Path, rgb: np.ndarray, compression: int) -> None:
    # Convert logical RGB to the BGR input convention expected by OpenCV's PNG
    # writer.  Standard readers such as Pillow will consequently see RGB.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(
        str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression]
    )
    if not ok:
        raise OSError(f"Failed to write RGB PNG: {path}")


def _write_depth_png(path: Path, depth: np.ndarray, compression: int) -> None:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be HxW, got {depth.shape}")
    if depth.dtype != np.uint16:
        if not np.issubdtype(depth.dtype, np.integer):
            raise ValueError(f"Depth frame must be integer millimetres, got {depth.dtype}")
        if np.any(depth < 0) or np.any(depth > np.iinfo(np.uint16).max):
            raise ValueError("Depth value is outside the uint16 range")
        depth = depth.astype(np.uint16)
    ok = cv2.imwrite(
        str(path), depth, [cv2.IMWRITE_PNG_COMPRESSION, compression]
    )
    if not ok:
        raise OSError(f"Failed to write depth PNG: {path}")


def _matrix_sequence(
    dataset: h5py.Dataset, shape: tuple[int, int], num_frames: int, field_name: str
) -> np.ndarray:
    values = np.asarray(dataset[()], dtype=np.float32)
    if values.shape == shape:
        values = np.repeat(values[None], num_frames, axis=0)
    expected_shape = (num_frames, *shape)
    if values.shape != expected_shape:
        raise ValueError(
            f"{field_name} must have shape {shape} or {expected_shape}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} contains NaN or infinity")
    return values


def _constant_intrinsic(camera: h5py.Group, num_frames: int, name: str) -> np.ndarray:
    if "intrinsic_matrix" not in camera:
        raise ValueError(f"{name} is missing intrinsic_matrix")
    values = _matrix_sequence(
        camera["intrinsic_matrix"], (3, 3), num_frames, f"{name}.intrinsic_matrix"
    )
    if not np.allclose(values, values[0], rtol=1e-6, atol=1e-6):
        max_delta = float(np.max(np.abs(values - values[0])))
        raise ValueError(
            f"{name}.intrinsic_matrix is not constant (max delta {max_delta:.3g}); "
            "a single per-view intrinsic cannot represent this episode"
        )
    return values[0].astype(np.float32, copy=True)


def _extrinsics(
    camera: h5py.Group, num_frames: int, name: str, convention: str
) -> np.ndarray:
    dataset_name = (
        "extrinsic_matrix"
        if convention == "world_to_camera"
        else "camera_pose_matrix"
    )
    if dataset_name not in camera:
        if convention == "camera_to_world" and "extrinsic_matrix" in camera:
            world_to_camera = _matrix_sequence(
                camera["extrinsic_matrix"],
                (4, 4),
                num_frames,
                f"{name}.extrinsic_matrix",
            )
            try:
                values = np.linalg.inv(world_to_camera).astype(np.float32)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{name}.extrinsic_matrix is singular") from exc
        else:
            raise ValueError(f"{name} is missing {dataset_name}")
    else:
        values = _matrix_sequence(
            camera[dataset_name], (4, 4), num_frames, f"{name}.{dataset_name}"
        )

    expected_last_row = np.asarray([0, 0, 0, 1], dtype=np.float32)
    if not np.allclose(values[:, 3, :], expected_last_row, rtol=0, atol=1e-5):
        raise ValueError(f"{name}.{dataset_name} has an invalid homogeneous row")
    return values[:, :3, :4].astype(np.float32, copy=True)


def _robot_vector(
    root: h5py.File, group_name: str, num_frames: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if group_name not in root:
        raise ValueError(f"Source episode is missing {group_name}/")
    group = root[group_name]
    parts: list[np.ndarray] = []
    schema: list[dict[str, Any]] = []
    offset = 0
    for field in ROBOT_FIELDS:
        if field not in group:
            raise ValueError(f"Source episode is missing {group_name}/{field}")
        values = np.asarray(group[field][()], dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != num_frames:
            raise ValueError(
                f"{group_name}/{field} must have shape (T, N) with T={num_frames}, "
                f"got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{group_name}/{field} contains NaN or infinity")
        parts.append(values)
        next_offset = offset + values.shape[1]
        schema.append({"name": field, "start": offset, "end": next_offset})
        offset = next_offset
    return np.concatenate(parts, axis=1), schema


def _scalar_value(root: h5py.File, path: str) -> Any | None:
    if path not in root:
        return None
    value = root[path][()]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_instructions(value: Any | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
        return [str(decoded)]
    return [str(value)]


def _prepare_destination(output_dir: Path, existing: str) -> bool:
    """Return False when an existing output should be skipped."""
    if not output_dir.exists():
        return True
    if existing == "skip":
        return False
    if existing == "error":
        raise FileExistsError(
            f"Output episode already exists: {output_dir}. "
            "Use --existing skip or --existing overwrite."
        )
    return True


def convert_episode(job: ConversionJob) -> dict[str, Any]:
    source_file = Path(job.source_file)
    output_dir = Path(job.output_dir)
    if not _prepare_destination(output_dir, job.existing):
        return {"status": "skipped", "source": str(source_file), "output": str(output_dir)}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / (
        f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary_dir.mkdir()

    try:
        with h5py.File(source_file, "r") as root:
            if "vision" not in root:
                raise ValueError("Source episode is missing vision/")
            vision = root["vision"]

            missing_cameras = [name for name in CAMERA_MAP if name not in vision]
            if missing_cameras:
                raise ValueError(
                    "Source episode is missing required cameras: "
                    + ", ".join(missing_cameras)
                )

            first_camera = vision[next(iter(CAMERA_MAP))]
            if "colors" not in first_camera:
                raise ValueError("Source episode is missing camera color frames")
            num_frames = len(first_camera["colors"])
            if num_frames == 0:
                raise ValueError("Source episode has no frames")

            state, robot_schema = _robot_vector(root, "state", num_frames)
            action, action_schema = _robot_vector(root, "action", num_frames)
            if robot_schema != action_schema:
                raise ValueError("State and action vector schemas differ")
            np.save(temporary_dir / "robot_state.npy", state, allow_pickle=False)
            np.save(temporary_dir / "robot_action.npy", action, allow_pickle=False)

            image_hw: list[int] | None = None
            for source_camera_name, output_view_name in CAMERA_MAP.items():
                camera = vision[source_camera_name]
                for field in ("colors", "depths"):
                    if field not in camera:
                        raise ValueError(f"{source_camera_name} is missing {field}")
                    if len(camera[field]) != num_frames:
                        raise ValueError(
                            f"{source_camera_name}/{field} has {len(camera[field])} "
                            f"frames, expected {num_frames}"
                        )

                intrinsic = _constant_intrinsic(
                    camera, num_frames, source_camera_name
                )
                extrinsic = _extrinsics(
                    camera,
                    num_frames,
                    source_camera_name,
                    job.extrinsics_convention,
                )

                image_dir = temporary_dir / "images" / output_view_name
                depth_dir = temporary_dir / "depths" / output_view_name
                intrinsic_dir = temporary_dir / "intrinsics"
                extrinsic_dir = temporary_dir / "extrinsics"
                image_dir.mkdir(parents=True)
                depth_dir.mkdir(parents=True)
                intrinsic_dir.mkdir(exist_ok=True)
                extrinsic_dir.mkdir(exist_ok=True)
                np.save(
                    intrinsic_dir / f"{output_view_name}.npy",
                    intrinsic,
                    allow_pickle=False,
                )
                np.save(
                    extrinsic_dir / f"{output_view_name}.npy",
                    extrinsic,
                    allow_pickle=False,
                )

                for frame_index in range(num_frames):
                    stem = f"{frame_index:06d}.png"
                    rgb = _decode_rgb(
                        camera["colors"][frame_index],
                        f"{source_camera_name}/colors[{frame_index}]",
                    )
                    depth = np.asarray(camera["depths"][frame_index])
                    if rgb.shape[:2] != depth.shape[:2]:
                        raise ValueError(
                            f"{source_camera_name} RGB/depth size mismatch at frame "
                            f"{frame_index}: {rgb.shape[:2]} vs {depth.shape[:2]}"
                        )
                    if image_hw is None:
                        image_hw = [int(rgb.shape[0]), int(rgb.shape[1])]
                    elif image_hw != [int(rgb.shape[0]), int(rgb.shape[1])]:
                        raise ValueError(
                            f"All views and frames must share one resolution; got "
                            f"{rgb.shape[:2]} and {tuple(image_hw)}"
                        )
                    _write_rgb_png(
                        image_dir / stem, rgb, compression=job.png_compression
                    )
                    _write_depth_png(
                        depth_dir / stem, depth, compression=job.png_compression
                    )

            metadata = {
                "format_version": "robotwin_4d_v1",
                "task": job.task,
                "episode": output_dir.name,
                "embodiment": job.embodiment,
                "source_file": str(source_file.resolve()),
                "num_frames": num_frames,
                "image_height": image_hw[0] if image_hw else None,
                "image_width": image_hw[1] if image_hw else None,
                "views": {
                    output_name: source_name
                    for source_name, output_name in CAMERA_MAP.items()
                },
                "rgb": {
                    "format": "PNG",
                    "dtype": "uint8",
                    "channel_order": "RGB",
                },
                "depth": {
                    "format": "PNG",
                    "dtype": "uint16",
                    "unit": "millimeter",
                    "invalid_value": 0,
                },
                "intrinsics": {
                    "shape": [3, 3],
                    "convention": "OpenCV pinhole pixel coordinates",
                    "per_view_constant": True,
                },
                "extrinsics": {
                    "shape": [num_frames, 3, 4],
                    "transform": job.extrinsics_convention,
                    "camera_coordinates": "OpenCV: +x right, +y down, +z forward",
                },
                "robot": {
                    "state_shape": list(state.shape),
                    "action_shape": list(action.shape),
                    "dtype": "float32",
                    "columns": robot_schema,
                },
                "frequency_hz": _scalar_value(root, "additional_info/frequency"),
                "instructions": _parse_instructions(_scalar_value(root, "instructions")),
            }
            with (temporary_dir / "metadata.json").open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2, ensure_ascii=False)
                file.write("\n")

        if output_dir.exists():
            # This branch is reachable only for the explicitly selected
            # --existing overwrite policy, after the replacement is complete.
            shutil.rmtree(output_dir)
        temporary_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return {
        "status": "converted",
        "source": str(source_file),
        "output": str(output_dir),
        "num_frames": num_frames,
    }


def discover_jobs(args: argparse.Namespace) -> list[ConversionJob]:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root:
        raise ValueError("--input-root and --output-root must be different directories")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    if args.tasks:
        task_dirs = [input_root / task for task in args.tasks]
        missing_tasks = [str(path) for path in task_dirs if not path.is_dir()]
        if missing_tasks:
            raise FileNotFoundError("Task directories not found: " + ", ".join(missing_tasks))
    else:
        task_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())

    selected_episode_ids = set(args.episode_ids) if args.episode_ids else None
    jobs: list[ConversionJob] = []
    for task_dir in task_dirs:
        data_dir = task_dir / args.embodiment / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Episode data directory not found: {data_dir}")
        episode_files = sorted(
            (
                path
                for path in data_dir.iterdir()
                if path.is_file() and EPISODE_PATTERN.match(path.name)
            ),
            key=_episode_number,
        )
        if selected_episode_ids is not None:
            episode_files = [
                path for path in episode_files if _episode_number(path) in selected_episode_ids
            ]
        if args.max_episodes is not None:
            episode_files = episode_files[: args.max_episodes]
        if not episode_files:
            raise FileNotFoundError(f"No selected episode HDF5 files found in {data_dir}")

        for source_file in episode_files:
            episode_id = _episode_number(source_file)
            output_dir = output_root / task_dir.name / f"episode_{episode_id:07d}"
            jobs.append(
                ConversionJob(
                    source_file=str(source_file),
                    output_dir=str(output_dir),
                    task=task_dir.name,
                    embodiment=args.embodiment,
                    extrinsics_convention=args.extrinsics_convention,
                    existing=args.existing,
                    png_compression=args.png_compression,
                )
            )
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RoboTwin 3D ALOHA HDF5 episodes into per-view RGB-D PNGs, "
            "camera matrices, and dense robot state/action arrays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/3d_aloha_dataset"),
        help="Directory containing <task>/<embodiment>/data/*.hdf5",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/4d_aloha_dataset"),
        help="Destination root for <task>/<episode>/ outputs",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task names to convert; omit to convert all tasks",
    )
    parser.add_argument(
        "--embodiment",
        default="aloha_agilex",
        help="Embodiment subdirectory below each task",
    )
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        type=int,
        help="Only convert these numeric episode IDs",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Maximum number of selected episodes per task",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of episodes converted concurrently",
    )
    parser.add_argument(
        "--extrinsics-convention",
        choices=("world_to_camera", "camera_to_world"),
        default="world_to_camera",
        help="Transform represented by the exported (T, 3, 4) matrices",
    )
    parser.add_argument(
        "--existing",
        choices=("error", "skip", "overwrite"),
        default="error",
        help="Policy for an already existing output episode",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=3,
        metavar="0..9",
        help="OpenCV lossless PNG compression level",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be at least 1")
    if args.episode_ids and any(value < 0 for value in args.episode_ids):
        parser.error("--episode-ids values must be non-negative")


def run(jobs: Sequence[ConversionJob], workers: int) -> int:
    converted = 0
    skipped = 0

    with tqdm(
        total=len(jobs),
        desc="Converting episodes",
        unit="episode",
        dynamic_ncols=True,
        smoothing=0.1,
    ) as progress:

        def record_result(result: dict[str, Any]) -> None:
            nonlocal converted, skipped
            if result["status"] == "converted":
                converted += 1
            else:
                skipped += 1
            progress.set_postfix(
                converted=converted,
                skipped=skipped,
                refresh=False,
            )
            progress.update()

        if workers == 1:
            for job in jobs:
                record_result(convert_episode(job))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(convert_episode, job): job for job in jobs}
                for future in as_completed(futures):
                    record_result(future.result())

    print(f"Done: {converted} converted, {skipped} skipped.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    try:
        jobs = discover_jobs(args)
        return run(jobs, args.workers)
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
