#!/usr/bin/env python3
"""
Visualize one extracted RoboTwin 3D episode frame as a colored PLY point cloud.

Example:
  python visualize_3d_pointcloud.py extracted_data/custom_aloha/episode0 --frame 0

Input directory layout:
  episode_dir/
    camera_data/
      images/<camera>/frame_000000.png
      depths/<camera>/frame_000000.png
      intrinsics/<camera>/intrinsic.npy
      extrinsics/<camera>/frame_000000.npy
"""

import argparse
import os
from typing import Iterable, List, Tuple

import cv2
import numpy as np


DEFAULT_CAMERAS = ["head_camera", "left_camera", "right_camera"]


def _as_4x4_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float32)
    if extrinsic.shape == (3, 4):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :4] = extrinsic.astype(np.float32)
        return mat
    raise ValueError(f"extrinsic must have shape (3, 4) or (4, 4), got {extrinsic.shape}")


def invert_se3(extrinsic: np.ndarray) -> np.ndarray:
    """Invert a rigid 3x4/4x4 SE(3) matrix."""
    mat = _as_4x4_extrinsic(extrinsic)
    inv = np.eye(4, dtype=np.float32)
    r = mat[:3, :3]
    t = mat[:3, 3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t
    return inv


def load_rgb(path: str) -> np.ndarray:
    rgb_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"failed to read RGB image: {path}")
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)


def load_depth_meters(path: str, depth_scale: float) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"failed to read depth image: {path}")
    return depth.astype(np.float32) / float(depth_scale)


def backproject_depth(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    extrinsic_mode: str,
    min_depth: float,
    max_depth: float,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(f"RGB/depth size mismatch: rgb={rgb.shape[:2]}, depth={depth_m.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must have shape (3, 3), got {intrinsic.shape}")

    depth = depth_m[::stride, ::stride]
    colors = rgb[::stride, ::stride]
    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    u *= stride
    v *= stride

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])

    valid = np.isfinite(depth) & (depth > min_depth)
    if max_depth > 0:
        valid &= depth < max_depth

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    cam_points = np.stack([x, y, z], axis=1).astype(np.float32)

    if extrinsic_mode == "world2cam":
        cam_to_world = invert_se3(extrinsic)
    elif extrinsic_mode == "cam2world":
        cam_to_world = _as_4x4_extrinsic(extrinsic)
    else:
        raise ValueError(f"invalid extrinsic mode: {extrinsic_mode}")

    r = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]
    world_points = cam_points @ r.T + t
    point_colors = colors[valid].astype(np.uint8)
    return world_points.astype(np.float32), point_colors


def write_ply_binary(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    if len(points) != len(colors):
        raise ValueError("points/colors length mismatch")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with open(path, "wb") as fp:
        fp.write(header)
        vertices.tofile(fp)


def frame_file(kind_dir: str, camera: str, frame_idx: int, suffix: str) -> str:
    return os.path.join(kind_dir, camera, f"frame_{frame_idx:06d}.{suffix}")


def build_point_cloud(
    episode_dir: str,
    frame_idx: int,
    cameras: Iterable[str],
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    stride: int,
    extrinsic_mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    camera_data = os.path.join(episode_dir, "camera_data")
    images_dir = os.path.join(camera_data, "images")
    depths_dir = os.path.join(camera_data, "depths")
    intrinsics_dir = os.path.join(camera_data, "intrinsics")
    extrinsics_dir = os.path.join(camera_data, "extrinsics")

    all_points = []
    all_colors = []
    used_cameras = []

    for camera in cameras:
        rgb_path = frame_file(images_dir, camera, frame_idx, "png")
        depth_path = frame_file(depths_dir, camera, frame_idx, "png")
        intrinsic_path = os.path.join(intrinsics_dir, camera, "intrinsic.npy")
        extrinsic_path = frame_file(extrinsics_dir, camera, frame_idx, "npy")

        missing = [p for p in (rgb_path, depth_path, intrinsic_path, extrinsic_path) if not os.path.exists(p)]
        if missing:
            print(f"WARN: skip {camera}, missing files:")
            for p in missing:
                print(f"  {p}")
            continue

        rgb = load_rgb(rgb_path)
        depth_m = load_depth_meters(depth_path, depth_scale)
        intrinsic = np.load(intrinsic_path).astype(np.float32)
        extrinsic = np.load(extrinsic_path).astype(np.float32)
        points, colors = backproject_depth(
            rgb=rgb,
            depth_m=depth_m,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            extrinsic_mode=extrinsic_mode,
            min_depth=min_depth,
            max_depth=max_depth,
            stride=stride,
        )

        print(f"{camera}: {len(points)} valid points")
        all_points.append(points)
        all_colors.append(colors)
        used_cameras.append(camera)

    if not all_points:
        raise RuntimeError("no valid camera data found")

    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0), used_cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a colored PLY point cloud from extracted RoboTwin 3D data.")
    parser.add_argument("episode_dir", help="Path like extracted_data/custom_aloha/episode0")
    parser.add_argument("--frame", "-f", type=int, required=True, help="Frame index to export")
    parser.add_argument("--output", "-o", default=None, help="Output PLY path")
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS, help="Camera names to fuse")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth PNG scale: depth_m = png / scale")
    parser.add_argument("--min-depth", type=float, default=1e-6, help="Minimum valid depth in meters")
    parser.add_argument("--max-depth", type=float, default=10.0, help="Maximum valid depth in meters, <=0 disables")
    parser.add_argument("--stride", type=int, default=1, help="Pixel stride for downsampling")
    parser.add_argument(
        "--extrinsic-mode",
        choices=["world2cam", "cam2world"],
        default="world2cam",
        help="Use world2cam for RoboTwin extrinsic_cv, matching robotwin.py behavior",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    output = args.output
    if output is None:
        output = os.path.join(args.episode_dir, f"frame_{args.frame:06d}_3cam_pointcloud.ply")

    points, colors, used_cameras = build_point_cloud(
        episode_dir=args.episode_dir,
        frame_idx=args.frame,
        cameras=args.cameras,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        stride=args.stride,
        extrinsic_mode=args.extrinsic_mode,
    )
    write_ply_binary(output, points, colors)
    print(f"Saved {len(points)} points from {used_cameras} to {output}")


if __name__ == "__main__":
    main()
