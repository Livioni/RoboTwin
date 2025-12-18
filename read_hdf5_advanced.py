#!/usr/bin/env python3
"""
高级 HDF5 数据读取脚本 - 支持单帧和批量数据提取

功能：
- 查看指定帧的详细信息
- 提取单帧或所有帧的图像
- 保存关节轨迹数据
- 批量提取完整数据集（支持输入目录，自动遍历其中所有 hdf5）
- 绘制关节运动轨迹
"""

import os
import h5py
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import json
import random
from typing import Dict, List, Optional, Tuple

def decode_image(encoded_data):
    """解码 JPEG 压缩的图像数据"""
    if isinstance(encoded_data, bytes):
        # 移除填充的 null 字节
        clean_data = encoded_data.rstrip(b'\x00')
        # 解码 JPEG
        img_array = np.frombuffer(clean_data, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        # 统一返回 RGB，避免后续保存/可视化颜色颠倒
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    else:
        return encoded_data

def _safe_read_array(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    try:
        return f[key][:]
    except Exception:
        return None

def _total_frames(f: h5py.File) -> int:
    """尽可能稳健地获取该 episode 的帧数"""
    candidates = [
        "/joint_action/left_gripper",
        "/joint_action/right_gripper",
        "/joint_action/left_arm",
        "/joint_action/right_arm",
    ]
    for k in candidates:
        if k in f:
            return int(len(f[k]))
    # fallback to observation any camera rgb length
    if "/observation" in f:
        obs = f["/observation"]
        for cam_name in obs.keys():
            if "rgb" in obs[cam_name]:
                try:
                    return int(len(obs[cam_name]["rgb"]))
                except Exception:
                    continue
    return 0

def _arm_motion_score(
    arm: Optional[np.ndarray],
    gripper: Optional[np.ndarray],
) -> float:
    """用关节序列的最大差分幅度作为运动强度打分。"""
    score = 0.0
    if arm is not None and len(arm) >= 2:
        try:
            score = max(score, float(np.nanmax(np.abs(np.diff(arm, axis=0)))))
        except Exception:
            pass
    if gripper is not None and len(gripper) >= 2:
        try:
            score = max(score, float(np.nanmax(np.abs(np.diff(gripper, axis=0)))))
        except Exception:
            pass
    return score

def choose_moving_arm(
    f: h5py.File,
    threshold: float = 1e-3,
    prefer: str = "left",
) -> Tuple[str, Dict[str, float]]:
    """
    判断哪个手臂在移动：
    - 仅一侧超过阈值 -> 选该侧
    - 两侧都超过阈值 -> 随机选一侧
    - 都不超过阈值 -> 选 prefer（若存在对应数据/相机），否则退化
    """
    left_arm = _safe_read_array(f, "/joint_action/left_arm")
    right_arm = _safe_read_array(f, "/joint_action/right_arm")
    left_gripper = _safe_read_array(f, "/joint_action/left_gripper")
    right_gripper = _safe_read_array(f, "/joint_action/right_gripper")

    left_score = _arm_motion_score(left_arm, left_gripper)
    right_score = _arm_motion_score(right_arm, right_gripper)
    scores = {"left": left_score, "right": right_score}

    moving = [k for k, v in scores.items() if v > threshold]
    if len(moving) == 1:
        return moving[0], scores
    if len(moving) == 2:
        return random.choice(moving), scores

    # none moving: prefer side if it exists, else whichever exists
    if prefer == "left" and (left_arm is not None or left_gripper is not None):
        return "left", scores
    if prefer == "right" and (right_arm is not None or right_gripper is not None):
        return "right", scores
    if left_arm is not None or left_gripper is not None:
        return "left", scores
    if right_arm is not None or right_gripper is not None:
        return "right", scores
    return prefer, scores

def list_cameras(f: h5py.File) -> List[str]:
    if "/observation" not in f:
        return []
    return sorted(list(f["/observation"].keys()))

def choose_cameras_for_arm(
    available: List[str],
    arm: str,
) -> List[str]:
    """从可用相机名里挑选对应手臂的相机（优先 left_camera/right_camera，其次模糊匹配）。"""
    if not available:
        return []

    preferred = "left_camera" if arm == "left" else "right_camera"
    if preferred in available:
        return [preferred]

    arm_lower = arm.lower()
    # e.g. left_wrist_camera, right_wrist, etc.
    fuzzy = [c for c in available if arm_lower in c.lower()]
    if fuzzy:
        # 只取更可能是“该手臂视角”的相机：包含 camera 关键字优先
        camera_like = [c for c in fuzzy if "camera" in c.lower()]
        return sorted(camera_like) if camera_like else sorted(fuzzy)

    return [available[0]]

def extract_frame_data(hdf5_file, frame_idx=0):
    """提取指定帧的所有数据"""
    data = {}

    # 关节动作
    if '/joint_action' in hdf5_file:
        joint_group = hdf5_file['/joint_action']
        data['left_arm'] = joint_group['left_arm'][frame_idx]
        data['left_gripper'] = joint_group['left_gripper'][frame_idx]
        data['right_arm'] = joint_group['right_arm'][frame_idx]
        data['right_gripper'] = joint_group['right_gripper'][frame_idx]

    # 末端姿态
    if '/endpose' in hdf5_file:
        endpose_group = hdf5_file['/endpose']
        data['left_endpose'] = endpose_group['left_endpose'][frame_idx]
        data['left_endpose_gripper'] = endpose_group['left_gripper'][frame_idx]
        data['right_endpose'] = endpose_group['right_endpose'][frame_idx]
        data['right_endpose_gripper'] = endpose_group['right_gripper'][frame_idx]

    # 相机图像和参数
    data['images'] = {}
    data['camera_params'] = {}
    if '/observation' in hdf5_file:
        obs_group = hdf5_file['/observation']
        for cam_name in obs_group.keys():
            cam_data = {}

            # RGB图像
            if 'rgb' in obs_group[cam_name]:
                encoded_img = obs_group[cam_name]['rgb'][frame_idx]
                decoded_img = decode_image(encoded_img)
                cam_data['rgb'] = decoded_img
                data['images'][cam_name] = decoded_img

            # 相机参数
            if 'intrinsic_cv' in obs_group[cam_name]:
                cam_data['intrinsic'] = obs_group[cam_name]['intrinsic_cv'][frame_idx]
            if 'extrinsic_cv' in obs_group[cam_name]:
                cam_data['extrinsic'] = obs_group[cam_name]['extrinsic_cv'][frame_idx]
            if 'cam2world_gl' in obs_group[cam_name]:
                cam_data['cam2world'] = obs_group[cam_name]['cam2world_gl'][frame_idx]
            if 'depth' in obs_group[cam_name]:
                cam_data['depth'] = obs_group[cam_name]['depth'][frame_idx]

            data['camera_params'][cam_name] = cam_data

    return data

def display_frame_info(frame_data, frame_idx):
    """显示帧信息的详细信息"""
    print(f"\n=== 第 {frame_idx} 帧数据详情 ===")

    # 关节状态
    print("关节状态:")
    print(f"  左臂关节角度: {frame_data['left_arm']}")
    print(f"  左夹爪: {frame_data['left_gripper']:.3f}")
    print(f"  右臂关节角度: {frame_data['right_arm']}")
    print(f"  右夹爪: {frame_data['right_gripper']:.3f}")

    # 末端姿态
    if 'left_endpose' in frame_data:
        print("\n末端姿态:")
        print(f"  左臂末端姿态: 位置({frame_data['left_endpose'][:3]}), 姿态({frame_data['left_endpose'][3:]})")
        print(f"  左臂末端夹爪: {frame_data['left_endpose_gripper']:.3f}")
        print(f"  右臂末端姿态: 位置({frame_data['right_endpose'][:3]}), 姿态({frame_data['right_endpose'][3:]})")
        print(f"  右臂末端夹爪: {frame_data['right_endpose_gripper']:.3f}")

    # 图像信息
    print("\n图像信息:")
    for cam_name, img in frame_data['images'].items():
        if img is not None:
            print(f"  {cam_name}: {img.shape} (HWC格式)")
        else:
            print(f"  {cam_name}: 解码失败")

def save_frame_images(frame_data, frame_idx, output_dir="extracted_images"):
    """保存指定帧的所有图像"""
    os.makedirs(output_dir, exist_ok=True)

    for cam_name, img in frame_data['images'].items():
        if img is not None:
            filename = f"frame_{frame_idx:04d}_{cam_name}.jpg"
            filepath = os.path.join(output_dir, filename)
            success = cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if success:
                print(f"保存图像: {filepath}")
            else:
                print(f"保存失败: {filepath}")

def extract_all_frames(hdf5_file):
    """提取 HDF5 文件中的所有帧数据"""
    print("正在提取所有帧数据...")

    # 获取总帧数
    total_frames = 0
    if '/joint_action/left_gripper' in hdf5_file:
        total_frames = len(hdf5_file['/joint_action/left_gripper'])

    print(f"总帧数: {total_frames}")

    all_frames_data = []

    for frame_idx in range(total_frames):
        if frame_idx % 20 == 0:  # 每20帧显示一次进度
            print(f"处理帧 {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

        frame_data = extract_frame_data(hdf5_file, frame_idx)
        all_frames_data.append(frame_data)

    print(f"完成! 共提取 {len(all_frames_data)} 帧数据")
    return all_frames_data

def save_all_frames_images(all_frames_data, output_dir="extracted_images", cameras=None):
    """保存所有帧的所有图像"""
    os.makedirs(output_dir, exist_ok=True)

    total_frames = len(all_frames_data)
    saved_count = 0

    print(f"开始保存 {total_frames} 帧的图像...")

    for frame_idx, frame_data in enumerate(all_frames_data):
        if frame_idx % 50 == 0:  # 每50帧显示一次进度
            print(f"保存帧 {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

        for cam_name, img in frame_data['images'].items():
            # 如果指定了相机列表，只保存指定相机
            if cameras is not None and cam_name not in cameras:
                continue

            if img is not None:
                filename = f"frame_{frame_idx:04d}_{cam_name}.jpg"
                filepath = os.path.join(output_dir, filename)
                success = cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if success:
                    saved_count += 1
                else:
                    print(f"保存失败: {filepath}")

    print(f"完成! 共保存 {saved_count} 张图像到 {output_dir}")
    return saved_count

def save_all_frames_camera_data(all_frames_data, output_dir="camera_data", cameras=None):
    """保存所有帧的相机数据 (RGB为PNG, extrinsic/intrinsic为npy文件)"""
    # 创建子目录
    images_dir = os.path.join(output_dir, "images")
    extrinsics_dir = os.path.join(output_dir, "extrinsics")
    intrinsics_dir = os.path.join(output_dir, "intrinsics")
    depths_dir = os.path.join(output_dir, "depths")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(extrinsics_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)

    total_frames = len(all_frames_data)
    saved_count = 0

    print(f"开始保存 {total_frames} 帧的相机数据...")

    for frame_idx, frame_data in enumerate(all_frames_data):
        if frame_idx % 20 == 0:  # 每20帧显示一次进度
            print(f"保存帧 {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

        camera_params = frame_data.get('camera_params', {})

        for cam_name, cam_data in camera_params.items():
            # 如果指定了相机列表，只保存指定相机
            if cameras is not None and cam_name not in cameras:
                continue

            # 保存RGB图像为PNG
            if 'rgb' in cam_data and cam_data['rgb'] is not None:
                rgb_filename = f"frame_{frame_idx:06d}_{cam_name}.png"
                rgb_filepath = os.path.join(images_dir, rgb_filename)
                # 使用PIL保存PNG
                img = Image.fromarray(cam_data['rgb'])
                img.save(rgb_filepath)
                saved_count += 1

            # 保存外参矩阵
            if 'extrinsic' in cam_data:
                ext_filename = f"frame_{frame_idx:06d}_{cam_name}.npy"
                ext_filepath = os.path.join(extrinsics_dir, ext_filename)
                np.save(ext_filepath, cam_data['extrinsic'])
                saved_count += 1

            # 保存内参矩阵
            if 'intrinsic' in cam_data:
                int_filename = f"frame_{frame_idx:06d}_{cam_name}.npy"
                int_filepath = os.path.join(intrinsics_dir, int_filename)
                np.save(int_filepath, cam_data['intrinsic'])
                saved_count += 1

            # 深度数据（如果存在）
            if 'depth' in cam_data and cam_data['depth'] is not None:
                depth_filename = f"frame_{frame_idx:06d}_{cam_name}.npy"
                depth_filepath = os.path.join(depths_dir, depth_filename)
                np.save(depth_filepath, cam_data['depth'])
                saved_count += 1

    print(f"完成! 共保存 {saved_count} 个相机数据文件到 {output_dir}")
    print(f"  - 图像: {images_dir}")
    print(f"  - 外参: {extrinsics_dir}")
    print(f"  - 内参: {intrinsics_dir}")
    print(f"  - 深度: {depths_dir}")
    return saved_count

def save_robot_data_from_hdf5(
    f: h5py.File,
    output_dir: str,
    arm: str,
) -> Dict[str, str]:
    """
    保存对应手臂的轨迹数据（joint_action / endpose），并返回保存的文件路径信息。
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    arm_prefix = "left" if arm == "left" else "right"
    joint_key = f"/joint_action/{arm_prefix}_arm"
    gripper_key = f"/joint_action/{arm_prefix}_gripper"

    joint = _safe_read_array(f, joint_key)
    gripper = _safe_read_array(f, gripper_key)
    if joint is not None:
        p = os.path.join(output_dir, f"{arm_prefix}_arm_joint_action.npy")
        np.save(p, joint)
        saved["joint_action"] = p
    if gripper is not None:
        p = os.path.join(output_dir, f"{arm_prefix}_gripper_action.npy")
        np.save(p, gripper)
        saved["gripper_action"] = p

    if "/endpose" in f:
        endpose_key = f"/endpose/{arm_prefix}_endpose"
        endpose_gripper_key = f"/endpose/{arm_prefix}_gripper"
        endpose = _safe_read_array(f, endpose_key)
        endpose_gripper = _safe_read_array(f, endpose_gripper_key)
        if endpose is not None:
            p = os.path.join(output_dir, f"{arm_prefix}_endpose.npy")
            np.save(p, endpose)
            saved["endpose"] = p
        if endpose_gripper is not None:
            p = os.path.join(output_dir, f"{arm_prefix}_endpose_gripper.npy")
            np.save(p, endpose_gripper)
            saved["endpose_gripper"] = p

    return saved

def save_camera_data_from_hdf5(
    f: h5py.File,
    output_dir: str,
    cameras: Optional[List[str]] = None,
    save_rgb: bool = True,
    save_intrinsic: bool = True,
    save_extrinsic: bool = True,
    save_depth: bool = True,
) -> int:
    """
    直接从 HDF5 流式读取并保存相机数据，避免一次性把所有帧/图像解码到内存。
    输出结构与旧版本一致：
      output_dir/
        images/
        intrinsics/
        extrinsics/
        depths/
    """
    available = list_cameras(f)
    if cameras is None:
        cameras = available
    else:
        cameras = [c for c in cameras if c in available]

    images_dir = os.path.join(output_dir, "images")
    extrinsics_dir = os.path.join(output_dir, "extrinsics")
    intrinsics_dir = os.path.join(output_dir, "intrinsics")
    depths_dir = os.path.join(output_dir, "depths")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(extrinsics_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)

    total_frames = _total_frames(f)
    if total_frames <= 0:
        print("无法获取帧数，跳过保存相机数据。")
        return 0

    saved_count = 0
    obs_group = f["/observation"] if "/observation" in f else None
    if obs_group is None:
        print("HDF5 中不存在 /observation，跳过相机数据。")
        return 0

    for frame_idx in range(total_frames):
        if frame_idx % 50 == 0:
            print(f"保存帧 {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

        for cam_name in cameras:
            cam_g = obs_group[cam_name]

            if save_rgb and "rgb" in cam_g:
                encoded_img = cam_g["rgb"][frame_idx]
                decoded_img = decode_image(encoded_img)
                if decoded_img is not None:
                    rgb_filename = f"frame_{frame_idx:06d}_{cam_name}.png"
                    rgb_filepath = os.path.join(images_dir, rgb_filename)
                    img = Image.fromarray(decoded_img)
                    img.save(rgb_filepath)
                    saved_count += 1

            if save_extrinsic and "extrinsic_cv" in cam_g:
                ext = cam_g["extrinsic_cv"][frame_idx]
                ext_filename = f"frame_{frame_idx:06d}_{cam_name}.npy"
                np.save(os.path.join(extrinsics_dir, ext_filename), ext)
                saved_count += 1

            if save_intrinsic and "intrinsic_cv" in cam_g:
                it = cam_g["intrinsic_cv"][frame_idx]
                int_filename = f"frame_{frame_idx:06d}_{cam_name}.npy"
                np.save(os.path.join(intrinsics_dir, int_filename), it)
                saved_count += 1

            if save_depth and "depth" in cam_g:
                depth = cam_g["depth"][frame_idx]
                if depth is not None:
                    depth_filename = f"frame_{frame_idx:06d}_{cam_name}.npy"
                    np.save(os.path.join(depths_dir, depth_filename), depth)
                    saved_count += 1

    return saved_count

def _is_hdf5_file(path: str) -> bool:
    p = path.lower()
    return p.endswith(".hdf5") or p.endswith(".h5")

def collect_hdf5_files(input_path: str) -> List[str]:
    """输入 file/dir 都可：dir 会递归收集所有 hdf5 文件并排序。"""
    if os.path.isfile(input_path) and _is_hdf5_file(input_path):
        return [input_path]

    files: List[str] = []
    if os.path.isdir(input_path):
        for root, _, fnames in os.walk(input_path):
            for fn in fnames:
                if _is_hdf5_file(fn):
                    files.append(os.path.join(root, fn))
    return sorted(files)

def episode_name_from_path(hdf5_path: str) -> str:
    base = os.path.basename(hdf5_path)
    stem = os.path.splitext(base)[0]
    return stem

def process_one_episode(
    hdf5_path: str,
    episode_output_dir: str,
    cameras: Optional[List[str]],
    auto_select_moving_arm: bool,
    arm_threshold: float,
    prefer_arm: str,
    overwrite: bool,
) -> None:
    if os.path.exists(episode_output_dir) and os.listdir(episode_output_dir) and not overwrite:
        print(f"输出目录已存在且非空，跳过（可用 --overwrite 强制）：{episode_output_dir}")
        return
    os.makedirs(episode_output_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        available_cams = list_cameras(f)

        chosen_arm = prefer_arm
        scores = {"left": 0.0, "right": 0.0}
        selected_cams = cameras
        if auto_select_moving_arm and cameras is None:
            chosen_arm, scores = choose_moving_arm(f, threshold=arm_threshold, prefer=prefer_arm)
            selected_cams = choose_cameras_for_arm(available_cams, chosen_arm)
        elif cameras is None:
            selected_cams = available_cams

        # 保存 meta
        meta = {
            "source_hdf5": os.path.abspath(hdf5_path),
            "total_frames": _total_frames(f),
            "available_cameras": available_cams,
            "auto_select_moving_arm": bool(auto_select_moving_arm),
            "arm_threshold": float(arm_threshold),
            "motion_scores": scores,
            "chosen_arm": chosen_arm,
            "selected_cameras": selected_cams,
        }
        with open(os.path.join(episode_output_dir, "meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)

        # 保存对应手臂轨迹
        robot_dir = os.path.join(episode_output_dir, "robot_data")
        save_robot_data_from_hdf5(f, robot_dir, chosen_arm)

        # 保存相机数据
        cam_dir = os.path.join(episode_output_dir, "camera_data")
        print(f"选择手臂: {chosen_arm} (left={scores['left']:.6g}, right={scores['right']:.6g})")
        print(f"保存相机: {selected_cams}")
        save_camera_data_from_hdf5(f, cam_dir, cameras=selected_cams)

def save_joint_trajectories(hdf5_path, output_file="joint_trajectories.npz"):
    """保存所有关节的轨迹数据为 numpy 数组"""
    print("正在提取关节轨迹数据...")

    with h5py.File(hdf5_path, 'r') as f:
        left_arm_data = f['/joint_action/left_arm'][:]
        right_arm_data = f['/joint_action/right_arm'][:]
        left_gripper_data = f['/joint_action/left_gripper'][:]
        right_gripper_data = f['/joint_action/right_gripper'][:]

        # 如果有末端姿态数据，也保存
        endpose_data = {}
        if '/endpose' in f:
            endpose_group = f['/endpose']
            endpose_data['left_endpose'] = endpose_group['left_endpose'][:]
            endpose_data['right_endpose'] = endpose_group['right_endpose'][:]
            endpose_data['left_endpose_gripper'] = endpose_group['left_gripper'][:]
            endpose_data['right_endpose_gripper'] = endpose_group['right_gripper'][:]

    # 保存为 npz 文件
    save_data = {
        'left_arm': left_arm_data,
        'right_arm': right_arm_data,
        'left_gripper': left_gripper_data,
        'right_gripper': right_gripper_data,
    }
    save_data.update(endpose_data)

    np.savez(output_file, **save_data)
    print(f"关节轨迹数据已保存到: {output_file}")

    # 显示数据统计
    print(f"数据形状: left_arm {left_arm_data.shape}, right_arm {right_arm_data.shape}")
    print(f"时间步数: {left_arm_data.shape[0]}, 关节数: {left_arm_data.shape[1]}")

    return save_data

def plot_trajectory(hdf5_path, joint_idx=0):
    """绘制指定关节的角度变化轨迹"""
    with h5py.File(hdf5_path, 'r') as f:
        left_arm_data = f['/joint_action/left_arm'][:]
        right_arm_data = f['/joint_action/right_arm'][:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 左臂关节轨迹
        ax1.plot(left_arm_data[:, joint_idx], label=f'Joint {joint_idx}')
        ax1.set_title(f'Left Arm Joint {joint_idx} Trajectory')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Joint Angle (rad)')
        ax1.grid(True)
        ax1.legend()

        # 右臂关节轨迹
        ax2.plot(right_arm_data[:, joint_idx], label=f'Joint {joint_idx}')
        ax2.set_title(f'Right Arm Joint {joint_idx} Trajectory')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Joint Angle (rad)')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('joint_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("轨迹图已保存为: joint_trajectory.png")

def main():
    parser = argparse.ArgumentParser(description="读取和分析 RoboTwin HDF5 数据文件")
    parser.add_argument("file_path", help="HDF5 文件路径（支持输入目录：会递归遍历其中所有 hdf5）")
    parser.add_argument("--frame", "-f", type=int, default=0, help="要显示的帧索引 (默认: 0)")

    # 单帧操作
    parser.add_argument("--extract-images", "-e", action="store_true", help="提取并保存当前帧图像")

    # 批量操作
    parser.add_argument("--save-camera-data", "-C", action="store_true", help="保存所有帧的相机数据（RGB, extrinsic, intrinsic）为 npy 文件")
    parser.add_argument("--auto-select-moving-arm", action="store_true", default=True,
                        help="自动判断哪个手臂在动，并只保存对应相机/手臂数据（默认开启）")
    parser.add_argument("--no-auto-select-moving-arm", dest="auto_select_moving_arm", action="store_false",
                        help="关闭自动判断移动手臂，改为保存所有相机或 --cameras 指定相机")
    parser.add_argument("--arm-threshold", type=float, default=1e-3, help="判断手臂是否移动的阈值 (默认: 1e-3)")
    parser.add_argument("--prefer-arm", choices=["left", "right"], default="left", help="当无法判断移动手臂时的默认选择")
    parser.add_argument("--overwrite", action="store_true", help="输出目录已存在时覆盖（否则跳过）")

    # 绘图选项
    parser.add_argument("--plot-trajectory", "-p", action="store_true", help="绘制关节轨迹")
    parser.add_argument("--joint-idx", "-j", type=int, default=0, help="要绘制的关节索引 (默认: 0)")

    # 输出选项
    parser.add_argument("--output-dir", "-o", default="extracted_data", help="输出目录 (默认: extracted_data)")
    parser.add_argument("--cameras", "-c", nargs="+", default=None,
                        help="指定要提取的相机名列表（不传则由 --auto-select-moving-arm 自动选择，或保存所有）")

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"文件不存在: {args.file_path}")
        return

    hdf5_files = collect_hdf5_files(args.file_path)
    if not hdf5_files:
        print(f"未找到任何 hdf5 文件: {args.file_path}")
        return

    # 单帧模式只支持单个文件
    if args.extract_images or args.plot_trajectory:
        if len(hdf5_files) != 1:
            raise ValueError("单帧/绘图模式请传入单个 hdf5 文件路径，而不是目录。")
        hdf5_path = hdf5_files[0]
        print(f"正在读取 HDF5 文件: {hdf5_path}")
        os.makedirs(args.output_dir, exist_ok=True)
        with h5py.File(hdf5_path, "r") as f:
            total_frames = _total_frames(f)
            print(f"数据总帧数: {total_frames}")
            if args.extract_images:
                print(f"\n--- 处理单帧 (第 {args.frame} 帧) ---")
                frame_data = extract_frame_data(f, args.frame)
                display_frame_info(frame_data, args.frame)
                save_frame_images(frame_data, args.frame, args.output_dir)
            elif args.plot_trajectory:
                plot_trajectory(hdf5_path, joint_idx=args.joint_idx)
        return

    if not args.save_camera_data:
        raise ValueError("Invalid argument：请指定 --save-camera-data 或 --extract-images")

    # 批量/单文件保存相机数据
    os.makedirs(args.output_dir, exist_ok=True)

    if len(hdf5_files) == 1 and os.path.isfile(args.file_path):
        # 单个文件：output_dir 直接作为 episode 输出目录
        hdf5_path = hdf5_files[0]
        print(f"\n--- 保存 episode: {hdf5_path} ---")
        process_one_episode(
            hdf5_path=hdf5_path,
            episode_output_dir=args.output_dir,
            cameras=args.cameras,
            auto_select_moving_arm=args.auto_select_moving_arm,
            arm_threshold=args.arm_threshold,
            prefer_arm=args.prefer_arm,
            overwrite=args.overwrite,
        )
    else:
        # 目录：output_dir 作为 root，自动创建 episode 子目录
        print(f"检测到 {len(hdf5_files)} 个 hdf5 文件，开始批量提取...")
        for i, hdf5_path in enumerate(hdf5_files):
            ep_name = episode_name_from_path(hdf5_path)
            ep_out = os.path.join(args.output_dir, ep_name)
            print(f"\n[{i+1}/{len(hdf5_files)}] episode={ep_name}")
            process_one_episode(
                hdf5_path=hdf5_path,
                episode_output_dir=ep_out,
                cameras=args.cameras,
                auto_select_moving_arm=args.auto_select_moving_arm,
                arm_threshold=args.arm_threshold,
                prefer_arm=args.prefer_arm,
                overwrite=args.overwrite,
            )

if __name__ == "__main__":
    main()
