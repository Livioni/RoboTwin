#!/usr/bin/env python3
"""
高级 HDF5 数据读取脚本 - 支持单帧和批量数据提取

功能：
- 查看指定帧的详细信息
- 提取单帧或所有帧的图像
- 保存关节轨迹数据
- 批量提取完整数据集
- 绘制关节运动轨迹
"""

import os
import h5py
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import io

def decode_image(encoded_data):
    """解码 JPEG 压缩的图像数据"""
    if isinstance(encoded_data, bytes):
        # 移除填充的 null 字节
        clean_data = encoded_data.rstrip(b'\x00')
        # 解码 JPEG
        img_array = np.frombuffer(clean_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    else:
        return encoded_data

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
    parser.add_argument("file_path", help="HDF5 文件路径")
    parser.add_argument("--frame", "-f", type=int, default=0, help="要显示的帧索引 (默认: 0)")

    # 单帧操作
    parser.add_argument("--extract-images", "-e", action="store_true", help="提取并保存当前帧图像")

    # 批量操作
    parser.add_argument("--save-camera-data", "-C", action="store_true", help="保存所有帧的相机数据（RGB, extrinsic, intrinsic）为 npy 文件")
    parser.add_argument("--save-trajectories", "-t", action="store_true", help="保存关节轨迹数据为 npz 文件")
    parser.add_argument("--extract-all-data", "-a", action="store_true", help="提取所有帧的完整数据")

    # 绘图选项
    parser.add_argument("--plot-trajectory", "-p", action="store_true", help="绘制关节轨迹")
    parser.add_argument("--joint-idx", "-j", type=int, default=0, help="要绘制的关节索引 (默认: 0)")

    # 输出选项
    parser.add_argument("--output-dir", "-o", default="extracted_data", help="输出目录 (默认: extracted_data)")
    parser.add_argument("--cameras", "-c", nargs="+", default=["left_camera"], help="指定要提取的相机 (默认: 全部)")

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"文件不存在: {args.file_path}")
        return

    print(f"正在读取 HDF5 文件: {args.file_path}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.file_path, 'r') as f:
        # 获取总帧数用于显示
        total_frames = len(f['/joint_action/left_gripper']) if '/joint_action/left_gripper' in f else 0
        print(f"数据总帧数: {total_frames}")

        # 单帧操作
        if args.extract_images:
            print(f"\n--- 处理单帧 (第 {args.frame} 帧) ---")
            frame_data = extract_frame_data(f, args.frame)
            display_frame_info(frame_data, args.frame)
            save_frame_images(frame_data, args.frame, args.output_dir)

        elif args.save_camera_data:
            print("\n--- 保存所有帧相机数据 ---")
            all_frames_data = extract_all_frames(f)
            saved_count = save_all_frames_camera_data(all_frames_data, args.output_dir, args.cameras)

        elif args.save_trajectories:
            print("\n--- 保存关节轨迹数据 ---")
            save_joint_trajectories(args.file_path, os.path.join(args.output_dir, "trajectories.npz"))

        # 绘图操作 (可以与其他操作同时进行)
        if args.plot_trajectory:
            print("\n--- 绘制关节轨迹 ---")
            plot_trajectory(args.file_path, args.joint_idx)

if __name__ == "__main__":
    main()
