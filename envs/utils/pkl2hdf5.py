import h5py, pickle
import json
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


CAMERA_MAP = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
    "front_camera": "cam_third_view",
}


def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for rgb_image in imgs:
        if rgb_image.ndim != 3 or rgb_image.shape[-1] != 3:
            raise ValueError(
                f"Expected an RGB image with shape (H, W, 3), got {rgb_image.shape}"
            )

        # Keep source channel values unchanged for XPolicyLab's OpenCV decoder.
        # OpenCV intentionally interprets this RGB array as BGR while encoding.
        success, encoded_image = cv2.imencode(".jpg", rgb_image)
        if not success:
            raise ValueError("OpenCV failed to encode an RGB frame as JPEG")
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def _ensure_2d(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    return values


def _to_4x4(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2 and values.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3, :4] = values
        return result
    if values.ndim == 3 and values.shape[1:] == (3, 4):
        result = np.repeat(np.eye(4, dtype=np.float32)[None], len(values), axis=0)
        result[:, :3, :4] = values
        return result
    return values


def _matrix_sequence(values, matrix_shape, frame_num, field_name):
    values = np.asarray(values, dtype=np.float32)
    if values.shape == matrix_shape:
        values = np.repeat(values[None], frame_num, axis=0)
    if values.ndim != 3 or values.shape[1:] != matrix_shape:
        raise ValueError(
            f"{field_name} must have shape {matrix_shape} or (T, {matrix_shape[0]}, {matrix_shape[1]}), "
            f"got {values.shape}"
        )
    if len(values) != frame_num:
        raise ValueError(
            f"{field_name} has {len(values)} frames, expected {frame_num}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} contains NaN or infinity")
    return values


def _camera_calibration(source_camera, frame_num, camera_name):
    if "intrinsic_cv" not in source_camera:
        raise ValueError(f"{camera_name} is missing intrinsic_cv")
    if "extrinsic_cv" not in source_camera:
        raise ValueError(f"{camera_name} is missing extrinsic_cv")

    intrinsic = _matrix_sequence(
        source_camera["intrinsic_cv"], (3, 3), frame_num, f"{camera_name}.intrinsic_cv"
    )
    extrinsic = _matrix_sequence(
        _to_4x4(source_camera["extrinsic_cv"]),
        (4, 4),
        frame_num,
        f"{camera_name}.extrinsic_cv",
    )
    expected_last_row = np.asarray([0, 0, 0, 1], dtype=np.float32)
    if not np.allclose(extrinsic[:, 3, :], expected_last_row, atol=1e-5):
        raise ValueError(f"{camera_name}.extrinsic_cv has an invalid homogeneous row")
    try:
        camera_pose = np.linalg.inv(extrinsic).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{camera_name}.extrinsic_cv is not invertible") from exc
    return intrinsic, extrinsic, camera_pose


def create_xpolicylab_hdf5(data, hdf5_path, instructions, frequency):
    joints = data["joint_action"]
    frame_num = len(joints["left_arm"])
    if frame_num < 2:
        raise ValueError("At least two frames are required to create state/action pairs")

    instruction_values = [str(value) for value in (instructions or []) if str(value)]
    if not instruction_values:
        instruction_values = [""]

    with h5py.File(hdf5_path, "w") as f:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        f.attrs["source_format"] = "RoboTwin"
        f.attrs["source_path"] = "native_collection"
        f.create_dataset("data_format_version", data="v1.0", dtype=string_dtype)
        f.create_dataset(
            "instructions",
            data=json.dumps(instruction_values, ensure_ascii=False),
            dtype=string_dtype,
        )
        f.create_group("additional_info").create_dataset(
            "frequency", data=np.asarray(frequency, dtype=np.int32)
        )

        state = f.create_group("state")
        action = f.create_group("action")
        joint_fields = [
            ("left_arm", "left_arm_joint_states"),
            ("left_gripper", "left_ee_joint_states"),
            ("right_arm", "right_arm_joint_states"),
            ("right_gripper", "right_ee_joint_states"),
        ]
        for source_name, target_name in joint_fields:
            if source_name not in joints:
                continue
            values = _ensure_2d(joints[source_name])
            state.create_dataset(target_name, data=values[:-1])
            action.create_dataset(target_name, data=values[1:])

        endpose = data.get("endpose", {})
        for source_name, target_name in [
            ("left_endpose", "left_ee_poses"),
            ("right_endpose", "right_ee_poses"),
        ]:
            if source_name not in endpose:
                continue
            values = _ensure_2d(endpose[source_name])
            state.create_dataset(target_name, data=values[:-1])
            action.create_dataset(target_name, data=values[1:])

        vision = f.create_group("vision")
        observations = data["observation"]
        for source_name, target_name in CAMERA_MAP.items():
            if source_name not in observations or "rgb" not in observations[source_name]:
                continue
            source_camera = observations[source_name]
            target_camera = vision.create_group(target_name)
            colors = np.asarray(source_camera["rgb"])
            if len(colors) != frame_num:
                raise ValueError(
                    f"{source_name}.rgb has {len(colors)} frames, expected {frame_num}"
                )
            colors = colors[:-1]
            encoded_colors, max_len = images_encoding(colors)
            target_camera.create_dataset(
                "colors", data=encoded_colors, dtype=f"S{max_len}"
            )
            target_camera.create_dataset(
                "shape", data=np.asarray(colors[0].shape, dtype=np.int32)
            )

            if "depth" in source_camera:
                depths = np.asarray(source_camera["depth"])
                if depths.ndim == 4 and depths.shape[-1] == 1:
                    depths = depths[..., 0]
                if depths.ndim != 3:
                    raise ValueError(
                        f"{source_name}.depth must have shape (T, H, W), got {depths.shape}"
                    )
                if len(depths) != frame_num:
                    raise ValueError(
                        f"{source_name}.depth has {len(depths)} frames, expected {frame_num}"
                    )
                depths = np.clip(np.rint(depths), 0, 65535).astype(np.uint16)[:-1]
                depth_dataset = target_camera.create_dataset(
                    "depths",
                    data=depths,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    chunks=(1, *depths.shape[1:]),
                )
                depth_dataset.attrs["unit"] = "millimeter"
                depth_dataset.attrs["invalid_value"] = np.uint16(0)

            intrinsic, extrinsic, camera_pose = _camera_calibration(
                source_camera, frame_num, source_name
            )
            intrinsic_dataset = target_camera.create_dataset(
                "intrinsic_matrix", data=intrinsic[:-1]
            )
            intrinsic_dataset.attrs["convention"] = "OpenCV pinhole; pixel coordinates"

            extrinsic_dataset = target_camera.create_dataset(
                "extrinsic_matrix", data=extrinsic[:-1]
            )
            extrinsic_dataset.attrs["transform"] = "world_to_camera"
            extrinsic_dataset.attrs["camera_coordinates"] = (
                "OpenCV: +x right, +y down, +z forward"
            )

            pose_dataset = target_camera.create_dataset(
                "camera_pose_matrix", data=camera_pose[:-1]
            )
            pose_dataset.attrs["transform"] = "camera_to_world"
            pose_dataset.attrs["camera_coordinates"] = (
                "OpenCV: +x right, +y down, +z forward"
            )

    return frame_num - 1


def pkl_files_to_hdf5_and_video(
    pkl_files,
    hdf5_path,
    video_path,
    *,
    instructions=None,
    frequency=15,
):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    images_to_video(np.array(data_list["observation"]["head_camera"]["rgb"]), out_path=video_path)
    return create_xpolicylab_hdf5(data_list, hdf5_path, instructions, frequency)


def process_folder_to_hdf5_video(
    folder_path,
    hdf5_path,
    video_path,
    *,
    instructions=None,
    frequency=15,
):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    return pkl_files_to_hdf5_and_video(
        pkl_files,
        hdf5_path,
        video_path,
        instructions=instructions,
        frequency=frequency,
    )
