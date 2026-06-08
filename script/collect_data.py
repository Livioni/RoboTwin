import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import math
import subprocess
import shutil
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _episode_status_path(save_path):
    return os.path.join(save_path, "episode_status.json")


def _load_episode_statuses(save_path, seed_list):
    status_path = _episode_status_path(save_path)
    statuses = []
    if os.path.exists(status_path):
        with open(status_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        statuses = data.get("episodes", data if isinstance(data, list) else [])

    normalized = []
    for idx, seed in enumerate(seed_list):
        status = statuses[idx] if idx < len(statuses) and isinstance(statuses[idx], dict) else {}
        normalized.append({
            "episode_idx": idx,
            "seed": int(status.get("seed", seed)),
            "success": bool(status.get("success", True)),
        })
    return normalized


def _save_episode_statuses(save_path, statuses):
    os.makedirs(save_path, exist_ok=True)
    with open(_episode_status_path(save_path), "w", encoding="utf-8") as file:
        json.dump({"episodes": statuses}, file, ensure_ascii=False, indent=2)


def _set_episode_status(statuses, episode_idx, seed, success):
    while len(statuses) <= episode_idx:
        statuses.append({
            "episode_idx": len(statuses),
            "seed": None,
            "success": True,
        })
    statuses[episode_idx] = {
        "episode_idx": int(episode_idx),
        "seed": int(seed),
        "success": bool(success),
    }


def _count_success(statuses):
    return sum(1 for status in statuses if status.get("success", True))


def _count_failed(statuses):
    return sum(1 for status in statuses if not status.get("success", True))


def _min_success_episode_count(args):
    episode_num = int(args["episode_num"])
    if args.get("min_success_episodes") is not None:
        return min(episode_num, max(0, int(args["min_success_episodes"])))
    min_success_rate = float(args.get("min_success_rate", 1.0))
    return min(episode_num, max(0, int(math.ceil(episode_num * min_success_rate))))


def _max_failed_episode_count(args):
    episode_num = int(args["episode_num"])
    if not bool(args.get("collect_failed_data", False)):
        return 0
    if args.get("max_failed_episodes") is not None:
        return min(episode_num, max(0, int(args["max_failed_episodes"])))
    if args.get("min_success_episodes") is not None or args.get("min_success_rate") is not None:
        return episode_num - _min_success_episode_count(args)

    max_failed_episode_rate = float(args.get("max_failed_episode_rate", 0.2))
    if not 0 <= max_failed_episode_rate <= 1:
        raise ValueError(f"max_failed_episode_rate must be in [0, 1], got {max_failed_episode_rate}")
    return min(episode_num, max(0, int(math.floor(episode_num * max_failed_episode_rate))))


def _failed_count_after_update(statuses, episode_idx, success):
    failed_count = _count_failed(statuses)
    if episode_idx < len(statuses):
        old_success = bool(statuses[episode_idx].get("success", True))
        if old_success and not success:
            failed_count += 1
        elif not old_success and success:
            failed_count -= 1
    elif not success:
        failed_count += 1
    return failed_count


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _episode_extract_enabled():
    return _env_flag("ROBOTWIN_EXTRACT_EACH_EPISODE", False)


def _episode_output_dir(args, episode_idx):
    output_root = os.environ.get("ROBOTWIN_EXTRACT_OUTPUT_DIR")
    if not output_root:
        return None
    return os.path.join(output_root, f"episode{episode_idx}")


def _episode_output_complete(args, episode_idx):
    episode_output_dir = _episode_output_dir(args, episode_idx)
    if episode_output_dir is None or not os.path.isdir(episode_output_dir):
        return False
    return (
        os.path.isfile(os.path.join(episode_output_dir, "meta.json"))
        and os.path.isdir(os.path.join(episode_output_dir, "camera_data"))
        and os.path.isdir(os.path.join(episode_output_dir, "robot_data"))
    )


def _delete_raw_episode_files(args, episode_idx, hdf5_path):
    raw_paths = [
        hdf5_path,
        os.path.join(args["save_path"], "video", f"episode{episode_idx}.mp4"),
        os.path.join(args["save_path"], "_traj_data", f"episode{episode_idx}.pkl"),
    ]
    for raw_path in raw_paths:
        if os.path.exists(raw_path):
            os.remove(raw_path)
            print(f"Deleted raw episode file: {raw_path}")


def _seed_path(save_path):
    return os.path.join(save_path, "seed.txt")


def _traj_data_path(save_path, episode_idx):
    return os.path.join(save_path, "_traj_data", f"episode{episode_idx}.pkl")


def _write_seed_list(save_path, seed_list):
    with open(_seed_path(save_path), "w") as file:
        for seed in seed_list:
            file.write("%s " % seed)


def _postprocess_episode(args, episode_idx, hdf5_path):
    if not _episode_extract_enabled():
        return

    episode_output_dir = _episode_output_dir(args, episode_idx)
    if not episode_output_dir:
        raise RuntimeError("ROBOTWIN_EXTRACT_OUTPUT_DIR is required when ROBOTWIN_EXTRACT_EACH_EPISODE=1")

    if not os.path.exists(hdf5_path):
        if _episode_output_complete(args, episode_idx):
            print(f"Episode {episode_idx} already extracted: {episode_output_dir}")
            return
        raise FileNotFoundError(f"Cannot extract missing HDF5 episode: {hdf5_path}")

    if os.path.isdir(episode_output_dir) and os.listdir(episode_output_dir):
        if _episode_output_complete(args, episode_idx) and not _env_flag("ROBOTWIN_EXTRACT_OVERWRITE", False):
            print(f"Episode {episode_idx} already extracted: {episode_output_dir}")
            if not _env_flag("ROBOTWIN_EXTRACT_KEEP_RAW", False):
                _delete_raw_episode_files(args, episode_idx, hdf5_path)
            return
        if not _env_flag("ROBOTWIN_EXTRACT_OVERWRITE", False):
            print(f"Removing incomplete extracted output before retry: {episode_output_dir}")
            shutil.rmtree(episode_output_dir)

    root_dir = os.path.dirname(parent_directory)
    read_hdf5_path = os.path.join(root_dir, "read_hdf5_advanced.py")
    cameras = os.environ.get("ROBOTWIN_EXTRACT_CAMERAS", "").split()
    video_fps = os.environ.get("ROBOTWIN_EXTRACT_VIDEO_FPS", "30")

    cmd = [
        sys.executable,
        read_hdf5_path,
        hdf5_path,
        "--save-camera-data",
        "--output-dir",
        episode_output_dir,
        "--no-auto-select-moving-arm",
        "--save-videos",
        "--video-fps",
        video_fps,
        "--save-both-arms",
    ]
    if cameras:
        cmd.extend(["--cameras", *cameras])
    if _env_flag("ROBOTWIN_EXTRACT_OVERWRITE", False):
        cmd.append("--overwrite")

    print(f"Extracting episode {episode_idx}: {hdf5_path} -> {episode_output_dir}")
    subprocess.run(cmd, check=True)

    if not _episode_output_complete(args, episode_idx):
        raise RuntimeError(f"Episode extraction finished but output is incomplete: {episode_output_dir}")

    if _env_flag("ROBOTWIN_EXTRACT_KEEP_RAW", False):
        print(f"Keeping raw episode file: {hdf5_path}")
    else:
        _delete_raw_episode_files(args, episode_idx, hdf5_path)


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path_yml = f"./task_config/{task_config}.yml"
    config_path_yaml = f"./task_config/{task_config}.yaml"
    if os.path.exists(config_path_yml):
        config_path = config_path_yml
    elif os.path.exists(config_path_yaml):
        config_path = config_path_yaml
    else:
        raise FileNotFoundError(
            f"Task config not found: {config_path_yml} or {config_path_yaml}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, fail_num, fail_count, seed_list = 0, 0, 0, []
    success_retry_limit = int(args.get("success_retry_limit", 100))
    max_retry_no_success = args.get("max_retry_no_success", success_retry_limit)
    collect_failed_data = bool(args.get("collect_failed_data", False))
    failed_episode_max_frames = args.get("failed_episode_max_frames", 100)
    max_failed_episodes = _max_failed_episode_count(args)
    aborted_no_success = False
    initial_epid = None
    allow_failed_fill = False

    def enable_failed_fill(reason):
        nonlocal allow_failed_fill
        if allow_failed_fill:
            return
        allow_failed_fill = True
        print(
            "WARN: failed episode quota reached and no new success was found after "
            f"{success_retry_limit} attempt(s); allowing failed episodes to fill "
            f"remaining slots to episode_num={args['episode_num']}. Reason: {reason}"
        )

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    print(f"Collect failed episodes: {collect_failed_data}")
    if collect_failed_data:
        print(f"Failed episode max frames: {failed_episode_max_frames}")
        print(
            f"Failed episode quota: {max_failed_episodes} / {args['episode_num']} "
            f"({max_failed_episodes / int(args['episode_num']):.1%})"
        )
        print(f"Success retry limit after failed quota is full: {success_retry_limit}")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(_seed_path(args["save_path"])):
            with open(_seed_path(args["save_path"]), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {len(seed_list)}")
        else:
            epid = int(time.time())
            print(f"Using current time as initial seed: {epid}")

        episode_statuses = _load_episode_statuses(args["save_path"], seed_list)
        initial_epid = epid

        existing_failed_num = _count_failed(episode_statuses)
        if existing_failed_num > max_failed_episodes:
            print(
                f"WARN: existing seed status already has {existing_failed_num} failed episodes, "
                f"above quota {max_failed_episodes}."
            )

        while len(seed_list) < args["episode_num"]:
            episode_idx = len(seed_list)
            try:
                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=epid, **args)
                TASK_ENV.play_once()

                episode_success = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
                failed_count_after_update = _failed_count_after_update(
                    episode_statuses, episode_idx, episode_success
                )
                can_save_failed_episode = (
                    collect_failed_data
                    and TASK_ENV.plan_success
                    and (allow_failed_fill or failed_count_after_update <= max_failed_episodes)
                )
                can_save_episode = episode_success or (not episode_success and can_save_failed_episode)

                if can_save_episode:
                    result = "success" if episode_success else "fail"
                    print(f"simulate data episode {episode_idx} {result}! (seed = {epid})")
                    seed_list.append(epid)
                    _set_episode_status(episode_statuses, episode_idx, epid, episode_success)
                    TASK_ENV.save_traj_data(episode_idx)
                    fail_count = 0
                else:
                    if (
                        collect_failed_data
                        and TASK_ENV.plan_success
                        and failed_count_after_update > max_failed_episodes
                    ):
                        reason = f"failed episode quota reached ({max_failed_episodes}/{args['episode_num']})"
                    else:
                        reason = "unsavable trajectory"
                    print(f"simulate data episode {episode_idx} fail! (seed = {epid}), skip {reason}")
                    fail_num += 1
                    fail_count += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {len(seed_list)} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                fail_count += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {len(seed_list)} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                fail_count += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            quota_is_full = collect_failed_data and _count_failed(episode_statuses) >= max_failed_episodes
            if quota_is_full and not allow_failed_fill and fail_count >= success_retry_limit:
                enable_failed_fill("seed collection could not find another successful episode")
                fail_count = 0
            elif (
                not allow_failed_fill
                and max_retry_no_success is not None
                and fail_count >= max_retry_no_success
            ):
                print(
                    f"Reached {fail_count} failed attempts without any saved episode, "
                    f"abort collection for task {args['task_name']}."
                )
                aborted_no_success = True
                break

            epid += 1

            _write_seed_list(args["save_path"], seed_list)
            _save_episode_statuses(args["save_path"], episode_statuses)

        if aborted_no_success:
            return

        total_tries = epid - initial_epid if initial_epid is not None else 0
        success_num = _count_success(episode_statuses)
        saved_fail_num = _count_failed(episode_statuses)
        print(
            f"\nComplete simulation, saved \033[92m{success_num}\033[0m success / "
            f"\033[91m{saved_fail_num}\033[0m fail episodes, "
            f"skipped \033[91m{fail_num}\033[0m attempts / {total_tries} tries \n"
        )
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(_seed_path(args["save_path"]), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]
        episode_statuses = _load_episode_statuses(args["save_path"], seed_list)
        existing_failed_num = _count_failed(episode_statuses)
        if existing_failed_num > max_failed_episodes:
            print(
                f"WARN: saved seed status has {existing_failed_num} failed episodes, "
                f"above quota {max_failed_episodes}."
            )

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        def regenerate_missing_traj_data(episode_idx, force=False, initial_failed_attempts=0):
            nonlocal epid, fail_num, fail_count, seed_list, episode_statuses, allow_failed_fill

            traj_path = _traj_data_path(args["save_path"], episode_idx)
            if os.path.exists(traj_path) and not force:
                return True

            if episode_idx >= len(seed_list):
                return False

            original_need_plan = args["need_plan"]
            original_save_data = args["save_data"]
            original_render_freq = args["render_freq"]
            args["need_plan"] = True
            args["save_data"] = False
            args["render_freq"] = 0

            next_seed = max(max(seed_list) + 1 if seed_list else int(time.time()), epid)
            candidate_seed = seed_list[episode_idx]
            tried_existing_seed = force
            attempts = initial_failed_attempts

            try:
                while True:
                    if tried_existing_seed:
                        candidate_seed = next_seed
                        next_seed += 1
                        epid = max(epid, next_seed)
                    tried_existing_seed = True

                    try:
                        print(
                            f"Regenerating missing trajectory for episode {episode_idx} "
                            f"(seed = {candidate_seed})"
                        )
                        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=candidate_seed, **args)
                        TASK_ENV.play_once()
                        attempts += 1

                        episode_success = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
                        failed_count_after_update = _failed_count_after_update(
                            episode_statuses, episode_idx, episode_success
                        )
                        replacing_old_failure = (
                            episode_idx < len(episode_statuses)
                            and not bool(episode_statuses[episode_idx].get("success", True))
                        )
                        can_save_failed_episode = (
                            collect_failed_data
                            and TASK_ENV.plan_success
                            and (
                                allow_failed_fill
                                or replacing_old_failure
                                or failed_count_after_update <= max_failed_episodes
                            )
                        )
                        if (
                            not episode_success
                            and not can_save_failed_episode
                            and collect_failed_data
                            and TASK_ENV.plan_success
                            and failed_count_after_update > max_failed_episodes
                            and attempts >= success_retry_limit
                        ):
                            enable_failed_fill(
                                f"episode {episode_idx} trajectory regeneration exceeded retry limit"
                            )
                            can_save_failed_episode = True

                        can_save_episode = episode_success or (not episode_success and can_save_failed_episode)

                        if can_save_episode:
                            if candidate_seed != seed_list[episode_idx]:
                                seed_list[episode_idx] = candidate_seed
                                _write_seed_list(args["save_path"], seed_list)
                            _set_episode_status(episode_statuses, episode_idx, candidate_seed, episode_success)
                            _save_episode_statuses(args["save_path"], episode_statuses)
                            TASK_ENV.save_traj_data(episode_idx)
                            TASK_ENV.close_env()
                            fail_count = 0
                            result = "success" if episode_success else "fail"
                            print(
                                f"Regenerated trajectory for episode {episode_idx} "
                                f"{result}! (seed = {candidate_seed})"
                            )
                            return True

                        fail_num += 1
                        fail_count += 1
                        TASK_ENV.close_env()
                        print(
                            f"Regenerate trajectory episode {episode_idx} failed "
                            f"(seed = {candidate_seed}), trying next seed"
                        )

                    except UnStableError as e:
                        attempts += 1
                        print(" -------------")
                        print(f"regenerate trajectory episode {episode_idx} fail! (seed = {candidate_seed})")
                        print("Error: ", e)
                        print(" -------------")
                        fail_num += 1
                        fail_count += 1
                        TASK_ENV.close_env()
                        time.sleep(0.3)
                    except Exception as e:
                        attempts += 1
                        print(" -------------")
                        print(f"regenerate trajectory episode {episode_idx} fail! (seed = {candidate_seed})")
                        print("Error: ", e)
                        print(" -------------")
                        fail_num += 1
                        fail_count += 1
                        TASK_ENV.close_env()
                        time.sleep(1)

                    quota_is_full = (
                        collect_failed_data
                        and _count_failed(episode_statuses) >= max_failed_episodes
                    )
                    if quota_is_full and not allow_failed_fill and attempts >= success_retry_limit:
                        enable_failed_fill(
                            f"episode {episode_idx} trajectory regeneration could not find success"
                        )
                    elif (
                        not allow_failed_fill
                        and max_retry_no_success is not None
                        and fail_count >= max_retry_no_success
                    ):
                        raise RuntimeError(
                            f"Reached {fail_count} failed attempts while regenerating "
                            f"episode {episode_idx} for task {args['task_name']}."
                        )
            finally:
                args["need_plan"] = original_need_plan
                args["save_data"] = original_save_data
                args["render_freq"] = original_render_freq

        def episode_already_done(idx):
            if exist_hdf5(idx):
                return True
            return _episode_extract_enabled() and _episode_output_complete(args, idx)

        while episode_already_done(st_idx):
            st_idx += 1

        target_episode_num = min(args["episode_num"], len(seed_list))
        if target_episode_num < args["episode_num"]:
            print(
                f"WARN: only {target_episode_num} saved seeds are available, "
                f"less than requested episode_num={args['episode_num']}."
            )

        for episode_idx in range(st_idx, target_episode_num):
            while True:
                print(f"\033[34mTask name: {args['task_name']}\033[0m")

                if episode_already_done(episode_idx):
                    print(f"Episode {episode_idx} already complete, skip.")
                    break

                regenerate_missing_traj_data(episode_idx)

                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

                traj_data = TASK_ENV.load_tran_data(episode_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                TASK_ENV.set_path_lst(args)

                info_file_path = os.path.join(args["save_path"], "scene_info.json")

                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)

                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                info = TASK_ENV.play_once()
                episode_success = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
                info["success"] = episode_success
                info["result"] = "success" if episode_success else "fail"
                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4)

                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                max_frames = None if episode_success else failed_episode_max_frames
                failed_count_after_update = _failed_count_after_update(
                    episode_statuses, episode_idx, episode_success
                )
                if (
                    not episode_success
                    and failed_count_after_update > max_failed_episodes
                    and not allow_failed_fill
                ):
                    TASK_ENV.remove_data_cache()
                    print(
                        f"WARN: episode {episode_idx} failed and would exceed failed episode quota "
                        f"({failed_count_after_update}>{max_failed_episodes}); resampling for success."
                    )
                    regenerate_missing_traj_data(
                        episode_idx,
                        force=True,
                        initial_failed_attempts=1,
                    )
                    continue

                TASK_ENV.merge_pkl_to_hdf5_video(episode_success=episode_success, max_frames=max_frames)
                TASK_ENV.remove_data_cache()
                _set_episode_status(episode_statuses, episode_idx, seed_list[episode_idx], episode_success)
                _save_episode_statuses(args["save_path"], episode_statuses)
                hdf5_path = os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5")
                _postprocess_episode(args, episode_idx, hdf5_path)

                if not episode_success and not collect_failed_data:
                    raise AssertionError("Collect Error")

                break

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
