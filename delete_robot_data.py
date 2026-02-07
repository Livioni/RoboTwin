#!/usr/bin/env python3
"""
Script to delete robot_data folders from all episodes in the datasets directory.
This script traverses the directory structure: datasets/[robot]/[task]/[episode]/robot_data/
and removes all robot_data folders.
"""

import os
import shutil
import argparse
from pathlib import Path


def delete_robot_data_folders(datasets_root: str, dry_run: bool = False):
    """
    Delete all robot_data folders in the datasets directory structure.

    Args:
        datasets_root: Root path of the datasets directory
        dry_run: If True, only print what would be deleted without actually deleting
    """
    datasets_path = Path(datasets_root)

    if not datasets_path.exists():
        print(f"Error: Datasets directory '{datasets_root}' does not exist.")
        return

    if not datasets_path.is_dir():
        print(f"Error: '{datasets_root}' is not a directory.")
        return

    total_deleted = 0
    total_errors = 0

    print(f"{'DRY RUN: ' if dry_run else ''}Scanning datasets directory: {datasets_root}")

    # 遍历所有机器文件夹
    for robot_dir in sorted(datasets_path.iterdir()):
        if not robot_dir.is_dir():
            continue

        print(f"\nProcessing robot: {robot_dir.name}")

        # 遍历所有任务文件夹
        for task_dir in sorted(robot_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            print(f"  Processing task: {task_dir.name}")

            # 遍历所有episode文件夹
            for episode_dir in sorted(task_dir.iterdir()):
                if not episode_dir.is_dir():
                    continue

                robot_data_path = episode_dir / "robot_data"

                if robot_data_path.exists():
                    if robot_data_path.is_dir():
                        if dry_run:
                            print(f"    Would delete: {robot_data_path}")
                            total_deleted += 1
                        else:
                            try:
                                shutil.rmtree(robot_data_path)
                                print(f"    Deleted: {robot_data_path}")
                                total_deleted += 1
                            except Exception as e:
                                print(f"    Error deleting {robot_data_path}: {e}")
                                total_errors += 1
                    else:
                        print(f"    Warning: {robot_data_path} exists but is not a directory")
                # 如果robot_data不存在，不做任何操作

    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"  Folders processed: {total_deleted}")
    if not dry_run:
        print(f"  Errors encountered: {total_errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Delete robot_data folders from all episodes in datasets directory"
    )
    parser.add_argument(
        "--datasets-root",
        default="datasets",
        help="Root directory of the datasets (default: datasets)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution)"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.confirm:
        response = input(
            f"This will permanently delete all robot_data folders in {args.datasets_root}. "
            "Are you sure? (type 'yes' to confirm): "
        )
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return

    delete_robot_data_folders(args.datasets_root, args.dry_run)


if __name__ == "__main__":
    main()