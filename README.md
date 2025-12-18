GPU_ID=0 ROBOT=piper CONFIG=custom_piper SAVE_ROOT=./save_path bash batch_collect_data.sh

bash test_one_task_one_robot.sh --gpu 0 --robot franka-panda --task beat_block_hammer