



 采集命令：

```bash
  bash collect_data.sh adjust_bottle 3d_aloha_dataset 0
```

批量采集全部 50 个任务（参数为 GPU ID）：

```bash
bash batch_collect_3d_aloha.sh 0
```

  转换命令：

```bash
conda activate xpolicy
export HF_LEROBOT_HOME=/home/shengyu/Documents/PENG/github/RoboTwin/lerobot

python XPolicyLab/scripts/transform_lerobot_v30_format.py \
    "3d_aloha_dataset.*.aloha_agilex" \
    --repo_id robotwin_3d_dataset_aloha_agilex_v30 \
    --max_episode 50 \
    --include_third_view \
    --include_depth \
    --include_camera_calibration
```

```bash
export HF_LEROBOT_HOME=/home/shengyu/Documents/PENG/github/RoboTwin/lerobot

# LeRobot v2.1 — all demo_clean tasks
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.*.aloha_agilex" \
  --repo_id robotwin_demo_clean_aloha_agilex \
  --max_episode 50

# LeRobot v3.0 — same selection
python XPolicyLab/scripts/transform_lerobot_v30_format.py \
  "demo_clean.*.aloha_agilex" \
  --repo_id robotwin_demo_clean_aloha_agilex_v30 \
  --max_episode 50

# Single task
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.beat_block_hammer.aloha_agilex" \
  --repo_id beat_block_hammer_demo_clean

```
