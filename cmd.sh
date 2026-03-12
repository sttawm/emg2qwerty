#!/bin/bash
python train_remote.py \
  --spot \
  --gpu NVIDIA_TESLA_V100 \
  --experiment learned_soft_rotation_inspect_by_time \
  lr_scheduler=cosine_annealing \
  optimizer.lr=1e-6 \
  trainer.max_epochs=10 \
  "checkpoint='gs://emg2qwerty-team-logs/logs/learned_soft_rotation_tdsconv/checkpoints/last.ckpt'" \
  datamodule.shuffle_train=false
