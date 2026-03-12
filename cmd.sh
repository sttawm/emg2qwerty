#!/bin/bash
CHECKPOINT="gs://emg2qwerty-team-logs/logs/learned_soft_rotation_per_band_v2/checkpoints/last.ckpt"

python train_remote.py \
  --spot \
  --gpu NVIDIA_TESLA_V100 \
  --experiment learned_soft_rotation_timeseries_v3 \
  lr_scheduler=cosine_annealing \
  optimizer.lr=1e-6 \
  trainer.max_epochs=10 \
  "checkpoint='${CHECKPOINT}'"
