#!/bin/bash
CHECKPOINT="gs://emg2qwerty-team-logs/logs/learned_soft_rotation_per_band_v2/checkpoints/last.ckpt"

python train_remote.py \
  --spot \
  --gpu NVIDIA_TESLA_V100 \
  --experiment learned_soft_rotation_v2_timeseries_every_single_one_v2 \
  lr_scheduler=cosine_annealing \
  optimizer.lr=1e-6 \
  trainer.max_epochs=10 \
  "checkpoint='${CHECKPOINT}'"
