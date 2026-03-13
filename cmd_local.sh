#!/bin/bash
CHECKPOINT="gs://emg2qwerty-team-logs/logs/learned_soft_rotation_per_band_v2/checkpoints/last.ckpt"

python -m emg2qwerty.train \
  trainer.accelerator=cpu \
  trainer.max_epochs=1 \
  batch_size=32 \
  num_workers=8 \
  optimizer.lr=1e-6 \
  lr_scheduler=cosine_annealing \
  "checkpoint='${CHECKPOINT}'"
