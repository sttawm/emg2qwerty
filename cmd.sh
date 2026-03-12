#!/bin/bash
python train_remote.py \
  --spot \
  --gpu NVIDIA_TESLA_V100 \
  --experiment learned_soft_rotation_per_band \
  trainer.max_epochs=70
