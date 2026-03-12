#!/bin/bash
python train_remote.py \
  --spot \
  --gpu NVIDIA_TESLA_V100 \
  --experiment learned_soft_rotation_tdsconv \
  trainer.max_epochs=70
