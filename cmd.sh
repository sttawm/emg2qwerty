#!/bin/bash
python train_remote.py \
  --spot \
  --gpu NVIDIA_TESLA_V100 \
  --experiment learned_soft_rotation_per_band_lstm_1x256 \
  trainer.max_epochs=80
