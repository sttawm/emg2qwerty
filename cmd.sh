#!/bin/bash
python train_remote.py \
  --spot \
  --gpu NVIDIA_TESLA_V100 \
  --experiment kenlm_fusion_v1 \
  --train-script "python train_fusion.py" \
  "fusion_module.acoustic_checkpoint='gs://emg2qwerty-team-logs/logs/lstm_1x256/checkpoints/epoch=61-step=7440.ckpt'"
