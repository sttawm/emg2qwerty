#!/bin/bash
# Local training script for EMG2QWERTY
#
# Usage:
#   ./train_local.sh                           # Train with default config
#   ./train_local.sh trainer.max_epochs=50     # Override epochs
#   ./train_local.sh user=generic              # Train on all users

set -e

export HYDRA_FULL_ERROR=1

# Enable MPS fallback for Mac M1/M2/M3 chips
# Allows unsupported operations to fall back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run training with cluster=local
python -m emg2qwerty.train cluster=local "$@"
