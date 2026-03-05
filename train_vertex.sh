#!/bin/bash
# Vertex AI training script for EMG2QWERTY
#
# This script runs inside a Docker container on Vertex AI
#
# Usage (in Docker container):
#   ./train_vertex.sh
#   ./train_vertex.sh trainer.max_epochs=100
#   ./train_vertex.sh user=generic

set -e

export HYDRA_FULL_ERROR=1

# Verify shared buckets are set
if [ -z "$SHARED_DATA_BUCKET" ]; then
    echo "WARNING: SHARED_DATA_BUCKET not set. Ensure bootstrap/shared_config.env is loaded."
fi
if [ -z "$SHARED_LOGS_BUCKET" ]; then
    echo "WARNING: SHARED_LOGS_BUCKET not set. Ensure bootstrap/shared_config.env is loaded."
fi

echo "Starting Vertex AI training job"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Check if running on GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "Training arguments: $@"

# Run training with cluster=vertex
python -m emg2qwerty.train cluster=vertex "$@"

echo "Training completed successfully"
