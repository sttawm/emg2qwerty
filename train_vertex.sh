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
    echo "ERROR: SHARED_DATA_BUCKET not set. Cannot download GCS data."
    exit 1
fi
if [ -z "$SHARED_LOGS_BUCKET" ]; then
    echo "ERROR: SHARED_LOGS_BUCKET not set. Cannot sync logs to GCS."
    exit 1
fi
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "WARNING: EXPERIMENT_NAME not set. Using 'default_experiment'."
    export EXPERIMENT_NAME="default_experiment"
fi

echo "Starting Vertex AI training job"
echo "Experiment: $EXPERIMENT_NAME"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Download data from GCS instead of mounting (FUSE not available in Vertex AI)
echo "Downloading data from GCS..."
mkdir -p /tmp/data
echo "Copying gs://${SHARED_DATA_BUCKET}/data/ to /tmp/data/"

if gsutil -m rsync -r gs://${SHARED_DATA_BUCKET}/data/ /tmp/data/; then
    echo "Data downloaded successfully"
    echo "Dataset files:"
    ls -lh /tmp/data/ | head -10
    echo "..."
    echo "Total data size: $(du -sh /tmp/data/ | cut -f1)"
    echo "File count: $(ls -1 /tmp/data/*.hdf5 2>/dev/null | wc -l) .hdf5 files"
else
    echo "ERROR: Failed to download data from GCS"
    exit 1
fi

# Check if running on GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "Training arguments: $@"

# Start background process to continuously sync logs to GCS
echo "Starting background log sync to GCS..."
(
    while true; do
        sleep 30
        # Check both /tmp/logs and /app/logs for compatibility
        if [ -d "/tmp/logs" ]; then
            echo "[$(date)] Syncing /tmp/logs to gs://${SHARED_LOGS_BUCKET}/logs/"
            gsutil -m rsync -r /tmp/logs/ gs://${SHARED_LOGS_BUCKET}/logs/ || echo "Warning: Log sync failed"
        fi
        if [ -d "/app/logs" ]; then
            echo "[$(date)] Syncing /app/logs to gs://${SHARED_LOGS_BUCKET}/logs/"
            gsutil -m rsync -r /app/logs/ gs://${SHARED_LOGS_BUCKET}/logs/ || echo "Warning: Log sync failed"
        fi
    done
) &
LOG_SYNC_PID=$!

# Run training with cluster=vertex
# Explicitly override dataset.root to use downloaded data location
# Override hydra output directory to ensure logs go to /tmp/logs for syncing
python -m emg2qwerty.train \
  cluster=vertex \
  dataset.root=/tmp/data \
  hydra.run.dir=/tmp/logs/${EXPERIMENT_NAME}/${now:%Y-%m-%d}/${now:%H-%M-%S} \
  "$@"

echo "Training completed"

# Kill background sync process
kill $LOG_SYNC_PID 2>/dev/null || true

# Final sync of logs to GCS
echo "Final log sync to GCS..."
SYNCED=false
if [ -d "/tmp/logs" ]; then
    echo "Syncing /tmp/logs to GCS..."
    gsutil -m rsync -r /tmp/logs/ gs://${SHARED_LOGS_BUCKET}/logs/
    SYNCED=true
fi
if [ -d "/app/logs" ]; then
    echo "Syncing /app/logs to GCS..."
    gsutil -m rsync -r /app/logs/ gs://${SHARED_LOGS_BUCKET}/logs/
    SYNCED=true
fi
if [ "$SYNCED" = true ]; then
    echo "Logs synced to gs://${SHARED_LOGS_BUCKET}/logs/"
else
    echo "Warning: No logs found at /tmp/logs or /app/logs"
fi

echo "Training job finished"
