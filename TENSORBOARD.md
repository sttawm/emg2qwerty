# TensorBoard Setup and Usage

This guide covers how to use TensorBoard for monitoring training locally and on Vertex AI.

## Local TensorBoard

### Option 1: Direct Launch

```bash
# Start TensorBoard pointing to your logs directory
tensorboard --logdir=./logs

# Open in browser
# http://localhost:6006
```

### Option 2: Launch During Training

Add to your terminal before training:

```bash
# Terminal 1: Start TensorBoard
tensorboard --logdir=./logs --port 6006

# Terminal 2: Start training
./train_local.sh
```

### Option 3: Using the Script

```bash
# Create a helper script
cat > view_tensorboard.sh << 'EOF'
#!/bin/bash
tensorboard --logdir=./logs --port 6006 --bind_all
EOF

chmod +x view_tensorboard.sh

# Run it
./view_tensorboard.sh
```

## Vertex AI TensorBoard

Vertex AI provides managed TensorBoard instances that automatically sync with your GCS logs.

### Step 1: Create Vertex AI TensorBoard Instance

```bash
# Set environment variables
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export TENSORBOARD_NAME="emg2qwerty-tensorboard"
export GCS_BUCKET="your-bucket-name"

# Create TensorBoard instance
gcloud ai tensorboards create \
  --display-name=${TENSORBOARD_NAME} \
  --region=${REGION}

# Get the TensorBoard resource name (save this)
export TENSORBOARD_ID=$(gcloud ai tensorboards list \
  --region=${REGION} \
  --filter="displayName:${TENSORBOARD_NAME}" \
  --format="value(name)")

echo "TensorBoard ID: ${TENSORBOARD_ID}"
```

### Step 2: Create Experiment

```bash
# Create an experiment to organize your training runs
gcloud ai tensorboard-experiments create emg2qwerty-training \
  --tensorboard=${TENSORBOARD_ID} \
  --region=${REGION}
```

### Step 3: Upload Logs from GCS

```bash
# Upload logs from a completed training run
gcloud ai tensorboard-experiments upload \
  --tensorboard=${TENSORBOARD_ID} \
  --tensorboard-experiment=emg2qwerty-training \
  --region=${REGION} \
  --source-logs=gs://${GCS_BUCKET}/logs/YYYY-MM-DD/HH-MM-SS/
```

### Step 4: View TensorBoard

```bash
# Get the web URL
gcloud ai tensorboards describe ${TENSORBOARD_ID} \
  --region=${REGION} \
  --format="value(url)"

# Or open directly in console
echo "https://console.cloud.google.com/vertex-ai/experiments/tensorboard/${TENSORBOARD_ID}?project=${PROJECT_ID}"
```

## Continuous Monitoring with Vertex AI

### Enable TensorBoard During Training

Modify your training job submission to include TensorBoard:

```bash
# Submit job with TensorBoard enabled
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=emg2qwerty-train-$(date +%Y%m%d-%H%M%S) \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=${IMAGE_URI} \
  --tensorboard=${TENSORBOARD_ID} \
  --service-account=vertex-training-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

Or update `vertex_submit.py` to include:

```python
# Add to the gcloud command in submit_training_job():
cmd.append(f"--tensorboard={TENSORBOARD_ID}")
```

## Automatic Log Sync with gsutil

For real-time syncing of local logs to GCS:

```bash
# Create sync script
cat > sync_logs.sh << 'EOF'
#!/bin/bash
GCS_BUCKET="your-bucket-name"

# Watch for new logs and sync to GCS
while true; do
  gsutil -m rsync -r ./logs gs://${GCS_BUCKET}/local-logs/
  sleep 60  # Sync every minute
done
EOF

chmod +x sync_logs.sh

# Run in background
./sync_logs.sh &
```

## TensorBoard Features

### What You Can Monitor

1. **Training Metrics**
   - Loss (train/val/test)
   - Character Error Rate (CER)
   - Learning rate

2. **Graphs**
   - Model architecture
   - Computational graph

3. **Distributions**
   - Weight distributions
   - Gradient distributions

4. **Scalars**
   - Custom metrics
   - System metrics (if enabled)

### Comparing Multiple Runs

```bash
# Local: Point to parent directory containing multiple runs
tensorboard --logdir=./logs

# Vertex AI: TensorBoard automatically compares runs in the same experiment
```

### Custom Metrics

Add custom metrics in your training code:

```python
# In lightning.py
from pytorch_lightning.loggers import TensorBoardLogger

# Add to your module
def training_step(self, batch, batch_idx):
    # ... existing code ...

    # Log custom metrics
    self.log('custom/my_metric', value)

    return loss
```

## Advanced: Remote TensorBoard Access

Access TensorBoard running on a remote machine:

```bash
# On remote machine
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

# On local machine (SSH tunnel)
ssh -N -L 6006:localhost:6006 user@remote-host

# Open http://localhost:6006 in your browser
```

## Troubleshooting

### TensorBoard Not Showing Metrics

1. Check logs are being written:
   ```bash
   ls -la ./logs/YYYY-MM-DD/HH-MM-SS/lightning_logs/version_0/
   ```

2. Verify TensorBoard is pointing to correct directory:
   ```bash
   tensorboard --logdir=./logs --inspect
   ```

3. Force reload in browser (Ctrl+F5 or Cmd+Shift+R)

### Vertex AI TensorBoard Not Updating

1. Check GCS permissions:
   ```bash
   gsutil ls gs://${GCS_BUCKET}/logs/
   ```

2. Verify logs are being written to GCS:
   ```bash
   gsutil ls -r gs://${GCS_BUCKET}/logs/YYYY-MM-DD/
   ```

3. Manually trigger upload:
   ```bash
   gcloud ai tensorboard-experiments upload \
     --tensorboard=${TENSORBOARD_ID} \
     --tensorboard-experiment=emg2qwerty-training \
     --region=${REGION} \
     --source-logs=gs://${GCS_BUCKET}/logs/
   ```

### Port Already in Use

```bash
# Kill existing TensorBoard
pkill -f tensorboard

# Or use a different port
tensorboard --logdir=./logs --port=6007
```

## Cost Optimization

**Vertex AI TensorBoard Pricing:**
- Managed TensorBoard instance: ~$0.10/hour
- Storage: GCS standard pricing

**Tips:**
- Delete old TensorBoard instances when not in use
- Use GCS lifecycle policies to delete old logs
- Consider local TensorBoard for development

```bash
# Delete TensorBoard instance when done
gcloud ai tensorboards delete ${TENSORBOARD_ID} --region=${REGION}
```

## Quick Reference

```bash
# Local TensorBoard
tensorboard --logdir=./logs

# View specific run
tensorboard --logdir=./logs/2024-03-15/10-30-00

# Vertex AI TensorBoard URL
https://console.cloud.google.com/vertex-ai/experiments/tensorboard

# Upload logs to Vertex AI
gcloud ai tensorboard-experiments upload \
  --tensorboard=${TENSORBOARD_ID} \
  --tensorboard-experiment=emg2qwerty-training \
  --region=${REGION} \
  --source-logs=gs://${GCS_BUCKET}/logs/
```
