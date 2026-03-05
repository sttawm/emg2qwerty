# EMG2QWERTY Training Guide

Complete guide for training models locally and on Google Cloud Vertex AI.

---

## 🎯 Team Collaboration Setup

This project uses a **team collaboration workflow**:
- ✅ Shared data bucket (upload once, everyone uses it)
- ✅ Shared logs bucket (all experiments in one TensorBoard)
- ✅ Each person has their own project for billing

**See: [Team Collaboration Workflow](#team-collaboration-workflow) below**

---

## Quick Start

### Local Training (Mac M1/M2/M3)

```bash
# 1. Setup conda environment
conda create -n emg2qwerty python=3.11 -y
conda activate emg2qwerty
pip install -r requirements.txt

# 2. Ensure dataset is in ./data/
ls ./data/

# 3. Start TensorBoard (optional, in separate terminal)
tensorboard --logdir=./logs

# 4. Run training
./train_local.sh

# 5. Monitor at http://localhost:6006
```

### Vertex AI Training (Team Setup)

```bash
# 1. One-time setup (creates your GCP project)
cd bootstrap && ./setup_teammate.sh

# 2. Submit training (configs auto-load, builds Docker automatically)
cd /path/to/emg2qwerty
python vertex_submit.py

# With custom args
python vertex_submit.py user=generic trainer.max_epochs=200

# Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs
# View logs: https://console.cloud.google.com/vertex-ai/experiments
```

## Detailed Workflows

### Team Collaboration Workflow

This workflow enables the team to:
- Share one dataset (uploaded once to GCS)
- Share one logs location (all experiments in one TensorBoard)
- Use individual GCP projects for compute/billing

**Architecture:**
- **Shared Resources** (Team Lead's Project):
  - Data bucket: `gs://emg2qwerty-team-data/`
  - Logs bucket: `gs://emg2qwerty-team-logs/`
- **Personal Resources** (Each Teammate's Project):
  - Compute (Vertex AI jobs, billed to personal project)
  - Artifact Registry (Docker images)

**Setup:**
- **Team lead**: See `TEAM_LEAD_GUIDE.md`
- **Teammates**: See `TEAMMATE_SETUP.md`

### Local Development Workflow

**Step 1: Environment Setup**
```bash
# Clone repository
git clone https://github.com/sttawm/emg2qwerty.git
cd emg2qwerty

# Create conda environment
conda create -n emg2qwerty python=3.11 -y
conda activate emg2qwerty

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Data Preparation**
```bash
# Ensure data is in the correct location
ls ./data/
# Should see .hdf5 files

# If dataset is elsewhere, create symlink
ln -s /path/to/your/data ./data
```

**Step 3: Start TensorBoard** (Optional but recommended)
```bash
# Terminal 1: TensorBoard
tensorboard --logdir=./logs --port=6006

# Open browser to http://localhost:6006
```

**Step 4: Run Training**
```bash
# Terminal 2: Training

# Default training (150 epochs, single user)
./train_local.sh

# Custom epochs
./train_local.sh trainer.max_epochs=50

# Train on all users
./train_local.sh user=generic

# Resume from checkpoint
./train_local.sh checkpoint=./logs/2024-03-15/10-30-00/checkpoints/last.ckpt
```

**Step 5: Monitor Training**
- TensorBoard: http://localhost:6006
- Terminal: Watch loss/metrics printed
- Logs: `./logs/YYYY-MM-DD/HH-MM-SS/`
- Checkpoints: `./logs/YYYY-MM-DD/HH-MM-SS/checkpoints/`

**Step 6: Test Model**
```bash
./train_local.sh \
  checkpoint=./logs/2024-03-15/10-30-00/checkpoints/best.ckpt \
  train=False \
  decoder=ctc_greedy
```

## Configuration Options

### Training Arguments

All arguments follow Hydra's override syntax:

```bash
# Model architecture
./train_local.sh model.block_channels=[32,32,32,32]

# Training hyperparameters
./train_local.sh \
  trainer.max_epochs=100 \
  batch_size=64 \
  optimizer.lr=0.001

# Data configuration
./train_local.sh \
  user=generic \
  num_workers=8

# Hardware
./train_local.sh \
  trainer.accelerator=gpu \
  trainer.devices=2
```

### Configuration Files

Located in `config/`:

- `base.yaml` - Main configuration
- `user/single_user.yaml` - Single user training
- `user/generic.yaml` - Multi-user training
- `model/tds_conv_ctc.yaml` - Model architecture
- `cluster/local.yaml` - Local training config
- `cluster/vertex.yaml` - Vertex AI training config

### Environment Variables

Set automatically by bootstrap scripts:

```bash
# Shared team resources (from shared_config.env)
export SHARED_DATA_BUCKET="emg2qwerty-team-data"
export SHARED_LOGS_BUCKET="emg2qwerty-team-logs"
export GCP_REGION="us-central1"

# Personal resources (from .teammate_config.env)
export GCP_PROJECT_ID="your-name-emg2qwerty"
export ARTIFACT_REGISTRY="us-central1-docker.pkg.dev/your-name-emg2qwerty/emg2qwerty-training"
export TEAMMATE_NAME="your-name"
```

## Common Tasks

### Resume Training from Checkpoint

**Local:**
```bash
./train_local.sh checkpoint=./logs/2024-03-15/10-30-00/checkpoints/last.ckpt
```

**Vertex AI:**
```bash
python vertex_submit.py checkpoint=gs://bucket/logs/2024-03-15/10-30-00/checkpoints/last.ckpt
```

### Train on Multiple Users

```bash
# Local
./train_local.sh user=generic trainer.devices=1

# Vertex AI (recommended for multi-user)
python vertex_submit.py user=generic --machine n1-highmem-8
```

### Hyperparameter Tuning

```bash
# Try different learning rates
for lr in 0.0001 0.0005 0.001 0.005; do
  ./train_local.sh optimizer.lr=$lr trainer.max_epochs=50
done

# On Vertex AI, submit multiple jobs
for lr in 0.0001 0.0005 0.001; do
  python vertex_submit.py optimizer.lr=$lr
done
```

### Export Model for Inference

```bash
# Test model
./train_local.sh \
  checkpoint=./models/best.ckpt \
  train=False \
  decoder=ctc_beam

# Convert to TorchScript (if needed)
# Add export script in scripts/
```

## Debugging

### Local Debugging

```bash
# Enable full error traces
export HYDRA_FULL_ERROR=1
./train_local.sh

# Reduce batch size for memory issues
./train_local.sh batch_size=16

# Use CPU if GPU issues
./train_local.sh trainer.accelerator=cpu

# Quick test run
./train_local.sh trainer.max_epochs=1 batch_size=4
```

### Vertex AI Debugging

```bash
# Check Docker image builds locally
docker build -t test -f Dockerfile .
docker run test --help

# Test with small instance
python vertex_submit.py --machine n1-standard-2 trainer.max_epochs=1 batch_size=4

# Stream logs in real-time
gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1
```

## Performance Tips

### Local Training (Mac M1/M2/M3)

- Use MPS accelerator (automatically detected)
- Increase `num_workers` to 4-8 for faster data loading
- Monitor with Activity Monitor for memory usage
- Expected: ~4-8 hours for 150 epochs

### Vertex AI Training

- **T4 GPU**: Good balance of cost/performance (~$0.35/hour)
- **V100 GPU**: 2-3x faster but more expensive (~$2.48/hour)
- **Spot instances**: 60-70% cheaper, but can be preempted
- **Multi-GPU**: Use for `user=generic` training only

### Storage Optimization

```bash
# Delete old logs locally
find ./logs -type d -mtime +30 -exec rm -rf {} \;

# Delete old GCS logs
gsutil -m rm -r gs://${GCS_BUCKET}/logs/2024-01-*

# Lifecycle policy for GCS
gsutil lifecycle set lifecycle.json gs://${GCS_BUCKET}
```

## Troubleshooting

### "CUDA out of memory"

```bash
# Reduce batch size
./train_local.sh batch_size=16

# Or use gradient accumulation
./train_local.sh batch_size=16 trainer.accumulate_grad_batches=2
```

### "Dataset not found"

**Local:**
```bash
ls ./data/  # Check data exists
./train_local.sh dataset.root=/path/to/data
```

**Vertex AI:**
```bash
gsutil ls gs://${GCS_BUCKET}/data/  # Verify upload
```

### "Permission denied" on Vertex AI

```bash
# Check service account permissions
gcloud projects get-iam-policy $PROJECT_ID

# Grant storage access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:vertex-training-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

## Best Practices

1. **Development**: Train locally with small epochs first
2. **Full Training**: Use Vertex AI for production runs
3. **Versioning**: Tag Docker images with git commit hash
4. **Monitoring**: Always use TensorBoard
5. **Checkpointing**: Save frequently, keep top 3 models
6. **Cost**: Use spot instances for experiments
7. **Organization**: Use clear job names with dates

## Next Steps

- [ ] Run teammate setup: `cd bootstrap && ./setup_teammate.sh`
- [ ] Test local training
- [ ] Submit first Vertex AI job
- [ ] Setup TensorBoard monitoring

## Additional Resources

- **Teammate Setup**: `TEAMMATE_SETUP.md`
- **Team Lead Guide**: `TEAM_LEAD_GUIDE.md`
- **Quick Start**: `VERTEX_QUICK_START.md`
- **TensorBoard**: `TENSORBOARD.md`
- **Vertex AI Docs**: https://cloud.google.com/vertex-ai/docs/training/custom-training
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
