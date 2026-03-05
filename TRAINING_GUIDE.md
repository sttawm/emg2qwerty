# EMG2QWERTY Training Guide

Complete guide for training models locally and on Google Cloud Vertex AI.

**Shared Team Resources:**
- Data: `gs://emg2qwerty-team-data/`
- Logs: `gs://emg2qwerty-team-logs/`

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
python train_remote.py

# With custom args
python train_remote.py user=generic trainer.max_epochs=200

# 3. Monitor with TensorBoard (install gcsfs first: pip install gcsfs)
tensorboard --logdir=gs://emg2qwerty-team-logs/logs/
# Open http://localhost:6006

# View jobs: https://console.cloud.google.com/vertex-ai/training/custom-jobs
```

## Local Development Workflow

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

## Common Tasks

### Resume Training from Checkpoint

**Local:**
```bash
./train_local.sh checkpoint=./logs/2024-03-15/10-30-00/checkpoints/last.ckpt
```

**Vertex AI:**
```bash
python train_remote.py checkpoint=gs://bucket/logs/2024-03-15/10-30-00/checkpoints/last.ckpt
```

### Train on Multiple Users

```bash
# Local
./train_local.sh user=generic trainer.devices=1

# Vertex AI (recommended for multi-user)
python train_remote.py user=generic --machine n1-highmem-8
```

### Hyperparameter Tuning

```bash
# Try different learning rates
for lr in 0.0001 0.0005 0.001 0.005; do
  ./train_local.sh optimizer.lr=$lr trainer.max_epochs=50
done

# On Vertex AI, submit multiple jobs
for lr in 0.0001 0.0005 0.001; do
  python train_remote.py optimizer.lr=$lr
done
```

### Test Model

```bash
./train_local.sh \
  checkpoint=./models/best.ckpt \
  train=False \
  decoder=ctc_beam
```
