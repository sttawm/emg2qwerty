# EMG2QWERTY - Teammate Setup Guide

Welcome to the team! This guide walks you through local setup first, then cloud training.

## Quick Reference

### Local Training
```bash
./train_local.sh user=single_user trainer.max_epochs=50
```

### Remote Training (Recommended)
```bash
python train_remote.py --spot --experiment my_experiment trainer.max_epochs=50
```

---

## Prerequisites

- Google account (personal Gmail works fine)
- Access granted by team lead (they ran `./grant_access.sh your-email@gmail.com`)

---

## Part 1: Local Setup (~5 minutes)

### 1. Clone Repository

```bash
git clone https://github.com/sttawm/emg2qwerty.git
cd emg2qwerty
```

### 2. Create Conda Environment

```bash
conda create -n emg2qwerty python=3.11 -y
conda activate emg2qwerty
pip install -r requirements.txt
```

### 3. Install Google Cloud SDK

**Mac (Homebrew):**
```bash
brew install google-cloud-sdk
```

**Linux/Other:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Verify:**
```bash
gcloud --version
```

### 4. Test Local Training (Optional but Recommended)

If you have access to the dataset locally:

```bash
# Make sure data is in ./data/
ls ./data/*.hdf5

# Run a quick test (5 epochs)
./train_local.sh trainer.max_epochs=5

# Monitor in TensorBoard (separate terminal)
tensorboard --logdir=./logs
# Open http://localhost:6006
```

This validates your environment before moving to cloud training.

---

## Part 2: Cloud Setup (~10 minutes)

### 5. Run GCP Setup Script

```bash
cd bootstrap
./setup_teammate.sh
```

This will:
- Create your personal GCP project (you'll need to enable billing)
- Setup Artifact Registry for your Docker images
- Verify you can access the shared team data and logs
- Grant your Vertex AI service account permissions
- **Grant you TensorBoard access to view remote logs**

**Follow the prompts** - it will ask you to:
- Choose a project ID (or accept the suggested one)
- Enable billing (required for GPU compute)

### 6. Setup TensorBoard for Remote Monitoring

```bash
# Install gcsfs (in your conda environment)
pip install gcsfs
```

Now you can monitor remote training in real-time:

```bash
tensorboard --logdir=gs://emg2qwerty-team-logs/logs/
# Open http://localhost:6006
```

---

## Training Remotely

### Basic Usage

```bash
# Default training (40 epochs, single_user, T4 GPU, spot instance)
python train_remote.py --spot --experiment baseline_v1

# Train on all users
python train_remote.py --spot --experiment multi_user user=generic

# More epochs
python train_remote.py --spot --experiment long_run trainer.max_epochs=200

# Different GPU (if you have quota)
python train_remote.py --spot --experiment v100_test --gpu NVIDIA_TESLA_V100
```

### Common Commands

```bash
# Baseline training
python train_remote.py --spot --experiment baseline

# Train on all users
python train_remote.py --spot --experiment generic_model user=generic

# Longer training
python train_remote.py --spot --experiment epoch_200 trainer.max_epochs=200

# Faster GPU with spot
python train_remote.py --spot --experiment v100_baseline --gpu NVIDIA_TESLA_V100

# Experiment with learning rates
python train_remote.py --spot --experiment lr_0001 optimizer.lr=0.001
```

---

## Monitoring Your Jobs

### TensorBoard (Real-time Metrics)

```bash
tensorboard --logdir=gs://emg2qwerty-team-logs/logs/
# Open http://localhost:6006
```

Logs sync every 30 seconds, so you'll see metrics update in near real-time.

### View Jobs in Console

https://console.cloud.google.com/vertex-ai/training/custom-jobs

### Stream Job Logs

```bash
# List your jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs from a specific job
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

---

## Billing

**You pay for:**
- Your GPU compute time only (~$0.35/hour for T4, ~$2.50/hour for V100)
- **Use spot instances with `--spot` to save ~70%** (all examples above use --spot)

**Spot instances:**
- 60-70% cheaper than regular instances
- Can be interrupted (job will restart automatically)
- Perfect for training jobs

---

## What Gets Shared

✅ **Shared (entire team):**
- Dataset: `gs://emg2qwerty-team-data/data/`
- Logs: `gs://emg2qwerty-team-logs/logs/`
- TensorBoard (everyone sees all experiments)

🔒 **Private (just you):**
- Your GCP project (your billing)
- Your Docker images
- Your job names (prefixed with your email)

---

**Happy training!** 🚀
