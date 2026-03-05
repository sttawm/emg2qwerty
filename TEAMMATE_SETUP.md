# EMG2QWERTY - Teammate Setup Guide

Welcome to the team! This guide will get you training on Vertex AI in ~10 minutes.

## Prerequisites

- Google account (personal Gmail works fine)
- Access granted by team lead (they ran `./grant_access.sh your-email@gmail.com`)

## Setup (One-Time, ~10 minutes)

### 1. Clone Repository

```bash
git clone https://github.com/sttawm/emg2qwerty.git
cd emg2qwerty
```

### 2. Run Setup Script

```bash
cd bootstrap
./setup_teammate.sh
```

This will:
- Create your personal GCP project (you'll need to enable billing)
- Setup Artifact Registry for your Docker images
- Verify you can access the shared team data and logs

**Follow the prompts** - it will ask you to:
- Choose a project ID (or accept the suggested one)
- Enable billing (required for GPU compute)

### 3. That's It!

You're ready to train.

## Training (Every Session)

```bash
# Go to repo
cd /path/to/emg2qwerty

# Submit training (default: 40 epochs, single_user, T4 GPU)
python vertex_submit.py

# With custom args (just like local training)
python vertex_submit.py user=generic trainer.max_epochs=100

# Different GPU
python vertex_submit.py --gpu NVIDIA_TESLA_V100

# Spot instances (cheaper, ~70% savings)
python vertex_submit.py --spot
```

## Monitoring

**View your jobs:**
https://console.cloud.google.com/vertex-ai/training/custom-jobs

**View all team experiments (TensorBoard):**
https://console.cloud.google.com/vertex-ai/experiments

**Stream logs:**
```bash
gcloud ai custom-jobs list --region=us-central1
gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1
```

## Billing

**You pay for:**
- Your GPU compute time only (~$0.35/hour for T4, ~$2.50/hour for V100)
- Use spot instances with `--spot` to save ~70%

**Team lead pays for:**
- Dataset storage (~$0.02/GB/month)
- Logs storage

**Free credits:**
- New GCP accounts get $300 free credits for 90 days

## Common Commands

```bash
# Default training
python vertex_submit.py

# Train on all users
python vertex_submit.py user=generic

# More epochs
python vertex_submit.py trainer.max_epochs=200

# Faster GPU
python vertex_submit.py --gpu NVIDIA_TESLA_V100

# Cheaper (spot instances)
python vertex_submit.py --spot

# Parameter sweep
python vertex_submit.py --multirun trainer.max_epochs=50,100,200
```

## Troubleshooting

**"GCP_PROJECT_ID not set"**
- Run: `cd bootstrap && ./setup_teammate.sh`

**"Permission denied" on buckets**
- Ask team lead to run: `./grant_access.sh your-email@gmail.com`

**Docker not running**
- Open Docker Desktop: `open -a Docker`

**Need help?**
- Check: `TRAINING_GUIDE.md` for detailed docs
- Or: `VERTEX_QUICK_START.md` for quick reference

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
