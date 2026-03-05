# Vertex AI Training - Quick Start

## First Time Setup (Each Teammate)

```bash
# 1. Clone repo
git clone https://github.com/sttawm/emg2qwerty.git
cd emg2qwerty

# 2. Run setup (creates your GCP project)
cd bootstrap && ./setup_teammate.sh
```

## Every Training Session

```bash
# Just cd to repo and run (configs auto-load)
cd /path/to/emg2qwerty
python vertex_submit.py

# With custom args
python vertex_submit.py user=generic trainer.max_epochs=200

# Different GPU
python vertex_submit.py --gpu NVIDIA_TESLA_V100 user=generic
```

## View Results

- **Jobs**: https://console.cloud.google.com/vertex-ai/training/custom-jobs
- **Logs/TensorBoard**: https://console.cloud.google.com/vertex-ai/experiments

## Common Commands

```bash
# List your jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1

# Different GPU
python vertex_submit.py --gpu NVIDIA_TESLA_V100

# Spot instances (cheaper)
python vertex_submit.py --spot
```

That's it! 🚀
