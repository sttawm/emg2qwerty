# Team Bootstrap Setup

This folder contains scripts to set up shared resources for the EMG2QWERTY team.

## Team Architecture

**Shared Resources (One Project):**
- Shared data bucket (uploaded once, read by all)
- Shared logs bucket (all teammates write their training logs here)
- One TensorBoard instance (shows all experiments)

**Per-Teammate (Individual Projects):**
- Own GCP project for billing/compute
- Own Artifact Registry for Docker images
- Own Vertex AI training jobs
- Access to shared buckets via IAM permissions

## Setup Workflow

### One-Time Setup (Team Lead)

**Step 1: Create shared resources**

```bash
# Run this script to create shared buckets
./setup_shared_resources.sh
```

This creates:
- Project: `emg2qwerty-team-shared` (or your choice)
- Data bucket: `gs://emg2qwerty-team-data/`
- Logs bucket: `gs://emg2qwerty-team-logs/`
- TensorBoard instance

**Step 2: Upload dataset**

```bash
# Upload dataset once to shared bucket
./upload_data.sh /path/to/local/data
```

### Per-Teammate Setup

**Team Lead: Grant access to new teammate**

```bash
# Grant access to teammate's Google account
./grant_access.sh teammate@example.com
```

**Teammate: Setup your environment**

```bash
# 1. Clone repository
git clone https://github.com/sttawm/emg2qwerty.git
cd emg2qwerty/bootstrap

# 2. Run teammate setup (creates your own project + connects to shared resources)
./setup_teammate.sh
```

This will:
- Create your personal GCP project (or use existing)
- Setup Artifact Registry in your project
- Configure access to shared data/logs buckets
- Set up environment variables

**Teammate: Start training**

```bash
# Build and submit training job
cd ..
python vertex_submit.py --build --submit
```

Your job will:
- Read data from shared bucket
- Write logs to shared bucket
- View all team experiments in Vertex AI TensorBoard UI

## Shared Configuration

The `shared_config.env` file contains team-wide settings:

```bash
# Shared resources (same for everyone)
SHARED_DATA_BUCKET=emg2qwerty-team-data
SHARED_LOGS_BUCKET=emg2qwerty-team-logs
SHARED_PROJECT_ID=emg2qwerty-team-shared

# Region (same for everyone)
GCP_REGION=us-central1
```

**View TensorBoard:**
- Vertex AI UI: https://console.cloud.google.com/vertex-ai/experiments
- Or locally: `tensorboard --logdir=gs://emg2qwerty-team-logs/logs`

Each teammate also has personal settings:

```bash
# Personal project (different for each teammate)
GCP_PROJECT_ID=teammate-emg2qwerty
ARTIFACT_REGISTRY=us-central1-docker.pkg.dev/teammate-emg2qwerty/training
```

## Files in This Folder

- `README.md` - This file
- `setup_shared_resources.sh` - Creates shared buckets (team lead runs once)
- `upload_data.sh` - Uploads dataset to shared bucket (team lead runs once)
- `grant_access.sh` - Grants teammate access to shared resources (team lead runs per teammate)
- `setup_teammate.sh` - Sets up teammate's personal environment (each teammate runs once)
- `shared_config.env` - Shared configuration (committed to git)
- `.teammate_config.env` - Personal configuration (gitignored)

## Cost Sharing

- **Shared bucket storage**: ~$0.02/GB/month (team pays once)
- **Individual compute**: Each teammate pays for their own training jobs
- **TensorBoard**: Team shares one instance (~$0.10/hour when active)

## Troubleshooting

**"Permission denied" accessing shared bucket:**
```bash
# Team lead re-runs grant script
./grant_access.sh your-email@example.com
```

**"Can't find shared bucket":**
```bash
# Verify shared bucket exists
gsutil ls gs://emg2qwerty-team-data/

# Check you have access
gsutil ls gs://emg2qwerty-team-logs/
```

**"TensorBoard not showing my runs":**
```bash
# Verify logs are in shared bucket
gsutil ls gs://emg2qwerty-team-logs/

# TensorBoard auto-updates every few minutes
# Or manually refresh the page
```

## Best Practices

1. **Naming convention**: Use your name in job names
   ```bash
   python vertex_submit.py --submit --job-name alice-experiment-1
   ```

2. **Clean up old logs**: Delete your old experiments from shared bucket
   ```bash
   gsutil rm -r gs://emg2qwerty-team-logs/2024-01-*/
   ```

3. **Communication**: Post in team chat when starting long jobs

4. **Documentation**: Update team wiki with your experiment notes

## Next Steps

- [Team Lead] Run `./setup_shared_resources.sh`
- [Team Lead] Run `./upload_data.sh`
- [Team Lead] For each teammate, run `./grant_access.sh teammate@email.com`
- [Teammates] Run `./setup_teammate.sh`
- [Everyone] Start training!
