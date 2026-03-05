# GCP and Vertex AI Setup Guide

This guide walks you through setting up Google Cloud Platform for training with Vertex AI.

## Prerequisites
- Google Cloud account
- `gcloud` CLI installed ([install guide](https://cloud.google.com/sdk/docs/install))
- Billing enabled on your GCP account

## Step 1: Create GCP Project

```bash
# Set your project ID (change this to your preferred name)
export PROJECT_ID="emg2qwerty-training"

# Create the project
gcloud projects create $PROJECT_ID --name="EMG2QWERTY Training"

# Set as active project
gcloud config set project $PROJECT_ID

# Link billing account (you'll need to enable this in console first)
# Go to: https://console.cloud.google.com/billing
```

## Step 2: Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Container Registry (for Docker images)
gcloud services enable containerregistry.googleapis.com

# Enable Artifact Registry (recommended over Container Registry)
gcloud services enable artifactregistry.googleapis.com

# Enable Cloud Storage
gcloud services enable storage.googleapis.com

# Enable Compute Engine (for VMs)
gcloud services enable compute.googleapis.com
```

## Step 3: Create GCS Bucket

```bash
# Set bucket name (must be globally unique)
export BUCKET_NAME="${PROJECT_ID}-training-data"
export REGION="us-central1"  # Choose your preferred region

# Create bucket
gcloud storage buckets create gs://${BUCKET_NAME} \
  --location=${REGION} \
  --uniform-bucket-level-access

# Verify bucket creation
gcloud storage buckets list
```

## Step 4: Create Artifact Registry Repository

```bash
# Create Docker repository for training images
gcloud artifacts repositories create emg2qwerty-training \
  --repository-format=docker \
  --location=${REGION} \
  --description="Docker images for EMG2QWERTY training"

# Configure Docker to use Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

## Step 5: Set Up Authentication

```bash
# Authenticate with your Google account
gcloud auth login

# Set application default credentials (for Python SDK)
gcloud auth application-default login

# (Optional) Create a service account for production
gcloud iam service-accounts create vertex-training-sa \
  --display-name="Vertex AI Training Service Account"

# Grant necessary permissions to the service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:vertex-training-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:vertex-training-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

## Step 6: Upload Dataset to GCS

```bash
# Upload your local dataset to GCS
# Assumes your dataset is in ./data/
gcloud storage cp -r ./data gs://${BUCKET_NAME}/data/

# Verify upload
gcloud storage ls gs://${BUCKET_NAME}/data/
```

## Step 7: Set Environment Variables

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
export GCP_PROJECT_ID="emg2qwerty-training"
export GCS_BUCKET="emg2qwerty-training-data"
export GCP_REGION="us-central1"
export ARTIFACT_REGISTRY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/emg2qwerty-training"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Step 8: Test Configuration

```bash
# Verify project is set
gcloud config get-value project

# Verify you can access the bucket
gcloud storage ls gs://${BUCKET_NAME}/

# Verify Vertex AI is accessible
gcloud ai models list --region=${GCP_REGION}
```

## Next Steps

You're now ready to:
1. Build the Docker image (see `Dockerfile`)
2. Push to Artifact Registry
3. Submit training jobs to Vertex AI (see `vertex_submit.py`)

## Useful Commands

```bash
# List all training jobs
gcloud ai custom-jobs list --region=${GCP_REGION}

# Stream logs from a job
gcloud ai custom-jobs stream-logs JOB_ID --region=${GCP_REGION}

# Cancel a running job
gcloud ai custom-jobs cancel JOB_ID --region=${GCP_REGION}
```

## Cost Management

Monitor your usage at: https://console.cloud.google.com/billing

**Estimated costs:**
- Training on n1-standard-4 with 1 T4 GPU: ~$0.50-1.00/hour
- GCS storage: ~$0.02/GB/month
- Artifact Registry: Free for first 0.5GB

**Tips to save money:**
- Use preemptible instances for non-critical training
- Delete old model checkpoints from GCS
- Use regional resources (cheaper than multi-regional)
