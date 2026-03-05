#!/bin/bash
#
# Setup Shared Resources for EMG2QWERTY Team
#
# This script creates shared GCS buckets and TensorBoard instance
# that all team members can access.
#
# Run this ONCE as the team lead.
#
# Usage:
#   ./setup_shared_resources.sh

set -e  # Exit on error

echo "=========================================="
echo "EMG2QWERTY Team - Shared Resources Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SHARED_PROJECT_ID="emg2qwerty-team-shared"
SHARED_DATA_BUCKET="emg2qwerty-team-data"
SHARED_LOGS_BUCKET="emg2qwerty-team-logs"
REGION="us-central1"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Shared Project ID: $SHARED_PROJECT_ID"
echo "  Data Bucket: gs://$SHARED_DATA_BUCKET"
echo "  Logs Bucket: gs://$SHARED_LOGS_BUCKET"
echo "  Region: $REGION"
echo ""

read -p "Do you want to use these settings? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Edit this script to change the configuration."
    exit 1
fi

# Step 1: Create or use existing project
echo ""
echo "Step 1: Setting up shared project..."
echo "======================================="

if gcloud projects describe $SHARED_PROJECT_ID &>/dev/null; then
    echo -e "${GREEN}✓${NC} Project $SHARED_PROJECT_ID already exists"
else
    echo "Creating project $SHARED_PROJECT_ID..."
    gcloud projects create $SHARED_PROJECT_ID \
        --name="EMG2QWERTY Team Shared"

    echo -e "${YELLOW}⚠${NC}  Please enable billing for this project:"
    echo "   https://console.cloud.google.com/billing/linkedaccount?project=$SHARED_PROJECT_ID"
    read -p "Press enter after enabling billing..."
fi

# Set active project
gcloud config set project $SHARED_PROJECT_ID

# Step 2: Enable required APIs
echo ""
echo "Step 2: Enabling required APIs..."
echo "======================================="

APIS=(
    "storage.googleapis.com"
    "aiplatform.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo "Enabling $api..."
    gcloud services enable $api
done

echo -e "${GREEN}✓${NC} APIs enabled"

# Step 3: Create shared data bucket
echo ""
echo "Step 3: Creating shared data bucket..."
echo "======================================="

if gsutil ls gs://$SHARED_DATA_BUCKET &>/dev/null; then
    echo -e "${GREEN}✓${NC} Data bucket already exists"
else
    echo "Creating gs://$SHARED_DATA_BUCKET..."
    gsutil mb -l $REGION gs://$SHARED_DATA_BUCKET
    gsutil uniformbucketlevelaccess set on gs://$SHARED_DATA_BUCKET

    echo -e "${GREEN}✓${NC} Data bucket created"
fi

# Step 4: Create shared logs bucket
echo ""
echo "Step 4: Creating shared logs bucket..."
echo "======================================="

if gsutil ls gs://$SHARED_LOGS_BUCKET &>/dev/null; then
    echo -e "${GREEN}✓${NC} Logs bucket already exists"
else
    echo "Creating gs://$SHARED_LOGS_BUCKET..."
    gsutil mb -l $REGION gs://$SHARED_LOGS_BUCKET
    gsutil uniformbucketlevelaccess set on gs://$SHARED_LOGS_BUCKET

    echo -e "${GREEN}✓${NC} Logs bucket created"
fi

# Step 5: Create shared config file
echo ""
echo "Step 5: Creating shared configuration file..."
echo "======================================="

cat > shared_config.env << EOF
# EMG2QWERTY Team - Shared Configuration
# This file is committed to git and shared by all teammates
# Generated on $(date)

# Shared GCP Project
export SHARED_PROJECT_ID="$SHARED_PROJECT_ID"

# Shared Storage Buckets
export SHARED_DATA_BUCKET="$SHARED_DATA_BUCKET"
export SHARED_LOGS_BUCKET="$SHARED_LOGS_BUCKET"

# Region
export GCP_REGION="$REGION"

# GCS Paths
export SHARED_DATA_PATH="gs://\${SHARED_DATA_BUCKET}/data"
export SHARED_LOGS_PATH="gs://\${SHARED_LOGS_BUCKET}/logs"
EOF

echo -e "${GREEN}✓${NC} Configuration saved to shared_config.env"

# Step 6: Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Shared Resources Created:"
echo "  ✓ Project: $SHARED_PROJECT_ID"
echo "  ✓ Data bucket: gs://$SHARED_DATA_BUCKET"
echo "  ✓ Logs bucket: gs://$SHARED_LOGS_BUCKET"
echo ""
echo "Next Steps:"
echo "  1. Upload dataset: ./upload_data.sh /path/to/data"
echo "  2. Grant access to teammates: ./grant_access.sh teammate@example.com"
echo "  3. Teammates run: ./setup_teammate.sh"
echo ""
echo "View TensorBoard (after training):"
echo "  • Vertex AI UI: https://console.cloud.google.com/vertex-ai/experiments?project=$SHARED_PROJECT_ID"
echo "  • Or locally: tensorboard --logdir=gs://$SHARED_LOGS_BUCKET/logs"
echo ""
echo "Configuration saved to: shared_config.env"
echo "=========================================="
