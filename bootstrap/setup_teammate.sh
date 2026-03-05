#!/bin/bash
#
# Teammate Environment Setup
#
# This script sets up a teammate's personal GCP project and connects
# it to the shared team resources.
#
# Each teammate runs this ONCE in their own environment.
#
# Prerequisites:
# - Team lead has run setup_shared_resources.sh
# - Team lead has run grant_access.sh for your email
#
# Usage:
#   ./setup_teammate.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "EMG2QWERTY - Teammate Setup"
echo "=========================================="
echo ""

# Load shared configuration
if [ -f "shared_config.env" ]; then
    source shared_config.env
    echo -e "${GREEN}✓${NC} Loaded shared team configuration"
else
    echo -e "${YELLOW}⚠${NC}  shared_config.env not found"
    echo "Please get this file from your team lead or git repository"
    exit 1
fi

echo ""
echo "Shared team resources:"
echo "  Data bucket: gs://$SHARED_DATA_BUCKET"
echo "  Logs bucket: gs://$SHARED_LOGS_BUCKET"
echo ""

# Step 1: Get teammate's email
echo "Step 1: Identifying your Google account..."
echo "======================================="

TEAMMATE_EMAIL=$(gcloud config get-value account 2>/dev/null || echo "")
if [ -z "$TEAMMATE_EMAIL" ]; then
    echo "Please login to gcloud first:"
    gcloud auth login
    TEAMMATE_EMAIL=$(gcloud config get-value account)
fi

echo -e "${GREEN}✓${NC} Logged in as: $TEAMMATE_EMAIL"

# Step 2: Personal project setup
echo ""
echo "Step 2: Setting up your personal project..."
echo "======================================="

# Suggest project ID based on email
EMAIL_PREFIX=$(echo $TEAMMATE_EMAIL | cut -d'@' -f1 | tr '.' '-')
SUGGESTED_PROJECT_ID="${EMAIL_PREFIX}-emg2qwerty"

echo ""
echo "You need a personal GCP project for:"
echo "  - Running your training jobs"
echo "  - Storing your Docker images"
echo "  - Billing your compute usage"
echo ""
read -p "Enter your project ID [$SUGGESTED_PROJECT_ID]: " USER_PROJECT_ID
USER_PROJECT_ID=${USER_PROJECT_ID:-$SUGGESTED_PROJECT_ID}

# Check if project exists
if gcloud projects describe $USER_PROJECT_ID &>/dev/null; then
    echo -e "${GREEN}✓${NC} Using existing project: $USER_PROJECT_ID"
else
    echo "Creating new project: $USER_PROJECT_ID..."
    # Truncate display name to max 30 chars
    DISPLAY_NAME="${EMAIL_PREFIX:0:17} - EMG2QWERTY"
    gcloud projects create $USER_PROJECT_ID \
        --name="$DISPLAY_NAME"

    echo -e "${YELLOW}⚠${NC}  Please enable billing for your project:"
    echo "   https://console.cloud.google.com/billing/linkedaccount?project=$USER_PROJECT_ID"
    read -p "Press enter after enabling billing..."
fi

# Set active project
gcloud config set project $USER_PROJECT_ID

# Step 3: Enable APIs
echo ""
echo "Step 3: Enabling required APIs..."
echo "======================================="

APIS=(
    "aiplatform.googleapis.com"
    "artifactregistry.googleapis.com"
    "storage.googleapis.com"
    "compute.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo "Enabling $api..."
    gcloud services enable $api --quiet
done

echo -e "${GREEN}✓${NC} APIs enabled"

# Step 4: Create Artifact Registry
echo ""
echo "Step 4: Creating Artifact Registry..."
echo "======================================="

REGISTRY_NAME="emg2qwerty-training"

if gcloud artifacts repositories describe $REGISTRY_NAME \
    --location=$GCP_REGION &>/dev/null; then
    echo -e "${GREEN}✓${NC} Artifact Registry already exists"
else
    echo "Creating Docker repository..."
    gcloud artifacts repositories create $REGISTRY_NAME \
        --repository-format=docker \
        --location=$GCP_REGION \
        --description="Docker images for EMG2QWERTY training"

    echo -e "${GREEN}✓${NC} Artifact Registry created"
fi

# Configure Docker
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev --quiet

# Step 5: Grant Vertex AI service account access to shared buckets
echo ""
echo "Step 5: Granting Vertex AI service account access..."
echo "======================================="

# Get project number (needed for service account email)
PROJECT_NUMBER=$(gcloud projects describe $USER_PROJECT_ID --format="value(projectNumber)")
VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"

echo "Vertex AI service account: $VERTEX_SA"
echo "Granting read access to data bucket..."
gsutil iam ch serviceAccount:${VERTEX_SA}:objectViewer gs://${SHARED_DATA_BUCKET}

echo "Granting write access to logs bucket..."
gsutil iam ch serviceAccount:${VERTEX_SA}:objectAdmin gs://${SHARED_LOGS_BUCKET}

echo "Granting you TensorBoard access..."
gsutil iam ch user:${TEAMMATE_EMAIL}:objectViewer gs://${SHARED_LOGS_BUCKET}

echo -e "${GREEN}✓${NC} Vertex AI service account configured"
echo -e "${GREEN}✓${NC} TensorBoard access granted to $TEAMMATE_EMAIL"

# Step 6: Verify access to shared resources
echo ""
echo "Step 6: Verifying access to shared resources..."
echo "======================================="

# Test data bucket access
echo -n "Checking data bucket access... "
if gsutil ls gs://$SHARED_DATA_BUCKET/data/ &>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}✗${NC}"
    echo ""
    echo "Error: Cannot access shared data bucket"
    echo "Ask your team lead to run: ./grant_access.sh $TEAMMATE_EMAIL"
    exit 1
fi

# Test logs bucket access
echo -n "Checking logs bucket access... "
if gsutil ls gs://$SHARED_LOGS_BUCKET/ &>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}✗${NC}"
    echo ""
    echo "Error: Cannot access shared logs bucket"
    echo "Ask your team lead to run: ./grant_access.sh $TEAMMATE_EMAIL"
    exit 1
fi

echo -e "${GREEN}✓${NC} All shared resources accessible"

# Step 7: Create personal configuration file
echo ""
echo "Step 7: Creating your configuration..."
echo "======================================="

cat > .teammate_config.env << EOF
# Your personal GCP project
export GCP_PROJECT_ID="$USER_PROJECT_ID"
export ARTIFACT_REGISTRY="${GCP_REGION}-docker.pkg.dev/${USER_PROJECT_ID}/${REGISTRY_NAME}"
export TEAMMATE_NAME="$EMAIL_PREFIX"
EOF

echo -e "${GREEN}✓${NC} Configuration saved to .teammate_config.env"

# Step 8: Set active project
gcloud config set project $USER_PROJECT_ID

# Step 9: Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo -e "${GREEN}Your Resources:${NC}"
echo "  ✓ Project: $USER_PROJECT_ID"
echo "  ✓ Artifact Registry: ${GCP_REGION}-docker.pkg.dev/${USER_PROJECT_ID}/${REGISTRY_NAME}"
echo ""
echo -e "${BLUE}Shared Resources:${NC}"
echo "  ✓ Data: gs://$SHARED_DATA_BUCKET/data/"
echo "  ✓ Logs: gs://$SHARED_LOGS_BUCKET/logs/"
echo ""
echo -e "${YELLOW}To start training:${NC}"
echo "  cd .."
echo "  python train_remote.py"
echo ""
echo "=========================================="
