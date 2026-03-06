#!/bin/bash
#
# Grant Teammate Access to Shared Resources
#
# This script grants a teammate access to:
# - Read from shared data bucket
# - Write to shared logs bucket
# - View TensorBoard
# - Vertex AI service account access (if project ID provided)
#
# Run this for EACH new teammate
#
# Usage:
#   ./grant_access.sh teammate@example.com [PROJECT_ID]
#
# Examples:
#   ./grant_access.sh alice@stanford.edu
#   ./grant_access.sh alice@stanford.edu alice-emg2qwerty

set -e

# Load shared configuration
if [ -f "shared_config.env" ]; then
    source shared_config.env
else
    echo "Error: shared_config.env not found. Run setup_shared_resources.sh first."
    exit 1
fi

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: ./grant_access.sh teammate@example.com [PROJECT_ID]"
    echo ""
    echo "Examples:"
    echo "  ./grant_access.sh alice@stanford.edu"
    echo "  ./grant_access.sh alice@stanford.edu alice-emg2qwerty"
    echo ""
    echo "If PROJECT_ID is provided, also grants Vertex AI service account access."
    exit 1
fi

TEAMMATE_EMAIL=$1
TEAMMATE_PROJECT_ID=$2

# Validate email format
if [[ ! $TEAMMATE_EMAIL =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo "Error: Invalid email format: $TEAMMATE_EMAIL"
    exit 1
fi

echo "=========================================="
echo "Grant Teammate Access"
echo "=========================================="
echo ""
echo "Teammate: $TEAMMATE_EMAIL"
echo "Shared Project: $SHARED_PROJECT_ID"
if [ -n "$TEAMMATE_PROJECT_ID" ]; then
    echo "Teammate Project: $TEAMMATE_PROJECT_ID"
    echo ""
    echo "Will grant:"
    echo "  - User bucket access"
    echo "  - Vertex AI service account bucket access"
else
    echo ""
    echo "Will grant:"
    echo "  - User bucket access only"
    echo ""
    echo "Note: To also grant Vertex AI service account access, run:"
    echo "  ./grant_access.sh $TEAMMATE_EMAIL PROJECT_ID"
fi
echo ""

# Confirm
read -p "Grant access to this teammate? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Set active project
gcloud config set project $SHARED_PROJECT_ID

echo ""
echo "Granting permissions..."
echo "=========================================="

# Grant data bucket read access
echo "1. Granting read access to data bucket..."
gsutil iam ch user:$TEAMMATE_EMAIL:objectViewer gs://$SHARED_DATA_BUCKET
echo "   ✓ Data bucket (read)"

# Grant logs bucket read/write access
echo "2. Granting read/write access to logs bucket..."
gsutil iam ch user:$TEAMMATE_EMAIL:objectAdmin gs://$SHARED_LOGS_BUCKET
echo "   ✓ Logs bucket (read/write)"

# Grant project-level viewer role for TensorBoard
echo "3. Granting TensorBoard viewer access..."
gcloud projects add-iam-policy-binding $SHARED_PROJECT_ID \
    --member="user:$TEAMMATE_EMAIL" \
    --role="roles/aiplatform.user" \
    --condition=None \
    --quiet
echo "   ✓ TensorBoard access"

# If project ID provided, grant Vertex AI service account access
if [ -n "$TEAMMATE_PROJECT_ID" ]; then
    echo ""
    echo "4. Granting Vertex AI service account access..."

    # Get project number from project ID
    PROJECT_NUMBER=$(gcloud projects describe $TEAMMATE_PROJECT_ID --format="value(projectNumber)" 2>/dev/null)

    if [ -z "$PROJECT_NUMBER" ]; then
        echo "   ⚠ Warning: Could not find project $TEAMMATE_PROJECT_ID"
        echo "   Skipping Vertex AI service account grants"
    else
        VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
        echo "   Vertex AI Service Account: $VERTEX_SA"

        echo "   Granting read access to data bucket..."
        gsutil iam ch serviceAccount:${VERTEX_SA}:objectViewer gs://${SHARED_DATA_BUCKET}

        echo "   Granting write access to logs bucket..."
        gsutil iam ch serviceAccount:${VERTEX_SA}:objectAdmin gs://${SHARED_LOGS_BUCKET}

        echo "   ✓ Vertex AI service account configured"
    fi
fi

echo ""
echo "=========================================="
echo "Access Granted!"
echo "=========================================="
echo ""
echo "Teammate $TEAMMATE_EMAIL now has:"
echo "  ✓ Read access to data bucket"
echo "  ✓ Read/write access to logs bucket"
echo "  ✓ TensorBoard viewer access"
if [ -n "$TEAMMATE_PROJECT_ID" ] && [ -n "$PROJECT_NUMBER" ]; then
    echo "  ✓ Vertex AI service account bucket access"
fi
echo ""
echo "Next steps for $TEAMMATE_EMAIL:"
echo "  1. Clone the repository"
echo "  2. cd emg2qwerty/bootstrap"
echo "  3. Run: ./setup_teammate.sh"
if [ -z "$TEAMMATE_PROJECT_ID" ]; then
    echo ""
    echo "After setup completes, run this script again with their project ID:"
    echo "  ./grant_access.sh $TEAMMATE_EMAIL PROJECT_ID"
fi
echo ""
echo "Verification:"
echo "  gsutil ls gs://$SHARED_DATA_BUCKET/data/"
echo "  gsutil ls gs://$SHARED_LOGS_BUCKET/"
echo "=========================================="
