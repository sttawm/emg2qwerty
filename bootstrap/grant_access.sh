#!/bin/bash
#
# Grant Teammate Access to Shared Resources
#
# This script grants a teammate access to:
# - Read from shared data bucket
# - Write to shared logs bucket
# - View TensorBoard
#
# Run this for EACH new teammate
#
# Usage:
#   ./grant_access.sh teammate@example.com

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
    echo "Usage: ./grant_access.sh teammate@example.com"
    echo ""
    echo "Example:"
    echo "  ./grant_access.sh alice@stanford.edu"
    exit 1
fi

TEAMMATE_EMAIL=$1

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

echo ""
echo "=========================================="
echo "Access Granted!"
echo "=========================================="
echo ""
echo "Teammate $TEAMMATE_EMAIL now has:"
echo "  ✓ Read access to data bucket"
echo "  ✓ Read/write access to logs bucket"
echo "  ✓ TensorBoard viewer access"
echo ""
echo "Next steps for $TEAMMATE_EMAIL:"
echo "  1. Clone the repository"
echo "  2. cd emg2qwerty/bootstrap"
echo "  3. Run: ./setup_teammate.sh"
echo ""
echo "Verification:"
echo "  gsutil ls gs://$SHARED_DATA_BUCKET/data/"
echo "  gsutil ls gs://$SHARED_LOGS_BUCKET/"
echo "=========================================="
