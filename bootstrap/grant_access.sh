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
    echo "Usage: ./grant_access.sh teammate@example.com [PROJECT_ID_OR_SERVICE_ACCOUNT]"
    echo ""
    echo "Examples:"
    echo "  # Just grant user bucket access:"
    echo "  ./grant_access.sh alice@stanford.edu"
    echo ""
    echo "  # Grant user + Vertex AI service account access:"
    echo "  ./grant_access.sh alice@stanford.edu alice-emg2qwerty"
    echo ""
    echo "  # Or provide the service account email directly (from error message):"
    echo "  ./grant_access.sh alice@stanford.edu service-123456789012@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
    echo ""
    echo "If teammate gets a 403 error, ask them to copy/paste the service account"
    echo "email from the error message and re-run with it as the second argument."
    exit 1
fi

TEAMMATE_EMAIL=$1
SECOND_ARG=$2

# Validate email format
if [[ ! $TEAMMATE_EMAIL =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo "Error: Invalid email format: $TEAMMATE_EMAIL"
    exit 1
fi

# Detect if second argument is a service account email or project ID
VERTEX_SA=""
TEAMMATE_PROJECT_ID=""

if [[ "$SECOND_ARG" == service-*@gcp-sa-aiplatform-cc.iam.gserviceaccount.com ]]; then
    # Second argument is a service account email
    VERTEX_SA="$SECOND_ARG"
elif [ -n "$SECOND_ARG" ]; then
    # Second argument is a project ID
    TEAMMATE_PROJECT_ID="$SECOND_ARG"
fi

echo "=========================================="
echo "Grant Teammate Access"
echo "=========================================="
echo ""
echo "Teammate: $TEAMMATE_EMAIL"
echo "Shared Project: $SHARED_PROJECT_ID"

if [ -n "$VERTEX_SA" ]; then
    echo "Vertex AI Service Account: $VERTEX_SA"
    echo ""
    echo "Will grant:"
    echo "  - User bucket access"
    echo "  - Vertex AI service account bucket access"
elif [ -n "$TEAMMATE_PROJECT_ID" ]; then
    echo "Teammate Project: $TEAMMATE_PROJECT_ID"
    echo ""
    echo "Will grant:"
    echo "  - User bucket access"
    echo "  - Vertex AI service account bucket access (auto-detected)"
else
    echo ""
    echo "Will grant:"
    echo "  - User bucket access only"
    echo ""
    echo "Note: To also grant Vertex AI service account access, run:"
    echo "  ./grant_access.sh $TEAMMATE_EMAIL PROJECT_ID"
    echo "Or:"
    echo "  ./grant_access.sh $TEAMMATE_EMAIL SERVICE_ACCOUNT_EMAIL"
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

# If Vertex SA or project ID provided, grant Vertex AI service account access
if [ -n "$VERTEX_SA" ] || [ -n "$TEAMMATE_PROJECT_ID" ]; then
    echo ""
    echo "4. Granting Vertex AI service account access..."

    # If VERTEX_SA not already set, look it up from project ID
    if [ -z "$VERTEX_SA" ]; then
        echo "   Looking up service account for project $TEAMMATE_PROJECT_ID..."
        PROJECT_NUMBER=$(gcloud projects describe $TEAMMATE_PROJECT_ID --format="value(projectNumber)" 2>/dev/null)

        if [ -z "$PROJECT_NUMBER" ]; then
            echo ""
            echo "   ❌ ERROR: Could not get project number for $TEAMMATE_PROJECT_ID"
            echo "   This is likely because you don't have permission to view their project."
            echo ""
            echo "   Solution: Ask $TEAMMATE_EMAIL to copy the service account email from their"
            echo "   Vertex AI training error message, then re-run:"
            echo "   ════════════════════════════════════════════════════════════════"
            echo "   ./grant_access.sh $TEAMMATE_EMAIL SERVICE_ACCOUNT_EMAIL"
            echo "   ════════════════════════════════════════════════════════════════"
            echo ""
            echo "   The error message will show something like:"
            echo "   'service-123456789012@gcp-sa-aiplatform-cc.iam.gserviceaccount.com'"
            echo ""
            exit 1
        fi

        VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
    fi

    echo "   Vertex AI Service Account: $VERTEX_SA"

    echo "   Granting read access to data bucket..."
    if ! gsutil iam ch serviceAccount:${VERTEX_SA}:objectViewer gs://${SHARED_DATA_BUCKET} 2>/dev/null; then
        echo ""
        echo "   ❌ ERROR: Service account does not exist yet"
        echo ""
        echo "   This service account is created automatically when Vertex AI is first used."
        echo "   Ask $TEAMMATE_EMAIL to run this command in their project to create it:"
        echo "   ════════════════════════════════════════════════════════════════"
        echo "   gcloud beta services identity create --service=aiplatform.googleapis.com"
        echo "   ════════════════════════════════════════════════════════════════"
        echo ""
        echo "   Then re-run this script with the same arguments."
        echo ""
        exit 1
    fi

    echo "   Granting write access to logs bucket..."
    gsutil iam ch serviceAccount:${VERTEX_SA}:objectAdmin gs://${SHARED_LOGS_BUCKET}

    echo "   ✓ Vertex AI service account configured"
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
if [ -n "$VERTEX_SA" ]; then
    echo "  ✓ Vertex AI service account bucket access"
fi
echo ""
echo "Next steps for $TEAMMATE_EMAIL:"
echo "  1. Clone the repository"
echo "  2. cd emg2qwerty/bootstrap"
echo "  3. Run: ./setup_teammate.sh"
if [ -z "$VERTEX_SA" ]; then
    echo ""
    echo "After setup completes, if they get a 403 error when training:"
    echo "  Ask them to send you the service account email from the error"
    echo "  Then run: ./grant_access.sh $TEAMMATE_EMAIL SERVICE_ACCOUNT_EMAIL"
fi
echo ""
echo "Verification:"
echo "  gsutil ls gs://$SHARED_DATA_BUCKET/data/"
echo "  gsutil ls gs://$SHARED_LOGS_BUCKET/"
echo "=========================================="
