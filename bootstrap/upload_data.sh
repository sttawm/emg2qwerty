#!/bin/bash
#
# Upload Dataset to Shared Bucket
#
# This script uploads the EMG2QWERTY dataset to the shared data bucket
# so all teammates can access it.
#
# Run this ONCE as the team lead after setup_shared_resources.sh
#
# Usage:
#   ./upload_data.sh /path/to/local/data

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
    echo "Usage: ./upload_data.sh /path/to/local/data"
    echo ""
    echo "Example:"
    echo "  ./upload_data.sh ../data"
    echo "  ./upload_data.sh /Users/yourname/datasets/emg2qwerty/data"
    exit 1
fi

DATA_DIR=$1

# Validate data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR does not exist"
    exit 1
fi

# Check for .hdf5 files
HDF5_COUNT=$(find "$DATA_DIR" -name "*.hdf5" -o -name "*.h5" | wc -l)
if [ $HDF5_COUNT -eq 0 ]; then
    echo "Warning: No .hdf5 or .h5 files found in $DATA_DIR"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=========================================="
echo "Upload Dataset to Shared Bucket"
echo "=========================================="
echo ""
echo "Local data directory: $DATA_DIR"
echo "Destination: gs://$SHARED_DATA_BUCKET/data/"
echo "HDF5 files found: $HDF5_COUNT"
echo ""

# Estimate size
echo "Calculating data size..."
DATA_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
echo "Total size: $DATA_SIZE"
echo ""

read -p "Proceed with upload? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 0
fi

# Upload data
echo ""
echo "Uploading data to gs://$SHARED_DATA_BUCKET/data/..."
echo "This may take several minutes depending on data size..."
echo ""

# Use gsutil with parallel uploads for faster transfer
gsutil -m rsync -r "$DATA_DIR" "gs://$SHARED_DATA_BUCKET/data/"

echo ""
echo "=========================================="
echo "Upload Complete!"
echo "=========================================="
echo ""
echo "Data uploaded to: gs://$SHARED_DATA_BUCKET/data/"
echo ""
echo "Verify upload:"
echo "  gsutil ls -lh gs://$SHARED_DATA_BUCKET/data/"
echo ""
echo "Next steps:"
echo "  1. Grant access to teammates: ./grant_access.sh teammate@example.com"
echo "  2. Teammates can now run training using the shared data"
echo "=========================================="
