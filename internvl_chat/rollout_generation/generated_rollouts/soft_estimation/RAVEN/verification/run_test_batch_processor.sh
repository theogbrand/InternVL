#!/bin/bash

# Simple script to run test_batch_processor.py in screen
# Usage: AZURE_API_KEY='your-key' ./run_test_batch_processor.sh [split_name]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCREEN_SESSION="test_batch_processor"
SPLIT=${1:-"test"}  # Default to "test" if not provided

# Check API key
if [[ -z "${AZURE_API_KEY}" ]]; then
    echo "Error: Set AZURE_API_KEY environment variable"
    echo "Usage: AZURE_API_KEY='your-key' ./run_test_batch_processor.sh [split_name]"
    exit 1
fi

# Kill existing screen session if it exists
screen -S "${SCREEN_SESSION}" -X quit 2>/dev/null || true

# Create screen session and run test
screen -S "${SCREEN_SESSION}" -dm bash -c "
    cd '${SCRIPT_DIR}'
    eval \"\$(conda shell.bash hook)\"
    conda activate mmr_processing
    export AZURE_API_KEY='${AZURE_API_KEY}'
    echo 'Starting test batch processor...'
    python test_batch_processor.py --split='${SPLIT}'
    echo 'Test completed. Press any key to exit.'
    read -n 1
"

echo "Started screen session: ${SCREEN_SESSION}"
echo "Processing split: ${SPLIT}"
echo "Attach with: screen -r ${SCREEN_SESSION}"
echo "Detach with: Ctrl+A then D"
echo "Stop with: Ctrl+C" 