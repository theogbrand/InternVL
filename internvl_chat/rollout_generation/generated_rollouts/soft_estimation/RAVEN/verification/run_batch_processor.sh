#!/bin/bash

# Script to run batch_processor.py to process ALL batches in screen
# Usage: AZURE_API_KEY='your-key' ./run_batch_processor.sh [screen_session_name]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCREEN_SESSION="${1:-batch_processor}"

# Check API key
if [[ -z "${AZURE_API_KEY}" ]]; then
    echo "Error: Set AZURE_API_KEY environment variable"
    echo "Usage: AZURE_API_KEY='your-key' ./run_batch_processor.sh"
    exit 1
fi

# Kill existing screen session if it exists
screen -S "${SCREEN_SESSION}" -X quit 2>/dev/null || true

# Create screen session and run batch processor
screen -S "${SCREEN_SESSION}" -dm bash -c "
    cd '${SCRIPT_DIR}'
    eval \"\$(conda shell.bash hook)\"
    conda activate mmr_processing
    export AZURE_API_KEY='${AZURE_API_KEY}'
    echo 'Starting batch processor for ALL batches...'
    python batch_processor.py
    echo 'Batch processing completed. Press any key to exit.'
    read -n 1
"

echo "Started screen session: ${SCREEN_SESSION}"
echo "Attach with: screen -r ${SCREEN_SESSION}"
echo "Detach with: Ctrl+A then D"
echo "Stop with: Ctrl+C" 