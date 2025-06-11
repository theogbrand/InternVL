#!/bin/bash

# Script to run batch_processor.py to process batches in screen
# Usage: AZURE_API_KEY='your-key' ./run_batch_processor.sh [screen_session_name] [start_index] [end_index]
# Example: AZURE_API_KEY='your-key' ./run_batch_processor.sh batch_processor 3 13

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCREEN_SESSION="${1:-batch_processor}"
START_INDEX="${2}"
END_INDEX="${3}"

# Check API key
if [[ -z "${AZURE_API_KEY}" ]]; then
    echo "Error: Set AZURE_API_KEY environment variable"
    echo "Usage: AZURE_API_KEY='your-key' ./run_batch_processor.sh [screen_session_name] [start_index] [end_index]"
    exit 1
fi

# Build Python command with optional arguments
PYTHON_CMD="python batch_processor.py"
if [[ -n "${START_INDEX}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --start-index ${START_INDEX}"
fi
if [[ -n "${END_INDEX}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --end-index ${END_INDEX}"
fi

# Kill existing screen session if it exists
screen -S "${SCREEN_SESSION}" -X quit 2>/dev/null || true

# Create screen session and run batch processor
screen -S "${SCREEN_SESSION}" -dm bash -c "
    cd '${SCRIPT_DIR}'
    eval \"\$(conda shell.bash hook)\"
    conda activate mmr_processing
    export AZURE_API_KEY='${AZURE_API_KEY}'
    if [[ -n '${START_INDEX}' || -n '${END_INDEX}' ]]; then
        echo 'Starting batch processor for range [${START_INDEX}, ${END_INDEX}]...'
    else
        echo 'Starting batch processor for ALL batches...'
    fi
    ${PYTHON_CMD}
    echo 'Batch processing completed. Press any key to exit.'
    read -n 1
"

echo "Started screen session: ${SCREEN_SESSION}"
if [[ -n "${START_INDEX}" || -n "${END_INDEX}" ]]; then
    echo "Processing batch range: [${START_INDEX}, ${END_INDEX}]"
else
    echo "Processing ALL batches"
fi
echo "Attach with: screen -r ${SCREEN_SESSION}"
echo "Detach with: Ctrl+A then D"
echo "Stop with: Ctrl+C" 