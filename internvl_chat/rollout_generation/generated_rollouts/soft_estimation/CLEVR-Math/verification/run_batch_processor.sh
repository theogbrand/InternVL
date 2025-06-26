#!/bin/bash

# Script to run batch_processor.py to process batches in screen
# Usage: AZURE_API_KEY='your-key' ./run_batch_processor.sh [screen_session_name] [batch_start_index] [batch_end_index] [split] [azure_endpoint] [check_interval]
# Example: AZURE_API_KEY='your-key' ./run_batch_processor.sh batch_processor 3 13 distribute_four https://your-endpoint.openai.azure.com/ 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCREEN_SESSION_BASE="${1:-batch_processor}"
START_INDEX="${2}"
END_INDEX="${3}"
SPLIT="${4:-distribute_four}"
AZURE_ENDPOINT="${5}"
CHECK_INTERVAL="${6:-1}"

# Build screen session name with format: SCREEN_SESSION_START_END_SPLIT
SCREEN_SESSION="${SCREEN_SESSION_BASE}"
if [[ -n "${START_INDEX}" ]]; then
    SCREEN_SESSION="${SCREEN_SESSION}_${START_INDEX}"
fi
if [[ -n "${END_INDEX}" ]]; then
    SCREEN_SESSION="${SCREEN_SESSION}_${END_INDEX}"
fi
SCREEN_SESSION="${SCREEN_SESSION}_${SPLIT}"

# Check API key
if [[ -z "${AZURE_API_KEY}" ]]; then
    echo "Error: Set AZURE_API_KEY environment variable"
    echo "Usage: AZURE_API_KEY='your-key' ./run_batch_processor.sh [screen_session_name] [batch_start_index] [batch_end_index] [split] [azure_endpoint] [check_interval]"
    echo "Parameters:"
    echo "  screen_session_name: Name for screen session (default: batch_processor)"
    echo "  batch_start_index: Start batch index, 1-indexed, inclusive (optional)"
    echo "  batch_end_index: End batch index, 1-indexed, inclusive (optional)"
    echo "  split: Split name (default: distribute_four)"
    echo "  azure_endpoint: Azure OpenAI endpoint URL (optional)"
    echo "  check_interval: Check interval in minutes (default: 1)"
    echo ""
    echo "Examples:"
    echo "  AZURE_API_KEY='key' ./run_batch_processor.sh"
    echo "  AZURE_API_KEY='key' ./run_batch_processor.sh batch_proc 3 distribute_nine"
    echo "  AZURE_API_KEY='key' ./run_batch_processor.sh batch_proc 3 distribute_nine https://custom.openai.azure.com/ 2"
    exit 1
fi

# Build Python command with optional arguments
PYTHON_CMD="python batch_processor.py --split ${SPLIT} --check-interval ${CHECK_INTERVAL}"

if [[ -n "${START_INDEX}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --start-index ${START_INDEX}"
fi
if [[ -n "${END_INDEX}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --end-index ${END_INDEX}"
fi
if [[ -n "${AZURE_ENDPOINT}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --azure-endpoint ${AZURE_ENDPOINT}"
fi
if [[ -n "${SPLIT}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --split ${SPLIT}"
fi 

# Kill existing screen session if it exists
screen -S "${SCREEN_SESSION}" -X quit 2>/dev/null || true

# Create screen session and run batch processor
screen -S "${SCREEN_SESSION}" -dm bash -c "
    cd '${SCRIPT_DIR}'
    eval \"\$(conda shell.bash hook)\"
    conda activate mmr_processing
    export AZURE_API_KEY='${AZURE_API_KEY}'
    export SPLIT='${SPLIT}'
    echo 'Batch Processor Configuration:'
    echo '  Split: ${SPLIT}'
    echo '  Check interval: ${CHECK_INTERVAL} minute(s)'
    if [[ -n '${AZURE_ENDPOINT}' ]]; then
        echo '  Azure endpoint: ${AZURE_ENDPOINT}'
    else
        echo '  Azure endpoint: Default (https://decla-mbncunfi-australiaeast.cognitiveservices.azure.com/)'
    fi
    if [[ -n '${START_INDEX}' || -n '${END_INDEX}' ]]; then
        echo '  Processing batch range: [${START_INDEX}, ${END_INDEX}]'
    else
        echo '  Processing: ALL batches'
    fi
    echo 'Starting batch processor...'
    echo ''
    ${PYTHON_CMD}
    echo ''
    echo 'Batch processing completed. Press any key to exit.'
    read -n 1
"

echo "Started screen session: ${SCREEN_SESSION}"
echo "Configuration:"
echo "  Split: ${SPLIT}"
echo "  Check interval: ${CHECK_INTERVAL} minute(s)"
if [[ -n "${AZURE_ENDPOINT}" ]]; then
    echo "  Azure endpoint: ${AZURE_ENDPOINT}"
else
    echo "  Azure endpoint: Default"
fi
if [[ -n "${START_INDEX}" || -n "${END_INDEX}" ]]; then
    echo "  Processing batch range: [${START_INDEX}, ${END_INDEX}]"
else
    echo "  Processing: ALL batches"
fi
echo ""
echo "Control commands:"
echo "  Attach: screen -r ${SCREEN_SESSION}"
echo "  Detach: Ctrl+A then D"
echo "  Stop: Ctrl+C" 