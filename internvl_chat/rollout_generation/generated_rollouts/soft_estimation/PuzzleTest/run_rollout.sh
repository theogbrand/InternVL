#!/bin/bash

# VQAv2 Rollout Runner with Streaming Support
# Usage: ./run_rollout.sh [action]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_SCRIPT="$SCRIPT_DIR/rollout.py"
OUTPUT_DIR="$SCRIPT_DIR/vqav2_int_rollouts_output"
LOG_DIR="$OUTPUT_DIR/screen_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_NAME="vqav2_rollout_$TIMESTAMP"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Function to start rollout
start_rollout() {
    echo "Starting VQAv2 rollout with streaming support..."
    
    # Check API key
    if [ -z "$AZURE_API_KEY" ]; then
        echo "Error: AZURE_API_KEY not set"
        echo "Run: export AZURE_API_KEY='your-key-here'"
        exit 1
    fi
    
    echo "Environment will be set up inside screen session..."
    
    # Run in screen session
    screen -dmS "$SCREEN_NAME" bash -c "
        # Set environment variables inside screen session
        export AZURE_API_KEY='$AZURE_API_KEY'
        export PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS=4
        
        # Activate conda environment inside screen session
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda activate mmr_processing
        
        cd '$SCRIPT_DIR'
        echo \"Starting rollout at \$(date)\"
        echo \"Output directory: $OUTPUT_DIR\"
        echo \"Log directory: $LOG_DIR\"
        
        # Run with unbuffered output and tee to both console and log
        python -u rollout.py 2>&1 | tee '$LOG_DIR/rollout_$TIMESTAMP.log'
        
        echo \"Rollout completed at \$(date)\"
        echo 'Press any key to close...'
        read -n 1
    "
    
    echo "Rollout started in screen session: $SCREEN_NAME"
    echo "Commands:"
    echo "  Attach: screen -r $SCREEN_NAME"
    echo "  Log: tail -f $LOG_DIR/screen_output_$TIMESTAMP.log"
    echo "  Stop: screen -S $SCREEN_NAME -X quit"
}

# Function to show running sessions
status() {
    echo "Active VQAv2 rollout sessions:"
    screen -list | grep vqav2_rollout || echo "No active sessions"
    echo ""
    echo "Recent logs:"
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -3 || echo "No logs found"
    echo ""
    echo "Recent output files:"
    ls -lt "$OUTPUT_DIR"/*.jsonl 2>/dev/null | head -3 || echo "No output files found"
}

# Function to cleanup
cleanup() {
    echo "Cleaning up..."
    # Kill all vqav2 rollout sessions
    screen -list | grep vqav2_rollout | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -S {} -X quit 2>/dev/null
    echo "Cleanup complete"
}

# Main logic
case "${1:-start}" in
    start)
        start_rollout
        ;;
    status)
        status
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 [start|status|cleanup]"
        echo ""
        echo "  start   - Start rollout (default)"
        echo "  status  - Show active sessions and logs"
        echo "  cleanup - Stop all sessions"
        exit 1
        ;;
esac 