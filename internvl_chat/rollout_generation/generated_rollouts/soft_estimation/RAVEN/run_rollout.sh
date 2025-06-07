#!/bin/bash

# RAVEN Rollout Runner with Screen Session Management
# Usage: ./run_rollout.sh [screen_name] [action]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_SCRIPT="$SCRIPT_DIR/rollout.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_NAME="${1:-raven_rollout_$TIMESTAMP}"
LOG_DIR="$SCRIPT_DIR/raven_rollouts_output"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to check if screen session exists
screen_exists() {
    screen -list | grep -q "\.$SCREEN_NAME\s"
}

# Function to get system resource info
show_resources() {
    echo "=== System Resources ==="
    echo "Memory usage:"
    free -h | head -2
    echo "CPU info:"
    top -bn1 | grep "Cpu(s)" | head -1
    echo "Disk usage for log directory:"
    df -h "$LOG_DIR" | tail -1
    echo "======================="
}

# Function to monitor the rollout process
monitor_rollout() {
    local logfile="$LOG_DIR/screen_output.log"
    local last_size=0
    
    echo "Monitoring rollout progress... (Press Ctrl+C to stop monitoring)"
    echo "Log file: $logfile"
    echo ""
    
    while screen_exists; do
        if [ -f "$logfile" ]; then
            current_size=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile" 2>/dev/null || echo "0")
            if [ "$current_size" -gt "$last_size" ]; then
                # Show new lines since last check
                tail -c +$((last_size + 1)) "$logfile" | head -20
                last_size=$current_size
            fi
        fi
        sleep 10
    done
    
    echo "Screen session ended."
}

# Function to start the rollout in screen
start_rollout() {
    echo "Starting RAVEN rollout in screen session: $SCREEN_NAME"
    echo "Estimated processing time: 3-4 hours for 1200 samples"
    echo ""
    
    show_resources
    echo ""
    
    # Check if AZURE_API_KEY is set
    if [ -z "$AZURE_API_KEY" ]; then
        echo "WARNING: AZURE_API_KEY environment variable is not set"
        echo "Please set it before running: export AZURE_API_KEY='your-key-here'"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
    
    # Start screen session with enhanced configuration
    screen -dmS "$SCREEN_NAME" bash -c "
        set -e
        trap 'echo \"Received termination signal, shutting down gracefully...\"; kill -TERM \$python_pid 2>/dev/null; wait \$python_pid 2>/dev/null; exit 0' SIGINT SIGTERM
        
        echo '=== RAVEN Rollout Session Started ==='
        echo \"Session: $SCREEN_NAME\"
        echo \"Time: \$(date)\"
        echo \"Script: $ROLLOUT_SCRIPT\"
        echo \"Log: $LOG_DIR/screen_output.log\"
        echo '====================================='
        
        echo 'Activating mmr_processing environment...'
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda activate mmr_processing
        
        echo 'Environment activated. Starting RAVEN rollout...'
        cd '$SCRIPT_DIR'
        
        # Set environment variables for optimal performance
        export PYTHONUNBUFFERED=1
        export TERM=screen-256color
        export OMP_NUM_THREADS=4
        export AZURE_API_KEY=\"\$AZURE_API_KEY\"
        
        # Show initial system state
        echo 'Initial system resources:'
        free -h | head -2
        echo ''
        
        # Run with proper signal handling and resource monitoring
        echo 'Launching Python script with PID tracking...'
        python -u rollout.py &
        python_pid=\$!
        echo \"Python process PID: \$python_pid\"
        
        # Monitor resources every 5 minutes
        (
            while kill -0 \$python_pid 2>/dev/null; do
                sleep 300  # 5 minutes
                echo \"[\$(date)] Resource check:\"
                ps -p \$python_pid -o pid,pcpu,pmem,etime,comm 2>/dev/null || break
                free -h | grep '^Mem:' || true
            done
        ) &
        monitor_pid=\$!
        
        # Wait for Python process to complete
        wait \$python_pid
        python_exit_code=\$?
        
        # Clean up monitor
        kill \$monitor_pid 2>/dev/null || true
        
        echo ''
        echo '=== Rollout Completed ==='
        echo \"Exit code: \$python_exit_code\"
        echo \"Time: \$(date)\"
        echo \"Final resource state:\"
        free -h | head -2
        echo '========================='
        
        if [ \$python_exit_code -eq 0 ]; then
            echo 'SUCCESS: Rollout completed successfully!'
        else
            echo 'ERROR: Rollout failed with exit code \$python_exit_code'
        fi
        
        echo ''
        echo 'Press any key to close this screen session...'
        read -n 1
    " 2>&1 | tee -a "$LOG_DIR/screen_output.log"
    
    if [ $? -eq 0 ]; then
        echo "Screen session '$SCREEN_NAME' started successfully"
        echo ""
        echo "Commands:"
        echo "  Attach to session: screen -r $SCREEN_NAME"
        echo "  Detach from session: Ctrl+A, then D"
        echo "  Monitor progress: $0 $SCREEN_NAME monitor"
        echo "  View logs: tail -f $LOG_DIR/screen_output.log"
        echo "  Stop session: $0 $SCREEN_NAME stop"
        echo ""
        echo "The rollout will run with massive parallel processing."
        echo "Expected completion: 3-4 hours for 1200 samples"
    else
        echo "Failed to start screen session"
        exit 1
    fi
}

# Function to attach to existing screen session
attach_rollout() {
    echo "Attaching to existing screen session: $SCREEN_NAME"
    echo "To detach: Ctrl+A, then D"
    screen -r "$SCREEN_NAME"
}

# Function to stop the rollout gracefully
stop_rollout() {
    if screen_exists; then
        echo "Sending graceful shutdown signal to screen session: $SCREEN_NAME"
        # Send SIGTERM to allow graceful shutdown
        screen -S "$SCREEN_NAME" -X stuff "^C"
        
        echo "Waiting 30 seconds for graceful shutdown..."
        sleep 30
        
        if screen_exists; then
            echo "Force terminating screen session..."
            screen -S "$SCREEN_NAME" -X quit
        fi
        
        echo "Screen session stopped"
    else
        echo "No screen session '$SCREEN_NAME' found"
    fi
}

# Function to show detailed status
show_status() {
    if screen_exists; then
        echo "✅ Screen session '$SCREEN_NAME' is RUNNING"
        screen -list | grep "$SCREEN_NAME"
        echo ""
        
        show_resources
        echo ""
        
        # Show recent log output if available
        local logfile="$LOG_DIR/screen_output.log"
        if [ -f "$logfile" ]; then
            echo "Recent log output (last 15 lines):"
            echo "═══════════════════════════════════"
            tail -n 15 "$logfile"
            echo "═══════════════════════════════════"
            echo ""
            
            # Try to extract progress information
            local total_samples=$(grep -o "Processing [0-9]* RAVEN samples" "$logfile" | tail -1 | grep -o "[0-9]*" | head -1)
            local completed_batches=$(grep -c "Batch [0-9]* completed" "$logfile")
            
            if [ -n "$total_samples" ] && [ -n "$completed_batches" ] && [ "$completed_batches" -gt 0 ]; then
                local batch_size=50  # Default batch size
                local completed_samples=$((completed_batches * batch_size))
                local progress=$((completed_samples * 100 / total_samples))
                echo "📊 Progress: $completed_samples/$total_samples samples (~$progress%)"
                echo "📈 Batches completed: $completed_batches"
            fi
        else
            echo "⚠️  Log file not found: $logfile"
        fi
    else
        echo "❌ Screen session '$SCREEN_NAME' is NOT running"
        
        # Check for recent log files
        if [ -f "$LOG_DIR/screen_output.log" ]; then
            echo ""
            echo "Last session log (final 10 lines):"
            echo "═══════════════════════════════════"
            tail -n 10 "$LOG_DIR/screen_output.log"
            echo "═══════════════════════════════════"
        fi
    fi
}

# Function to clean up old logs and outputs
cleanup() {
    echo "Cleaning up old logs and temporary files..."
    
    # Remove old log files (older than 7 days)
    find "$LOG_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Remove temporary image files
    find /tmp -name "temp_image_*" -mtime +1 -delete 2>/dev/null || true
    
    echo "Cleanup completed"
}

# Main logic
ACTION="${2:-start}"

case "$ACTION" in
    start)
        if screen_exists; then
            echo "Screen session '$SCREEN_NAME' already exists"
            echo "Choose action:"
            echo "1) Attach to existing session"
            echo "2) Stop and restart"
            echo "3) Show status"
            read -p "Enter choice (1-3): " -n 1 -r
            echo
            case $REPLY in
                1) attach_rollout ;;
                2) stop_rollout && sleep 2 && start_rollout ;;
                3) show_status ;;
                *) echo "Invalid choice" && exit 1 ;;
            esac
        else
            start_rollout
        fi
        ;;
    attach)
        if screen_exists; then
            attach_rollout
        else
            echo "No screen session '$SCREEN_NAME' found"
            exit 1
        fi
        ;;
    stop)
        stop_rollout
        ;;
    status)
        show_status
        ;;
    monitor)
        if screen_exists; then
            monitor_rollout
        else
            echo "No screen session '$SCREEN_NAME' found"
            exit 1
        fi
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "RAVEN Rollout Runner"
        echo "Usage: $0 [screen_name] [action]"
        echo ""
        echo "Actions:"
        echo "  start    - Start new rollout or attach if already running (default)"
        echo "  attach   - Attach to existing rollout session"
        echo "  stop     - Stop the rollout session gracefully"
        echo "  status   - Show detailed rollout status and progress"
        echo "  monitor  - Monitor rollout progress in real-time"
        echo "  cleanup  - Clean up old logs and temporary files"
        echo ""
        echo "Examples:"
        echo "  $0                           # Start with default name"
        echo "  $0 my_rollout start          # Start with custom name"
        echo "  $0 my_rollout status         # Check status"
        echo "  $0 my_rollout monitor        # Monitor progress"
        exit 1
        ;;
esac 