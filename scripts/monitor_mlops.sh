#!/bin/bash
# MLOps Hard Samples Monitor Script
# Monitors data/hard_samples directory and triggers fine-tuning when threshold is reached
#
# Usage:
#   ./scripts/monitor_mlops.sh           # Run in foreground
#   nohup ./scripts/monitor_mlops.sh &   # Run in background
#   tail -f logs/mlops_monitor.log       # View logs
#   pkill -f "monitor_mlops.sh"          # Stop

# Configuration
HARD_SAMPLES_DIR="data/hard_samples"
THRESHOLD=500
CHECK_INTERVAL=300  # 5 minutes in seconds
PROJECT_ROOT="/root/captcharecognition"
LOG_FILE="logs/mlops_trigger.log"
MONITOR_LOG="logs/mlops_monitor.log"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

# Ensure log directory exists
mkdir -p logs

# Function to log messages
log_message() {
    local message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message" | tee -a "$MONITOR_LOG"
}

# Function to count hard samples
count_hard_samples() {
    if [ -d "$HARD_SAMPLES_DIR" ]; then
        local count=$(find "$HARD_SAMPLES_DIR" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
        echo "$count"
    else
        echo "0"
    fi
}

# Function to check if mlops training is running
is_mlops_running() {
    if pgrep -f "main.py --mode mlops" > /dev/null 2>&1; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Function to trigger mlops fine-tuning
trigger_mlops() {
    local count="$1"
    local trigger_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log trigger event
    echo "Triggered at $trigger_time, count: $count" >> "$LOG_FILE"
    
    log_message "TRIGGER: Hard samples ($count) >= threshold ($THRESHOLD), starting MLOps fine-tuning..."
    
    # Run mlops fine-tuning
    cd "$PROJECT_ROOT"
    python main.py --mode mlops \
        --hard_sample_dir "$HARD_SAMPLES_DIR" \
        --threshold "$THRESHOLD" \
        --lr 1e-6 \
        --epochs 5 2>&1 | tee -a "$MONITOR_LOG"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_message "SUCCESS: MLOps fine-tuning completed"
    else
        log_message "ERROR: MLOps fine-tuning failed with exit code $exit_code"
    fi
    
    cd "$PROJECT_ROOT"
}

# Main loop
log_message "MLOps Monitor started"
log_message "Configuration:"
log_message "  Hard samples directory: $HARD_SAMPLES_DIR"
log_message "  Threshold: $THRESHOLD"
log_message "  Check interval: ${CHECK_INTERVAL}s (5 minutes)"
log_message "  Trigger log: $LOG_FILE"
log_message "  Monitor log: $MONITOR_LOG"
log_message "------------------------------------------------"

while true; do
    # Count hard samples
    count=$(count_hard_samples)
    
    # Log current count every hour (every 12 iterations)
    if [ $((count % 12)) -eq 0 ]; then
        log_message "Current hard sample count: $count"
    fi
    
    # Check if threshold is reached and no mlops process is running
    if [ "$count" -ge "$THRESHOLD" ] && ! is_mlops_running; then
        log_message "Threshold reached: $count >= $THRESHOLD"
        trigger_mlops "$count"
    elif [ "$count" -ge "$THRESHOLD" ] && is_mlops_running; then
        log_message "Threshold reached ($count), but MLOps process already running, skipping"
    fi
    
    # Wait for next check
    sleep "$CHECK_INTERVAL"
done
