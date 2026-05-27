#!/bin/bash
# Pre-training environment check script
# Purpose: Clean up old training processes, ensure a clean environment

set -e

TRAIN_PATTERNS=("train_baseline_vgg.py" "python -m models.train" "python3 -m models.train")

found_pids=""

for pattern in "${TRAIN_PATTERNS[@]}"; do
    pids=$(ps aux | grep "$pattern" | grep -v grep | grep -v "check_training_prerequisites" | awk '{print $2}' || true)
    if [ -n "$pids" ]; then
        found_pids="$found_pids $pids"
    fi
done

if [ -n "$found_pids" ]; then
    echo "Found old training processes (PID:$found_pids), terminating..."
    pkill -9 -f train_baseline_vgg.py 2>/dev/null || true
    pkill -9 -f "python.*models.train" 2>/dev/null || true
    sleep 2
    
    remaining=$(ps aux | grep -E "train_baseline_vgg.py|python.*models.train" | grep -v grep | grep -v "check_training_prerequisites" | wc -l)
    if [ "$remaining" -gt 0 ]; then
        echo "Warning: $remaining training process(es) still not cleaned up"
        exit 1
    fi
fi

echo "Check passed, environment is clean. Ready to start new training."
