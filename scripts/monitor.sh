#!/bin/bash

echo "Starting monitoring dashboard..."

# Activate environment
source .venv/bin/activate

# Start dashboard
python -m src.monitoring.dashboard \
    --port 8050 \
    --log-dir logs/dashboard

# Watch logs in another terminal
echo "To watch logs:"
echo "tail -f logs/training.log" 