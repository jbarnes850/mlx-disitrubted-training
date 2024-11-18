#!/bin/bash
set -e  # Exit on error

echo "Initializing MLX distributed cluster..."

# Load configuration
CONFIG_FILE=${1:-"configs/distributed_config.json"}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Setup hosts
./scripts/setup_hosts.sh

# Verify environment on all nodes
./scripts/verify_setup.sh distributed

# Start coordinator
echo "Starting coordinator..."
python scripts/start_coordinator.py --config "$CONFIG_FILE" &
COORD_PID=$!

# Wait for coordinator to initialize
sleep 5

# Start workers
echo "Starting workers..."
python scripts/start_worker.py --config "$CONFIG_FILE"

# Wait for all processes
wait $COORD_PID 