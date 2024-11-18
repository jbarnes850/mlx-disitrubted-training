#!/bin/bash

# Quick start script for MLX distributed training
echo "MLX Distributed Training Quick Start"

# Source environment
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found. Running setup..."
    ./scripts/verify_setup.sh
fi
source venv/bin/activate

# Verify dependencies are installed
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

# Run verification with mode
MODE=${1:-"single"}
./scripts/verify_setup.sh $MODE

# Start training based on mode
case $MODE in
    "single")
        echo "Starting single-device training..."
        python train.py --config configs/single_device_config.json
        ;;
    "distributed")
        echo "Starting distributed training..."
        ./scripts/init_cluster.sh
        ;;
    *)
        echo "Invalid mode. Use 'single' or 'distributed'"
        exit 1
        ;;
esac 