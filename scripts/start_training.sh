#!/bin/bash
set -e

# Parse arguments
ROLE=${1:-"primary"}
CONFIG=${2:-"configs/distributed_config.json"}
DEVICE_ID=${3:-0}

echo "Starting MLX distributed training - Role: $ROLE"

# Activate environment
source .venv/bin/activate

# Verify setup
./scripts/verify_setup.sh distributed

# Start training based on role
case $ROLE in
    "primary")
        echo "Starting primary node..."
        python train.py \
            --role primary \
            --config "$CONFIG" \
            --device-id "$DEVICE_ID"
        ;;
    "secondary")
        echo "Starting secondary node..."
        python train.py \
            --role secondary \
            --config "$CONFIG" \
            --device-id "$DEVICE_ID"
        ;;
    *)
        echo "Invalid role. Use 'primary' or 'secondary'"
        exit 1
        ;;
esac 