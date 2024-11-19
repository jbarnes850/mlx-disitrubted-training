#!/bin/bash
set -e

# Default values
CONFIG_FILE=${1:-"configs/personal_ai_config.json"}
DEVICE=${2:-"gpu"}
WORLD_SIZE=${3:-1}
PRIVACY_LEVEL=${4:-"high"}

# Source environment
source .venv/bin/activate

# Verify dependencies
python -c "import mlx.core; import mlx.nn" || {
    echo "MLX not properly installed"
    exit 1
}

# Create data directories
mkdir -p data/base_knowledge
mkdir -p data/personal_data
mkdir -p checkpoints

# Verify personal data access and encryption
echo "Verifying personal data security..."
python scripts/verify_setup.py \
    --check-encryption \
    --privacy-level "$PRIVACY_LEVEL" \
    || exit 1

# Start distributed training
echo "Starting personal AI training..."
echo "Config: $CONFIG_FILE"
echo "Device: $DEVICE"
echo "World Size: $WORLD_SIZE"
echo "Privacy Level: $PRIVACY_LEVEL"

# Run training
python train_distributed.py \
    --base-knowledge-dir data/base_knowledge \
    --personal-data-dir data/personal_data \
    --privacy-level "$PRIVACY_LEVEL" \
    --device "$DEVICE" \
    --world-size "$WORLD_SIZE" \
    --batch-size 32 \
    --grad-accum-steps 4 \
    --max-steps 100000 \
    --warmup-steps 2000 \
    --checkpoint-dir checkpoints \
    --log-file logs/training.log

# Verify training results
echo "Verifying training results..."
python scripts/verify_results.py \
    --model-dir checkpoints \
    --privacy-check
