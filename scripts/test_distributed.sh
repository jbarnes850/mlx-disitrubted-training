#!/bin/bash

# Test distributed setup with small model
echo "Testing distributed setup..."

# Create test config
cat > configs/test_config.json << EOL
{
    "model": {
        "num_layers": 4,
        "dims": 256,
        "num_heads": 4,
        "vocab_size": 1000
    },
    "training": {
        "batch_size": {
            "primary": 8,
            "secondary": 4
        },
        "gradient_accumulation_steps": 4,
        "max_steps": 10
    }
}
EOL

# Run test training
python train.py \
    --config configs/test_config.json \
    --test-mode \
    --role primary &

sleep 2

python train.py \
    --config configs/test_config.json \
    --test-mode \
    --role secondary

# Check results
python scripts/verify_results.py 