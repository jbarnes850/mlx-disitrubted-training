#!/bin/bash

# Test distributed setup with small model
echo "Testing distributed setup..."

# Check if running in CI
if [ "$CI" = "true" ]; then
    echo "Running in CI environment - using minimal configuration"
    CONFIG_FILE="configs/ci_test_config.json"
else
    echo "Running in local environment"
    CONFIG_FILE="configs/test_config.json"
fi

# Create test config based on environment
if [ "$CI" = "true" ]; then
    # Minimal config for CI
    cat > configs/ci_test_config.json << EOL
{
    "model": {
        "num_layers": 2,
        "dims": 128,
        "num_heads": 2,
        "vocab_size": 1000
    },
    "training": {
        "batch_size": {
            "primary": 4,
            "secondary": 2
        },
        "gradient_accumulation_steps": 2,
        "max_steps": 5
    }
}
EOL
else
    # Standard test config for local testing
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
fi

# Run test training based on environment
if [ "$CI" = "true" ]; then
    echo "Running minimal distributed test in CI..."
    # Use mpirun for CI testing
    mpirun -n 2 python train.py --config $CONFIG_FILE --test-mode
else
    echo "Running full distributed test locally..."
    # Run distributed processes
    python train.py \
        --config $CONFIG_FILE \
        --test-mode \
        --role primary &

    sleep 2

    python train.py \
        --config $CONFIG_FILE \
        --test-mode \
        --role secondary
fi

# Check results with appropriate timeout
if [ "$CI" = "true" ]; then
    timeout 60s python scripts/verify_results.py --ci-mode
else
    python scripts/verify_results.py
fi

# Exit with verify_results.py exit code
exit $?