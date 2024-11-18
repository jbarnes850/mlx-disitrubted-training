#!/bin/bash
set -e

echo "Running test suite..."

# Activate environment
source .venv/bin/activate

# Run tests with coverage
pytest tests/ \
    --cov=src \
    --cov-report=html \
    -v

# Run distributed tests if multiple devices
if [ "$1" == "distributed" ]; then
    echo "Running distributed tests..."
    mpirun -np 2 pytest tests/test_distributed.py -v
fi

echo "Tests complete!" 