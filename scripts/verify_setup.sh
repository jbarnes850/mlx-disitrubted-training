#!/bin/bash

# Verify MLX distributed training setup
echo "Verifying MLX distributed setup..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade pip
python3 -m pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.10" | bc -l) )); then
    echo "Error: Python 3.10+ required (current: $python_version)"
    exit 1
fi

# Verify key dependencies
echo "Verifying dependencies..."
python3 -c "
import mlx.core as mx
import transformers
import datasets
import fastapi
import dash

print(f'MLX version: {mx.__version__}')
print(f'Metal available: {mx.metal.is_available()}')
if mx.metal.is_available():
    print(f'Memory limit: {mx.metal.get_memory_limit() / (1024**3):.2f} GB')
"

# Verify network setup for distributed training
if [ "$1" == "distributed" ]; then
    echo "Verifying distributed setup..."
    
    # Check MPI
    if ! command -v mpirun &> /dev/null; then
        echo "Error: MPI not installed"
        exit 1
    fi
    
    # Test network speed
    if command -v iperf3 &> /dev/null; then
        echo "Testing network speed..."
        iperf3 -c secondary-device.local || {
            echo "Warning: Network speed test failed"
        }
    fi
fi

# Run basic tests
echo "Running tests..."
pytest tests/ -v

echo "Setup verification complete!"