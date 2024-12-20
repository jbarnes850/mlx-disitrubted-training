#!/bin/bash
set -e

echo "Setting up MLX distributed training environment..."

# Check Python version
python3 -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install/upgrade pip
python3 -m pip install --upgrade pip

# Install MPI (macOS specific)
if ! command -v mpirun &> /dev/null; then
    if command -v brew &> /dev/null; then
        brew install mpich
    else
        echo "Error: Homebrew not found. Please install Homebrew first."
        exit 1
    fi
fi

# Install core dependencies first
pip install -r requirements.txt

# Install development dependencies if requested
if [ "$1" = "--dev" ]; then
    echo "Installing development dependencies..."
    pip install -e ".[dev]"
fi

# Install monitoring tools if requested
if [ "$1" = "--monitoring" ]; then
    echo "Installing monitoring dependencies..."
    pip install ".[monitoring]"
fi

# Verify installation
python3 scripts/verify_setup.py

echo "Setup complete! Run 'source .venv/bin/activate' to activate the environment."