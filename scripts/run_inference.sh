#!/bin/bash

# Run inference server for MLX distributed model
echo "Starting MLX model inference server..."

# Source environment
source venv/bin/activate

# Verify dependencies
if ! python3 -c "import fastapi, uvicorn" &> /dev/null; then
    echo "Installing inference dependencies..."
    pip install -r requirements.txt
fi

# Configuration
MODEL_PATH=${1:-"checkpoints/model_latest.safetensors"}
CONFIG_FILE="configs/inference_config.json"
PORT=${2:-8000}

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Start server
echo "Starting inference server on port $PORT..."
python3 src/inference/server.py \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG_FILE" \
    --port "$PORT" 