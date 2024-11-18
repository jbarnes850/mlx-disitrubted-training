import mlx.core as mx
from src.models.unified_model import UnifiedModel, ModelConfig
from src.inference.inference_server import InferenceServer, InferenceConfig
import uvicorn
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)

def main():
    # Load configurations
    with open("configs/distributed_config.json") as f:
        config = json.load(f)
    
    # Initialize model
    model_config = ModelConfig(
        num_layers=32,
        vocab_size=32000,
        dims=4096,
        mlp_dims=11008,
        num_heads=32
    )
    
    # Setup inference server
    server = InferenceServer(
        model_path=config["inference"]["model_path"],
        model_config=model_config,
        inference_config=InferenceConfig(**config["inference"]),
        memory_limit_gb=config["memory"]["inference_memory_gb"]
    )
    
    # Run server
    uvicorn.run(
        server.app,
        host=config["inference"]["host"],
        port=config["inference"]["port"]
    )

if __name__ == "__main__":
    main() 