#!/usr/bin/env python3
"""Comprehensive benchmarking suite for MLX Distributed Training"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.utils.benchmark_utils import (
    run_training_benchmark,
    run_inference_benchmark,
    run_memory_benchmark,
    run_network_benchmark
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument("--config", default="configs/distributed_config.json")
    parser.add_argument("--model-sizes", nargs="+", default=["1B"])
    args = parser.parse_args()
    
    results = {}
    
    # Run benchmarks for each model size
    for size in args.model_sizes:
        print(f"\nBenchmarking {size} model...")
        
        results[size] = {
            "training": run_training_benchmark(size),
            "inference": run_inference_benchmark(size),
            "memory": run_memory_benchmark(size),
            "network": run_network_benchmark()
        }
        
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main() 