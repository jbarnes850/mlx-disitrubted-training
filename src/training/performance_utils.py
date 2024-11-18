import mlx.core as mx
import psutil
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class PerformanceConfig:
    initial_batch_size: int
    min_batch_size: int
    max_batch_size: int
    target_memory_usage: float = 0.85  # Target GPU memory utilization
    checkpointing_layers: Optional[list] = None
    gradient_accumulation_steps: int = 1

class PerformanceOptimizer:
    """Manages training performance optimizations"""
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.current_batch_size = config.initial_batch_size
        self.memory_history = []
        self.throughput_history = []
        
    def optimize_batch_size(self, current_memory_usage: float) -> int:
        """Dynamically adjust batch size based on memory usage"""
        self.memory_history.append(current_memory_usage)
        
        # Use moving average for stability
        avg_memory = np.mean(self.memory_history[-10:])
        
        # Adjust batch size
        if avg_memory > self.config.target_memory_usage:
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif avg_memory < self.config.target_memory_usage * 0.8:
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            
        return self.current_batch_size
    
    def setup_gradient_checkpointing(
        self,
        model: mx.nn.Module,
        checkpoint_layers: Optional[list] = None
    ) -> mx.nn.Module:
        """Configure gradient checkpointing for memory efficiency"""
        if checkpoint_layers is None:
            # Default to checkpointing every other transformer layer
            checkpoint_layers = list(range(1, len(model.layers), 2))
            
        for idx in checkpoint_layers:
            layer = model.layers[idx]
            # Enable checkpointing for attention and MLP computations
            layer.attention = mx.checkpoint(layer.attention)
            layer.feed_forward = mx.checkpoint(layer.feed_forward)
            
        return model
    
    def optimize_compute_schedule(
        self,
        batch: Dict[str, mx.array]
    ) -> Tuple[mx.Stream, mx.Stream]:
        """Create optimized compute schedule"""
        # Create separate streams for compute and memory operations
        compute_stream = mx.Stream(mx.gpu)
        memory_stream = mx.Stream(mx.cpu)
        
        # Prefetch next batch while computing current batch
        with mx.stream(memory_stream):
            # Ensure inputs are on GPU
            for k, v in batch.items():
                if not isinstance(v, mx.array):
                    batch[k] = mx.array(v)
                    
        return compute_stream, memory_stream 