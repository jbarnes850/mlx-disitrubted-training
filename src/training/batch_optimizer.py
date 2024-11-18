import mlx.core as mx
from dataclasses import dataclass
from typing import Dict, Optional
import logging
import numpy as np

@dataclass
class BatchConfig:
    initial_batch_size: int
    min_batch_size: int = 1
    max_batch_size: int = 128
    target_memory_usage: float = 0.85
    adjustment_rate: float = 0.1
    warmup_steps: int = 100

class BatchOptimizer:
    """Dynamic batch size optimization"""
    def __init__(self, config: BatchConfig):
        self.config = config
        self.current_batch_size = config.initial_batch_size
        self.step = 0
        self.memory_history = []
        self.logger = logging.getLogger(__name__)
        
    def update(self, current_memory_gb: float) -> Dict[str, Any]:
        """Update batch size based on memory usage"""
        self.step += 1
        self.memory_history.append(current_memory_gb)
        
        # Skip during warmup
        if self.step < self.config.warmup_steps:
            return {"batch_size": self.current_batch_size}
            
        # Calculate memory trend
        memory_trend = self._calculate_trend()
        
        # Adjust batch size
        if memory_trend > self.config.target_memory_usage:
            # Decrease batch size
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * (1 - self.config.adjustment_rate))
            )
        elif memory_trend < self.config.target_memory_usage * 0.8:
            # Increase batch size
            new_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * (1 + self.config.adjustment_rate))
            )
        else:
            return {"batch_size": self.current_batch_size}
            
        # Log adjustment
        if new_batch_size != self.current_batch_size:
            self.logger.info(
                f"Adjusting batch size: {self.current_batch_size} -> {new_batch_size}"
            )
            self.current_batch_size = new_batch_size
            
        return {
            "batch_size": self.current_batch_size,
            "memory_trend": memory_trend
        }
        
    def _calculate_trend(self) -> float:
        """Calculate memory usage trend"""
        if len(self.memory_history) < 10:
            return 0.0
            
        recent = self.memory_history[-10:]
        return np.mean(recent) / mx.metal.get_memory_limit() 