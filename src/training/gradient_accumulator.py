import mlx.core as mx
from dataclasses import dataclass
from typing import Dict, Optional
import logging
import numpy as np

@dataclass
class AccumulationConfig:
    initial_steps: int = 4
    max_steps: int = 32
    min_steps: int = 1
    target_batch_size: int = 1024
    memory_threshold: float = 0.9

class GradientAccumulator:
    """Dynamic gradient accumulation"""
    def __init__(self, config: AccumulationConfig):
        self.config = config
        self.current_steps = config.initial_steps
        self.logger = logging.getLogger(__name__)
        self.grad_norms = []
        
    def update(
        self,
        current_batch_size: int,
        memory_usage: float,
        grad_norm: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update accumulation steps"""
        # Track gradient norms
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
            
        # Calculate effective batch size
        effective_batch_size = current_batch_size * self.current_steps
        
        # Adjust based on memory and target batch size
        if memory_usage > self.config.memory_threshold:
            # Increase accumulation to reduce memory
            new_steps = min(
                self.config.max_steps,
                self.current_steps + 1
            )
        elif effective_batch_size < self.config.target_batch_size:
            # Increase accumulation to reach target
            new_steps = min(
                self.config.max_steps,
                self.current_steps + 1
            )
        else:
            new_steps = self.current_steps
            
        # Adjust based on gradient stability
        if len(self.grad_norms) >= 10:
            grad_std = np.std(self.grad_norms[-10:])
            if grad_std > 1.0:
                # Increase accumulation for stability
                new_steps = min(
                    self.config.max_steps,
                    new_steps + 1
                )
                
        # Log changes
        if new_steps != self.current_steps:
            self.logger.info(
                f"Adjusting gradient accumulation: {self.current_steps} -> {new_steps}"
            )
            self.current_steps = new_steps
            
        return {
            "accumulation_steps": self.current_steps,
            "effective_batch_size": current_batch_size * self.current_steps
        }
        
    def should_update(self, step: int) -> bool:
        """Check if gradients should be applied"""
        return step % self.current_steps == 0 