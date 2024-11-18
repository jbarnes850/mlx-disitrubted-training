import mlx.core as mx
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import numpy as np

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    # Sequence length curriculum
    min_seq_length: int = 128
    max_seq_length: int = 2048
    length_warmup_steps: int = 1000
    
    # Difficulty curriculum
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0
    difficulty_warmup_steps: int = 2000
    
    # Mixing strategy
    dataset_mixing_schedule: str = "linear"  # linear, exponential, cosine
    
class CurriculumScheduler:
    """Manages curriculum learning schedules"""
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.step = 0
        
    def get_sequence_length(self) -> int:
        """Get current sequence length based on curriculum"""
        if self.step >= self.config.length_warmup_steps:
            return self.config.max_seq_length
            
        progress = self.step / self.config.length_warmup_steps
        return int(
            self.config.min_seq_length + 
            progress * (self.config.max_seq_length - self.config.min_seq_length)
        )
        
    def get_difficulty(self) -> float:
        """Get current difficulty threshold"""
        if self.step >= self.config.difficulty_warmup_steps:
            return self.config.max_difficulty
            
        progress = self.step / self.config.difficulty_warmup_steps
        return (
            self.config.min_difficulty + 
            progress * (self.config.max_difficulty - self.config.min_difficulty)
        )
        
    def get_dataset_weights(self, datasets: Dict[str, float]) -> Dict[str, float]:
        """Get dataset mixing weights based on curriculum"""
        if self.config.dataset_mixing_schedule == "linear":
            progress = min(1.0, self.step / self.config.difficulty_warmup_steps)
        elif self.config.dataset_mixing_schedule == "exponential":
            progress = 1 - np.exp(-5 * self.step / self.config.difficulty_warmup_steps)
        else:  # cosine
            progress = np.cos(np.pi * self.step / self.config.difficulty_warmup_steps - np.pi) / 2 + 0.5
            
        # Adjust weights based on progress
        weights = {}
        for dataset, base_weight in datasets.items():
            if "instruction" in dataset or "alpaca" in dataset:
                # Increase instruction data weight over time
                weights[dataset] = base_weight * (0.5 + 0.5 * progress)
            elif "code" in dataset:
                # Gradually introduce code data
                weights[dataset] = base_weight * progress
            else:
                weights[dataset] = base_weight
                
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
        
    def step_curriculum(self):
        """Update curriculum step"""
        self.step += 1
        
    def get_current_config(self) -> Dict[str, Any]:
        """Get current curriculum settings"""
        return {
            "sequence_length": self.get_sequence_length(),
            "difficulty": self.get_difficulty(),
            "step": self.step
        } 