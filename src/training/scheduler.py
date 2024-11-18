import mlx.core as mx
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging
import math

@dataclass
class SchedulerConfig:
    """Configuration for learning rate and early stopping"""
    # Learning rate scheduling
    initial_lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 2000
    decay_steps: int = 50000
    schedule_type: str = "cosine"  # linear, cosine, exponential
    
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4
    min_epochs: int = 3
    
    # Monitoring
    eval_frequency: int = 100
    smoothing_factor: float = 0.95  # For loss smoothing

class TrainingScheduler:
    """Manages learning rate scheduling and early stopping"""
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history: List[float] = []
        self.smoothed_loss = None
        self.logger = logging.getLogger(__name__)
        
    def get_lr(self) -> float:
        """Get current learning rate based on schedule"""
        if self.step < self.config.warmup_steps:
            # Linear warmup
            return self.config.initial_lr * (self.step / self.config.warmup_steps)
            
        # Apply chosen schedule after warmup
        progress = (self.step - self.config.warmup_steps) / self.config.decay_steps
        progress = min(1.0, progress)
        
        if self.config.schedule_type == "linear":
            return self._linear_schedule(progress)
        elif self.config.schedule_type == "cosine":
            return self._cosine_schedule(progress)
        else:  # exponential
            return self._exponential_schedule(progress)
            
    def should_stop(self, current_loss: float, epoch: int) -> bool:
        """Check if training should stop"""
        # Update loss history
        self.loss_history.append(current_loss)
        
        # Update smoothed loss
        if self.smoothed_loss is None:
            self.smoothed_loss = current_loss
        else:
            self.smoothed_loss = (
                self.config.smoothing_factor * self.smoothed_loss +
                (1 - self.config.smoothing_factor) * current_loss
            )
        
        # Don't stop before minimum epochs
        if epoch < self.config.min_epochs:
            return False
            
        # Check improvement
        if (self.best_loss - self.smoothed_loss) > self.config.min_delta:
            self.best_loss = self.smoothed_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Log warning if close to stopping
        if self.patience_counter >= self.config.patience - 2:
            self.logger.warning(
                f"Close to early stopping: {self.patience_counter}/{self.config.patience}"
            )
            
        return self.patience_counter >= self.config.patience
        
    def step_scheduler(self):
        """Update scheduler state"""
        self.step += 1
        
    def get_state(self) -> Dict[str, Any]:
        """Get current scheduler state"""
        return {
            "step": self.step,
            "current_lr": self.get_lr(),
            "best_loss": self.best_loss,
            "patience_counter": self.patience_counter,
            "smoothed_loss": self.smoothed_loss
        }
        
    def _linear_schedule(self, progress: float) -> float:
        """Linear learning rate decay"""
        return self.config.initial_lr * (1 - progress)
        
    def _cosine_schedule(self, progress: float) -> float:
        """Cosine learning rate decay"""
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.config.min_lr + (self.config.initial_lr - self.config.min_lr) * cosine_decay
        
    def _exponential_schedule(self, progress: float) -> float:
        """Exponential learning rate decay"""
        return self.config.initial_lr * math.exp(-5 * progress)
        
    def plot_schedule(self, steps: int = 1000) -> Dict[str, List[float]]:
        """Generate learning rate schedule for visualization"""
        original_step = self.step
        schedule = []
        
        for step in range(steps):
            self.step = step
            schedule.append(self.get_lr())
            
        self.step = original_step
        return {
            "steps": list(range(steps)),
            "learning_rates": schedule
        }

class AdaptiveLRScheduler(TrainingScheduler):
    """Advanced scheduler with adaptive learning rate"""
    def __init__(
        self,
        config: SchedulerConfig,
        adaptation_factor: float = 0.5,
        max_lr_reduction: int = 3
    ):
        super().__init__(config)
        self.adaptation_factor = adaptation_factor
        self.max_lr_reduction = max_lr_reduction
        self.lr_reductions = 0
        self.loss_window: List[float] = []
        self.window_size = 100
        
    def adapt_lr(self, current_loss: float) -> float:
        """Adaptively adjust learning rate based on loss trend"""
        self.loss_window.append(current_loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)
            
        # Check if loss is plateauing
        if len(self.loss_window) == self.window_size:
            recent_mean = np.mean(self.loss_window[-10:])
            old_mean = np.mean(self.loss_window[:10])
            
            # If loss hasn't improved significantly
            if (old_mean - recent_mean) / old_mean < 0.01:
                if self.lr_reductions < self.max_lr_reduction:
                    self.config.initial_lr *= self.adaptation_factor
                    self.lr_reductions += 1
                    self.logger.info(f"Reducing learning rate to {self.config.initial_lr}")
                    
        return self.get_lr() 