import mlx.core as mx
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

@dataclass
class ProfileConfig:
    log_frequency: int = 100
    profile_memory: bool = True
    profile_compute: bool = True
    output_path: Optional[str] = "profile_results.json"

class DistributedProfiler:
    """Profile distributed training performance"""
    def __init__(self, config: ProfileConfig):
        self.config = config
        self.step = 0
        self.epoch = 0
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
    def start_epoch(self) -> None:
        """Mark start of epoch"""
        self.epoch_start = time.time()
        
    def end_epoch(self) -> None:
        """Record epoch metrics"""
        epoch_time = time.time() - self.epoch_start
        self.metrics['epoch_time'].append(epoch_time)
        self.epoch += 1
        
        if self.config.output_path:
            self._save_metrics()
    
    def step_metrics(
        self,
        loss: float,
        grad_norm: Optional[float] = None,
        **kwargs
    ) -> None:
        """Record per-step metrics"""
        self.metrics['loss'].append(loss)
        if grad_norm is not None:
            self.metrics['grad_norm'].append(grad_norm)
            
        # Record custom metrics
        for k, v in kwargs.items():
            self.metrics[k].append(v)
        
        # Profile system metrics
        if self.config.profile_memory and self.step % self.config.log_frequency == 0:
            self._profile_memory()
            
        if self.config.profile_compute and self.step % self.config.log_frequency == 0:
            self._profile_compute()
            
        self.step += 1
    
    def _profile_memory(self) -> None:
        """Record memory usage metrics"""
        if mx.metal.is_available():
            self.metrics['metal_active'].append(
                mx.metal.get_active_memory() / (1024**3)
            )
            self.metrics['metal_peak'].append(
                mx.metal.get_peak_memory() / (1024**3)
            )
            self.metrics['metal_cache'].append(
                mx.metal.get_cache_memory() / (1024**3)
            )
    
    def _profile_compute(self) -> None:
        """Record compute utilization metrics"""
        # Record steps per second
        steps_per_sec = self.step / (time.time() - self.start_time)
        self.metrics['steps_per_sec'].append(steps_per_sec)
        
    def _save_metrics(self) -> None:
        """Save metrics to disk"""
        if self.config.output_path:
            output_path = Path(self.config.output_path)
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
    def summary(self) -> Dict[str, Any]:
        """Get training summary metrics"""
        total_time = time.time() - self.start_time
        
        return {
            'total_time': total_time,
            'epochs': self.epoch,
            'steps': self.step,
            'avg_loss': sum(self.metrics['loss']) / len(self.metrics['loss']),
            'steps_per_sec': self.step / total_time,
            'peak_memory_gb': max(self.metrics['metal_peak']) if self.metrics['metal_peak'] else None
        } 