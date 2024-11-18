import mlx.core as mx
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CheckpointConfig:
    save_dir: str
    save_frequency: int
    keep_last_k: int = 5
    save_optimizer: bool = True

class CheckpointManager:
    """Manage model checkpoints"""
    def __init__(
        self,
        config: CheckpointConfig,
        model: mx.nn.Module,
        optimizer: Optional[mx.optimizers.Optimizer] = None
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        
        # Setup save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.step = 0
        self.saved_checkpoints = []
        
    def save(
        self,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save checkpoint"""
        if step is not None:
            self.step = step
            
        # Save model weights
        model_path = self.save_dir / f"model_{self.step}.safetensors"
        mx.save(model_path, self.model.parameters())
        
        # Save optimizer state
        if self.optimizer and self.config.save_optimizer:
            opt_path = self.save_dir / f"optimizer_{self.step}.safetensors"
            mx.save(opt_path, self.optimizer.state)
            
        # Save metrics
        if metrics:
            metrics_path = self.save_dir / f"metrics_{self.step}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        # Update checkpoint tracking
        self.saved_checkpoints.append(self.step)
        self._cleanup_old_checkpoints()
        
        logging.info(f"Saved checkpoint at step {self.step}")
        
    def load(
        self,
        step: Optional[int] = None,
        model_only: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint"""
        # Find latest checkpoint if step not specified
        if step is None:
            if not self.saved_checkpoints:
                return None
            step = max(self.saved_checkpoints)
            
        # Load model weights
        model_path = self.save_dir / f"model_{step}.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"No checkpoint found at step {step}")
            
        params = mx.load(model_path)
        self.model.update(params)
        
        # Load optimizer state
        if not model_only and self.optimizer and self.config.save_optimizer:
            opt_path = self.save_dir / f"optimizer_{step}.safetensors"
            if opt_path.exists():
                opt_state = mx.load(opt_path)
                self.optimizer.state.update(opt_state)
                
        # Load metrics
        metrics = None
        metrics_path = self.save_dir / f"metrics_{step}.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
                
        logging.info(f"Loaded checkpoint from step {step}")
        return metrics
        
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints"""
        if self.config.keep_last_k > 0:
            checkpoints_to_remove = sorted(self.saved_checkpoints)[:-self.config.keep_last_k]
            
            for step in checkpoints_to_remove:
                # Remove model weights
                model_path = self.save_dir / f"model_{step}.safetensors"
                if model_path.exists():
                    model_path.unlink()
                    
                # Remove optimizer state
                if self.config.save_optimizer:
                    opt_path = self.save_dir / f"optimizer_{step}.safetensors"
                    if opt_path.exists():
                        opt_path.unlink()
                        
                # Remove metrics
                metrics_path = self.save_dir / f"metrics_{step}.json"
                if metrics_path.exists():
                    metrics_path.unlink()
                    
                self.saved_checkpoints.remove(step) 