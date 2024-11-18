import mlx.core as mx
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import asyncio
from pathlib import Path
import json
import time
import numpy as np

@dataclass
class EvalConfig:
    """Configuration for continuous evaluation"""
    eval_frequency: int = 100  # Steps between evaluations
    metrics: List[str] = None  # Metrics to track
    min_eval_samples: int = 1000  # Minimum samples for evaluation
    save_results: bool = True
    output_dir: str = "eval_results"
    alert_thresholds: Dict[str, float] = None

class ContinuousEvaluator:
    """Continuous evaluation during training"""
    def __init__(
        self,
        model: Any,
        eval_datasets: Dict[str, Any],
        config: EvalConfig
    ):
        self.model = model
        self.eval_datasets = eval_datasets
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics tracking
        self.metrics = {
            "perplexity": [],
            "accuracy": [],
            "loss": []
        }
        
        if config.metrics:
            self.metrics.update({m: [] for m in config.metrics})
            
        # Setup output directory
        if config.save_results:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            
    async def evaluate_step(
        self,
        step: int,
        current_loss: float
    ) -> Dict[str, float]:
        """Run evaluation at current step"""
        if step % self.config.eval_frequency != 0:
            return {}
            
        self.logger.info(f"Running evaluation at step {step}")
        results = {
            "step": step,
            "training_loss": current_loss,
            "timestamp": time.time()
        }
        
        # Evaluate on each dataset
        for dataset_name, dataset in self.eval_datasets.items():
            scores = await self._evaluate_dataset(dataset)
            results[dataset_name] = scores
            
        # Update metrics history
        for metric, value in results.items():
            if metric in self.metrics:
                self.metrics[metric].append(value)
                
        # Save results
        if self.config.save_results:
            self._save_results(results)
            
        # Check alerts
        self._check_alerts(results)
        
        return results
        
    async def _evaluate_dataset(
        self,
        dataset: Any
    ) -> Dict[str, float]:
        """Evaluate model on a specific dataset"""
        total_loss = 0
        total_samples = 0
        
        for batch in dataset:
            # Forward pass
            loss = self.model(batch)
            total_loss += loss.item()
            total_samples += len(batch)
            
            if total_samples >= self.config.min_eval_samples:
                break
                
        return {
            "loss": total_loss / total_samples,
            "perplexity": np.exp(total_loss / total_samples)
        }
        
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        output_file = Path(self.config.output_dir) / f"eval_{results['step']}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    def _check_alerts(self, results: Dict[str, Any]):
        """Check results against alert thresholds"""
        if not self.config.alert_thresholds:
            return
            
        for metric, threshold in self.config.alert_thresholds.items():
            if metric in results:
                value = results[metric]
                if isinstance(value, dict):
                    value = value.get('loss', float('inf'))  # Default to loss for dataset results
                    
                if value > threshold:
                    self.logger.warning(
                        f"Alert: {metric} ({value:.4f}) exceeded threshold ({threshold:.4f})"
                    )
                    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary"""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[metric] = {
                    "current": values[-1],
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                
        return summary 