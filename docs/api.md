# API Documentation

## Models

### UnifiedModel

```python
class UnifiedModel(nn.Module):
    """Unified language model with RoPE and SwiGLU"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model.
        
        Args:
            config (ModelConfig): Model configuration
        """
        
    def generate(
        self,
        x: mx.array,
        max_length: int,
        temperature: float = 1.0
    ):
        """
        Generate tokens autoregressively.
        
        Args:
            x (mx.array): Input tokens
            max_length (int): Maximum sequence length
            temperature (float): Sampling temperature
            
        Yields:
            mx.array: Generated tokens
        """
```

## Training

### DistributedTrainer

```python
class DistributedTrainer:
    """Manages distributed training across devices"""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: optim.Optimizer
    ):
        """
        Initialize trainer.
        
        Args:
            config (TrainingConfig): Training configuration
            model (nn.Module): Model to train
            optimizer (optim.Optimizer): Optimizer instance
        """
        
    def train_step(
        self,
        batch: Dict[str, mx.array]
    ) -> Tuple[float, Dict[str, mx.array]]:
        """
        Execute single training step.
        
        Args:
            batch (Dict[str, mx.array]): Input batch
            
        Returns:
            Tuple[float, Dict[str, mx.array]]: Loss and gradients
        """
```

## Inference

### InferenceServer

```python
class InferenceServer:
    """Serves model for inference"""
    
    def __init__(
        self,
        model_path: str,
        model_config: ModelConfig,
        inference_config: InferenceConfig
    ):
        """
        Initialize server.
        
        Args:
            model_path (str): Path to model weights
            model_config (ModelConfig): Model configuration
            inference_config (InferenceConfig): Inference settings
        """
```

## Monitoring

### PerformanceDashboard

```python
class PerformanceDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize dashboard.
        
        Args:
            config (DashboardConfig): Dashboard configuration
        """
        
    def track_training_metrics(
        self,
        throughput: float,
        communication_time: float,
        cache_hit_rate: float
    ):
        """
        Track training metrics.
        
        Args:
            throughput (float): Training throughput
            communication_time (float): Communication overhead
            cache_hit_rate (float): Cache hit rate
        """
``` 