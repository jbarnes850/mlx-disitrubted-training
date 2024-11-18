import mlx.core as mx
from mlx.utils import tree_map
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from src.utils.network_utils import DistributedCommunicator, NetworkConfig
from src.utils.memory_utils import AdvancedMemoryManager, MemoryConfig
from src.evaluation.continuous_eval import ContinuousEvaluator, EvalConfig
from src.monitoring.dashboard import PerformanceDashboard, DashboardConfig
from src.training.scheduler import TrainingScheduler, SchedulerConfig
from mpi4py import MPI
import asyncio
import time

@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    batch_size: int
    gradient_accumulation_steps: int
    max_memory_gb: float
    prefetch_batches: int
    mixed_precision: bool
    device_map: Dict[str, str]
    primary_host: str = "localhost"
    primary_port: int = 29500
    sync_weights_every: int = 100
    eval_frequency: int = 100
    checkpoint_frequency: int = 1000
    max_steps: int = 100000
    enable_monitoring: bool = True

class DistributedTrainer:
    """Enhanced distributed trainer with comprehensive monitoring and optimization"""
    def __init__(
        self,
        config: TrainingConfig,
        model: mx.nn.Module,
        optimizer: mx.optimizers.Optimizer,
        eval_datasets: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.step = 0
        
        # Initialize distributed setup
        self.world = mx.distributed.init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.world.rank
        self.size = self.world.size
        
        # Configure MPI parameters
        if self.size > 1:
            MPI.Info.Set("btl_tcp_links", "4")
        
        # Initialize components
        self.model = model
        self.optimizer = optimizer
        self.setup_components(eval_datasets)
        
    def setup_components(self, eval_datasets: Optional[Dict[str, Any]]):
        """Initialize all training components"""
        # Setup memory management
        self.memory_manager = AdvancedMemoryManager(
            MemoryConfig(max_memory_gb=self.config.max_memory_gb)
        )
        self.model = self.memory_manager.optimize_memory_layout(self.model)
        
        # Setup network communication
        self.network = DistributedCommunicator(
            NetworkConfig(
                primary_host=self.config.primary_host,
                primary_port=self.config.primary_port
            )
        )
        
        # Setup scheduler
        self.scheduler = TrainingScheduler(
            SchedulerConfig(
                initial_lr=self.optimizer.learning_rate,
                warmup_steps=2000
            )
        )
        
        # Setup evaluation if datasets provided
        if eval_datasets:
            self.evaluator = ContinuousEvaluator(
                model=self.model,
                eval_datasets=eval_datasets,
                config=EvalConfig(
                    eval_frequency=self.config.eval_frequency
                )
            )
        else:
            self.evaluator = None
            
        # Setup monitoring
        if self.config.enable_monitoring:
            self.dashboard = PerformanceDashboard(
                DashboardConfig(
                    alert_thresholds={
                        "loss": 10.0,
                        "gpu_utilization": 0.95,
                        "memory_usage": 0.9
                    }
                )
            )
        
    async def train_step(
        self,
        batch: Dict[str, mx.array],
        stream: Optional[mx.Stream] = None
    ) -> Tuple[float, Dict[str, mx.array]]:
        """Execute optimized training step"""
        stream = stream or mx.default_stream()
        
        with mx.stream(stream):
            try:
                # Get current learning rate
                current_lr = self.scheduler.get_lr()
                self.optimizer.learning_rate = current_lr
                
                # Forward and backward pass
                loss, grads = mx.value_and_grad(self.model)(batch)
                
                # Gradient accumulation
                if self.step % self.config.gradient_accumulation_steps == 0:
                    # Average gradients in distributed setting
                    if self.world.size > 1:
                        grads = await self.network.sync_gradients(grads)
                    
                    # Optimizer step
                    self.optimizer.update(self.model, grads)
                    
                    # Sync weights if needed
                    if self.config.sync_weights_every > 0 and self.step % self.config.sync_weights_every == 0:
                        self.model.parameters = await self.network.broadcast_weights(self.model.parameters)
                
                # Update metrics
                if self.config.enable_monitoring:
                    self.dashboard.track_training_metrics(
                        loss=loss.item(),
                        throughput=self._compute_throughput(batch),
                        learning_rate=current_lr,
                        gradient_norm=self._compute_gradient_norm(grads)
                    )
                
                # Run evaluation if needed
                if self.evaluator and self.step % self.config.eval_frequency == 0:
                    eval_results = await self.evaluator.evaluate_step(self.step, loss.item())
                    self._handle_eval_results(eval_results)
                
                # Ensure computations are complete
                mx.eval(loss, self.model.parameters())
                
                # Update step
                self.step += 1
                self.scheduler.step_scheduler()
                
                return loss.item(), grads
                
            except Exception as e:
                logging.error(f"Error in training step: {str(e)}")
                if not self.recover_from_error(e):
                    raise
                
    async def train_epoch(
        self, 
        dataloader: Any,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with optimizations"""
        total_loss = 0
        num_batches = 0
        
        # Create streams for compute/memory overlap
        compute_stream = mx.Stream(mx.gpu)
        transfer_stream = mx.Stream(mx.cpu)
        
        # Check memory and adjust batch size if needed
        current_memory = mx.metal.get_active_memory() / (1024**3)
        new_batch_size = self.adjust_batch_size(current_memory)
        if new_batch_size != self.config.batch_size:
            self.config.batch_size = new_batch_size
            logging.info(f"Adjusted batch size to {new_batch_size}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Prefetch next batch
            if batch_idx + 1 < len(dataloader):
                with mx.stream(transfer_stream):
                    next_batch = dataloader.prefetch_batch()
            
            # Train on current batch
            loss, _ = await self.train_step(batch, stream=compute_stream)
            
            # Update metrics
            total_loss += loss
            num_batches += 1
            
            # Periodic memory cleanup
            if batch_idx % 100 == 0:
                self.memory_manager.defragment()
        
        return {
            "epoch": epoch,
            "avg_loss": total_loss / num_batches,
            "batches": num_batches,
            "world_size": self.world.size,
            "learning_rate": self.scheduler.get_lr()
        }
        
    def _compute_throughput(self, batch: Dict[str, mx.array]) -> float:
        """Compute training throughput"""
        tokens_per_batch = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
        return tokens_per_batch / self.get_last_step_time()
        
    def _compute_gradient_norm(self, grads: Dict[str, mx.array]) -> float:
        """Compute gradient norm"""
        return mx.norm(mx.concatenate([g.flatten() for g in grads.values()]))
        
    def _handle_eval_results(self, results: Dict[str, Any]):
        """Handle evaluation results"""
        if self.config.enable_monitoring:
            self.dashboard.track_eval_metrics(results)
            
        # Check for early stopping
        if self.scheduler.should_stop(results.get("loss", float('inf')), self.step):
            logging.info("Early stopping triggered")
            raise StopIteration("Early stopping")
            
    def get_last_step_time(self) -> float:
        """Get time taken for last step"""
        return getattr(self, '_last_step_time', 1.0)
        
    def shutdown(self):
        """Clean shutdown of all components"""
        if self.config.enable_monitoring:
            self.dashboard.shutdown()
        self.memory_manager.defragment()
        logging.info("Trainer shutdown complete")
        return grads