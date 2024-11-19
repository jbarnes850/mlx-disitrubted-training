import mlx.core as mx
from mlx.utils import tree_map
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from src.utils.network_utils import DistributedCommunicator, NetworkConfig
from src.utils.memory_utils import AdvancedMemoryManager
from src.training.batch_optimizer import BatchOptimizer, BatchOptimizerConfig
from src.models.personal_ai import PersonalAIModel, PersonalAIConfig
from src.datasets.personal_dataset import PersonalDataset
from mpi4py import MPI
import asyncio
import time
import math

@dataclass
class OptimizedTrainingConfig:
    """Enhanced training configuration for distributed training"""
    # Model configuration
    model_config: PersonalAIConfig
    
    # Training hyperparameters
    initial_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    
    # Distributed training settings
    world_size: int = 1
    primary_device: str = "gpu"
    sync_interval: int = 100
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    
    # Resource management
    max_memory_gb: float = 80
    prefetch_batches: int = 2
    network_threshold_mbps: int = 1000
    
    # Training duration
    max_steps: int = 100000
    warmup_steps: int = 2000
    
    # Checkpoint directory
    checkpoint_dir: str = "checkpoints"

class OptimizedTrainer:
    """Enhanced distributed trainer with OpenELM optimizations"""
    def __init__(
        self,
        config: OptimizedTrainingConfig,
        model: PersonalAIModel,
        dataset: PersonalDataset,
        optimizer: Any
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        
        # Initialize distributed communication
        self.comm = DistributedCommunicator(
            world_size=config.world_size,
            network_config=NetworkConfig(
                bandwidth_threshold_mbps=config.network_threshold_mbps
            )
        )
        
        # Setup memory management
        self.memory_manager = AdvancedMemoryManager(
            max_memory_gb=config.max_memory_gb,
            model=model
        )
        
        # Initialize batch optimizer
        self.batch_optimizer = BatchOptimizer(
            config=BatchOptimizerConfig(
                initial_batch_size=config.initial_batch_size,
                grad_accum_steps=config.gradient_accumulation_steps,
                max_grad_norm=config.max_grad_norm
            ),
            model=model,
            optimizer=optimizer
        )
        
        self.step = 0
        self._setup_logging()
        
    async def train(self):
        """Main training loop with personal data integration"""
        while self.step < self.config.max_steps:
            try:
                # Get next batch (combines base knowledge and personal data)
                input_ids, attention_mask, context = self.dataset.get_training_batch()
                
                # Forward pass with context integration
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    context_embedding=context
                )
                
                # Compute loss and optimize
                loss = self._compute_loss(outputs, input_ids)
                self.batch_optimizer.backward_step(loss)
                
                # Sync gradients across devices
                if self.step % self.config.sync_interval == 0:
                    await self.comm.sync_gradients(self.model)
                
                # Log progress
                if self.step % 10 == 0:
                    self._log_progress(loss)
                
                # Save checkpoint
                if self.step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                self.step += 1
                
            except Exception as e:
                logging.error(f"Error during training step {self.step}: {str(e)}")
                raise
                
        logging.info("Training completed successfully")
        self._save_checkpoint()
        
    def _compute_loss(self, outputs, targets):
        """Compute loss with proper masking"""
        logits = outputs.logits[:, :-1]  # Remove last prediction
        targets = targets[:, 1:]         # Remove first token (shift right)
        
        # Compute cross entropy loss
        loss = mx.mean(
            mx.losses.cross_entropy(logits, targets)
        )
        return loss
        
    def _log_progress(self, loss):
        """Log training progress"""
        logging.info(
            f"Step {self.step}/{self.config.max_steps} "
            f"Loss: {loss.item():.4f} "
            f"Memory: {self.memory_manager.get_memory_usage_gb():.2f}GB"
        )
        
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }
        mx.save(
            f"{self.config.checkpoint_dir}/checkpoint_{self.step}.npz",
            checkpoint
        )
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )
