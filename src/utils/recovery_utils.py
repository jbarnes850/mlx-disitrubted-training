import mlx.core as mx
import logging
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
import time
from pathlib import Path
import asyncio
import numpy as np

@dataclass
class RecoveryConfig:
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_batch_size: Optional[int] = None
    enable_checkpointing: bool = True
    network_timeout: float = 30.0
    memory_threshold: float = 0.95
    gradient_norm_threshold: float = 10.0

class ErrorRecovery:
    """Enhanced error recovery system"""
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.retry_count = 0
        self.last_checkpoint = None
        self.setup_checkpointing()
        
    def setup_checkpointing(self):
        """Initialize checkpoint management"""
        if self.config.enable_checkpointing:
            self.checkpoint_path = Path("checkpoints")
            self.checkpoint_path.mkdir(exist_ok=True)
            
    async def handle_network_error(self, error: Exception) -> bool:
        """Handle network-related errors"""
        self.logger.error(f"Network error: {str(error)}")
        
        try:
            # Wait for network recovery
            await asyncio.sleep(self.config.retry_delay)
            
            # Verify network connectivity
            if not await self._verify_network():
                return False
                
            # Restore from last checkpoint if available
            if self.last_checkpoint:
                await self._restore_checkpoint()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Network recovery failed: {str(e)}")
            return False
            
    async def handle_memory_error(self, current_memory: float) -> Dict[str, Any]:
        """Handle memory-related errors"""
        self.logger.warning(f"Memory usage critical: {current_memory:.2f}GB")
        
        try:
            # Clear memory
            mx.metal.clear_cache()
            mx.metal.reset_peak_memory()
            
            # Calculate new batch size
            new_config = await self._adjust_training_config(current_memory)
            
            # Verify memory after adjustment
            if not await self._verify_memory():
                raise RuntimeError("Memory still critical after adjustment")
                
            return new_config
            
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {str(e)}")
            raise
            
    async def handle_gradient_error(self, grad_norm: float) -> Dict[str, Any]:
        """Handle gradient-related errors"""
        self.logger.warning(f"Gradient norm critical: {grad_norm:.2f}")
        
        try:
            # Adjust training parameters
            new_config = {
                "learning_rate": self.current_lr * 0.5,
                "gradient_clip_norm": self.current_clip_norm * 0.8,
                "gradient_accumulation_steps": self.current_accum_steps * 2
            }
            
            # Verify stability
            if not await self._verify_stability(new_config):
                raise RuntimeError("Training still unstable after adjustment")
                
            return new_config
            
        except Exception as e:
            self.logger.error(f"Gradient recovery failed: {str(e)}")
            raise
            
    async def _verify_network(self) -> bool:
        """Verify network connectivity"""
        try:
            # Test basic connectivity
            test_tensor = mx.array([1.0])
            result = mx.distributed.all_sum(test_tensor)
            
            # Verify bandwidth
            bandwidth = await self._test_bandwidth()
            if bandwidth < 1_000:  # 1 Gbps minimum
                return False
                
            return True
            
        except Exception:
            return False
            
    async def _verify_memory(self) -> bool:
        """Verify memory usage"""
        if not mx.metal.is_available():
            return True
            
        current_usage = mx.metal.get_active_memory() / mx.metal.get_memory_limit()
        return current_usage < self.config.memory_threshold
        
    async def _verify_stability(self, config: Dict[str, Any]) -> bool:
        """Verify training stability"""
        # Run a few test steps
        test_losses = []
        for _ in range(5):
            try:
                loss = await self._test_step(config)
                test_losses.append(loss)
            except Exception:
                return False
                
        # Check loss stability
        loss_std = np.std(test_losses)
        return loss_std < 1.0
        
    async def _adjust_training_config(self, current_memory: float) -> Dict[str, Any]:
        """Dynamically adjust training configuration"""
        memory_ratio = current_memory / mx.metal.get_memory_limit()
        
        if memory_ratio > self.config.memory_threshold:
            # Calculate new batch size
            new_batch_size = int(self.current_batch_size * 0.75)
            new_batch_size = max(1, new_batch_size)
            
            # Adjust gradient accumulation
            new_accum_steps = int(self.current_accum_steps * 1.5)
            
            return {
                "batch_size": new_batch_size,
                "gradient_accumulation_steps": new_accum_steps,
                "enable_checkpoint": True
            }
            
        return {}
        
    async def _restore_checkpoint(self):
        """Restore from checkpoint"""
        if self.last_checkpoint and self.last_checkpoint.exists():
            self.logger.info(f"Restoring from checkpoint: {self.last_checkpoint}")
            # Implement checkpoint restoration logic
            pass
        
    async def with_recovery(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with automatic recovery"""
        while self.retry_count < self.config.max_retries:
            try:
                result = await operation(*args, **kwargs)
                self.retry_count = 0  # Reset on success
                return result
                
            except mx.metal.OutOfMemoryError:
                self.logger.warning("Out of memory error, attempting recovery...")
                if not await self._handle_oom_error():
                    raise
                    
            except Exception as e:
                self.logger.error(f"Operation failed: {str(e)}")
                if not await self._handle_general_error():
                    raise
                    
            self.retry_count += 1
            await asyncio.sleep(self.config.retry_delay)
            
        raise RuntimeError(f"Operation failed after {self.config.max_retries} retries")
        
    async def _handle_oom_error(self) -> bool:
        """Handle out of memory errors"""
        try:
            # Clear memory
            mx.metal.clear_cache()
            mx.metal.reset_peak_memory()
            
            # Reduce batch size if configured
            if self.config.fallback_batch_size:
                self.logger.info(f"Reducing batch size to {self.config.fallback_batch_size}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {str(e)}")
            return False
            
    async def _handle_general_error(self) -> bool:
        """Handle general errors"""
        try:
            # Basic recovery steps
            mx.metal.clear_cache()
            time.sleep(self.config.retry_delay)
            return True
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {str(e)}")
            return False
            
    async def setup_checkpointing(self):
        """Initialize checkpoint management"""
        # Add automatic checkpointing
        self.checkpoint_interval = 100
        self.checkpoint_path = Path("checkpoints")
        self.checkpoint_path.mkdir(exist_ok=True) 