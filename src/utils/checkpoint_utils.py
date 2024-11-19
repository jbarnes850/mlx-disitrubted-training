import mlx.core as mx
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import shutil
import os
import aiofiles
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mlx.distributed import DistributedCommunicator

@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    ram_dir: str = "/dev/shm/mlx_checkpoints"
    save_optimizer: bool = True
    keep_last_k: int = 3
    async_save: bool = True
    compression: bool = True
    ram_buffer_size: int = 512  # MB

class CheckpointManager:
    """Manage model checkpoints"""
    def __init__(
        self,
        config: CheckpointConfig,
        model: mx.nn.Module,
        optimizer: Optional[mx.optimizers.Optimizer] = None,
        network_communicator: Optional['DistributedCommunicator'] = None
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.network_communicator = network_communicator
        
        # Setup directories
        self.ram_dir = Path(config.ram_dir)
        self.disk_dir = Path(config.save_dir)
        self.ram_dir.mkdir(parents=True, exist_ok=True)
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-allocate RAM buffer for faster saves
        if config.async_save:
            self.ram_buffer_size = 1024 * 1024 * config.ram_buffer_size
            buffer_file = self.ram_dir / "buffer"
            if not buffer_file.exists():
                os.system(f"dd if=/dev/zero of={buffer_file} bs={self.ram_buffer_size} count=1")
                
        # Track checkpoints and network state
        self.step = 0
        self.saved_checkpoints = []
        
    async def save(
        self,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        save_network_stats: bool = True
    ) -> None:
        """Asynchronous checkpoint save with network stats"""
        if step is not None:
            self.step = step
            
        # Fast save to RAM
        ram_path = self.ram_dir / f"step_{self.step}"
        
        try:
            # Save model state
            model_state = self.model.state_dict()
            if self.config.compression:
                model_state = self._compress_state(model_state)
                
            # Add network stats if available
            if save_network_stats and self.network_communicator:
                network_stats = self.network_communicator.get_bandwidth_stats()
                if metrics is None:
                    metrics = {}
                metrics['network_stats'] = network_stats
                
            # Save to RAM first
            await self._save_to_ram(ram_path, model_state, metrics)
            
            if self.config.async_save:
                # Schedule disk save
                asyncio.create_task(self._save_to_disk(ram_path))
            else:
                # Synchronous disk save
                await self._save_to_disk(ram_path)
                
            # Update checkpoint history
            self.saved_checkpoints.append(ram_path)
            await self._cleanup_old_checkpoints()
            
        except Exception as e:
            logging.error(f"Checkpoint save failed: {str(e)}")
            raise
            
    async def _save_to_ram(
        self,
        path: Path,
        model_state: Dict[str, Any],
        metrics: Optional[Dict[str, Any]]
    ) -> None:
        """Fast save to RAM filesystem"""
        save_dict = {
            "model": model_state,
            "step": self.step,
            "metrics": metrics or {}
        }
        
        if self.optimizer and self.config.save_optimizer:
            save_dict["optimizer"] = self.optimizer.state_dict()
            
        # Use pre-allocated buffer for faster writes
        buffer_file = self.ram_dir / "buffer"
        if buffer_file.exists():
            shutil.copy2(buffer_file, path)
            
        async with aiofiles.open(path, "wb") as f:
            await f.write(json.dumps(save_dict).encode())
            
    async def _save_to_disk(self, ram_path: Path) -> None:
        """Move checkpoint from RAM to disk"""
        disk_path = self.disk_dir / ram_path.name
        try:
            # Copy from RAM to disk
            shutil.copy2(ram_path, disk_path)
            # Remove RAM copy
            ram_path.unlink()
        except Exception as e:
            logging.error(f"Failed to save checkpoint to disk: {str(e)}")
            raise
            
    def _compress_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compress model state"""
        compressed = {}
        for k, v in state.items():
            if isinstance(v, mx.array):
                # Use float16 for weights, keep other tensors as is
                if "weight" in k or "bias" in k:
                    compressed[k] = v.astype(mx.float16)
                else:
                    compressed[k] = v
            else:
                compressed[k] = v
        return compressed
        
    async def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond keep_last_k"""
        while len(self.saved_checkpoints) > self.config.keep_last_k:
            old_checkpoint = self.saved_checkpoints.pop(0)
            try:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            except Exception as e:
                logging.warning(f"Failed to remove old checkpoint: {str(e)}")
                
    async def load(self, path: Path) -> Dict[str, Any]:
        """Load checkpoint"""
        try:
            async with aiofiles.open(path, "rb") as f:
                checkpoint = json.loads(await f.read())
                
            # Load model state
            self.model.load_state_dict(checkpoint["model"])
            
            # Load optimizer state if available
            if self.optimizer and "optimizer" in checkpoint and self.config.save_optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                
            self.step = checkpoint["step"]
            return checkpoint.get("metrics", {})
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
            raise