import mlx.core as mx
import logging
import json
import socket
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio

from src.training.distributed_trainer import DistributedTrainer
from src.monitoring.dashboard import PerformanceDashboard
from src.utils.recovery_utils import ErrorRecovery, RecoveryConfig
from src.utils.memory_utils import AdvancedMemoryManager, MemoryConfig
from src.utils.network_utils import DistributedCommunicator, NetworkConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkerNode:
    """Worker node for distributed training"""
    def __init__(self, config_path: str = "configs/distributed_config.json"):
        self.config = self._load_config(config_path)
        self.setup_components()
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return json.load(f)
            
    def setup_components(self):
        """Initialize worker components"""
        # Initialize memory management
        self.memory_manager = AdvancedMemoryManager(
            MemoryConfig(
                max_memory_gb=self.config["training"]["max_memory_gb"]["secondary"]
            )
        )
        
        # Initialize error recovery
        self.recovery = ErrorRecovery(RecoveryConfig())
        
        # Initialize network communication
        self.network = DistributedCommunicator(
            NetworkConfig(
                primary_host=self.config["training"]["primary_host"]
            )
        )
        
        # Initialize monitoring
        self.dashboard = PerformanceDashboard(self.config["monitoring"])
        
    async def initialize_worker(self) -> bool:
        """Initialize worker node"""
        try:
            logger.info("Initializing worker node...")
            
            # Initialize MLX distributed
            self.world = mx.distributed.init()
            logger.info(f"Worker initialized with rank {self.world.rank}")
            
            # Verify network connectivity
            if not await self.network.verify_cluster():
                raise RuntimeError("Network verification failed")
                
            # Setup memory limits
            if mx.metal.is_available():
                memory_limit = self.config["training"]["max_memory_gb"]["secondary"] * 1024 * 1024 * 1024
                mx.metal.set_memory_limit(memory_limit)
                logger.info(f"Memory limit set to {memory_limit / 1e9:.1f} GB")
                
            # Start monitoring
            self.dashboard.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Worker initialization failed: {str(e)}")
            return False
            
    async def run(self):
        """Main worker loop"""
        try:
            # Initialize worker
            if not await self.initialize_worker():
                raise RuntimeError("Worker initialization failed")
                
            logger.info("Worker ready for training")
            
            while True:
                try:
                    # Wait for and process coordinator commands
                    command = await self.network.receive_command()
                    
                    if command["type"] == "train":
                        await self.handle_training(command["data"])
                    elif command["type"] == "evaluate":
                        await self.handle_evaluation(command["data"])
                    elif command["type"] == "shutdown":
                        break
                        
                except Exception as e:
                    if not await self.handle_error(e):
                        raise
                        
        except Exception as e:
            logger.error(f"Worker failed: {str(e)}")
        finally:
            await self.shutdown()
            
    async def handle_training(self, batch_data: Dict[str, Any]):
        """Handle training step"""
        try:
            # Process batch
            loss, grads = await self.trainer.train_step(batch_data)
            
            # Sync gradients
            synced_grads = await self.network.sync_gradients(grads)
            
            # Update metrics
            self.dashboard.track_training_metrics(
                loss=loss,
                grad_norm=mx.norm(synced_grads).item(),
                memory_gb=mx.metal.get_active_memory() / 1e9
            )
            
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}")
            raise
            
    async def handle_evaluation(self, eval_data: Dict[str, Any]):
        """Handle evaluation step"""
        try:
            results = await self.trainer.evaluate(eval_data)
            await self.network.send_results(results)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
            
    async def handle_error(self, error: Exception) -> bool:
        """Handle worker errors"""
        logger.error(f"Error occurred: {str(error)}")
        
        try:
            if isinstance(error, mx.metal.OutOfMemoryError):
                return await self.recovery.handle_memory_error(
                    mx.metal.get_active_memory() / 1e9
                )
            elif isinstance(error, ConnectionError):
                return await self.recovery.handle_network_error(error)
            else:
                # Clear memory and retry
                mx.metal.clear_cache()
                return True
                
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {str(recovery_error)}")
            return False
            
    async def shutdown(self):
        """Clean shutdown of worker"""
        logger.info("Shutting down worker...")
        
        try:
            # Stop monitoring
            self.dashboard.shutdown()
            
            # Clean up memory
            self.memory_manager.defragment()
            
            # Notify coordinator
            await self.network.send_command({
                "type": "shutdown",
                "status": "clean"
            })
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")
            
        logger.info("Worker shutdown complete")

async def main():
    """Main worker entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="MLX Worker Node")
    parser.add_argument("--config", default="configs/distributed_config.json")
    args = parser.parse_args()
    
    worker = WorkerNode(args.config)
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main()) 