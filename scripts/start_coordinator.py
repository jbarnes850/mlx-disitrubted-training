import mlx.core as mx
import logging
import json
from pathlib import Path
from src.training.distributed_trainer import DistributedTrainer
from src.models.unified_model import UnifiedModel, ModelConfig
from src.monitoring.dashboard import PerformanceDashboard
import socket
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingCoordinator:
    """Coordinates distributed training across devices"""
    def __init__(self, config_path: str = "configs/distributed_config.json"):
        self.config = self._load_config(config_path)
        self.dashboard = PerformanceDashboard(self.config["monitoring"])
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return json.load(f)
            
    def verify_devices(self) -> bool:
        """Verify all devices are ready"""
        devices = self.config["training"]["device_map"]
        for device, role in devices.items():
            if device != "primary":
                try:
                    # Test SSH connection
                    result = subprocess.run(
                        ["ssh", f"{device}", "python3 -c 'import mlx'"],
                        capture_output=True
                    )
                    if result.returncode != 0:
                        logger.error(f"Failed to connect to {device}")
                        return False
                except Exception as e:
                    logger.error(f"Device verification failed: {str(e)}")
                    return False
        return True
        
    def initialize_training(self):
        """Initialize distributed training"""
        logger.info("Initializing distributed training...")
        
        # Verify devices
        if not self.verify_devices():
            raise RuntimeError("Device verification failed")
            
        # Initialize model
        model_config = ModelConfig(**self.config["model"])
        model = UnifiedModel(model_config)
        
        # Start monitoring
        self.dashboard.start()
        
        # Initialize trainer
        trainer = DistributedTrainer(
            config=self.config["training"],
            model=model
        )
        
        return trainer
        
    def start(self):
        """Start training coordination"""
        try:
            trainer = self.initialize_training()
            logger.info("Starting distributed training")
            trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.shutdown()
            
    def shutdown(self):
        """Clean shutdown of all components"""
        logger.info("Shutting down coordinator")
        self.dashboard.shutdown()

def main():
    coordinator = TrainingCoordinator()
    coordinator.start()

if __name__ == "__main__":
    main() 