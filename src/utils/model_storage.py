import mlx.core as mx
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
import shutil
import time

logger = logging.getLogger(__name__)

class ModelStorage:
    """Handles model saving and versioning"""
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_model(
        self,
        model: Any,
        metrics: Dict[str, float],
        version: Optional[str] = None,
        is_checkpoint: bool = False
    ) -> str:
        """Save model with metadata"""
        # Generate version if not provided
        if not version:
            version = time.strftime("%Y%m%d_%H%M%S")
            
        # Create save directory
        save_dir = self.base_path / ("checkpoints" if is_checkpoint else "releases") / version
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model weights
            mx.save(save_dir / "weights.safetensors", model.parameters())
            
            # Save model config
            config = {
                "version": version,
                "timestamp": time.time(),
                "metrics": metrics,
                "model_config": model.config if hasattr(model, "config") else {},
                "is_checkpoint": is_checkpoint
            }
            
            with open(save_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
                
            # If it's a release, create latest symlink
            if not is_checkpoint:
                latest = self.base_path / "releases" / "latest"
                if latest.exists():
                    latest.unlink()
                latest.symlink_to(save_dir)
                
            logger.info(f"Model saved successfully to {save_dir}")
            return str(save_dir)
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            if save_dir.exists():
                shutil.rmtree(save_dir)
            raise

    def load_latest(self) -> Dict[str, Any]:
        """Load latest release model"""
        latest_path = self.base_path / "releases" / "latest"
        if not latest_path.exists():
            raise FileNotFoundError("No latest model found")
            
        return self.load_version(latest_path.resolve().name)
        
    def load_version(self, version: str) -> Dict[str, Any]:
        """Load specific model version"""
        version_path = self.base_path / "releases" / version
        if not version_path.exists():
            raise FileNotFoundError(f"Model version {version} not found")
            
        # Load weights
        weights = mx.load(version_path / "weights.safetensors")
        
        # Load config
        with open(version_path / "config.json") as f:
            config = json.load(f)
            
        return {
            "weights": weights,
            "config": config
        } 