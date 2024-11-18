import mlx.core as mx
import logging
import json
import os
from pathlib import Path
import subprocess
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_ci_environment() -> bool:
    """Check if we're running in a CI environment"""
    return os.environ.get('CI', 'false').lower() == 'true'

def verify_hardware() -> bool:
    """Verify hardware capabilities"""
    try:
        # Skip detailed hardware checks in CI environment
        if is_ci_environment():
            logger.info("Running in CI environment - skipping detailed hardware checks")
            return True

        # Check Metal availability
        if not mx.metal.is_available():
            logger.error("Metal backend not available")
            return False
            
        # Get memory limits
        memory_limit = mx.metal.get_memory_limit() / (1024**3)  # Convert to GB
        if memory_limit < 32:
            logger.error(f"Insufficient memory: {memory_limit:.1f}GB (min 32GB required)")
            return False
            
        logger.info(f"Hardware verification passed (Memory: {memory_limit:.1f}GB)")
        return True
        
    except Exception as e:
        if is_ci_environment():
            logger.warning(f"Hardware check exception in CI (expected): {str(e)}")
            return True
        logger.error(f"Hardware verification failed: {str(e)}")
        return False

def verify_network() -> bool:
    """Verify network setup"""
    try:
        # Run network tests
        result = subprocess.run(
            ["python", "scripts/test_network.py"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error("Network verification failed")
            logger.error(result.stderr)
            return False
            
        logger.info("Network verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Network verification failed: {str(e)}")
        return False

def verify_data_pipeline() -> bool:
    """Verify data pipeline"""
    try:
        # Test dataset loading
        from src.training.data_utils import OptimizedDataManager, DataConfig
        data_manager = OptimizedDataManager(DataConfig())
        dataset = data_manager.load_datasets()
        
        # Verify first batch
        batch = next(iter(dataset))
        if not all(k in batch for k in ["input_ids", "labels"]):
            logger.error("Invalid batch format")
            return False
            
        logger.info("Data pipeline verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Data pipeline verification failed: {str(e)}")
        return False

def verify_model_loading() -> bool:
    """Verify model loading"""
    try:
        # Load model configuration
        config_path = Path("configs/distributed_config.json")
        if not config_path.exists():
            logger.error("Model configuration not found")
            return False
            
        with open(config_path) as f:
            config = json.load(f)
            
        # Try loading model
        from src.models.unified_model import UnifiedModel, ModelConfig
        model = UnifiedModel(ModelConfig(**config["model"]))
        
        # Test forward pass
        batch = mx.random.randint(0, config["model"]["vocab_size"], (2, 16))
        output = model(batch)
        
        if output.shape != (2, 16, config["model"]["vocab_size"]):
            logger.error("Invalid model output shape")
            return False
            
        logger.info("Model loading verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Model loading verification failed: {str(e)}")
        return False

def verify_distributed() -> bool:
    """Verify distributed setup"""
    try:
        world = mx.distributed.init()
        logger.info(f"Distributed verification passed (Rank {world.rank}/{world.size})")
        return True
    except Exception as e:
        logger.error(f"Distributed verification failed: {str(e)}")
        return False

def run_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks"""
    try:
        # Run benchmarks
        result = subprocess.run(
            ["python", "scripts/benchmark.py"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error("Benchmark failed")
            return {}
            
        # Parse results
        import json
        return json.loads(result.stdout)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        return {}

def main():
    """Run all verification checks"""
    if is_ci_environment():
        # In CI, only run essential checks
        checks = {
            "Hardware Check": verify_hardware,
            "Network Check": verify_network,
            "Distributed Setup": verify_distributed
        }
    else:
        # Run all checks in non-CI environment
        checks = {
            "Hardware Check": verify_hardware,
            "Network Check": verify_network,
            "Data Pipeline": verify_data_pipeline,
            "Model Loading": verify_model_loading,
            "Distributed Setup": verify_distributed
        }
    
    all_passed = True
    for name, check in checks.items():
        logger.info(f"\nRunning {name}...")
        if not check():
            all_passed = False
            logger.error(f"{name} failed")
        else:
            logger.info(f"{name} passed")
    
    if all_passed:
        if not is_ci_environment():
            logger.info("\nRunning benchmarks...")
            results = run_benchmarks()
            if results:
                logger.info("All verification checks and benchmarks passed!")
        else:
            logger.info("All CI verification checks passed!")
        return 0
    
    logger.error("\nVerification failed")
    return 1

if __name__ == "__main__":
    exit(main()) 