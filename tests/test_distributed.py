import pytest
import mlx.core as mx
import os
from src.utils.network_utils import DistributedCommunicator, NetworkConfig
from src.training.distributed_trainer import DistributedTrainer
from src.utils.memory_utils import MemoryManager
from src.models.unified_model import UnifiedModel, ModelConfig
from src.data.data_manager import DataManager, DataConfig

def test_distributed_setup():
    """Test basic distributed functionality"""
    world = mx.distributed.init()
    assert world.size >= 1
    assert 0 <= world.rank < world.size

def test_memory_management():
    """Test memory management"""
    manager = MemoryManager(max_memory_gb=32)
    stats = manager.get_memory_stats()
    assert "metal_active_gb" in stats
    assert stats["metal_active_gb"] >= 0

def test_model_initialization():
    """Test model initialization"""
    config = ModelConfig(
        num_layers=2,
        vocab_size=1000,
        dims=128,
        mlp_dims=512,
        num_heads=4
    )
    model = UnifiedModel(config)
    assert isinstance(model, UnifiedModel)

@pytest.mark.skipif(mx.distributed.get_world().size < 2, 
                   reason="Requires multiple devices")
def test_gradient_sync():
    """Test gradient synchronization"""
    comm = DistributedCommunicator(NetworkConfig())
    x = mx.array([1.0, 2.0, 3.0])
    result = mx.distributed.all_sum(x)
    assert result.shape == x.shape 

def test_full_pipeline():
    """Test complete training pipeline"""
    # Add test for data loading, model init, and training step
    config = {
        "training": {
            "batch_size": {
                "primary": 4,    # Small batch for testing
                "secondary": 2
            },
            "gradient_accumulation_steps": 2,
            "max_steps": 5
        }
    }
    
    # Test data pipeline
    data_manager = DataManager(DataConfig())
    dataset = data_manager.load_datasets()
    assert dataset is not None
    
    # Test model initialization
    model = UnifiedModel(ModelConfig())
    assert model is not None
    
    # Test training step
    trainer = DistributedTrainer(config, model)
    loss = trainer.train_step(next(iter(dataset)))
    assert loss is not None