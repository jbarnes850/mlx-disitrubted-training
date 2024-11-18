import mlx.core as mx
import pytest
import time
from src.models.unified_model import UnifiedModel, ModelConfig
from src.training.distributed_trainer import DistributedTrainer, TrainingConfig
from src.training.performance_utils import PerformanceOptimizer, PerformanceConfig
from src.monitoring.dashboard import PerformanceDashboard, DashboardConfig

@pytest.fixture
def performance_config():
    return PerformanceConfig(
        initial_batch_size=32,
        min_batch_size=8,
        max_batch_size=128,
        target_memory_usage=0.85,
        gradient_accumulation_steps=4
    )

@pytest.fixture
def model_config():
    return ModelConfig(
        num_layers=12,
        vocab_size=32000,
        dims=768,
        mlp_dims=3072,
        num_heads=12
    )

def test_training_throughput(model_config, performance_config):
    """Test training performance meets targets"""
    model = UnifiedModel(model_config)
    optimizer = mx.optimizers.Adam(learning_rate=1e-4)
    
    # Setup training
    batch = {
        "input_ids": mx.random.randint(0, 32000, (32, 512)),
        "labels": mx.random.randint(0, 32000, (32, 512))
    }
    
    # Measure throughput
    start_time = time.time()
    num_tokens = 0
    
    for _ in range(10):  # Run multiple steps
        loss, _ = model.train_step(batch)
        num_tokens += batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
        
    duration = time.time() - start_time
    tokens_per_second = num_tokens / duration
    
    # Assert performance targets
    assert tokens_per_second > 10000, f"Throughput too low: {tokens_per_second:.2f} tokens/sec"

def test_inference_latency(model_config):
    """Test inference latency requirements"""
    model = UnifiedModel(model_config)
    
    # Setup inference
    prompt = mx.random.randint(0, 32000, (1, 32))
    
    # Measure latency
    latencies = []
    
    for _ in range(100):  # Generate 100 tokens
        start_time = time.time()
        next_token = next(model.generate(prompt, max_length=1))
        mx.eval(next_token)  # Force computation
        latencies.append(time.time() - start_time)
        
    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 0.1, f"Latency too high: {avg_latency*1000:.2f}ms per token"

def test_memory_efficiency(model_config, performance_config):
    """Test memory usage efficiency"""
    if not mx.metal.is_available():
        pytest.skip("Metal backend not available")
        
    initial_memory = mx.metal.get_active_memory()
    model = UnifiedModel(model_config)
    
    # Measure memory overhead
    memory_used = mx.metal.get_active_memory() - initial_memory
    memory_overhead = memory_used / (model_config.dims * model_config.num_layers * 4)  # 4 bytes per parameter
    
    assert memory_overhead < 1.3, f"Memory overhead too high: {memory_overhead:.2f}x model size"

def test_scaling_efficiency():
    """Test distributed scaling efficiency"""
    if mx.distributed.get_world_size() < 2:
        pytest.skip("Multiple devices required for scaling test")
        
    # TODO: Implement distributed scaling tests
    pass

def test_gpu_utilization(model_config, performance_config):
    """Test GPU utilization meets targets"""
    if not mx.metal.is_available():
        pytest.skip("Metal backend not available")
        
    model = UnifiedModel(model_config)
    dashboard = PerformanceDashboard(DashboardConfig())
    
    # Run training and monitor GPU utilization
    batch = {
        "input_ids": mx.random.randint(0, 32000, (32, 512)),
        "labels": mx.random.randint(0, 32000, (32, 512))
    }
    
    for _ in range(100):
        loss, _ = model.train_step(batch)
        mx.eval(loss)
        
    gpu_stats = dashboard.metrics["gpu_utilization"].get_stats()
    assert gpu_stats["mean"] > 0.9, f"GPU utilization too low: {gpu_stats['mean']*100:.2f}%"
    
    dashboard.shutdown() 

def test_1b_model_throughput():
    """Test 1B model training throughput"""
    # Add specific throughput tests for 1B model
    # Target: >5000 tokens/sec on M2/M3

def test_memory_scaling():
    """Test memory scaling with model size"""
    # Add memory scaling tests 