import mlx.core as mx
import pytest
from src.models.unified_model import UnifiedModel, ModelConfig
from src.training.distributed_trainer import DistributedTrainer, TrainingConfig
from src.utils.memory_utils import MemoryManager
from src.training.scheduler import TrainingScheduler, SchedulerConfig
from src.data.validation import DataValidator, ValidationConfig
from src.evaluation.continuous_eval import ContinuousEvaluator, EvalConfig

@pytest.fixture
def small_model_config():
    return ModelConfig(
        num_layers=2,
        vocab_size=1000,
        dims=128,
        mlp_dims=512,
        num_heads=4
    )

@pytest.fixture
def training_config():
    return TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        max_memory_gb=4,
        prefetch_batches=1,
        mixed_precision=True,
        device_map={"primary": "gpu"}
    )

def test_model_forward(small_model_config):
    model = UnifiedModel(small_model_config)
    batch = mx.random.randint(0, 1000, (2, 16))
    
    output = model(batch)
    assert output.shape == (2, 16, 1000)

def test_distributed_trainer(small_model_config, training_config):
    model = UnifiedModel(small_model_config)
    optimizer = mx.optimizers.Adam(learning_rate=1e-4)
    trainer = DistributedTrainer(training_config, model, optimizer)
    
    batch = {
        "input_ids": mx.random.randint(0, 1000, (2, 16)),
        "labels": mx.random.randint(0, 1000, (2, 16))
    }
    
    loss, grads = trainer.train_step(batch)
    assert isinstance(loss, float)
    assert all(isinstance(g, mx.array) for g in grads.values())

def test_memory_management():
    manager = MemoryManager(max_memory_gb=4)
    stats = manager.get_memory_stats()
    
    assert "system_used_gb" in stats
    if mx.metal.is_available():
        assert "metal_active_gb" in stats 

def test_early_stopping():
    """Test early stopping functionality"""
    # Setup
    config = SchedulerConfig(
        patience=3,
        min_delta=1e-4,
        min_epochs=2
    )
    scheduler = TrainingScheduler(config)
    
    # Test case 1: Should not stop before min_epochs
    for epoch in range(config.min_epochs):
        assert not scheduler.should_stop(1.0, epoch)
    
    # Test case 2: Should stop after patience steps without improvement
    loss_values = [1.0, 0.9, 0.89, 0.889, 0.8889]  # Diminishing improvements
    for epoch, loss in enumerate(loss_values, start=config.min_epochs):
        if epoch >= config.min_epochs + config.patience:
            assert scheduler.should_stop(loss, epoch)
        else:
            assert not scheduler.should_stop(loss, epoch)
    
    # Test case 3: Should not stop if loss improves significantly
    scheduler = TrainingScheduler(config)
    loss_values = [1.0, 0.7, 0.5, 0.3, 0.1]  # Significant improvements
    for epoch, loss in enumerate(loss_values, start=config.min_epochs):
        assert not scheduler.should_stop(loss, epoch)

def test_data_validation():
    """Test data validation functionality"""
    # Setup
    config = ValidationConfig(
        perplexity_threshold=100.0,
        repetition_threshold=0.3,
        toxicity_threshold=0.1,
        min_length=10,
        max_length=100
    )
    validator = DataValidator(config)
    
    # Test case 1: Valid text
    valid_text = "This is a high-quality text sample with good diversity and reasonable length."
    result = validator.validate_sample(valid_text)
    assert result["valid"]
    
    # Test case 2: Too short
    short_text = "Too short"
    result = validator.validate_sample(short_text)
    assert not result["valid"]
    assert result["reason"] == "too_short"
    
    # Test case 3: High repetition
    repetitive_text = "This is a test. This is a test. This is a test. " * 10
    result = validator.validate_sample(repetitive_text)
    assert not result["valid"]
    assert result["reason"] == "high_repetition"
    
    # Test case 4: Batch validation
    texts = [
        "Valid text one with good quality.",
        "Too short",
        "Valid text two with good diversity and length."
    ]
    results = validator.validate_batch(texts, return_scores=True)
    assert len(results) == 3
    assert results[0]["valid"]
    assert not results[1]["valid"]
    assert results[2]["valid"]

@pytest.mark.asyncio
async def test_continuous_eval():
    """Test continuous evaluation functionality"""
    # Setup model and datasets
    model_config = ModelConfig(
        num_layers=2,
        vocab_size=1000,
        dims=128,
        mlp_dims=512,
        num_heads=4
    )
    model = UnifiedModel(model_config)
    
    # Create dummy evaluation datasets
    eval_datasets = {
        "validation": {
            "input_ids": mx.random.randint(0, 1000, (100, 32)),
            "labels": mx.random.randint(0, 1000, (100, 32))
        },
        "test": {
            "input_ids": mx.random.randint(0, 1000, (50, 32)),
            "labels": mx.random.randint(0, 1000, (50, 32))
        }
    }
    
    # Initialize evaluator
    config = EvalConfig(
        eval_frequency=10,
        min_eval_samples=50,
        save_results=True
    )
    evaluator = ContinuousEvaluator(model, eval_datasets, config)
    
    # Test case 1: Basic evaluation
    step = 10  # Evaluation step
    current_loss = 1.0
    results = await evaluator.evaluate_step(step, current_loss)
    
    assert "step" in results
    assert "training_loss" in results
    assert "validation" in results
    assert "test" in results
    
    # Test case 2: Metrics tracking
    assert len(evaluator.metrics["loss"]) > 0
    assert "perplexity" in evaluator.metrics
    
    # Test case 3: Skip evaluation if not at frequency
    step = 5  # Not evaluation step
    results = await evaluator.evaluate_step(step, current_loss)
    assert results == {}
    
    # Test case 4: Summary statistics
    summary = evaluator.get_summary()
    assert "loss" in summary
    assert "current" in summary["loss"]
    assert "mean" in summary["loss"]

@pytest.mark.asyncio
async def test_full_training_loop():
    """Test integration of early stopping, validation, and evaluation"""
    # Setup components
    model_config = ModelConfig(
        num_layers=2,
        vocab_size=1000,
        dims=128,
        mlp_dims=512,
        num_heads=4
    )
    model = UnifiedModel(model_config)
    
    scheduler = TrainingScheduler(SchedulerConfig())
    validator = DataValidator(ValidationConfig())
    
    eval_datasets = {
        "validation": {
            "input_ids": mx.random.randint(0, 1000, (100, 32)),
            "labels": mx.random.randint(0, 1000, (100, 32))
        }
    }
    evaluator = ContinuousEvaluator(model, eval_datasets, EvalConfig())
    
    # Simulate training loop
    try:
        for step in range(100):
            # Generate dummy batch
            batch = {
                "input_ids": mx.random.randint(0, 1000, (32, 32)),
                "labels": mx.random.randint(0, 1000, (32, 32))
            }
            
            # Validate batch
            if not validator.validate_sample(batch):
                continue
                
            # Forward pass
            loss = model(batch["input_ids"])
            current_lr = scheduler.get_lr()
            
            # Evaluation
            if step % 10 == 0:
                eval_results = await evaluator.evaluate_step(step, loss.item())
                
                # Check early stopping
                if scheduler.should_stop(eval_results["validation"]["loss"], step // 10):
                    break
                    
            scheduler.step_scheduler()
            
    except Exception as e:
        pytest.fail(f"Training loop failed: {str(e)}")