import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
import json
import logging
from src.models.unified_model import UnifiedModel, ModelConfig
from src.training.data_utils import FineWebDataManager, FineWebConfig, DataIterator
from src.training.distributed_trainer import DistributedTrainer, TrainingConfig
from src.utils.profile_utils import DistributedProfiler, ProfileConfig
from src.utils.checkpoint_utils import CheckpointManager, CheckpointConfig

logging.basicConfig(level=logging.INFO)

def main():
    # Load configurations
    with open("configs/distributed_config.json") as f:
        config = json.load(f)
    
    # Initialize model
    model_config = ModelConfig(
        num_layers=32,
        vocab_size=32000,
        dims=4096,
        mlp_dims=11008,
        num_heads=32
    )
    model = UnifiedModel(model_config)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        grad_clip=config["training"]["grad_clip"]
    )
    
    # Initialize distributed training
    trainer = DistributedTrainer(
        config=TrainingConfig(**config["training"]),
        model=model,
        optimizer=optimizer
    )
    
    # Setup data loading
    data_manager = FineWebDataManager(
        config=FineWebConfig(**config["data"]),
        world_size=trainer.world.size,
        rank=trainer.world.rank
    )
    dataset = data_manager.load_dataset()
    dataloader = DataIterator(
        dataset=dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    # Setup profiling and checkpointing
    profiler = DistributedProfiler(ProfileConfig(**config["monitoring"]))
    checkpoint_manager = CheckpointManager(
        config=CheckpointConfig(**config["checkpointing"]),
        model=model,
        optimizer=optimizer
    )
    
    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        profiler.start_epoch()
        
        for batch in dataloader:
            loss, grads = trainer.train_step(batch)
            profiler.step_metrics(
                loss=loss,
                grad_norm=mx.norm(grads)
            )
            
            # Save checkpoint periodically
            if profiler.step % config["checkpointing"]["save_frequency"] == 0:
                checkpoint_manager.save(
                    step=profiler.step,
                    metrics=profiler.summary()
                )
        
        profiler.end_epoch()
        logging.info(f"Epoch {epoch}: {profiler.summary()}")

if __name__ == "__main__":
    main() 