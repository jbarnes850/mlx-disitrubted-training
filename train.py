import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import argparse
import json
import logging
from pathlib import Path
from src.models.unified_model import UnifiedModel, ModelConfig
from src.training.distributed_trainer import DistributedTrainer, TrainingConfig
from src.training.data_utils import OptimizedDataManager, DataConfig
from src.utils.checkpoint_utils import CheckpointManager, CheckpointConfig
from src.utils.profile_utils import DistributedProfiler, ProfileConfig
from src.training.batch_optimizer import BatchOptimizer, BatchConfig
from src.training.gradient_accumulator import GradientAccumulator, AccumulationConfig
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="MLX Distributed Training")
    parser.add_argument("--config", default="configs/distributed_config.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hostfile", default="hostfile")
    return parser.parse_args()

async def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Initialize components
    model = UnifiedModel(ModelConfig(**config["model"]))
    optimizer = optim.AdamW(learning_rate=1e-4)
    
    # Initialize optimizers
    batch_optimizer = BatchOptimizer(BatchConfig(
        initial_batch_size=config["training"]["batch_size"]["primary"]
    ))
    
    gradient_accumulator = GradientAccumulator(AccumulationConfig(
        initial_steps=config["training"]["gradient_accumulation_steps"]
    ))
    
    trainer = DistributedTrainer(
        config=TrainingConfig(**config["training"]),
        model=model,
        optimizer=optimizer
    )
    
    # Setup optimized data pipeline
    data_manager = OptimizedDataManager(
        config=DataConfig(),
        world_size=trainer.world.size,
        rank=trainer.world.rank
    )
    dataset = data_manager.load_datasets()
    data_manager.start_prefetch()  # Start async prefetching
    
    # Setup checkpointing
    checkpoint_manager = CheckpointManager(
        config=CheckpointConfig(
            save_dir=args.checkpoint_dir,
            save_frequency=100
        ),
        model=model,
        optimizer=optimizer
    )
    
    # Resume if requested
    if args.resume:
        checkpoint_manager.load()
    
    # Setup profiler
    profiler = DistributedProfiler(ProfileConfig(**config["monitoring"]))
    
    # Training loop with optimizations
    try:
        step = 0
        while step < config["training"]["max_steps"]:
            # Get batch with prefetching
            batch = data_manager.get_batch()
            
            # Update batch size based on memory
            current_memory = mx.metal.get_active_memory() / (1024**3)
            batch_config = batch_optimizer.update(current_memory)
            
            if batch_config["batch_size"] != trainer.config.batch_size:
                trainer.config.batch_size = batch_config["batch_size"]
                logging.info(f"Adjusted batch size to {batch_config['batch_size']}")
            
            # Training step with gradient accumulation
            loss, grads = await trainer.train_step(batch)
            
            # Update gradient accumulation steps
            accum_config = gradient_accumulator.update(
                trainer.config.batch_size,
                current_memory,
                mx.norm(grads).item()
            )
            
            if accum_config["accumulation_steps"] != trainer.config.gradient_accumulation_steps:
                trainer.config.gradient_accumulation_steps = accum_config["accumulation_steps"]
                logging.info(f"Adjusted gradient accumulation to {accum_config['accumulation_steps']}")
            
            # Log metrics
            profiler.step_metrics(
                loss=loss,
                memory_gb=current_memory,
                batch_size=trainer.config.batch_size,
                grad_norm=mx.norm(grads).item(),
                effective_batch_size=accum_config["effective_batch_size"]
            )
            
            # Save checkpoint
            if step % config["training"]["checkpoint_frequency"] == 0:
                checkpoint_manager.save(
                    step=step,
                    metrics=profiler.summary()
                )
            
            step += 1
            
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise
    finally:
        # Clean shutdown
        trainer.shutdown()
        profiler.save()
        data_manager.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import asyncio
    asyncio.run(main()) 