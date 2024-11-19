import mlx.core as mx
import mlx.optimizers as optim
import argparse
import logging
from src.models.personal_ai import PersonalAIModel, PersonalAIConfig
from src.training.optimized_trainer import OptimizedTrainer, OptimizedTrainingConfig
from src.data.personal_dataset import PersonalDataset
from src.data.personal_context import PersonalDataProcessor, PrivacyConfig
import asyncio

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed training for Personal AI Model")
    
    # Model configuration
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--model-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=22)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-gqa-groups", type=int, default=4)
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    
    # Distributed training
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--max-memory-gb", type=float, default=80)
    
    # Data and checkpointing
    parser.add_argument("--base-knowledge-dir", type=str, required=True,
                      help="Directory containing base knowledge datasets")
    parser.add_argument("--personal-data-dir", type=str, required=True,
                      help="Directory for secure personal data storage")
    parser.add_argument("--privacy-level", type=str, default="high",
                      choices=["low", "medium", "high"],
                      help="Privacy protection level")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-file", type=str, default="training.log")
    
    return parser.parse_args()

def setup_logging(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

def create_model_config(args) -> PersonalAIConfig:
    return PersonalAIConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_length,
        hidden_size=args.model_dim,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_gqa_groups * 4,
        head_dim=args.head_dim,
        gqa_groups=args.num_gqa_groups
    )

def create_training_config(args, model_config) -> OptimizedTrainingConfig:
    return OptimizedTrainingConfig(
        model_config=model_config,
        initial_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        world_size=args.world_size,
        primary_device=args.device,
        max_memory_gb=args.max_memory_gb
    )

def create_optimizer(model: PersonalAIModel, config: OptimizedTrainingConfig):
    return optim.AdamW(
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )

async def main():
    args = parse_args()
    setup_logging(args)
    
    # Create configurations
    model_config = create_model_config(args)
    training_config = create_training_config(args, model_config)
    
    # Initialize model
    model = PersonalAIModel(model_config)
    optimizer = create_optimizer(model, training_config)
    
    # Setup personal data processing
    privacy_config = PrivacyConfig(
        privacy_level=args.privacy_level,
        local_storage_only=True,
        encryption_enabled=True
    )
    
    data_processor = PersonalDataProcessor(
        privacy_config=privacy_config,
        data_dir=args.personal_data_dir
    )
    
    # Create combined dataset
    dataset = PersonalDataset(
        data_processor=data_processor,
        base_knowledge_path=args.base_knowledge_dir,
        batch_size=args.batch_size,
        sequence_length=args.max_length,
        world_size=args.world_size,
        rank=0  # Set based on MPI rank in distributed setting
    )
    
    # Initialize trainer
    trainer = OptimizedTrainer(
        config=training_config,
        model=model,
        dataset=dataset,
        optimizer=optimizer
    )
    
    # Start training
    logging.info("Starting distributed training with personal data integration")
    await trainer.train()

if __name__ == "__main__":
    asyncio.run(main())
