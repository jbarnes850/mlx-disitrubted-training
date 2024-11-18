"""Minimal example of distributed training"""
import mlx.core as mx
from src.training.distributed_trainer import DistributedTrainer
from src.models.unified_model import UnifiedModel

def main():
    # Initialize model
    model = UnifiedModel.from_pretrained("mlx-community/tinyllama-1b")
    
    # Setup distributed training
    trainer = DistributedTrainer(
        model=model,
        batch_size=16,
        gradient_accumulation=4
    )
    
    # Train
    trainer.train(
        dataset="HuggingFaceH4/ultrachat_200k",
        num_epochs=1
    )

if __name__ == "__main__":
    main() 