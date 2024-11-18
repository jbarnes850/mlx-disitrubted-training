import mlx.core as mx
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
from datasets import load_dataset
from src.training.data_utils import DataManager, DataConfig

@dataclass
class InstructConfig:
    dataset_name: str = "tatsu-lab/alpaca"
    num_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_length: int = 2048
    warmup_steps: int = 100

class InstructionTuner:
    """Handles instruction tuning of pretrained models"""
    def __init__(
        self,
        model,
        config: InstructConfig,
        world_size: int = 1,
        rank: int = 0
    ):
        self.model = model
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.logger = logging.getLogger(__name__)
        
    def load_instruction_dataset(self):
        """Load instruction tuning dataset"""
        data_config = DataConfig(
            datasets=[self.config.dataset_name],
            streaming=True,
            max_length=self.config.max_length
        )
        
        data_manager = DataManager(
            config=data_config,
            world_size=self.world_size,
            rank=self.rank
        )
        
        return data_manager.load_datasets()
        
    def prepare_instruction_batch(
        self,
        examples: Dict[str, list]
    ) -> Dict[str, mx.array]:
        """Format instruction-response pairs"""
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        responses = examples["output"]
        
        # Format as instruction tuning
        texts = [
            f"Instruction: {inst}\nInput: {inp}\nResponse: {resp}"
            for inst, inp, resp in zip(instructions, inputs, responses)
        ]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="np"
        )
        
        return {
            "input_ids": mx.array(tokenized["input_ids"]),
            "attention_mask": mx.array(tokenized["attention_mask"]),
            "labels": mx.array(tokenized["input_ids"])
        }
        
    def train(self):
        """Run instruction tuning"""
        # Load dataset
        dataset = self.load_instruction_dataset()
        
        # Setup optimizer with learning rate schedule
        optimizer = mx.optimizers.Adam(
            learning_rate=self.config.learning_rate
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch}")
            
            for batch in dataset:
                # Forward pass
                loss = self.model.loss(**batch)
                
                # Backward pass
                gradients = mx.grad(self.model.parameters)
                optimizer.update(self.model, gradients)
                
                self.logger.info(f"Loss: {loss.item():.4f}") 