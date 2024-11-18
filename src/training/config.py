@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_memory_gb: float = 160
    prefetch_batches: int = 2
    mixed_precision: bool = True
    use_flash_attention: bool = True
    use_gqa: bool = True
    sliding_window: int = 512
    rope_scaling: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10 