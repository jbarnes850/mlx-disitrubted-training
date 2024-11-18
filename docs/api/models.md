# Models API Reference

## UnifiedModel

The `UnifiedModel` class implements a transformer-based architecture optimized for distributed training on Apple Silicon devices.

### Parameters

- `num_layers` (int): Number of transformer layers
- `dims` (int): Model dimension size
- `num_heads` (int): Number of attention heads
- `vocab_size` (int): Size of the vocabulary

### Methods

#### forward(x: mx.array) -> mx.array

Forward pass through the model.

#### distributed_forward(x: mx.array, world_size: int) -> mx.array

Forward pass optimized for distributed training.

#### save_checkpoint(path: str) -> None

Save model weights and optimizer state.

#### load_checkpoint(path: str) -> None

Load model weights and optimizer state.
