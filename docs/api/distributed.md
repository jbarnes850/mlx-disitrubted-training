# Distributed Training API Reference

## DistributedTrainer

The `DistributedTrainer` class manages distributed training across multiple Apple Silicon devices.

### Parameters

- `config` (dict): Configuration dictionary containing model and training parameters
  - `model`: Model configuration
  - `training`: Training hyperparameters
  - `distributed`: Distributed training settings

### Methods

#### prepare_dataset(dataset: Dataset) -> None

Prepare and shard the dataset across available devices.

#### train(model: UnifiedModel, dataset: Dataset, dashboard: Optional[PerformanceDashboard] = None) -> None

Execute distributed training.

#### synchronize_gradients() -> None

Synchronize gradients across all devices.

#### broadcast_weights() -> None

Broadcast updated weights to all devices.

## NetworkManager

The `NetworkManager` class handles inter-device communication.

### Methods

#### setup_connections() -> None

Initialize network connections between devices.

#### verify_bandwidth() -> float

Test and verify network bandwidth between devices.

#### optimize_communication() -> None

Optimize communication patterns based on network topology.
