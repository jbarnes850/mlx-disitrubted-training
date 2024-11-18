# MLX Distributed Training

A high-performance distributed training framework for MLX that enables efficient model training across multiple Apple Silicon devices.

![MLX Version](https://img.shields.io/badge/MLX-%3E%3D0.0.8-blue)
![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue)
![macOS](https://img.shields.io/badge/macOS-Sonoma%2014.0%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-Passing-success)

## Introduction to Distributed Training with MLX

MLX enables efficient distributed training across multiple Apple Silicon devices by leveraging the [MLX distributed communication framework](https://ml-explore.github.io/mlx/build/html/usage/distributed.html). This implementation focuses on data-parallel training, where each device holds a complete model copy but processes different data batches.

Distributed training is a crucial advancement in AI that democratizes access to large-scale model development. By allowing multiple devices to work together, we can:

- Train larger models that wouldn't fit on a single device
- Dramatically reduce training time through parallel processing
- Enable collaborative research across distributed teams
- Lower the barrier to entry for AI research and development

This project aims to make distributed training accessible to researchers, educators, and developers using consumer Apple hardware. While cloud providers offer powerful distributed training capabilities, having these tools available on local hardware opens new possibilities for:

- Academic research without expensive cloud costs
- Privacy-preserving AI development on sensitive data
- Community-driven model development

Our implementation leverages the power of Apple Silicon to deliver efficient distributed training that was previously only possible with specialized hardware. We believe this can help accelerate AI innovation while keeping it open, accessible, and focused on public benefit.

### Why MLX for Distributed Training?

1. **Native Apple Silicon Support**:
   - MLX is built specifically for Apple's Metal architecture
   - Direct access to Metal Performance Shaders (MPS)
   - Optimized for the Neural Engine and unified memory

2. **Efficient Communication**:
   - Uses MPI for high-performance inter-device communication
   - Supports gradient synchronization through all-reduce operations
   - Optimized for Apple's networking stack

3. **Memory Efficiency**:
   - Unified memory architecture eliminates CPU-GPU transfers
   - Dynamic batch sizing based on available memory
   - Gradient checkpointing for large models
   - Real-time performance monitoring
   - HuggingFace datasets integration
   - Inference server deployment

### Key Features

- Distributed training optimized for Apple Silicon
- Streaming dataset support with HuggingFace integration
- Real-time performance monitoring dashboard
- Automatic memory optimization
- Production-ready inference server

## Quick Start

1. **Environment Setup**:

```bash
# Clone repository
git clone https://github.com/jbarnes850/mlx_distributed
cd mlx_distributed

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

2. **Verify Setup**:

```bash
# Run verification script
./scripts/verify_setup.sh

# Test network connectivity
python scripts/test_network.py
```

3. **Start Training**:

```bash
# On primary device (e.g., Mac Studio M2 Ultra)
./scripts/start_training.sh --role primary

# On secondary device (e.g., MacBook M3 Max)
./scripts/start_training.sh --role secondary
```

4. **Monitor Progress**:

```bash
# Open dashboard
open http://localhost:8050

# Watch logs
tail -f logs/training.log
```

## Network Requirements

- High-speed connection (10Gbps+ recommended)
- Low latency (<1ms between devices)
- SSH access configured between devices

## Documentation

- [MLX Distributed Documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [Setup Guide](docs/setup_guide.md)
- [Performance Tuning](docs/performance_tuning.md)
- [API Reference](docs/api.md)
- [Best Practices](docs/best_practices.md)

## Implementation Details

Our distributed training implementation follows MLX's recommended practices:

1. **Data Parallelism**:
   - Each device maintains a complete model copy
   - Data is sharded across devices
   - Gradients synchronized using `mx.distributed.all_sum`
   - Weights broadcast periodically for consistency

2. **Memory Management**:
   - Dynamic batch sizing based on device capabilities
   - Gradient accumulation for effective larger batches
   - Activation checkpointing for memory efficiency
   - Streaming data loading to manage memory usage

3. **Performance Optimization**:
   - Mixed precision training
   - Gradient compression for efficient communication
   - Multiple TCP links for improved bandwidth
   - Sliding window attention for long sequences

4. **Monitoring and Recovery**:
   - Real-time performance dashboard
   - Automatic error recovery
   - Checkpoint management
   - Network health monitoring

For more details on MLX's distributed capabilities, see:
- [Distributed Communication Guide](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX Examples Repository](https://github.com/ml-explore/mlx-examples)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- MLX Team at Apple
- MLX Community Contributors
