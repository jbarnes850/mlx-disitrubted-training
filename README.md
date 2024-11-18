# MLX Distributed Training

A high-performance distributed training framework for MLX that enables efficient model training across multiple Apple Silicon devices.

![MLX Version](https://img.shields.io/badge/MLX-%3E%3D0.20.0-blue)
![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue)
![macOS](https://img.shields.io/badge/macOS-Sonoma%2014.0%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-Passing-success)

## Introduction to Distributed Training with MLX

This project explores the potential of distributed training on Apple Silicon, specifically targeting the development of large language models. By leveraging [MLX's distributed communication framework](https://ml-explore.github.io/mlx/build/html/usage/distributed.html), we're pushing the boundaries of what's possible with consumer hardware.

The primary goal is ambitious yet practical: train a 1B parameter model using a network of Mac devices that outperforms state-of-the-art results (such as llama 3.2). Traditional approaches to training models of this scale typically require expensive cloud resources or specialized hardware. This implementation demonstrates that with efficient distributed algorithms and Apple's unified architecture, we can achieve comparable results using devices many developers already own.

This framework is designed for ML engineers and researchers interested in:
- Implementing and optimizing distributed training systems
- Exploring novel approaches to model parallelism and gradient synchronization
- Understanding the practical aspects of training large language models
- Contributing to the advancement of decentralized ML infrastructure

### Why MLX for Distributed Training?

After extensive experimentation with various frameworks, MLX emerged as the optimal choice for distributed training on Apple Silicon for several compelling reasons:

1. **Native Silicon Architecture Integration**
   - Direct compilation to Metal, maximizing M-series chip performance
   - Seamless utilization of the Neural Engine and unified memory
   - Optimized memory bandwidth and computational throughput
   - Performance that consistently outpaces traditional frameworks on Apple hardware

2. **Advanced Communication Architecture**
   - High-efficiency MPI-based inter-device communication
   - Zero-copy gradient synchronization through optimized all-reduce operations
   - Network stack specifically tuned for Apple's hardware ecosystem
   - Minimal overhead in multi-device coordination

3. **Sophisticated Memory Management**
   - Leverages unified memory architecture for optimal resource utilization
   - Implements dynamic batch size adjustment based on device capabilities
   - Advanced gradient checkpointing for memory-constrained scenarios
   - Comprehensive monitoring and profiling capabilities

Our research and development focus on several key areas:
- Scaling transformer architectures to 1B-3B parameters across distributed Mac systems
- Implementing novel data streaming and caching strategies
- Exploring hybrid parallelism techniques (data, model, and pipeline)
- Developing robust distributed training protocols

This project serves as both a practical implementation and a research platform, enabling the ML community to explore distributed training techniques without the traditional barriers to entry. We welcome contributions from engineers and researchers interested in advancing the field of distributed ML training.

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