# MLX Distributed Training Setup Guide

## System Requirements

### Hardware Requirements

- Primary Device: Mac Studio M2 Ultra (76-core GPU, 192GB memory)
- Secondary Device: MacBook M3 Max (40-core GPU, 128GB memory)
- High-speed network connection between devices (10Gbps recommended)

### Software Requirements

- macOS Sonoma 14.0 or later
- Python 3.10 or later
- Xcode Command Line Tools

## Installation Steps

1. **System Preparation**

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify Metal support
python3 -c "import metal; print(metal.device_count())"
```

2. **Environment Setup**

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

3. **Network Configuration**

```bash
# Configure SSH keys between devices
ssh-keygen -t ed25519
ssh-copy-id user@secondary-device

# Test connection
ssh user@secondary-device "python3 -c 'import mlx; print(mlx.__version__)'"
```

4. **Verify Installation**

```bash
# Run verification script
python scripts/verify_setup.py

# Run basic tests
pytest tests/
```

## Configuration

1. **Distributed Setup**

Edit `configs/distributed_config.json`:

```json
{
    "training": {
        "device_map": {
            "primary": "gpu",
            "secondary": "gpu"
        },
        "max_memory_gb": {
            "primary": 160,
            "secondary": 96
        }
    }
}
```

2. **Memory Configuration**

Adjust based on your hardware:

```json
{
    "memory": {
        "parameter_memory_gb": 4,
        "gradient_memory_gb": 4,
        "optimizer_memory_gb": 8,
        "workspace_memory_gb": 2
    }
}
```

## Troubleshooting

### Common Issues

1. **Metal Device Not Found**

```bash
# Check Metal availability
python3 -c "import mlx.core as mx; print(mx.metal.is_available())"
```

2. **Memory Errors**

- Reduce batch size
- Enable gradient checkpointing
- Adjust memory limits

3. **Network Issues**

```bash
# Test network speed
iperf3 -c secondary-device
```

### Performance Verification

Run benchmark suite:

```bash
python scripts/benchmark.py
```

Expected results:

- GPU Utilization: >90%
- Memory Efficiency: <30% overhead
- Network Latency: <1ms between devices 