# MLX Distributed Training Quick Start Guide

## Prerequisites

- macOS Sonoma 14.0+
- Python 3.10+
- Multiple Apple Silicon devices (M2/M3 series)
- High-speed network connection between devices

## 1. Initial Setup

First, clone and set up the repository:

```bash
# Clone repository
git clone https://github.com/jbarnes850/mlx_distributed_training
cd mlx_distributed_training

# Run setup script
./scripts/setup_env.sh

# Verify MLX installation
python3 -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

## 2. System Verification

Run the verification script to check all components:

```bash
./scripts/verify_setup.sh
```

Expected output:

```plaintext
✓ Python 3.10+ detected
✓ MLX installation verified
✓ Metal backend available
✓ Dependencies installed
✓ Basic tests passed
```

## 3. Network Configuration

Configure SSH access between devices:

```bash
# On primary device
ssh-keygen -t ed25519 -f ~/.ssh/mlx_key
ssh-copy-id -i ~/.ssh/mlx_key.pub user@secondary-device.local

# Test connection
ssh -i ~/.ssh/mlx_key user@secondary-device.local "python3 -c 'import mlx'"
```

## 4. Performance Benchmark

Run benchmarks to verify system performance:

```bash
python scripts/benchmark.py
```

Expected metrics:

- Training throughput: >10,000 tokens/sec
- Inference latency: <100ms per token
- GPU utilization: >90%
- Memory efficiency: <30% overhead

## 5. Start Training

### Single Device

```bash
./scripts/quickstart.sh single
```

### Multiple Devices

```bash
# On primary device
./scripts/quickstart.sh distributed

# The script automatically handles secondary device setup
```

Monitor training progress:

```bash
# Watch metrics
tail -f logs/profile_results.json

# Monitor GPU usage
python3 -c "
import mlx.core as mx
print(f'GPU Memory: {mx.metal.get_active_memory() / (1024**3):.2f} GB')
"
```

## 6. Run Inference

After training completes:

```bash
# Start inference server
./scripts/run_inference.sh checkpoints/model_latest.safetensors 8000

# Test inference (in another terminal)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are", "max_tokens": 50}'
```

## Troubleshooting

### Common Issues

1. **Memory Errors**

```bash
# Clear Metal cache
python3 -c "import mlx.core as mx; mx.metal.clear_cache()"

# Reduce batch size in config
vim configs/distributed_config.json
# Set "batch_size": 16
```

2. **Network Issues**

```bash
# Test network speed
iperf3 -c secondary-device.local

# Expected: >10Gbps for optimal performance
```

3. **Performance Issues**

```bash
# Enable MPI optimization
mpirun --mca btl_tcp_links 4 python train.py
```

### Verification Commands

Check system status:

```bash
# Metal status
python3 -c "
import mlx.core as mx
print(f'Metal available: {mx.metal.is_available()}')
print(f'Memory limit: {mx.metal.get_memory_limit() / (1024**3):.2f} GB')
print(f'Active memory: {mx.metal.get_active_memory() / (1024**3):.2f} GB')
"

# Network status
python3 -c "
import mlx.core as mx
world = mx.distributed.init()
print(f'World size: {world.size}')
print(f'Rank: {world.rank}')
"
```

## Next Steps

1. Tune Performance:
   - Adjust batch size based on memory usage
   - Enable gradient checkpointing for large models
   - Optimize network configuration

2. Monitor Training:
   - Watch GPU utilization
   - Track memory usage
   - Monitor network bandwidth

3. Production Deployment:
   - Set up monitoring alerts
   - Configure automatic checkpointing
   - Implement error recovery
