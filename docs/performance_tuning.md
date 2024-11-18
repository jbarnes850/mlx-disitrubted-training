# Performance Tuning Guide

## Memory Optimization

### 1. Gradient Checkpointing

```python
from src.training.performance_utils import PerformanceOptimizer

# Configure checkpointing
optimizer = PerformanceOptimizer(config)
model = optimizer.setup_gradient_checkpointing(
    model,
    checkpoint_layers=[1, 3, 5, 7]  # Checkpoint every other layer
)
```

### 2. Dynamic Batch Sizing

```python
# Monitor and adjust batch size
current_memory = mx.metal.get_active_memory() / (1024**3)  # GB
new_batch_size = optimizer.optimize_batch_size(current_memory)
```

### 3. Mixed Precision Training

```python
# Enable mixed precision in config
config.training.mixed_precision = True
```

## Compute Optimization

### 1. Stream Management

```python
# Create separate streams for compute and memory ops
compute_stream = mx.Stream(mx.gpu)
memory_stream = mx.Stream(mx.cpu)

with mx.stream(compute_stream):
    # Compute operations
    loss, grads = model.train_step(batch)

with mx.stream(memory_stream):
    # Memory operations
    next_batch = dataloader.prefetch()
```

### 2. Operation Fusion

MLX automatically fuses operations, but you can help by:

```python
# Group related operations
def fused_forward(self, x):
    # These operations will be fused
    x = self.linear1(x)
    x = mx.relu(x)
    return self.linear2(x)
```

### 3. Compute Scheduling

```python
# Profile-based scheduling
scheduler = ComputeScheduler(
    compute_intensity=0.8,  # Ratio of compute to memory ops
    pipeline_depth=2  # Number of batches in flight
)
```

## Network Optimization

### 1. Gradient Synchronization

```python
# Optimize all-reduce operations
def optimized_all_reduce(grads):
    # Combine small tensors
    flat_grads = mx.concatenate([g.flatten() for g in grads])
    # Single all-reduce
    reduced = mx.distributed.all_sum(flat_grads)
    # Reshape back
    return [g.reshape(orig.shape) for g, orig in zip(
        mx.split(reduced, [g.size for g in grads]),
        grads
    )]
```

### 2. Communication Overlap

```python
# Overlap computation and communication
with mx.stream(compute_stream):
    # Forward pass
    loss = model(batch)

with mx.stream(comm_stream):
    # Start gradient synchronization
    grads = optimizer.reduce_gradients()
```

## Monitoring and Tuning

### 1. Performance Metrics

```python
from src.monitoring.dashboard import PerformanceDashboard

dashboard = PerformanceDashboard()
dashboard.track_training_metrics(
    throughput=tokens_per_second,
    communication_time=sync_time,
    cache_hit_rate=cache_hits/total_access
)
```

### 2. Memory Profiling

```python
# Monitor memory usage
memory_stats = dashboard.get_memory_stats()
print(f"Active Memory: {memory_stats['active_gb']:.2f} GB")
print(f"Peak Memory: {memory_stats['peak_gb']:.2f} GB")
```

### 3. Automatic Tuning

```python
# Enable autotuning
trainer.enable_autotuning(
    target_memory_usage=0.85,
    target_throughput=10000,
    adaptation_rate=0.1
)
```

## Best Practices

1. **Memory Management**
   - Use gradient checkpointing for large models
   - Enable mixed precision training
   - Monitor and adjust batch sizes dynamically

2. **Compute Efficiency**
   - Use separate streams for compute and memory ops
   - Profile and optimize operation fusion
   - Implement efficient gradient synchronization

3. **Monitoring**
   - Track key performance metrics
   - Set up alerts for performance degradation
   - Regular profiling and optimization

4. **Testing**
   - Benchmark performance regularly
   - Test scaling efficiency
   - Validate memory usage patterns
