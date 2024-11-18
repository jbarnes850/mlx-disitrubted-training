# Performance Tuning Guide

## Technical Specifications

### Current Hardware Configuration
- **Primary Node**: Mac Studio M2 Ultra
  - 24-core GPU
  - 160GB unified memory limit
  - Batch size: 32
- **Secondary Node**: MacBook M3 Max
  - 16-core GPU
  - 96GB unified memory limit
  - Batch size: 16

### Training Process (In Progress)
- **Estimated Duration**:
  - Best case: 3-4 weeks
  - Realistic case: 4-6 weeks
  - Assumes 24/7 operation
  - For comparison: Similar models (1.3B parameters) take ~6 days on 8x A100 GPUs

### Performance Optimizations
- **Memory Management**:
  - Dynamic batch size adjustment
  - Gradient accumulation (16 steps)
  - Memory defragmentation every 100 batches
  - Gradient checkpointing on alternate layers
  
- **Distributed Training**:
  - 4 TCP links for network communication
  - Asynchronous data prefetching
  - Optimized gradient synchronization
  - Weight broadcasting optimization

- **Computation**:
  - Mixed precision training
  - Separate compute/memory streams
  - Flash Attention implementation
  - Grouped Query Attention (GQA)
  - Optimized memory layout

### Training Characteristics
- Training is ~3x slower than inference due to:
  - Gradient computation and synchronization
  - Weight updates and broadcasting
  - Memory management overhead
  - Network communication latency

### Memory Considerations
- Primary bottleneck is memory bandwidth between devices
- Dynamic batch size adjustment based on memory usage
- Streaming data loading to manage memory pressure
- Gradient accumulation to handle memory constraints

### Monitoring and Stability
- Continuous performance monitoring
- Automatic batch size optimization
- Training stability checks
- Early stopping based on loss convergence
- Adaptive learning rate scheduling

These specifications represent our current optimized configuration for training a 1B parameter model on consumer Apple Silicon hardware. The training time estimates are based on empirical measurements and system monitoring data from our training implementation.


## Implementation Details

### Memory Management Implementation

#### 1. Dynamic Batch Size Adjustment
```python
def optimize_batch_size(self, current_memory_usage: float) -> int:
    """Dynamically adjust batch size based on memory usage"""
    self.memory_history.append(current_memory_usage)
    
    # Use moving average for stability
    avg_memory = np.mean(self.memory_history[-10:])
    
    # Adjust batch size
    if avg_memory > self.config.target_memory_usage:
        self.current_batch_size = max(
            self.config.min_batch_size,
            int(self.current_batch_size * 0.8)
        )
    elif avg_memory < self.config.target_memory_usage * 0.8:
        self.current_batch_size = min(
            self.config.max_batch_size,
            int(self.current_batch_size * 1.2)
        )
```

#### 2. Gradient Accumulation
```python
def update(self, current_batch_size: int, memory_usage: float) -> Dict[str, Any]:
    """Update accumulation steps"""
    effective_batch_size = current_batch_size * self.current_steps
    
    # Adjust based on memory and target batch size
    if memory_usage > self.config.memory_threshold:
        new_steps = min(self.config.max_steps, self.current_steps + 1)
    elif effective_batch_size < self.config.target_batch_size:
        new_steps = min(self.config.max_steps, self.current_steps + 1)
    else:
        new_steps = self.current_steps
```

#### 3. Memory Defragmentation
```python
# In DistributedTrainer.train_epoch
if batch_idx % 100 == 0:
    self.memory_manager.defragment()
```

### Distributed Training Implementation

#### 1. Network Communication
```python
# Configure MPI parameters for optimal communication
if self.size > 1:
    MPI.Info.Set("btl_tcp_links", "4")

# Gradient synchronization
if self.world.size > 1:
    grads = await self.network.sync_gradients(grads)

# Weight synchronization
if self.config.sync_weights_every > 0 and self.step % self.config.sync_weights_every == 0:
    self.model.parameters = await self.network.broadcast_weights(self.model.parameters)
```

#### 2. Data Prefetching
```python
def start_prefetch(self):
    """Start background prefetching"""
    def prefetch_worker():
        try:
            for batch in self.dataset:
                processed = self.preprocess_function(batch)
                self.prefetch_queue.put(processed)
        except Exception as e:
            self.logger.error(f"Prefetch error: {str(e)}")
```

### Performance Monitoring Implementation

#### 1. Training Metrics
```python
def check_training_health(self, loss: float, step: int) -> Dict[str, Any]:
    """Monitor training stability"""
    metrics = {
        "loss_std": np.std(self.loss_history[-100:]),
        "loss_trend": self._calculate_trend(),
        "gradient_norm": self._compute_gradient_norm(),
        "learning_rate": self._get_current_lr(step)
    }
    
    if metrics["loss_std"] > 5.0:
        self.logger.warning("High loss variance detected")
```

#### 2. Memory Tracking
```python
def _calculate_trend(self) -> float:
    """Calculate memory usage trend"""
    if len(self.memory_history) < 10:
        return 0.0
        
    recent = self.memory_history[-10:]
    return np.mean(recent) / mx.metal.get_memory_limit()
```

### Optimization Guidelines

1. **Memory Management**
   - Start with smaller batch sizes and gradually increase
   - Monitor memory usage trends over time
   - Use gradient accumulation when memory constrained
   - Enable gradient checkpointing on alternate layers

2. **Network Communication**
   - Use multiple TCP links for better bandwidth
   - Implement asynchronous data prefetching
   - Optimize gradient synchronization frequency
   - Monitor network latency and adjust accordingly

3. **Training Stability**
   - Track loss variance and gradient norms
   - Implement early stopping with patience
   - Use adaptive learning rates
   - Monitor training metrics continuously

4. **Hardware Utilization**
   - Balance workload across devices
   - Monitor GPU utilization
   - Optimize memory transfer patterns
   - Use separate compute and memory streams

These implementations reflect our current optimized configuration for training large models on Apple Silicon hardware. Adjust parameters based on your specific hardware setup and training requirements.

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
