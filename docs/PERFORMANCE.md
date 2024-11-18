# Performance Optimization Guide

## Hardware Recommendations

### Single Device
- M2 Ultra or M3 Max recommended
- 64GB+ unified memory
- Fast SSD for dataset streaming

### Distributed Setup
- Primary: M2 Ultra (192GB)
- Secondary: M3 Max (128GB)
- 10Gbps+ network connection

## Memory Optimization

1. Gradient Checkpointing
```python
trainer.enable_checkpointing(layers=[1,3,5,7])
```

2. Dynamic Batch Sizing
```python
trainer.enable_dynamic_batching(
    target_memory_usage=0.85
)
```

[Additional sections...] 