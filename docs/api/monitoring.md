# Monitoring API Reference

## PerformanceDashboard

The `PerformanceDashboard` class provides real-time monitoring and visualization of distributed training.

### Methods

#### display_metrics() -> None

Display current training metrics and performance indicators.

#### log_performance(metrics: Dict[str, float]) -> None

Log performance metrics for visualization.

#### export_report(path: str) -> None

Export performance report to specified path.

## MetricsCollector

The `MetricsCollector` class handles collection and aggregation of training metrics.

### Methods

#### collect_device_metrics() -> Dict[str, float]

Collect metrics from individual devices.

#### aggregate_metrics(device_metrics: List[Dict[str, float]]) -> Dict[str, float]

Aggregate metrics across all devices.

#### calculate_efficiency() -> float

Calculate distributed training efficiency.

## SystemMonitor

The `SystemMonitor` class monitors system resource utilization.

### Methods

#### monitor_gpu_usage() -> Dict[str, float]

Monitor GPU utilization and memory usage.

#### monitor_network_usage() -> Dict[str, float]

Monitor network bandwidth utilization.

#### monitor_memory_usage() -> Dict[str, float]

Monitor system memory usage.
