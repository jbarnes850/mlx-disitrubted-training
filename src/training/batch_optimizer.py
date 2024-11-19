import mlx.core as mx
import psutil
import iperf3
import logging
import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterator, Union, Tuple
import metal
from concurrent.futures import ThreadPoolExecutor

@dataclass
class BatchOptimizerConfig:
    initial_batch_size: int = 32
    min_batch_size: int = 8
    max_batch_size: int = 128
    memory_threshold: float = 0.85
    network_threshold_mbps: int = 1000
    cpu_threshold: float = 0.90
    adjustment_factor: float = 0.8
    monitoring_interval: int = 30
    warmup_steps: int = 100
    trend_window: int = 10
    prefetch_size: int = 2
    prefetch_workers: int = 2
    use_network_monitor: bool = True

class BatchOptimizer:
    """Optimizes batch processing with resource monitoring"""
    
    def __init__(
        self, 
        model: mx.nn.Module, 
        optimizer: mx.optimizers.Optimizer,
        network_communicator: Optional['DistributedCommunicator'] = None,
        config: Optional[BatchOptimizerConfig] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.network_communicator = network_communicator
        self.config = config or BatchOptimizerConfig()
        self.current_batch_size = self.config.initial_batch_size
        self.step = 0
        
        # Resource monitoring state
        self._stop_monitoring = False
        self._monitor_thread = None
        self._resource_metrics = {
            'memory_pressure': False,
            'network_pressure': False,
            'cpu_pressure': False,
            'bandwidth_mbps': 0,
        }
        self._metric_history = []
        
        # Prefetching setup
        self._prefetch_queue = queue.Queue(maxsize=self.config.prefetch_size)
        self._prefetch_pool = ThreadPoolExecutor(max_workers=self.config.prefetch_workers)
        self._current_iterator = None
        
        # Initialize monitoring
        self._start_resource_monitoring()
        
    def __del__(self):
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join()
        if self._prefetch_pool:
            self._prefetch_pool.shutdown()
            
    def _start_resource_monitoring(self):
        """Start background resource monitoring thread"""
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        
    def _monitor_resources(self):
        """Background thread for monitoring system resources"""
        while not self._stop_monitoring:
            try:
                # Check Metal memory
                self._resource_metrics['memory_pressure'] = self._check_memory_pressure()
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self._resource_metrics['cpu_pressure'] = cpu_percent > self.config.cpu_threshold * 100
                
                # Check network bandwidth using communicator if available
                if self.network_communicator and self.config.use_network_monitor:
                    bandwidth = self.network_communicator.get_current_bandwidth()
                    self._resource_metrics['bandwidth_mbps'] = bandwidth
                    self._resource_metrics['network_pressure'] = bandwidth < self.config.network_threshold_mbps
                else:
                    # Fallback to iperf3
                    bandwidth = self._measure_network_bandwidth()
                    self._resource_metrics['bandwidth_mbps'] = bandwidth
                    self._resource_metrics['network_pressure'] = bandwidth < self.config.network_threshold_mbps
                
                # Log resource status
                logging.info(f"Resource metrics: {self._resource_metrics}")
                
            except Exception as e:
                logging.error(f"Error monitoring resources: {str(e)}")
                
            time.sleep(self.config.monitoring_interval)
            
    def _check_memory_pressure(self) -> bool:
        """Check Metal GPU memory pressure"""
        try:
            device = metal.MTLCreateSystemDefaultDevice()
            if device:
                mem_used = 1 - (device.recommendedMaxWorkingSetSize - device.currentAllocatedSize) / device.recommendedMaxWorkingSetSize
                return mem_used > self.config.memory_threshold
        except Exception as e:
            logging.warning(f"Failed to check Metal memory: {str(e)}")
        return False
        
    def _measure_network_bandwidth(self) -> float:
        """Measure network bandwidth using iperf3"""
        try:
            client = iperf3.Client()
            client.duration = 1  # Quick test
            client.server_hostname = '127.0.0.1'  # Local test
            result = client.run()
            if result:
                return result.sent_Mbps
        except Exception as e:
            logging.warning(f"Failed to measure network bandwidth: {str(e)}")
        return float('inf')  # Assume good bandwidth if check fails
        
    def adjust_batch_size(self) -> bool:
        """Dynamically adjust batch size based on resource pressure"""
        self.step += 1
        
        # During warmup, be more conservative with adjustments
        if self.step < self.config.warmup_steps:
            adjustment_factor = self.config.adjustment_factor * 0.5
        else:
            adjustment_factor = self.config.adjustment_factor
        
        # Store current metrics for trend analysis
        self._metric_history.append(self._resource_metrics.copy())
        if len(self._metric_history) > self.config.trend_window:
            self._metric_history.pop(0)
        
        # Calculate pressure based on recent history
        recent_pressure = self._calculate_pressure_trend()
        
        should_reduce = recent_pressure > 0.5  # Threshold for pressure trend
        
        if should_reduce:
            new_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * adjustment_factor)
            )
            if new_size != self.current_batch_size:
                old_size = self.current_batch_size
                self.current_batch_size = new_size
                logging.info(
                    f"Step {self.step}: Adjusted batch size from {old_size} to {new_size} "
                    f"(pressure: {recent_pressure:.2f})"
                )
                return True
        return False
        
    def _calculate_pressure_trend(self) -> float:
        """Calculate weighted average of resource pressure over recent history"""
        if not self._metric_history:
            return 0.0
            
        pressure_scores = []
        for metrics in self._metric_history:
            score = sum([
                metrics['memory_pressure'],
                metrics['network_pressure'],
                metrics['cpu_pressure']
            ]) / 3.0
            pressure_scores.append(score)
            
        # More weight to recent measurements
        weights = [0.5 + (i / (2 * len(pressure_scores))) for i in range(len(pressure_scores))]
        return sum(s * w for s, w in zip(pressure_scores, weights)) / sum(weights)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        return {
            'batch_size': self.current_batch_size,
            **self._resource_metrics
        }
        
    def _convert_to_mlx_array(self, data: Any) -> Union[mx.array, Tuple[mx.array, ...]]:
        """Convert input data to MLX array format"""
        if isinstance(data, (tuple, list)):
            return tuple(
                mx.array(item) if not isinstance(item, mx.array) else item 
                for item in data
            )
        return mx.array(data) if not isinstance(data, mx.array) else data

    def _prefetch_worker(self, iterator: Iterator) -> Optional[Any]:
        """Worker function for prefetching batches"""
        try:
            batch = next(iterator)
            return self._convert_to_mlx_array(batch)
        except StopIteration:
            return None
        except Exception as e:
            logging.error(f"Error in prefetch worker: {str(e)}")
            return None

    def start_prefetching(self, iterator: Iterator) -> None:
        """Start prefetching from an iterator"""
        self._current_iterator = iterator
        
        # Initial prefetch
        for _ in range(self.config.prefetch_size):
            future = self._prefetch_pool.submit(self._prefetch_worker, iterator)
            try:
                batch = future.result(timeout=1.0)  # 1 second timeout
                if batch is not None:
                    self._prefetch_queue.put(batch)
            except Exception as e:
                logging.warning(f"Failed to prefetch initial batch: {str(e)}")

    def get_next_batch(self) -> Optional[Any]:
        """Get next batch from prefetch queue and trigger new prefetch"""
        if self._current_iterator is None:
            return None
            
        try:
            # Get next batch from queue
            batch = self._prefetch_queue.get_nowait()
            
            # Submit new prefetch task
            future = self._prefetch_pool.submit(self._prefetch_worker, self._current_iterator)
            future.add_done_callback(
                lambda f: self._handle_prefetch_result(f.result())
            )
            
            return batch
        except queue.Empty:
            logging.warning("Prefetch queue empty - possible prefetch delay")
            return self._prefetch_worker(self._current_iterator)
            
    def _handle_prefetch_result(self, result: Optional[Any]) -> None:
        """Handle prefetch worker results"""
        if result is not None:
            try:
                self._prefetch_queue.put_nowait(result)
            except queue.Full:
                logging.debug("Prefetch queue full - skipping")
        else:
            logging.debug("End of iterator reached in prefetch worker")

    def get_prefetch_stats(self) -> Dict[str, Any]:
        """Get prefetching statistics"""
        return {
            'queue_size': self._prefetch_queue.qsize(),
            'max_queue_size': self.config.prefetch_size,
            'active_workers': len([f for f in self._prefetch_pool._threads if f.is_alive()]),
            'max_workers': self.config.prefetch_workers
        }