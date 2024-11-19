import mlx.core as mx
import socket
import logging
import asyncio
import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    primary_host: str = "localhost"
    primary_port: int = 29500
    timeout: float = 30.0
    retry_attempts: int = 3
    buffer_size: int = 8192
    use_quantization: bool = True
    quantization_bits: int = 8
    tcp_links: int = 4
    metal_buffer_pool_size: int = 1024  # MB
    metal_queue_depth: int = 3

class DistributedCommunicator:
    """Handles network communication for distributed training"""
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.world = mx.distributed.get_world()
        self._setup_logging()
        self._configure_metal()
        self._last_bandwidth = float('inf')
        self._bandwidth_history = []
        
        # Configure TCP links for better bandwidth
        if self.config.tcp_links > 1:
            os.environ["OMPI_MCA_btl_tcp_links"] = str(self.config.tcp_links)
            
    def _configure_metal(self):
        """Configure Metal-specific optimizations"""
        # Enable Metal buffer pooling
        os.environ["MLX_BUFFER_POOL_SIZE"] = str(self.config.metal_buffer_pool_size)
        
        # Set Metal command queue depth
        os.environ["MLX_METAL_QUEUE_DEPTH"] = str(self.config.metal_queue_depth)
        
        # Enable Metal graph optimization
        os.environ["MLX_METAL_GRAPH_OPTIMIZE"] = "1"
        
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        
    async def sync_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Synchronize gradients across devices with optional quantization"""
        try:
            if self.world.size == 1:
                return gradients
                
            if self.config.use_quantization:
                # Quantize gradients for efficient communication
                quantized_grads = {}
                for name, grad in gradients.items():
                    # Compute scale for quantization
                    max_val = mx.max(mx.abs(grad))
                    scale = max_val / ((1 << (self.config.quantization_bits - 1)) - 1)
                    
                    # Quantize to int8/int16
                    quantized = mx.array(
                        mx.clip(grad / scale * ((1 << (self.config.quantization_bits - 1)) - 1), 
                               -(1 << (self.config.quantization_bits - 1)), 
                               (1 << (self.config.quantization_bits - 1)) - 1),
                        dtype=mx.int8 if self.config.quantization_bits == 8 else mx.int16
                    )
                    
                    # Store quantized gradient and scale
                    quantized_grads[name] = (quantized, scale)
                    
                # Synchronize quantized gradients
                synced_grads = {}
                for name, (quantized, scale) in quantized_grads.items():
                    # All-reduce the quantized gradients
                    synced_quantized = await self.world.all_reduce(quantized, "sum")
                    
                    # Dequantize
                    synced_grads[name] = (synced_quantized.astype(mx.float32) * scale) / self.world.size
                    
                return synced_grads
            else:
                # Synchronize full-precision gradients
                synced_grads = {}
                for name, grad in gradients.items():
                    synced_grads[name] = await self.world.all_reduce(grad, "sum") / self.world.size
                    
                return synced_grads
                
        except Exception as e:
            self.logger.error(f"Gradient synchronization failed: {str(e)}")
            raise
            
    async def broadcast_weights(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Broadcast model weights from primary to secondary devices"""
        try:
            if self.world.rank == 0:
                # Primary device broadcasts weights
                for name, w in weights.items():
                    mx.distributed.broadcast(w, 0)
            else:
                # Secondary devices receive weights
                for name, w in weights.items():
                    weights[name] = mx.distributed.broadcast(w, 0)
            return weights
            
        except Exception as e:
            self.logger.error(f"Weight broadcast failed: {str(e)}")
            raise
            
    def verify_connection(self) -> bool:
        """Verify network connectivity between devices"""
        try:
            # Simple ping-pong test
            test_tensor = mx.array([self.world.rank], dtype=mx.float32)
            result = mx.distributed.all_sum(test_tensor)
            expected_sum = sum(range(self.world.size))
            
            if result[0] == expected_sum:
                self.logger.info("Network verification successful")
                return True
            else:
                self.logger.error("Network verification failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Network verification failed: {str(e)}")
            return False 

    async def verify_cluster(self) -> bool:
        """Verify full cluster connectivity"""
        try:
            # Test bandwidth
            bandwidth = await self.test_bandwidth()
            if bandwidth < 8_000:  # 8 Gbps minimum
                self.logger.error(f"Insufficient bandwidth: {bandwidth} Mbps")
                return False
                
            # Test latency
            latency = await self.test_latency()
            if latency > 2:  # 2ms maximum
                self.logger.error(f"High latency: {latency}ms")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Cluster verification failed: {e}")
            return False 

    def get_current_bandwidth(self) -> float:
        """Get current network bandwidth in Mbps"""
        return self._last_bandwidth
        
    async def test_bandwidth(self) -> float:
        """Test current network bandwidth"""
        try:
            # Send a large tensor to measure bandwidth
            test_size = 100 * 1024 * 1024  # 100MB
            test_data = mx.random.uniform(shape=(test_size // 4,), dtype=mx.float32)
            
            start_time = time.time()
            await self.world.all_reduce(test_data, "sum")
            end_time = time.time()
            
            # Calculate bandwidth in Mbps
            duration = end_time - start_time
            bandwidth = (test_size * 8 * (self.world.size - 1)) / (duration * 1024 * 1024)
            
            self._last_bandwidth = bandwidth
            self._bandwidth_history.append(bandwidth)
            if len(self._bandwidth_history) > 10:
                self._bandwidth_history.pop(0)
                
            return bandwidth
            
        except Exception as e:
            self.logger.error(f"Bandwidth test failed: {str(e)}")
            return float('inf')
            
    def get_bandwidth_stats(self) -> Dict[str, float]:
        """Get bandwidth statistics"""
        if not self._bandwidth_history:
            return {'current': float('inf'), 'average': float('inf'), 'min': float('inf')}
            
        return {
            'current': self._last_bandwidth,
            'average': sum(self._bandwidth_history) / len(self._bandwidth_history),
            'min': min(self._bandwidth_history)
        }