import mlx.core as mx
import socket
import logging
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    primary_host: str = "localhost"
    primary_port: int = 29500
    timeout: float = 30.0
    retry_attempts: int = 3
    buffer_size: int = 8192

class DistributedCommunicator:
    """Handles network communication for distributed training"""
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.world = mx.distributed.get_world()
        self._setup_logging()
        
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        
    async def sync_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Synchronize gradients across devices"""
        try:
            # Use MLX's built-in all_reduce for gradient synchronization
            synced_grads = {}
            for name, grad in gradients.items():
                synced_grads[name] = mx.distributed.all_sum(grad) / self.world.size
            return synced_grads
            
        except Exception as e:
            self.logger.error(f"Gradient sync failed: {str(e)}")
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