import mlx.core as mx
from typing import Optional
import logging

class DeviceOptimizer:
    """Optimizes model for on-device inference"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def optimize_for_device(self):
        """Apply device-specific optimizations"""
        # Enable Metal optimizations
        if mx.metal.is_available():
            self._enable_metal_optimizations()
            
        # Optimize memory usage
        self._optimize_memory()
        
        # Cache computation graphs
        self._cache_graphs()
        
        return self.model
        
    def _enable_metal_optimizations(self):
        """Enable Metal-specific optimizations"""
        # Set optimal thread count
        mx.metal.set_num_threads(6)  # Balanced for M-series chips
        
        # Enable fast math
        mx.metal.set_fast_math(True)
        
        # Set optimal tile size
        mx.metal.set_max_tile_size(1024)
        
    def _optimize_memory(self):
        """Optimize memory usage"""
        # Set sliding window attention
        self.model.config.sliding_window = 512
        
        # Enable gradient checkpointing
        for layer in self.model.layers:
            layer.attention = mx.checkpoint(layer.attention)
            
        # Use grouped-query attention
        self.model.config.use_gqa = True
        self.model.config.kv_heads = 4
        
    def _cache_graphs(self):
        """Cache computation graphs for faster inference"""
        # Compile and cache forward pass
        @mx.compile
        def forward_pass(x):
            return self.model(x)
            
        # Cache generation graph
        @mx.compile
        def generate_token(x, cache):
            return self.model.generate_token(x, cache)
            
        self.model.forward_cached = forward_pass
        self.model.generate_cached = generate_token 