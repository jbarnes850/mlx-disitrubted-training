import mlx.core as mx
from typing import Optional, Dict, List, Any
import logging
import psutil
from dataclasses import dataclass
import numpy as np

@dataclass
class MemoryConfig:
    """Memory management configuration"""
    max_memory_gb: float
    cache_fraction: float = 0.25
    wired_fraction: float = 0.125
    chunk_size: int = 128
    prefetch_buffer: int = 2
    enable_defrag: bool = True
    monitor_interval: float = 1.0
    activation_checkpointing: bool = True
    gradient_accumulation_steps: int = 16
    attention_slice_size: int = 1024

class AdvancedMemoryManager:
    """Enhanced memory management with advanced optimizations"""
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.peak_memory = 0
        self.setup_limits()
        
    def setup_limits(self):
        """Configure memory limits with optimizations"""
        if mx.metal.is_available():
            memory_bytes = int(self.config.max_memory_gb * 1024 * 1024 * 1024)
            
            # Set Metal memory limits
            mx.metal.set_memory_limit(memory_bytes)
            mx.metal.set_cache_limit(int(memory_bytes * self.config.cache_fraction))
            mx.metal.set_wired_limit(int(memory_bytes * self.config.wired_fraction))
            
            self.logger.info(f"Memory limits configured: {self.config.max_memory_gb}GB total")
            
    def optimize_memory_layout(self, model: Any) -> Any:
        """Optimize model memory layout"""
        # Enable gradient checkpointing
        model = self._setup_checkpointing(model)
        
        # Setup attention chunking
        model = self._setup_chunked_attention(model)
        
        # Enable activation recomputation
        model = self._setup_activation_recompute(model)
        
        return model
        
    def _setup_checkpointing(self, model: Any) -> Any:
        """Setup gradient checkpointing"""
        checkpointed_layers = []
        for i, layer in enumerate(model.layers):
            if i % 2 == 0:  # Checkpoint every other layer
                layer.attention = mx.checkpoint(layer.attention)
                layer.mlp = mx.checkpoint(layer.mlp)
                checkpointed_layers.append(i)
                
        self.logger.info(f"Gradient checkpointing enabled for layers: {checkpointed_layers}")
        return model
        
    def _setup_chunked_attention(self, model: Any) -> Any:
        """Setup chunked attention computation"""
        for layer in model.layers:
            if hasattr(layer, 'attention'):
                layer.attention.chunk_size = self.config.chunk_size
                
        self.logger.info(f"Attention chunking enabled with size: {self.config.chunk_size}")
        return model
        
    def _setup_activation_recompute(self, model: Any) -> Any:
        """Setup activation recomputation"""
        recompute_layers = ["mlp", "attention"]
        for layer in model.layers:
            for name in recompute_layers:
                if hasattr(layer, name):
                    setattr(layer, name, mx.recompute(getattr(layer, name)))
                    
        self.logger.info(f"Activation recomputation enabled for: {recompute_layers}")
        return model
        
    def monitor_memory(self) -> Dict[str, float]:
        """Get detailed memory statistics"""
        stats = {
            "system_used_gb": psutil.Process().memory_info().rss / (1024**3),
            "system_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        if mx.metal.is_available():
            active_mem = mx.metal.get_active_memory()
            peak_mem = mx.metal.get_peak_memory()
            self.peak_memory = max(self.peak_memory, peak_mem)
            
            stats.update({
                "metal_active_gb": active_mem / (1024**3),
                "metal_peak_gb": peak_mem / (1024**3),
                "metal_peak_ever_gb": self.peak_memory / (1024**3),
                "metal_cache_gb": mx.metal.get_cache_memory() / (1024**3),
                "utilization": active_mem / mx.metal.get_memory_limit()
            })
            
        return stats
        
    def defragment(self):
        """Perform memory defragmentation"""
        if not self.config.enable_defrag:
            return
            
        if mx.metal.is_available():
            # Clear cache
            mx.metal.clear_cache()
            
            # Reset peak memory tracking
            mx.metal.reset_peak_memory()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Memory defragmentation completed")
            
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary"""
        stats = self.monitor_memory()
        
        # Calculate efficiency metrics
        if mx.metal.is_available():
            stats["fragmentation"] = (
                stats["metal_peak_gb"] - stats["metal_active_gb"]
            ) / stats["metal_peak_gb"]
            
            stats["cache_efficiency"] = (
                stats["metal_cache_gb"] / (self.config.max_memory_gb * self.config.cache_fraction)
            )
            
        return {
            "current": stats,
            "config": self.config.__dict__,
            "warnings": self._get_memory_warnings(stats)
        }
        
    def _get_memory_warnings(self, stats: Dict[str, float]) -> List[str]:
        """Generate memory-related warnings"""
        warnings = []
        
        if stats.get("utilization", 0) > 0.95:
            warnings.append("High memory utilization")
            
        if stats.get("fragmentation", 0) > 0.3:
            warnings.append("High memory fragmentation")
            
        return warnings