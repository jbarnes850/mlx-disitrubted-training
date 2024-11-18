import mlx.core as mx
from typing import Dict, Any
import numpy as np

class ModelQuantizer:
    """Handles model quantization for efficient inference"""
    def __init__(self, bits: int = 4):
        self.bits = bits
        
    def quantize_weights(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Quantize model weights"""
        quantized = {}
        for name, param in weights.items():
            if 'embedding' in name or 'norm' in name:
                # Skip certain layers
                quantized[name] = param
                continue
                
            # Compute scale for quantization
            max_val = mx.abs(param).max()
            scale = (2 ** (self.bits - 1) - 1) / max_val
            
            # Quantize
            quantized_param = mx.round(param * scale)
            quantized_param = mx.clip(
                quantized_param,
                -2 ** (self.bits - 1),
                2 ** (self.bits - 1) - 1
            )
            
            # Dequantize
            quantized[name] = quantized_param / scale
            
        return quantized 