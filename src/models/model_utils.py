import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple
import math

def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: Optional[mx.array] = None,
    scale: Optional[float] = None
) -> mx.array:
    """Optimized scaled dot-product attention"""
    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])
        
    # Compute attention scores
    scores = mx.matmul(query, key.transpose(-2, -1)) * scale
    
    if mask is not None:
        scores = mx.where(mask, scores, float('-inf'))
    
    # Compute attention weights
    weights = mx.softmax(scores, axis=-1)
    
    # Apply attention to values
    return mx.matmul(weights, value)

def apply_rotary_embeddings(
    x: mx.array,
    freqs: mx.array
) -> mx.array:
    """Apply rotary positional embeddings"""
    # Split input into real and imaginary components
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    
    # Apply rotation
    freqs_real = mx.cos(freqs)
    freqs_imag = mx.sin(freqs)
    
    out_real = x_real * freqs_real - x_imag * freqs_imag
    out_imag = x_real * freqs_imag + x_imag * freqs_real
    
    # Combine components
    return mx.stack([out_real, out_imag], axis=-1).reshape(*x.shape)

def get_rotary_frequencies(
    dim: int,
    max_seq_length: int,
    base: int = 10000
) -> mx.array:
    """Generate rotary embedding frequencies"""
    freqs = mx.exp(
        -mx.arange(0, dim, 2) * (math.log(base) / dim)
    )
    pos = mx.arange(max_seq_length)
    freqs = mx.outer(pos, freqs)
    return freqs 