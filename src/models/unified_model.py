import math
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

@dataclass
class ModelConfig:
    """Enhanced configuration for better performance/size trade-off"""
    num_layers: int = 24        # Reduced from 32 for efficiency
    vocab_size: int = 32000
    dims: int = 2048           # Optimized for memory/performance
    mlp_dims: int = 5504       # 2.7x hidden dim (like Llama)
    num_heads: int = 16        # Adjusted for dims
    max_seq_length: int = 4096 # Increased context window
    rope_base: int = 1000000   # Extended RoPE scaling
    sliding_window: int = 256  # Add sliding window attention

class UnifiedAttention(nn.Module):
    """Multi-head attention with RoPE (Rotary Position Embedding)"""
    def __init__(self, dims: int, num_heads: int, rope_traditional: bool = True):
        super().__init__()
        self.num_heads = num_heads
        
        # RoPE positional embeddings
        self.rope = nn.RoPE(dims // num_heads, traditional=rope_traditional)
        
        # Projection layers
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)
        
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Project queries, keys, and values
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)
        
        # Extract shapes
        B, L, D = queries.shape
        num_heads = self.num_heads
        
        # Reshape for multi-head attention
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        
        # Apply RoPE and handle cache for inference
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        # Scaled dot-product attention
        scale = 1 / math.sqrt(queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        
        attention = mx.softmax(scores, axis=-1)
        values_hat = (attention @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # Final projection
        output = self.out_proj(values_hat)
        
        return output, (keys, values)

class UnifiedEncoderLayer(nn.Module):
    """Transformer encoder layer with RMSNorm and SwiGLU"""
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()
        
        # Attention and normalization
        self.attention = UnifiedAttention(dims, num_heads)
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        
        # MLP with SwiGLU
        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)
        
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Attention with residual
        residual = x
        x = self.norm1(x)
        x, cache = self.attention(x, mask=mask, cache=cache)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.norm2(x)
        
        # SwiGLU activation
        gate = self.linear1(x)
        x = self.linear2(x)
        x = gate * mx.sigmoid(gate) * x
        x = self.linear3(x)
        
        return residual + x, cache

class UnifiedModel(nn.Module):
    """Unified language model with RoPE and SwiGLU"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.dims)
        
        # Transformer layers
        self.layers = [
            UnifiedEncoderLayer(
                config.dims,
                config.mlp_dims,
                config.num_heads
            ) for _ in range(config.num_layers)
        ]
        
        # Output head
        self.norm = nn.RMSNorm(config.dims)
        self.out_proj = nn.Linear(config.dims, config.vocab_size, bias=False)
        
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        # Create causal mask if not provided
        if mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(self.embedding.weight.dtype)
        
        # Forward pass
        x = self.embedding(x)
        
        for layer in self.layers:
            x, _ = layer(x, mask=mask)
            
        x = self.norm(x)
        return self.out_proj(x)
    
    def generate(
        self,
        x: mx.array,
        max_length: int,
        temperature: float = 1.0
    ):
        """Generate tokens autoregressively"""
        # Initialize cache for attention layers
        cache: List[Optional[Tuple[mx.array, mx.array]]] = [None] * len(self.layers)
        
        # Process prompt
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)
        
        x = self.embedding(x)
        for i, layer in enumerate(self.layers):
            x, cache[i] = layer(x, mask=mask)
            
        x = self.norm(x)
        y = self.out_proj(x[:, -1])
        y = mx.random.categorical(y * (1/temperature))
        
        yield y
        
        # Generate tokens
        while True:
            x = y[:, None]
            x = self.embedding(x)
            
            for i, layer in enumerate(self.layers):
                x, cache[i] = layer(x, cache=cache[i])
                
            x = self.norm(x)
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1/temperature))
            
            yield y