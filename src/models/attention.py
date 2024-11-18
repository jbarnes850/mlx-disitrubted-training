import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Dict
import math

class EnhancedAttention(nn.Module):
    """Enhanced attention with multiple improvements"""
    def __init__(self, config):
        super().__init__()
        # Sliding window attention
        self.window_size = config.sliding_window
        # Grouped-query attention
        self.num_kv_heads = config.kv_heads
        self.num_heads = config.num_heads
        self.head_dim = config.dims // config.num_heads
        
        # Rotary embeddings with extended scaling
        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=True,
            base=config.rope_base
        )
        
        # Optimized projections
        self.q_proj = nn.Linear(config.dims, config.dims, bias=False)
        self.k_proj = nn.Linear(config.dims, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dims, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dims, config.dims, bias=False)
        
    def forward(self, x, mask=None, cache=None):
        B, L, D = x.shape
        
        # Apply sliding window attention
        if self.window_size and L > self.window_size:
            attention_mask = self._create_sliding_window_mask(L)
            if mask is not None:
                mask = mask & attention_mask
        
        # Project to q, k, v
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, -1)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, -1)
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # Handle cache for inference
        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=1)
            v = mx.concatenate([value_cache, v], axis=1)
            
        # Repeat k,v for grouped-query attention
        if self.use_gqa:
            k = mx.repeat(k, self.num_heads // self.num_kv_heads, axis=2)
            v = mx.repeat(v, self.num_heads // self.num_kv_heads, axis=2)
            
        # Scaled dot-product attention
        scale = 1 / math.sqrt(self.head_dim)
        
        if self.use_flash and not self.training:
            # Use flash attention for inference
            attn_output = self._flash_attention(q, k, v, mask, scale)
        else:
            # Regular attention for training
            scores = (q * scale) @ k.transpose(0, 1, 3, 2)
            if mask is not None:
                scores = scores + mask
            attn_weights = mx.softmax(scores, axis=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = attn_weights @ v
            
        # Reshape and project output
        output = self.o_proj(attn_output.reshape(B, L, -1))
        
        return output, (k, v)
        
    def _flash_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Optional[mx.array],
        scale: float
    ) -> mx.array:
        """Flash attention implementation"""
        # TODO: Implement flash attention optimization
        # This is a placeholder for the actual implementation
        scores = (q * scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        attn_weights = mx.softmax(scores, axis=-1)
        return attn_weights @ v 