import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict, Any
from src.models.model_utils import make_divisible

@dataclass
class PersonalAIConfig:
    """Configuration for the Personal AI model"""
    vocab_size: int = 32000
    max_context_length: int = 8192
    num_transformer_layers: int = 22
    model_dim: int = 2048
    head_dim: int = 64
    num_gqa_groups: int = 4
    ffn_multipliers: Tuple[float, float] = (0.5, 4.0)
    qkv_multipliers: Tuple[float, float] = (0.5, 1.0)
    normalize_qk_projections: bool = True
    share_input_output_layers: bool = True
    activation_fn_name: str = "swish"
    normalization_layer_name: str = "rms_norm"
    rope_freq_constant: int = 10000
    rope_max_length: int = 16384  # 2x max_context_length for flexibility

class MultiHeadCausalAttention(nn.Module):
    """Enhanced Multi-head causal attention with GQA support"""
    def __init__(self, config: PersonalAIConfig, layer_idx: int):
        super().__init__()
        # Calculate variable dimensions based on layer index
        qkv_multiplier = config.qkv_multipliers[0] + (
            (config.qkv_multipliers[1] - config.qkv_multipliers[0]) 
            * (layer_idx / (config.num_transformer_layers - 1))
        )
        qkv_dim = make_divisible(
            int(config.model_dim * qkv_multiplier),
            divisor=config.head_dim * config.num_gqa_groups
        )
        
        self.num_q_heads = qkv_dim // config.head_dim
        self.num_kv_heads = self.num_q_heads // config.num_gqa_groups
        
        # Projections
        self.qkv_proj = nn.Linear(
            input_dims=config.model_dim,
            output_dims=(self.num_q_heads + 2 * self.num_kv_heads) * config.head_dim,
            bias=False
        )
        
        self.pos_embedding = nn.RoPE(
            config.head_dim,
            base=config.rope_freq_constant,
            max_position=config.rope_max_length
        )
        
        if config.normalize_qk_projections:
            self.q_norm = nn.RMSNorm(config.head_dim)
            self.k_norm = nn.RMSNorm(config.head_dim)
        else:
            self.q_norm = self.k_norm = None
            
        self.out_proj = nn.Linear(
            input_dims=self.num_q_heads * config.head_dim,
            output_dims=config.model_dim,
            bias=False
        )
        
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5

    def __call__(
        self,
        x: mx.array,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_kv_cache: bool = False,
        causal_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        batch_size, seq_length, _ = x.shape
        
        # Combined QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(
            batch_size, seq_length,
            self.num_q_heads + 2 * self.num_kv_heads,
            self.head_dim
        )
        qkv = qkv.transpose(0, 2, 1, 3)
        
        # Split into Q, K, V
        q, k, v = mx.split(
            qkv,
            [self.num_q_heads, self.num_q_heads + self.num_kv_heads],
            axis=1
        )
        
        # Apply normalization if configured
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
            
        # Handle KV caching
        if use_kv_cache and past_key_value is not None:
            past_k, past_v = past_key_value
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
            
        # Apply rotary embeddings
        q = self.pos_embedding(q)
        k = self.pos_embedding(k)
        
        # Handle GQA
        if self.num_q_heads > self.num_kv_heads:
            k = mx.repeat(k, self.num_q_heads // self.num_kv_heads, axis=1)
            v = mx.repeat(v, self.num_q_heads // self.num_kv_heads, axis=1)
            
        # Scaled dot-product attention
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            mask=causal_mask
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        out = self.out_proj(attn_output)
        
        if use_kv_cache:
            return out, (k, v)
        return out, None

class FeedForwardNetwork(nn.Module):
    """Enhanced FFN with variable width and GLU activation"""
    def __init__(self, config: PersonalAIConfig, layer_idx: int):
        super().__init__()
        # Calculate variable FFN dimension based on layer index
        ffn_multiplier = config.ffn_multipliers[0] + (
            (config.ffn_multipliers[1] - config.ffn_multipliers[0]) 
            * (layer_idx / (config.num_transformer_layers - 1))
        )
        intermediate_dim = make_divisible(
            int(ffn_multiplier * config.model_dim),
            divisor=256
        )
        
        # GLU-style FFN
        self.gate_proj = nn.Linear(
            config.model_dim,
            intermediate_dim,
            bias=False
        )
        self.up_proj = nn.Linear(
            config.model_dim,
            intermediate_dim,
            bias=False
        )
        self.down_proj = nn.Linear(
            intermediate_dim,
            config.model_dim,
            bias=False
        )
        self.act = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerLayer(nn.Module):
    """Transformer layer with OpenELM optimizations"""
    def __init__(self, config: PersonalAIConfig, layer_idx: int):
        super().__init__()
        self.pre_attention_norm = nn.RMSNorm(config.model_dim)
        self.attention = MultiHeadCausalAttention(config, layer_idx)
        self.pre_ffn_norm = nn.RMSNorm(config.model_dim)
        self.ffn = FeedForwardNetwork(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_kv_cache: bool = False,
        causal_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Pre-norm attention
        attn_norm = self.pre_attention_norm(x)
        attn_output, past_key_value = self.attention(
            attn_norm,
            past_key_value,
            use_kv_cache,
            causal_mask
        )
        x = x + attn_output
        
        # Pre-norm FFN
        ffn_norm = self.pre_ffn_norm(x)
        x = x + self.ffn(ffn_norm)
        
        return x, past_key_value

class PersonalAIModel(nn.Module):
    """Personal AI model optimized for on-device deployment"""
    def __init__(self, config: PersonalAIConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            dims=config.model_dim
        )
        
        # Context projection
        self.context_projection = nn.Linear(
            input_dims=256,  # Default context size
            output_dims=config.model_dim,
            bias=False
        )
        
        # Transformer layers
        self.layers = [
            TransformerLayer(config, i)
            for i in range(config.num_transformer_layers)
        ]
        
        # Final norm
        self.norm = nn.RMSNorm(config.model_dim)
        
        # Optional tied embeddings
        if config.share_input_output_layers:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.model_dim,
                config.vocab_size,
                bias=False
            )
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters following OpenELM's scheme"""
        def _normal_init(std: float):
            return lambda x: mx.random.normal(x.shape) * std
            
        # Embedding initialization
        emb_std = self.config.model_dim ** -0.5
        self.token_embedding.weight = _normal_init(emb_std)(
            self.token_embedding.weight
        )
        
        # Layer-specific initialization
        layer_std = (self.config.model_dim ** -0.5) * (
            (2 * self.config.num_transformer_layers) ** -0.5
        )
        
        for layer in self.layers:
            # Attention projections
            layer.attention.qkv_proj.weight = _normal_init(layer_std)(
                layer.attention.qkv_proj.weight
            )
            layer.attention.out_proj.weight = _normal_init(layer_std)(
                layer.attention.out_proj.weight
            )
            
            # FFN projections
            layer.ffn.gate_proj.weight = _normal_init(layer_std)(
                layer.ffn.gate_proj.weight
            )
            layer.ffn.up_proj.weight = _normal_init(layer_std)(
                layer.ffn.up_proj.weight
            )
            layer.ffn.down_proj.weight = _normal_init(layer_std)(
                layer.ffn.down_proj.weight
            )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        context_embedding: Optional[mx.array] = None,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
        use_kv_cache: bool = False,
    ) -> Dict[str, mx.array]:
        """Forward pass with context integration"""
        # Get input embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add context if provided
        if context_embedding is not None:
            context_proj = self.context_projection(context_embedding)
            hidden_states = hidden_states + context_proj[:, None, :]
        
        # Create causal mask
        if attention_mask is None:
            causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(
                hidden_states.shape[1]
            )
        else:
            causal_mask = attention_mask
        
        # Process through layers
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values else None
            
            hidden_states, present_kv = layer(
                hidden_states,
                past_key_value=past_key_value,
                use_kv_cache=use_kv_cache,
                causal_mask=causal_mask
            )
            present_key_values.append(present_kv)
            
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Get logits
        if self.lm_head is None:
            logits = mx.linear(hidden_states, self.token_embedding.weight.T)
        else:
            logits = self.lm_head(hidden_states)
            
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "present_key_values": present_key_values if use_kv_cache else None
        }
