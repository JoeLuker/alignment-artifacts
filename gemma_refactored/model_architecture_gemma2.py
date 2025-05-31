"""
Core Gemma-2 model architecture components with activation capture.
This version is compatible with Gemma-2 models (without QK-norm).
"""

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional
import mlx.core as mx
import mlx.nn as nn

from .activation_capture import activation_store, scaled_dot_product_attention_with_activations
from .model_loading import BaseModelArgs


@partial(mx.compile, shapeless=True)
def clip_residual(x, y):
    """Clip residual connection to avoid float16 overflow."""
    if x.dtype != mx.float16 or y.dtype != mx.float16:  # Check both types
        return x + y
    bound = mx.finfo(mx.float16).max
    result_f32 = x.astype(mx.float32) + y.astype(mx.float32)
    result_f32_clipped = mx.clip(result_f32, -bound, bound)
    return result_f32_clipped.astype(mx.float16)


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    dtype=mx.bool_  # Original returned boolean
):
    """Create a boolean causal mask for auto-regressive generation."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds <= rinds + window_size)
    return mask.astype(dtype)  # Ensure specified dtype


def create_attention_mask(
    h: mx.array, 
    cache: Optional[Any] = None, 
    return_array: bool = False, 
    dtype=mx.float32
):
    """Create an attention mask for transformer layers. Returns additive float mask or None."""
    T = h.shape[1]
    if T <= 1 and not return_array: 
        return None  # No mask needed for single token by default

    offset = 0
    window_size = None
    # Handle cache being a list or single object
    cache_obj = None
    if cache is not None:
        if isinstance(cache, list) and cache: 
            cache_obj = cache[0]
        elif not isinstance(cache, list): 
            cache_obj = cache

    if cache_obj is not None:
        if hasattr(cache_obj, "offset"): 
            offset = cache_obj.offset
        if hasattr(cache_obj, "max_size") and cache_obj.max_size is not None:
            window_size = cache_obj.max_size
            offset = min(window_size, offset)  # Original logic
            return_array = return_array or (offset + T > window_size and T > 0)

    # Generate mask if needed (T > 1 or forced by windowing/return_array)
    if return_array or T > 1:
        bool_mask = create_causal_mask(T, offset, window_size=window_size)
        # Convert boolean mask to additive float mask
        return mx.where(bool_mask, 0.0, -1e9).astype(dtype)
    else:
        return None


def create_additive_causal_mask(N: int, offset: int = 0, dtype=mx.float32):
    """Create additive causal mask directly."""
    return create_attention_mask(
        mx.zeros((1, N, 1)),  # Dummy tensor with correct sequence length
        cache=None,
        return_array=True,
        dtype=dtype
    )


class RMSNorm(nn.Module):
    """RMS Normalization layer with activation capture."""
    
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        # Assign unique name for activation logging
        self._activation_name_prefix = f"rmsnorm_{id(self)}"

    def __call__(self, x):
        activation_store.register(f"{self._activation_name_prefix}.input", x)
        output = mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
        activation_store.register(f"{self._activation_name_prefix}.output", output)
        return output


class MLP(nn.Module):
    """MLP layer used in Gemma3 with activation capture."""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        # Assign unique name for activation logging
        self.name = f"mlp_{id(self)}"

    def __call__(self, x) -> mx.array:
        activation_store.register(f"{self.name}.input", x)
        
        # Capture intermediate projections
        gate_out = self.gate_proj(x)
        activation_store.register(f"{self.name}.gate_proj_out", gate_out)
        
        up_out = self.up_proj(x)
        activation_store.register(f"{self.name}.up_proj_out", up_out)
        
        # Capture activation output
        gelu_out = nn.gelu_approx(gate_out)
        activation_store.register(f"{self.name}.gelu_approx_out", gelu_out)
        
        # Capture element-wise product
        prod = gelu_out * up_out
        activation_store.register(f"{self.name}.gate_up_product", prod)
        
        # Capture final output
        output = self.down_proj(prod)
        activation_store.register(f"{self.name}.output", output)
        
        return output


class Attention(nn.Module):
    """Multi-head attention module for Gemma3 with activation capture."""
    
    def __init__(self, args: "ModelArgs", layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        # Handle potential missing num_key_value_heads in config
        self.n_kv_heads = n_kv_heads = getattr(args, 'num_key_value_heads', n_heads)
        self.repeats = n_heads // n_kv_heads
        assert n_heads % n_kv_heads == 0, f"n_heads({n_heads}) must be divisible by n_kv_heads({n_kv_heads})"
        
        # Handle potential missing head_dim
        self.head_dim = head_dim = getattr(args, 'head_dim', dim // n_heads)
        self.layer_idx = layer_idx

        # Use query_pre_attn_scalar if present, default based on head_dim
        query_scalar = getattr(args, 'query_pre_attn_scalar', None)
        if query_scalar is not None:
            self.scale = query_scalar ** -0.5
        else:
            self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # Gemma-2 doesn't have q_norm/k_norm (QK-norm was introduced in Gemma-3)
        # self.q_norm = RMSNorm(dims=head_dim, eps=args.rms_norm_eps)
        # self.k_norm = RMSNorm(dims=head_dim, eps=args.rms_norm_eps)

        # Sliding window check based on pattern (if defined)
        self.is_sliding = False
        sliding_pattern = getattr(args, 'sliding_window_pattern', None)
        if sliding_pattern is not None:
            self.is_sliding = (layer_idx + 1) % sliding_pattern != 0

        # Determine RoPE base frequency
        rope_base = 10000.0  # Default
        if hasattr(args, 'rope_theta') and args.rope_theta: 
            rope_base = args.rope_theta
        elif hasattr(args, 'rope_local_base_freq') and args.rope_local_base_freq:
            # Use local base freq if available (for sliding window models)
            rope_base = args.rope_local_base_freq
        
        # Override with global freq for non-sliding layers if both are defined
        if (hasattr(args, 'rope_global_base_freq') and args.rope_global_base_freq and 
            hasattr(args, 'rope_local_base_freq') and args.rope_local_base_freq and
            not self.is_sliding):
            rope_base = args.rope_global_base_freq

        self.rope = nn.RoPE(
            head_dim,
            traditional=getattr(args, 'rope_traditional', False),
            base=rope_base,
        )
        
        # Naming for activation store
        self.name = f"model.layers.{layer_idx}.self_attn"
        # Gemma-2 doesn't have q_norm/k_norm
        # self.q_norm._activation_name_prefix = f"{self.name}.q_norm"
        # self.k_norm._activation_name_prefix = f"{self.name}.k_norm"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,  # Expects additive float mask or None
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        activation_store.register(f"{self.name}.input", x)

        # Projections
        queries = self.q_proj(x)
        activation_store.register(f"{self.name}.q_proj", queries)
        
        keys = self.k_proj(x)
        activation_store.register(f"{self.name}.k_proj", keys)
        
        values = self.v_proj(x)
        activation_store.register(f"{self.name}.v_proj", values)

        # Reshape & Transpose
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        activation_store.register(f"{self.name}.queries_reshaped", queries)
        
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        activation_store.register(f"{self.name}.keys_reshaped", keys)
        
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        activation_store.register(f"{self.name}.values_reshaped", values)

        # Gemma-2 doesn't have QK-norm, so skip normalization
        # queries = self.q_norm(queries)
        activation_store.register(f"{self.name}.queries_normed", queries)
        
        # keys = self.k_norm(keys)
        activation_store.register(f"{self.name}.keys_normed", keys)

        # RoPE & Cache
        offset = 0
        if cache is not None:
            offset = getattr(cache, 'offset', 0)  # Safely get offset
            activation_store.register(f"{self.name}.cache_offset", mx.array(offset))
            
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            activation_store.register(f"{self.name}.queries_rope", queries)
            activation_store.register(f"{self.name}.keys_rope", keys)
            
            activation_store.register(f"{self.name}.keys_before_cache_update", keys)
            activation_store.register(f"{self.name}.values_before_cache_update", values)
            
            keys, values = cache.update_and_fetch(keys, values)  # Assumes cache has this method
            activation_store.register(f"{self.name}.keys_from_cache", keys)
            activation_store.register(f"{self.name}.values_from_cache", values)
        else:
            # Apply RoPE without cache offset
            queries = self.rope(queries)
            keys = self.rope(keys)
            activation_store.register(f"{self.name}.queries_rope", queries)
            activation_store.register(f"{self.name}.keys_rope", keys)

        # GQA/MQA Repeat Keys/Values
        if self.repeats > 1:
            keys = mx.repeat(keys, repeats=self.repeats, axis=1)
            activation_store.register(f"{self.name}.keys_repeated", keys)
            
            values = mx.repeat(values, repeats=self.repeats, axis=1)
            activation_store.register(f"{self.name}.values_repeated", values)

        # Store the mask before potential slicing
        activation_store.register(f"{self.name}.mask_before_slicing", mask)

        # Original mask slicing logic
        if isinstance(mask, mx.array) and mask.ndim > 1 and mask.shape[-1] != keys.shape[-2]:
            mask = mask[..., -keys.shape[-2]:]  # Slice the last dimension
            activation_store.register(f"{self.name}.mask_sliced", mask)

        # Register the mask that will actually be used in the attention calculation
        activation_store.register(f"{self.name}.final_mask_used", mask)

        # Use manual SDPA for activation capture
        output = scaled_dot_product_attention_with_activations(
            queries, keys, values, scale=self.scale, mask=mask, attn_name=self.name
        )
        activation_store.register(f"{self.name}.attn_output", output)

        # Reshape & Output Projection
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        activation_store.register(f"{self.name}.attn_output_reshaped", output)
        
        output = self.o_proj(output)
        activation_store.register(f"{self.name}.output", output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block for Gemma3 with activation capture."""
    
    def __init__(self, args: "ModelArgs", layer_idx: int):
        super().__init__()
        # Keep original attributes
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        
        # Use modified sub-modules
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Naming for activation store
        self.name = f"model.layers.{layer_idx}"
        
        # Assign clearer names to child norms for activation logging
        self.input_layernorm._activation_name_prefix = f"{self.name}.input_layernorm"
        self.post_attention_layernorm._activation_name_prefix = f"{self.name}.post_attention_layernorm"
        self.pre_feedforward_layernorm._activation_name_prefix = f"{self.name}.pre_feedforward_layernorm"
        self.post_feedforward_layernorm._activation_name_prefix = f"{self.name}.post_feedforward_layernorm"
        
        # Assign name to child MLP for activation logging
        self.mlp.name = f"{self.name}.mlp"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        activation_store.register(f"{self.name}.input", x)
        
        # Attention Path
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        activation_store.register(f"{self.name}.attn_output_raw", r)
        
        h = clip_residual(x, self.post_attention_layernorm(r))
        activation_store.register(f"{self.name}.attn_residual_out", h)

        # MLP Path
        mlp_in = self.pre_feedforward_layernorm(h)
        activation_store.register(f"{self.name}.pre_ff_layernorm_out", mlp_in)
        
        r = self.mlp(mlp_in)
        activation_store.register(f"{self.name}.mlp_output_raw", r)
        
        out = clip_residual(h, self.post_feedforward_layernorm(r))
        activation_store.register(f"{self.name}.output", out)
        
        return out


class Gemma3Model(nn.Module):
    """Core Gemma3 model implementation with activation capture."""
    
    def __init__(self, args: "ModelArgs"):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        
        # Use modified TransformerBlock
        self.layers = [
            TransformerBlock(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        
        # Use modified RMSNorm
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Naming for activation store
        self.name = "model"
        self.norm._activation_name_prefix = f"{self.name}.norm"

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        activation_store.register(f"{self.name}.input_ids", inputs)
        
        h = self.embed_tokens(inputs)
        activation_store.register(f"{self.name}.embed_tokens_out", h)

        # Scaling factor - ensure correct dtype and capture
        hidden_size_sqrt = mx.array(self.args.hidden_size**0.5)
        h = h * hidden_size_sqrt.astype(h.dtype)
        activation_store.register(f"{self.name}.scaled_embeddings", h)

        # Initialize cache if not provided
        if cache is None:
            cache = [None] * len(self.layers)

        # Mask creation logic
        active_mask = mask  # Start with the passed mask (could be None)
        if mask is None:
            sliding_pattern = getattr(self.args, 'sliding_window_pattern', None)
            if sliding_pattern is not None:
                # Get cache state for relevant layers
                global_layer_idx = sliding_pattern - 1
                global_cache_slice = cache[global_layer_idx : global_layer_idx + 1] if global_layer_idx < len(cache) else None
                
                full_mask = create_attention_mask(h, global_cache_slice, return_array=True, dtype=h.dtype)
                sliding_window_mask = create_attention_mask(h, cache, return_array=True, dtype=h.dtype)
                
                activation_store.register(f"{self.name}.computed_full_mask", full_mask)
                activation_store.register(f"{self.name}.computed_sliding_mask", sliding_window_mask)
            else:
                # If no pattern, default to standard causal mask if T > 1
                full_mask = create_attention_mask(h, cache=None, return_array=True, dtype=h.dtype)
                sliding_window_mask = full_mask  # No distinction without pattern
                activation_store.register(f"{self.name}.computed_default_mask", full_mask)

        # Pass through layers
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            # Determine which mask to use
            local_mask_to_use = active_mask
            if active_mask is None and 'sliding_pattern' in locals() and sliding_pattern is not None:
                is_global_layer = (i % sliding_pattern == sliding_pattern - 1)
                local_mask_to_use = full_mask if is_global_layer else sliding_window_mask
                activation_store.register(f"{layer.name}.mask_type_selected", "full" if is_global_layer else "sliding")
            elif active_mask is None:
                local_mask_to_use = full_mask if 'full_mask' in locals() else None

            h = layer(h, local_mask_to_use, c)

        h = self.norm(h)
        return h


class Model(nn.Module):
    """Complete model with language modeling head."""
    
    def __init__(self, args: "ModelArgs"):
        super().__init__()
        self.args = args
        self.model = Gemma3Model(args)
        if not getattr(args, "tie_word_embeddings", True):
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if hasattr(self, "lm_head"):
            return self.lm_head(out)
        else:
            return self.model.embed_tokens.as_linear(out)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        """Sanitize weights to match model structure."""
        weights = dict(weights)
        
        # Handle embedding/lm_head weight sharing
        embed_key = "model.embed_tokens.weight"
        head_key = "lm_head.weight"
        if head_key not in weights and embed_key in weights:
            print(f"Sanitizing weights: Copying {embed_key} to {head_key}")
            weights[head_key] = weights[embed_key]
        
        # Remove unused precomputed freqs 
        keys_to_remove = []
        for key in weights.keys():
            if "rotary_emb.inv_freq" in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del weights[key]
            
        return weights


@dataclass
class ModelArgs(BaseModelArgs):
    """Model configuration arguments."""
    vocab_size: int = 262144
    hidden_size: int = 1152
    intermediate_size: int = 6912
    num_hidden_layers: int = 26
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    tie_word_embeddings: bool = True
    query_pre_attn_scalar: Optional[float] = None
    sliding_window_pattern: Optional[int] = None
    sliding_window: Optional[int] = None
    rope_local_base_freq: Optional[float] = None
    rope_global_base_freq: Optional[float] = None