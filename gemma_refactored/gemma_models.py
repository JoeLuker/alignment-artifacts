"""
Gemma model implementations adapted from mlx-lm.
Supports Gemma, Gemma2, and other variants.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float = 1e-6
    vocab_size: int = 256000
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    model_type: str = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 8192
    query_pre_attn_scalar: Optional[int] = None
    # For multimodal
    vision_config: Optional[dict] = None
    text_config: Optional[dict] = None
    
    @classmethod
    def from_dict(cls, params):
        import inspect
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads or n_heads

        self.repeats = n_heads // n_kv_heads

        head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = (args.query_pre_attn_scalar or head_dim) ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        rope_scale = 1.0
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            # Handle both KVCache objects and tuple format
            if hasattr(cache, 'update_and_fetch'):
                # KVCache object - use its method
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset) 
                keys, values = cache.update_and_fetch(keys, values)
            else:
                # Tuple format - original logic
                key_cache, value_cache = cache
                queries = self.rope(queries, offset=key_cache.shape[2])
                keys = self.rope(keys, offset=key_cache.shape[2])
                keys = mx.concatenate([key_cache, keys], axis=2)
                values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h = layer(h, mask, cache[e])

        return self.norm(h)


class Model(nn.Module):
    """Main model class that handles both text-only and multimodal Gemma."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type or "gemma"
        
        # Always create text model, even for multimodal
        # (we'll use it text-only mode)
        if args.vision_config:
            print("Note: Loading multimodal model in text-only mode")
            # Use text config if available
            if hasattr(args, 'text_config') and args.text_config:
                # Update args with text config
                for k, v in args.text_config.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
        
        self.model = GemmaModel(args)
        if not hasattr(args, "tie_word_embeddings") or args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is not None:
            return self.lm_head(out)
        else:
            return self.model.embed_tokens.as_linear(out)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        # Handle different weight naming conventions
        new_weights = {}
        
        for k, v in weights.items():
            # Skip vision-related weights
            if "vision" in k:
                continue
            
            # Handle multimodal model weight names
            if k.startswith("language_model."):
                # Remove language_model prefix for multimodal models
                new_k = k.replace("language_model.", "")
                new_weights[new_k] = v
            elif k.startswith("model.") or not any(k.startswith(p) for p in ["embed_tokens", "norm", "layers"]):
                # Already has model prefix or needs one
                new_weights[k] = v
            else:
                # Add model prefix
                new_weights[f"model.{k}"] = v
        
        # Special handling for quantized embeddings in multimodal models
        # The embedding might be stored with scales/biases for quantization
        embed_key = "model.embed_tokens.weight"
        if embed_key in new_weights and hasattr(self.args, 'vision_config') and self.args.vision_config:
            # Check if we have quantized embedding components
            embed_scales = new_weights.get(f"{embed_key[:-7]}.scales")
            embed_biases = new_weights.get(f"{embed_key[:-7]}.biases")
            if embed_scales is not None or embed_biases is not None:
                print("Note: Quantized embeddings detected, keeping original weight shape")
        
        # Handle tied embeddings
        if hasattr(self.args, "tie_word_embeddings") and self.args.tie_word_embeddings:
            if "lm_head.weight" in new_weights:
                del new_weights["lm_head.weight"]
        
        return new_weights