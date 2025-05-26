"""Simple multimodal Gemma implementation based on mlx-vlm."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from .gemma_models import GemmaModel, ModelArgs, RMSNorm


@dataclass
class VisionConfig:
    """Vision model configuration."""
    hidden_size: int = 1152
    image_size: int = 336
    patch_size: int = 14
    num_patches: int = 576
    layer_norm_eps: float = 1e-6


@dataclass  
class MultimodalConfig(ModelArgs):
    """Config for multimodal Gemma models."""
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    mm_tokens_per_image: int = 256
    mm_emb_dim: int = 3840
    
    @classmethod
    def from_dict(cls, params):
        # First create base ModelArgs
        import inspect
        base_params = {k: v for k, v in params.items() 
                      if k in inspect.signature(ModelArgs).parameters}
        
        # Create vision config if present
        vision_params = params.get('vision_config', {})
        if vision_params:
            # Only pass recognized parameters
            vision_config = VisionConfig(
                hidden_size=vision_params.get('hidden_size', 1152),
                image_size=vision_params.get('image_size', 336),
                patch_size=vision_params.get('patch_size', 14),
                num_patches=vision_params.get('num_patches', 576),
                layer_norm_eps=vision_params.get('layer_norm_eps', 1e-6)
            )
        else:
            vision_config = VisionConfig()
        
        # Get multimodal specific params
        mm_tokens = params.get('mm_tokens_per_image', 256)
        mm_emb_dim = params.get('mm_emb_dim', params.get('hidden_size', 3840))
        
        return cls(
            **base_params,
            vision_config=vision_config,
            mm_tokens_per_image=mm_tokens,
            mm_emb_dim=mm_emb_dim
        )


class MultiModalProjector(nn.Module):
    """Projects vision features to text embedding space."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        vision_hidden_size = config.vision_config.hidden_size
        text_hidden_size = config.hidden_size
        
        # Normalization
        self.mm_soft_emb_norm = RMSNorm(vision_hidden_size, eps=config.vision_config.layer_norm_eps)
        
        # Projection to text space
        self.mm_input_projection = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)
        
        # Pooling setup
        patches_per_image = config.vision_config.image_size // config.vision_config.patch_size
        self.patches_per_image = patches_per_image
        self.tokens_per_side = int(config.mm_tokens_per_image ** 0.5)
        self.kernel_size = patches_per_image // self.tokens_per_side
    
    def __call__(self, x: mx.array) -> mx.array:
        # x shape: [batch, num_patches, hidden_size]
        b, l, d = x.shape
        
        # Simple average pooling to reduce patches
        # Reshape to 2D grid
        side = int(l ** 0.5)
        x = x.reshape(b, side, side, d)
        
        # Pool
        pooled = []
        for i in range(0, side, self.kernel_size):
            for j in range(0, side, self.kernel_size):
                patch = x[:, i:i+self.kernel_size, j:j+self.kernel_size, :]
                pooled.append(patch.mean(axis=(1, 2), keepdims=True))
        
        x = mx.concatenate(pooled, axis=1).squeeze(2)
        
        # Normalize and project
        x = self.mm_soft_emb_norm(x)
        x = self.mm_input_projection(x)
        
        return x


class DummyVisionModel(nn.Module):
    """Dummy vision model for testing."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        
    def __call__(self, pixel_values: mx.array) -> mx.array:
        # Return dummy features
        batch_size = pixel_values.shape[0]
        num_patches = self.config.num_patches
        return mx.zeros((batch_size, num_patches, self.embed_dim))


class MultimodalGemma(GemmaModel):
    """Gemma with multimodal support."""
    
    def __init__(self, args: MultimodalConfig):
        # Initialize base Gemma
        super().__init__(args)
        
        # Add vision components
        self.vision_tower = DummyVisionModel(args.vision_config)
        self.multi_modal_projector = MultiModalProjector(args)
        
        # Store config
        self.config = args
    
    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_token_index: int = 256000,  # Default image token
    ) -> mx.array:
        """Get embeddings combining text and vision inputs."""
        
        # Get text embeddings
        if input_ids is not None:
            text_embeds = self.embed_tokens(input_ids)
        else:
            batch_size = pixel_values.shape[0] if pixel_values is not None else 1
            text_embeds = mx.zeros((batch_size, 1, self.config.hidden_size))
        
        # Process vision if provided
        if pixel_values is not None:
            # Get vision features
            vision_features = self.vision_tower(pixel_values)
            vision_embeds = self.multi_modal_projector(vision_features)
            
            # Find image token positions
            if input_ids is not None:
                image_positions = input_ids == image_token_index
                
                # Insert vision embeddings at image token positions
                # This is simplified - in practice you'd handle multiple images
                if mx.any(image_positions):
                    # For simplicity, just concatenate
                    # In real implementation, you'd replace tokens properly
                    text_embeds = text_embeds  # Keep text for now
            
            return mx.concatenate([text_embeds, vision_embeds], axis=1)
        
        return text_embeds
    
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        pixel_values: Optional[mx.array] = None,
    ):
        # Get embeddings (text + vision if provided)
        if inputs_embeds is None:
            if pixel_values is not None:
                h = self.get_input_embeddings(inputs, pixel_values)
            else:
                h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds
            
        # Continue with standard Gemma forward pass
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        
        return self.lm_head(self.norm(h)), cache


class MultimodalGemmaForCausalLM(nn.Module):
    """Wrapper for multimodal Gemma."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.model = MultimodalGemma(config)
        self.config = config
    
    def __call__(
        self, 
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        cache=None,
    ):
        return self.model(input_ids, cache=cache, pixel_values=pixel_values)
    
    @property
    def layers(self):
        return self.model.layers
    
    @property 
    def head_dim(self):
        return self.model.head_dim
    
    @property
    def n_kv_heads(self):
        return self.model.n_kv_heads