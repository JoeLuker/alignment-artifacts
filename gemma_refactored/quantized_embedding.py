"""
Quantized embedding layer for pre-quantized embeddings.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class QuantizedEmbedding(nn.Module):
    """Embedding layer that works with pre-quantized weights."""
    
    def __init__(self, num_embeddings: int, dims: int, group_size: int = 64, bits: int = 4,
                 weight=None, scales=None, biases=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.group_size = group_size
        self.bits = bits
        
        # Calculate quantized dimensions
        self.n_groups = dims // group_size
        self.quantized_dims = dims // 8  # For 4-bit quantization
        
        # Initialize as parameters so MLX can track them
        if weight is not None:
            self.weight = weight
            self.scales = scales
            self.biases = biases
        else:
            # Initialize empty if not provided
            self.weight = mx.zeros((num_embeddings, self.quantized_dims))
            self.scales = mx.zeros((num_embeddings, self.n_groups))
            self.biases = mx.zeros((num_embeddings, self.n_groups))
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with dequantization."""
        # Get quantized embeddings (uint32)
        quantized = mx.take(self.weight, x, axis=0)
        
        # Get scales and biases for the selected embeddings
        scales = mx.take(self.scales, x, axis=0)
        biases = mx.take(self.biases, x, axis=0)
        
        # Unpack 4-bit values from uint32 using vectorized operations
        # Each uint32 contains 8 4-bit values
        batch_shape = quantized.shape[:-1]
        
        # Create shifts and masks for all 8 positions at once
        shifts = mx.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=mx.uint32)
        masks = mx.array([0xF] * 8, dtype=mx.uint32) << shifts
        
        # Expand quantized for broadcasting
        quantized_expanded = quantized[..., None]  # Add dimension for 8 values
        
        # Extract all 8 values at once
        unpacked = (quantized_expanded & masks) >> shifts
        
        # Reshape to flatten the unpacked dimension
        unpacked = unpacked.reshape(*batch_shape, -1)
        
        # Convert to float and normalize from [0, 15] to [-8, 7]
        unpacked = unpacked.astype(mx.float32) - 8.0
        
        # Reshape for per-group scaling
        unpacked = unpacked.reshape(*batch_shape, self.n_groups, self.group_size)
        
        # Apply scales and biases per group
        scales = scales.reshape(*batch_shape, self.n_groups, 1)
        biases = biases.reshape(*batch_shape, self.n_groups, 1)
        
        # Dequantize: output = unpacked * scale + bias
        dequantized = unpacked * scales + biases
        
        # Reshape back to original dimensions
        dequantized = dequantized.reshape(*batch_shape, self.dims)
        
        return dequantized
    
    def as_linear(self, x: mx.array) -> mx.array:
        """Use as a linear layer for tied embeddings."""
        # Unpack all embeddings using vectorized operations
        shifts = mx.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=mx.uint32)
        masks = mx.array([0xF] * 8, dtype=mx.uint32) << shifts
        
        # Expand weight for broadcasting
        weight_expanded = self.weight[..., None]
        
        # Extract all values at once
        unpacked = (weight_expanded & masks) >> shifts
        unpacked = unpacked.reshape(self.num_embeddings, -1)
        
        # Convert to float and normalize
        unpacked = unpacked.astype(mx.float32) - 8.0
        
        # Reshape for per-group scaling
        unpacked = unpacked.reshape(self.num_embeddings, self.n_groups, self.group_size)
        
        # Apply scales and biases
        scales = self.scales.reshape(self.num_embeddings, self.n_groups, 1)
        biases = self.biases.reshape(self.num_embeddings, self.n_groups, 1)
        
        dequantized = unpacked * scales + biases
        dequantized_weight = dequantized.reshape(self.num_embeddings, self.dims)
        
        # Use as linear transformation
        return x @ dequantized_weight.T