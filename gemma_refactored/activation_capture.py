"""
Activation capture system for monitoring model internals during inference.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union
import mlx.core as mx
import os
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('ACTIVATION_DEBUG', '').lower() == 'true' else logging.INFO)


class ActivationStore:
    """Store for model activations with hierarchical naming."""
    
    def __init__(self):
        self.activations: Dict[str, Any] = {}
        self.enabled: bool = True
        self._capture_this_step = True  # Control capture per step
        self.debug = os.environ.get('ACTIVATION_DEBUG', '').lower() == 'true'
        self._expected_mlp_layers = 26  # Expected number of MLP layers for 1B model

    def reset(self):
        """Clear all stored activations for the next step."""
        self.activations = {}
        self._capture_this_step = True  # Re-enable for next step by default

    def skip_capture_for_step(self):
        """Disable capture only for the current step's registrations."""
        self._capture_this_step = False

    def register(self, name: str, tensor: Any):
        """Register an activation tensor if enabled *for this step*."""
        if self.enabled and self._capture_this_step and tensor is not None:
            try:
                # Store the tensor directly. Evaluation happens during saving.
                self.activations[name] = tensor
            except Exception as e:
                logger.warning(f"Could not register activation '{name}'. Error: {e}")

    def disable(self):
        """Disable activation collection entirely."""
        self.enabled = False

    def enable(self):
        """Enable activation collection entirely."""
        self.enabled = True

    def get_captured_activations(self) -> Dict[str, Any]:
        """Get all activations captured so far in the current step."""
        if self.debug:
            self._check_invariants()
        return self.activations
    
    def _check_invariants(self):
        """Check invariants about captured activations when debug is enabled."""
        if not self.debug:
            return
            
        # Count MLP outputs
        mlp_outputs = [k for k in self.activations.keys() if 'mlp.output' in k]
        mlp_count = len(mlp_outputs)
        
        # Check we have the expected number of MLP outputs
        if mlp_count != self._expected_mlp_layers and mlp_count > 0:
            logger.warning(f"Expected {self._expected_mlp_layers} MLP outputs, found {mlp_count}")
            missing = set(range(self._expected_mlp_layers)) - {i for i in range(self._expected_mlp_layers) if f'model.layers.{i}.mlp.output' in self.activations}
            logger.warning(f"Missing layers: {missing}")
        
        # Check activation shapes and types
        for name, activation in self.activations.items():
            if 'mlp.output' in name and isinstance(activation, mx.array):
                # MLP outputs should have shape (batch, seq_len, hidden_size)
                if len(activation.shape) != 3:
                    logger.warning(f"MLP output {name} has unexpected shape: {activation.shape}")
                
                # Check for NaN/Inf
                if mx.any(mx.isnan(activation)) or mx.any(mx.isinf(activation)):
                    logger.warning(f"MLP output {name} contains NaN or Inf values!")
        
        # Log summary
        if self.debug:
            logger.debug(f"Captured {len(self.activations)} total activations")
            logger.debug(f"MLP outputs: {mlp_count}/{self._expected_mlp_layers}")
            
            # Count by type
            type_counts = {}
            for name in self.activations.keys():
                if '.mlp.' in name:
                    if '.output' in name:
                        key = 'mlp.output'
                    elif '.input' in name:
                        key = 'mlp.input'
                    else:
                        key = 'mlp.other'
                elif '.self_attn.' in name:
                    key = 'self_attn'
                elif 'layernorm' in name:
                    key = 'layernorm'
                else:
                    key = 'other'
                type_counts[key] = type_counts.get(key, 0) + 1
            
            logger.debug(f"Activation types: {type_counts}")


# Global activation store instance
activation_store = ActivationStore()


def save_activations(
    activations: Dict[str, Any], 
    output_dir: str, 
    step: Optional[Union[int, str]] = None, 
    compress: bool = True
):
    """Save activations to disk as NumPy arrays. Handles mx.array evaluation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"activations_step_{step}.npz" if step is not None else "activations.npz"
    filepath = output_path / filename
    
    numpy_activations = {}
    total_size_mb = 0
    skipped_count = 0
    error_count = 0
    debug = os.environ.get('ACTIVATION_DEBUG', '').lower() == 'true'
    
    logger.info(f"Preparing to save {len(activations)} activations for step {step}...")
    
    if debug:
        # Count MLP outputs before saving
        mlp_outputs_before = [k for k in activations.keys() if 'mlp.output' in k]
        logger.debug(f"MLP outputs before save: {len(mlp_outputs_before)}")
    
    for name, tensor in activations.items():
        try:
            if isinstance(tensor, mx.array):
                mx.eval(tensor)  # Evaluate JUST before saving
                # Handle bfloat16 by converting to float32 first
                if tensor.dtype == mx.bfloat16:
                    tensor = tensor.astype(mx.float32)
                np_array = np.array(tensor)
            elif isinstance(tensor, (np.ndarray, list, tuple, int, float, bool, str)):
                np_array = np.array(tensor)  # Handles primitives and strings
            else:
                try:  # Attempt conversion for other types
                    np_array = np.array(tensor)
                except Exception:
                    logger.debug(f"Skipping unsupported activation type '{name}' (type: {type(tensor)})")
                    skipped_count += 1
                    continue

            numpy_activations[name] = np_array
            size_bytes = getattr(np_array, 'nbytes', 0)  # Estimate size safely
            total_size_mb += size_bytes / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error processing activation '{name}' (type: {type(tensor)}): {e}")
            error_count += 1

    if not numpy_activations:
        logger.warning(f"No numeric/convertible activations to save for step {step}.")
        return

    try:
        if compress:
            np.savez_compressed(filepath, **numpy_activations)
        else:
            np.savez(filepath, **numpy_activations)
        logger.info(f"Saved {len(numpy_activations)} activations ({total_size_mb:.2f} MB) to {filepath}")
        if skipped_count > 0: 
            logger.info(f"Skipped {skipped_count} non-convertible activations.")
        if error_count > 0: 
            logger.warning(f"Encountered errors processing {error_count} activations.")
            
        if debug:
            # Count MLP outputs after saving
            mlp_outputs_after = [k for k in numpy_activations.keys() if 'mlp.output' in k]
            logger.debug(f"MLP outputs after save: {len(mlp_outputs_after)}")
            if len(mlp_outputs_after) < len(mlp_outputs_before):
                logger.warning(f"Lost MLP outputs during save!")
                missing = set(mlp_outputs_before) - set(mlp_outputs_after)
                logger.warning(f"Missing: {missing}")
    except Exception as e:
        logger.error(f"Error saving activations to {filepath}: {e}")


def scaled_dot_product_attention_with_activations(
    queries: mx.array, 
    keys: mx.array, 
    values: mx.array, 
    scale: float,
    mask: Optional[mx.array] = None, 
    attn_name: str = "attention"
) -> mx.array:
    """Computes SDPA and captures intermediate activations."""
    # QK^T
    scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale
    activation_store.register(f"{attn_name}.scores", scores)

    # Masking (assuming additive mask from create_attention_mask)
    if mask is not None:
        scores = scores + mask.astype(scores.dtype)
        activation_store.register(f"{attn_name}.scores_masked", scores)
    else:
        activation_store.register(f"{attn_name}.scores_masked", scores)  # Register raw scores if no mask

    # Softmax
    attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
    activation_store.register(f"{attn_name}.weights", attn_weights)

    # Weighted Values
    output = attn_weights @ values
    activation_store.register(f"{attn_name}.output_weighted_values", output)
    return output