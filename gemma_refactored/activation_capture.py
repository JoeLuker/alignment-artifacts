"""
Activation capture system for monitoring model internals during inference.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union
import mlx.core as mx


class ActivationStore:
    """Store for model activations with hierarchical naming."""
    
    def __init__(self):
        self.activations: Dict[str, Any] = {}
        self.enabled: bool = True
        self._capture_this_step = True  # Control capture per step

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
                print(f"Warning: Could not register activation '{name}'. Error: {e}")

    def disable(self):
        """Disable activation collection entirely."""
        self.enabled = False

    def enable(self):
        """Enable activation collection entirely."""
        self.enabled = True

    def get_captured_activations(self) -> Dict[str, Any]:
        """Get all activations captured so far in the current step."""
        return self.activations


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
    
    print(f"Preparing to save {len(activations)} activations for step {step}...")
    
    for name, tensor in activations.items():
        try:
            if isinstance(tensor, mx.array):
                mx.eval(tensor)  # Evaluate JUST before saving
                np_array = np.array(tensor)
            elif isinstance(tensor, (np.ndarray, list, tuple, int, float, bool, str)):
                np_array = np.array(tensor)  # Handles primitives and strings
            else:
                try:  # Attempt conversion for other types
                    np_array = np.array(tensor)
                except Exception:
                    print(f"Skipping unsupported activation type '{name}' (type: {type(tensor)})")
                    skipped_count += 1
                    continue

            numpy_activations[name] = np_array
            size_bytes = getattr(np_array, 'nbytes', 0)  # Estimate size safely
            total_size_mb += size_bytes / (1024 * 1024)
        except Exception as e:
            print(f"Error processing activation '{name}' (type: {type(tensor)}): {e}")
            error_count += 1

    if not numpy_activations:
        print(f"No numeric/convertible activations to save for step {step}.")
        return

    try:
        if compress:
            np.savez_compressed(filepath, **numpy_activations)
        else:
            np.savez(filepath, **numpy_activations)
        print(f"Saved {len(numpy_activations)} activations ({total_size_mb:.2f} MB) to {filepath}")
        if skipped_count > 0: 
            print(f"Skipped {skipped_count} non-convertible activations.")
        if error_count > 0: 
            print(f"Encountered errors processing {error_count} activations.")
    except Exception as e:
        print(f"Error saving activations to {filepath}: {e}")


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