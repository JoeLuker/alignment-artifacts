"""
Model loading utilities for Gemma models.
"""

import json
import inspect
from pathlib import Path
from typing import Dict, Any, Tuple, Type
from dataclasses import dataclass

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def load_config(model_path: Path) -> Dict[str, Any]:
    """Load the model configuration from a JSON file."""
    # Try multiple common config names
    config_files = ["config.json", "model_config.json", "params.json"]
    for filename in config_files:
        config_path = model_path / filename
        if config_path.is_file():
            try:
                with open(config_path, "r") as f:
                    print(f"Loaded configuration from {filename}")
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}. Error: {e}")
                continue  # Try next file
    raise FileNotFoundError(f"No config file ({', '.join(config_files)}) found in {model_path}")


def get_model_path(path_or_hf_repo: str) -> Path:
    """
    Convert a string path to a Path object or download from Hugging Face if needed.
    """
    model_path = Path(path_or_hf_repo)

    # If the path doesn't exist locally, try to download from Hugging Face
    if not model_path.exists():
        if snapshot_download is None:
            raise ImportError(
                "huggingface_hub is required for downloading models. "
                "Please install it with `pip install huggingface_hub`."
            )
        try:
            print(f"Downloading {path_or_hf_repo} from Hugging Face Hub...")
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
            print(f"Download complete. Model saved to {model_path}")
        except Exception as e:
            raise ValueError(
                f"Failed to download model from Hugging Face: {path_or_hf_repo}. "
                f"Error: {str(e)}\n"
                "If you're trying to access a private repository, "
                "make sure you're authenticated with `huggingface-cli login`."
            )
    
    # Basic check after resolving path
    if not list(model_path.glob("*.safetensors")):
        print(f"Warning: No *.safetensors files found in {model_path}. Weight loading might fail.")
    return model_path


@dataclass
class BaseModelArgs:
    """Base class for model arguments."""
    
    @classmethod
    def from_dict(cls, params):
        # Keep original filtering logic
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def _get_classes(config: Dict[str, Any]) -> Tuple[Type, Type]:
    """
    Get the model and model args classes based on the model type.
    """
    # Check if it's a multimodal model
    is_multimodal = (
        "vision_config" in config or 
        "text_config" in config or
        "gemma3forconditionalgenerationconfig" in str(config.get("architectures", [])).lower()
    )
    
    if is_multimodal:
        # For now, use standard Gemma for multimodal models
        # TODO: Implement proper multimodal support
        print("Note: Using standard Gemma implementation for multimodal model")
        # Don't update config here - we'll handle it in the model loading
        from .gemma_models import Model, ModelArgs
        return Model, ModelArgs
    
    # Check for standard Gemma variants
    model_type = config.get("model_type", "gemma")
    if model_type in ["gemma", "gemma2"]:
        from .gemma_models import Model, ModelArgs
        return Model, ModelArgs
    
    # Default to original implementation
    from .model_architecture import Model, ModelArgs
    return Model, ModelArgs


def load_tokenizer(model_path: Path, config_extra={}, eos_token_ids=None):
    """Load tokenizer from model path."""
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for tokenizer loading. "
            "Please install it with `pip install transformers`."
        )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            **config_extra
        )
        
        # Handle special EOS token configuration
        if eos_token_ids is not None:
            if hasattr(tokenizer, 'eos_token_id'):
                original_eos = tokenizer.eos_token_id
                tokenizer.eos_token_id = eos_token_ids[0] if isinstance(eos_token_ids, list) else eos_token_ids
                print(f"Updated tokenizer EOS token ID from {original_eos} to {tokenizer.eos_token_id}")
        
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_path}: {e}")
        return None


def load_model_weights(model, model_path: Path, config: dict):
    """Load model weights from safetensors files."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten, tree_flatten
    
    weight_files = list(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    
    print(f"Loading weights from {len(weight_files)} file(s)...")
    weights = {}
    
    for weight_file in weight_files:
        file_weights = mx.load(str(weight_file))
        weights.update(file_weights)
    
    # Handle multimodal model weight prefixes
    if any(k.startswith("language_model.") for k in weights.keys()):
        print("Detected multimodal model weights, remapping keys...")
        remapped_weights = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                # Remove the language_model. prefix
                new_key = k.replace("language_model.", "", 1)
                remapped_weights[new_key] = v
            else:
                remapped_weights[k] = v
        weights = remapped_weights
    
    # Handle quantization if present
    quantization = config.get("quantization")
    if quantization and isinstance(quantization, dict):
        print(f"Applying quantization: {quantization}")
        
        def class_predicate(p, m):
            # Check if layer is quantizable and has required weight components
            is_quantizable = isinstance(m, (nn.Linear, nn.Embedding))
            has_scales = f"{p}.scales" in weights
            has_biases = f"{p}.biases" in weights
            
            # Special handling for multimodal embeddings
            if "embed_tokens" in p and ("language_model" in p or "text_config" in config):
                # The embedding might be pre-quantized in multimodal models
                print(f"Note: Quantized embeddings detected, keeping original weight shape")
                return False  # Don't re-quantize already quantized embeddings
            
            return is_quantizable and has_scales and has_biases
        
        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )
    else:
        print("No quantization config found or 'quantization' not a dict.")
    
    # Sanitize weights using model's sanitize method
    if hasattr(model, 'sanitize'):
        weights = model.sanitize(weights)
    
    # Additional sanitization for multimodal models
    if "vision_config" in config or "text_config" in config:
        # Remove actual vision implementation weights since we use dummy vision
        vision_keys = [k for k in weights.keys() if "vision_tower" in k and "embed" in k]
        projector_keys = [k for k in weights.keys() if "multi_modal_projector" in k]
        
        # Keep projector weights but remove vision embeddings
        for k in vision_keys:
            if k in weights:
                del weights[k]
        
        if vision_keys:
            print(f"Note: Removed {len(vision_keys)} vision embedding weights (using dummy vision tower)")
        
        # Log what we're keeping
        remaining_vision_keys = [k for k in weights.keys() if "vision" in k or "projector" in k]
        if remaining_vision_keys:
            print(f"Keeping {len(remaining_vision_keys)} multimodal component weights")
    
    # Filter weights to match model parameters (after potential quantization)
    final_params = dict(tree_flatten(model.parameters()))
    filtered_weights = {k: v for k, v in weights.items() if k in final_params}
    ignored_weights = {k: v for k, v in weights.items() if k not in final_params}
    
    if ignored_weights:
        print(f"Warning: Ignoring {len(ignored_weights)} weight(s) not found in model parameters: {list(ignored_weights.keys())[:5]}...")
    
    if not filtered_weights:
        raise ValueError("No matching weights found between loaded files and model parameters. Check weight keys and model structure.")
    
    # Load filtered weights
    model.load_weights(list(filtered_weights.items()))
    print(f"Successfully loaded {len(filtered_weights)} weight tensors")
    model.eval()
    return model