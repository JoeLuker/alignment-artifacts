"""Suppression functionality for alignment artifacts."""

import numpy as np
import mlx.core as mx
from pathlib import Path
import json
from typing import Dict, Optional, List
from ..utils.logging_config import get_logger

logger = get_logger("suppressor")


class MLPWrapper:
    """Wrapper that intercepts MLP calls to apply suppression."""
    
    def __init__(self, mlp, suppression_vec, scale, layer_idx, suppressor):
        self.mlp = mlp
        self.suppression_vec = suppression_vec
        self.scale = scale
        self.layer_idx = layer_idx
        self.suppressor = suppressor
        
        # Copy all attributes from the original MLP
        for attr in dir(mlp):
            if not attr.startswith('_') and attr != '__call__':
                try:
                    setattr(self, attr, getattr(mlp, attr))
                except:
                    pass
    
    def __call__(self, x):
        """Apply suppression during forward pass."""
        # Call original MLP
        output = self.mlp(x)
        
        # Apply suppression
        if self.scale > 0:
            # Convert to float32 for precision
            output_f32 = output.astype(mx.float32)
            
            # Apply suppression
            if output.ndim == 3:
                direction_reshaped = self.suppression_vec.reshape(1, 1, -1)
                suppressed = output_f32 - self.scale * direction_reshaped
            else:
                suppressed = output_f32 - self.scale * self.suppression_vec
            
            # Convert back
            output = suppressed.astype(output.dtype)
            
            self.suppressor.intervention_count += 1
            # Log first few and then every 50th intervention at DEBUG level
            if (self.suppressor.intervention_count <= 5):
                logger.debug(f"✓ Applied suppression #{self.suppressor.intervention_count} "
                           f"at layer {self.layer_idx}")
            elif (self.suppressor.intervention_count % 50 == 0):
                logger.debug(f"✓ Applied suppression #{self.suppressor.intervention_count} "
                           f"at layer {self.layer_idx}")
        
        return output


class AlignmentArtifactSuppressor:
    """Suppressor that modifies MLP outputs to reduce alignment artifacts."""
    
    def __init__(self, 
                 activations_dir: Path,
                 target_layers: Optional[List[int]] = None,
                 categories: Optional[List[str]] = None,
                 scale: float = 1.0):
        
        self.scale = scale
        self.target_layers = target_layers or [1, 2, 3, 4, 5, 6]
        self.suppression_vectors = {}
        self.intervention_count = 0
        
        # Compute suppression vectors
        self._compute_vectors(activations_dir, categories)
        
        # Store original MLPs
        self.original_mlps = {}
    
    def _compute_vectors(self, activations_dir: Path, 
                        categories: Optional[List[str]] = None):
        """Compute suppression vectors from saved activations."""
        
        # Load metadata - try multiple locations
        metadata_path = None
        for path in [
            Path("prompts_metadata.json"),
            Path(__file__).parent.parent.parent / "prompts_metadata.json",
            Path.cwd() / "prompts_metadata.json"
        ]:
            if path.exists():
                metadata_path = path
                break
        
        if not metadata_path:
            raise FileNotFoundError("Could not find prompts_metadata.json")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        prompt_mappings = metadata['prompt_mappings']
        
        # Filter by categories if specified
        if categories:
            prompt_mappings = [
                p for p in prompt_mappings 
                if p['category'] in categories
            ]
        
        # Separate indices
        natural_indices = [
            p['index'] for p in prompt_mappings 
            if p['type'] == 'natural'
        ]
        artifact_indices = [
            p['index'] for p in prompt_mappings 
            if p['type'] == 'artifact'
        ]
        
        logger.info(f"Using {len(natural_indices)} natural and "
                   f"{len(artifact_indices)} artifact prompts")
        if categories:
            logger.info(f"Categories: {categories}")
        
        batch_dirs = sorted([
            d for d in activations_dir.iterdir() 
            if d.is_dir() and d.name.startswith("batch_")
        ])
        logger.info(f"Found {len(batch_dirs)} batch directories in {activations_dir}")
        
        # First, load all activation files once
        loaded_activations = {}
        for batch_dir in batch_dirs:
            batch_num = int(batch_dir.name.split('_')[1]) - 1
            
            for step in range(20):  # num_steps
                # Handle nested batch directories
                step_file = batch_dir / batch_dir.name / f"activations_step_{step}.npz"
                if not step_file.exists():
                    step_file = batch_dir / f"activations_step_{step}.npz"
                if not step_file.exists():
                    continue
                    
                try:
                    data = np.load(step_file)
                    loaded_activations[(batch_num, step)] = data
                    if step == 0 and batch_num == 0:
                        logger.debug(f"  Loaded {step_file.name} with {len(data.files)} keys")
                except Exception as e:
                    logger.warning(f"Error loading {step_file}: {e}")
                    continue
        
        logger.debug(f"Loaded {len(loaded_activations)} activation files")
        
        for layer in self.target_layers:
            all_natural = []
            all_artifact = []
            
            # Collect activations from loaded data
            for (batch_num, step), data in loaded_activations.items():
                batch_offset = batch_num * 100
                key = f"model.layers.{layer}.mlp.output"
                
                if key not in data:
                    continue
                
                batch_data = data[key]
                
                # Extract activations
                for idx in natural_indices:
                    if batch_offset <= idx < batch_offset + 100:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = (batch_data[local_idx, 0, :] 
                                  if batch_data.ndim == 3 
                                  else batch_data[local_idx, :])
                            all_natural.append(vec)
                
                for idx in artifact_indices:
                    if batch_offset <= idx < batch_offset + 100:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = (batch_data[local_idx, 0, :] 
                                  if batch_data.ndim == 3 
                                  else batch_data[local_idx, :])
                            all_artifact.append(vec)
            
            logger.debug(f"Layer {layer}: collected {len(all_natural)} natural, {len(all_artifact)} artifact")
            
            if all_natural and all_artifact:
                # Compute direction (artifact - natural)
                natural_mean = np.mean(all_natural, axis=0).astype(np.float32)
                artifact_mean = np.mean(all_artifact, axis=0).astype(np.float32)
                direction = artifact_mean - natural_mean
                
                # Store with original magnitude
                norm = np.linalg.norm(direction)
                if norm > 0:
                    self.suppression_vectors[layer] = mx.array(
                        direction, dtype=mx.float32
                    )
                    logger.info(f"✓ Layer {layer}: computed suppression vector "
                              f"(norm={norm:.3f})")
            else:
                logger.warning(f"Layer {layer}: insufficient data")
    
    def patch_model(self, model):
        """Patch the model's MLP modules to apply suppression."""
        logger.info(f"Patching model with suppression vectors")
        logger.info(f"Target layers: {self.target_layers}")
        logger.debug(f"Available suppression vectors: {list(self.suppression_vectors.keys())}")
        
        for layer_idx in self.target_layers:
            if layer_idx < len(model.layers) and layer_idx in self.suppression_vectors:
                layer = model.layers[layer_idx]
                
                # Store original MLP
                self.original_mlps[layer_idx] = layer.mlp
                
                # Create wrapper
                wrapper = MLPWrapper(
                    layer.mlp,
                    self.suppression_vectors[layer_idx],
                    self.scale,
                    layer_idx,
                    self
                )
                
                # Replace MLP with wrapper
                layer.mlp = wrapper
                logger.info(f"✓ Patched layer {layer_idx}")
    
    def unpatch_model(self, model):
        """Restore original MLP modules."""
        for layer_idx, original_mlp in self.original_mlps.items():
            if layer_idx < len(model.layers):
                model.layers[layer_idx].mlp = original_mlp
        self.original_mlps.clear()