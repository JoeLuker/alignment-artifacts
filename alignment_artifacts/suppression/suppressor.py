"""Suppression functionality for alignment artifacts."""

import numpy as np
import mlx.core as mx
from pathlib import Path
import json
from typing import Dict, Optional, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        self.mode = suppressor.mode
        self.proximity_threshold = suppressor.proximity_threshold
        self.distance_threshold = suppressor.distance_threshold
        self.decay_rate = suppressor.decay_rate

        # Normalize suppression vector for calculations
        self.suppression_vec_norm = mx.sqrt(mx.sum(self.suppression_vec ** 2))
        self.suppression_vec_normalized = self.suppression_vec / (self.suppression_vec_norm + 1e-8)
        
        # Get artifact centroid for sphere mode
        if self.mode == "sphere":
            self.artifact_centroid = suppressor.artifact_centroids.get(layer_idx)
        elif self.mode == "subspace":
            self.artifact_subspace = suppressor.artifact_subspaces.get(layer_idx)
            self.artifact_mean = suppressor.artifact_means.get(layer_idx)

        # Copy all attributes from the original MLP
        for attr in dir(mlp):
            if not attr.startswith("_") and attr != "__call__":
                try:
                    setattr(self, attr, getattr(mlp, attr))
                except:
                    pass

    def __call__(self, x):
        """Apply suppression during forward pass."""
        # Call original MLP
        output = self.mlp(x)

        # Apply suppression based on mode
        if self.scale > 0:
            # Convert to float32 for precision
            output_f32 = output.astype(mx.float32)
            
            # Reshape direction vector to match output dimensions
            if output.ndim == 3:
                direction_reshaped = self.suppression_vec.reshape(1, 1, -1)
                direction_normalized = self.suppression_vec_normalized.reshape(1, 1, -1)
            else:
                direction_reshaped = self.suppression_vec
                direction_normalized = self.suppression_vec_normalized

            if self.mode == "always":
                # Original behavior - always apply suppression
                suppressed = output_f32 - self.scale * direction_reshaped
                output = suppressed.astype(output.dtype)
                self.suppressor.intervention_count += 1
                
            elif self.mode == "proximity":
                # Only suppress if activation is close to artifact direction
                # Compute projection onto artifact direction
                output_norm = mx.sqrt(mx.sum(output_f32 ** 2, axis=-1, keepdims=True))
                output_normalized = output_f32 / (output_norm + 1e-8)
                
                # Cosine similarity (dot product of normalized vectors)
                cosine_sim = mx.sum(output_normalized * direction_normalized, axis=-1, keepdims=True)
                
                # Apply suppression only where similarity exceeds threshold
                mask = mx.abs(cosine_sim) > self.proximity_threshold
                suppression = self.scale * direction_reshaped
                
                # Apply masked suppression
                suppressed = mx.where(mask, output_f32 - suppression, output_f32)
                output = suppressed.astype(output.dtype)
                
                # Count actual interventions
                interventions = mx.sum(mask).item()
                if interventions > 0:
                    self.suppressor.intervention_count += interventions
                    
            elif self.mode == "adaptive":
                # Scale suppression by cosine similarity
                # Compute cosine similarity
                output_norm = mx.sqrt(mx.sum(output_f32 ** 2, axis=-1, keepdims=True))
                output_normalized = output_f32 / (output_norm + 1e-8)
                
                cosine_sim = mx.sum(output_normalized * direction_normalized, axis=-1, keepdims=True)
                
                # Scale suppression by absolute cosine similarity
                # (0 when orthogonal, full strength when aligned)
                adaptive_scale = self.scale * mx.abs(cosine_sim)
                suppressed = output_f32 - adaptive_scale * direction_reshaped
                output = suppressed.astype(output.dtype)
                
                # Count as intervention if any suppression was applied
                if mx.max(mx.abs(cosine_sim)) > 0.1:
                    self.suppressor.intervention_count += 1
                    
            elif self.mode == "sphere":
                # Distance-based suppression: repulsive sphere around artifact centroid
                if self.artifact_centroid is not None:
                    # Reshape centroid to match output dimensions
                    if output.ndim == 3:
                        centroid_reshaped = self.artifact_centroid.reshape(1, 1, -1)
                    else:
                        centroid_reshaped = self.artifact_centroid
                    
                    # Compute distance from artifact centroid
                    distance_vec = output_f32 - centroid_reshaped
                    distance = mx.sqrt(mx.sum(distance_vec ** 2, axis=-1, keepdims=True))
                    
                    # Apply repulsion if within threshold distance
                    mask = distance < self.distance_threshold
                    
                    # Compute repulsion direction (away from centroid)
                    # Normalize distance vector to get direction
                    distance_normalized = distance_vec / (distance + 1e-8)
                    
                    # Apply repulsion: push away from centroid
                    # Exponential decay: stronger repulsion when closer, exponential falloff
                    # Formula: scale * exp(-decay_rate * distance / threshold)
                    normalized_distance = distance / self.distance_threshold
                    repulsion_strength = self.scale * mx.exp(-self.decay_rate * normalized_distance)
                    repulsion = repulsion_strength * distance_normalized
                    
                    # Apply masked repulsion
                    suppressed = mx.where(mask, output_f32 + repulsion, output_f32)
                    output = suppressed.astype(output.dtype)
                    
                    # Count actual interventions
                    interventions = mx.sum(mask).item()
                    if interventions > 0:
                        self.suppressor.intervention_count += interventions
                else:
                    # No centroid available, fall back to always mode
                    suppressed = output_f32 - self.scale * direction_reshaped
                    output = suppressed.astype(output.dtype)
                    self.suppressor.intervention_count += 1
                    
            elif self.mode == "subspace":
                # Subspace projection suppression: Remove components in artifact subspace
                if self.artifact_subspace is not None and self.artifact_mean is not None:
                    # Reshape for proper dimensions
                    if output.ndim == 3:
                        batch_size, seq_len, hidden_dim = output.shape
                        output_flat = output_f32.reshape(-1, hidden_dim)  # (batch*seq, hidden)
                        mean_reshaped = self.artifact_mean.reshape(1, -1)  # (1, hidden)
                        subspace_reshaped = self.artifact_subspace  # (n_components, hidden)
                    else:
                        output_flat = output_f32
                        mean_reshaped = self.artifact_mean
                        subspace_reshaped = self.artifact_subspace
                    
                    # Center the activations
                    centered_output = output_flat - mean_reshaped
                    
                    # Project onto the artifact subspace
                    # projection = output @ subspace.T @ subspace  (removes artifact components)
                    projections = mx.matmul(centered_output, subspace_reshaped.T)  # (batch*seq, n_components)
                    artifact_components = mx.matmul(projections, subspace_reshaped)  # (batch*seq, hidden)
                    
                    # Remove the artifact components (subtract projection)
                    suppressed_flat = centered_output - self.scale * artifact_components
                    
                    # Add back the mean
                    suppressed_flat = suppressed_flat + mean_reshaped
                    
                    # Reshape back to original dimensions
                    if output.ndim == 3:
                        suppressed = suppressed_flat.reshape(batch_size, seq_len, hidden_dim)
                    else:
                        suppressed = suppressed_flat
                    
                    output = suppressed.astype(output.dtype)
                    
                    # Count interventions based on magnitude of removed components
                    intervention_magnitude = mx.mean(mx.sum(artifact_components ** 2, axis=-1))
                    if intervention_magnitude > 0.01:  # Threshold for meaningful intervention
                        self.suppressor.intervention_count += 1
                else:
                    # No subspace available, fall back to always mode
                    suppressed = output_f32 - self.scale * direction_reshaped
                    output = suppressed.astype(output.dtype)
                    self.suppressor.intervention_count += 1

            # Log interventions
            if self.suppressor.intervention_count <= 5:
                logger.debug(
                    f"✓ Applied {self.mode} suppression #{self.suppressor.intervention_count} "
                    f"at layer {self.layer_idx}"
                )
            elif self.suppressor.intervention_count % 50 == 0:
                logger.debug(
                    f"✓ Applied {self.mode} suppression #{self.suppressor.intervention_count} "
                    f"at layer {self.layer_idx}"
                )

        return output


class AlignmentArtifactSuppressor:
    """Suppressor that modifies MLP outputs to reduce alignment artifacts."""

    def __init__(
        self,
        activations_dir: Path,
        target_layers: Optional[List[int]] = None,
        categories: Optional[List[str]] = None,
        scale: float = 1.0,
        mode: str = "auto",  # "auto", "always", "proximity", "adaptive", "sphere", "subspace"
        proximity_threshold: float = 0.5,  # For proximity mode
        distance_threshold: float = 2.0,  # For sphere mode
        decay_rate: float = 2.0,  # For sphere mode exponential decay
        pca_components: int = 5,  # Number of PCA components for subspace mode
    ):

        self.scale = scale
        self.mode = mode
        self.proximity_threshold = proximity_threshold
        self.distance_threshold = distance_threshold
        self.decay_rate = decay_rate
        self.pca_components = pca_components
        self.target_layers = target_layers or [1, 2, 3, 4, 5, 6]
        self.suppression_vectors = {}
        self.artifact_centroids = {}  # For sphere mode
        self.artifact_subspaces = {}  # For subspace mode (PCA components)
        self.artifact_means = {}  # For subspace mode centering
        self.intervention_count = 0

        # Validate mode
        if mode not in ["auto", "always", "proximity", "adaptive", "sphere", "subspace"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'auto', 'always', 'proximity', 'adaptive', 'sphere', or 'subspace'")

        # Auto-select mode based on model sophistication if requested
        if mode == "auto":
            self.mode = self._auto_select_mode()
            logger.info(f"Auto-selected mode: {self.mode}")

        # Compute suppression vectors
        self._compute_vectors(activations_dir, categories)

        # Store original MLPs
        self.original_mlps = {}

    def _auto_select_mode(self) -> str:
        """
        Automatically select the optimal suppression mode based on model characteristics.
        Scales sophistication with model complexity.
        """
        # Try to determine model size from activations directory or config
        model_params = self._estimate_model_parameters()
        hidden_dim = self._estimate_hidden_dimension()
        
        logger.info(f"Estimated model parameters: {model_params:.1f}M, hidden_dim: {hidden_dim}")
        
        # Model sophistication tiers
        if model_params < 2000:  # < 2B parameters
            # Small models: Simple geometric patterns, use proximity mode
            # These models have clear, concentrated artifact patterns
            return "proximity"
            
        elif model_params < 10000:  # 2B - 10B parameters  
            # Medium models: More complex patterns, use adaptive mode
            # Patterns start becoming distributed but still manageable
            return "adaptive"
            
        elif model_params < 50000:  # 10B - 50B parameters
            # Large models: Complex distributed patterns, use sphere mode
            # Need distance-based intervention with exponential decay
            return "sphere"
            
        else:  # > 50B parameters
            # Very large models: Highly distributed patterns, use subspace mode
            # Require sophisticated multi-dimensional intervention
            return "subspace"

    def _estimate_model_parameters(self) -> float:
        """Estimate model parameters from model config or heuristics."""
        try:
            # Try to load model config from cache
            cache_dir = Path(str(self.activations_dir))
            config_file = cache_dir / "model_config.json"
            
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                # Calculate approximate parameters
                hidden_size = config.get("hidden_size", 1152)
                num_layers = config.get("num_hidden_layers", 24)
                vocab_size = config.get("vocab_size", 256000)
                intermediate_size = config.get("intermediate_size", hidden_size * 4)
                
                # Rough parameter estimation for transformer models
                # Embedding: vocab_size * hidden_size
                # Attention: 4 * hidden_size^2 per layer (Q, K, V, O projections)
                # MLP: 2 * hidden_size * intermediate_size per layer  
                # Layer norm: 2 * hidden_size per layer
                
                embedding_params = vocab_size * hidden_size
                attention_params = num_layers * 4 * hidden_size * hidden_size
                mlp_params = num_layers * 2 * hidden_size * intermediate_size
                norm_params = num_layers * 2 * hidden_size
                
                total_params = embedding_params + attention_params + mlp_params + norm_params
                
                return total_params / 1_000_000  # Convert to millions
                
        except Exception as e:
            logger.debug(f"Could not estimate parameters from config: {e}")
        
        # Fallback: estimate from hidden dimension
        hidden_dim = self._estimate_hidden_dimension()
        
        # Rough heuristic based on common model architectures
        if hidden_dim <= 1152:  # Like Gemma-1B
            return 1000  # ~1B parameters
        elif hidden_dim <= 2048:  # Like Gemma-2B  
            return 2000  # ~2B parameters
        elif hidden_dim <= 4096:  # Like Llama-7B
            return 7000  # ~7B parameters
        elif hidden_dim <= 5120:  # Like Llama-13B
            return 13000  # ~13B parameters
        elif hidden_dim <= 8192:  # Like Llama-70B
            return 70000  # ~70B parameters
        else:
            return 100000  # Very large model

    def _estimate_hidden_dimension(self) -> int:
        """Estimate hidden dimension from activation files."""
        try:
            # Look for any activation file to determine dimensions
            cache_dir = Path(str(self.activations_dir))
            
            # Find first batch directory
            batch_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])
            
            if batch_dirs:
                batch_dir = batch_dirs[0]
                batch_data_dir = batch_dir / batch_dir.name
                
                prefill_file = batch_data_dir / "activations_step_prefill.npz"
                if prefill_file.exists():
                    data = np.load(prefill_file)
                    
                    # Look for any MLP output to get hidden dimension
                    for key in data.files:
                        if "mlp.output" in key:
                            shape = data[key].shape
                            if len(shape) >= 3:  # (batch, seq, hidden)
                                return shape[-1]  # hidden dimension
                                
        except Exception as e:
            logger.debug(f"Could not estimate hidden dimension: {e}")
        
        # Default fallback
        return 1152  # Gemma-1B default

    def _compute_vectors(
        self, activations_dir: Path, categories: Optional[List[str]] = None
    ):
        """Compute suppression vectors from saved activations."""

        # Load metadata - try multiple locations
        metadata_path = None
        for path in [
            Path("prompts_metadata.json"),
            Path(__file__).parent.parent.parent / "prompts_metadata.json",
            Path.cwd() / "prompts_metadata.json",
        ]:
            if path.exists():
                metadata_path = path
                break

        if not metadata_path:
            raise FileNotFoundError("Could not find prompts_metadata.json")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        prompt_mappings = metadata["prompt_mappings"]

        # Filter by categories if specified
        if categories:
            prompt_mappings = [
                p for p in prompt_mappings if p["category"] in categories
            ]

        # Separate indices
        natural_indices = [
            p["index"] for p in prompt_mappings if p["type"] == "natural"
        ]
        artifact_indices = [
            p["index"] for p in prompt_mappings if p["type"] == "artifact"
        ]

        logger.info(
            f"Using {len(natural_indices)} natural and "
            f"{len(artifact_indices)} artifact prompts"
        )
        if categories:
            logger.info(f"Categories: {categories}")

        batch_dirs = sorted(
            [
                d
                for d in activations_dir.iterdir()
                if d.is_dir() and d.name.startswith("batch_")
            ]
        )
        logger.info(f"Found {len(batch_dirs)} batch directories in {activations_dir}")

        # First, load all activation files once
        loaded_activations = {}
        for batch_dir in batch_dirs:
            batch_num = int(batch_dir.name.split("_")[1]) - 1

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
                        logger.debug(
                            f"  Loaded {step_file.name} with {len(data.files)} keys"
                        )
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
                            vec = (
                                batch_data[local_idx, 0, :]
                                if batch_data.ndim == 3
                                else batch_data[local_idx, :]
                            )
                            all_natural.append(vec)

                for idx in artifact_indices:
                    if batch_offset <= idx < batch_offset + 100:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = (
                                batch_data[local_idx, 0, :]
                                if batch_data.ndim == 3
                                else batch_data[local_idx, :]
                            )
                            all_artifact.append(vec)

            logger.debug(
                f"Layer {layer}: collected {len(all_natural)} natural, {len(all_artifact)} artifact"
            )

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
                    
                    # For sphere mode, store the artifact centroid as the sphere center
                    if self.mode == "sphere":
                        self.artifact_centroids[layer] = mx.array(
                            artifact_mean, dtype=mx.float32
                        )
                        logger.info(
                            f"✓ Layer {layer}: computed suppression vector (norm={norm:.3f}) "
                            f"and artifact centroid"
                        )
                    elif self.mode == "subspace":
                        # PCA-based subspace analysis
                        logger.info(f"Computing PCA subspace for layer {layer}...")
                        
                        # Combine all data for better subspace identification
                        combined_data = np.vstack([all_natural, all_artifact])
                        
                        # Standardize the data
                        scaler = StandardScaler()
                        combined_scaled = scaler.fit_transform(combined_data)
                        
                        # Separate back into natural and artifact after scaling
                        n_natural = len(all_natural)
                        natural_scaled = combined_scaled[:n_natural]
                        artifact_scaled = combined_scaled[n_natural:]
                        
                        # Compute the difference space (artifact - natural for each sample)
                        # This captures the directions that distinguish artifacts from natural
                        if len(natural_scaled) == len(artifact_scaled):
                            # If we have paired data, use pairwise differences
                            difference_vectors = artifact_scaled - natural_scaled
                        else:
                            # Otherwise, use all pairwise differences (more comprehensive)
                            difference_vectors = []
                            for art in artifact_scaled:
                                for nat in natural_scaled:
                                    difference_vectors.append(art - nat)
                            difference_vectors = np.array(difference_vectors)
                        
                        # Apply PCA to find the principal directions of artifact space
                        pca = PCA(n_components=min(self.pca_components, difference_vectors.shape[1]))
                        pca.fit(difference_vectors)
                        
                        # Store the PCA components (principal artifact directions)
                        # These represent the multi-dimensional subspace of artifacts
                        components = pca.components_.astype(np.float32)  # Shape: (n_components, feature_dim)
                        self.artifact_subspaces[layer] = mx.array(components)
                        
                        # Store the mean for centering
                        artifact_mean_scaled = scaler.transform(artifact_mean.reshape(1, -1))[0]
                        self.artifact_means[layer] = mx.array(artifact_mean_scaled.astype(np.float32))
                        
                        # Store the traditional single vector as well for fallback
                        self.suppression_vectors[layer] = mx.array(direction, dtype=mx.float32)
                        
                        # Report explained variance
                        explained_var = pca.explained_variance_ratio_
                        total_var = np.sum(explained_var)
                        
                        logger.info(
                            f"✓ Layer {layer}: computed {self.pca_components}-component PCA subspace "
                            f"(explains {total_var:.1%} of artifact variance, "
                            f"top component: {explained_var[0]:.1%})"
                        )
                    else:
                        logger.info(
                            f"✓ Layer {layer}: computed suppression vector "
                            f"(norm={norm:.3f})"
                        )
            else:
                logger.warning(f"Layer {layer}: insufficient data")

    def patch_model(self, model):
        """Patch the model's MLP modules to apply suppression."""
        logger.info(f"Patching model with {self.mode} suppression vectors")
        logger.info(f"Target layers: {self.target_layers}")
        if self.mode == "proximity":
            logger.info(f"Proximity threshold: {self.proximity_threshold}")
        elif self.mode == "sphere":
            logger.info(f"Distance threshold: {self.distance_threshold}")
        logger.debug(
            f"Available suppression vectors: {list(self.suppression_vectors.keys())}"
        )

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
                    self,
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
