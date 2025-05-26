"""Artifact analysis functionality."""

import numpy as np
from pathlib import Path
from typing import Dict, List
import json

from .metrics import compute_category_metrics


class ArtifactAnalyzer:
    """Analyzes alignment artifacts in model activations."""
    
    def __init__(self, metadata: Dict):
        self.metadata = metadata
        self.prompt_mappings = metadata['prompt_mappings']
        
        # Organize prompts by category
        self.categories = {}
        for mapping in self.prompt_mappings:
            cat = mapping['category']
            if cat not in self.categories:
                self.categories[cat] = {'natural': [], 'artifact': []}
            self.categories[cat][mapping['type']].append(mapping['index'])
    
    def load_model_config(self, activations_dir: Path) -> tuple:
        """Load model configuration from saved config file."""
        config_file = activations_dir / "model_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            return (
                config.get('num_hidden_layers', 26), 
                config.get('hidden_size', 1152)
            )
        else:
            # Try to infer from activation files
            return self._infer_config_from_activations(activations_dir)
    
    def _infer_config_from_activations(self, activations_dir: Path) -> tuple:
        """Infer model config from activation files."""
        batch_dirs = sorted([
            d for d in activations_dir.iterdir() 
            if d.is_dir() and d.name.startswith("batch_")
        ])
        
        if batch_dirs:
            # Check first activation file (handle nested batch directories)
            activation_files = list(batch_dirs[0].glob("*/activations_step_*.npz"))
            if not activation_files:
                activation_files = list(batch_dirs[0].glob("activations_step_*.npz"))
            
            for step_file in sorted(activation_files):
                data = np.load(step_file)
                # Find highest layer number
                max_layer = -1
                hidden_dim = None
                for key in data.keys():
                    if 'model.layers.' in key and '.mlp.output' in key:
                        layer_num = int(key.split('.')[2])
                        max_layer = max(max_layer, layer_num)
                        if hidden_dim is None:
                            hidden_dim = data[key].shape[-1]
                if max_layer >= 0:
                    return max_layer + 1, hidden_dim
        
        # Default fallback
        print("Warning: Could not determine model config, using defaults")
        return 26, 1152
    
    def analyze(self, activations_dir: Path) -> Dict:
        """Analyze activations and return results."""
        # Load model config
        num_layers, hidden_dim = self.load_model_config(activations_dir)
        
        # Find batch directories
        batch_dirs = sorted([
            d for d in activations_dir.iterdir() 
            if d.is_dir() and d.name.startswith("batch_")
        ])
        
        # Analyze each layer
        results = {'by_layer': {}, 'by_category': {}}
        
        # Quick analysis of key layers
        key_layers = list(range(min(10, num_layers)))  # First 10 layers
        
        for layer in key_layers:
            layer_results = {}
            
            for category, indices in self.categories.items():
                natural_acts = []
                artifact_acts = []
                
                # Collect activations
                for batch_dir in batch_dirs:
                    batch_num = int(batch_dir.name.split('_')[1]) - 1
                    batch_offset = batch_num * 100
                    
                    for step in range(20):
                        # Handle nested batch directories
                        step_file = batch_dir / batch_dir.name / f"activations_step_{step}.npz"
                        if not step_file.exists():
                            step_file = batch_dir / f"activations_step_{step}.npz"
                        if not step_file.exists():
                            continue
                        
                        data = np.load(step_file)
                        key = f"model.layers.{layer}.mlp.output"
                        
                        if key not in data:
                            continue
                        
                        batch_data = data[key]
                        
                        # Extract activations for this category
                        for idx in indices['natural']:
                            if batch_offset <= idx < batch_offset + 100:
                                local_idx = idx - batch_offset
                                if local_idx < batch_data.shape[0]:
                                    vec = (batch_data[local_idx, 0, :] 
                                          if batch_data.ndim == 3 
                                          else batch_data[local_idx, :])
                                    natural_acts.append(vec)
                        
                        for idx in indices['artifact']:
                            if batch_offset <= idx < batch_offset + 100:
                                local_idx = idx - batch_offset
                                if local_idx < batch_data.shape[0]:
                                    vec = (batch_data[local_idx, 0, :] 
                                          if batch_data.ndim == 3 
                                          else batch_data[local_idx, :])
                                    artifact_acts.append(vec)
                
                if natural_acts and artifact_acts:
                    metrics = compute_category_metrics(
                        np.array(natural_acts), 
                        np.array(artifact_acts)
                    )
                    if metrics:
                        layer_results[category] = metrics
            
            results['by_layer'][layer] = layer_results
        
        # Find best layers
        best_layers = {}
        for layer, cats in results['by_layer'].items():
            if cats:
                avg_d = np.mean([c['cohens_d'] for c in cats.values()])
                best_layers[layer] = avg_d
        
        sorted_layers = sorted(best_layers.items(), key=lambda x: x[1], reverse=True)
        results['best_layers'] = [l[0] for l in sorted_layers[:6]]  # Top 6 layers
        
        return results