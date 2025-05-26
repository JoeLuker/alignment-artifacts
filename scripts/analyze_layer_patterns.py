#!/usr/bin/env python3
"""
Analyze which layers show strongest alignment artifact patterns.
Similar to the helper pattern analysis but for alignment artifacts.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple

def load_model_config_from_activations(activations_dir: Path) -> Tuple[int, int]:
    """Load model configuration from saved config file or infer from activations."""
    config_file = activations_dir / "model_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config.get('num_hidden_layers', 26), config.get('hidden_size', 1152)
    else:
        # Try to infer from activation files
        batch_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])
        if batch_dirs:
            # Check first activation file
            for step_file in sorted(batch_dirs[0].glob("activations_step_*.npz")):
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


def analyze_layer_patterns(
    activations_dir: Path = Path("./collected_activations_no_rep"),
    metadata_file: Path = Path("prompts_metadata.json")
) -> Tuple[List[int], Dict[int, Dict]]:
    """Find which layers show strongest separation between natural and artifact prompts."""
    
    # Load model configuration
    num_layers, expected_hidden_dim = load_model_config_from_activations(activations_dir)
    print(f"Model config: {num_layers} layers, hidden dim {expected_hidden_dim}")
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    prompt_mappings = metadata['prompt_mappings']
    natural_indices = [p['index'] for p in prompt_mappings if p['type'] == 'natural']
    artifact_indices = [p['index'] for p in prompt_mappings if p['type'] == 'artifact']
    
    print(f"Analyzing patterns across {num_layers} layers...")
    print(f"Natural prompts: {len(natural_indices)}, Artifact prompts: {len(artifact_indices)}")
    print("="*60)
    
    # Check all layers
    layer_scores = {}
    
    for layer in range(num_layers):
        print(f"\nAnalyzing layer {layer}...")
        
        # Load activations for this layer
        natural_states = []
        artifact_states = []
        
        # Process all batches
        batch_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])
        
        for batch_dir in batch_dirs:
            batch_num = int(batch_dir.name.split('_')[1]) - 1
            batch_offset = batch_num * 100  # batch size 100
            
            # Load all steps
            for step in range(20):
                step_file = batch_dir / f"activations_step_{step}.npz"
                if not step_file.exists():
                    continue
                
                data = np.load(step_file)
                key = f"model.layers.{layer}.mlp.output"
                
                if key not in data:
                    continue
                
                batch_data = data[key]
                
                # Extract natural prompts
                for idx in natural_indices:
                    if batch_offset <= idx < batch_offset + 100:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = batch_data[local_idx, 0, :] if batch_data.ndim == 3 else batch_data[local_idx, :]
                            natural_states.append(vec)
                
                # Extract artifact prompts
                for idx in artifact_indices:
                    if batch_offset <= idx < batch_offset + 100:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = batch_data[local_idx, 0, :] if batch_data.ndim == 3 else batch_data[local_idx, :]
                            artifact_states.append(vec)
        
        if natural_states and artifact_states:
            natural_array = np.array(natural_states)
            artifact_array = np.array(artifact_states)
            
            # Compute separation metrics
            nat_mean = natural_array.mean(axis=0)
            art_mean = artifact_array.mean(axis=0)
            direction = art_mean - nat_mean
            
            # Cohen's d
            nat_std = natural_array.std(axis=0).mean()
            art_std = artifact_array.std(axis=0).mean()
            pooled_std = np.sqrt((nat_std + art_std) / 2)
            cohens_d = np.linalg.norm(direction) / (pooled_std + 1e-8)
            
            # Test linear separability
            nat_proj = natural_array @ direction
            art_proj = artifact_array @ direction
            threshold = (nat_proj.mean() + art_proj.mean()) / 2
            
            # Classification accuracy
            nat_correct = (nat_proj < threshold).sum()
            art_correct = (art_proj > threshold).sum()
            accuracy = (nat_correct + art_correct) / (len(nat_proj) + len(art_proj))
            
            layer_scores[layer] = {
                'cohens_d': float(cohens_d),
                'accuracy': float(accuracy),
                'direction_norm': float(np.linalg.norm(direction)),
                'mean_separation': float(art_proj.mean() - nat_proj.mean()),
                'n_samples': len(natural_states) + len(artifact_states)
            }
            
            print(f"  Cohen's d: {cohens_d:.3f}")
            print(f"  Classification accuracy: {accuracy:.1%}")
            print(f"  Direction norm: {np.linalg.norm(direction):.3f}")
    
    # Find top layers
    print("\n" + "="*60)
    print("SUMMARY - Top layers by Cohen's d:")
    print("="*60)
    
    sorted_layers = sorted(layer_scores.items(), 
                          key=lambda x: x[1]['cohens_d'], 
                          reverse=True)
    
    top_layers = []
    for layer, scores in sorted_layers[:10]:
        print(f"Layer {layer:2d}: d={scores['cohens_d']:6.3f}, "
              f"acc={scores['accuracy']:.1%}, "
              f"sep={scores['mean_separation']:6.3f}")
        if scores['cohens_d'] > 1.0:  # Strong effect
            top_layers.append(layer)
    
    # Visualize
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Cohen's d by layer
    plt.subplot(2, 2, 1)
    layers = list(range(num_layers))
    cohens_ds = [layer_scores.get(l, {}).get('cohens_d', 0) for l in layers]
    
    bars = plt.bar(layers, cohens_ds, 
                    color=['red' if l in top_layers else 'blue' for l in layers])
    plt.xlabel('Layer')
    plt.ylabel("Cohen's d")
    plt.title('Alignment Artifact Strength by Layer')
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Strong effect')
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    plt.legend()
    
    # Plot 2: Accuracy by layer
    plt.subplot(2, 2, 2)
    accuracies = [layer_scores.get(l, {}).get('accuracy', 0.5) for l in layers]
    plt.bar(layers, accuracies, 
            color=['red' if l in top_layers else 'blue' for l in layers])
    plt.xlabel('Layer')
    plt.ylabel('Classification Accuracy')
    plt.title('Linear Separability by Layer')
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.5)
    plt.ylim(0.4, 1.0)
    
    # Plot 3: Effect size vs accuracy scatter
    plt.subplot(2, 2, 3)
    for layer, scores in layer_scores.items():
        color = 'red' if layer in top_layers else 'blue'
        plt.scatter(scores['cohens_d'], scores['accuracy'], 
                   s=50, color=color, alpha=0.7)
        if layer in top_layers:
            plt.annotate(f'L{layer}', 
                        (scores['cohens_d'], scores['accuracy']),
                        xytext=(2, 2), textcoords='offset points', 
                        fontsize=8)
    
    plt.xlabel("Cohen's d")
    plt.ylabel('Classification Accuracy')
    plt.title('Effect Size vs Detectability')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Top layers detail
    plt.subplot(2, 2, 4)
    top_n = min(8, len(sorted_layers))
    top_layer_nums = [l[0] for l in sorted_layers[:top_n]]
    top_cohen_ds = [l[1]['cohens_d'] for l in sorted_layers[:top_n]]
    
    plt.barh(range(top_n), top_cohen_ds, color='darkred')
    plt.yticks(range(top_n), [f'Layer {l}' for l in top_layer_nums])
    plt.xlabel("Cohen's d")
    plt.title(f'Top {top_n} Layers by Effect Size')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_fig = Path('results/figures/layer_analysis_alignment_artifacts.png')
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=150)
    print(f"\nVisualization saved to: {output_fig}")
    
    # Save detailed results
    results = {
        'layer_scores': layer_scores,
        'top_layers': top_layers,
        'sorted_layers': [(l, s) for l, s in sorted_layers]
    }
    
    output_json = Path('results/data/layer_analysis_results.json')
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_json}")
    
    return top_layers, layer_scores


def compute_multi_layer_directions(
    layers: List[int],
    activations_dir: Path = Path("./collected_activations_no_rep"),
    metadata_file: Path = Path("prompts_metadata.json"),
    categories: List[str] = None
) -> Dict[int, np.ndarray]:
    """Compute suppression directions for multiple layers."""
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    prompt_mappings = metadata['prompt_mappings']
    
    # Filter by categories if specified
    if categories:
        prompt_mappings = [p for p in prompt_mappings if p['category'] in categories]
        print(f"Filtering to categories: {categories}")
    
    natural_indices = [p['index'] for p in prompt_mappings if p['type'] == 'natural']
    artifact_indices = [p['index'] for p in prompt_mappings if p['type'] == 'artifact']
    
    print(f"\nComputing directions for layers: {layers}")
    print(f"Using {len(natural_indices)} natural and {len(artifact_indices)} artifact prompts")
    
    directions = {}
    
    for layer in layers:
        natural_states = []
        artifact_states = []
        
        # Load activations
        batch_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])
        
        for batch_dir in batch_dirs:
            batch_num = int(batch_dir.name.split('_')[1]) - 1
            batch_offset = batch_num * 100
            
            for step in range(20):
                step_file = batch_dir / f"activations_step_{step}.npz"
                if not step_file.exists():
                    continue
                
                data = np.load(step_file)
                key = f"model.layers.{layer}.mlp.output"
                
                if key not in data:
                    continue
                
                batch_data = data[key]
                
                # Extract states
                for idx in natural_indices:
                    if batch_offset <= idx < batch_offset + 100:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = batch_data[local_idx, 0, :] if batch_data.ndim == 3 else batch_data[local_idx, :]
                            natural_states.append(vec)
                
                for idx in artifact_indices:
                    if batch_offset <= idx < batch_offset + 100:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = batch_data[local_idx, 0, :] if batch_data.ndim == 3 else batch_data[local_idx, :]
                            artifact_states.append(vec)
        
        if natural_states and artifact_states:
            nat_array = np.array(natural_states)
            art_array = np.array(artifact_states)
            
            # Compute direction
            direction = art_array.mean(axis=0) - nat_array.mean(axis=0)
            directions[layer] = direction
            
            print(f"  Layer {layer}: direction norm = {np.linalg.norm(direction):.3f}")
    
    # Save directions
    output_npz = Path('saved_models/suppression_vectors/multi_layer_directions.npz')
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, 
             **{f'layer_{l}': d for l, d in directions.items()})
    
    metadata = {
        'layers': layers,
        'categories': categories,
        'norms': {l: float(np.linalg.norm(d)) for l, d in directions.items()}
    }
    
    output_meta = Path('saved_models/suppression_vectors/multi_layer_directions_metadata.json')
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(output_meta, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDirections saved to: {output_npz}")
    print(f"Metadata saved to: {output_meta}")
    
    return directions


if __name__ == "__main__":
    # Step 1: Find which layers matter most
    print("STEP 1: Discovering alignment artifact layers...")
    top_layers, layer_scores = analyze_layer_patterns()
    
    # Step 2: Compute directions for top layers
    if top_layers:
        print(f"\n\nSTEP 2: Computing directions for top {len(top_layers)} layers...")
        directions = compute_multi_layer_directions(top_layers[:6])  # Use top 6 layers
        
        print(f"\n\n✅ Analysis complete!")
        print(f"Top layers identified: {top_layers[:6]}")
        print("\nNext steps:")
        print("1. Use these directions in inference_with_suppression_simple.py")
        print("2. Or load them with: np.load('multi_layer_directions.npz')")
    else:
        print("\n❌ No layers with strong effects found!")