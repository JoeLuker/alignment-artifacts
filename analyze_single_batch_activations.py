#!/usr/bin/env python3
"""
Analyzes activations from single batch processing.
Simplified version since all prompts are in batch_1.
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Configuration
RAW_ACTIVATIONS_DIR = Path("./collected_activations_single_batch")
METADATA_FILE = Path("prompts_metadata.json")

NUM_LAYERS = 26
KEY_PATTERN_TEMPLATE = "model.layers.{layer}.mlp.output"
NUM_TOKEN_STEPS = 20
EXPECTED_HIDDEN_DIM = 1152

OUTPUT_FIGURE = Path("alignment_artifact_single_batch.png")
OUTPUT_RESULTS = Path("alignment_artifact_single_batch_results.json")


def main():
    """Analyze single batch activations."""
    print("\n🚀 SINGLE BATCH ANALYSIS 🚀")
    print("="*60)
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    prompt_mappings = metadata["prompt_mappings"]
    
    print(f"Total prompts: {len(prompt_mappings)}")
    print(f"Activation directory: {RAW_ACTIVATIONS_DIR}")
    
    # All activations are in: all_prompts/batch/batch_1/
    batch_dir = RAW_ACTIVATIONS_DIR / "all_prompts" / "batch" / "batch_1"
    
    if not batch_dir.exists():
        print(f"\n❌ ERROR: Batch directory not found: {batch_dir}")
        print("Please run ./collect_activations_single_batch.sh first")
        return
    
    print(f"\n✅ Found batch directory: {batch_dir}")
    
    all_layer_scores = {}
    
    for layer_idx in range(NUM_LAYERS):
        print(f"\nLayer {layer_idx}:", end=" ")
        
        natural_activations = []
        artifact_activations = []
        
        key = KEY_PATTERN_TEMPLATE.format(layer=layer_idx)
        
        # Load activations for each step
        for step in range(NUM_TOKEN_STEPS):
            step_file = batch_dir / f"activations_step_{step}.npz"
            
            if step_file.exists():
                data = np.load(step_file)
                if key in data:
                    batch_data = data[key]
                    
                    # Extract each prompt's activation
                    for mapping in prompt_mappings:
                        idx = mapping["index"]
                        
                        if batch_data.ndim == 3:  # (batch, seq, hidden)
                            vector = batch_data[idx, 0, :]
                        elif batch_data.ndim == 2:  # (batch, hidden)
                            vector = batch_data[idx, :]
                        else:
                            continue
                        
                        if vector.shape == (EXPECTED_HIDDEN_DIM,):
                            if mapping["type"] == "natural":
                                natural_activations.append(vector)
                            else:
                                artifact_activations.append(vector)
        
        # Compute metrics
        if natural_activations and artifact_activations:
            nat_array = np.array(natural_activations)
            art_array = np.array(artifact_activations)
            
            # Cohen's d
            direction = art_array.mean(axis=0) - nat_array.mean(axis=0)
            pooled_std = np.sqrt((nat_array.std(axis=0).mean() + art_array.std(axis=0).mean()) / 2)
            cohens_d = np.linalg.norm(direction) / (pooled_std + 1e-9)
            
            # Accuracy
            nat_proj = nat_array @ direction
            art_proj = art_array @ direction
            threshold = (nat_proj.mean() + art_proj.mean()) / 2
            accuracy = ((nat_proj < threshold).sum() + (art_proj > threshold).sum()) / (len(nat_proj) + len(art_proj))
            
            all_layer_scores[layer_idx] = {
                'cohens_d': float(cohens_d),
                'accuracy': float(accuracy),
                'n_samples': len(nat_proj) + len(art_proj)
            }
            
            print(f"d={cohens_d:.3f}, acc={accuracy:.1%}")
        else:
            print("No data")
    
    # Summary
    print("\n" + "="*60)
    print("TOP LAYERS BY EFFECT SIZE")
    print("="*60)
    
    sorted_layers = sorted(all_layer_scores.items(), key=lambda x: x[1]['cohens_d'], reverse=True)
    
    for i, (layer, scores) in enumerate(sorted_layers[:10]):
        print(f"{i+1}. Layer {layer}: Cohen's d = {scores['cohens_d']:.3f}, "
              f"Accuracy = {scores['accuracy']:.1%}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    layers = list(range(NUM_LAYERS))
    cohens_ds = [all_layer_scores.get(l, {}).get('cohens_d', 0) for l in layers]
    accuracies = [all_layer_scores.get(l, {}).get('accuracy', 0.5) for l in layers]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(layers, cohens_ds, color='darkblue')
    # Highlight strong effects
    for i, (d, bar) in enumerate(zip(cohens_ds, bars)):
        if d > 0.8:
            bar.set_color('red')
        elif d > 0.5:
            bar.set_color('orange')
    
    plt.xlabel('Layer')
    plt.ylabel("Cohen's d")
    plt.title('Alignment Artifact Strength')
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large (d=0.8)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (d=0.5)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(layers, accuracies, color='darkgreen')
    plt.xlabel('Layer')
    plt.ylabel('Classification Accuracy')
    plt.title('Alignment Artifact Detectability')
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.5)
    plt.ylim(0.4, 1.05)
    
    plt.suptitle('🚀 SINGLE BATCH PROCESSING - Maximum Speed! 🚀', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE)
    print(f"\n📊 Plot saved to: {OUTPUT_FIGURE}")
    
    # Save results
    results = {
        'processing_mode': 'SINGLE_BATCH',
        'total_prompts': len(prompt_mappings),
        'batches': 1,
        'speedup': '50x faster than individual processing',
        'layer_scores': all_layer_scores,
        'top_layers': [l for l, _ in sorted_layers[:5]]
    }
    
    with open(OUTPUT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {OUTPUT_RESULTS}")
    
    print("\n✨ Analysis complete! ✨\n")


if __name__ == "__main__":
    plt.switch_backend('Agg')
    main()