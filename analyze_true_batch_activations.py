#!/usr/bin/env python3
"""
Analyzes activations from true batch processing.
Uses the flat prompt structure with separate metadata.
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# --- Configuration ---
RAW_ACTIVATIONS_BASE_DIR = Path("./collected_activations_true_batch")
METADATA_FILE = Path("prompts_metadata.json")
BATCH_SIZE = 50  # Single batch with ALL prompts

NUM_LAYERS = 26
KEY_PATTERN_TEMPLATE = "model.layers.{layer}.mlp.output"
NUM_TOKEN_STEPS_TO_COLLECT = 20
EXPECTED_HIDDEN_DIM = 1152

OUTPUT_FIGURE_PATH = Path("alignment_artifact_analysis_true_batch.png")
OUTPUT_RESULTS_PATH = Path("alignment_artifact_analysis_true_batch_results.json")
# --- End Configuration ---


def load_metadata(metadata_path: Path) -> Dict:
    """Load prompt metadata."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def extract_batch_activations(
    activations_dir: Path,
    batch_idx: int,
    prompt_indices: List[int],
    layer_idx: int,
    num_steps: int
) -> Dict[int, List[np.ndarray]]:
    """
    Extract activations for specific prompts from a batch.
    
    Returns:
        Dict mapping prompt index to list of activation vectors
    """
    activations_by_prompt = {idx: [] for idx in prompt_indices}
    key_to_load = KEY_PATTERN_TEMPLATE.format(layer=layer_idx)
    
    # The batch activations are stored in: all_prompts/batch/batch_N/
    batch_dir = activations_dir / "all_prompts" / "batch" / f"batch_{batch_idx}"
    
    if not batch_dir.exists():
        print(f"  Warning: Batch directory not found: {batch_dir}")
        return activations_by_prompt
    
    for step_idx in range(num_steps):
        step_file = batch_dir / f"activations_step_{step_idx}.npz"
        if step_file.exists():
            try:
                data = np.load(step_file)
                if key_to_load in data:
                    batch_activations = data[key_to_load]
                    
                    # Extract activations for each prompt in this batch
                    for prompt_idx in prompt_indices:
                        batch_position = prompt_idx % BATCH_SIZE
                        
                        if batch_activations.ndim == 3:
                            # (batch_size, seq_len, hidden_dim)
                            if batch_position < batch_activations.shape[0]:
                                vector = batch_activations[batch_position, 0, :]
                            else:
                                continue
                        elif batch_activations.ndim == 2:
                            # (batch_size, hidden_dim)
                            if batch_position < batch_activations.shape[0]:
                                vector = batch_activations[batch_position, :]
                            else:
                                continue
                        else:
                            print(f"  Warning: Unexpected shape {batch_activations.shape}")
                            continue
                        
                        if vector.shape == (EXPECTED_HIDDEN_DIM,):
                            activations_by_prompt[prompt_idx].append(vector)
            except Exception as e:
                print(f"  Error loading {step_file}: {e}")
    
    return activations_by_prompt


def main_analysis():
    """Main analysis for true batch processing."""
    # Load metadata
    if not METADATA_FILE.exists():
        print(f"ERROR: Metadata file not found: {METADATA_FILE}")
        return
    
    metadata = load_metadata(METADATA_FILE)
    prompt_mappings = metadata["prompt_mappings"]
    
    print("\n--- Starting Analysis of True Batch Activations ---")
    print(f"Reading activations from: {RAW_ACTIVATIONS_BASE_DIR}")
    print(f"Using metadata from: {METADATA_FILE}")
    print(f"Total prompts: {len(prompt_mappings)}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*80)
    
    all_layer_scores = {}
    
    for layer_idx in range(NUM_LAYERS):
        print(f"\nAnalyzing Layer {layer_idx}...")
        
        # Organize prompts by batch
        batches = {}
        for mapping in prompt_mappings:
            prompt_idx = mapping["index"]
            batch_idx = prompt_idx // BATCH_SIZE + 1  # batch_1, batch_2, etc.
            
            if batch_idx not in batches:
                batches[batch_idx] = []
            batches[batch_idx].append(prompt_idx)
        
        # Collect activations
        all_natural_states = []
        all_artifact_states = []
        
        for batch_idx, prompt_indices in batches.items():
            # Extract activations for all prompts in this batch
            batch_activations = extract_batch_activations(
                RAW_ACTIVATIONS_BASE_DIR,
                batch_idx,
                prompt_indices,
                layer_idx,
                NUM_TOKEN_STEPS_TO_COLLECT
            )
            
            # Sort activations by type
            for prompt_idx, activations in batch_activations.items():
                if activations:  # Only if we got activations
                    mapping = prompt_mappings[prompt_idx]
                    if mapping["type"] == "natural":
                        all_natural_states.extend(activations)
                    else:  # artifact
                        all_artifact_states.extend(activations)
        
        # Analyze if we have data
        if all_natural_states and all_artifact_states:
            natural_array = np.array(all_natural_states)
            artifact_array = np.array(all_artifact_states)
            
            print(f"  Layer {layer_idx}: {natural_array.shape[0]} natural, "
                  f"{artifact_array.shape[0]} artifact activation vectors")
            
            # Calculate metrics
            nat_mean = natural_array.mean(axis=0)
            artifact_mean = artifact_array.mean(axis=0)
            direction_vector = artifact_mean - nat_mean
            
            # Cohen's d
            std_nat = natural_array.std(axis=0).mean()
            std_art = artifact_array.std(axis=0).mean()
            pooled_std = np.sqrt((std_nat + std_art) / 2)
            cohens_d = np.linalg.norm(direction_vector) / (pooled_std + 1e-9)
            
            # Classification accuracy
            natural_projections = natural_array @ direction_vector
            artifact_projections = artifact_array @ direction_vector
            
            threshold = (natural_projections.mean() + artifact_projections.mean()) / 2
            correct_natural = (natural_projections < threshold).sum()
            correct_artifact = (artifact_projections > threshold).sum()
            total = len(natural_projections) + len(artifact_projections)
            accuracy = (correct_natural + correct_artifact) / total
            
            all_layer_scores[layer_idx] = {
                'cohens_d': float(cohens_d),
                'accuracy': float(accuracy),
                'direction_norm': float(np.linalg.norm(direction_vector)),
                'mean_separation': float(artifact_projections.mean() - natural_projections.mean()),
                'num_natural': natural_array.shape[0],
                'num_artifact': artifact_array.shape[0]
            }
            
            print(f"    Cohen's d: {cohens_d:.3f}, Accuracy: {accuracy:.1%}")
    
    # Visualization and summary
    if not all_layer_scores:
        print("\nNo scores computed. Check activation collection.")
        return
    
    print("\n" + "="*80)
    print("SUMMARY - Top layers by Cohen's d:")
    print("="*80)
    
    sorted_layers = sorted(all_layer_scores.items(), 
                          key=lambda x: x[1]['cohens_d'], 
                          reverse=True)
    
    for layer_idx, scores in sorted_layers[:10]:
        print(f"Layer {layer_idx}: d={scores['cohens_d']:.3f}, "
              f"acc={scores['accuracy']:.1%}, "
              f"N={scores['num_natural'] + scores['num_artifact']} samples")
    
    # Plot results
    plt.figure(figsize=(14, 7))
    
    layers = list(range(NUM_LAYERS))
    cohens_ds = [all_layer_scores.get(l, {}).get('cohens_d', 0) for l in layers]
    accuracies = [all_layer_scores.get(l, {}).get('accuracy', 0.5) for l in layers]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(layers, cohens_ds, color='royalblue')
    # Highlight strong effects
    for i, d in enumerate(cohens_ds):
        if d > 0.8:
            bars[i].set_color('red')
    plt.xlabel('Layer Index')
    plt.ylabel("Cohen's d")
    plt.title('Alignment Artifact Strength (True Batch Processing)')
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.7, label="d=0.8")
    plt.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5, label="d=0.5")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(layers, accuracies, color='royalblue')
    plt.xlabel('Layer Index')
    plt.ylabel('Classification Accuracy')
    plt.title('Alignment Artifact Detectability (True Batch Processing)')
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.7, label="80%")
    plt.ylim(0.45, 1.05)
    plt.legend()
    
    plt.suptitle("Alignment Artifacts Analysis - Single Batch Processing (50x faster!)", 
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH)
    print(f"\nVisualization saved to {OUTPUT_FIGURE_PATH}")
    
    # Save detailed results
    results = {
        'layer_scores': all_layer_scores,
        'top_layers': [l[0] for l in sorted_layers if l[1]['cohens_d'] > 0.8],
        'config': {
            'activations_dir': str(RAW_ACTIVATIONS_BASE_DIR),
            'metadata_file': str(METADATA_FILE),
            'batch_size': BATCH_SIZE,
            'num_layers': NUM_LAYERS,
            'key_pattern': KEY_PATTERN_TEMPLATE,
            'num_steps': NUM_TOKEN_STEPS_TO_COLLECT
        },
        'summary': {
            'total_prompts_processed': len(prompt_mappings),
            'batches_processed': len(prompt_mappings) // BATCH_SIZE + (1 if len(prompt_mappings) % BATCH_SIZE else 0),
            'processing_speedup': 'Maximum speed - 50x faster than individual processing!'
        }
    }
    
    with open(OUTPUT_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_RESULTS_PATH}")


if __name__ == "__main__":
    try:
        plt.switch_backend('Agg')
    except:
        pass
    
    main_analysis()