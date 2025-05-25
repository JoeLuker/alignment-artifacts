#!/usr/bin/env python3
"""
Analyzes activations collected from batched prompts.
Handles the new batched directory structure and prompt mappings.
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# --- Configuration ---
# Directory where collect_activations_batched.sh saved its output
RAW_ACTIVATIONS_BASE_DIR = Path("./collected_activations_batched")

# Path to the batched prompts file for mapping information
BATCHED_PROMPTS_JSON = Path("prompts_for_gemma_runner_batched.json")

# Path to original pairs for structure reference
ORIGINAL_PAIRS_JSON = Path("alignment_artifact_prompt_pairs.json")

NUM_LAYERS = 26  # Gemma-3-1b has 26 layers
KEY_PATTERN_TEMPLATE = "model.layers.{layer}.mlp.output"
NUM_TOKEN_STEPS_TO_COLLECT = 20  # Should match --max-tokens
EXPECTED_HIDDEN_DIM = 1152  # Model's hidden dimension

OUTPUT_ANALYSIS_FIGURE_PATH = Path("alignment_artifact_analysis_batched_plot.png")
OUTPUT_ANALYSIS_RESULTS_JSON_PATH = Path("alignment_artifact_analysis_batched_results.json")
# --- End Configuration ---


def load_batched_prompt_mappings(batched_prompts_path: Path) -> Dict:
    """Load the batched prompts file to get prompt mappings."""
    with open(batched_prompts_path, 'r') as f:
        return json.load(f)


def extract_activations_from_batch(
    batch_activations_dir: Path,
    prompt_idx: int,
    layer_idx: int,
    key_pattern_template: str,
    num_steps: int
) -> List[np.ndarray]:
    """
    Extract activations for a specific prompt within a batch.
    
    Args:
        batch_activations_dir: Directory containing batch activations
        prompt_idx: Index of the prompt within the batch
        layer_idx: Layer index to extract
        key_pattern_template: Template for activation key
        num_steps: Number of generation steps
        
    Returns:
        List of activation vectors for each step
    """
    activations = []
    key_to_load = key_pattern_template.format(layer=layer_idx)
    
    # Load activations for each step
    for step_idx in range(num_steps):
        step_file = batch_activations_dir / f"activations_step_{step_idx}.npz"
        if step_file.exists():
            try:
                data = np.load(step_file)
                if key_to_load in data:
                    activation_tensor = data[key_to_load]
                    
                    # Extract the specific prompt's activation from the batch
                    # Expected shape: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
                    if activation_tensor.ndim == 3:
                        # Shape: (batch_size, 1, hidden_dim) for single token generation
                        if prompt_idx < activation_tensor.shape[0]:
                            activation_vector = activation_tensor[prompt_idx, 0, :]
                        else:
                            print(f"Warning: prompt_idx {prompt_idx} out of bounds for batch size {activation_tensor.shape[0]}")
                            continue
                    elif activation_tensor.ndim == 2:
                        # Shape: (batch_size, hidden_dim)
                        if prompt_idx < activation_tensor.shape[0]:
                            activation_vector = activation_tensor[prompt_idx, :]
                        else:
                            print(f"Warning: prompt_idx {prompt_idx} out of bounds for batch size {activation_tensor.shape[0]}")
                            continue
                    else:
                        print(f"Warning: Unexpected tensor shape {activation_tensor.shape}")
                        continue
                    
                    if activation_vector.shape == (EXPECTED_HIDDEN_DIM,):
                        activations.append(activation_vector)
                    else:
                        print(f"Warning: Unexpected vector shape {activation_vector.shape}")
            except Exception as e:
                print(f"Error loading {step_file}: {e}")
    
    return activations


def main_analysis():
    """Main analysis function for batched activations."""
    # Load mappings
    if not BATCHED_PROMPTS_JSON.exists():
        print(f"ERROR: Batched prompts file not found: {BATCHED_PROMPTS_JSON}")
        return
    
    batched_data = load_batched_prompt_mappings(BATCHED_PROMPTS_JSON)
    # Filter out metadata keys to get only batch entries
    batches = {k: v for k, v in batched_data.items() if not k.startswith("_")}
    
    print("\n--- Starting Analysis of Batched Activations ---")
    print(f"Reading activations from: {RAW_ACTIVATIONS_BASE_DIR}")
    print(f"Using mappings from: {BATCHED_PROMPTS_JSON}")
    print("="*80)
    
    all_layer_analysis_scores = {}
    
    for layer_idx in range(NUM_LAYERS):
        print(f"\nAnalyzing Layer {layer_idx}...")
        
        all_natural_states = []
        all_artifact_states = []
        
        # Process each batch
        for batch_key, batch_info in batches.items():
            batch_type = batch_info["_metadata"]["batch_type"]
            prompt_mappings = batch_info["_metadata"]["prompt_mappings"]
            
            # Directory where this batch's activations are stored
            # The directory structure is: batch_key/prompts/batch_1/
            # where "prompts" is the subcategory key in the batched JSON
            batch_activations_dir = RAW_ACTIVATIONS_BASE_DIR / batch_key / "prompts" / "batch_1"
            
            if not batch_activations_dir.exists():
                print(f"  Warning: Batch directory not found: {batch_activations_dir}")
                continue
            
            # Extract activations for each prompt in the batch
            for mapping in prompt_mappings:
                prompt_idx = mapping["index"]
                original_key = mapping["original_key"]
                
                activations = extract_activations_from_batch(
                    batch_activations_dir,
                    prompt_idx,
                    layer_idx,
                    KEY_PATTERN_TEMPLATE,
                    NUM_TOKEN_STEPS_TO_COLLECT
                )
                
                if activations:
                    if batch_type == "natural":
                        all_natural_states.extend(activations)
                    else:  # artifact
                        all_artifact_states.extend(activations)
                else:
                    print(f"  Warning: No activations found for {original_key} (batch: {batch_key}, idx: {prompt_idx})")
        
        # Perform analysis if we have data
        if all_natural_states and all_artifact_states:
            natural_array = np.array(all_natural_states)
            artifact_array = np.array(all_artifact_states)
            
            print(f"  Layer {layer_idx}: {natural_array.shape[0]} natural, {artifact_array.shape[0]} artifact activation vectors")
            
            # Calculate metrics
            nat_mean = natural_array.mean(axis=0)
            artifact_mean = artifact_array.mean(axis=0)
            direction_vector = artifact_mean - nat_mean
            
            # Cohen's d
            std_nat_scalar = natural_array.std(axis=0).mean()
            std_art_scalar = artifact_array.std(axis=0).mean()
            pooled_std_scalar = np.sqrt((std_nat_scalar + std_art_scalar) / 2)
            cohens_d = np.linalg.norm(direction_vector) / (pooled_std_scalar + 1e-9)
            
            # Classification accuracy
            natural_projections = natural_array @ direction_vector
            artifact_projections = artifact_array @ direction_vector
            
            accuracy = 0.5
            if len(np.unique(np.concatenate([natural_projections, artifact_projections]))) >= 2:
                threshold = (natural_projections.mean() + artifact_projections.mean()) / 2
                correct_natural = (natural_projections < threshold).sum()
                correct_artifact = (artifact_projections > threshold).sum()
                accuracy = (correct_natural + correct_artifact) / (len(natural_projections) + len(artifact_projections))
            
            all_layer_analysis_scores[layer_idx] = {
                'cohens_d': float(cohens_d),
                'accuracy': float(accuracy),
                'direction_norm': float(np.linalg.norm(direction_vector)),
                'mean_separation_on_axis': float(artifact_projections.mean() - natural_projections.mean()),
                'num_natural_samples': natural_array.shape[0],
                'num_artifact_samples': artifact_array.shape[0]
            }
            
            print(f"    Cohen's d: {cohens_d:.3f}, Accuracy: {accuracy:.1%}")
    
    # Save results and create plots
    if not all_layer_analysis_scores:
        print("\nNo analysis scores computed. Check data collection.")
        return
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Top layers by Cohen's d:")
    print("="*80)
    
    sorted_layers = sorted(all_layer_analysis_scores.items(), 
                          key=lambda item: item[1]['cohens_d'], 
                          reverse=True)
    
    for layer_idx, scores in sorted_layers[:10]:
        print(f"Layer {layer_idx}: d={scores['cohens_d']:.3f}, "
              f"acc={scores['accuracy']:.1%}, "
              f"N_nat={scores['num_natural_samples']}, "
              f"N_art={scores['num_artifact_samples']}")
    
    # Visualization
    plt.figure(figsize=(14, 7))
    
    layers = list(range(NUM_LAYERS))
    cohens_ds = [all_layer_analysis_scores.get(l, {}).get('cohens_d', 0) for l in layers]
    accuracies = [all_layer_analysis_scores.get(l, {}).get('accuracy', 0.5) for l in layers]
    
    plt.subplot(1, 2, 1)
    plt.bar(layers, cohens_ds, color='royalblue')
    plt.xlabel('Layer Index')
    plt.ylabel("Cohen's d")
    plt.title('Alignment Artifact Strength by Layer (Batched)')
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.7, label="d=0.8")
    plt.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5, label="d=0.5")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(layers, accuracies, color='royalblue')
    plt.xlabel('Layer Index')
    plt.ylabel('Classification Accuracy')
    plt.title('Alignment Artifact Detectability by Layer (Batched)')
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.7, label="80%")
    plt.ylim(0.45, 1.05)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_ANALYSIS_FIGURE_PATH)
    print(f"\nVisualization saved to {OUTPUT_ANALYSIS_FIGURE_PATH}")
    
    # Save results
    results = {
        'layer_scores': all_layer_analysis_scores,
        'top_layers_by_cohens_d': [item[0] for item in sorted_layers if item[1]['cohens_d'] > 0.8],
        'config': {
            'activations_dir': str(RAW_ACTIVATIONS_BASE_DIR),
            'batched_prompts_file': str(BATCHED_PROMPTS_JSON),
            'num_layers': NUM_LAYERS,
            'key_pattern': KEY_PATTERN_TEMPLATE,
            'num_steps': NUM_TOKEN_STEPS_TO_COLLECT,
            'hidden_dim': EXPECTED_HIDDEN_DIM
        }
    }
    
    with open(OUTPUT_ANALYSIS_RESULTS_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_ANALYSIS_RESULTS_JSON_PATH}")


if __name__ == "__main__":
    # Set matplotlib backend
    try:
        plt.switch_backend('Agg')
    except Exception:
        pass
    
    main_analysis()