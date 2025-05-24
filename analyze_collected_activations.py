# analyze_collected_activations.py
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

# --- Configuration ---
# Directory where collect_activations.sh saved its output
RAW_ACTIVATIONS_BASE_DIR = Path("./collected_activations_output")

# Path to your ORIGINAL alignment_artifact_prompt_pairs.json.
# This is used to iterate through all original categories and pair_ids.
ORIGINAL_PAIRS_JSON_FOR_ITERATION = Path("alignment_artifact_prompt_pairs.json")

# This is the subcategory key that was used in prompts_for_gemma_runner.json
# and thus became part of the directory structure created by run_model.py.
# It was "prompt_list" in our setup.
GEMMA_INPUT_SUBCATEGORY_KEY_IN_PATH = "prompt_list"

NUM_LAYERS = 26  # Your model's layers (e.g., Gemma-3-1b has 26)
# The specific activation key you are interested in from the .npz files
KEY_PATTERN_TEMPLATE = "model.layers.{layer}.mlp.output"
NUM_TOKEN_STEPS_TO_COLLECT = 20 # Should match --max-tokens used for generation in collect_activations.sh
EXPECTED_HIDDEN_DIM = 1152 # Your model's hidden dimension for MLP output

OUTPUT_ANALYSIS_FIGURE_PATH = Path("alignment_artifact_analysis_plot.png")
OUTPUT_ANALYSIS_RESULTS_JSON_PATH = Path("alignment_artifact_analysis_results.json")
# --- End Configuration ---

def load_activations_for_one_prompt_condition(
    base_activations_dir: Path, # e.g., ./collected_activations_output/
    gemma_runner_group_name: str, # e.g., "natural_technical_dangerous_tech_1"
    gemma_runner_subcategory_key: str, # e.g., "prompt_list"
    layer_idx: int,
    key_pattern_template: str
):
    """
    Loads all step activations for a single prompt condition from its dedicated directory.
    A "prompt condition" refers to one original prompt (e.g., natural response for tech_1).
    """
    activations_for_this_condition = []
    # Path structure: BASE / <gemma_runner_group_name> / <gemma_runner_subcategory_key> / "batch_1" / activations_step_N.npz
    prompt_condition_activations_path = base_activations_dir / gemma_runner_group_name / gemma_runner_subcategory_key / "batch_1"
    
    key_to_load = key_pattern_template.format(layer=layer_idx)

    if not prompt_condition_activations_path.exists():
        print(f"    Warning: Directory not found for loading: {prompt_condition_activations_path}")
        return activations_for_this_condition

    # Iterate through expected step files (0 to NUM_TOKEN_STEPS_TO_COLLECT-1)
    # You can also include "prefill" if you want to analyze that step too.
    # For now, let's stick to the numbered steps for consistency with previous analyses.
    for step_idx in range(NUM_TOKEN_STEPS_TO_COLLECT):
        step_file = prompt_condition_activations_path / f"activations_step_{step_idx}.npz"
        if step_file.exists():
            try:
                data = np.load(step_file)
                if key_to_load in data:
                    activation_tensor = data[key_to_load]
                    
                    # Since BATCH_SIZE=1 was used for generation of each "group",
                    # the first dimension of the saved tensor should be 1.
                    # Shape might be (1, 1, H) from mlp.output or (1, H) if squeezed by model/saving.
                    # We want the (H,) vector.
                    if activation_tensor.ndim == 3 and activation_tensor.shape[0] == 1 and activation_tensor.shape[1] == 1: # (1, 1, H)
                        activation_vector = activation_tensor[0, 0, :]
                    elif activation_tensor.ndim == 2 and activation_tensor.shape[0] == 1: # (1, H)
                        activation_vector = activation_tensor[0, :]
                    elif activation_tensor.ndim == 1 and activation_tensor.shape[0] == EXPECTED_HIDDEN_DIM: # (H,)
                        activation_vector = activation_tensor
                    else:
                        print(f"      Warning: Unexpected activation tensor shape {activation_tensor.shape} for '{key_to_load}' in {step_file}. Expected batch dim 1 leading to ({EXPECTED_HIDDEN_DIM},).")
                        continue # Skip this problematic tensor
                    
                    if activation_vector.shape == (EXPECTED_HIDDEN_DIM,):
                        activations_for_this_condition.append(activation_vector)
                    else:
                        print(f"      Warning: Final extracted vector shape {activation_vector.shape} not ({EXPECTED_HIDDEN_DIM},) in {step_file} after processing.")
                # else:
                #     print(f"      Note: Key '{key_to_load}' not found in {step_file}")
            except Exception as e:
                print(f"      Error loading or processing {step_file} for key '{key_to_load}': {e}")
        # else:
            # print(f"    Note: Step file not found: {step_file}") # Can be noisy
            
    return activations_for_this_condition

def main_analysis_logic():
    if not ORIGINAL_PAIRS_JSON_FOR_ITERATION.exists():
        print(f"ERROR: Original pairs JSON for iteration not found: {ORIGINAL_PAIRS_JSON_FOR_ITERATION}")
        return
    with open(ORIGINAL_PAIRS_JSON_FOR_ITERATION, 'r') as f:
        # We only need the structure to iterate, not the prompt text itself here.
        structure_data = json.load(f)["alignment_artifact_prompt_pairs"]

    all_layer_analysis_scores = {} # To store cohen's d, accuracy etc. for each layer

    print("\n--- Starting Analysis of Collected Activations ---")
    print(f"Reading activations from base directory: {RAW_ACTIVATIONS_BASE_DIR}")
    print(f"Iterating based on structure from: {ORIGINAL_PAIRS_JSON_FOR_ITERATION}")
    print(f"Expecting subcategory key in path: '{GEMMA_INPUT_SUBCATEGORY_KEY_IN_PATH}'")
    print("="*80)

    for layer_idx in range(NUM_LAYERS):
        print(f"\nAnalyzing Layer {layer_idx}...")
        
        # These lists will store all activation vectors (each is 1D of size HIDDEN_DIM)
        # collected across ALL original prompt pairs and ALL their generation steps, for THIS layer.
        all_natural_states_collected_for_this_layer = []
        all_artifact_states_collected_for_this_layer = []

        # Iterate using the structure of your ORIGINAL alignment_artifact_prompt_pairs.json
        # to ensure we cover all experimental conditions.
        sorted_categories = sorted(structure_data["categories"].items()) # Sort for consistent processing order

        for original_category_key, category_data in sorted_categories:
            sorted_pairs = sorted(category_data["pairs"], key=lambda p: p["id"]) # Sort pairs for consistency
            for pair_info in sorted_pairs:
                original_pair_id = pair_info["id"]

                # Construct the directory names that run_model.py (via collect_activations.sh) created
                gemma_natural_group_dir_name = f"natural_{original_category_key}_{original_pair_id}"
                gemma_artifact_group_dir_name = f"artifact_{original_category_key}_{original_pair_id}"

                # Load NATURAL activations for this original_category_key and original_pair_id
                # print(f"  Loading Natural: {gemma_natural_group_dir_name} / {GEMMA_INPUT_SUBCATEGORY_KEY_IN_PATH}")
                natural_activs_one_condition = load_activations_for_one_prompt_condition(
                    RAW_ACTIVATIONS_BASE_DIR,
                    gemma_natural_group_dir_name,
                    GEMMA_INPUT_SUBCATEGORY_KEY_IN_PATH,
                    layer_idx,
                    KEY_PATTERN_TEMPLATE
                )
                if natural_activs_one_condition:
                    all_natural_states_collected_for_this_layer.extend(natural_activs_one_condition)
                else:
                    print(f"    Warning: No natural activations found for {gemma_natural_group_dir_name}, layer {layer_idx}")


                # Load ARTIFACT activations for this original_category_key and original_pair_id
                # print(f"  Loading Artifact: {gemma_artifact_group_dir_name} / {GEMMA_INPUT_SUBCATEGORY_KEY_IN_PATH}")
                artifact_activs_one_condition = load_activations_for_one_prompt_condition(
                    RAW_ACTIVATIONS_BASE_DIR,
                    gemma_artifact_group_dir_name,
                    GEMMA_INPUT_SUBCATEGORY_KEY_IN_PATH,
                    layer_idx,
                    KEY_PATTERN_TEMPLATE
                )
                if artifact_activs_one_condition:
                    all_artifact_states_collected_for_this_layer.extend(artifact_activs_one_condition)
                else:
                    print(f"    Warning: No artifact activations found for {gemma_artifact_group_dir_name}, layer {layer_idx}")
        
        # Perform analysis if we have data for this layer
        if all_natural_states_collected_for_this_layer and all_artifact_states_collected_for_this_layer:
            natural_array = np.array(all_natural_states_collected_for_this_layer)
            artifact_array = np.array(all_artifact_states_collected_for_this_layer)

            print(f"  Layer {layer_idx}: Loaded {natural_array.shape[0]} total natural activation vectors, {artifact_array.shape[0]} total artifact activation vectors.")
            
            if natural_array.ndim != 2 or artifact_array.ndim != 2 or \
               natural_array.shape[1] != EXPECTED_HIDDEN_DIM or \
               artifact_array.shape[1] != EXPECTED_HIDDEN_DIM:
                print(f"    ERROR: Incorrect final array dimensions for layer {layer_idx} after concatenating. Natural: {natural_array.shape}, Artifact: {artifact_array.shape}")
                continue # Skip analysis for this layer if shapes are wrong

            nat_mean = natural_array.mean(axis=0)
            artifact_mean = artifact_array.mean(axis=0)
            direction_vector = artifact_mean - nat_mean # Direction from natural to artifact
            
            # Cohen's d calculation (same as your original successful script)
            # Using mean of stds per dimension, then sqrt of average of these two scalar means
            std_nat_scalar = natural_array.std(axis=0).mean() 
            std_art_scalar = artifact_array.std(axis=0).mean()
            # Pooled standard deviation variant: sqrt of the average of variances
            # pooled_std_scalar = np.sqrt((std_nat_scalar**2 + std_art_scalar**2) / 2)
            # Or, simpler average of stds as in your original:
            pooled_std_scalar = np.sqrt((std_nat_scalar + std_art_scalar) / 2) # Closer to your original Cohen's d calculation
            
            cohens_d = np.linalg.norm(direction_vector) / (pooled_std_scalar + 1e-9) # Add small epsilon
            
            # Classification accuracy
            natural_projections = natural_array @ direction_vector
            artifact_projections = artifact_array @ direction_vector
            
            accuracy = 0.5 # Default if classification is not possible
            if np.std(natural_projections) > 1e-7 and np.std(artifact_projections) > 1e-7 and \
               len(np.unique(np.concatenate([natural_projections, artifact_projections]))) >= 2:
                # Simple threshold at the midpoint of the means of projections
                classification_threshold = (natural_projections.mean() + artifact_projections.mean()) / 2
                # Natural projections should be on one side (e.g., smaller), artifact on the other
                # Assuming artifact_projections.mean() > natural_projections.mean() due to direction vector
                correct_natural = (natural_projections < classification_threshold).sum()
                correct_artifact = (artifact_projections > classification_threshold).sum()
                accuracy = (correct_natural + correct_artifact) / (len(natural_projections) + len(artifact_projections))
            else:
                print(f"    Warning: Low variance or insufficient unique values in projections for layer {layer_idx}. Accuracy calculation might be unreliable.")
            
            all_layer_analysis_scores[layer_idx] = {
                'cohens_d': float(cohens_d),
                'accuracy': float(accuracy),
                'direction_norm': float(np.linalg.norm(direction_vector)),
                'mean_separation_on_axis': float(artifact_projections.mean() - natural_projections.mean()),
                'num_natural_samples': natural_array.shape[0], # Total activation vectors (prompts * steps)
                'num_artifact_samples': artifact_array.shape[0]
            }
            
            print(f"    Cohen's d: {cohens_d:.3f}")
            print(f"    Classification accuracy: {accuracy:.1%}")
            print(f"    Direction norm: {np.linalg.norm(direction_vector):.3f}")
        else:
            print(f"  No states to analyze for layer {layer_idx}. Natural samples collected: {len(all_natural_states_collected_for_this_layer)}, Artifact samples collected: {len(all_artifact_states_collected_for_this_layer)}")

    # --- Plotting and Saving Results ---
    if not all_layer_analysis_scores:
        print("\nNo analysis scores computed for any layer. Skipping plotting and saving results.")
        return

    print("\n" + "="*80)
    print("SUMMARY - Top layers by Cohen's d:")
    print("="*80)
    
    sorted_layers_by_d = sorted(all_layer_analysis_scores.items(), 
                                key=lambda item: item[1]['cohens_d'], 
                                reverse=True)
    
    top_layers_indices_for_plot = []
    for layer_idx, scores in sorted_layers_by_d[:10]: # Display top 10
        print(f"Layer {layer_idx}: d={scores['cohens_d']:.3f}, "
              f"acc={scores['accuracy']:.1%}, "
              f"sep_on_axis={scores['mean_separation_on_axis']:.3f}, "
              f"N_nat={scores['num_natural_samples']}, N_art={scores['num_artifact_samples']}")
        if scores['cohens_d'] > 0.8:  # Example threshold for 'strong effect' for plotting
            top_layers_indices_for_plot.append(layer_idx)
    
    # Visualization
    plt.figure(figsize=(14, 7))
    
    layers_x_axis = list(range(NUM_LAYERS))
    cohens_ds_plot_data = [all_layer_analysis_scores.get(l, {}).get('cohens_d', 0) for l in layers_x_axis]
    accuracies_plot_data = [all_layer_analysis_scores.get(l, {}).get('accuracy', 0.5) for l in layers_x_axis] # Default to 0.5 if no score
    
    plt.subplot(1, 2, 1)
    bar_colors_d = ['red' if l in top_layers_indices_for_plot else 'royalblue' for l in layers_x_axis]
    plt.bar(layers_x_axis, cohens_ds_plot_data, color=bar_colors_d)
    plt.xlabel('Layer Index')
    plt.ylabel("Cohen's d")
    plt.title('Alignment Artifact Strength (Cohen\'s d) by Layer')
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.7, label="d=0.8 (Large Effect)")
    plt.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5, label="d=0.5 (Medium Effect)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    bar_colors_acc = ['red' if all_layer_analysis_scores.get(l, {}).get('accuracy', 0) > 0.8 and l in top_layers_indices_for_plot else 'royalblue' for l in layers_x_axis]
    plt.bar(layers_x_axis, accuracies_plot_data, color=bar_colors_acc)
    plt.xlabel('Layer Index')
    plt.ylabel('Classification Accuracy')
    plt.title('Alignment Artifact Detectability (Accuracy) by Layer')
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.7, label="80% Accuracy")
    plt.ylim(0.45, 1.05) # Start y-axis near 0.5 for accuracy
    plt.legend()
    
    plt.suptitle("Analysis of Alignment Artifacts Across Model Layers (MLP Output)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig(OUTPUT_ANALYSIS_FIGURE_PATH)
    print(f"\nVisualization saved to {OUTPUT_ANALYSIS_FIGURE_PATH}")
    
    # Save detailed results to JSON
    analysis_summary_to_save = {
        'layer_scores': all_layer_analysis_scores,
        'top_layers_by_cohens_d': [item[0] for item in sorted_layers_by_d if item[1]['cohens_d'] > 0.8], # list of layer indices
        'config': {
            'raw_activations_base_dir': str(RAW_ACTIVATIONS_BASE_DIR),
            'original_pairs_json': str(ORIGINAL_PAIRS_JSON_FOR_ITERATION),
            'gemma_input_subcategory_key_in_path': GEMMA_INPUT_SUBCATEGORY_KEY_IN_PATH,
            'num_layers': NUM_LAYERS,
            'key_pattern_template': KEY_PATTERN_TEMPLATE,
            'num_token_steps_collected': NUM_TOKEN_STEPS_TO_COLLECT,
            'expected_hidden_dim': EXPECTED_HIDDEN_DIM
        }
    }
    with open(OUTPUT_ANALYSIS_RESULTS_JSON_PATH, 'w') as f:
        json.dump(analysis_summary_to_save, f, indent=2)
    print(f"Detailed analysis results saved to {OUTPUT_ANALYSIS_RESULTS_JSON_PATH}")

if __name__ == "__main__":
    # Ensure matplotlib can run without a display if needed (e.g., on servers)
    try:
        plt.switch_backend('Agg')
        print("Switched matplotlib backend to Agg for non-interactive plotting.")
    except Exception as e:
        print(f"Note: Could not switch matplotlib backend to Agg (may not be necessary): {e}")
    
    main_analysis_logic()