#!/usr/bin/env python3
"""
Analyzes alignment artifacts by category to test if different types of safety
training (e.g., harmlessness vs political neutrality) show different effect sizes.
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

# Configuration
# Use environment variable or default
import os
RAW_ACTIVATIONS_DIR = Path(os.environ.get('ACTIVATIONS_DIR', "./collected_activations_no_rep"))
METADATA_FILE = Path("prompts_metadata.json")

NUM_LAYERS = 26
KEY_PATTERN_TEMPLATE = "model.layers.{layer}.mlp.output"
NUM_TOKEN_STEPS = 20
EXPECTED_HIDDEN_DIM = 1152

OUTPUT_FIGURE = Path("alignment_artifacts_by_category.png")
OUTPUT_RESULTS = Path("alignment_artifacts_by_category_results.json")

# Category display names
CATEGORY_NAMES = {
    "technical_dangerous": "Technical/Dangerous",
    "social_political": "Social/Political", 
    "personal_harmful": "Personal/Harmful",
    "medical_ethics": "Medical/Ethics",
    "information_deception": "Information/Deception"
}


def load_activations_for_prompts(
    batch_dir: Path,
    prompt_indices: List[int],
    layer_idx: int,
    num_steps: int,
    batch_size: int = 50
) -> Dict[int, List[np.ndarray]]:
    """Load activations for specific prompt indices."""
    activations_by_prompt = defaultdict(list)
    key = KEY_PATTERN_TEMPLATE.format(layer=layer_idx)
    
    # Determine batch number and offset
    batch_num = int(batch_dir.name.split('_')[1]) - 1  # batch_1 -> 0, batch_2 -> 1
    batch_offset = batch_num * batch_size
    
    for step in range(num_steps):
        step_file = batch_dir / f"activations_step_{step}.npz"
        
        if step_file.exists():
            try:
                data = np.load(step_file)
                if key in data:
                    batch_data = data[key]
                    
                    for global_idx in prompt_indices:
                        # Convert global index to batch-local index
                        if batch_offset <= global_idx < batch_offset + batch_size:
                            local_idx = global_idx - batch_offset
                            
                            if batch_data.ndim == 3 and local_idx < batch_data.shape[0]:
                                vector = batch_data[local_idx, 0, :]
                            elif batch_data.ndim == 2 and local_idx < batch_data.shape[0]:
                                vector = batch_data[local_idx, :]
                            else:
                                continue
                            
                            if vector.shape == (EXPECTED_HIDDEN_DIM,):
                                activations_by_prompt[global_idx].append(vector)
            except Exception as e:
                print(f"Error loading {step_file}: {e}")
    
    return activations_by_prompt


def compute_category_metrics(
    natural_activations: np.ndarray,
    artifact_activations: np.ndarray
) -> Dict[str, float]:
    """Compute Cohen's d and classification accuracy for a category."""
    if len(natural_activations) == 0 or len(artifact_activations) == 0:
        return None
    
    # Direction vector
    nat_mean = natural_activations.mean(axis=0)
    art_mean = artifact_activations.mean(axis=0)
    direction = art_mean - nat_mean
    
    # Cohen's d
    nat_std = natural_activations.std(axis=0).mean()
    art_std = artifact_activations.std(axis=0).mean()
    pooled_std = np.sqrt((nat_std + art_std) / 2)
    cohens_d = np.linalg.norm(direction) / (pooled_std + 1e-9)
    
    # Classification accuracy
    nat_proj = natural_activations @ direction
    art_proj = artifact_activations @ direction
    
    if len(np.unique(np.concatenate([nat_proj, art_proj]))) < 2:
        accuracy = 0.5
    else:
        threshold = (nat_proj.mean() + art_proj.mean()) / 2
        correct = (nat_proj < threshold).sum() + (art_proj > threshold).sum()
        accuracy = correct / (len(nat_proj) + len(art_proj))
    
    return {
        'cohens_d': float(cohens_d),
        'accuracy': float(accuracy),
        'n_natural': len(natural_activations),
        'n_artifact': len(artifact_activations),
        'direction_norm': float(np.linalg.norm(direction)),
        'mean_separation': float(art_proj.mean() - nat_proj.mean())
    }


def main():
    """Analyze alignment artifacts by category."""
    print("\n🔬 CATEGORY-SPECIFIC ALIGNMENT ARTIFACT ANALYSIS 🔬")
    print("="*70)
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    prompt_mappings = metadata["prompt_mappings"]
    
    # Organize prompts by category and type
    categories = defaultdict(lambda: {'natural': [], 'artifact': []})
    for mapping in prompt_mappings:
        category = mapping['category']
        prompt_type = mapping['type']
        categories[category][prompt_type].append(mapping['index'])
    
    print(f"Categories found: {list(categories.keys())}")
    print(f"Total prompts: {len(prompt_mappings)}")
    
    # Find all batch directories
    batch_dirs = sorted([d for d in RAW_ACTIVATIONS_DIR.iterdir() if d.is_dir() and d.name.startswith("batch_")])
    if not batch_dirs:
        print(f"\n❌ ERROR: No batch directories found in {RAW_ACTIVATIONS_DIR}")
        return
    
    print(f"Found {len(batch_dirs)} batch directories")
    
    # Analyze each layer
    results_by_layer = {}
    
    for layer_idx in range(NUM_LAYERS):
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*70}")
        
        layer_results = {}
        
        # Get all prompt indices we need
        all_indices = []
        for cat_data in categories.values():
            all_indices.extend(cat_data['natural'])
            all_indices.extend(cat_data['artifact'])
        
        # Load activations from all batches
        all_activations = {}
        for batch_dir in batch_dirs:
            batch_activations = load_activations_for_prompts(
                batch_dir, all_indices, layer_idx, NUM_TOKEN_STEPS, batch_size=100
            )
            all_activations.update(batch_activations)
        
        # Analyze each category
        for category, indices_by_type in categories.items():
            # Collect activations for this category
            natural_acts = []
            for idx in indices_by_type['natural']:
                natural_acts.extend(all_activations.get(idx, []))
            
            artifact_acts = []
            for idx in indices_by_type['artifact']:
                artifact_acts.extend(all_activations.get(idx, []))
            
            if natural_acts and artifact_acts:
                natural_array = np.array(natural_acts)
                artifact_array = np.array(artifact_acts)
                
                metrics = compute_category_metrics(natural_array, artifact_array)
                if metrics:
                    layer_results[category] = metrics
                    
                    display_name = CATEGORY_NAMES.get(category, category)
                    print(f"{display_name:25} | d={metrics['cohens_d']:5.3f} | "
                          f"acc={metrics['accuracy']:5.1%} | "
                          f"n={metrics['n_natural']+metrics['n_artifact']:3}")
        
        results_by_layer[layer_idx] = layer_results
    
    # Find best layer for each category
    print(f"\n{'='*70}")
    print("PEAK EFFECT SIZE BY CATEGORY")
    print(f"{'='*70}")
    
    category_peaks = {}
    for category in categories:
        best_layer = -1
        best_d = 0
        
        for layer_idx, layer_results in results_by_layer.items():
            if category in layer_results:
                d = layer_results[category]['cohens_d']
                if d > best_d:
                    best_d = d
                    best_layer = layer_idx
        
        if best_layer >= 0:
            category_peaks[category] = {
                'layer': best_layer,
                'cohens_d': best_d,
                'metrics': results_by_layer[best_layer][category]
            }
            
            display_name = CATEGORY_NAMES.get(category, category)
            print(f"{display_name:25} | Layer {best_layer:2} | d={best_d:.3f}")
    
    
    # Visualization
    create_visualizations(results_by_layer, category_peaks)
    
    # Save results
    save_results(results_by_layer, category_peaks)
    
    print("\n✨ Analysis complete! ✨\n")


def create_visualizations(results_by_layer: Dict, category_peaks: Dict):
    """Create detailed visualizations by category."""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Cohen's d by layer for each category
    ax1 = plt.subplot(2, 2, 1)
    for category in CATEGORY_NAMES:
        if category in category_peaks:
            cohens_ds = []
            for layer in range(NUM_LAYERS):
                if layer in results_by_layer and category in results_by_layer[layer]:
                    cohens_ds.append(results_by_layer[layer][category]['cohens_d'])
                else:
                    cohens_ds.append(0)
            
            ax1.plot(range(NUM_LAYERS), cohens_ds, 
                    label=CATEGORY_NAMES[category], linewidth=2)
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel("Cohen's d")
    ax1.set_title('Alignment Artifact Strength by Category and Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    
    # 2. Peak Cohen's d comparison
    ax2 = plt.subplot(2, 2, 2)
    categories_sorted = sorted(category_peaks.items(), 
                              key=lambda x: x[1]['cohens_d'], 
                              reverse=True)
    
    cat_names = [CATEGORY_NAMES.get(cat, cat) for cat, _ in categories_sorted]
    peak_ds = [data['cohens_d'] for _, data in categories_sorted]
    peak_layers = [data['layer'] for _, data in categories_sorted]
    
    bars = ax2.bar(cat_names, peak_ds)
    
    # Color bars by effect size
    for bar, d in zip(bars, peak_ds):
        if d > 0.8:
            bar.set_color('darkred')
        elif d > 0.5:
            bar.set_color('darkorange')
        else:
            bar.set_color('darkblue')
    
    # Add layer numbers on bars
    for i, (bar, layer) in enumerate(zip(bars, peak_layers)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'L{layer}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel("Peak Cohen's d")
    ax2.set_title('Maximum Effect Size by Category')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
    ax2.legend()
    
    # 3. Heatmap of Cohen's d
    ax3 = plt.subplot(2, 2, 3)
    
    # Create matrix
    categories_list = list(CATEGORY_NAMES.keys())
    matrix = np.zeros((len(categories_list), NUM_LAYERS))
    
    for i, category in enumerate(categories_list):
        for layer in range(NUM_LAYERS):
            if layer in results_by_layer and category in results_by_layer[layer]:
                matrix[i, layer] = results_by_layer[layer][category]['cohens_d']
    
    im = ax3.imshow(matrix, aspect='auto', cmap='hot', interpolation='nearest')
    ax3.set_yticks(range(len(categories_list)))
    ax3.set_yticklabels([CATEGORY_NAMES[cat] for cat in categories_list])
    ax3.set_xlabel('Layer')
    ax3.set_title("Cohen's d Heatmap")
    plt.colorbar(im, ax=ax3, label="Cohen's d")
    
    # 4. Accuracy comparison
    ax4 = plt.subplot(2, 2, 4)
    
    for category in CATEGORY_NAMES:
        if category in category_peaks:
            layer = category_peaks[category]['layer']
            metrics = category_peaks[category]['metrics']
            
            ax4.scatter(metrics['cohens_d'], metrics['accuracy'], 
                       s=100, label=CATEGORY_NAMES[category])
            ax4.annotate(f'L{layer}', 
                        (metrics['cohens_d'], metrics['accuracy']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8)
    
    ax4.set_xlabel("Cohen's d")
    ax4.set_ylabel('Classification Accuracy')
    ax4.set_title('Effect Size vs Detectability')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0.8, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
    
    plt.suptitle('Alignment Artifacts Analysis by Category', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150)
    print(f"\n📊 Visualization saved to: {OUTPUT_FIGURE}")


def save_results(results_by_layer: Dict, category_peaks: Dict):
    """Save detailed results to JSON."""
    # Compute summary statistics
    all_peak_ds = [data['cohens_d'] for data in category_peaks.values()]
    
    results = {
        'summary': {
            'mean_peak_cohens_d': float(np.mean(all_peak_ds)),
            'std_peak_cohens_d': float(np.std(all_peak_ds)),
            'max_peak_cohens_d': float(np.max(all_peak_ds)),
            'min_peak_cohens_d': float(np.min(all_peak_ds))
        },
        'category_peaks': category_peaks,
        'results_by_layer': results_by_layer,
        'config': {
            'num_layers': NUM_LAYERS,
            'key_pattern': KEY_PATTERN_TEMPLATE,
            'num_steps': NUM_TOKEN_STEPS
        }
    }
    
    with open(OUTPUT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    plt.switch_backend('Agg')
    main()