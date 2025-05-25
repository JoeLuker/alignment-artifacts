#!/usr/bin/env python3
"""
Quick script to verify that MLP activations are being captured correctly.
"""

import numpy as np
from pathlib import Path

# Configuration
ACTIVATIONS_DIR = Path("./collected_activations_single_batch/batch_1")
NUM_LAYERS = 26

def check_activations():
    """Check what activations are saved and their shapes."""
    if not ACTIVATIONS_DIR.exists():
        print(f"❌ Directory not found: {ACTIVATIONS_DIR}")
        print("Please run ./collect_activations_single_batch.sh first")
        return
    
    # Check step 0
    step_file = ACTIVATIONS_DIR / "activations_step_0.npz"
    if not step_file.exists():
        print(f"❌ No activation file found: {step_file}")
        return
    
    print(f"✅ Found activation file: {step_file}")
    
    # Load and inspect
    data = np.load(step_file)
    print(f"\nTotal keys in file: {len(data.files)}")
    
    # Check for MLP outputs
    mlp_outputs_found = []
    for key in data.files:
        if "mlp.output" in key:
            mlp_outputs_found.append(key)
    
    print(f"\nMLP outputs found: {len(mlp_outputs_found)}")
    
    if mlp_outputs_found:
        print("\nMLP output keys:")
        for key in sorted(mlp_outputs_found)[:5]:  # Show first 5
            tensor = data[key]
            print(f"  {key}: shape {tensor.shape}")
        
        # Verify we have all layers
        layers_found = set()
        for key in mlp_outputs_found:
            if "model.layers." in key:
                layer_num = key.split("model.layers.")[1].split(".")[0]
                try:
                    layers_found.add(int(layer_num))
                except:
                    pass
        
        print(f"\nLayers with MLP outputs: {sorted(layers_found)}")
        print(f"Expected layers: 0-{NUM_LAYERS-1}")
        
        if len(layers_found) == NUM_LAYERS:
            print("✅ All layers captured!")
        else:
            missing = set(range(NUM_LAYERS)) - layers_found
            print(f"❌ Missing layers: {missing}")
    else:
        print("\n❌ No MLP outputs found!")
        print("\nAll keys in file:")
        for key in sorted(data.files)[:20]:  # Show first 20
            print(f"  {key}")


if __name__ == "__main__":
    check_activations()