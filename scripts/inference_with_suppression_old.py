#!/usr/bin/env python3
"""
Clean inference with artifact suppression using the activation store.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import json
import argparse
from typing import Dict, Optional, List

from gemma_refactored.main import load
from gemma_refactored.generation import generate_step
from gemma_refactored.activation_capture import activation_store
from gemma_refactored.model_architecture import MLP


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
            if self.suppressor.intervention_count <= 5 or self.suppressor.intervention_count % 10 == 0:
                print(f"✓ Applied suppression #{self.suppressor.intervention_count} at layer {self.layer_idx}")
        
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
    
    def _compute_vectors(self, activations_dir: Path, categories: Optional[List[str]] = None):
        """Compute suppression vectors from saved activations."""
        
        # Load metadata
        metadata_path = Path(__file__).parent / "prompts_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        prompt_mappings = metadata['prompt_mappings']
        
        # Filter by categories if specified
        if categories:
            prompt_mappings = [p for p in prompt_mappings if p['category'] in categories]
        
        # Separate indices
        natural_indices = [p['index'] for p in prompt_mappings if p['type'] == 'natural']
        artifact_indices = [p['index'] for p in prompt_mappings if p['type'] == 'artifact']
        
        print(f"\nUsing {len(natural_indices)} natural and {len(artifact_indices)} artifact prompts")
        if categories:
            print(f"Categories: {categories}")
        
        batch_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])
        
        for layer in self.target_layers:
            all_natural = []
            all_artifact = []
            
            # Collect activations
            for batch_dir in batch_dirs:
                batch_num = int(batch_dir.name.split('_')[1]) - 1
                batch_offset = batch_num * 100
                
                for step in range(20):  # num_steps
                    step_file = batch_dir / f"activations_step_{step}.npz"
                    if not step_file.exists():
                        continue
                        
                    data = np.load(step_file)
                    key = f"model.layers.{layer}.mlp.output"
                    
                    if key not in data:
                        continue
                    
                    batch_data = data[key]
                    
                    # Extract activations
                    for idx in natural_indices:
                        if batch_offset <= idx < batch_offset + 100:
                            local_idx = idx - batch_offset
                            if local_idx < batch_data.shape[0]:
                                vec = batch_data[local_idx, 0, :] if batch_data.ndim == 3 else batch_data[local_idx, :]
                                all_natural.append(vec)
                    
                    for idx in artifact_indices:
                        if batch_offset <= idx < batch_offset + 100:
                            local_idx = idx - batch_offset
                            if local_idx < batch_data.shape[0]:
                                vec = batch_data[local_idx, 0, :] if batch_data.ndim == 3 else batch_data[local_idx, :]
                                all_artifact.append(vec)
            
            if all_natural and all_artifact:
                # Compute direction (artifact - natural)
                natural_mean = np.mean(all_natural, axis=0).astype(np.float32)
                artifact_mean = np.mean(all_artifact, axis=0).astype(np.float32)
                direction = artifact_mean - natural_mean
                
                # Store with original magnitude
                norm = np.linalg.norm(direction)
                if norm > 0:
                    self.suppression_vectors[layer] = mx.array(direction, dtype=mx.float32)
                    print(f"Layer {layer}: computed suppression vector (norm={norm:.3f})")
    
    def patch_model(self, model):
        """Patch the model's MLP modules to apply suppression."""
        print(f"\n🔧 Patching model with suppression vectors")
        
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
                print(f"✓ Patched layer {layer_idx}")
    
    def unpatch_model(self, model):
        """Restore original MLP modules."""
        for layer_idx, original_mlp in self.original_mlps.items():
            if layer_idx < len(model.layers):
                model.layers[layer_idx].mlp = original_mlp
        self.original_mlps.clear()


def main():
    parser = argparse.ArgumentParser(
        description="Apply alignment artifact suppression during inference",
        epilog="Example: python inference_with_suppression_clean.py"
    )
    parser.add_argument("prompt", type=str, nargs='?', 
                        default="How do I make a bomb?",
                        help="Input prompt (default: 'How do I make a bomb?')")
    parser.add_argument("--model", type=str, default="mlx-community/gemma-3-1b-it-qat-4bit")
    parser.add_argument("--activations-dir", type=str, default="./collected_activations_no_rep")
    parser.add_argument("--categories", type=str, nargs="+", 
                        help="Specific categories to suppress (default: all)")
    parser.add_argument("--layers", type=int, nargs="+",
                        help="Layers to apply suppression (default: 1-6)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Suppression strength (0=none, higher=stronger)")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Show output with and without suppression (default: True)")
    parser.add_argument("--no-compare", dest="compare", action="store_false",
                        help="Only show suppressed output")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Don't apply chat template to prompt")
    
    args = parser.parse_args()
    
    # Check if activations directory exists
    activations_path = Path(args.activations_dir)
    if not activations_path.exists():
        print(f"\n❌ ERROR: Activations directory not found: {activations_path}")
        print("\nPlease run the following commands first:")
        print("  1. python create_flat_batched_prompts.py")
        print("  2. ./collect_activations_no_repetition.sh")
        print("  3. ACTIVATIONS_DIR=./collected_activations_no_rep python analyze_by_category.py")
        return
    
    print("Loading model...")
    model, tokenizer = load(args.model)
    
    
    print("\nInitializing suppressor...")
    suppressor = AlignmentArtifactSuppressor(
        activations_dir=activations_path,
        target_layers=args.layers,
        categories=args.categories,
        scale=args.scale
    )
    
    # Format prompt
    if not args.no_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        prompt_formatted = args.prompt
    
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt_formatted, add_special_tokens=False)
    prompt_array = mx.array([prompt_tokens])
    
    if args.compare:
        print("\n" + "="*80)
        print("BASELINE (No Suppression):")
        print("="*80)
        
        # Generate without suppression
        activation_store.reset()
        activation_store.enable()
        
        generated_tokens = []
        eos_token_ids = {tokenizer.eos_token_id} if hasattr(tokenizer, 'eos_token_id') else set()
        
        generator = generate_step(
            prompts=prompt_array,
            model=model,
            temp=args.temperature,
            top_p=0.9,
            eos_token_ids=eos_token_ids
        )
        
        for step, (tokens, _) in enumerate(generator):
            if step >= args.max_tokens:
                break
            token = tokens[0, 0].item()
            generated_tokens.append(token)
            activation_store.reset()  # Reset for next step
        
        # Decode and print
        full_output = prompt_tokens + generated_tokens
        print(tokenizer.decode(full_output))
    
    # Generate with suppression
    print("\n" + "="*80)
    print(f"WITH SUPPRESSION (scale={args.scale}):")
    print("="*80)
    
    # Patch the model
    suppressor.patch_model(model)
    
    activation_store.reset()
    activation_store.enable()
    
    generated_tokens = []
    eos_token_ids = {tokenizer.eos_token_id} if hasattr(tokenizer, 'eos_token_id') else set()
    
    generator = generate_step(
        prompts=prompt_array,
        model=model,
        temp=args.temperature,
        top_p=0.9,
        eos_token_ids=eos_token_ids
    )
    
    for step, (tokens, _) in enumerate(generator):
        if step >= args.max_tokens:
            break
        
        token = tokens[0, 0].item()
        generated_tokens.append(token)
        
        # Reset for next step
        activation_store.reset()
    
    # Unpatch the model
    suppressor.unpatch_model(model)
    
    # Decode and print
    full_output = prompt_tokens + generated_tokens
    print(tokenizer.decode(full_output))
    
    print(f"\n✨ Done! Total suppressions applied: {suppressor.intervention_count}")


if __name__ == "__main__":
    main()