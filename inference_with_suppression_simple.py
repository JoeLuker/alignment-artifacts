#!/usr/bin/env python3
"""
Simple inference with artifact suppression using steering vectors.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import json
import argparse
from typing import Dict, Optional, List

from gemma_refactored.main import load
from gemma_refactored.generation import generate_step


def compute_suppression_vectors(
    activations_dir: Path,
    target_layers: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
    num_steps: int = 20
) -> Dict[int, mx.array]:
    """Compute suppression vectors from saved activations."""
    
    # Default to layers where artifacts are strongest (based on analysis)
    if target_layers is None:
        target_layers = [1, 2, 3, 4, 5, 6]  # Early layers show strongest artifacts
    
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
    
    suppression_vectors = {}
    batch_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])
    
    for layer in target_layers:
        all_natural = []
        all_artifact = []
        
        # Collect activations across all batches and steps
        for batch_dir in batch_dirs:
            batch_num = int(batch_dir.name.split('_')[1]) - 1
            batch_offset = batch_num * 50
            
            for step in range(num_steps):
                step_file = batch_dir / f"activations_step_{step}.npz"
                if not step_file.exists():
                    continue
                    
                data = np.load(step_file)
                key = f"model.layers.{layer}.mlp.output"
                
                if key not in data:
                    continue
                
                batch_data = data[key]
                
                # Extract activations for our indices
                for idx in natural_indices:
                    if batch_offset <= idx < batch_offset + 50:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = batch_data[local_idx, 0, :] if batch_data.ndim == 3 else batch_data[local_idx, :]
                            all_natural.append(vec)
                
                for idx in artifact_indices:
                    if batch_offset <= idx < batch_offset + 50:
                        local_idx = idx - batch_offset
                        if local_idx < batch_data.shape[0]:
                            vec = batch_data[local_idx, 0, :] if batch_data.ndim == 3 else batch_data[local_idx, :]
                            all_artifact.append(vec)
        
        if all_natural and all_artifact:
            # Compute mean difference vector
            natural_mean = np.mean(all_natural, axis=0)
            artifact_mean = np.mean(all_artifact, axis=0)
            diff_vector = artifact_mean - natural_mean
            
            # Normalize
            norm = np.linalg.norm(diff_vector)
            if norm > 0:
                diff_vector = diff_vector / norm
                suppression_vectors[layer] = mx.array(diff_vector)
                print(f"Layer {layer}: computed suppression vector (norm={norm:.3f})")
    
    return suppression_vectors


def inject_suppression(model, suppression_vectors, scale=1.0):
    """Inject suppression vectors into model layers."""
    hooks = []
    
    def make_hook(layer_idx, direction, scale):
        def hook(module, inputs, outputs):
            # Subtract the artifact direction
            if outputs.ndim == 3:
                # Reshape direction for batch processing
                direction_reshaped = direction.reshape(1, 1, -1)
                return outputs - scale * direction_reshaped
            else:
                return outputs - scale * direction
        return hook
    
    # Add hooks to specified layers
    for layer_idx, direction in suppression_vectors.items():
        if layer_idx < len(model.model.layers):
            hook_fn = make_hook(layer_idx, direction, scale)
            handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
            hooks.append(handle)
    
    return hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Input prompt")
    parser.add_argument("--model", type=str, default="mlx-community/gemma-3-1b-it-qat-4bit")
    parser.add_argument("--activations-dir", type=str, default="./collected_activations_no_rep")
    parser.add_argument("--categories", type=str, nargs="+", 
                        help="Specific categories to suppress (default: all)")
    parser.add_argument("--layers", type=int, nargs="+",
                        help="Layers to apply suppression (default: 1-6)")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Suppression strength (0=none, higher=stronger)")
    parser.add_argument("--compare", action="store_true",
                        help="Show output with and without suppression")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Don't apply chat template to prompt")
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load(args.model)
    
    print("\nComputing suppression vectors...")
    suppression_vectors = compute_suppression_vectors(
        Path(args.activations_dir),
        target_layers=args.layers,
        categories=args.categories
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
    prompt_array = mx.array([prompt_tokens])  # Add batch dimension
    
    if args.compare:
        print("\n" + "="*80)
        print("BASELINE (No Suppression):")
        print("="*80)
        
        # Generate without suppression
        generated_tokens = []
        eos_token_ids = {tokenizer.eos_token_id} if hasattr(tokenizer, 'eos_token_id') else set()
        
        generator = generate_step(
            prompts=prompt_array,
            model=model,
            temp=args.temperature,
            top_p=0.9,
            eos_token_ids=eos_token_ids
        )
        
        for step, (tokens, probs) in enumerate(generator):
            if step >= args.max_tokens:
                break
            token = tokens[0, 0].item()
            generated_tokens.append(token)
            
        # Decode and print full output
        full_output = prompt_tokens + generated_tokens
        print(tokenizer.decode(full_output))
    
    # Apply suppression
    print("\n" + "="*80)
    print(f"WITH SUPPRESSION (scale={args.scale}):")
    print("="*80)
    
    hooks = inject_suppression(model, suppression_vectors, args.scale)
    
    try:
        generated_tokens = []
        eos_token_ids = {tokenizer.eos_token_id} if hasattr(tokenizer, 'eos_token_id') else set()
        
        generator = generate_step(
            prompts=prompt_array,
            model=model,
            temp=args.temperature,
            top_p=0.9,
            eos_token_ids=eos_token_ids
        )
        
        for step, (tokens, probs) in enumerate(generator):
            if step >= args.max_tokens:
                break
            token = tokens[0, 0].item()
            generated_tokens.append(token)
            
        # Decode and print full output
        full_output = prompt_tokens + generated_tokens
        print(tokenizer.decode(full_output))
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()