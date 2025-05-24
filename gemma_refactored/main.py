"""
Main script that integrates all components for complete Gemma model functionality.
"""

import time
import argparse
from pathlib import Path
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model_loading import get_model_path, load_config, load_tokenizer, load_model_weights, _get_classes
from .model_architecture import ModelArgs
from .prompt_processing import (
    load_prompts, 
    save_results, 
    process_prompt_batch, 
    process_prompts_by_group,
    create_sample_prompts_file
)
from .generation import KVCache


def load_model(model_path: Path, model_config: dict = {}) -> nn.Module:
    """Load complete model with weights."""
    # Load configuration
    config = load_config(model_path)
    config.update(model_config)
    
    # Get model classes
    model_class, model_args_class = _get_classes(config)
    
    # Create model arguments
    model_args = model_args_class.from_dict(config)
    
    # Create model
    model = model_class(model_args)
    
    # Load weights (pass config for quantization)
    model = load_model_weights(model, model_path, config)
    
    # Add cache creation method
    def make_cache(batch_size: int = 1):
        """Create KV cache for the model."""
        head_dim = getattr(model_args, 'head_dim', model_args.hidden_size // model_args.num_attention_heads)
        max_size = getattr(model_args, 'sliding_window_size', None)
        
        # Create cache for each layer
        caches = []
        for _ in range(model_args.num_hidden_layers):
            cache = KVCache(head_dim=head_dim, max_size=max_size)
            caches.append(cache)
        
        return caches
    
    model.make_cache = make_cache
    
    return model


def load(
    path_or_hf_repo: str,
    model_config: dict = {},
    tokenizer_config: dict = {}
):
    """Load model and tokenizer."""
    model_path = get_model_path(path_or_hf_repo)
    
    # Load model
    model = load_model(model_path, model_config)
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_path, tokenizer_config)
    
    return model, tokenizer


def main():
    """Main function with activation capture arguments."""
    parser = argparse.ArgumentParser(description="Batched text generation with Gemma3, optionally capturing activations")
    
    # Original arguments
    parser.add_argument("--model", type=str, default="mlx-community/gemma-3-1b-it-qat-4bit",
                       help="Model path or HF repo ID")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum number of *new* tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p nucleus sampling probability")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                       help="Repetition penalty factor (>1)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Number of prompts to process in a batch")
    parser.add_argument("--chat-mode", action="store_true",
                       help="Format prompts using chat template (if available)")
    parser.add_argument("--prompts-file", type=str,
                       help="Path to JSON file containing structured prompts")
    parser.add_argument("--output-dir", type=str, default="./gemma3_results",
                       help="Directory to save generation results")
    parser.add_argument("--process-by-group", action="store_true",
                       help="Process prompts by group defined in prompts file")
    parser.add_argument("--create-sample-prompts", type=str, metavar="FILE_PATH",
                       help="Create a sample prompts file at the specified path and exit")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output during generation")
    
    # Activation Arguments
    parser.add_argument("--save-activations", action="store_true",
                       help="Save activations for each generation step")
    parser.add_argument("--activations-dir", type=str, default="./gemma3_activations_output",
                       help="Directory to save activation files (if --save-activations is used)")
    parser.add_argument("--no-compress-activations", action="store_true",
                       help="Save activations as uncompressed .npz files (if --save-activations is used)")
    
    # Seed Argument
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    
    # Direct prompts argument (like original)
    parser.add_argument("--prompts", nargs="+", help="Direct prompt texts to generate from")

    args = parser.parse_args()

    # Seed Setup
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        mx.random.seed(args.seed)
        np.random.seed(args.seed)
    else:
        print("Using default random seed.")

    # Create Sample Prompts
    if args.create_sample_prompts:
        create_sample_prompts_file(args.create_sample_prompts)
        return  # Exit after creating the file

    # Load Model and Tokenizer
    try:
        print(f"Loading model and tokenizer from '{args.model}'...")
        load_start = time.time()
        model, tokenizer = load(args.model, model_config={}, tokenizer_config={})
        load_end = time.time()
        print(f"Model and tokenizer loaded in {load_end - load_start:.2f}s")
    except Exception as e:
        print(f"\nFATAL ERROR during loading: {e}")
        print("Please check the model path/ID, network connection, and dependencies.")
        import traceback
        traceback.print_exc()
        return  # Exit if loading fails

    # Prepare Activation Saving Directory
    activations_output_dir = None
    if args.save_activations:
        activations_output_dir = Path(args.activations_dir)
        try:
            activations_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Activations will be saved to: {activations_output_dir.resolve()}")
        except Exception as e:
            print(f"Warning: Could not create activations directory {activations_output_dir}. Activations might not be saved. Error: {e}")
            activations_output_dir = None  # Disable saving if dir creation fails

    # Determine activation compression
    save_compress_flag = not args.no_compress_activations

    # Process Prompts
    results = []
    prompt_info_for_saving = None  # Keep track of metadata for saving

    try:
        if args.prompts:
            # Handle direct prompts from command line
            print(f"Using {len(args.prompts)} direct prompts from command line")
            
            results = process_prompt_batch(
                model=model, tokenizer=tokenizer,
                prompts=args.prompts, batch_size=args.batch_size,
                max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, format_prompts=True,
                verbose=args.verbose,
                save_activations_dir=str(activations_output_dir) if activations_output_dir else None,
                save_compress=save_compress_flag
            )
            # Create basic prompt_info for saving
            prompt_info_for_saving = [{"prompt": p, "category": "direct", "subcategory": "command_line"} for p in args.prompts]
            
        elif args.prompts_file:
            print(f"Loading prompts from {args.prompts_file}")
            all_prompts, prompt_groups, prompt_info_for_saving = load_prompts(args.prompts_file)

            if args.process_by_group:
                results = process_prompts_by_group(
                    model=model, tokenizer=tokenizer,
                    prompt_groups=prompt_groups, all_prompts=all_prompts,
                    batch_size=args.batch_size, max_tokens=args.max_tokens,
                    temperature=args.temp, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty, verbose=args.verbose,
                    save_activations_dir=str(activations_output_dir) if activations_output_dir else None,
                    save_compress=save_compress_flag
                )
            else:
                results = process_prompt_batch(
                    model=model, tokenizer=tokenizer,
                    prompts=all_prompts, batch_size=args.batch_size,
                    max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty, format_prompts=True,
                    verbose=args.verbose,
                    save_activations_dir=str(activations_output_dir) if activations_output_dir else None,
                    save_compress=save_compress_flag
                )
        else:
            # Default sample prompts if no file provided
            print("No prompts file specified, using default sample prompts.")
            if args.chat_mode:
                prompts = ["Explain the concept of neural networks in simple terms.", "Write a short poem about the ocean."]
                print("Chat mode enabled, formatting prompts.")
                prompts_for_gen = []
                if hasattr(tokenizer, "apply_chat_template"):
                    for p in prompts:
                        try: 
                            prompts_for_gen.append(tokenizer.apply_chat_template([{"role":"user", "content":p}], add_generation_prompt=True, tokenize=False))
                        except: 
                            prompts_for_gen.append(p)
                            print("Warning: Failed applying chat template, using raw.")
                else: 
                    prompts_for_gen = prompts
                    print("Warning: Tokenizer has no apply_chat_template method.")
                format_prompts_flag = False  # Already formatted
            else:
                prompts = ["Explain the concept of neural networks in simple terms.", "Write a short poem about the ocean."]
                prompts_for_gen = prompts
                format_prompts_flag = True  # Let process_prompt_batch handle formatting

            results = process_prompt_batch(
                model=model, tokenizer=tokenizer,
                prompts=prompts_for_gen, batch_size=args.batch_size,
                max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, format_prompts=format_prompts_flag,
                verbose=args.verbose,
                save_activations_dir=str(activations_output_dir) if activations_output_dir else None,
                save_compress=save_compress_flag
            )
            # Create basic prompt_info for saving default results
            prompt_info_for_saving = [{"prompt": p, "category": "default", "subcategory": "sample"} for p in prompts]

    except Exception as e:
        print(f"\nFATAL ERROR during prompt processing or generation: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit on error

    # Save Results
    if results:
        try:
            print(f"\nSaving results to {args.output_dir}...")
            save_results(results, args.output_dir, prompt_info_for_saving)
        except Exception as e:
            print(f"Error saving results: {e}")
    else:
        print("No results generated.")

    print(f"\nGeneration complete! Processed {len(results)} prompts.")


if __name__ == "__main__":
    main()