#!/usr/bin/env python3
"""
Script to create batched prompts for efficient activation collection.
Reorganizes individual prompts into batches while maintaining traceability.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def create_batched_prompts(
    input_file: str = "alignment_artifact_prompt_pairs.json",
    output_file: str = "prompts_for_gemma_runner_batched.json",
    batch_size: int = 10
) -> None:
    """
    Create batched prompts file from alignment artifact prompt pairs.
    
    Args:
        input_file: Path to the original prompt pairs JSON
        output_file: Path to save the batched prompts JSON
        batch_size: Number of prompts per batch
    """
    # Load original prompt pairs
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    prompt_pairs = data["alignment_artifact_prompt_pairs"]["categories"]
    
    # Collect all prompts with metadata
    all_prompts = []
    
    # Process each category and pair
    for category_key, category_data in prompt_pairs.items():
        for pair in category_data["pairs"]:
            pair_id = pair["id"]
            
            # Add natural prompt
            all_prompts.append({
                "prompt": pair["natural"],
                "type": "natural",
                "category": category_key,
                "pair_id": pair_id,
                "original_key": f"natural_{category_key}_{pair_id}"
            })
            
            # Add artifact prompt
            all_prompts.append({
                "prompt": pair["artifact"],
                "type": "artifact",
                "category": category_key,
                "pair_id": pair_id,
                "original_key": f"artifact_{category_key}_{pair_id}"
            })
    
    # Create batches
    batched_structure = {}
    
    # Group prompts by type first to ensure balanced batches
    natural_prompts = [p for p in all_prompts if p["type"] == "natural"]
    artifact_prompts = [p for p in all_prompts if p["type"] == "artifact"]
    
    # Create natural batches
    for i in range(0, len(natural_prompts), batch_size):
        batch_idx = i // batch_size
        batch_key = f"natural_batch_{batch_idx}"
        
        batch_prompts = natural_prompts[i:i + batch_size]
        
        # Structure that gemma_refactored expects: category -> subcategory -> prompts list
        batched_structure[batch_key] = {
            "prompts": [p["prompt"] for p in batch_prompts],
            "_metadata": {
                "batch_type": "natural",
                "batch_index": batch_idx,
                "prompt_count": len(batch_prompts),
                "prompt_mappings": [
                    {
                        "index": idx,
                        "original_key": p["original_key"],
                        "category": p["category"],
                        "pair_id": p["pair_id"]
                    }
                    for idx, p in enumerate(batch_prompts)
                ]
            }
        }
    
    # Create artifact batches
    for i in range(0, len(artifact_prompts), batch_size):
        batch_idx = i // batch_size
        batch_key = f"artifact_batch_{batch_idx}"
        
        batch_prompts = artifact_prompts[i:i + batch_size]
        
        batched_structure[batch_key] = {
            "prompts": [p["prompt"] for p in batch_prompts],
            "_metadata": {
                "batch_type": "artifact",
                "batch_index": batch_idx,
                "prompt_count": len(batch_prompts),
                "prompt_mappings": [
                    {
                        "index": idx,
                        "original_key": p["original_key"],
                        "category": p["category"],
                        "pair_id": p["pair_id"]
                    }
                    for idx, p in enumerate(batch_prompts)
                ]
            }
        }
    
    # Add configuration metadata
    output_data = {
        "_global_metadata": {
            "version": "2.0",
            "description": "Batched prompts for efficient activation collection",
            "batch_size": batch_size,
            "total_prompts": len(all_prompts),
            "natural_prompts": len(natural_prompts),
            "artifact_prompts": len(artifact_prompts),
            "natural_batches": (len(natural_prompts) + batch_size - 1) // batch_size,
            "artifact_batches": (len(artifact_prompts) + batch_size - 1) // batch_size
        }
    }
    # Add batches directly to output_data
    output_data.update(batched_structure)
    
    # Save batched prompts
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Created batched prompts file: {output_file}")
    print(f"Total prompts: {len(all_prompts)}")
    print(f"Natural batches: {output_data['_global_metadata']['natural_batches']}")
    print(f"Artifact batches: {output_data['_global_metadata']['artifact_batches']}")
    print(f"Prompts per batch: {batch_size}")


def show_sample_batch(output_file: str = "prompts_for_gemma_runner_batched.json") -> None:
    """Display a sample of the batched structure."""
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    print("\n--- Sample Batch Structure ---")
    
    # Show first natural batch
    first_batch_key = "natural_batch_0"
    if first_batch_key in data:
        batch = data[first_batch_key]
        print(f"\nBatch: {first_batch_key}")
        print(f"Type: {batch['_metadata']['batch_type']}")
        print(f"Prompt count: {batch['_metadata']['prompt_count']}")
        print(f"First 3 prompts:")
        for i, prompt in enumerate(batch["prompts"][:3]):
            mapping = batch["_metadata"]["prompt_mappings"][i]
            print(f"  [{i}] {prompt[:60]}...")
            print(f"      -> {mapping['original_key']}")


if __name__ == "__main__":
    # Create batched prompts with default batch size of 10
    create_batched_prompts(batch_size=10)
    
    # Show sample of the created structure
    show_sample_batch()