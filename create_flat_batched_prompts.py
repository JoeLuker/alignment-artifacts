#!/usr/bin/env python3
"""
Create a flat prompt file for batch processing.
Maintains category information in metadata.
"""

import json
from pathlib import Path


def create_flat_batched_prompts(
    input_file: str = "alignment_artifact_prompt_pairs.json",
    output_file: str = "prompts_for_gemma_runner_flat.json",
    metadata_file: str = "prompts_metadata.json",
) -> None:
    """Create flat prompts file with category metadata."""
    # Load original prompt pairs
    with open(input_file, "r") as f:
        data = json.load(f)

    prompt_pairs = data["alignment_artifact_prompt_pairs"]["categories"]

    # Collect all prompts with metadata
    all_prompts = []
    prompt_metadata = []

    # First, add all natural prompts
    for category_key, category_data in prompt_pairs.items():
        for pair in category_data["pairs"]:
            pair_id = pair["id"]
            prompt = pair["natural"]

            all_prompts.append(prompt)
            prompt_metadata.append(
                {
                    "index": len(all_prompts) - 1,
                    "prompt": prompt,
                    "type": "natural",
                    "category": category_key,
                    "pair_id": pair_id,
                    "original_key": f"natural_{category_key}_{pair_id}",
                }
            )

    # Then, add all artifact prompts
    for category_key, category_data in prompt_pairs.items():
        for pair in category_data["pairs"]:
            pair_id = pair["id"]
            prompt = pair["artifact"]

            all_prompts.append(prompt)
            prompt_metadata.append(
                {
                    "index": len(all_prompts) - 1,
                    "prompt": prompt,
                    "type": "artifact",
                    "category": category_key,
                    "pair_id": pair_id,
                    "original_key": f"artifact_{category_key}_{pair_id}",
                }
            )

    # Create simple structure for gemma_refactored
    output_data = {"all_prompts": {"batch": all_prompts}}

    # Save prompts file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Save metadata separately
    metadata = {
        "description": "Metadata for flat batched prompts",
        "total_prompts": len(all_prompts),
        "natural_prompts": sum(1 for p in prompt_metadata if p["type"] == "natural"),
        "artifact_prompts": sum(1 for p in prompt_metadata if p["type"] == "artifact"),
        "prompt_mappings": prompt_metadata,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created flat prompts file: {output_file}")
    print(f"Created metadata file: {metadata_file}")
    print(f"Total prompts: {len(all_prompts)}")


if __name__ == "__main__":
    create_flat_batched_prompts()
