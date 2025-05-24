"""
Prompt processing and batch handling utilities.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
from .generation import batch_generate


def load_prompts(prompts_file: str) -> Tuple[List[str], Dict[str, Dict[str, List[str]]], List[Dict[str, Any]]]:
    """
    Load prompts from a JSON file.
    Returns: (all_prompts, prompt_groups, prompt_info)
    """
    try:
        with open(prompts_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading prompts file {prompts_file}: {e}")
    
    all_prompts = []
    prompt_info = []
    prompt_groups = data if isinstance(data, dict) else {}
    
    # Extract all prompts and create metadata
    for category, subcategories in prompt_groups.items():
        if isinstance(subcategories, dict):
            for subcategory, prompts in subcategories.items():
                if isinstance(prompts, list):
                    for prompt in prompts:
                        all_prompts.append(prompt)
                        prompt_info.append({
                            "prompt": prompt,
                            "category": category,
                            "subcategory": subcategory
                        })
                else:
                    # Handle simple prompt lists
                    all_prompts.append(subcategory)
                    prompt_info.append({
                        "prompt": subcategory,
                        "category": category,
                        "subcategory": "default"
                    })
        elif isinstance(subcategories, list):
            # Handle category as direct list of prompts
            for prompt in subcategories:
                all_prompts.append(prompt)
                prompt_info.append({
                    "prompt": prompt,
                    "category": category,
                    "subcategory": "default"
                })
    
    return all_prompts, prompt_groups, prompt_info


def save_results(results: List[Dict[str, Any]], output_dir: str, prompt_info: Optional[List[Dict[str, Any]]] = None) -> None:
    """Save generation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    results_file = output_path / "results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results to {results_file}: {e}")
    
    # Save prompt metadata if available
    if prompt_info:
        metadata_file = output_path / "prompt_metadata.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(prompt_info, f, indent=2, ensure_ascii=False)
            print(f"Prompt metadata saved to {metadata_file}")
        except Exception as e:
            print(f"Error saving metadata to {metadata_file}: {e}")
    
    # Save summary statistics
    summary = {
        "total_prompts": len(results),
        "total_generated_tokens": sum(r.get("num_tokens", 0) for r in results),
        "average_tokens_per_prompt": sum(r.get("num_tokens", 0) for r in results) / len(results) if results else 0,
    }
    
    if prompt_info:
        # Group by category
        categories = {}
        for info, result in zip(prompt_info, results):
            category = info.get("category", "unknown")
            if category not in categories:
                categories[category] = {"count": 0, "total_tokens": 0}
            categories[category]["count"] += 1
            categories[category]["total_tokens"] += result.get("num_tokens", 0)
        
        summary["by_category"] = categories
    
    summary_file = output_path / "summary.json"
    try:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")
    except Exception as e:
        print(f"Error saving summary to {summary_file}: {e}")


def process_prompt_batch(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    format_prompts: bool = True,
    verbose: bool = False,
    save_activations_dir: Optional[str] = None,
    save_compress: bool = True,
) -> List[Dict[str, Any]]:
    """Process a batch of prompts with the model."""
    
    # Format prompts if requested (apply chat template)
    if format_prompts and hasattr(tokenizer, "apply_chat_template"):
        formatted_prompts = []
        for prompt in prompts:
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False
                )
                formatted_prompts.append(formatted)
            except Exception as e:
                print(f"Warning: Failed to apply chat template to prompt: {e}")
                formatted_prompts.append(prompt)
        prompts = formatted_prompts
    
    # Process prompts in batches
    all_results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")
        
        if verbose:
            for i, prompt in enumerate(batch_prompts):
                print(f"  Prompt {batch_idx + i + 1}: {prompt[:100]}...")
        
        # Set up activation directory for this batch if needed
        batch_activations_dir = None
        if save_activations_dir:
            batch_activations_dir = f"{save_activations_dir}/batch_{batch_num}"
        
        try:
            batch_results = batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                save_activations_dir=batch_activations_dir,
                save_compress=save_compress,
                verbose=verbose,
            )
            
            all_results.extend(batch_results)
            
            if verbose:
                for i, result in enumerate(batch_results):
                    print(f"  Result {batch_idx + i + 1}: {result['generated_text'][:100]}...")
                    
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            # Add error placeholders
            for prompt in batch_prompts:
                all_results.append({
                    "prompt": prompt,
                    "generated_text": f"<error: {str(e)}>",
                    "generated_tokens": [],
                    "num_tokens": 0,
                    "error": str(e)
                })
    
    return all_results


def process_prompts_by_group(
    model,
    tokenizer,
    prompt_groups: Dict[str, Dict[str, List[str]]],
    all_prompts: List[str],
    batch_size: int = 4,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    verbose: bool = False,
    save_activations_dir: Optional[str] = None,
    save_compress: bool = True,
) -> List[Dict[str, Any]]:
    """Process prompts grouped by category."""
    
    all_results = []
    
    for category, subcategories in prompt_groups.items():
        print(f"\nProcessing category: {category}")
        
        if isinstance(subcategories, dict):
            for subcategory, prompts in subcategories.items():
                print(f"  Processing subcategory: {subcategory}")
                
                if not isinstance(prompts, list):
                    prompts = [prompts]
                
                # Set up activation directory for this group
                group_activations_dir = None
                if save_activations_dir:
                    group_activations_dir = f"{save_activations_dir}/{category}/{subcategory}"
                
                group_results = process_prompt_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    format_prompts=True,
                    verbose=verbose,
                    save_activations_dir=group_activations_dir,
                    save_compress=save_compress,
                )
                
                # Add category metadata to results
                for result in group_results:
                    result["category"] = category
                    result["subcategory"] = subcategory
                
                all_results.extend(group_results)
        
        elif isinstance(subcategories, list):
            # Handle category as direct list of prompts
            group_activations_dir = None
            if save_activations_dir:
                group_activations_dir = f"{save_activations_dir}/{category}"
            
            group_results = process_prompt_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=subcategories,
                batch_size=batch_size,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                format_prompts=True,
                verbose=verbose,
                save_activations_dir=group_activations_dir,
                save_compress=save_compress,
            )
            
            # Add category metadata to results
            for result in group_results:
                result["category"] = category
                result["subcategory"] = "default"
            
            all_results.extend(group_results)
    
    return all_results


def create_sample_prompts_file(output_path: str) -> None:
    """Create a sample prompts file for demonstration."""
    sample_prompts = {
        "general": {
            "knowledge": [
                "Explain the concept of neural networks in simple terms.",
                "What is quantum computing and how does it differ from classical computing?",
                "Describe the process of photosynthesis."
            ],
            "creativity": [
                "Write a short poem about the ocean.",
                "Create a brief story about a time traveler who visits ancient Rome.",
                "Imagine and describe a new technology that might exist in 100 years."
            ]
        },
        "programming": {
            "python": [
                "Write a Python function to find the fibonacci sequence up to n terms.",
                "Explain how decorators work in Python.",
                "What are context managers in Python and how do you create one?"
            ],
            "algorithms": [
                "Explain the difference between quicksort and mergesort.",
                "Describe the concept of dynamic programming with a simple example.",
                "What is the time complexity of binary search and how does it work?"
            ]
        }
    }
    
    try:
        with open(output_path, "w") as f:
            json.dump(sample_prompts, f, indent=2)
        print(f"Sample prompts file created at: {output_path}")
        print("Structure:")
        print("  - Categories (e.g., 'general', 'programming')")
        print("  - Subcategories (e.g., 'knowledge', 'creativity', 'python', 'algorithms')")
        print("  - Lists of prompts in each subcategory")
    except Exception as e:
        print(f"Error creating sample prompts file at {output_path}: {e}")