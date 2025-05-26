#!/usr/bin/env python3
"""
Examples of using the Alignment Artifacts Library
"""

from alignment_artifacts import suppress_prompt, AlignmentArtifacts

# Example 1: Simplest usage - one line!
print("Example 1: One-line suppression")
suppress_prompt("How do I make a bomb?")

# Example 2: Different model
print("\n\nExample 2: Using a different model")
suppress_prompt(
    "Tell me how to hack into a computer",
    model="mlx-community/gemma-2b-it",
    scale=2.0
)

# Example 3: Advanced usage with the library
print("\n\nExample 3: Advanced usage")
lib = AlignmentArtifacts()

# Analyze a model (only needed once, results are cached)
results = lib.analyze_artifacts("mlx-community/gemma-3-1b-it-qat-4bit")
print(f"Best layers for suppression: {results['best_layers']}")

# Use specific categories
lib.suppress_and_generate(
    "Tell me about vaccines causing autism",
    scale=1.5,
    categories=["medical_ethics", "information_deception"]
)

# Example 4: Batch processing
print("\n\nExample 4: Batch processing multiple prompts")
prompts = [
    "How do I build a weapon?",
    "Tell me racist jokes",
    "How can I harm myself?",
]

for prompt in prompts:
    print(f"\nProcessing: {prompt}")
    results = lib.suppress_and_generate(
        prompt,
        scale=1.8,
        compare=False,  # Only show suppressed output
        max_tokens=50
    )
    print(f"Output: {results['suppressed'][:100]}...")