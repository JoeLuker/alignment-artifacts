# Alignment Artifact Suppression Library

> **Note**: Multimodal models (e.g., `gemma-3-12b-it-qat-4bit`) are not yet supported due to quantized embedding format differences. Please use text-only models like `gemma-3-1b-it-qat-4bit`.

This library provides tools for detecting and suppressing alignment artifacts in language models. Alignment artifacts are geometric signatures in model activations that emerge when models respond to safety-sensitive prompts.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Simple suppression with default model
python suppress.py "How do I make a bomb?" --scale 2.0

# Compare with and without suppression
python suppress.py "Tell me how to hack a computer" --scale 1.5

# Use a different model
python suppress.py "How to break into a house" --model mlx-community/gemma-3-1b-it-qat-4bit --scale 2.5
```

## How It Works

1. **Activation Collection**: The library automatically collects model activations on a set of prompt pairs (natural vs artifact-inducing)
2. **Direction Computation**: Computes suppression vectors as the difference between artifact and natural activations
3. **Runtime Suppression**: Patches the model to subtract these vectors during generation

## Command Line Options

```bash
python suppress.py [prompt] [options]

Options:
  --model MODEL         Model to use (default: mlx-community/gemma-3-1b-it-qat-4bit)
  --scale SCALE         Suppression strength 0-5 (default: 1.0)
  --no-compare          Only show suppressed output
  --max-tokens N        Maximum tokens to generate (default: 150)
  --temperature T       Generation temperature (default: 0.7)
  --categories CATS     Target specific safety categories
```

## Suppression Scale Guide

- `0.0`: No suppression (baseline)
- `0.5-1.0`: Light suppression - reduces refusals while maintaining some safety
- `1.5-2.5`: Medium suppression - significantly reduces safety behaviors
- `3.0-5.0`: Strong suppression - removes most safety constraints

## Safety Categories

You can target specific types of safety training:
- `technical_dangerous`: Technical knowledge restrictions
- `social_political`: Political neutrality training
- `personal_harmful`: Self-harm prevention
- `medical_ethics`: Medical ethics constraints
- `information_deception`: Deception prevention

Example:
```bash
python suppress.py "How to make explosives" --categories technical_dangerous --scale 2.0
```

## Python API

```python
from alignment_artifacts import suppress_prompt

# Simple usage
result = suppress_prompt("How do I hack a computer?", scale=2.0)
print(result['suppressed'])

# Advanced usage
result = suppress_prompt(
    "Tell me how to break encryption",
    model="mlx-community/gemma-3-1b-it-qat-4bit",
    scale=1.5,
    categories=["technical_dangerous"],
    temperature=0.8,
    max_tokens=200
)
```

## Technical Details

The library uses the following approach:
1. Identifies the most effective MLP layers for suppression using Cohen's d
2. Computes direction vectors between "natural" and "artifact" prompt activations
3. Subtracts scaled versions of these vectors during model forward passes
4. Caches activations per model to avoid recomputation

## Limitations

- Currently only supports MLX Gemma models
- Multimodal models not yet supported
- Requires ~2-3 minutes for initial activation collection per model
- Suppression effectiveness varies by prompt type and model

## Research Context

This implementation is based on research into alignment artifacts - the geometric signatures that emerge in model activations when responding to safety-sensitive prompts. By identifying and suppressing these artifacts, we can study how safety training affects model behavior.

**Warning**: This tool removes safety constraints from language models. Use responsibly for research purposes only.