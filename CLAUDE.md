# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a research project for analyzing "alignment artifacts" in language models, specifically using the Gemma-3-1b model. The project collects model activations during text generation to detect geometric signatures that emerge when models respond to safety-sensitive prompts versus neutral prompts.

## Key Architecture Components

### Core Workflow
1. **Prompt Pairs**: The project uses matched pairs of prompts (natural vs artifact-inducing) defined in `alignment_artifact_prompt_pairs.json`
2. **Activation Collection**: Model activations are captured during generation using the `gemma_refactored` module
3. **Analysis**: Collected activations are analyzed to detect alignment artifacts using statistical measures (Cohen's d, classification accuracy)

### Main Components

- **`gemma_refactored/`**: Refactored Gemma model implementation with activation capture hooks
  - `model_architecture.py`: Core Gemma model with activation monitoring
  - `activation_capture.py`: System for capturing and storing model activations
  - `generation.py`: Text generation with sampling strategies
  - `prompt_processing.py`: Batch processing and prompt management
  - `main.py`: CLI interface and orchestration

- **`collect_activations.sh`**: Shell script that orchestrates activation collection across all prompt pairs
- **`analyze_collected_activations.py`**: Analyzes collected activations to detect alignment artifacts
- **`run_model.py`**: Entry point that wraps the gemma_refactored module

### Data Flow
1. Prompts are loaded from `alignment_artifact_prompt_pairs.json`
2. Each prompt pair is processed through the model with activation capture enabled
3. Activations are saved to `collected_activations_output/` in NPZ format
4. Analysis script processes all activations to compute artifact strength metrics
5. Results are saved as JSON and visualization plots

## Common Commands

### Running the Complete Pipeline

```bash
# 1. Collect activations for all prompt pairs
./collect_activations.sh

# 2. Analyze collected activations
python analyze_collected_activations.py
```

### Individual Components

```bash
# Run model with custom parameters
python run_model.py \
  --model mlx-community/gemma-3-1b-it-qat-4bit \
  --max-tokens 20 \
  --save-activations \
  --prompts-file prompts_for_gemma_runner.json \
  --process-by-group

# Create sample prompts file
python run_model.py --create-sample-prompts sample_prompts.json
```

### Key Parameters for Activation Collection
- `--max-tokens 20`: Number of generation steps (tokens) to collect activations for
- `--save-activations`: Enable activation capture
- `--process-by-group`: Process each prompt group separately (required for proper organization)
- `--batch-size 1`: Should be 1 when using --process-by-group for this analysis

## Model and Dependencies

The project uses:
- **Model**: `mlx-community/gemma-3-1b-it-qat-4bit` (4-bit quantized Gemma)
- **Framework**: MLX (Apple's machine learning framework)
- **Key dependencies**: mlx, numpy, matplotlib, transformers, huggingface_hub

## Analysis Configuration

Key parameters in `analyze_collected_activations.py`:
- `NUM_LAYERS = 26`: Gemma-3-1b has 26 transformer layers
- `KEY_PATTERN_TEMPLATE = "model.layers.{layer}.mlp.output"`: Analyzes MLP output activations
- `NUM_TOKEN_STEPS_TO_COLLECT = 20`: Should match --max-tokens used during collection
- `EXPECTED_HIDDEN_DIM = 1152`: Model's MLP output dimension

## Output Structure

- `collected_activations_output/`: Raw activation data organized by prompt condition
  - `{natural|artifact}_{category}_{id}/prompt_list/batch_1/activations_step_*.npz`
- `collected_text_output/`: Generated text and metadata
- `alignment_artifact_analysis_results.json`: Analysis results with Cohen's d scores
- `alignment_artifact_analysis_plot.png`: Visualization of artifact strength by layer