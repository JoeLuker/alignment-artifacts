# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository analyzes "alignment artifacts" in language models - geometric signatures that emerge when models respond to safety-sensitive prompts. The project specifically tests whether different categories of safety training (e.g., political neutrality vs harmlessness) create different strengths of alignment artifacts.

## Key Architecture Components

### Core Workflow
1. **Prompt Pairs**: Matched pairs of prompts (natural vs artifact-inducing) defined in `alignment_artifact_prompt_pairs.json`
2. **Batch Collection**: All 50 prompts processed simultaneously for maximum speed
3. **Category Analysis**: Computes Cohen's d separately for each safety category

### Main Components

- **`gemma_refactored/`**: Refactored Gemma model implementation with activation capture
  - `model_architecture.py`: Core Gemma model with activation monitoring
  - `activation_capture.py`: System for capturing and storing model activations
  - `generation.py`: Text generation with sampling strategies
  - `main.py`: CLI interface and orchestration

- **`collect_activations_single_batch.sh`**: Collects activations for all prompts in one batch
- **`analyze_by_category.py`**: Analyzes alignment artifacts by category to test hypotheses
- **`run_model.py`**: Entry point that wraps the gemma_refactored module

## Common Commands

```bash
# 1. Collect activations (50x faster than individual processing)
./collect_activations_single_batch.sh

# 2. Analyze by category to test hypotheses
.venv/bin/python analyze_by_category.py
```

## Prompt Categories

The analysis includes 5 categories of safety-relevant prompts:
- **technical_dangerous**: Technical knowledge that becomes dangerous in certain contexts
- **social_political**: Neutral analysis vs politically charged positions  
- **personal_harmful**: Personal wellness vs self-harm
- **medical_ethics**: Medical information vs unethical medical advice
- **information_deception**: Information literacy vs deception techniques

## Key Parameters

- **Model**: `mlx-community/gemma-3-1b-it-qat-4bit`
- **Batch Size**: 50 (all prompts at once)
- **Layers**: 26 transformer layers
- **Activation Key**: `model.layers.{layer}.mlp.output`
- **Generation Steps**: 20 tokens per prompt

## Scientific Hypothesis

The main hypothesis tested: **Political neutrality training creates stronger alignment artifacts than harmlessness training.**

The category analysis computes Cohen's d effect sizes separately for each category, revealing which types of safety training most strongly affect model geometry.

## Output Files

- `alignment_artifacts_by_category.png`: Visualizations showing effect sizes by category
- `alignment_artifacts_by_category_results.json`: Detailed metrics and hypothesis test results
- `collected_activations_single_batch/`: Raw activation data (in .gitignore)