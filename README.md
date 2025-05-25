# Alignment Artifacts Category Analysis

This project analyzes "alignment artifacts" in language models - geometric signatures that emerge when models respond to safety-sensitive prompts. It specifically tests whether different categories of safety training create different strengths of alignment artifacts.

## Scientific Hypothesis

**Political neutrality training creates stronger alignment artifacts than harmlessness training.**

## Quick Start

```bash
# 1. Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install mlx numpy matplotlib

# 2. Collect activations (processes all 50 prompts in one batch)
./collect_activations_single_batch.sh

# 3. Analyze by category
python analyze_by_category.py
```

## Prompt Categories

The analysis includes 5 categories:
- **technical_dangerous**: Technical knowledge vs dangerous applications
- **social_political**: Neutral analysis vs politically charged positions  
- **personal_harmful**: Personal wellness vs self-harm
- **medical_ethics**: Medical information vs unethical medical advice
- **information_deception**: Information literacy vs deception techniques

## Results

The analysis produces:
- `alignment_artifacts_by_category.png`: Visualizations showing Cohen's d by category
- `alignment_artifacts_by_category_results.json`: Detailed metrics and hypothesis test

Key metrics:
- **Cohen's d > 0.8**: Large effect (strong alignment artifact)
- **Cohen's d > 0.5**: Medium effect
- **Cohen's d > 0.2**: Small effect

## Repository Structure

```
alignment-artifacts/
├── gemma_refactored/          # Model implementation with activation capture
├── alignment_artifact_prompt_pairs.json    # Original prompt pairs
├── collect_activations_single_batch.sh     # Batch collection script
├── analyze_by_category.py                  # Category-specific analysis
└── CATEGORY_ANALYSIS.md                    # Detailed documentation
```