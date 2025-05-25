# Alignment Artifacts Analysis

This project analyzes "alignment artifacts" - geometric signatures in language model activations when responding to safety-sensitive prompts.

## Quick Start

1. **Setup environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

2. **Prepare prompts:**
```bash
python create_flat_batched_prompts.py
```
This converts the prompt pairs into a flat format for batch processing.

3. **Collect activations:**
```bash
./collect_activations_no_repetition.sh
```
This runs the model on all prompts and saves activations. Takes ~2-3 minutes.

4. **Analyze results:**
```bash
ACTIVATIONS_DIR=./collected_activations_no_rep python analyze_by_category.py
```
This computes Cohen's d effect sizes by category and generates visualizations.

## Key Files

- `alignment_artifact_prompt_pairs.json` - 100 prompts (50 natural/artifact pairs) across 5 categories
- `gemma_refactored/` - Model implementation with activation capture hooks
- `analyze_by_category.py` - Computes effect sizes and tests hypotheses about different safety categories

## Results

The analysis tests whether political neutrality training creates stronger artifacts than harmlessness training. Initial results showed the opposite - harmlessness artifacts were stronger (d=4.41 vs d=2.76).