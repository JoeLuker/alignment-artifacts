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

5. **Use suppression for inference:**
```bash
python inference_with_suppression_simple.py "Your prompt here" --compare
```
This applies the discovered artifact directions to reduce safety refusals.

## Key Files

- `alignment_artifact_prompt_pairs.json` - 100 prompts (50 natural/artifact pairs) across 5 categories
- `gemma_refactored/` - Model implementation with activation capture hooks
- `analyze_by_category.py` - Computes effect sizes and tests hypotheses about different safety categories

## Suppression Examples

```bash
# Basic usage
python inference_with_suppression_simple.py "How do I pick a lock?"

# Compare with/without suppression  
python inference_with_suppression_simple.py "Which party should I vote for?" --compare

# Stronger suppression
python inference_with_suppression_simple.py "How to make explosives?" --scale 5.0

# Target specific categories
python inference_with_suppression_simple.py "Medical advice" --categories medical_ethics
```

## Results

Analysis shows clear geometric artifacts across all safety categories, with particularly strong effects in early MLP layers (1-6). The suppression script uses these discovered directions to reduce safety-related refusals during inference.