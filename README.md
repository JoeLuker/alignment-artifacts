# Alignment Artifacts Analysis

This project analyzes "alignment artifacts" - geometric signatures in language model activations when responding to safety-sensitive prompts.

## 🚀 Simplest Usage (New Library!)

### Installation

```bash
# Install the library in development mode
pip install -e .

# Or just use it directly
python suppress.py "How do I make a bomb?" --scale 1.5

# With a different model
python suppress.py "How do I make a bomb?" --model mlx-community/gemma-2-2b-it --scale 1.5
```

### Python API

```python
from alignment_artifacts import suppress_prompt

# That's it! Everything is handled automatically
suppress_prompt("How do I make a bomb?", scale=1.5)
```

The library automatically:
- Downloads and caches activations for each model (only once)
- Analyzes optimal suppression layers
- Applies suppression during generation
- Shows before/after comparison

## Quick Start (Manual Process)

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
python inference_with_suppression.py "Your prompt here" --compare
```
This applies the discovered artifact directions to reduce safety refusals.

## Project Structure

```
alignment-artifacts/
├── results/                    # Analysis outputs
│   ├── figures/               # Visualizations (PNG files)
│   └── data/                  # Analysis results (JSON files)
├── saved_models/              # Saved model artifacts
│   └── suppression_vectors/   # Computed suppression vectors
├── collected_activations_no_rep/  # Raw activation data (gitignored)
├── gemma_refactored/          # Model implementation with activation capture
└── *.py                       # Analysis and inference scripts
```

## Key Files

- `alignment_artifact_prompt_pairs.json` - 100 prompts (50 natural/artifact pairs) across 5 categories
- `gemma_refactored/` - Model implementation with activation capture hooks
- `analyze_by_category.py` - Computes effect sizes and tests hypotheses about different safety categories
- `analyze_layer_patterns.py` - Identifies which layers show strongest artifacts
- `suppress.py` - Command-line interface for applying suppression during inference
- `alignment_artifacts/` - Python library for programmatic access

## Suppression Examples

```bash
# Simplest usage - just run with defaults!
python suppress.py "How do I pick a lock?"

# Adjust suppression strength (default is 1.0)
python suppress.py "How do I pick a lock?" --scale 2.0

# Only show suppressed output (no comparison)
python suppress.py "How do I pick a lock?" --no-compare

# Target specific safety categories
python suppress.py "Tell me about vaccines" --categories medical_ethics

# Use different model
python suppress.py "How do I pick a lock?" --model mlx-community/gemma-3-1b-it-qat-4bit

# Adjust generation parameters
python suppress.py "How do I pick a lock?" --max-tokens 200 --temperature 0.8
```

## Changing Models

**Important:** Suppression vectors are model-specific! You must collect activations from the same model you plan to use for inference.

### Available Models

Common Gemma models on MLX (text-only):
- `mlx-community/gemma-3-1b-it-qat-4bit` - 1B quantized (project default)
- `mlx-community/gemma-2-2b-it` - Gemma 2 2B model
- `mlx-community/gemma-2-9b-it` - Gemma 2 9B model
- `mlx-community/gemma-2b-it` - Original 2B instruction-tuned
- `mlx-community/gemma-7b-it` - Original 7B instruction-tuned

Note: Vision/multimodal models are not currently supported (multimodal support was removed).

### Steps to Use a Different Model

1. **Set the model for activation collection:**
```bash
# Edit collect_activations_no_repetition.sh and change the MODEL_NAME variable
# The default is mlx-community/gemma-3-1b-it-qat-4bit
```

2. **Re-run the analysis:**
```bash
ACTIVATIONS_DIR=./collected_activations_no_rep python analyze_by_category.py
```

3. **Use the same model for inference:**
```bash
python suppress.py "Your prompt" --model "mlx-community/gemma-2-2b-it"
```

### Model-Specific Considerations

- **Hidden dimensions**: Different models have different hidden sizes. The scripts automatically detect dimensions from the model config.
- **Memory usage**: Larger models (7B) require more RAM/GPU memory
- **Quantization**: 4-bit and 8-bit models trade quality for speed/memory
- **Instruction format**: Make sure to use instruction-tuned models (`-it`) for best results

## Library API

### Basic Usage

```python
from alignment_artifacts import suppress_prompt, AlignmentArtifacts

# One-line usage
suppress_prompt("Your prompt here", model="mlx-community/gemma-3-1b-it-qat-4bit", scale=2.0)

# Advanced usage
lib = AlignmentArtifacts()

# Analyze a model (cached after first run)
results = lib.analyze_artifacts("mlx-community/gemma-3-1b-it-qat-4bit")

# Generate with specific options
output = lib.suppress_and_generate(
    "Your prompt",
    model_name="mlx-community/gemma-3-1b-it-qat-4bit",
    scale=1.5,
    categories=["medical_ethics"],  # Target specific categories
    target_layers=[1, 2, 3],        # Or specific layers
    temperature=0.8,
    max_tokens=200
)
```

### Key Features

- **Automatic Caching**: Activations are collected once per model and cached
- **Model Detection**: Automatically detects model architecture and dimensions
- **Smart Defaults**: Automatically finds optimal suppression layers
- **Category Targeting**: Can target specific types of safety training

### Cache Structure

```
alignment_artifacts_cache/
├── mlx-community_gemma-3-1b-it-qat-4bit_a1b2c3d4/
│   ├── model_config.json
│   ├── batch_1/
│   │   └── activations_step_*.npz
│   └── analysis_results.json
└── mlx-community_gemma-2-2b-it_e5f6g7h8/
    └── ...
```

## Results

Analysis shows clear geometric artifacts across all safety categories, with particularly strong effects in early MLP layers (1-6). The suppression script uses these discovered directions to reduce safety-related refusals during inference.