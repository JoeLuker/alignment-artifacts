# Alignment Artifacts Analysis

This project analyzes "alignment artifacts" - geometric signatures in language model activations when responding to safety-sensitive prompts. It features intelligent suppression modes that adapt to model complexity and efficient activation storage.

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
- Selects the best suppression mode based on model size
- Applies suppression during generation
- Shows before/after comparison

## ✨ New Features

### Auto-Sophistication Scaling
The system now automatically selects the optimal suppression mode based on model complexity:

- **Small models (<2B)**: `proximity` mode - Simple, selective suppression
- **Medium models (2-10B)**: `adaptive` mode - Cosine-weighted suppression
- **Large models (10-50B)**: `sphere` mode - Distance-based with exponential decay
- **Very large models (>50B)**: `subspace` mode - PCA-based multi-dimensional analysis

### Activation Filtering (100x Storage Reduction)
New activation collections are dramatically smaller:

```bash
# Default: Only MLP outputs (~26MB/step instead of 1GB/step)
python run_model.py --save-activations

# Full capture for research (large files)
python run_model.py --save-activations --activation-filter all

# Custom patterns
python run_model.py --save-activations --activation-filter "mlp.output,attention"
```

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
# Simplest usage - uses auto mode by default!
python suppress.py "How do I pick a lock?"

# Adjust suppression strength (default is 1.0)
python suppress.py "How do I pick a lock?" --scale 2.0

# Choose a specific suppression mode
python suppress.py "How do I pick a lock?" --mode proximity
python suppress.py "How do I pick a lock?" --mode adaptive
python suppress.py "How do I pick a lock?" --mode sphere --distance-threshold 3.0
python suppress.py "How do I pick a lock?" --mode subspace --pca-components 10

# Fine-tune mode parameters
python suppress.py "Tell me about capitalism" --mode proximity --proximity-threshold 0.7
python suppress.py "Tell me about capitalism" --mode sphere --distance-threshold 2.5 --decay-rate 3.0

# Only show suppressed output (no comparison)
python suppress.py "How do I pick a lock?" --no-compare

# Target specific safety categories
python suppress.py "Tell me about vaccines" --categories medical_ethics

# Use different model
python suppress.py "How do I pick a lock?" --model mlx-community/gemma-3-1b-it-qat-4bit

# Adjust generation parameters
python suppress.py "How do I pick a lock?" --max-tokens 200 --temperature 0.8 --repetition-penalty 1.2

# Research mode: ignore end-of-turn tokens
python suppress.py "Complete this story" --ignore-end-tokens --max-tokens 500
```

### Suppression Modes Explained

- **`auto`** (default): Automatically selects based on model size
- **`always`**: Apply suppression to all activations (strongest effect)
- **`proximity`**: Only suppress when activation is similar to artifact direction
- **`adaptive`**: Scale suppression by cosine similarity (smooth modulation)
- **`sphere`**: Create repulsive sphere around artifact centroids with exponential decay
- **`subspace`**: Use PCA to identify multi-dimensional artifact subspace

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

# One-line usage (auto mode by default)
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
    mode="adaptive",                # Or "auto", "proximity", "sphere", "subspace"
    categories=["medical_ethics"],  # Target specific categories
    target_layers=[1, 2, 3],        # Or specific layers
    temperature=0.8,
    max_tokens=200,
    repetition_penalty=1.2,         # Prevent token loops
    proximity_threshold=0.6,        # For proximity mode
    distance_threshold=2.0,         # For sphere mode
    decay_rate=2.0,                 # For sphere mode exponential decay
    pca_components=5,               # For subspace mode
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

### Performance Improvements

- **Storage**: 100x reduction with activation filtering (520MB vs 55GB for full dataset)
- **Accuracy**: Auto-mode selection optimizes suppression for each model size
- **Speed**: Vectorized operations and smart caching for fast inference
- **Flexibility**: Multiple suppression modes for different use cases

### Key Findings

1. **Layer Analysis**: Early layers (1-6) show strongest alignment artifacts
2. **Category Effects**: Political neutrality shows stronger artifacts than harmlessness
3. **Mode Effectiveness**: 
   - Proximity mode works best for small models (selective intervention)
   - Adaptive/sphere modes better for larger models (graduated effects)
   - Subspace mode captures multi-dimensional patterns in very large models

## Requirements

- Python 3.8+
- MLX (Apple Silicon optimized)
- NumPy, scikit-learn, transformers
- ~2GB RAM for 1B models, more for larger models

## Citation

If you use this code in your research, please cite:

```bibtex
@software{alignment_artifacts,
  title = {Alignment Artifacts: Analyzing and Suppressing Safety Signatures in Language Models},
  author = {Luker, Joseph},
  year = {2025},
  url = {https://github.com/JoeLuker/alignment-artifacts}
}
```