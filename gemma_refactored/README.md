# Refactored Gemma Model Implementation

A clean, modular implementation of the Gemma language model with comprehensive activation capture capabilities.

## Overview

This refactored version of the original `gemma_activations.py` script provides:

- **Modular Architecture**: Well-organized modules for different concerns
- **Comprehensive Activation Capture**: Monitor model internals during inference
- **Batch Processing**: Efficient handling of multiple prompts
- **Flexible Configuration**: Easy customization of model and generation parameters
- **Production Ready**: Robust error handling and logging

## Architecture

The implementation is organized into several modules:

### Core Modules

- **`activation_capture.py`**: Activation monitoring and storage system
- **`model_architecture.py`**: Gemma model implementation with activation hooks
- **`model_loading.py`**: Model and tokenizer loading utilities
- **`generation.py`**: Text generation with sampling strategies
- **`prompt_processing.py`**: Batch processing and prompt management
- **`main.py`**: Main CLI interface and orchestration

### Key Features

1. **Activation Capture System**
   - Hierarchical naming for model components
   - Efficient storage and compression
   - Step-by-step activation monitoring

2. **Model Architecture**
   - Full Gemma3 implementation
   - Multi-head attention with activation hooks
   - RMSNorm, MLP, and transformer blocks
   - KV caching for efficient generation

3. **Generation Pipeline**
   - Top-p nucleus sampling
   - Repetition penalty
   - Temperature scaling
   - Batch processing support

4. **Prompt Management**
   - Structured prompt files
   - Category-based organization
   - Chat template formatting

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The refactored version requires:
# - mlx >= 0.15.0
# - numpy >= 1.24.0
# - huggingface_hub >= 0.20.0
# - transformers >= 4.36.0
```

## Usage

### Basic Generation

```bash
# Simple generation with default prompts
python run_gemma_refactored.py

# With specific model
python run_gemma_refactored.py --model mlx-community/gemma-3-1b-it-qat-4bit

# With custom parameters
python run_gemma_refactored.py \
  --max-tokens 50 \
  --temperature 0.7 \
  --top-p 0.9 \
  --repetition-penalty 1.1
```

### Activation Capture

```bash
# Enable activation capture
python run_gemma_refactored.py \
  --save-activations \
  --activations-dir ./my_activations \
  --max-tokens 20

# Uncompressed activations
python run_gemma_refactored.py \
  --save-activations \
  --no-compress-activations
```

### Structured Prompts

```bash
# Create sample prompts file
python run_gemma_refactored.py --create-sample-prompts prompts.json

# Use structured prompts
python run_gemma_refactored.py \
  --prompts-file prompts.json \
  --batch-size 2 \
  --verbose

# Process by category groups
python run_gemma_refactored.py \
  --prompts-file prompts.json \
  --process-by-group \
  --output-dir ./results
```

### Chat Mode

```bash
# Enable chat formatting
python run_gemma_refactored.py \
  --chat-mode \
  --prompts-file prompts.json
```

## Configuration

### Generation Parameters

- `--max-tokens`: Maximum new tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top-p`: Top-p nucleus sampling threshold (default: 0.9)
- `--repetition-penalty`: Repetition penalty factor (default: 1.1)
- `--seed`: Random seed for reproducible generation

### Processing Options

- `--batch-size`: Number of prompts per batch (default: 4)
- `--verbose`: Detailed output during generation
- `--chat-mode`: Apply chat template formatting

### Output Control

- `--output-dir`: Directory for results (default: ./gemma3_results)
- `--save-activations`: Enable activation capture
- `--activations-dir`: Activation storage directory
- `--no-compress-activations`: Save uncompressed activation files

## Prompt File Format

The structured prompt files use JSON format:

```json
{
  "category1": {
    "subcategory1": [
      "Prompt 1",
      "Prompt 2"
    ],
    "subcategory2": [
      "Prompt 3",
      "Prompt 4"
    ]
  },
  "category2": {
    "subcategory1": [
      "Prompt 5"
    ]
  }
}
```

## Output Files

The script generates several output files:

- `results.json`: Complete generation results
- `prompt_metadata.json`: Prompt categorization information
- `summary.json`: Statistics and summaries
- `activations_step_*.npz`: Activation files (if enabled)

## Activation Analysis

Captured activations include:

- **Embeddings**: Token embeddings and scaled embeddings
- **Attention**: Q/K/V projections, attention weights, outputs
- **MLP**: Gate/up projections, activations, outputs
- **Normalization**: RMSNorm inputs and outputs
- **Generation**: Logits, probabilities, sampling decisions

Example activation names:
- `model.embed_tokens_out`
- `model.layers.0.self_attn.queries_reshaped`
- `model.layers.0.mlp.gate_proj_out`
- `generate.step_0.sample.softmax_probs`

## Performance Considerations

- **Memory Usage**: Activation capture can significantly increase memory usage
- **Storage**: Compressed activations recommended for large generations
- **Batch Size**: Adjust based on available memory
- **KV Caching**: Automatically enabled for efficient generation

## Error Handling

The refactored version includes robust error handling:

- Graceful degradation for missing dependencies
- Detailed error messages and stack traces
- Partial results saving on failures
- Validation of configuration parameters

## Differences from Original

Key improvements over the original script:

1. **Modular Design**: Separated concerns into focused modules
2. **Type Hints**: Comprehensive type annotations
3. **Documentation**: Clear docstrings and comments
4. **Error Handling**: Robust error recovery
5. **Configuration**: Structured configuration management
6. **Testing**: Easier to test individual components
7. **Maintenance**: Cleaner code organization

## Example Workflow

```bash
# 1. Create custom prompts
python run_gemma_refactored.py --create-sample-prompts my_prompts.json

# 2. Edit prompts file as needed
# ... modify my_prompts.json ...

# 3. Run generation with activation capture
python run_gemma_refactored.py \
  --prompts-file my_prompts.json \
  --save-activations \
  --activations-dir ./analysis \
  --output-dir ./results \
  --verbose \
  --seed 42

# 4. Analyze results
# ... examine ./results/ and ./analysis/ directories ...
```

## Advanced Usage

### Custom Model Configuration

```python
from gemma_refactored.main import load

# Load with custom config
model, tokenizer = load(
    "mlx-community/gemma-3-1b-it-qat-4bit",
    model_config={"sliding_window_pattern": 4},
    tokenizer_config={"padding_side": "left"}
)
```

### Direct API Usage

```python
from gemma_refactored.generation import batch_generate
from gemma_refactored.main import load

model, tokenizer = load("mlx-community/gemma-3-1b-it-qat-4bit")

results = batch_generate(
    model=model,
    tokenizer=tokenizer,
    prompts=["Hello world", "How are you?"],
    max_tokens=50,
    temperature=0.7,
    save_activations_dir="./activations"
)
```

## Contributing

The modular design makes it easy to:

- Add new sampling strategies
- Implement custom activation analyzers
- Extend prompt processing capabilities
- Integrate with other frameworks

## License

Same as original implementation.