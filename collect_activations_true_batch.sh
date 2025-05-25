#!/bin/bash

PROMPT_FILE_FOR_GEMMA="./prompts_for_gemma_runner_flat.json"
METADATA_FILE="./prompts_metadata.json"

# Output directory where run_model.py will save activations
RAW_ACTIVATIONS_DIR="./collected_activations_true_batch"
# Directory for text generation results
TEXT_RESULTS_DIR="./collected_text_true_batch"

MODEL_NAME="mlx-community/gemma-3-1b-it-qat-4bit"
MAX_TOKENS=20       # Number of activation steps (tokens) to collect per prompt
BATCH_SIZE=50       # Process ALL prompts in a single batch!

# Ensure output directories exist
mkdir -p "$RAW_ACTIVATIONS_DIR"
mkdir -p "$TEXT_RESULTS_DIR"

echo "--- Starting TRUE Batched Activation Collection ---"
echo "Using prompt file: $PROMPT_FILE_FOR_GEMMA"
echo "Metadata file: $METADATA_FILE"
echo "Outputting raw activations to: $RAW_ACTIVATIONS_DIR"
echo "Outputting text results to: $TEXT_RESULTS_DIR"
echo "Model: $MODEL_NAME"
echo "Max tokens (steps) per prompt: $MAX_TOKENS"
echo "Batch size: $BATCH_SIZE (SINGLE BATCH - ALL prompts at once!)"
echo ""
echo "Expected processing:"
echo "  - 50 prompts total"
echo "  - 1 single batch containing all prompts"
echo "  - Maximum possible speed - 50x faster than individual processing!"

# Run the model with true batch processing
.venv/bin/python ./run_model.py \
    --model "$MODEL_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --prompts-file "$PROMPT_FILE_FOR_GEMMA" \
    --save-activations \
    --activations-dir "$RAW_ACTIVATIONS_DIR" \
    --output-dir "$TEXT_RESULTS_DIR" \
    --batch-size "$BATCH_SIZE" \
    --verbose \
    # Add other generation parameters if needed:
    # --temp 0.8 \
    # --top-p 0.9 \
    # --repetition-penalty 1.1 \
    # --seed 42

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "--- TRUE Batched Activation Collection Completed Successfully ---"
    echo "Raw activations saved to: $RAW_ACTIVATIONS_DIR"
    echo "Text results saved to: $TEXT_RESULTS_DIR"
    echo ""
    echo "Next step: Run the analysis script"
    echo "  .venv/bin/python analyze_true_batch_activations.py"
else
    echo ""
    echo "--- ERROR: Activation Collection Failed (Exit Code: $EXIT_CODE) ---"
fi