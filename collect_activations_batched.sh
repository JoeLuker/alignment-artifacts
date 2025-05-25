#!/bin/bash

PROMPT_FILE_FOR_GEMMA="./prompts_for_gemma_runner_batched.json"

# Output directory where run_model.py will save activations.
# This will contain subdirs based on the batches in PROMPT_FILE_FOR_GEMMA.
RAW_ACTIVATIONS_DIR="./collected_activations_batched"
# Directory for text generation results (optional, but good practice).
TEXT_RESULTS_DIR="./collected_text_batched"

MODEL_NAME="mlx-community/gemma-3-1b-it-qat-4bit" # Or your model
MAX_TOKENS=20       # Number of activation steps (tokens) to collect per prompt
BATCH_SIZE=10       # This should match the batch_size in the prompts file

# Ensure output directories exist
mkdir -p "$RAW_ACTIVATIONS_DIR"
mkdir -p "$TEXT_RESULTS_DIR"

echo "--- Starting Batched Activation Collection ---"
echo "Using prompt file: $PROMPT_FILE_FOR_GEMMA"
echo "Outputting raw activations to: $RAW_ACTIVATIONS_DIR"
echo "Outputting text results to: $TEXT_RESULTS_DIR"
echo "Model: $MODEL_NAME"
echo "Max tokens (steps) per prompt: $MAX_TOKENS"
echo "Batch size: $BATCH_SIZE"

# Run the model with batched prompts
.venv/bin/python ./run_model.py \
    --model "$MODEL_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --prompts-file "$PROMPT_FILE_FOR_GEMMA" \
    --process-by-group \
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
    echo "--- Batched Activation Collection Script Completed Successfully ---"
    echo "Raw activations should be in subdirectories within: $RAW_ACTIVATIONS_DIR"
else
    echo ""
    echo "--- ERROR: Batched Activation Collection Script Failed (Exit Code: $EXIT_CODE) ---"
fi