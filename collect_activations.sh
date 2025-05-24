#!/bin/bash

PROMPT_FILE_FOR_GEMMA="./prompts_for_gemma_runner.json"

# Output directory where run_model.py will save activations.
# This will contain subdirs based on the groups in PROMPT_FILE_FOR_GEMMA.
RAW_ACTIVATIONS_DIR="./collected_activations_output"
# Directory for text generation results (optional, but good practice).
TEXT_RESULTS_DIR="./collected_text_output"

MODEL_NAME="mlx-community/gemma-3-1b-it-qat-4bit" # Or your model
MAX_TOKENS=20       # Number of activation steps (tokens) to collect per prompt
BATCH_SIZE=1        # CRITICAL: Since each "group" in PROMPT_FILE_FOR_GEMMA
                    # represents one original prompt, and we want activations
                    # for each in a separate directory created by --process-by-group,
                    # the batch size for processing these "groups" should be 1.

# Ensure output directories exist
mkdir -p "$RAW_ACTIVATIONS_DIR"
mkdir -p "$TEXT_RESULTS_DIR"

echo "--- Starting Activation Collection ---"
echo "Using prompt file: $PROMPT_FILE_FOR_GEMMA"
echo "Outputting raw activations to: $RAW_ACTIVATIONS_DIR"
echo "Outputting text results to: $TEXT_RESULTS_DIR"
echo "Model: $MODEL_NAME"
echo "Max tokens (steps) per prompt: $MAX_TOKENS"
echo "Batch size (for groups): $BATCH_SIZE (should be 1 for this strategy)"

# Make sure your venv is active if you run this script directly
# If run from a terminal where venv is active, this is not strictly needed.
# source .venv/bin/activate 

python ./run_model.py \
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
    echo "--- Activation Collection Script Completed Successfully ---"
    echo "Raw activations should be in subdirectories within: $RAW_ACTIVATIONS_DIR"
else
    echo ""
    echo "--- ERROR: Activation Collection Script Failed (Exit Code: $EXIT_CODE) ---"
fi