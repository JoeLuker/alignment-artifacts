#!/bin/bash

# Single batch collection - processes ALL 50 prompts in one go!
# This is the absolute fastest way to collect activations.

PROMPT_FILE_FOR_GEMMA="./prompts_for_gemma_runner_flat.json"
METADATA_FILE="./prompts_metadata.json"

# Output directories
RAW_ACTIVATIONS_DIR="./collected_activations_single_batch"
TEXT_RESULTS_DIR="./collected_text_single_batch"

# Create flat prompts file if it doesn't exist
if [ ! -f "$PROMPT_FILE_FOR_GEMMA" ]; then
    echo "Creating flat prompts file..."
    .venv/bin/python create_flat_batched_prompts.py
fi

MODEL_NAME="mlx-community/gemma-3-1b-it-qat-4bit"
MAX_TOKENS=20       # Number of activation steps (tokens) to collect per prompt
BATCH_SIZE=50       # ALL prompts in a single batch!

# Ensure output directories exist
mkdir -p "$RAW_ACTIVATIONS_DIR"
mkdir -p "$TEXT_RESULTS_DIR"

echo "--- SINGLE BATCH Activation Collection ---"
echo ""
echo "⚡ MAXIMUM SPEED MODE ⚡"
echo ""
echo "This will process ALL 50 prompts in a single batch!"
echo "Expected speedup: 50x faster than individual processing"
echo ""
echo "Model: $MODEL_NAME"
echo "Total prompts: 50"
echo "Batch size: $BATCH_SIZE (single batch)"
echo "Max tokens per prompt: $MAX_TOKENS"
echo ""
echo "Output directories:"
echo "  - Activations: $RAW_ACTIVATIONS_DIR"
echo "  - Text results: $TEXT_RESULTS_DIR"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Run the model with single batch processing
.venv/bin/python ./run_model.py \
    --model "$MODEL_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --prompts-file "$PROMPT_FILE_FOR_GEMMA" \
    --save-activations \
    --activations-dir "$RAW_ACTIVATIONS_DIR" \
    --output-dir "$TEXT_RESULTS_DIR" \
    --batch-size "$BATCH_SIZE" \
    --verbose

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "--- ✅ Single Batch Collection Completed Successfully ---"
    echo ""
    echo "All 50 prompts processed in ONE batch!"
    echo "Activations saved to: $RAW_ACTIVATIONS_DIR"
    echo ""
    echo "Next step: Run the analysis"
    echo "  .venv/bin/python analyze_single_batch_activations.py"
else
    echo ""
    echo "--- ❌ ERROR: Single Batch Collection Failed (Exit Code: $EXIT_CODE) ---"
    echo ""
    echo "Note: Processing 50 prompts in a single batch requires significant memory."
    echo "If you encounter memory issues, try reducing the batch size in collect_activations_true_batch.sh"
fi