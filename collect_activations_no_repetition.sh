#!/bin/bash

# Collection without repetition penalty to avoid serialization

PROMPT_FILE_FOR_GEMMA="./prompts_for_gemma_runner_flat.json"
RAW_ACTIVATIONS_DIR="./collected_activations_no_rep"
TEXT_RESULTS_DIR="./collected_text_no_rep"

# Create flat prompts file if needed
if [ ! -f "$PROMPT_FILE_FOR_GEMMA" ]; then
    echo "Creating flat prompts file..."
    .venv/bin/python create_flat_batched_prompts.py
fi

MODEL_NAME="mlx-community/gemma-3-1b-it-qat-4bit"
MAX_TOKENS=20
BATCH_SIZE=100

mkdir -p "$RAW_ACTIVATIONS_DIR"
mkdir -p "$TEXT_RESULTS_DIR"

echo "--- FAST Activation Collection (No Repetition Penalty) ---"
echo ""
echo "This disables repetition penalty to avoid serialization bottlenecks."
echo "Should be MUCH faster - true parallel processing!"
echo ""

# Run WITHOUT repetition penalty (which causes serialization)
.venv/bin/python ./run_model.py \
    --model "$MODEL_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --prompts-file "$PROMPT_FILE_FOR_GEMMA" \
    --save-activations \
    --activations-dir "$RAW_ACTIVATIONS_DIR" \
    --output-dir "$TEXT_RESULTS_DIR" \
    --batch-size "$BATCH_SIZE" \
    --repetition-penalty 1.0 \
    --verbose

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "--- ✅ Fast Collection Completed ---"
    echo ""
    echo "To analyze: ACTIVATIONS_DIR=./collected_activations_no_rep .venv/bin/python analyze_by_category.py"
else
    echo ""
    echo "--- ❌ ERROR: Collection Failed ---"
fi