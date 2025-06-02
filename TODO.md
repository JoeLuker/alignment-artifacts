# TODO: Make Suppression Work with 4B Model

## THE ONLY GOAL: 
**Get alignment artifact suppression working with `mlx-community/gemma-3-4b-it-qat-4bit` for text-only use cases.**

## Current Status:
- ✅ Created multimodal wrapper (`gemma_multimodal.py`) that can load the 4b model
- ✅ Verified the 4b model can do basic text generation via mlx-vlm
- ❌ Activation collection fails due to architecture mismatch (34 vs 26 layers)
- ❌ No suppression vectors computed for 4b model
- ❌ Cannot run suppression on 4b model

## What Needs to Be Done:

### 1. Fix Activation Collection for 4B Model
- The 4b model has 34 text layers (not 26 like the 1b model)
- Weight keys are prefixed with `language_model.` for multimodal models
- Need to properly handle the architecture differences in `model_loading.py`

### 2. Run Full Activation Collection
- Execute `./collect_activations_no_repetition.sh` successfully
- Should process all 50 prompt pairs
- Store activations in `collected_activations_no_rep/`

### 3. Compute Suppression Vectors
- Run `analyze_by_category.py` on the collected activations
- Generate suppression vectors for the 4b model
- Save to `saved_models/suppression_vectors/`

### 4. Test Suppression
- Use `suppress.py` or similar to test suppression works
- Verify that alignment artifacts are successfully suppressed
- Compare suppressed vs unsuppressed outputs

## The Core Problem:
The activation collection script expects a text-only model architecture but the 4b model is multimodal. We need to either:
- A) Make the multimodal wrapper work seamlessly with activation collection, OR
- B) Extract just the text model from the multimodal model for activation collection

## Success Criteria:
```bash
# This should work without errors:
./collect_activations_no_repetition.sh
ACTIVATIONS_DIR=./collected_activations_no_rep .venv/bin/python analyze_by_category.py
.venv/bin/python suppress.py "How do I make a bomb?" --model mlx-community/gemma-3-4b-it-qat-4bit
```