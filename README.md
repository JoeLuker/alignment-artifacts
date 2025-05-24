# Alignment Artifact Analysis in Gemma Model

This project investigates the geometric signatures of "alignment artifacts" (safety-trained or evasive behaviors) within the MLP activations of the Gemma 3-1B language model. It provides a pipeline to:

1.  Generate input prompts structured for systematic activation collection.
2.  Run the `gemma_refactored` model to collect activations for these prompts.
3.  Analyze the collected activations to quantify the geometric separation between "natural" responses and "alignment artifact" responses across different model layers using Cohen's d and classification accuracy.

## Project Structure

```
alignment-artifacts/
├── .venv/                                # Python virtual environment
├── gemma_refactored/                     # Core Gemma model implementation with activation capture
│   ├── __init__.py
│   ├── activation_capture.py
│   ├── generation.py
│   ├── main.py                           # Main entry point for gemma_refactored logic
│   ├── model_architecture.py
│   ├── model_loading.py
│   ├── prompt_processing.py
│   └── README.md
├── run_model.py                          # Top-level script to run gemma_refactored.main.main()
├── alignment_artifact_prompt_pairs.json  # Defines natural/artifact prompt pairs for the experiment
├── create_gemma_runner_input.py          # Converts ^^ to format needed by run_model.py
├── prompts_for_gemma_runner.json         # Output of create_gemma_runner_input.py
├── collect_activations.sh                # Shell script to run activation collection
├── collected_activations_output/         # Directory where raw activations are saved by collect_activations.sh
│   └── natural_technical_dangerous_tech_1/ # Example group directory
│       └── prompt_list/                  # Subcategory key used in prompts_for_gemma_runner.json
│           └── batch_1/                  # Batch directory (since batch_size=1 per group)
│               ├── activations_step_0.npz
│               └── ...
├── analyze_collected_activations.py      # Script to analyze collected activations
├── alignment_artifact_analysis_plot.png  # Output plot from analysis
├── alignment_artifact_analysis_results.json # Output JSON from analysis
└── requirements.txt                      # Python dependencies
```

## Setup

1.  **Clone the repository (or create the project structure):**
    Ensure you have the `gemma_refactored` module and the necessary scripts (`run_model.py`, etc.).

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure `requirements.txt` includes `mlx`, `numpy`, `huggingface_hub`, `transformers`, and `matplotlib`.

## Workflow

The process involves three main steps:

### Step 1: Prepare Input Prompts for Activation Collection

The `alignment_artifact_prompt_pairs.json` file defines the core experimental prompts. This file needs to be converted into a format suitable for the `run_model.py` script when using its `--process-by-group` feature.

```bash
python create_gemma_runner_input.py
```
This will:
*   Read `alignment_artifact_prompt_pairs.json`.
*   Generate `prompts_for_gemma_runner.json`. This new JSON will have unique top-level keys for each original prompt condition (e.g., `natural_technical_dangerous_tech_1`, `artifact_social_political_pol_3`). Each key maps to a sub-dictionary containing a list with the single prompt string.

### Step 2: Collect Activations

The `collect_activations.sh` script automates running the Gemma model for all prompts defined in `prompts_for_gemma_runner.json` and saves their activations.

```bash
bash collect_activations.sh
```
This script will:
*   Call `python ./run_model.py` with appropriate arguments.
*   Use the `--process-by-group` flag, treating each top-level key in `prompts_for_gemma_runner.json` as a separate group.
*   Use `--batch-size 1` to ensure activations for each original prompt are processed individually.
*   Save activations into the `./collected_activations_output/` directory. The structure will be:
    `./collected_activations_output/<group_name>/<subcategory_key_from_json>/batch_1/activations_step_N.npz`
    (e.g., `./collected_activations_output/natural_technical_dangerous_tech_1/prompt_list/batch_1/activations_step_0.npz`)

**Configuration in `collect_activations.sh`:**
*   `MODEL_NAME`: Specify the Gemma model to use.
*   `MAX_TOKENS`: Number of generation steps (tokens) for which to save activations. This should match `NUM_TOKEN_STEPS_TO_COLLECT` in the analysis script.
*   Other generation parameters (temp, top-p, etc.) can be added.

### Step 3: Analyze Collected Activations

Once activations are collected, run the analysis script:

```bash
python analyze_collected_activations.py
```
This script will:
*   Read `alignment_artifact_prompt_pairs.json` to know the structure of original categories and pairs.
*   Load the MLP output activations (or other specified activations via `KEY_PATTERN_TEMPLATE`) from the corresponding directories in `./collected_activations_output/` for each layer.
*   Pool all "natural" activation vectors and all "artifact" activation vectors across all prompt pairs and generation steps for each layer.
*   Calculate Cohen's d and classification accuracy for each layer.
*   Generate `alignment_artifact_analysis_plot.png` visualizing these metrics.
*   Save detailed numerical results to `alignment_artifact_analysis_results.json`.

**Configuration in `analyze_collected_activations.py`:**
*   `RAW_ACTIVATIONS_BASE_DIR`: Path to the output of `collect_activations.sh`.
*   `ORIGINAL_PAIRS_JSON_FOR_ITERATION`: Path to your original `alignment_artifact_prompt_pairs.json`.
*   `GEMMA_INPUT_SUBCATEGORY_KEY_IN_PATH`: The subcategory key used in `prompts_for_gemma_runner.json` (e.g., `"prompt_list"`). This forms part of the path to the activation files.
*   `NUM_LAYERS`, `KEY_PATTERN_TEMPLATE`, `NUM_TOKEN_STEPS_TO_COLLECT`, `EXPECTED_HIDDEN_DIM`: Model and analysis parameters.

## Key Findings (Example from a Run)

Analysis of MLP output activations typically reveals significant geometric separation between natural and artifactual responses across multiple layers of the Gemma model. Cohen's d values can exceed 2.0 (e.g., Layer 1: d ≈ 2.9, Layer 6: d ≈ 2.5), indicating a very large effect size. Classification accuracy based on a simple linear boundary often surpasses 80% in mid-to-late layers.

These findings suggest that alignment techniques induce substantial and measurable geometric shifts in the model's activation space.

## Future Work & Extensions

*   Analyze activations from different model components (e.g., attention weights, residual stream).
*   Test interventions based on the discovered separation vectors.
*   Compare artifact signatures across different models or alignment methods.
*   Optimize the activation collection process for speed (e.g., using larger batch sizes for `run_model.py` and adapting the analysis script to handle batched activation files).