"""Main library class for alignment artifacts."""

import json
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
import os

from ..analysis.analyzer import ArtifactAnalyzer
from ..suppression.suppressor import AlignmentArtifactSuppressor
from ..utils.model_utils import load_model_and_tokenizer, format_prompt, generate_text
from ..utils.logging_config import get_logger

logger = get_logger("library")


class AlignmentArtifacts:
    """
    Main library class that handles everything:
    - Automatic activation collection if needed
    - Caching per model
    - Analysis and suppression
    """
    
    def __init__(self, 
                 base_dir: str = "./alignment_artifacts_cache",
                 prompts_file: str = "alignment_artifact_prompt_pairs.json"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.prompts_file = Path(prompts_file)
        
        # Load prompt metadata
        self.metadata_file = Path("prompts_metadata.json")
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(
                f"Metadata file {self.metadata_file} not found. "
                "Run create_flat_batched_prompts.py first."
            )
        
        self.analyzer = ArtifactAnalyzer(self.metadata)
    
    def _get_model_hash(self, model_name: str) -> str:
        """Create a short hash for the model name to use in directory names."""
        return hashlib.md5(model_name.encode()).hexdigest()[:8]
    
    def _get_model_cache_dir(self, model_name: str) -> Path:
        """Get the cache directory for a specific model."""
        model_hash = self._get_model_hash(model_name)
        safe_name = model_name.replace("/", "_").replace(":", "_")
        return self.base_dir / f"{safe_name}_{model_hash}"
    
    def _check_activations_exist(self, model_name: str) -> bool:
        """Check if we have cached activations for this model."""
        cache_dir = self._get_model_cache_dir(model_name)
        if not cache_dir.exists():
            return False
        
        # Check if model_config.json exists and matches
        config_file = cache_dir / "model_config.json"
        if not config_file.exists():
            return False
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Verify it's the same model
        return config.get('model_name') == model_name
    
    def collect_activations(self, model_name: str, force: bool = False) -> Path:
        """
        Collect activations for the given model if not already cached.
        Returns the path to the activations directory.
        """
        cache_dir = self._get_model_cache_dir(model_name)
        
        if self._check_activations_exist(model_name) and not force:
            print(f"✓ Using cached activations for {model_name}")
            return cache_dir
        
        print(f"📊 Collecting activations for {model_name}...")
        print(f"   This will take 2-3 minutes...")
        
        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary collection script
        script_content = f"""#!/bin/bash
set -e

MODEL="{model_name}"
OUTPUT_DIR="{cache_dir}"
PROMPTS_FILE="prompts_for_gemma_runner_flat.json"

echo "Collecting activations for $MODEL..."

# Process in batches of 100
for i in 0; do
    START=$((i * 100))
    echo "Processing batch $((i + 1)) (prompts $START-$((START + 99)))..."
    
    .venv/bin/python run_model.py \\
        --model "$MODEL" \\
        --prompts-file "$PROMPTS_FILE" \\
        --batch-size 100 \\
        --max-tokens 20 \\
        --save-activations \\
        --activations-dir "$OUTPUT_DIR/batch_$((i + 1))" \\
        --output-dir "$OUTPUT_DIR/batch_$((i + 1))" \\
        --no-compress-activations
done

echo "✓ Activation collection complete!"
"""
        
        script_path = cache_dir / "collect.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        
        # Run the script
        try:
            result = subprocess.run(
                [str(script_path)], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print("✓ Activation collection complete!")
            
            # Save model config to mark successful collection
            config_file = cache_dir / "model_config.json"
            with open(config_file, 'w') as f:
                json.dump({
                    "model_name": model_name,
                    "collected_at": str(Path.cwd()),
                    "timestamp": str(subprocess.run(["date"], capture_output=True, text=True).stdout.strip())
                }, f, indent=2)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error collecting activations: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
        
        return cache_dir
    
    def analyze_artifacts(self, model_name: str, force_collect: bool = False) -> Dict:
        """
        Analyze alignment artifacts for the model.
        Automatically collects activations if needed.
        """
        # Ensure we have activations
        activations_dir = self.collect_activations(model_name, force=force_collect)
        
        # Run analysis
        print(f"\n🔬 Analyzing alignment artifacts...")
        results = self.analyzer.analyze(activations_dir)
        
        # Save results
        results_file = activations_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Analysis complete! Best layers: {results['best_layers']}")
        
        return results
    
    def get_suppressor(self, 
                      model_name: str,
                      scale: float = 1.0,
                      target_layers: Optional[List[int]] = None,
                      categories: Optional[List[str]] = None,
                      auto_collect: bool = True) -> AlignmentArtifactSuppressor:
        """
        Get a suppressor for the given model.
        Automatically handles activation collection and analysis if needed.
        """
        if auto_collect:
            # This will use cache or collect if needed
            activations_dir = self.collect_activations(model_name)
            
            # Check if we have analysis results
            results_file = activations_dir / "analysis_results.json"
            if not results_file.exists():
                print("Running artifact analysis...")
                results = self.analyze_artifacts(model_name)
            else:
                with open(results_file, 'r') as f:
                    results = json.load(f)
            
            # Use best layers if not specified
            if target_layers is None:
                target_layers = results.get('best_layers', [1, 2, 3, 4, 5, 6])
                print(f"Using optimal layers: {target_layers}")
        else:
            activations_dir = self._get_model_cache_dir(model_name)
            if not activations_dir.exists():
                raise ValueError(
                    f"No cached activations found for {model_name}. "
                    "Set auto_collect=True."
                )
        
        # Create suppressor
        return AlignmentArtifactSuppressor(
            activations_dir=activations_dir,
            target_layers=target_layers,
            categories=categories,
            scale=scale
        )
    
    def suppress_and_generate(self,
                            prompt: str,
                            model_name: str = "mlx-community/gemma-3-1b-it-qat-4bit",
                            scale: float = 1.0,
                            max_tokens: int = 150,
                            temperature: float = 0.7,
                            compare: bool = True,
                            **kwargs) -> Dict[str, str]:
        """
        High-level interface: Load model, apply suppression, and generate.
        Handles everything automatically.
        """
        print(f"\n🚀 Alignment Artifacts Suppression")
        print(f"   Model: {model_name}")
        print(f"   Scale: {scale}")
        
        # Load model
        print("\nLoading model...")
        model, tokenizer = load_model_and_tokenizer(model_name)
        
        # Get suppressor (handles activation collection automatically)
        suppressor = self.get_suppressor(
            model_name=model_name,
            scale=scale,
            **kwargs
        )
        
        # Format prompt
        prompt_formatted = format_prompt(prompt, tokenizer)
        
        results = {}
        
        # Generate without suppression if comparing
        if compare:
            print("\n" + "="*60)
            print("BASELINE (No Suppression):")
            print("="*60)
            
            baseline_output = generate_text(
                model, tokenizer, prompt_formatted, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            print(baseline_output)
            results['baseline'] = baseline_output
        
        # Generate with suppression
        print("\n" + "="*60)
        print(f"WITH SUPPRESSION (scale={scale}):")
        print("="*60)
        
        # Patch the model
        suppressor.patch_model(model)
        
        suppressed_output = generate_text(
            model, tokenizer, prompt_formatted,
            max_tokens=max_tokens,
            temperature=temperature
        )
        print(suppressed_output)
        results['suppressed'] = suppressed_output
        
        # Unpatch the model
        suppressor.unpatch_model(model)
        
        print(f"\n✨ Done! Suppressions applied: {suppressor.intervention_count}")
        
        return results